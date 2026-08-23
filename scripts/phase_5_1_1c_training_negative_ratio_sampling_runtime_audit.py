"""
Phase 5.1.1c — Training Negative-Ratio and Sampling-Runtime Audit

AUDIT ONLY.

This script DOES NOT:
- generate negative rows;
- instantiate any RNG;
- select/freeze a negative:positive ratio;
- select/freeze a probability distribution;
- select/freeze replacement semantics;
- select/freeze resampling cadence;
- instantiate the ITRS model;
- create an optimizer;
- train.

It consumes the frozen Phase-5.1.1b training-negative semantics and
quantifies the implications of several possible runtime contracts.

Inherited frozen rule
---------------------
For training target T_h, h = 1..59:

    eligible_negative(o,b,h) :=
        startup_role_node(b)
        AND NOT EXISTS positive_event(o,b,s)
                     WITH segment_number(s) <= h

Future-positive pairs remain eligible because excluding them would use
future labels.

Paper-specified facts
---------------------
- Training negatives are randomly sampled.
- Batch size = 512.
- Evaluation uses 1 positive + 99 random negatives.
- Evaluation metrics = HR@10 and NDCG@10.

Still paper-unspecified
-----------------------
- training negative:positive ratio;
- exact random distribution;
- replacement semantics;
- resampling cadence;
- negative-sampling RNG/runtime contract.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

NEGATIVE_SEMANTICS_CONTRACT = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_1_1b_training_negative_semantics_contract.json"
)

PHASE_5_1_1A_SCOPE = Path(
    "data/experimental/phase_5/audits/phase_5_1_1a/"
    "negative_pool_feasibility_and_collision_overall_by_scope.csv"
)

OUT_DIR = Path(
    "data/experimental/phase_5/audits/phase_5_1_1c"
)

RATIO_AUDIT_PATH = (
    OUT_DIR
    / "training_negative_ratio_workload_audit.csv"
)

DISTRIBUTION_OPTIONS_PATH = (
    OUT_DIR
    / "training_negative_distribution_option_register.csv"
)

REPLACEMENT_OPTIONS_PATH = (
    OUT_DIR
    / "training_negative_replacement_option_register.csv"
)

RESAMPLING_OPTIONS_PATH = (
    OUT_DIR
    / "training_negative_resampling_option_register.csv"
)

DIVERSITY_PATH = (
    OUT_DIR
    / "epoch_resampling_diversity_sensitivity.csv"
)

RNG_OPTIONS_PATH = (
    OUT_DIR
    / "training_negative_rng_option_register.csv"
)

AUDIT_MANIFEST_PATH = (
    OUT_DIR
    / "phase_5_1_1c_audit_manifest.json"
)


# =============================================================================
# Frozen / inherited values
# =============================================================================

EXPECTED_TRAINING_POSITIVES = 1_073_249
EXPECTED_STARTUP_UNIVERSE = 311_589

PAPER_BATCH_SIZE = 512

SELECTED_UNIVERSE = "global_role_universe"

SELECTED_EXCLUSION = (
    "exclude_prior_and_target_period_positive_pairs"
)


# These are diagnostic probes only.
#
# 99 is included as the paper-specified EVALUATION negative count,
# not because 99 is assumed appropriate for training.
#
# 4 is included because it is a common NCF-style training reference,
# not because ITRS reports it.
RATIO_PROBES = (
    1,
    4,
    10,
    20,
    50,
    99,
)


# Epoch counts here are sensitivity probes only.
# Phase 5 has NOT yet selected an epoch count.
EPOCH_PROBES = (
    1,
    5,
    10,
    20,
    50,
)


# =============================================================================
# Helpers
# =============================================================================

def banner(text: str) -> None:
    print("\n" + "=" * 118)
    print(text)
    print("=" * 118)


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def div0(num: float, den: float) -> float:
    if den == 0:
        return float("nan")

    return float(num) / float(den)


def duplicate_probability_with_replacement(
    population_size: int,
    draws: int,
) -> float:
    """
    Exact probability of at least one duplicate when `draws`
    independent draws are made WITH replacement from N items.

    P(no duplicate) =
        N/N * (N-1)/N * ... * (N-draws+1)/N
    """

    require(
        population_size > 0,
        "population_size must be positive",
    )

    require(
        draws >= 0,
        "draws must be nonnegative",
    )

    if draws <= 1:
        return 0.0

    if draws > population_size:
        return 1.0

    log_p_unique = 0.0

    for i in range(draws):
        log_p_unique += math.log(
            (
                population_size - i
            )
            / population_size
        )

    p_unique = math.exp(log_p_unique)

    return 1.0 - p_unique


def expected_distinct_across_epochs(
    population_size: float,
    negatives_per_epoch: int,
    epochs: int,
) -> float:
    """
    Expected distinct negative candidates seen by ONE positive
    after E independently regenerated epochs when each epoch
    draws K unique negatives uniformly without replacement.

    Probability that one particular candidate is not chosen
    during one epoch:

        1 - K/N

    Across E independent epochs:

        (1 - K/N)^E

    Therefore expected distinct candidates:

        N * [1 - (1 - K/N)^E]

    No random sampling is performed here.
    """

    require(
        population_size > 0,
        "population_size must be positive",
    )

    require(
        0 <= negatives_per_epoch <= population_size,
        "invalid negatives_per_epoch",
    )

    require(
        epochs >= 1,
        "epochs must be >= 1",
    )

    p_not_selected_one_epoch = (
        1.0
        - (
            negatives_per_epoch
            / population_size
        )
    )

    return (
        population_size
        * (
            1.0
            - (
                p_not_selected_one_epoch
                ** epochs
            )
        )
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.1.1c — "
        "TRAINING NEGATIVE-RATIO AND SAMPLING-RUNTIME AUDIT"
    )

    print("Negative rows generated:             0")
    print("RNG instantiated:                    NO")
    print("Ratio frozen:                        NO")
    print("Distribution frozen:                 NO")
    print("Replacement semantics frozen:        NO")
    print("Resampling cadence frozen:           NO")
    print("Model instantiated:                  NO")
    print("Optimizer created:                   NO")
    print("Training performed:                  NO")

    # =========================================================================
    # Input existence
    # =========================================================================

    banner("AUTHORITATIVE INPUT EXISTENCE")

    for path in (
        NEGATIVE_SEMANTICS_CONTRACT,
        PHASE_5_1_1A_SCOPE,
    ):

        require(
            path.exists(),
            f"Missing authoritative input: {path}",
        )

        print(f"FOUND  {path}")

    # =========================================================================
    # Read frozen 5.1.1b contract
    # =========================================================================

    banner("PHASE 5.1.1b CONTRACT RECHECK")

    contract = json.loads(
        NEGATIVE_SEMANTICS_CONTRACT.read_text(
            encoding="utf-8"
        )
    )

    require(
        contract.get("phase") == "5.1.1b",
        "Unexpected contract phase",
    )

    require(
        contract.get("status") == "FROZEN",
        "Phase 5.1.1b contract is not frozen",
    )

    require(
        contract.get("negative_samples_generated") is False,
        "5.1.1b unexpectedly generated negatives",
    )

    require(
        contract.get("training_performed") is False,
        "5.1.1b unexpectedly performed training",
    )

    candidate_contract = contract[
        "training_negative_candidate_universe"
    ]

    exclusion_contract = contract[
        "training_negative_exclusion"
    ]

    require(
        int(candidate_contract["count"])
        == EXPECTED_STARTUP_UNIVERSE,
        "Startup candidate-universe count drift",
    )

    require(
        exclusion_contract[
            "exclude_prior_positive_pairs"
        ]
        is True,
        "Prior-positive exclusion is not frozen",
    )

    require(
        exclusion_contract[
            "exclude_other_positive_pairs_in_target_segment"
        ]
        is True,
        "Same-target positive exclusion is not frozen",
    )

    require(
        exclusion_contract[
            "exclude_future_positive_pairs"
        ]
        is False,
        "Future-positive exclusion unexpectedly enabled",
    )

    require(
        exclusion_contract[
            "use_T60_labels_for_T1_T59_sampling"
        ]
        is False,
        "T60 labels unexpectedly allowed in training sampling",
    )

    print(
        "Candidate universe:       "
        f"{candidate_contract['count']:,} Startup role nodes  PASS"
    )

    print(
        "Prior positives excluded: YES  PASS"
    )

    print(
        "Target positives excluded:YES  PASS"
    )

    print(
        "Future positives excluded:NO   PASS"
    )

    print(
        "T60 labels used in train: NO   PASS"
    )

    # =========================================================================
    # Re-read 5.1.1a selected-policy evidence
    # =========================================================================

    banner("SELECTED POLICY FEASIBILITY EVIDENCE")

    scope = pd.read_csv(
        PHASE_5_1_1A_SCOPE
    )

    selected_rows = scope.loc[
        (
            scope["scope"]
            == "training_T1_T59"
        )
        & (
            scope["candidate_universe"]
            == SELECTED_UNIVERSE
        )
        & (
            scope["exclusion_policy"]
            == SELECTED_EXCLUSION
        )
    ].copy()

    require(
        len(selected_rows) == 1,
        (
            "Expected exactly one Phase-5.1.1a "
            "selected-policy training row"
        ),
    )

    selected = selected_rows.iloc[0]

    training_positives = int(
        selected["positive_events"]
    )

    negative_pool_min = int(
        selected["negative_pool_min"]
    )

    negative_pool_mean = float(
        selected["negative_pool_mean"]
    )

    future_positive_probability = float(
        selected[
            "p_uniform_draw_hits_future_positive"
        ]
    )

    require(
        training_positives
        == EXPECTED_TRAINING_POSITIVES,
        "Training positive count drift",
    )

    require(
        negative_pool_min >= 99,
        (
            "Negative pool unexpectedly cannot support "
            "the 99-negative reference probe"
        ),
    )

    require(
        float(
            selected[
                "p_uniform_draw_hits_historical_positive"
            ]
        )
        == 0.0,
        "Historical collisions reappeared",
    )

    require(
        float(
            selected[
                "p_uniform_draw_hits_other_target_positive"
            ]
        )
        == 0.0,
        "Same-target collisions reappeared",
    )

    require(
        future_positive_probability > 0.0,
        (
            "Future-positive diagnostic unexpectedly vanished; "
            "future labels may have entered eligibility."
        ),
    )

    print(
        f"Training positives:           {training_positives:,}"
    )

    print(
        f"Minimum eligible pool:        {negative_pool_min:,}"
    )

    print(
        f"Mean eligible pool:           {negative_pool_mean:,.6f}"
    )

    print(
        "Future-positive diagnostic:  "
        f"{future_positive_probability:.9f}"
    )

    # =========================================================================
    # Ratio workload audit
    # =========================================================================

    banner("NEGATIVE:POSITIVE RATIO WORKLOAD AUDIT")

    ratio_rows = []

    for k in RATIO_PROBES:

        negatives_per_epoch = (
            training_positives
            * k
        )

        labeled_examples_per_epoch = (
            training_positives
            * (
                1 + k
            )
        )

        batches_per_epoch = math.ceil(
            labeled_examples_per_epoch
            / PAPER_BATCH_SIZE
        )

        negative_fraction = (
            k
            / (
                1 + k
            )
        )

        min_pool_fraction = (
            k
            / negative_pool_min
        )

        mean_pool_fraction = (
            k
            / negative_pool_mean
        )

        duplicate_probability_min_pool = (
            duplicate_probability_with_replacement(
                negative_pool_min,
                k,
            )
        )

        expected_unique_with_replacement_min_pool = (
            negative_pool_min
            * (
                1.0
                - (
                    (
                        negative_pool_min - 1
                    )
                    / negative_pool_min
                )
                ** k
            )
        )

        # This is a hypothetical reference under uniform sampling.
        # Uniform sampling itself is NOT yet frozen.
        expected_future_positive_draws_uniform = (
            negatives_per_epoch
            * future_positive_probability
        )

        # Rejection-sampling diagnostics:
        #
        # If a raw Startup node is proposed uniformly from the full
        # 311,589-node universe and rejected when ineligible:
        #
        # acceptance probability = eligible_pool / universe_size
        #
        # Expected proposals per accepted sample = 1 / acceptance_probability
        #
        # This is only an implementation-feasibility diagnostic.
        worst_acceptance_probability = (
            negative_pool_min
            / EXPECTED_STARTUP_UNIVERSE
        )

        mean_acceptance_probability = (
            negative_pool_mean
            / EXPECTED_STARTUP_UNIVERSE
        )

        worst_expected_proposals_per_accepted = (
            1.0
            / worst_acceptance_probability
        )

        mean_expected_proposals_per_accepted = (
            1.0
            / mean_acceptance_probability
        )

        ratio_rows.append(
            {
                "negatives_per_positive": k,
                "classification": (
                    "AUDIT_PROBE_NOT_FROZEN"
                ),
                "training_positive_events": (
                    training_positives
                ),
                "negative_examples_per_epoch": (
                    negatives_per_epoch
                ),
                "labeled_examples_per_epoch": (
                    labeled_examples_per_epoch
                ),
                "batches_per_epoch_batch512": (
                    batches_per_epoch
                ),
                "positive_fraction": (
                    1.0
                    / (
                        1 + k
                    )
                ),
                "negative_fraction": (
                    negative_fraction
                ),
                "workload_multiplier_vs_positive_only": (
                    1 + k
                ),
                "workload_multiplier_vs_1_to_1": (
                    (
                        1 + k
                    )
                    / 2.0
                ),
                "minimum_eligible_pool": (
                    negative_pool_min
                ),
                "mean_eligible_pool": (
                    negative_pool_mean
                ),
                "negative_fraction_of_min_pool_per_positive": (
                    min_pool_fraction
                ),
                "negative_fraction_of_mean_pool_per_positive": (
                    mean_pool_fraction
                ),
                "without_replacement_feasible_for_all_events": (
                    k
                    <= negative_pool_min
                ),
                "duplicate_probability_if_with_replacement_at_min_pool": (
                    duplicate_probability_min_pool
                ),
                "expected_unique_negatives_if_with_replacement_at_min_pool": (
                    expected_unique_with_replacement_min_pool
                ),
                "uniform_reference_expected_future_positive_draws_per_epoch": (
                    expected_future_positive_draws_uniform
                ),
                "uniform_reference_only": True,
                "worst_rejection_sampling_acceptance_probability": (
                    worst_acceptance_probability
                ),
                "mean_rejection_sampling_acceptance_probability": (
                    mean_acceptance_probability
                ),
                "worst_expected_raw_proposals_per_accepted_negative": (
                    worst_expected_proposals_per_accepted
                ),
                "mean_expected_raw_proposals_per_accepted_negative": (
                    mean_expected_proposals_per_accepted
                ),
            }
        )

    ratio_df = pd.DataFrame(
        ratio_rows
    )

    print(
        ratio_df[
            [
                "negatives_per_positive",
                "negative_examples_per_epoch",
                "labeled_examples_per_epoch",
                "batches_per_epoch_batch512",
                "negative_fraction",
                (
                    "duplicate_probability_if_with_"
                    "replacement_at_min_pool"
                ),
                (
                    "uniform_reference_expected_"
                    "future_positive_draws_per_epoch"
                ),
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # Distribution-option register
    # =========================================================================

    banner("RANDOM-DISTRIBUTION OPTION AUDIT")

    distribution_df = pd.DataFrame(
        [
            {
                "option": "uniform_over_eligible_startups",
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "requires_additional_data": False,
                "uses_future_information": False,
                "changes_random_sampling_into_hard_sampling": False,
                "main_advantage": (
                    "Minimal interpretation of random sampling; "
                    "every eligible Startup has equal probability."
                ),
                "main_risk": (
                    "May generate many easy negatives because "
                    "candidate universe is very large."
                ),
                "audit_assessment": (
                    "STRONG_CANDIDATE_FOR_STRICT_REPRODUCTION"
                ),
            },
            {
                "option": "historical_popularity_weighted_through_target",
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "requires_additional_data": True,
                "uses_future_information": False,
                "changes_random_sampling_into_hard_sampling": False,
                "main_advantage": (
                    "Can emphasize startups plausible under "
                    "historical market activity."
                ),
                "main_risk": (
                    "Introduces a weighting model not reported by ITRS."
                ),
                "audit_assessment": (
                    "NOT_PREFERRED_FOR_STRICT_REPRODUCTION"
                ),
            },
            {
                "option": "global_popularity_weighted_T0_T60",
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "requires_additional_data": True,
                "uses_future_information": True,
                "changes_random_sampling_into_hard_sampling": False,
                "main_advantage": (
                    "Simple static popularity distribution."
                ),
                "main_risk": (
                    "Uses future T_{h+1}..T60 activity to define "
                    "historical training probabilities."
                ),
                "audit_assessment": (
                    "REJECT_FOR_TEMPORAL_LEAKAGE"
                ),
            },
            {
                "option": "model_score_hard_negative_sampling",
                "classification": (
                    "METHOD_EXTENSION_NOT_STRICT_REPRODUCTION"
                ),
                "requires_additional_data": False,
                "uses_future_information": False,
                "changes_random_sampling_into_hard_sampling": True,
                "main_advantage": (
                    "Potentially more informative gradient signal."
                ),
                "main_risk": (
                    "Changes the reported random-negative training "
                    "method and couples sampling to current model state."
                ),
                "audit_assessment": (
                    "REJECT_FOR_PRIMARY_REPRODUCTION"
                ),
            },
        ]
    )

    print(
        distribution_df[
            [
                "option",
                "uses_future_information",
                "audit_assessment",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # Replacement semantics
    # =========================================================================

    banner("REPLACEMENT-SEMANTICS OPTION AUDIT")

    replacement_df = pd.DataFrame(
        [
            {
                "option": (
                    "without_replacement_within_each_positive"
                ),
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "duplicate_negative_within_positive_possible": False,
                "feasible_at_K99_for_all_training_events": (
                    negative_pool_min >= 99
                ),
                "main_advantage": (
                    "Guarantees K distinct negative Startups "
                    "for each positive event."
                ),
                "main_risk": (
                    "Paper does not explicitly state replacement semantics."
                ),
                "audit_assessment": (
                    "STRONG_CANDIDATE"
                ),
            },
            {
                "option": (
                    "with_replacement_within_each_positive"
                ),
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "duplicate_negative_within_positive_possible": True,
                "feasible_at_K99_for_all_training_events": True,
                "main_advantage": (
                    "Simplest independent-draw interpretation."
                ),
                "main_risk": (
                    "Can duplicate the same negative Startup "
                    "inside one positive's training set."
                ),
                "audit_assessment": (
                    "NO_OBSERVED_NEED"
                ),
            },
        ]
    )

    print(
        replacement_df[
            [
                "option",
                "duplicate_negative_within_positive_possible",
                "audit_assessment",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # Epoch-resampling diversity
    # =========================================================================

    banner("FIXED VS PER-EPOCH NEGATIVE DIVERSITY")

    diversity_rows = []

    for k in RATIO_PROBES:

        for epochs in EPOCH_PROBES:

            expected_distinct_min_pool = (
                expected_distinct_across_epochs(
                    float(negative_pool_min),
                    k,
                    epochs,
                )
            )

            expected_distinct_mean_pool = (
                expected_distinct_across_epochs(
                    negative_pool_mean,
                    k,
                    epochs,
                )
            )

            diversity_rows.append(
                {
                    "negatives_per_positive": k,
                    "epochs_probe": epochs,

                    # Fixed once means the same K negatives
                    # are reused forever.
                    "fixed_once_distinct_negatives_per_positive": (
                        k
                    ),

                    # Per-epoch case assumes independent,
                    # uniform, without-replacement draws
                    # inside each epoch.
                    "per_epoch_expected_distinct_at_min_pool": (
                        expected_distinct_min_pool
                    ),
                    "per_epoch_expected_distinct_at_mean_pool": (
                        expected_distinct_mean_pool
                    ),
                    "diversity_multiplier_vs_fixed_min_pool": (
                        div0(
                            expected_distinct_min_pool,
                            k,
                        )
                    ),
                    "diversity_multiplier_vs_fixed_mean_pool": (
                        div0(
                            expected_distinct_mean_pool,
                            k,
                        )
                    ),
                    "classification": (
                        "SENSITIVITY_ONLY_NOT_EPOCH_SELECTION"
                    ),
                }
            )

    diversity_df = pd.DataFrame(
        diversity_rows
    )

    print(
        diversity_df.loc[
            diversity_df[
                "negatives_per_positive"
            ]
            == 4,
            [
                "negatives_per_positive",
                "epochs_probe",
                "fixed_once_distinct_negatives_per_positive",
                "per_epoch_expected_distinct_at_mean_pool",
                "diversity_multiplier_vs_fixed_mean_pool",
            ],
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # Resampling cadence register
    # =========================================================================

    banner("RESAMPLING-CADENCE OPTION AUDIT")

    resampling_df = pd.DataFrame(
        [
            {
                "option": "fixed_once_before_training",
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "negative_diversity_across_epochs": "LOW",
                "resume_reproducibility_complexity": "LOW",
                "depends_on_batch_order": False,
                "main_advantage": (
                    "Simple deterministic training dataset."
                ),
                "main_risk": (
                    "Every positive sees the exact same negatives "
                    "throughout all epochs."
                ),
                "audit_assessment": (
                    "VALID_BUT_LOW_DIVERSITY"
                ),
            },
            {
                "option": "regenerate_once_per_epoch",
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "negative_diversity_across_epochs": "HIGH",
                "resume_reproducibility_complexity": "LOW_TO_MODERATE",
                "depends_on_batch_order": False,
                "main_advantage": (
                    "Increases negative coverage while preserving "
                    "a fixed K:1 class ratio each epoch."
                ),
                "main_risk": (
                    "Requires explicit epoch-indexed RNG semantics "
                    "for exact reruns/resume."
                ),
                "audit_assessment": (
                    "STRONG_CANDIDATE"
                ),
            },
            {
                "option": "sample_on_each_batch_or_step",
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "negative_diversity_across_epochs": "HIGH",
                "resume_reproducibility_complexity": "HIGH",
                "depends_on_batch_order": True,
                "main_advantage": (
                    "Maximum online variation."
                ),
                "main_risk": (
                    "Sampling becomes coupled to shuffle order, "
                    "worker state, batching, checkpoint location, "
                    "and RNG-consumption order."
                ),
                "audit_assessment": (
                    "NOT_PREFERRED"
                ),
            },
        ]
    )

    print(
        resampling_df[
            [
                "option",
                "negative_diversity_across_epochs",
                "resume_reproducibility_complexity",
                "audit_assessment",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # RNG strategy audit
    # =========================================================================

    banner("NEGATIVE-SAMPLING RNG OPTION AUDIT")

    rng_df = pd.DataFrame(
        [
            {
                "option": "shared_global_rng_state",
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE_CANDIDATE"
                ),
                "exact_epoch_reconstruction": False,
                "robust_to_resume_from_epoch_boundary": False,
                "coupled_to_unrelated_rng_consumption": True,
                "audit_assessment": (
                    "NOT_PREFERRED"
                ),
                "reason": (
                    "Exact negatives depend on every previous "
                    "random operation."
                ),
            },
            {
                "option": "single_dedicated_sampler_rng_stream",
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE_CANDIDATE"
                ),
                "exact_epoch_reconstruction": False,
                "robust_to_resume_from_epoch_boundary": (
                    "ONLY_IF_RNG_STATE_CHECKPOINTED"
                ),
                "coupled_to_unrelated_rng_consumption": False,
                "audit_assessment": (
                    "VALID"
                ),
                "reason": (
                    "Separates negative sampling from model and "
                    "dataloader RNG, but stream state must be preserved."
                ),
            },
            {
                "option": "deterministic_epoch_indexed_sampler_seed",
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE_CANDIDATE"
                ),
                "exact_epoch_reconstruction": True,
                "robust_to_resume_from_epoch_boundary": True,
                "coupled_to_unrelated_rng_consumption": False,
                "audit_assessment": (
                    "STRONG_CANDIDATE"
                ),
                "reason": (
                    "Each epoch's negatives can be reconstructed "
                    "independently from base seed + epoch index."
                ),
            },
        ]
    )

    print(
        rng_df[
            [
                "option",
                "exact_epoch_reconstruction",
                "audit_assessment",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner("FINAL AUDIT INVARIANTS")

    require(
        ratio_df[
            "without_replacement_feasible_for_all_events"
        ].all(),
        (
            "At least one ratio probe is infeasible "
            "without replacement"
        ),
    )

    require(
        int(
            ratio_df.loc[
                ratio_df[
                    "negatives_per_positive"
                ]
                == 99,
                "labeled_examples_per_epoch",
            ].iloc[0]
        )
        == (
            training_positives
            * 100
        ),
        "99:1 workload arithmetic failed",
    )

    require(
        int(
            ratio_df.loc[
                ratio_df[
                    "negatives_per_positive"
                ]
                == 4,
                "negative_examples_per_epoch",
            ].iloc[0]
        )
        == (
            training_positives
            * 4
        ),
        "4:1 workload arithmetic failed",
    )

    print(
        "5.1.1b frozen contract loaded:       PASS"
    )
    print(
        "Training positive count preserved:   PASS"
    )
    print(
        "All ratio probes feasible:           PASS"
    )
    print(
        "99-without-replacement feasible:     PASS"
    )
    print(
        "No historical collisions reappear:  PASS"
    )
    print(
        "No target collisions reappear:       PASS"
    )
    print(
        "Future-positive status not excluded: PASS"
    )
    print(
        "Negative rows generated:             0"
    )
    print(
        "RNG instantiated:                    NO"
    )
    print(
        "Training performed:                  NO"
    )
    print(
        "Sampling runtime frozen:             NO"
    )

    # =========================================================================
    # Write outputs
    # =========================================================================

    banner("WRITE AUDIT-ONLY OUTPUTS")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ratio_df.to_csv(
        RATIO_AUDIT_PATH,
        index=False,
    )

    distribution_df.to_csv(
        DISTRIBUTION_OPTIONS_PATH,
        index=False,
    )

    replacement_df.to_csv(
        REPLACEMENT_OPTIONS_PATH,
        index=False,
    )

    resampling_df.to_csv(
        RESAMPLING_OPTIONS_PATH,
        index=False,
    )

    diversity_df.to_csv(
        DIVERSITY_PATH,
        index=False,
    )

    rng_df.to_csv(
        RNG_OPTIONS_PATH,
        index=False,
    )

    manifest = {
        "phase": "5.1.1c",
        "title": (
            "Training Negative-Ratio and "
            "Sampling-Runtime Audit"
        ),
        "status": "AUDIT_COMPLETE_NOT_FROZEN",

        "negative_samples_generated": False,
        "rng_instantiated": False,
        "model_instantiated": False,
        "optimizer_created": False,
        "training_performed": False,

        "inherited_frozen_semantics": {
            "candidate_universe": (
                "all 311589 frozen Startup role nodes"
            ),
            "positive_exclusion": (
                "exclude Investor-Startup pairs with "
                "first_positive_segment <= target segment"
            ),
            "future_positive_exclusion": False,
        },

        "paper_specified": {
            "training_negative_method": "random",
            "batch_size": PAPER_BATCH_SIZE,
            "evaluation_negative_count": 99,
            "evaluation_metrics": [
                "HR@10",
                "NDCG@10",
            ],
        },

        "ratio_probes_only": list(
            RATIO_PROBES
        ),

        "epoch_probes_only": list(
            EPOCH_PROBES
        ),

        "important_interpretation": {
            "ratio_99": (
                "Paper-specified for evaluation only; "
                "not inherited as training ratio."
            ),
            "ratio_4": (
                "NCF-style external methodological precedent; "
                "not an ITRS paper parameter."
            ),
            "future_positive_probability": (
                "Diagnostic only. Future positives remain legal "
                "historical negatives under frozen temporal semantics."
            ),
        },

        "still_unresolved_after_audit": [
            "training negative:positive ratio",
            "training random distribution",
            "training replacement semantics",
            "training resampling cadence",
            "training negative RNG/seed runtime contract",
            "training epoch count",
            "early stopping",
            "weight decay",
            (
                "evaluation candidate-generation "
                "runtime contract"
            ),
        ],

        "outputs": [
            str(RATIO_AUDIT_PATH),
            str(DISTRIBUTION_OPTIONS_PATH),
            str(REPLACEMENT_OPTIONS_PATH),
            str(RESAMPLING_OPTIONS_PATH),
            str(DIVERSITY_PATH),
            str(RNG_OPTIONS_PATH),
        ],

        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    AUDIT_MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    for path in (
        RATIO_AUDIT_PATH,
        DISTRIBUTION_OPTIONS_PATH,
        REPLACEMENT_OPTIONS_PATH,
        RESAMPLING_OPTIONS_PATH,
        DIVERSITY_PATH,
        RNG_OPTIONS_PATH,
        AUDIT_MANIFEST_PATH,
    ):
        print(f"WROTE  {path}")

    # =========================================================================
    # Decision-facing summary
    # =========================================================================

    banner("DECISION-FACING SUMMARY — NOTHING NEW FROZEN")

    focal_ratios = ratio_df.loc[
        ratio_df[
            "negatives_per_positive"
        ].isin(
            [
                1,
                4,
                10,
                99,
            ]
        ),
        [
            "negatives_per_positive",
            "labeled_examples_per_epoch",
            "batches_per_epoch_batch512",
            "negative_fraction",
            (
                "duplicate_probability_if_with_"
                "replacement_at_min_pool"
            ),
            (
                "uniform_reference_expected_"
                "future_positive_draws_per_epoch"
            ),
        ],
    ]

    print(
        focal_ratios.to_string(
            index=False
        )
    )

    print()
    print(
        "Interpretation:"
    )
    print(
        "1. Candidate-pool capacity does not force the ratio."
    )
    print(
        "2. 99 is an evaluation setting, not a known training ratio."
    )
    print(
        "3. Uniform eligible sampling introduces the fewest "
        "additional assumptions."
    )
    print(
        "4. Without-replacement sampling is feasible for every "
        "audited ratio."
    )
    print(
        "5. Per-epoch regeneration increases diversity without "
        "changing the class ratio."
    )
    print(
        "6. Epoch-indexed sampler seeds provide the cleanest "
        "reconstruction/resume semantics."
    )

    banner(
        "PHASE 5.1.1c COMPLETE"
    )

    print(
        "AUDIT COMPLETE — NO NEGATIVES GENERATED — "
        "NO NEW SAMPLING DECISION FROZEN"
    )


if __name__ == "__main__":
    main()