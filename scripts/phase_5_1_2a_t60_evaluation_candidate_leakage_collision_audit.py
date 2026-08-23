"""
Phase 5.1.2a — T60 Evaluation Candidate-Generation
Leakage and Collision Audit

AUDIT ONLY.

This script DOES NOT:
- sample evaluation negatives;
- instantiate an RNG;
- freeze an evaluation candidate universe;
- freeze an evaluation exclusion policy;
- freeze evaluation replacement semantics;
- freeze an evaluation RNG seed;
- instantiate the model;
- create an optimizer;
- train;
- evaluate HR/NDCG.

Paper-specified evaluation fact
-------------------------------
Each validation/test case contains:
    1 positive + 99 randomly sampled negatives

Metrics:
    HR@10
    NDCG@10

Paper-unspecified
-----------------
- negative candidate universe;
- whether historical positives are excluded;
- whether other T60 positives are excluded;
- whether validation/test labels may influence candidate construction;
- exact random distribution;
- replacement semantics;
- evaluation sampling seed / persistence.

This audit compares candidate and exclusion policies algebraically.
NO RANDOM DRAW IS MADE.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

TEMPORAL_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "interactions_itrs_temporal_split.parquet"
)

NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/model_ready/"
    "node_index.parquet"
)

TRAINING_NEGATIVE_CONTRACT = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_1_1d_training_negative_sampling_runtime_contract.json"
)

OUT_DIR = Path(
    "data/experimental/phase_5/audits/phase_5_1_2a"
)

CANDIDATE_REGISTER_PATH = (
    OUT_DIR
    / "t60_evaluation_candidate_universe_register.csv"
)

POLICY_REGISTER_PATH = (
    OUT_DIR
    / "t60_evaluation_exclusion_policy_risk_register.csv"
)

SPLIT_OVERLAP_PATH = (
    OUT_DIR
    / "t60_validation_test_overlap_audit.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "t60_evaluation_pool_collision_summary.csv"
)

EVENT_DETAIL_PATH = (
    OUT_DIR
    / "t60_evaluation_pool_collision_event_detail.parquet"
)

MANIFEST_PATH = (
    OUT_DIR
    / "phase_5_1_2a_audit_manifest.json"
)


# =============================================================================
# Frozen expected counts
# =============================================================================

EXPECTED = {
    "temporal_rows": 1_195_937,
    "t0_t59_rows": 1_173_422,
    "t60_rows": 22_515,
    "validation_rows": 2_251,
    "test_rows": 20_264,

    "node_rows": 477_564,
    "startup_nodes": 311_589,

    "startups_t0_t60": 309_306,
    "startups_pre_t60": 304_591,

    "t60_unique_pairs": 22_327,
    "t60_repeated_pairs": 172,

    "validation_test_pair_overlap": 33,
    "validation_test_funding_round_overlap": 1_315,
}

EVALUATION_NEGATIVE_COUNT = 99


# =============================================================================
# Candidate universes
# =============================================================================

CANDIDATE_UNIVERSES = (
    "global_role_universe",
    "experiment_any",
    "pre_t60_any",
)


# =============================================================================
# Exclusion policies
# =============================================================================

EXCLUSION_POLICIES = (
    "exclude_focal_only",
    "exclude_prior_pairs_plus_focal",
    "exclude_prior_plus_same_split_t60_pairs",
    "exclude_prior_plus_all_t60_pairs",
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


def probability_at_least_one_collision_without_replacement(
    population_size: int,
    collision_items: int,
    draws: int,
) -> float:
    """
    Exact:
        P(at least one collision)
        = 1 - C(N-M, K) / C(N, K)

    Implemented as a log-product to avoid very large combinations.

    No RNG is used.
    """

    N = int(population_size)
    M = int(collision_items)
    K = int(draws)

    require(N >= 0, "population_size must be nonnegative")
    require(M >= 0, "collision_items must be nonnegative")
    require(K >= 0, "draws must be nonnegative")
    require(M <= N, "collision_items cannot exceed population")

    if K == 0 or M == 0:
        return 0.0

    if K > N:
        return float("nan")

    if N - M < K:
        return 1.0

    log_p_none = 0.0

    for r in range(K):
        log_p_none += math.log(
            (N - M - r)
            / (N - r)
        )

    p_none = math.exp(log_p_none)

    return 1.0 - p_none


def duplicate_probability_with_replacement(
    population_size: int,
    draws: int,
) -> float:
    """
    Probability that at least one duplicate occurs if K draws
    are made WITH replacement from N candidates.

    Diagnostic only.
    """

    N = int(population_size)
    K = int(draws)

    require(N > 0, "population_size must be positive")
    require(K >= 0, "draws must be nonnegative")

    if K <= 1:
        return 0.0

    if K > N:
        return 1.0

    log_p_all_distinct = 0.0

    for r in range(K):
        log_p_all_distinct += math.log(
            (N - r) / N
        )

    return 1.0 - math.exp(
        log_p_all_distinct
    )


def set_for(
    mapping: dict,
    key,
) -> set:
    return mapping.get(
        key,
        set(),
    )


def build_grouped_sets(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> dict:
    return (
        df.groupby(
            group_col,
            sort=False,
            observed=True,
        )[value_col]
        .agg(lambda x: set(x.tolist()))
        .to_dict()
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.1.2a — "
        "T60 EVALUATION CANDIDATE-GENERATION "
        "LEAKAGE AND COLLISION AUDIT"
    )

    print("Evaluation negatives sampled:       0")
    print("RNG instantiated:                   NO")
    print("Candidate universe frozen:          NO")
    print("Exclusion policy frozen:            NO")
    print("Replacement semantics frozen:       NO")
    print("Evaluation seed frozen:             NO")
    print("Model instantiated:                 NO")
    print("Training performed:                 NO")
    print("Evaluation performed:               NO")

    # =========================================================================
    # Inputs
    # =========================================================================

    banner("AUTHORITATIVE INPUT EXISTENCE")

    for path in (
        TEMPORAL_PATH,
        NODE_INDEX_PATH,
        TRAINING_NEGATIVE_CONTRACT,
    ):
        require(
            path.exists(),
            f"Missing authoritative input: {path}",
        )

        print(f"FOUND  {path}")

    # =========================================================================
    # Recheck completed training-negative contract
    # =========================================================================

    banner("PHASE 5.1.1d TRAINING CONTRACT RECHECK")

    training_contract = json.loads(
        TRAINING_NEGATIVE_CONTRACT.read_text(
            encoding="utf-8"
        )
    )

    require(
        training_contract["phase"]
        == "5.1.1d",
        "Unexpected training-negative contract phase",
    )

    require(
        training_contract["status"]
        == "FROZEN",
        "Phase 5.1.1d contract is not frozen",
    )

    require(
        training_contract[
            "paper_unspecified_reproduction_choices"
        ][
            "training_negative_positive_ratio"
        ]
        == 4,
        "Frozen 4:1 training ratio drift",
    )

    require(
        training_contract[
            "paper_unspecified_reproduction_choices"
        ][
            "training_distribution"
        ]
        == "uniform_over_eligible_startups",
        "Frozen training distribution drift",
    )

    require(
        training_contract[
            "inherited_semantics"
        ][
            "exclude_future_positive_pairs"
        ]
        is False,
        "Training future-label guard drift",
    )

    print("Training-negative contract:          FROZEN  PASS")
    print("Training K:                          4       PASS")
    print("Training distribution:               uniform PASS")
    print("Future positives excluded in train:  NO      PASS")

    # =========================================================================
    # Load frozen data
    # =========================================================================

    banner("FROZEN PHASE-2 / PHASE-3 INPUT INTEGRITY")

    temporal = pd.read_parquet(
        TEMPORAL_PATH,
        columns=[
            "interaction_id",
            "funding_round_id",
            "investor_id",
            "startup_id",
            "segment_number",
            "experiment_split",
        ],
    )

    nodes = pd.read_parquet(
        NODE_INDEX_PATH,
        columns=[
            "node_index",
            "node_type",
            "raw_entity_id",
        ],
    )

    temporal["segment_number"] = (
        temporal["segment_number"]
        .astype(np.int16)
    )

    temporal["experiment_split"] = (
        temporal["experiment_split"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    require(
        len(temporal)
        == EXPECTED["temporal_rows"],
        "Temporal row-count drift",
    )

    require(
        len(nodes)
        == EXPECTED["node_rows"],
        "Node-index row-count drift",
    )

    startup_nodes = nodes.loc[
        nodes["node_type"] == "startup",
        "raw_entity_id",
    ].dropna()

    global_startups = set(
        startup_nodes.tolist()
    )

    require(
        len(global_startups)
        == EXPECTED["startup_nodes"],
        "Frozen Startup universe drift",
    )

    pre_t60 = temporal.loc[
        temporal["segment_number"] <= 59
    ].copy()

    t60 = temporal.loc[
        temporal["segment_number"] == 60
    ].copy()

    require(
        len(pre_t60)
        == EXPECTED["t0_t59_rows"],
        "T0-T59 count drift",
    )

    require(
        len(t60)
        == EXPECTED["t60_rows"],
        "T60 count drift",
    )

    validation = t60.loc[
        t60["experiment_split"]
        == "validation"
    ].copy()

    test = t60.loc[
        t60["experiment_split"]
        == "test"
    ].copy()

    require(
        len(validation)
        == EXPECTED["validation_rows"],
        "Validation count drift",
    )

    require(
        len(test)
        == EXPECTED["test_rows"],
        "Test count drift",
    )

    require(
        len(validation) + len(test)
        == len(t60),
        "T60 contains unexpected experiment_split values",
    )

    print(
        f"Temporal rows:             {len(temporal):>10,}  PASS"
    )
    print(
        f"T0-T59 rows:              {len(pre_t60):>10,}  PASS"
    )
    print(
        f"T60 rows:                 {len(t60):>10,}  PASS"
    )
    print(
        f"Validation rows:          {len(validation):>10,}  PASS"
    )
    print(
        f"Test rows:                {len(test):>10,}  PASS"
    )
    print(
        f"Startup role nodes:       {len(global_startups):>10,}  PASS"
    )

    # =========================================================================
    # Split-overlap recheck
    # =========================================================================

    banner("T60 VALIDATION / TEST OVERLAP AUDIT")

    validation_pairs = set(
        zip(
            validation["investor_id"],
            validation["startup_id"],
        )
    )

    test_pairs = set(
        zip(
            test["investor_id"],
            test["startup_id"],
        )
    )

    pair_overlap = (
        validation_pairs
        & test_pairs
    )

    validation_rounds = set(
        validation[
            "funding_round_id"
        ].dropna()
    )

    test_rounds = set(
        test[
            "funding_round_id"
        ].dropna()
    )

    round_overlap = (
        validation_rounds
        & test_rounds
    )

    t60_pair_counts = (
        t60.groupby(
            [
                "investor_id",
                "startup_id",
            ],
            sort=False,
            observed=True,
        )
        .size()
    )

    t60_unique_pairs = len(
        t60_pair_counts
    )

    t60_repeated_pairs = int(
        (
            t60_pair_counts > 1
        ).sum()
    )

    require(
        t60_unique_pairs
        == EXPECTED["t60_unique_pairs"],
        "T60 unique-pair count drift",
    )

    require(
        t60_repeated_pairs
        == EXPECTED["t60_repeated_pairs"],
        "T60 repeated-pair count drift",
    )

    require(
        len(pair_overlap)
        == EXPECTED[
            "validation_test_pair_overlap"
        ],
        "Validation/test pair-overlap drift",
    )

    require(
        len(round_overlap)
        == EXPECTED[
            "validation_test_funding_round_overlap"
        ],
        "Validation/test funding-round overlap drift",
    )

    split_overlap_df = pd.DataFrame(
        [
            {
                "metric": "t60_unique_pairs",
                "value": t60_unique_pairs,
                "expected": EXPECTED[
                    "t60_unique_pairs"
                ],
                "result": "PASS",
            },
            {
                "metric": "t60_repeated_pairs",
                "value": t60_repeated_pairs,
                "expected": EXPECTED[
                    "t60_repeated_pairs"
                ],
                "result": "PASS",
            },
            {
                "metric": (
                    "validation_test_pair_overlap"
                ),
                "value": len(pair_overlap),
                "expected": EXPECTED[
                    "validation_test_pair_overlap"
                ],
                "result": "PASS",
            },
            {
                "metric": (
                    "validation_test_funding_round_overlap"
                ),
                "value": len(round_overlap),
                "expected": EXPECTED[
                    "validation_test_funding_round_overlap"
                ],
                "result": "PASS",
            },
        ]
    )

    print(
        f"T60 unique pairs:                    "
        f"{t60_unique_pairs:,}  PASS"
    )

    print(
        f"T60 repeated pairs:                  "
        f"{t60_repeated_pairs:,}  PASS"
    )

    print(
        f"Validation/test pair overlap:         "
        f"{len(pair_overlap):,}  PASS"
    )

    print(
        f"Validation/test funding-round overlap:"
        f" {len(round_overlap):,}  PASS"
    )

    # =========================================================================
    # Candidate universes
    # =========================================================================

    banner("T60 CANDIDATE-UNIVERSE AUDIT")

    experiment_any_startups = set(
        temporal[
            "startup_id"
        ].dropna()
    )

    pre_t60_startups = set(
        pre_t60[
            "startup_id"
        ].dropna()
    )

    require(
        len(experiment_any_startups)
        == EXPECTED["startups_t0_t60"],
        "T0-T60 observed Startup count drift",
    )

    require(
        len(pre_t60_startups)
        == EXPECTED["startups_pre_t60"],
        "Pre-T60 observed Startup count drift",
    )

    require(
        experiment_any_startups
        <= global_startups,
        "Observed startup missing from frozen Startup role universe",
    )

    require(
        pre_t60_startups
        <= global_startups,
        "Pre-T60 startup missing from frozen Startup role universe",
    )

    universe_sets = {
        "global_role_universe": (
            global_startups
        ),
        "experiment_any": (
            experiment_any_startups
        ),
        "pre_t60_any": (
            pre_t60_startups
        ),
    }

    universe_rows = []

    for universe_name, universe in universe_sets.items():

        focal_inside = (
            t60["startup_id"]
            .isin(universe)
        )

        validation_inside = (
            validation["startup_id"]
            .isin(universe)
        )

        test_inside = (
            test["startup_id"]
            .isin(universe)
        )

        if universe_name == "global_role_universe":

            classification = (
                "DATASET_ADAPTATION_CANDIDATE"
            )

            uses_t60_labels_for_membership = False

            interpretation = (
                "Matches frozen training universe; "
                "static/transductive entity availability."
            )

        elif universe_name == "experiment_any":

            classification = (
                "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
            )

            uses_t60_labels_for_membership = True

            interpretation = (
                "Membership depends on whether Startup appears "
                "somewhere in T0-T60, including held-out T60."
            )

        else:

            classification = (
                "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
            )

            uses_t60_labels_for_membership = False

            interpretation = (
                "Investment-observed by T59 only; "
                "cold-start focal positives are inserted separately "
                "and therefore face an asymmetric negative universe."
            )

        universe_rows.append(
            {
                "candidate_universe": (
                    universe_name
                ),
                "candidate_universe_size": (
                    len(universe)
                ),
                "t60_focal_inside_share": float(
                    focal_inside.mean()
                ),
                "t60_focal_outside_share": float(
                    1.0
                    - focal_inside.mean()
                ),
                "validation_focal_outside_share": float(
                    1.0
                    - validation_inside.mean()
                ),
                "test_focal_outside_share": float(
                    1.0
                    - test_inside.mean()
                ),
                "uses_t60_labels_for_membership": (
                    uses_t60_labels_for_membership
                ),
                "classification": (
                    classification
                ),
                "interpretation": (
                    interpretation
                ),
            }
        )

    universe_df = pd.DataFrame(
        universe_rows
    )

    print(
        universe_df[
            [
                "candidate_universe",
                "candidate_universe_size",
                "t60_focal_outside_share",
                "uses_t60_labels_for_membership",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # First-positive pair timing
    # =========================================================================

    banner("PAIR FIRST-POSITIVE TIMING FOR T60")

    pair_first = (
        temporal.groupby(
            [
                "investor_id",
                "startup_id",
            ],
            sort=False,
            observed=True,
        )["segment_number"]
        .min()
        .rename("first_positive_segment")
        .reset_index()
    )

    historical_pairs = pair_first.loc[
        pair_first[
            "first_positive_segment"
        ] < 60
    ].copy()

    target_new_pairs = pair_first.loc[
        pair_first[
            "first_positive_segment"
        ] == 60
    ].copy()

    require(
        len(historical_pairs)
        + len(target_new_pairs)
        == len(pair_first),
        (
            "Unexpected pair-first segment beyond "
            "the T0-T60 experiment"
        ),
    )

    historical_by_investor = (
        build_grouped_sets(
            historical_pairs,
            "investor_id",
            "startup_id",
        )
    )

    target_new_by_investor = (
        build_grouped_sets(
            target_new_pairs,
            "investor_id",
            "startup_id",
        )
    )

    # Unique T60 pair membership in validation/test.
    validation_pair_df = (
        validation[
            [
                "investor_id",
                "startup_id",
            ]
        ]
        .drop_duplicates()
    )

    test_pair_df = (
        test[
            [
                "investor_id",
                "startup_id",
            ]
        ]
        .drop_duplicates()
    )

    target_new_key_set = set(
        zip(
            target_new_pairs["investor_id"],
            target_new_pairs["startup_id"],
        )
    )

    validation_target_new_pair_df = (
        validation_pair_df.loc[
            [
                pair in target_new_key_set
                for pair in zip(
                    validation_pair_df["investor_id"],
                    validation_pair_df["startup_id"],
                )
            ]
        ]
    )

    test_target_new_pair_df = (
        test_pair_df.loc[
            [
                pair in target_new_key_set
                for pair in zip(
                    test_pair_df["investor_id"],
                    test_pair_df["startup_id"],
                )
            ]
        ]
    )

    validation_target_new_by_investor = (
        build_grouped_sets(
            validation_target_new_pair_df,
            "investor_id",
            "startup_id",
        )
    )

    test_target_new_by_investor = (
        build_grouped_sets(
            test_target_new_pair_df,
            "investor_id",
            "startup_id",
        )
    )

    print(
        f"Historical positive pairs (<T60): "
        f"{len(historical_pairs):,}"
    )

    print(
        f"New positive pairs at T60:        "
        f"{len(target_new_pairs):,}"
    )

    # =========================================================================
    # Policy register
    # =========================================================================

    banner("EVALUATION EXCLUSION-POLICY RISK REGISTER")

    policy_df = pd.DataFrame(
        [
            {
                "exclusion_policy": (
                    "exclude_focal_only"
                ),
                "uses_pre_t60_history": False,
                "uses_other_validation_labels": False,
                "uses_other_test_labels": False,
                "historical_positive_collision_possible": True,
                "other_t60_positive_collision_possible": True,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "audit_assessment": (
                    "WEAK_BASELINE_ONLY"
                ),
                "reason": (
                    "Can label an already-observed historical "
                    "Investor-Startup pair as negative."
                ),
            },
            {
                "exclusion_policy": (
                    "exclude_prior_pairs_plus_focal"
                ),
                "uses_pre_t60_history": True,
                "uses_other_validation_labels": False,
                "uses_other_test_labels": False,
                "historical_positive_collision_possible": False,
                "other_t60_positive_collision_possible": True,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "audit_assessment": (
                    "STRICT_TARGET_LABEL_INDEPENDENT_CANDIDATE"
                ),
                "reason": (
                    "Uses only pre-T60 history plus the focal label; "
                    "other T60 outcomes cannot clean the pool."
                ),
            },
            {
                "exclusion_policy": (
                    "exclude_prior_plus_same_split_t60_pairs"
                ),
                "uses_pre_t60_history": True,
                "uses_other_validation_labels": True,
                "uses_other_test_labels": True,
                "historical_positive_collision_possible": False,
                "other_t60_positive_collision_possible": True,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "audit_assessment": (
                    "SPLIT_LABEL_AWARE"
                ),
                "reason": (
                    "Candidate construction depends on all positive "
                    "labels in the focal validation/test split and "
                    "still may leave positives belonging only to "
                    "the opposite split."
                ),
            },
            {
                "exclusion_policy": (
                    "exclude_prior_plus_all_t60_pairs"
                ),
                "uses_pre_t60_history": True,
                "uses_other_validation_labels": True,
                "uses_other_test_labels": True,
                "historical_positive_collision_possible": False,
                "other_t60_positive_collision_possible": False,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "audit_assessment": (
                    "POSITIVE_SAFE_BUT_HELDOUT_LABEL_AWARE"
                ),
                "reason": (
                    "Guarantees no held-out T60 positive is sampled "
                    "as a negative, but uses the full T60 outcome set "
                    "when constructing each candidate list."
                ),
            },
        ]
    )

    print(
        policy_df[
            [
                "exclusion_policy",
                "uses_other_validation_labels",
                "uses_other_test_labels",
                "audit_assessment",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # Exact event-level collision audit
    # =========================================================================

    banner(
        "EXACT 99-NEGATIVE COLLISION AUDIT — NO SAMPLING"
    )

    event_rows = []

    t60_eval = t60[
        [
            "interaction_id",
            "investor_id",
            "startup_id",
            "experiment_split",
        ]
    ].copy()

    for event in t60_eval.itertuples(
        index=False
    ):

        investor = event.investor_id
        focal_startup = event.startup_id
        split = event.experiment_split

        historical_all = set_for(
            historical_by_investor,
            investor,
        )

        target_new_all = set_for(
            target_new_by_investor,
            investor,
        )

        validation_target_new_all = set_for(
            validation_target_new_by_investor,
            investor,
        )

        test_target_new_all = set_for(
            test_target_new_by_investor,
            investor,
        )

        if split == "validation":

            same_split_target_new_all = (
                validation_target_new_all
            )

            opposite_split_target_new_all = (
                test_target_new_all
            )

        elif split == "test":

            same_split_target_new_all = (
                test_target_new_all
            )

            opposite_split_target_new_all = (
                validation_target_new_all
            )

        else:
            raise AssertionError(
                f"Unexpected T60 split: {split}"
            )

        for universe_name in CANDIDATE_UNIVERSES:

            universe = universe_sets[
                universe_name
            ]

            U = len(universe)

            focal_in_universe = (
                focal_startup
                in universe
            )

            historical = (
                historical_all
                & universe
            )

            target_new = (
                target_new_all
                & universe
            )

            same_split_target_new = (
                same_split_target_new_all
                & universe
            )

            opposite_split_target_new = (
                opposite_split_target_new_all
                & universe
            )

            for policy in EXCLUSION_POLICIES:

                if policy == "exclude_focal_only":

                    excluded = set()

                    if focal_in_universe:
                        excluded.add(
                            focal_startup
                        )

                elif policy == "exclude_prior_pairs_plus_focal":

                    excluded = set(
                        historical
                    )

                    if focal_in_universe:
                        excluded.add(
                            focal_startup
                        )

                elif policy == "exclude_prior_plus_same_split_t60_pairs":

                    excluded = (
                        set(historical)
                        | set(
                            same_split_target_new
                        )
                    )

                    if focal_in_universe:
                        excluded.add(
                            focal_startup
                        )

                elif policy == "exclude_prior_plus_all_t60_pairs":

                    excluded = (
                        set(historical)
                        | set(target_new)
                    )

                    if focal_in_universe:
                        excluded.add(
                            focal_startup
                        )

                else:
                    raise AssertionError(
                        f"Unknown policy: {policy}"
                    )

                negative_pool_size = (
                    U
                    - len(excluded)
                )

                require(
                    negative_pool_size >= 0,
                    "Negative pool became negative",
                )

                remaining_historical = (
                    historical
                    - excluded
                )

                remaining_target_new = (
                    target_new
                    - excluded
                )

                remaining_same_split_target_new = (
                    same_split_target_new
                    - excluded
                )

                remaining_opposite_split_target_new = (
                    opposite_split_target_new
                    - excluded
                )

                historical_collision_count = (
                    len(
                        remaining_historical
                    )
                )

                t60_collision_count = (
                    len(
                        remaining_target_new
                    )
                )

                same_split_collision_count = (
                    len(
                        remaining_same_split_target_new
                    )
                )

                opposite_split_collision_count = (
                    len(
                        remaining_opposite_split_target_new
                    )
                )

                if negative_pool_size > 0:

                    p_one_hist = (
                        historical_collision_count
                        / negative_pool_size
                    )

                    p_one_t60 = (
                        t60_collision_count
                        / negative_pool_size
                    )

                    expected_hist_in_99 = (
                        EVALUATION_NEGATIVE_COUNT
                        * p_one_hist
                    )

                    expected_t60_in_99 = (
                        EVALUATION_NEGATIVE_COUNT
                        * p_one_t60
                    )

                    p_any_hist_in_99 = (
                        probability_at_least_one_collision_without_replacement(
                            negative_pool_size,
                            historical_collision_count,
                            EVALUATION_NEGATIVE_COUNT,
                        )
                    )

                    p_any_t60_in_99 = (
                        probability_at_least_one_collision_without_replacement(
                            negative_pool_size,
                            t60_collision_count,
                            EVALUATION_NEGATIVE_COUNT,
                        )
                    )

                    duplicate_p_if_with_replacement = (
                        duplicate_probability_with_replacement(
                            negative_pool_size,
                            EVALUATION_NEGATIVE_COUNT,
                        )
                    )

                else:

                    p_one_hist = float("nan")
                    p_one_t60 = float("nan")

                    expected_hist_in_99 = (
                        float("nan")
                    )

                    expected_t60_in_99 = (
                        float("nan")
                    )

                    p_any_hist_in_99 = (
                        float("nan")
                    )

                    p_any_t60_in_99 = (
                        float("nan")
                    )

                    duplicate_p_if_with_replacement = (
                        float("nan")
                    )

                event_rows.append(
                    {
                        "interaction_id": (
                            event.interaction_id
                        ),
                        "experiment_split": (
                            split
                        ),
                        "candidate_universe": (
                            universe_name
                        ),
                        "exclusion_policy": (
                            policy
                        ),
                        "focal_startup_in_negative_universe": (
                            focal_in_universe
                        ),
                        "negative_pool_size": (
                            negative_pool_size
                        ),
                        "pool_supports_99": (
                            negative_pool_size
                            >= EVALUATION_NEGATIVE_COUNT
                        ),

                        "remaining_historical_positive_candidates": (
                            historical_collision_count
                        ),
                        "remaining_other_t60_positive_candidates": (
                            t60_collision_count
                        ),
                        "remaining_same_split_t60_positive_candidates": (
                            same_split_collision_count
                        ),
                        "remaining_opposite_split_t60_positive_candidates": (
                            opposite_split_collision_count
                        ),

                        "p_one_uniform_draw_hits_historical_positive": (
                            p_one_hist
                        ),
                        "p_one_uniform_draw_hits_other_t60_positive": (
                            p_one_t60
                        ),

                        "expected_historical_positive_collisions_in_99": (
                            expected_hist_in_99
                        ),
                        "expected_other_t60_positive_collisions_in_99": (
                            expected_t60_in_99
                        ),

                        "p_99_contains_at_least_one_historical_positive": (
                            p_any_hist_in_99
                        ),
                        "p_99_contains_at_least_one_other_t60_positive": (
                            p_any_t60_in_99
                        ),

                        "duplicate_probability_if_99_with_replacement": (
                            duplicate_p_if_with_replacement
                        ),
                    }
                )

    event_df = pd.DataFrame(
        event_rows
    )

    # =========================================================================
    # Aggregate
    # =========================================================================

    banner("AGGREGATE VALIDATION / TEST / T60 SUMMARY")

    summary_rows = []

    scopes = {
        "validation": (
            event_df[
                "experiment_split"
            ]
            == "validation"
        ),
        "test": (
            event_df[
                "experiment_split"
            ]
            == "test"
        ),
        "t60_overall": pd.Series(
            True,
            index=event_df.index,
        ),
    }

    for scope_name, scope_mask in scopes.items():

        scoped = event_df.loc[
            scope_mask
        ]

        for universe_name in CANDIDATE_UNIVERSES:

            for policy in EXCLUSION_POLICIES:

                part = scoped.loc[
                    (
                        scoped[
                            "candidate_universe"
                        ]
                        == universe_name
                    )
                    & (
                        scoped[
                            "exclusion_policy"
                        ]
                        == policy
                    )
                ]

                require(
                    len(part) > 0,
                    (
                        "Missing aggregate slice: "
                        f"{scope_name} / "
                        f"{universe_name} / "
                        f"{policy}"
                    ),
                )

                summary_rows.append(
                    {
                        "scope": (
                            scope_name
                        ),
                        "candidate_universe": (
                            universe_name
                        ),
                        "exclusion_policy": (
                            policy
                        ),
                        "positive_events": (
                            len(part)
                        ),

                        "focal_outside_negative_universe_share": (
                            1.0
                            - part[
                                "focal_startup_in_negative_universe"
                            ].mean()
                        ),

                        "negative_pool_min": int(
                            part[
                                "negative_pool_size"
                            ].min()
                        ),

                        "negative_pool_mean": float(
                            part[
                                "negative_pool_size"
                            ].mean()
                        ),

                        "share_events_with_pool_lt_99": float(
                            (
                                ~part[
                                    "pool_supports_99"
                                ]
                            ).mean()
                        ),

                        "mean_p_one_draw_hits_historical_positive": float(
                            part[
                                "p_one_uniform_draw_hits_historical_positive"
                            ].mean()
                        ),

                        "mean_p_one_draw_hits_other_t60_positive": float(
                            part[
                                "p_one_uniform_draw_hits_other_t60_positive"
                            ].mean()
                        ),

                        "mean_expected_historical_positive_collisions_in_99": float(
                            part[
                                "expected_historical_positive_collisions_in_99"
                            ].mean()
                        ),

                        "mean_expected_other_t60_positive_collisions_in_99": float(
                            part[
                                "expected_other_t60_positive_collisions_in_99"
                            ].mean()
                        ),

                        "mean_p_99_contains_at_least_one_historical_positive": float(
                            part[
                                "p_99_contains_at_least_one_historical_positive"
                            ].mean()
                        ),

                        "mean_p_99_contains_at_least_one_other_t60_positive": float(
                            part[
                                "p_99_contains_at_least_one_other_t60_positive"
                            ].mean()
                        ),

                        "share_events_with_any_other_t60_positive_candidate": float(
                            (
                                part[
                                    "remaining_other_t60_positive_candidates"
                                ]
                                > 0
                            ).mean()
                        ),

                        "share_events_with_opposite_split_t60_positive_candidate": float(
                            (
                                part[
                                    "remaining_opposite_split_t60_positive_candidates"
                                ]
                                > 0
                            ).mean()
                        ),

                        "mean_duplicate_probability_if_99_with_replacement": float(
                            part[
                                "duplicate_probability_if_99_with_replacement"
                            ].mean()
                        ),
                    }
                )

    summary_df = pd.DataFrame(
        summary_rows
    )

    global_summary = summary_df.loc[
        (
            summary_df[
                "scope"
            ]
            == "t60_overall"
        )
        & (
            summary_df[
                "candidate_universe"
            ]
            == "global_role_universe"
        )
    ]

    print(
        global_summary[
            [
                "exclusion_policy",
                "positive_events",
                "negative_pool_min",
                "negative_pool_mean",
                "mean_p_one_draw_hits_historical_positive",
                "mean_p_one_draw_hits_other_t60_positive",
                "mean_expected_other_t60_positive_collisions_in_99",
                "mean_p_99_contains_at_least_one_other_t60_positive",
                "share_events_with_pool_lt_99",
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
        len(t60)
        == EXPECTED["t60_rows"],
        "T60 population changed",
    )

    require(
        len(validation)
        == EXPECTED["validation_rows"],
        "Validation population changed",
    )

    require(
        len(test)
        == EXPECTED["test_rows"],
        "Test population changed",
    )

    require(
        event_df[
            "pool_supports_99"
        ].all(),
        (
            "At least one audited evaluation pool "
            "cannot support 99 negatives"
        ),
    )

    # Causal policy + global universe must remove
    # all historical-positive collisions.
    causal_global = summary_df.loc[
        (
            summary_df[
                "scope"
            ]
            == "t60_overall"
        )
        & (
            summary_df[
                "candidate_universe"
            ]
            == "global_role_universe"
        )
        & (
            summary_df[
                "exclusion_policy"
            ]
            == "exclude_prior_pairs_plus_focal"
        )
    ]

    require(
        len(causal_global) == 1,
        "Missing causal-global summary row",
    )

    require(
        float(
            causal_global.iloc[0][
                "mean_p_one_draw_hits_historical_positive"
            ]
        )
        == 0.0,
        (
            "Historical-positive collisions remain "
            "under prior-plus-focal policy"
        ),
    )

    # Full T60-label-aware policy must remove
    # all other T60 positive collisions.
    label_aware_global = summary_df.loc[
        (
            summary_df[
                "scope"
            ]
            == "t60_overall"
        )
        & (
            summary_df[
                "candidate_universe"
            ]
            == "global_role_universe"
        )
        & (
            summary_df[
                "exclusion_policy"
            ]
            == "exclude_prior_plus_all_t60_pairs"
        )
    ]

    require(
        len(label_aware_global) == 1,
        "Missing full-label-aware summary row",
    )

    require(
        float(
            label_aware_global.iloc[0][
                "mean_p_one_draw_hits_other_t60_positive"
            ]
        )
        == 0.0,
        (
            "T60-positive collisions remain under "
            "all-T60-positive exclusion"
        ),
    )

    print(
        "Frozen T60 row count unchanged:            PASS"
    )
    print(
        "Frozen validation row count unchanged:     PASS"
    )
    print(
        "Frozen test row count unchanged:           PASS"
    )
    print(
        "Validation/test pair overlap unchanged:    PASS"
    )
    print(
        "Funding-round overlap unchanged:           PASS"
    )
    print(
        "All audited pools support 99 negatives:    PASS"
    )
    print(
        "Prior-plus-focal removes history collision:PASS"
    )
    print(
        "All-T60 exclusion removes T60 collision:   PASS"
    )
    print(
        "Evaluation negatives generated:            0"
    )
    print(
        "RNG instantiated:                          NO"
    )
    print(
        "Training performed:                        NO"
    )
    print(
        "Evaluation performed:                      NO"
    )
    print(
        "Evaluation contract frozen:                NO"
    )

    # =========================================================================
    # Write
    # =========================================================================

    banner("WRITE AUDIT-ONLY OUTPUTS")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    universe_df.to_csv(
        CANDIDATE_REGISTER_PATH,
        index=False,
    )

    policy_df.to_csv(
        POLICY_REGISTER_PATH,
        index=False,
    )

    split_overlap_df.to_csv(
        SPLIT_OVERLAP_PATH,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    event_df.to_parquet(
        EVENT_DETAIL_PATH,
        index=False,
    )

    manifest = {
        "phase": "5.1.2a",
        "title": (
            "T60 Evaluation Candidate-Generation "
            "Leakage and Collision Audit"
        ),
        "status": (
            "AUDIT_COMPLETE_NOT_FROZEN"
        ),

        "evaluation_negative_samples_generated": False,
        "rng_instantiated": False,
        "model_instantiated": False,
        "optimizer_created": False,
        "training_performed": False,
        "evaluation_performed": False,

        "paper_specified": {
            "positive_per_case": 1,
            "random_negatives_per_case": (
                EVALUATION_NEGATIVE_COUNT
            ),
            "metrics": [
                "HR@10",
                "NDCG@10",
            ],
        },

        "candidate_universes_audited": list(
            CANDIDATE_UNIVERSES
        ),

        "exclusion_policies_audited": list(
            EXCLUSION_POLICIES
        ),

        "collision_probability_contract": (
            "Analytical only; 99-negative at-least-one "
            "probability computed under hypothetical "
            "uniform sampling without replacement."
        ),

        "important_interpretation_guards": [
            (
                "Other T60 positives are genuine held-out "
                "positive outcomes, not automatically valid negatives."
            ),
            (
                "Using other T60 labels to remove those candidates "
                "is label-aware candidate construction."
            ),
            (
                "Not using other T60 labels preserves target-label "
                "independence but can create multi-positive collisions."
            ),
            (
                "pre_t60_any may be temporally clean for negative "
                "membership but creates warm-negative/cold-positive "
                "candidate asymmetry."
            ),
            (
                "The frozen event-level validation/test split is "
                "not modified by this audit."
            ),
        ],

        "still_unresolved_original_phase_5_handoff_decisions": [
            "training epoch count",
            "early stopping",
            "weight decay",
            (
                "evaluation candidate-generation "
                "runtime contract"
            ),
        ],

        "outputs": [
            str(CANDIDATE_REGISTER_PATH),
            str(POLICY_REGISTER_PATH),
            str(SPLIT_OVERLAP_PATH),
            str(SUMMARY_PATH),
            str(EVENT_DETAIL_PATH),
        ],
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    for path in (
        CANDIDATE_REGISTER_PATH,
        POLICY_REGISTER_PATH,
        SPLIT_OVERLAP_PATH,
        SUMMARY_PATH,
        EVENT_DETAIL_PATH,
        MANIFEST_PATH,
    ):
        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Decision-facing summary
    # =========================================================================

    banner(
        "DECISION-FACING SUMMARY — "
        "NOTHING NEW FROZEN"
    )

    print()
    print(
        "Candidate universes:"
    )

    print(
        universe_df[
            [
                "candidate_universe",
                "candidate_universe_size",
                "t60_focal_outside_share",
                "uses_t60_labels_for_membership",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Global-role-universe exclusion policies:"
    )

    print(
        global_summary[
            [
                "exclusion_policy",
                "mean_p_one_draw_hits_historical_positive",
                "mean_p_one_draw_hits_other_t60_positive",
                "mean_expected_other_t60_positive_collisions_in_99",
                "mean_p_99_contains_at_least_one_other_t60_positive",
                "share_events_with_opposite_split_t60_positive_candidate",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Interpretation guards:")
    print(
        "1. exclude_focal_only can relabel historical positives "
        "as negatives."
    )
    print(
        "2. exclude_prior_pairs_plus_focal uses no other T60 labels."
    )
    print(
        "3. same-split exclusion is held-out-label-aware and can "
        "still leave opposite-split positives."
    )
    print(
        "4. all-T60 exclusion is positive-safe but uses the complete "
        "held-out outcome set."
    )
    print(
        "5. Candidate-pool capacity is not expected to constrain "
        "the paper-specified 99 negatives."
    )
    print(
        "6. No evaluation candidate policy is frozen by this script."
    )

    banner(
        "PHASE 5.1.2a COMPLETE"
    )

    print(
        "AUDIT COMPLETE — "
        "NO EVALUATION NEGATIVES GENERATED — "
        "NO EVALUATION CONTRACT FROZEN"
    )


if __name__ == "__main__":
    main()