"""
Phase 5.1.1a — Negative-Sampling Feasibility and Temporal-Collision Audit

Audit only. This script:
- reads the frozen Phase-2 temporal split and Phase-3 node index;
- audits several candidate-universe and exclusion-policy options;
- computes exact pool sizes/collision probabilities algebraically;
- DOES NOT sample negatives, instantiate an RNG, train, or freeze a policy.

Inherited target indexing:
- T0 = compressed history only
- T1..T59 = training target periods
- T60 = frozen validation/test target period
"""

from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Frozen inputs / audit outputs
# -----------------------------------------------------------------------------

TEMPORAL_PATH = Path(
    "data/experimental/phase_2/model_ready/interactions_itrs_temporal_split.parquet"
)
NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/model_ready/node_index.parquet"
)

OUT_DIR = Path(
    "data/experimental/phase_5/audits/phase_5_1_1a"
)

CANDIDATE_COMPOSITION_PATH = (
    OUT_DIR / "candidate_universe_temporal_composition_by_segment.csv"
)
POLICY_RISK_PATH = (
    OUT_DIR / "negative_sampling_policy_risk_register.csv"
)
SEGMENT_AUDIT_PATH = (
    OUT_DIR / "negative_pool_feasibility_and_collision_by_segment.csv"
)
SCOPE_AUDIT_PATH = (
    OUT_DIR / "negative_pool_feasibility_and_collision_overall_by_scope.csv"
)
DELTA_PATH = (
    OUT_DIR / "future_and_target_label_exclusion_deltas.csv"
)
MANIFEST_PATH = (
    OUT_DIR / "phase_5_1_1a_audit_manifest.json"
)


EXPECTED = {
    "temporal_rows": 1_195_937,
    "t0_rows": 100_173,
    "t1_t59_rows": 1_073_249,
    "t0_t59_rows": 1_173_422,
    "t60_rows": 22_515,
    "validation_rows": 2_251,
    "test_rows": 20_264,
    "node_rows": 477_564,
    "investor_nodes": 165_975,
    "startup_nodes": 311_589,
}


# Capacity probes only.
#
# IMPORTANT:
# These values DO NOT freeze the training negative:positive ratio.
# 99 is included because ITRS explicitly uses 99 negatives at evaluation.
REFERENCE_NEGATIVE_COUNTS = (1, 4, 10, 20, 50, 99)


# -----------------------------------------------------------------------------
# Candidate-universe policies to AUDIT, not freeze
# -----------------------------------------------------------------------------

CANDIDATE_UNIVERSES = (
    "global_role_universe",
    "experiment_any",
    "pre_t60_any",
    "prefix_prior",
    "prefix_through_target",
)


# -----------------------------------------------------------------------------
# Positive-exclusion policies to AUDIT, not freeze
# -----------------------------------------------------------------------------

EXCLUSION_POLICIES = (
    "exclude_current_positive_only",
    "exclude_prior_positive_pairs_plus_current",
    "exclude_prior_and_target_period_positive_pairs",
    "exclude_all_experiment_positive_pairs",
)


# =============================================================================
# Generic helpers
# =============================================================================

def banner(text: str) -> None:
    print("\n" + "=" * 118)
    print(text)
    print("=" * 118)


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def div0(num, den):
    if den == 0:
        return np.nan
    return float(num) / float(den)


# =============================================================================
# Candidate-universe semantics
# =============================================================================

def startup_eligible_mask(
    first_seg: pd.Series,
    h: int,
    universe: str,
) -> np.ndarray:
    """
    Return whether each Startup is eligible to be a NEGATIVE candidate
    under the named audit universe at target segment T_h.

    first_seg:
        Startup's first observed investment segment in the frozen
        T0..T60 temporal experiment.

    Important:
        global_role_universe can also contain Startup role nodes that
        never appear in T0..T60. Those nodes are represented in its
        universe size, even though this mask operates only over Startups
        actually observed in the temporal experiment.
    """

    fs = first_seg.to_numpy(
        dtype=np.int16,
        copy=False,
    )

    if universe in {
        "global_role_universe",
        "experiment_any",
    }:
        return np.ones(
            len(fs),
            dtype=bool,
        )

    if universe == "pre_t60_any":
        return fs <= 59

    if universe == "prefix_prior":
        return fs < h

    if universe == "prefix_through_target":
        return fs <= h

    raise KeyError(
        f"Unknown candidate universe: {universe}"
    )


def universe_size(
    first_counts: dict[int, int],
    experiment_n: int,
    h: int,
    universe: str,
) -> int:
    """
    Exact size of each negative-candidate universe.
    """

    if universe == "global_role_universe":
        return EXPECTED["startup_nodes"]

    if universe == "experiment_any":
        return experiment_n

    if universe == "pre_t60_any":
        return sum(
            value
            for segment, value in first_counts.items()
            if segment <= 59
        )

    if universe == "prefix_prior":
        return sum(
            value
            for segment, value in first_counts.items()
            if segment < h
        )

    if universe == "prefix_through_target":
        return sum(
            value
            for segment, value in first_counts.items()
            if segment <= h
        )

    raise KeyError(
        f"Unknown candidate universe: {universe}"
    )


# =============================================================================
# Exclusion-policy algebra
# =============================================================================

def policy_arrays(
    policy: str,
    N: int,
    H: np.ndarray,
    T: np.ndarray,
    F: np.ndarray,
    F60: np.ndarray,
):
    """
    Construct exact negative-pool and collision counts.

    Per investor at target T_h:

        N = negative-candidate universe size

        H = number of eligible Investor-Startup pairs whose first
            positive is before T_h

        T = number of eligible Investor-Startup pairs whose first
            positive is in T_h

        F = number of eligible Investor-Startup pairs whose first
            positive is after T_h

        F60 = subset of F whose first positive occurs specifically
              in T60

    Three focal-event classes are represented by the tuple positions:

        0 = focal positive is negative-eligible and pair is historical
        1 = focal positive is negative-eligible and pair first occurs at T_h
        2 = focal Startup lies outside this negative candidate universe

    Return:
        pools,
        historical collisions,
        other-target collisions,
        future collisions,
        T60-future collisions

    No random sampling occurs.
    """

    Nvec = np.full_like(
        H,
        int(N),
        dtype=np.int64,
    )

    zero = np.zeros_like(
        H,
        dtype=np.int64,
    )

    # -------------------------------------------------------------------------
    # Policy 1:
    # remove only the focal positive if it belongs to the negative universe
    # -------------------------------------------------------------------------
    if policy == "exclude_current_positive_only":

        return (
            (
                Nvec - 1,
                Nvec - 1,
                Nvec,
            ),
            (
                H - 1,
                H,
                H,
            ),
            (
                T,
                T - 1,
                T,
            ),
            (
                F,
                F,
                F,
            ),
            (
                F60,
                F60,
                F60,
            ),
        )

    # -------------------------------------------------------------------------
    # Policy 2:
    # remove every historical positive for the investor + focal positive
    # -------------------------------------------------------------------------
    if policy == "exclude_prior_positive_pairs_plus_current":

        return (
            (
                Nvec - H,
                Nvec - H - 1,
                Nvec - H,
            ),
            (
                zero,
                zero,
                zero,
            ),
            (
                T,
                T - 1,
                T,
            ),
            (
                F,
                F,
                F,
            ),
            (
                F60,
                F60,
                F60,
            ),
        )

    # -------------------------------------------------------------------------
    # Policy 3:
    # remove every investor positive through the target segment
    # -------------------------------------------------------------------------
    if policy == "exclude_prior_and_target_period_positive_pairs":

        pool = (
            Nvec
            - H
            - T
        )

        return (
            (
                pool,
                pool,
                pool,
            ),
            (
                zero,
                zero,
                zero,
            ),
            (
                zero,
                zero,
                zero,
            ),
            (
                F,
                F,
                F,
            ),
            (
                F60,
                F60,
                F60,
            ),
        )

    # -------------------------------------------------------------------------
    # Policy 4:
    # remove every pair that is ever positive in T0..T60
    #
    # This is intentionally audited because, during T1..T59,
    # this consumes future outcome labels.
    # -------------------------------------------------------------------------
    if policy == "exclude_all_experiment_positive_pairs":

        pool = (
            Nvec
            - H
            - T
            - F
        )

        return (
            (
                pool,
                pool,
                pool,
            ),
            (
                zero,
                zero,
                zero,
            ),
            (
                zero,
                zero,
                zero,
            ),
            (
                zero,
                zero,
                zero,
            ),
            (
                zero,
                zero,
                zero,
            ),
        )

    raise KeyError(
        f"Unknown exclusion policy: {policy}"
    )


# =============================================================================
# Exact event-weighted aggregation
# =============================================================================

def summarize_policy(
    stats: pd.DataFrame,
    N: int,
    policy: str,
) -> dict:
    """
    Compute exact event-weighted negative-pool and collision diagnostics.

    No Investor x Startup Cartesian candidate table is materialized.
    No random negatives are generated.
    """

    event_classes = [
        stats[
            "eligible_history"
        ].to_numpy(
            dtype=np.int64
        ),
        stats[
            "eligible_target"
        ].to_numpy(
            dtype=np.int64
        ),
        stats[
            "ineligible"
        ].to_numpy(
            dtype=np.int64
        ),
    ]

    H = stats[
        "history"
    ].to_numpy(
        dtype=np.int64
    )

    T = stats[
        "target_new"
    ].to_numpy(
        dtype=np.int64
    )

    F = stats[
        "future"
    ].to_numpy(
        dtype=np.int64
    )

    F60 = stats[
        "t60_future"
    ].to_numpy(
        dtype=np.int64
    )

    (
        pools,
        historical_collisions,
        target_collisions,
        future_collisions,
        t60_collisions,
    ) = policy_arrays(
        policy,
        N,
        H,
        T,
        F,
        F60,
    )

    total_events = int(
        sum(
            event_count.sum()
            for event_count in event_classes
        )
    )

    focal_outside = int(
        event_classes[2].sum()
    )

    require(
        total_events > 0,
        "Unexpected empty target/split group",
    )

    pool_sum = 0.0
    pool_min = None
    pool_max = None

    collision_sums = np.zeros(
        5,
        dtype=np.float64,
    )

    probability_sums = np.zeros(
        5,
        dtype=np.float64,
    )

    zero_pool_events = 0

    capacity_lt = {
        k: 0
        for k in REFERENCE_NEGATIVE_COUNTS
    }

    for class_index, event_count in enumerate(
        event_classes
    ):

        if int(
            event_count.sum()
        ) == 0:
            continue

        pool = np.asarray(
            pools[class_index],
            dtype=np.int64,
        )

        hist_collision = np.asarray(
            historical_collisions[class_index],
            dtype=np.int64,
        )

        target_collision = np.asarray(
            target_collisions[class_index],
            dtype=np.int64,
        )

        future_collision = np.asarray(
            future_collisions[class_index],
            dtype=np.int64,
        )

        t60_collision = np.asarray(
            t60_collisions[class_index],
            dtype=np.int64,
        )

        active = (
            event_count > 0
        )

        # Values such as H-1 or T-1 can be -1 for inactive rows.
        # They only need to be valid where this event class actually exists.
        require(
            np.all(
                pool[active] >= 0
            ),
            f"{policy}: negative pool became negative",
        )

        require(
            np.all(
                hist_collision[active] >= 0
            ),
            f"{policy}: historical collision count became negative",
        )

        require(
            np.all(
                target_collision[active] >= 0
            ),
            f"{policy}: target collision count became negative",
        )

        require(
            np.all(
                future_collision[active] >= 0
            ),
            f"{policy}: future collision count became negative",
        )

        require(
            np.all(
                t60_collision[active] >= 0
            ),
            f"{policy}: T60 future collision count became negative",
        )

        require(
            np.all(
                (
                    hist_collision
                    + target_collision
                    + future_collision
                )[active]
                <= pool[active]
            ),
            f"{policy}: collision classes exceed candidate pool",
        )

        require(
            np.all(
                t60_collision[active]
                <= future_collision[active]
            ),
            f"{policy}: T60 future class is not subset of future class",
        )

        pool_active = (
            pool[active]
        )

        if pool_min is None:
            pool_min = int(
                pool_active.min()
            )
            pool_max = int(
                pool_active.max()
            )
        else:
            pool_min = min(
                pool_min,
                int(
                    pool_active.min()
                ),
            )

            pool_max = max(
                pool_max,
                int(
                    pool_active.max()
                ),
            )

        pool_sum += float(
            np.sum(
                event_count
                * pool
            )
        )

        eventual_collision = (
            hist_collision
            + target_collision
            + future_collision
        )

        collision_arrays = (
            hist_collision,
            target_collision,
            future_collision,
            t60_collision,
            eventual_collision,
        )

        for j, collision in enumerate(
            collision_arrays
        ):

            collision_sums[j] += float(
                np.sum(
                    event_count
                    * collision
                )
            )

        positive_pool = (
            pool > 0
        )

        zero_pool_events += int(
            event_count[
                ~positive_pool
            ].sum()
        )

        for j, collision in enumerate(
            collision_arrays
        ):

            probability = np.zeros(
                len(pool),
                dtype=np.float64,
            )

            probability[
                positive_pool
            ] = (
                collision[
                    positive_pool
                ]
                / pool[
                    positive_pool
                ]
            )

            probability_sums[j] += float(
                np.sum(
                    event_count
                    * probability
                )
            )

        for k in REFERENCE_NEGATIVE_COUNTS:

            capacity_lt[k] += int(
                event_count[
                    pool < k
                ].sum()
            )

    result = {
        "positive_events": total_events,
        "focal_outside_negative_universe_events": focal_outside,
        "focal_outside_negative_universe_share": div0(
            focal_outside,
            total_events,
        ),
        "negative_pool_min": int(
            pool_min
        ),
        "negative_pool_mean": div0(
            pool_sum,
            total_events,
        ),
        "negative_pool_max": int(
            pool_max
        ),
        "mean_historical_positive_pairs_remaining": div0(
            collision_sums[0],
            total_events,
        ),
        "mean_other_target_positive_pairs_remaining": div0(
            collision_sums[1],
            total_events,
        ),
        "mean_future_positive_pairs_remaining": div0(
            collision_sums[2],
            total_events,
        ),
        "mean_t60_future_positive_pairs_remaining": div0(
            collision_sums[3],
            total_events,
        ),
        "mean_any_eventual_positive_pairs_remaining": div0(
            collision_sums[4],
            total_events,
        ),
        "p_uniform_draw_hits_historical_positive": div0(
            probability_sums[0],
            total_events,
        ),
        "p_uniform_draw_hits_other_target_positive": div0(
            probability_sums[1],
            total_events,
        ),
        "p_uniform_draw_hits_future_positive": div0(
            probability_sums[2],
            total_events,
        ),
        "p_uniform_draw_hits_t60_future_positive": div0(
            probability_sums[3],
            total_events,
        ),
        "p_uniform_draw_hits_any_eventual_positive": div0(
            probability_sums[4],
            total_events,
        ),
        "zero_pool_events": zero_pool_events,
    }

    for k in REFERENCE_NEGATIVE_COUNTS:

        result[
            f"events_with_pool_lt_{k}"
        ] = (
            capacity_lt[k]
        )

        result[
            f"share_events_with_pool_lt_{k}"
        ] = div0(
            capacity_lt[k],
            total_events,
        )

    return result


# =============================================================================
# Reaggregate segment-level results
# =============================================================================

def combine_weighted(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:

    weighted_cols = [
        "negative_pool_mean",
        "mean_historical_positive_pairs_remaining",
        "mean_other_target_positive_pairs_remaining",
        "mean_future_positive_pairs_remaining",
        "mean_t60_future_positive_pairs_remaining",
        "mean_any_eventual_positive_pairs_remaining",
        "p_uniform_draw_hits_historical_positive",
        "p_uniform_draw_hits_other_target_positive",
        "p_uniform_draw_hits_future_positive",
        "p_uniform_draw_hits_t60_future_positive",
        "p_uniform_draw_hits_any_eventual_positive",
    ]

    rows = []

    for keys, group in df.groupby(
        group_cols,
        sort=True,
        dropna=False,
    ):

        if not isinstance(
            keys,
            tuple,
        ):
            keys = (
                keys,
            )

        total_events = int(
            group[
                "positive_events"
            ].sum()
        )

        result = dict(
            zip(
                group_cols,
                keys,
            )
        )

        result[
            "positive_events"
        ] = total_events

        result[
            "candidate_universe_size_min"
        ] = int(
            group[
                "candidate_universe_size"
            ].min()
        )

        result[
            "candidate_universe_size_max"
        ] = int(
            group[
                "candidate_universe_size"
            ].max()
        )

        result[
            "negative_pool_min"
        ] = int(
            group[
                "negative_pool_min"
            ].min()
        )

        result[
            "negative_pool_max"
        ] = int(
            group[
                "negative_pool_max"
            ].max()
        )

        result[
            "zero_pool_events"
        ] = int(
            group[
                "zero_pool_events"
            ].sum()
        )

        focal_outside = int(
            group[
                "focal_outside_negative_universe_events"
            ].sum()
        )

        result[
            "focal_outside_negative_universe_events"
        ] = focal_outside

        result[
            "focal_outside_negative_universe_share"
        ] = div0(
            focal_outside,
            total_events,
        )

        for column in weighted_cols:

            result[
                column
            ] = div0(
                (
                    group[
                        column
                    ]
                    * group[
                        "positive_events"
                    ]
                ).sum(),
                total_events,
            )

        for k in REFERENCE_NEGATIVE_COUNTS:

            count_col = (
                f"events_with_pool_lt_{k}"
            )

            count = int(
                group[
                    count_col
                ].sum()
            )

            result[
                count_col
            ] = count

            result[
                f"share_events_with_pool_lt_{k}"
            ] = div0(
                count,
                total_events,
            )

        rows.append(
            result
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Methodological risk register
# =============================================================================

def build_policy_risk_register() -> pd.DataFrame:

    return pd.DataFrame(
        [
            [
                "paper_training_method",
                "random_training_negatives",
                "PAPER_SPECIFIED",
                (
                    "Random negatives are specified; "
                    "training ratio and detailed exclusions are not."
                ),
                "INHERITED_NOT_SAMPLED",
            ],
            [
                "paper_evaluation_protocol",
                "1_positive_plus_99_random_negatives",
                "PAPER_SPECIFIED",
                (
                    "99 negatives per positive are specified; "
                    "runtime eligibility/exclusion semantics remain open."
                ),
                "INHERITED_NOT_SAMPLED",
            ],
            [
                "candidate_universe",
                "global_role_universe",
                "DATASET_ADAPTATION_CANDIDATE",
                (
                    "Uses all 311,589 frozen Startup role nodes. "
                    "No target labels define membership, but the universe "
                    "is static/transductive rather than historically versioned."
                ),
                "AUDIT_OPTION_NOT_FROZEN",
            ],
            [
                "candidate_universe",
                "experiment_any",
                "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE",
                (
                    "Uses whether a Startup appears anywhere in T0..T60, "
                    "so historical targets use future interaction observations "
                    "to define eligibility."
                ),
                "AUDIT_OPTION_NOT_FROZEN",
            ],
            [
                "candidate_universe",
                "pre_t60_any",
                "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE",
                (
                    "Uses whether a Startup appears anywhere in T0..T59; "
                    "still future-aware for early historical targets."
                ),
                "AUDIT_OPTION_NOT_FROZEN",
            ],
            [
                "candidate_universe",
                "prefix_prior",
                "DATASET_ADAPTATION_CANDIDATE",
                (
                    "Requires Startup first observed investment < T_h. "
                    "Temporally causal for interaction visibility, but visibility "
                    "is only a proxy for company availability and excludes "
                    "interaction-cold candidates."
                ),
                "AUDIT_OPTION_NOT_FROZEN",
            ],
            [
                "candidate_universe",
                "prefix_through_target",
                "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE",
                (
                    "Allows first observed investment <= T_h, "
                    "so target-period labels help determine eligibility."
                ),
                "AUDIT_OPTION_NOT_FROZEN",
            ],
            [
                "exclusion_policy",
                "exclude_current_positive_only",
                "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE",
                (
                    "Can label other historical positives or other "
                    "same-segment positives as negatives."
                ),
                "AUDIT_OPTION_NOT_FROZEN",
            ],
            [
                "exclusion_policy",
                "exclude_prior_positive_pairs_plus_current",
                "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE",
                (
                    "Causally removes prior Investor-Startup positives; "
                    "other same-target-period positives may remain."
                ),
                "AUDIT_OPTION_NOT_FROZEN",
            ],
            [
                "exclusion_policy",
                "exclude_prior_and_target_period_positive_pairs",
                "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE",
                (
                    "Avoids same-period positive collisions. "
                    "In training it is target-label-aware; in T60 evaluation "
                    "it consumes held-out target labels beyond the focal positive."
                ),
                "AUDIT_OPTION_NOT_FROZEN",
            ],
            [
                "exclusion_policy",
                "exclude_all_experiment_positive_pairs",
                "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE",
                (
                    "Removes all observed-positive collisions, but historical "
                    "training uses future interaction labels and therefore leaks "
                    "temporal outcome information."
                ),
                "AUDIT_OPTION_NOT_FROZEN",
            ],
            [
                "audit_method",
                "exact_algebra_without_sampling",
                "IMPLEMENTATION_EQUIVALENT_AUDIT_CHOICE",
                (
                    "Exact counts/probabilities are computed without random "
                    "draws, so the audit cannot pre-commit the future "
                    "sampling contract."
                ),
                "AUDIT_ONLY",
            ],
        ],
        columns=[
            "decision_family",
            "option",
            "classification",
            "main_risk_or_implication",
            "status_after_5_1_1a",
        ],
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.1.1a — "
        "NEGATIVE-SAMPLING FEASIBILITY AND TEMPORAL-COLLISION AUDIT"
    )

    print(
        "NO NEGATIVES WILL BE SAMPLED."
    )
    print(
        "NO RNG WILL BE CREATED."
    )
    print(
        "NO TRAINING WILL BE PERFORMED."
    )
    print(
        "NO NEGATIVE-SAMPLING POLICY WILL BE FROZEN."
    )

    # -------------------------------------------------------------------------
    # Frozen file existence
    # -------------------------------------------------------------------------

    for path in (
        TEMPORAL_PATH,
        NODE_INDEX_PATH,
    ):

        require(
            path.exists(),
            f"Missing frozen input: {path}",
        )

        print(
            f"FOUND  {path}"
        )

    # -------------------------------------------------------------------------
    # Load only audited schema fields
    # -------------------------------------------------------------------------

    temporal = pd.read_parquet(
        TEMPORAL_PATH,
        columns=[
            "interaction_id",
            "investor_id",
            "startup_id",
            "segment_number",
            "segment_label",
            "temporal_role",
            "experiment_split",
        ],
    )

    nodes = pd.read_parquet(
        NODE_INDEX_PATH,
        columns=[
            "node_index",
            "node_id",
            "node_type",
            "raw_entity_id",
        ],
    )

    temporal[
        "segment_number"
    ] = temporal[
        "segment_number"
    ].astype(
        np.int16
    )

    for column in (
        "interaction_id",
        "investor_id",
        "startup_id",
        "experiment_split",
    ):

        temporal[
            column
        ] = temporal[
            column
        ].astype(
            "string"
        )

    nodes[
        "node_type"
    ] = nodes[
        "node_type"
    ].astype(
        "string"
    )

    nodes[
        "raw_entity_id"
    ] = nodes[
        "raw_entity_id"
    ].astype(
        "string"
    )

    # =========================================================================
    # Frozen integrity audit
    # =========================================================================

    banner(
        "FROZEN INPUT INTEGRITY"
    )

    require(
        len(
            temporal
        )
        == EXPECTED[
            "temporal_rows"
        ],
        "Temporal row-count drift",
    )

    require(
        temporal[
            "interaction_id"
        ].notna().all(),
        "Null interaction_id",
    )

    require(
        not temporal[
            "interaction_id"
        ].duplicated().any(),
        "Duplicate interaction_id",
    )

    require(
        temporal[
            "investor_id"
        ].notna().all(),
        "Null investor_id",
    )

    require(
        temporal[
            "startup_id"
        ].notna().all(),
        "Null startup_id",
    )

    require(
        set(
            temporal[
                "segment_number"
            ].unique()
        )
        == set(
            range(
                61
            )
        ),
        "segment_number must be exactly 0..60",
    )

    checks = {
        "t0_rows": int(
            (
                temporal[
                    "segment_number"
                ]
                == 0
            ).sum()
        ),
        "t1_t59_rows": int(
            temporal[
                "segment_number"
            ].between(
                1,
                59,
            ).sum()
        ),
        "t0_t59_rows": int(
            (
                temporal[
                    "segment_number"
                ]
                <= 59
            ).sum()
        ),
        "t60_rows": int(
            (
                temporal[
                    "segment_number"
                ]
                == 60
            ).sum()
        ),
        "validation_rows": int(
            (
                (
                    temporal[
                        "segment_number"
                    ]
                    == 60
                )
                & (
                    temporal[
                        "experiment_split"
                    ]
                    == "validation"
                )
            ).sum()
        ),
        "test_rows": int(
            (
                (
                    temporal[
                        "segment_number"
                    ]
                    == 60
                )
                & (
                    temporal[
                        "experiment_split"
                    ]
                    == "test"
                )
            ).sum()
        ),
    }

    for name, actual in checks.items():

        require(
            actual
            == EXPECTED[
                name
            ],
            (
                f"{name}: expected "
                f"{EXPECTED[name]:,}, got {actual:,}"
            ),
        )

        print(
            f"{name:22s} "
            f"{actual:>10,}  PASS"
        )

    require(
        len(
            nodes
        )
        == EXPECTED[
            "node_rows"
        ],
        "Node row-count drift",
    )

    require(
        np.array_equal(
            nodes[
                "node_index"
            ].to_numpy(
                dtype=np.int64
            ),
            np.arange(
                EXPECTED[
                    "node_rows"
                ],
                dtype=np.int64,
            ),
        ),
        "node_index is not contiguous 0..477563",
    )

    node_counts = (
        nodes[
            "node_type"
        ].value_counts()
    )

    require(
        int(
            node_counts.get(
                "investor",
                0,
            )
        )
        == EXPECTED[
            "investor_nodes"
        ],
        "Investor node-count drift",
    )

    require(
        int(
            node_counts.get(
                "startup",
                0,
            )
        )
        == EXPECTED[
            "startup_nodes"
        ],
        "Startup node-count drift",
    )

    startup_node_ids = pd.Index(
        nodes.loc[
            nodes[
                "node_type"
            ]
            == "startup",
            "raw_entity_id",
        ].unique()
    )

    experiment_startup_ids = pd.Index(
        temporal[
            "startup_id"
        ].unique()
    )

    missing_from_nodes = (
        experiment_startup_ids.difference(
            startup_node_ids
        )
    )

    require(
        len(
            missing_from_nodes
        )
        == 0,
        (
            "Temporal experiment contains Startup IDs "
            "absent from frozen Startup role-node universe"
        ),
    )

    print(
        f"{'node_rows':22s} "
        f"{len(nodes):>10,}  PASS"
    )

    print(
        f"{'investor_nodes':22s} "
        f"{int(node_counts['investor']):>10,}  PASS"
    )

    print(
        f"{'startup_nodes':22s} "
        f"{int(node_counts['startup']):>10,}  PASS"
    )

    # =========================================================================
    # Startup / pair first-observation maps
    # =========================================================================

    banner(
        "TEMPORAL FIRST-OBSERVATION MAPS"
    )

    startup_first = (
        temporal.groupby(
            "startup_id",
            sort=False,
            observed=True,
        )[
            "segment_number"
        ]
        .min()
        .rename(
            "startup_first_segment"
        )
        .reset_index()
    )

    startup_first[
        "startup_first_segment"
    ] = startup_first[
        "startup_first_segment"
    ].astype(
        np.int16
    )

    experiment_startup_count = len(
        startup_first
    )

    never_observed_in_experiment = (
        EXPECTED[
            "startup_nodes"
        ]
        - experiment_startup_count
    )

    require(
        never_observed_in_experiment
        >= 0,
        (
            "Experiment Startup count exceeds "
            "frozen Startup role-node count"
        ),
    )

    first_counts = {
        int(
            segment
        ): int(
            count
        )
        for segment, count
        in startup_first[
            "startup_first_segment"
        ]
        .value_counts()
        .sort_index()
        .items()
    }

    pair_first = (
        temporal.groupby(
            [
                "investor_id",
                "startup_id",
            ],
            sort=False,
            observed=True,
        )[
            "segment_number"
        ]
        .min()
        .rename(
            "pair_first_segment"
        )
        .reset_index()
        .merge(
            startup_first,
            on="startup_id",
            how="left",
            validate="many_to_one",
        )
    )

    pair_first[
        "pair_first_segment"
    ] = pair_first[
        "pair_first_segment"
    ].astype(
        np.int16
    )

    targets = (
        temporal.loc[
            temporal[
                "segment_number"
            ].between(
                1,
                60,
            ),
            [
                "interaction_id",
                "investor_id",
                "startup_id",
                "segment_number",
                "experiment_split",
            ],
        ]
        .merge(
            pair_first[
                [
                    "investor_id",
                    "startup_id",
                    "pair_first_segment",
                    "startup_first_segment",
                ]
            ],
            on=[
                "investor_id",
                "startup_id",
            ],
            how="left",
            validate="many_to_one",
        )
    )

    require(
        targets[
            "pair_first_segment"
        ].notna().all(),
        "Missing Investor-Startup first-positive mapping",
    )

    require(
        targets[
            "startup_first_segment"
        ].notna().all(),
        "Missing Startup first-observation mapping",
    )

    require(
        (
            targets[
                "pair_first_segment"
            ]
            <= targets[
                "segment_number"
            ]
        ).all(),
        (
            "A target event appears before "
            "the pair's first-positive segment"
        ),
    )

    require(
        (
            targets[
                "startup_first_segment"
            ]
            <= targets[
                "segment_number"
            ]
        ).all(),
        (
            "A target event appears before "
            "the Startup's first-observation segment"
        ),
    )

    print(
        "Startups observed in T0..T60:      "
        f"{experiment_startup_count:,}"
    )

    print(
        "Startup role nodes not in T0..T60: "
        f"{never_observed_in_experiment:,}"
    )

    print(
        "Investor-Startup pairs T0..T60:    "
        f"{len(pair_first):,}"
    )

    print(
        "Target events T1..T60:             "
        f"{len(targets):,}"
    )

    # =========================================================================
    # Candidate-universe composition by target
    # =========================================================================

    banner(
        "CANDIDATE-UNIVERSE TEMPORAL COMPOSITION"
    )

    composition_rows = []

    startup_first_values = (
        startup_first[
            "startup_first_segment"
        ].to_numpy(
            dtype=np.int16
        )
    )

    for h in range(
        1,
        61,
    ):

        for universe in CANDIDATE_UNIVERSES:

            N = universe_size(
                first_counts,
                experiment_startup_count,
                h,
                universe,
            )

            eligible_mask = startup_eligible_mask(
                startup_first[
                    "startup_first_segment"
                ],
                h,
                universe,
            )

            eligible_first = (
                startup_first_values[
                    eligible_mask
                ]
            )

            history_count = int(
                (
                    eligible_first
                    < h
                ).sum()
            )

            target_count = int(
                (
                    eligible_first
                    == h
                ).sum()
            )

            future_count = int(
                (
                    eligible_first
                    > h
                ).sum()
            )

            unobserved_count = (
                never_observed_in_experiment
                if universe
                == "global_role_universe"
                else 0
            )

            require(
                (
                    history_count
                    + target_count
                    + future_count
                    + unobserved_count
                )
                == N,
                (
                    "Candidate universe composition mismatch "
                    f"for T{h}, {universe}"
                ),
            )

            composition_rows.append(
                {
                    "target_segment": h,
                    "target_label": f"T{h}",
                    "candidate_universe": universe,
                    "candidate_universe_size": N,
                    "startup_first_observed_before_target": (
                        history_count
                    ),
                    "startup_first_observed_in_target": (
                        target_count
                    ),
                    "startup_first_observed_after_target": (
                        future_count
                    ),
                    "startup_first_observed_in_T60_after_target": int(
                        (
                            (
                                eligible_first
                                == 60
                            )
                            & (
                                eligible_first
                                > h
                            )
                        ).sum()
                    ),
                    "startup_role_nodes_not_observed_in_T0_T60": (
                        unobserved_count
                    ),
                    "share_before_target": div0(
                        history_count,
                        N,
                    ),
                    "share_in_target": div0(
                        target_count,
                        N,
                    ),
                    "share_after_target": div0(
                        future_count,
                        N,
                    ),
                    "share_unobserved_T0_T60": div0(
                        unobserved_count,
                        N,
                    ),
                }
            )

    composition_df = pd.DataFrame(
        composition_rows
    )

    # =========================================================================
    # Exact collision / feasibility audit
    # =========================================================================

    banner(
        "EXACT NEGATIVE-POOL / COLLISION AUDIT — NO SAMPLING"
    )

    audit_rows = []

    for h in range(
        1,
        61,
    ):

        events_h = targets.loc[
            targets[
                "segment_number"
            ]
            == h
        ].copy()

        if h <= 59:

            split_names = (
                "train",
            )

            require(
                set(
                    events_h[
                        "experiment_split"
                    ].unique()
                )
                == {
                    "train"
                },
                (
                    f"T{h} contains a non-training "
                    "experiment_split value"
                ),
            )

        else:

            split_names = (
                "validation",
                "test",
            )

            require(
                set(
                    events_h[
                        "experiment_split"
                    ].unique()
                )
                == {
                    "validation",
                    "test",
                },
                (
                    "T60 experiment_split values "
                    "are not exactly validation/test"
                ),
            )

        active_investors = pd.Index(
            events_h[
                "investor_id"
            ].unique()
        )

        pairs_h = pair_first.loc[
            pair_first[
                "investor_id"
            ].isin(
                active_investors
            )
        ].copy()

        for universe in CANDIDATE_UNIVERSES:

            N = universe_size(
                first_counts,
                experiment_startup_count,
                h,
                universe,
            )

            # -------------------------------------------------------------
            # Investor-positive pair categories within negative eligibility
            # -------------------------------------------------------------

            pair_eligible = startup_eligible_mask(
                pairs_h[
                    "startup_first_segment"
                ],
                h,
                universe,
            )

            eligible_pairs = pairs_h.loc[
                pair_eligible,
                [
                    "investor_id",
                    "pair_first_segment",
                ],
            ].copy()

            eligible_pairs[
                "pair_class"
            ] = np.select(
                [
                    eligible_pairs[
                        "pair_first_segment"
                    ].to_numpy(
                        dtype=np.int16
                    )
                    < h,

                    eligible_pairs[
                        "pair_first_segment"
                    ].to_numpy(
                        dtype=np.int16
                    )
                    == h,
                ],
                [
                    "history",
                    "target_new",
                ],
                default="future",
            )

            pair_counts = (
                eligible_pairs.groupby(
                    [
                        "investor_id",
                        "pair_class",
                    ],
                    observed=True,
                )
                .size()
                .unstack(
                    fill_value=0
                )
            )

            for column in (
                "history",
                "target_new",
                "future",
            ):

                if column not in pair_counts.columns:

                    pair_counts[
                        column
                    ] = 0

            # -------------------------------------------------------------
            # Special diagnostic:
            # future pairs whose first positive is specifically T60
            # -------------------------------------------------------------

            if h < 60:

                t60_future = (
                    pairs_h.loc[
                        pair_eligible
                        & (
                            pairs_h[
                                "pair_first_segment"
                            ].to_numpy(
                                dtype=np.int16
                            )
                            == 60
                        )
                    ]
                    .groupby(
                        "investor_id",
                        observed=True,
                    )
                    .size()
                    .rename(
                        "t60_future"
                    )
                )

                pair_counts = pair_counts.join(
                    t60_future,
                    how="left",
                )

            else:

                pair_counts[
                    "t60_future"
                ] = 0

            pair_counts = (
                pair_counts.fillna(
                    0
                )[
                    [
                        "history",
                        "target_new",
                        "future",
                        "t60_future",
                    ]
                ]
                .astype(
                    np.int64
                )
            )

            # -------------------------------------------------------------
            # Focal-positive event classes
            # -------------------------------------------------------------

            focal_eligible = startup_eligible_mask(
                events_h[
                    "startup_first_segment"
                ],
                h,
                universe,
            )

            event_view = events_h[
                [
                    "investor_id",
                    "pair_first_segment",
                    "experiment_split",
                ]
            ].copy()

            event_view[
                "eligible"
            ] = focal_eligible

            event_view[
                "event_class"
            ] = np.where(
                ~event_view[
                    "eligible"
                ].to_numpy(),
                "ineligible",
                np.where(
                    event_view[
                        "pair_first_segment"
                    ].to_numpy(
                        dtype=np.int16
                    )
                    < h,
                    "eligible_history",
                    "eligible_target",
                ),
            )

            # -------------------------------------------------------------
            # Preserve frozen validation/test identities independently
            # -------------------------------------------------------------

            for split_name in split_names:

                split_events = event_view.loc[
                    event_view[
                        "experiment_split"
                    ]
                    == split_name
                ]

                event_counts = (
                    split_events.groupby(
                        [
                            "investor_id",
                            "event_class",
                        ],
                        observed=True,
                    )
                    .size()
                    .unstack(
                        fill_value=0
                    )
                )

                for column in (
                    "eligible_history",
                    "eligible_target",
                    "ineligible",
                ):

                    if column not in event_counts.columns:

                        event_counts[
                            column
                        ] = 0

                event_counts = (
                    event_counts[
                        [
                            "eligible_history",
                            "eligible_target",
                            "ineligible",
                        ]
                    ]
                    .astype(
                        np.int64
                    )
                )

                stats = (
                    event_counts.join(
                        pair_counts,
                        how="left",
                    )
                    .fillna(
                        0
                    )
                    .astype(
                        np.int64
                    )
                )

                require(
                    (
                        (
                            stats[
                                "eligible_history"
                            ]
                            == 0
                        )
                        | (
                            stats[
                                "history"
                            ]
                            >= 1
                        )
                    ).all(),
                    (
                        "Eligible historical focal event "
                        "has no eligible historical pair"
                    ),
                )

                require(
                    (
                        (
                            stats[
                                "eligible_target"
                            ]
                            == 0
                        )
                        | (
                            stats[
                                "target_new"
                            ]
                            >= 1
                        )
                    ).all(),
                    (
                        "Eligible target-period focal event "
                        "has no eligible target-new pair"
                    ),
                )

                # ---------------------------------------------------------
                # Evaluate every candidate/exclusion combination exactly.
                # ---------------------------------------------------------

                for policy in EXCLUSION_POLICIES:

                    summary = summarize_policy(
                        stats,
                        N,
                        policy,
                    )

                    audit_rows.append(
                        {
                            "target_segment": h,
                            "target_label": f"T{h}",
                            "experiment_split": split_name,
                            "candidate_universe": universe,
                            "exclusion_policy": policy,
                            "candidate_universe_size": N,
                            **summary,
                        }
                    )

        print(
            f"T{h:02d}: "
            f"{len(events_h):>7,} positives | "
            f"{len(active_investors):>6,} active investors | "
            "5 universes × 4 exclusions audited"
        )

    segment_df = pd.DataFrame(
        audit_rows
    )

    # =========================================================================
    # Overall training / evaluation summaries
    # =========================================================================

    banner(
        "TRAINING / EVALUATION SCOPE AGGREGATION"
    )

    scope_input = (
        segment_df.copy()
    )

    scope_input[
        "scope"
    ] = np.where(
        scope_input[
            "target_segment"
        ]
        <= 59,
        "training_T1_T59",
        "evaluation_T60",
    )

    scope_df = combine_weighted(
        scope_input,
        [
            "scope",
            "candidate_universe",
            "exclusion_policy",
        ],
    )

    # =========================================================================
    # Quantify exactly what target/future-aware filtering removes
    # =========================================================================

    banner(
        "TARGET-LABEL / FUTURE-LABEL EXCLUSION DELTAS"
    )

    delta_rows = []

    key_columns = [
        "target_segment",
        "target_label",
        "experiment_split",
        "candidate_universe",
    ]

    for keys, group in segment_df.groupby(
        key_columns,
        sort=True,
        dropna=False,
    ):

        by_policy = {
            row[
                "exclusion_policy"
            ]: row
            for _,
            row
            in group.iterrows()
        }

        prior = by_policy[
            "exclude_prior_positive_pairs_plus_current"
        ]

        through_target = by_policy[
            "exclude_prior_and_target_period_positive_pairs"
        ]

        all_experiment = by_policy[
            "exclude_all_experiment_positive_pairs"
        ]

        delta_rows.append(
            {
                **dict(
                    zip(
                        key_columns,
                        keys,
                    )
                ),
                "positive_events": int(
                    prior[
                        "positive_events"
                    ]
                ),
                "mean_pool_reduction_from_excluding_other_target_positives": (
                    float(
                        prior[
                            "negative_pool_mean"
                        ]
                    )
                    - float(
                        through_target[
                            "negative_pool_mean"
                        ]
                    )
                ),
                "p_target_positive_collision_before": float(
                    prior[
                        "p_uniform_draw_hits_other_target_positive"
                    ]
                ),
                "p_target_positive_collision_after": float(
                    through_target[
                        "p_uniform_draw_hits_other_target_positive"
                    ]
                ),
                "mean_pool_reduction_from_excluding_future_positives": (
                    float(
                        through_target[
                            "negative_pool_mean"
                        ]
                    )
                    - float(
                        all_experiment[
                            "negative_pool_mean"
                        ]
                    )
                ),
                "p_future_positive_collision_before": float(
                    through_target[
                        "p_uniform_draw_hits_future_positive"
                    ]
                ),
                "p_future_positive_collision_after": float(
                    all_experiment[
                        "p_uniform_draw_hits_future_positive"
                    ]
                ),
                "target_exclusion_interpretation": (
                    "T1-T59: training-label-aware within target period; "
                    "T60: held-out target-label usage beyond focal positive."
                ),
                "future_exclusion_interpretation": (
                    "T1-T59: future-label leakage if used operationally; "
                    "T60: no later experiment period exists."
                ),
            }
        )

    delta_df = pd.DataFrame(
        delta_rows
    )

    risk_df = (
        build_policy_risk_register()
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL AUDIT INVARIANTS"
    )

    # Candidate/exclusion alternatives duplicate the same positive populations.
    # Deduplicate first before checking frozen positive counts.
    base_counts = (
        segment_df[
            [
                "target_segment",
                "experiment_split",
                "positive_events",
            ]
        ]
        .drop_duplicates()
    )

    training_events = int(
        base_counts.loc[
            base_counts[
                "experiment_split"
            ]
            == "train",
            "positive_events",
        ].sum()
    )

    validation_events = int(
        base_counts.loc[
            base_counts[
                "experiment_split"
            ]
            == "validation",
            "positive_events",
        ].sum()
    )

    test_events = int(
        base_counts.loc[
            base_counts[
                "experiment_split"
            ]
            == "test",
            "positive_events",
        ].sum()
    )

    require(
        training_events
        == EXPECTED[
            "t1_t59_rows"
        ],
        "Training target-event count changed",
    )

    require(
        validation_events
        == EXPECTED[
            "validation_rows"
        ],
        "Validation target-event count changed",
    )

    require(
        test_events
        == EXPECTED[
            "test_rows"
        ],
        "Test target-event count changed",
    )

    # Policy identities should reproduce their intended algebra exactly.

    prior_rows = segment_df.loc[
        segment_df[
            "exclusion_policy"
        ]
        == "exclude_prior_positive_pairs_plus_current"
    ]

    through_rows = segment_df.loc[
        segment_df[
            "exclusion_policy"
        ]
        == "exclude_prior_and_target_period_positive_pairs"
    ]

    all_rows = segment_df.loc[
        segment_df[
            "exclusion_policy"
        ]
        == "exclude_all_experiment_positive_pairs"
    ]

    require(
        np.allclose(
            prior_rows[
                "mean_historical_positive_pairs_remaining"
            ],
            0.0,
        ),
        (
            "Prior-positive exclusion unexpectedly "
            "left historical collisions"
        ),
    )

    require(
        np.allclose(
            through_rows[
                "mean_historical_positive_pairs_remaining"
            ],
            0.0,
        ),
        (
            "Through-target exclusion unexpectedly "
            "left historical collisions"
        ),
    )

    require(
        np.allclose(
            through_rows[
                "mean_other_target_positive_pairs_remaining"
            ],
            0.0,
        ),
        (
            "Through-target exclusion unexpectedly "
            "left target-period collisions"
        ),
    )

    require(
        np.allclose(
            all_rows[
                "mean_any_eventual_positive_pairs_remaining"
            ],
            0.0,
        ),
        (
            "All-experiment exclusion unexpectedly "
            "left observed-positive collisions"
        ),
    )

    print(
        "Frozen training target count unchanged:      PASS"
    )
    print(
        "Frozen validation target count unchanged:    PASS"
    )
    print(
        "Frozen test target count unchanged:          PASS"
    )
    print(
        "Frozen Startup role universe unchanged:      PASS"
    )
    print(
        "Historical-collision algebra:                PASS"
    )
    print(
        "Target-collision algebra:                    PASS"
    )
    print(
        "Future-collision algebra:                    PASS"
    )
    print(
        "Negative rows generated:                     0"
    )
    print(
        "RNG used:                                    NO"
    )
    print(
        "Training performed:                          NO"
    )
    print(
        "Sampling policy frozen:                      NO"
    )

    # =========================================================================
    # Write audit-only outputs
    # =========================================================================

    banner(
        "WRITE AUDIT-ONLY OUTPUTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    composition_df.to_csv(
        CANDIDATE_COMPOSITION_PATH,
        index=False,
    )

    risk_df.to_csv(
        POLICY_RISK_PATH,
        index=False,
    )

    segment_df.to_csv(
        SEGMENT_AUDIT_PATH,
        index=False,
    )

    scope_df.to_csv(
        SCOPE_AUDIT_PATH,
        index=False,
    )

    delta_df.to_csv(
        DELTA_PATH,
        index=False,
    )

    manifest = {
        "phase": "5.1.1a",
        "title": (
            "Negative-Sampling Feasibility "
            "and Temporal-Collision Audit"
        ),
        "status": (
            "AUDIT_COMPLETE_NOT_FROZEN"
        ),
        "negative_samples_generated": False,
        "rng_used": False,
        "training_performed": False,
        "optimizer_created": False,
        "frozen_inputs": {
            "temporal": str(
                TEMPORAL_PATH
            ),
            "node_index": str(
                NODE_INDEX_PATH
            ),
        },
        "inherited_indexing": {
            "T0": (
                "compressed_history_only"
            ),
            "training_targets": (
                "T1..T59"
            ),
            "evaluation_target": (
                "T60"
            ),
            "target_history": (
                "T_h uses T0..T(h-1)"
            ),
        },
        "paper_specified": {
            "training_negative_method": (
                "random negatives"
            ),
            "evaluation_candidates": (
                "1 positive + 99 random negatives"
            ),
            "evaluation_metrics": [
                "HR@10",
                "NDCG@10",
            ],
        },
        "still_unfrozen_phase_5_decisions": [
            "training negative:positive ratio",
            "training negative candidate eligibility",
            "training historical negative exclusion",
            "training epoch count",
            "early stopping",
            "weight decay",
            (
                "evaluation candidate-generation "
                "runtime contract"
            ),
        ],
        "candidate_universes_audited": list(
            CANDIDATE_UNIVERSES
        ),
        "exclusion_policies_audited": list(
            EXCLUSION_POLICIES
        ),
        "reference_negative_counts_capacity_only": list(
            REFERENCE_NEGATIVE_COUNTS
        ),
        "semantic_guard": (
            "A future-positive pair is not automatically "
            "an invalid negative at T_h. It is a "
            "temporal-collision diagnostic. Excluding it "
            "from historical training would require future "
            "label knowledge."
        ),
        "outputs": [
            str(
                CANDIDATE_COMPOSITION_PATH
            ),
            str(
                POLICY_RISK_PATH
            ),
            str(
                SEGMENT_AUDIT_PATH
            ),
            str(
                SCOPE_AUDIT_PATH
            ),
            str(
                DELTA_PATH
            ),
        ],
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    for path in (
        CANDIDATE_COMPOSITION_PATH,
        POLICY_RISK_PATH,
        SEGMENT_AUDIT_PATH,
        SCOPE_AUDIT_PATH,
        DELTA_PATH,
        MANIFEST_PATH,
    ):

        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Decision-facing terminal output
    # =========================================================================

    banner(
        "DECISION-FACING SUMMARY — NOTHING FROZEN"
    )

    print(
        "\nT60 candidate-universe composition:"
    )

    print(
        composition_df.loc[
            composition_df[
                "target_segment"
            ]
            == 60,
            [
                "candidate_universe",
                "candidate_universe_size",
                "startup_first_observed_before_target",
                "startup_first_observed_in_target",
                "startup_role_nodes_not_observed_in_T0_T60",
            ],
        ].to_string(
            index=False
        )
    )

    summary_columns = [
        "candidate_universe",
        "exclusion_policy",
        "positive_events",
        "focal_outside_negative_universe_share",
        "negative_pool_min",
        "negative_pool_mean",
        "p_uniform_draw_hits_historical_positive",
        "p_uniform_draw_hits_other_target_positive",
        "p_uniform_draw_hits_future_positive",
        "share_events_with_pool_lt_99",
    ]

    print(
        "\nTraining T1-T59:"
    )

    print(
        scope_df.loc[
            scope_df[
                "scope"
            ]
            == "training_T1_T59",
            summary_columns,
        ].to_string(
            index=False
        )
    )

    print(
        "\nEvaluation T60:"
    )

    print(
        scope_df.loc[
            scope_df[
                "scope"
            ]
            == "evaluation_T60",
            summary_columns,
        ].to_string(
            index=False
        )
    )

    print(
        "\nInterpretation guards:"
    )

    print(
        "1. A future-positive at T_h is a diagnostic "
        "class, not automatically a false negative at T_h."
    )

    print(
        "2. Excluding future positives during T1-T59 "
        "training uses future labels -> temporal leakage."
    )

    print(
        "3. Excluding all other T60 positives for each "
        "focal evaluation event uses held-out T60 labels."
    )

    print(
        "4. prefix_prior is causal only for observed "
        "investment visibility, not company existence."
    )

    print(
        "5. Reference counts 1/4/10/20/50/99 are "
        "feasibility probes only, not a ratio decision."
    )

    banner(
        "PHASE 5.1.1a COMPLETE"
    )

    print(
        "AUDIT COMPLETE — NO NEGATIVES GENERATED — "
        "NO NEGATIVE-SAMPLING DECISION FROZEN"
    )


if __name__ == "__main__":
    main()