"""
Phase 5.1.1b — Freeze Training Negative-Sampling Semantics

Purpose
-------
Freeze only:

1. training negative candidate eligibility;
2. training historical/current-target positive exclusion.

This script DOES NOT:
- generate negative samples;
- create an RNG;
- choose the training negative:positive ratio;
- choose a sampling seed;
- choose with/without-replacement semantics;
- freeze evaluation candidate generation;
- instantiate a model;
- create an optimizer;
- train.

Frozen rule
-----------
For training target T_h, h = 1..59:

    candidate startup b is eligible for investor o iff:

        1. b belongs to the frozen Phase-3 Startup role universe; and

        2. there is no observed positive Investor-Startup pair (o,b)
           with first-positive segment <= h.

Future-positive pairs with first-positive segment > h remain eligible.

This is deliberately different from excluding all eventual positives,
because the latter would consume future labels during historical training.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

PHASE2_TEMPORAL = Path(
    "data/experimental/phase_2/model_ready/"
    "interactions_itrs_temporal_split.parquet"
)

PHASE3_NODE_INDEX = Path(
    "data/experimental/phase_3/model_ready/"
    "node_index.parquet"
)

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/phase_5_1_1a"
)

AUDIT_SCOPE = (
    AUDIT_DIR
    / "negative_pool_feasibility_and_collision_overall_by_scope.csv"
)

AUDIT_COMPOSITION = (
    AUDIT_DIR
    / "candidate_universe_temporal_composition_by_segment.csv"
)

AUDIT_MANIFEST = (
    AUDIT_DIR
    / "phase_5_1_1a_audit_manifest.json"
)

OUT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

CONTRACT_PATH = (
    OUT_DIR
    / "phase_5_1_1b_training_negative_semantics_contract.json"
)

DECISION_REGISTER_PATH = (
    OUT_DIR
    / "phase_5_1_1b_negative_semantics_decision_register.csv"
)

FREEZE_AUDIT_PATH = (
    OUT_DIR
    / "phase_5_1_1b_negative_semantics_freeze_audit.csv"
)


# =============================================================================
# Frozen expected values
# =============================================================================

EXPECTED = {
    "temporal_rows": 1_195_937,
    "training_target_rows_T1_T59": 1_073_249,
    "t60_rows": 22_515,
    "node_rows": 477_564,
    "investor_nodes": 165_975,
    "startup_nodes": 311_589,
}

SELECTED_UNIVERSE = "global_role_universe"

SELECTED_TRAINING_EXCLUSION = (
    "exclude_prior_and_target_period_positive_pairs"
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.1.1b — "
        "FREEZE TRAINING NEGATIVE-SAMPLING SEMANTICS"
    )

    print("Negative samples generated:       NO")
    print("RNG created:                      NO")
    print("Training ratio selected:          NO")
    print("Evaluation runtime frozen:        NO")
    print("Model instantiated:               NO")
    print("Optimizer created:                NO")
    print("Training performed:               NO")

    # =========================================================================
    # Existence
    # =========================================================================

    banner("AUTHORITATIVE INPUT EXISTENCE")

    required_paths = (
        PHASE2_TEMPORAL,
        PHASE3_NODE_INDEX,
        AUDIT_SCOPE,
        AUDIT_COMPOSITION,
        AUDIT_MANIFEST,
    )

    for path in required_paths:
        require(
            path.exists(),
            f"Missing required input: {path}",
        )

        print(f"FOUND  {path}")

    # =========================================================================
    # Load frozen Phase-2 / Phase-3 identities
    # =========================================================================

    banner("FROZEN PHASE-2 / PHASE-3 IDENTITY RECHECK")

    temporal = pd.read_parquet(
        PHASE2_TEMPORAL,
        columns=[
            "interaction_id",
            "investor_id",
            "startup_id",
            "segment_number",
            "experiment_split",
        ],
    )

    nodes = pd.read_parquet(
        PHASE3_NODE_INDEX,
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

    require(
        len(temporal) == EXPECTED["temporal_rows"],
        "Frozen temporal row count drift",
    )

    require(
        not temporal["interaction_id"].duplicated().any(),
        "Duplicate interaction_id detected",
    )

    training = temporal.loc[
        temporal["segment_number"].between(1, 59)
    ].copy()

    t60 = temporal.loc[
        temporal["segment_number"] == 60
    ].copy()

    require(
        len(training)
        == EXPECTED["training_target_rows_T1_T59"],
        "Frozen T1-T59 training population drift",
    )

    require(
        len(t60) == EXPECTED["t60_rows"],
        "Frozen T60 population drift",
    )

    require(
        len(nodes) == EXPECTED["node_rows"],
        "Frozen node-index row count drift",
    )

    node_counts = nodes["node_type"].value_counts()

    require(
        int(node_counts.get("investor", 0))
        == EXPECTED["investor_nodes"],
        "Frozen Investor node count drift",
    )

    require(
        int(node_counts.get("startup", 0))
        == EXPECTED["startup_nodes"],
        "Frozen Startup node count drift",
    )

    startup_nodes = nodes.loc[
        nodes["node_type"] == "startup",
        [
            "node_index",
            "raw_entity_id",
        ],
    ].copy()

    require(
        startup_nodes["raw_entity_id"].notna().all(),
        "Null Startup raw_entity_id",
    )

    require(
        not startup_nodes["raw_entity_id"].duplicated().any(),
        "Duplicate Startup raw_entity_id",
    )

    print(
        f"Temporal rows:       {len(temporal):>10,}  PASS"
    )
    print(
        f"T1-T59 targets:      {len(training):>10,}  PASS"
    )
    print(
        f"T60 events:          {len(t60):>10,}  PASS"
    )
    print(
        f"Role nodes:          {len(nodes):>10,}  PASS"
    )
    print(
        f"Startup candidates:  {len(startup_nodes):>10,}  PASS"
    )

    # =========================================================================
    # Reconstruct first-positive pair timing
    # =========================================================================

    banner("FIRST-POSITIVE TEMPORAL SEMANTICS")

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

    pair_first["first_positive_segment"] = (
        pair_first["first_positive_segment"]
        .astype(np.int16)
    )

    require(
        pair_first["first_positive_segment"]
        .between(0, 60)
        .all(),
        "Invalid first-positive segment",
    )

    print(
        "Unique Investor-Startup positive pairs: "
        f"{len(pair_first):,}"
    )

    # =========================================================================
    # Load Phase-5.1.1a evidence
    # =========================================================================

    banner("PHASE 5.1.1a AUDIT EVIDENCE")

    scope = pd.read_csv(AUDIT_SCOPE)
    composition = pd.read_csv(AUDIT_COMPOSITION)

    manifest = json.loads(
        AUDIT_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    require(
        manifest.get("negative_samples_generated") is False,
        "5.1.1a unexpectedly generated negatives",
    )

    require(
        manifest.get("rng_used") is False,
        "5.1.1a unexpectedly used RNG",
    )

    require(
        manifest.get("training_performed") is False,
        "5.1.1a unexpectedly trained",
    )

    selected_training_rows = scope.loc[
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
            == SELECTED_TRAINING_EXCLUSION
        )
    ].copy()

    require(
        len(selected_training_rows) == 1,
        (
            "Expected exactly one selected "
            "training audit row"
        ),
    )

    selected = selected_training_rows.iloc[0]

    require(
        int(selected["positive_events"])
        == EXPECTED["training_target_rows_T1_T59"],
        "Selected audit row has wrong training count",
    )

    require(
        float(
            selected[
                "focal_outside_negative_universe_share"
            ]
        )
        == 0.0,
        (
            "Selected universe excludes "
            "some training focal positives"
        ),
    )

    require(
        int(selected["negative_pool_min"]) >= 99,
        (
            "Selected policy has fewer than "
            "99 candidates for at least one event"
        ),
    )

    require(
        float(
            selected[
                "p_uniform_draw_hits_historical_positive"
            ]
        )
        == 0.0,
        (
            "Selected policy leaves "
            "historical-positive collisions"
        ),
    )

    require(
        float(
            selected[
                "p_uniform_draw_hits_other_target_positive"
            ]
        )
        == 0.0,
        (
            "Selected policy leaves "
            "same-target-positive collisions"
        ),
    )

    require(
        float(
            selected[
                "share_events_with_pool_lt_99"
            ]
        )
        == 0.0,
        "Selected policy is not 99-candidate feasible",
    )

    future_collision_probability = float(
        selected[
            "p_uniform_draw_hits_future_positive"
        ]
    )

    require(
        future_collision_probability > 0.0,
        (
            "Expected future-positive diagnostic "
            "to remain non-zero; otherwise future "
            "labels may have been excluded"
        ),
    )

    print(
        "Selected candidate universe: "
        f"{SELECTED_UNIVERSE}"
    )

    print(
        "Selected exclusion policy:   "
        f"{SELECTED_TRAINING_EXCLUSION}"
    )

    print(
        "Training positives:          "
        f"{int(selected['positive_events']):,}"
    )

    print(
        "Minimum negative pool:       "
        f"{int(selected['negative_pool_min']):,}"
    )

    print(
        "Mean negative pool:          "
        f"{float(selected['negative_pool_mean']):,.6f}"
    )

    print(
        "Historical collision p:      "
        f"{float(selected['p_uniform_draw_hits_historical_positive']):.9f}"
    )

    print(
        "Same-target collision p:     "
        f"{float(selected['p_uniform_draw_hits_other_target_positive']):.9f}"
    )

    print(
        "Future-positive diagnostic p:"
        f" {future_collision_probability:.9f}"
    )

    # =========================================================================
    # Explicit rejection evidence
    # =========================================================================

    banner("REJECTED CANDIDATE-UNIVERSE OPTIONS")

    training_prefix = scope.loc[
        (
            scope["scope"]
            == "training_T1_T59"
        )
        & (
            scope["candidate_universe"]
            == "prefix_prior"
        )
        & (
            scope["exclusion_policy"]
            == "exclude_prior_positive_pairs_plus_current"
        )
    ]

    require(
        len(training_prefix) == 1,
        "Missing prefix_prior training audit row",
    )

    training_prefix_outside = float(
        training_prefix.iloc[0][
            "focal_outside_negative_universe_share"
        ]
    )

    t60_prefix = scope.loc[
        (
            scope["scope"]
            == "evaluation_T60"
        )
        & (
            scope["candidate_universe"]
            == "prefix_prior"
        )
        & (
            scope["exclusion_policy"]
            == "exclude_prior_positive_pairs_plus_current"
        )
    ]

    require(
        len(t60_prefix) == 1,
        "Missing prefix_prior T60 audit row",
    )

    t60_prefix_outside = float(
        t60_prefix.iloc[0][
            "focal_outside_negative_universe_share"
        ]
    )

    print(
        "prefix_prior focal-positive exclusion:"
    )
    print(
        f"  T1-T59 training: {training_prefix_outside:.6%}"
    )
    print(
        f"  T60 evaluation:  {t60_prefix_outside:.6%}"
    )

    require(
        training_prefix_outside > 0.40,
        (
            "Expected prefix_prior to exclude "
            "a substantial training-positive population"
        ),
    )

    require(
        t60_prefix_outside > 0.40,
        (
            "Expected prefix_prior to exclude "
            "a substantial T60-positive population"
        ),
    )

    # =========================================================================
    # Semantic contract
    # =========================================================================

    banner("FROZEN TRAINING NEGATIVE SEMANTICS")

    semantic_expression = (
        "eligible_negative(o,b,h) := "
        "startup_role_node(b) AND "
        "NOT EXISTS positive_event(o,b,s) "
        "WITH segment_number(s) <= h; "
        "h in {1,...,59}"
    )

    print(semantic_expression)

    print()
    print(
        "Future-positive pairs with first-positive segment > h:"
        " ELIGIBLE"
    )

    print(
        "Pairs positive at any segment <= h:"
        " INELIGIBLE"
    )

    print(
        "Structural connectivity filter:"
        " NONE"
    )

    print(
        "Description-availability filter:"
        " NONE"
    )

    print(
        "Startup first-investment eligibility filter:"
        " NONE"
    )

    print(
        "T60 labels allowed in T1-T59 eligibility:"
        " NO"
    )

    # =========================================================================
    # Decision register
    # =========================================================================

    decisions = pd.DataFrame(
        [
            {
                "decision_id": (
                    "training_negative_method"
                ),
                "value": (
                    "random_negative_sampling"
                ),
                "classification": (
                    "PAPER_SPECIFIED"
                ),
                "status": (
                    "INHERITED"
                ),
                "rationale": (
                    "ITRS explicitly states that "
                    "negative instances are sampled randomly."
                ),
            },
            {
                "decision_id": (
                    "training_negative_candidate_universe"
                ),
                "value": (
                    "all_311589_frozen_startup_role_nodes"
                ),
                "classification": (
                    "DATASET_ADAPTATION+"
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_1_1b"
                ),
                "rationale": (
                    "Preserves the frozen Phase-3 role universe, "
                    "retains interaction-cold positives, and avoids "
                    "defining historical eligibility from future "
                    "interaction observations."
                ),
            },
            {
                "decision_id": (
                    "training_positive_pair_exclusion"
                ),
                "value": (
                    "exclude_all_pairs_positive_through_target_segment"
                ),
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_1_1b"
                ),
                "rationale": (
                    "Prevents already-observed and same-target "
                    "positive pairs from receiving contradictory "
                    "negative labels."
                ),
            },
            {
                "decision_id": (
                    "training_future_positive_exclusion"
                ),
                "value": (
                    "do_not_exclude_future_positive_pairs"
                ),
                "classification": (
                    "TEMPORAL_LEAKAGE_GUARD"
                ),
                "status": (
                    "FROZEN_PHASE_5_1_1b"
                ),
                "rationale": (
                    "Future investment outcomes must not affect "
                    "historical negative eligibility."
                ),
            },
            {
                "decision_id": (
                    "startup_structural_coverage_filter"
                ),
                "value": "none",
                "classification": (
                    "INHERITED_PHASE_3_CONSTRAINT"
                ),
                "status": "FROZEN",
                "rationale": (
                    "Structural isolates remain in the canonical "
                    "recommendation universe."
                ),
            },
            {
                "decision_id": (
                    "startup_description_availability_filter"
                ),
                "value": "none",
                "classification": (
                    "INHERITED_PHASE_4_CONSTRAINT"
                ),
                "status": "FROZEN",
                "rationale": (
                    "Missing description/category inputs were already "
                    "represented explicitly rather than filtered."
                ),
            },
            {
                "decision_id": (
                    "training_negative_positive_ratio"
                ),
                "value": None,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": "DEFERRED",
                "rationale": (
                    "Phase 5.1.1a shows candidate capacity does not "
                    "constrain the choice; ratio requires its own "
                    "audit/selection."
                ),
            },
            {
                "decision_id": (
                    "training_random_distribution"
                ),
                "value": None,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": "DEFERRED",
                "rationale": (
                    "Paper says randomly sampled but does not state "
                    "uniformity or another probability distribution."
                ),
            },
            {
                "decision_id": (
                    "training_sampling_replacement_semantics"
                ),
                "value": None,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": "DEFERRED",
                "rationale": (
                    "Depends on the eventual negative count and "
                    "runtime sampling contract."
                ),
            },
            {
                "decision_id": (
                    "evaluation_candidate_generation_runtime_contract"
                ),
                "value": None,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": "DEFERRED",
                "rationale": (
                    "ITRS fixes 99 negatives but not detailed "
                    "eligibility, seed, persistence, or replacement "
                    "semantics."
                ),
            },
        ]
    )

    # =========================================================================
    # Freeze audit
    # =========================================================================

    checks = [
        (
            "phase2_temporal_row_count",
            len(temporal)
            == EXPECTED["temporal_rows"],
        ),
        (
            "training_target_count",
            len(training)
            == EXPECTED["training_target_rows_T1_T59"],
        ),
        (
            "t60_population_unchanged",
            len(t60)
            == EXPECTED["t60_rows"],
        ),
        (
            "phase3_node_count",
            len(nodes)
            == EXPECTED["node_rows"],
        ),
        (
            "startup_universe_count",
            len(startup_nodes)
            == EXPECTED["startup_nodes"],
        ),
        (
            "selected_universe_has_zero_focal_exclusion",
            float(
                selected[
                    "focal_outside_negative_universe_share"
                ]
            )
            == 0.0,
        ),
        (
            "selected_policy_removes_historical_collisions",
            float(
                selected[
                    "p_uniform_draw_hits_historical_positive"
                ]
            )
            == 0.0,
        ),
        (
            "selected_policy_removes_same_target_collisions",
            float(
                selected[
                    "p_uniform_draw_hits_other_target_positive"
                ]
            )
            == 0.0,
        ),
        (
            "selected_policy_preserves_future_positive_eligibility",
            future_collision_probability > 0.0,
        ),
        (
            "negative_pool_supports_99_reference_candidates",
            int(
                selected[
                    "negative_pool_min"
                ]
            )
            >= 99,
        ),
        (
            "no_negative_sampling_performed",
            True,
        ),
        (
            "no_training_performed",
            True,
        ),
    ]

    freeze_audit = pd.DataFrame(
        [
            {
                "check": name,
                "result": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
            }
            for name, passed in checks
        ]
    )

    require(
        (
            freeze_audit["result"]
            == "PASS"
        ).all(),
        "One or more freeze checks failed",
    )

    # =========================================================================
    # Contract
    # =========================================================================

    contract = {
        "phase": "5.1.1b",
        "title": (
            "Training Negative-Sampling Semantics Selection and Freeze"
        ),
        "status": "FROZEN",
        "negative_samples_generated": False,
        "rng_created": False,
        "training_performed": False,
        "optimizer_created": False,

        "paper_specified": {
            "training_negative_sampling": "random",
            "evaluation_candidates": (
                "1 positive + 99 randomly sampled negatives"
            ),
            "evaluation_metrics": [
                "HR@10",
                "NDCG@10",
            ],
        },

        "training_target_periods": {
            "first": 1,
            "last": 59,
            "T0_role": "compressed_history_only",
            "T60_role": "held_out_evaluation",
        },

        "training_negative_candidate_universe": {
            "source": str(PHASE3_NODE_INDEX),
            "node_type": "startup",
            "count": EXPECTED["startup_nodes"],
            "membership_rule": (
                "all frozen Phase-3 Startup role nodes"
            ),
            "classification": (
                "DATASET_ADAPTATION+"
                "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
            ),
            "historical_availability_interpretation": (
                "static/transductive role universe; "
                "not historically versioned company availability"
            ),
        },

        "training_negative_exclusion": {
            "rule": (
                "For target T_h, exclude every Investor-Startup pair "
                "whose first observed positive segment is <= h."
            ),
            "mathematical_contract": semantic_expression,
            "exclude_prior_positive_pairs": True,
            "exclude_other_positive_pairs_in_target_segment": True,
            "exclude_future_positive_pairs": False,
            "use_future_labels": False,
            "use_T60_labels_for_T1_T59_sampling": False,
            "classification": (
                "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
            ),
        },

        "filters_not_allowed": {
            "structural_connectivity": False,
            "description_nonmissing": False,
            "category_nonmissing": False,
            "prior_startup_interaction": False,
            "minimum_investor_history": False,
            "future_positive_status": False,
        },

        "audit_evidence": {
            "training_positive_events": int(
                selected["positive_events"]
            ),
            "minimum_negative_pool": int(
                selected["negative_pool_min"]
            ),
            "mean_negative_pool": float(
                selected["negative_pool_mean"]
            ),
            "focal_outside_universe_share": float(
                selected[
                    "focal_outside_negative_universe_share"
                ]
            ),
            "historical_positive_collision_probability": float(
                selected[
                    "p_uniform_draw_hits_historical_positive"
                ]
            ),
            "same_target_positive_collision_probability": float(
                selected[
                    "p_uniform_draw_hits_other_target_positive"
                ]
            ),
            "future_positive_diagnostic_probability": (
                future_collision_probability
            ),
            "events_with_pool_lt_99_share": float(
                selected[
                    "share_events_with_pool_lt_99"
                ]
            ),
            "prefix_prior_training_focal_exclusion_share": (
                training_prefix_outside
            ),
            "prefix_prior_t60_focal_exclusion_share": (
                t60_prefix_outside
            ),
        },

        "still_unresolved": [
            "training negative:positive ratio",
            "training random sampling distribution",
            "training sampling with/without replacement",
            "training negative RNG/seed runtime contract",
            "training epoch count",
            "early stopping",
            "weight decay",
            "evaluation candidate-generation runtime contract",
        ],

        "authoritative_input_hashes": {
            str(PHASE2_TEMPORAL): sha256_file(
                PHASE2_TEMPORAL
            ),
            str(PHASE3_NODE_INDEX): sha256_file(
                PHASE3_NODE_INDEX
            ),
            str(AUDIT_SCOPE): sha256_file(
                AUDIT_SCOPE
            ),
            str(AUDIT_COMPOSITION): sha256_file(
                AUDIT_COMPOSITION
            ),
            str(AUDIT_MANIFEST): sha256_file(
                AUDIT_MANIFEST
            ),
        },
    }

    # =========================================================================
    # Write
    # =========================================================================

    banner("WRITE FROZEN PHASE-5.1.1b CONTRACT")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    decisions.to_csv(
        DECISION_REGISTER_PATH,
        index=False,
    )

    freeze_audit.to_csv(
        FREEZE_AUDIT_PATH,
        index=False,
    )

    write_json(
        CONTRACT_PATH,
        contract,
    )

    for path in (
        CONTRACT_PATH,
        DECISION_REGISTER_PATH,
        FREEZE_AUDIT_PATH,
    ):
        print(f"WROTE  {path}")

    # =========================================================================
    # Final status
    # =========================================================================

    banner("PHASE 5.1.1b FINAL STATUS")

    print(
        "Training negative candidate universe: "
        "FROZEN"
    )
    print(
        "  -> all 311,589 frozen Startup role nodes"
    )

    print(
        "Training positive-pair exclusion:      "
        "FROZEN"
    )
    print(
        "  -> exclude first_positive_segment <= target T_h"
    )

    print(
        "Training future-positive exclusion:    "
        "FROZEN"
    )
    print(
        "  -> NO"
    )

    print(
        "Training negative:positive ratio:      "
        "DEFERRED"
    )

    print(
        "Training random distribution:          "
        "DEFERRED"
    )

    print(
        "Evaluation runtime contract:           "
        "DEFERRED"
    )

    print(
        "Negative samples generated:            "
        "0"
    )

    print(
        "Training performed:                    "
        "NO"
    )

    banner(
        "PHASE 5.1.1b COMPLETE / PARTIAL NEGATIVE CONTRACT FROZEN"
    )


if __name__ == "__main__":
    main()