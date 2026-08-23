"""
Phase 5.1.2b — Freeze T60 Evaluation Candidate-Generation Runtime Contract

This phase freezes the complete evaluation-side candidate-generation
contract.

NO NEGATIVES ARE GENERATED HERE.

Frozen here
-----------
1. Evaluation candidate universe:
      all 311,589 frozen Phase-3 Startup role nodes.

2. Negative eligibility for focal T60 event (o,b):
      exclude all Investor-Startup pairs positive before T60
      plus the focal positive startup.

3. Other T60 positive labels:
      NOT used to clean the candidate pool.

4. Evaluation negative count:
      99 per positive event (PAPER_SPECIFIED).

5. Distribution:
      uniform over eligible candidates.

6. Replacement:
      without replacement within one evaluation case.

7. Persistence:
      generate once and reuse permanently across validation/test passes.

8. Evaluation case identity:
      frozen Phase-2 interaction_id (event-level).

9. RNG:
      dedicated deterministic per-case seed,
      namespace ITRS_PHASE5_EVAL_NEGATIVE,
      base seed 42.

10. Post-generation full-T60 collision checks:
      diagnostic only; they MUST NOT trigger resampling.

This script does NOT:
- instantiate an RNG;
- generate evaluation negatives;
- generate training negatives;
- instantiate the model;
- create an optimizer;
- train;
- evaluate HR@10/NDCG@10;
- choose epoch count;
- choose early stopping;
- choose weight decay.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


# =============================================================================
# Paths
# =============================================================================

TRAINING_NEGATIVE_CONTRACT = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_1_1d_training_negative_sampling_runtime_contract.json"
)

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/phase_5_1_2a"
)

UNIVERSE_REGISTER = (
    AUDIT_DIR
    / "t60_evaluation_candidate_universe_register.csv"
)

POLICY_REGISTER = (
    AUDIT_DIR
    / "t60_evaluation_exclusion_policy_risk_register.csv"
)

COLLISION_SUMMARY = (
    AUDIT_DIR
    / "t60_evaluation_pool_collision_summary.csv"
)

SPLIT_OVERLAP_AUDIT = (
    AUDIT_DIR
    / "t60_validation_test_overlap_audit.csv"
)

AUDIT_MANIFEST = (
    AUDIT_DIR
    / "phase_5_1_2a_audit_manifest.json"
)

OUT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

CONTRACT_PATH = (
    OUT_DIR
    / "phase_5_1_2b_t60_evaluation_candidate_runtime_contract.json"
)

DECISION_REGISTER_PATH = (
    OUT_DIR
    / "phase_5_1_2b_t60_evaluation_candidate_decision_register.csv"
)

FREEZE_AUDIT_PATH = (
    OUT_DIR
    / "phase_5_1_2b_t60_evaluation_candidate_freeze_audit.csv"
)


# =============================================================================
# Frozen values
# =============================================================================

CANDIDATE_UNIVERSE = (
    "global_role_universe"
)

EXCLUSION_POLICY = (
    "exclude_prior_pairs_plus_focal"
)

EVALUATION_NEGATIVES_PER_POSITIVE = 99

SAMPLING_DISTRIBUTION = (
    "uniform_over_eligible_startups"
)

REPLACEMENT_SEMANTICS = (
    "without_replacement_within_case"
)

PERSISTENCE = (
    "generate_once_and_reuse"
)

CASE_IDENTITY = (
    "interaction_id"
)

EVALUATION_BASE_SEED = 42

SEED_NAMESPACE = (
    "ITRS_PHASE5_EVAL_NEGATIVE"
)

BIT_GENERATOR = (
    "numpy.random.PCG64"
)

EXPECTED = {
    "startup_universe": 311_589,
    "t60_events": 22_515,
    "validation_events": 2_251,
    "test_events": 20_264,
    "validation_test_pair_overlap": 33,
    "validation_test_funding_round_overlap": 1_315,
}


# =============================================================================
# Helpers
# =============================================================================

def banner(text: str) -> None:
    print("\n" + "=" * 118)
    print(text)
    print("=" * 118)


def require(
    condition: bool,
    message: str,
) -> None:
    if not bool(condition):
        raise AssertionError(message)


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def derive_case_seed_without_rng(
    experiment_split: str,
    interaction_id: str,
) -> int:
    """
    Deterministically derive one evaluation-case seed.

    IMPORTANT:
    This function performs hashing only.
    It does NOT instantiate an RNG.

    Contract:
        material =
            ITRS_PHASE5_EVAL_NEGATIVE
            |42
            |<experiment_split>
            |<interaction_id>

    First 8 SHA256 digest bytes:
        little-endian unsigned integer

    Then:
        modulo (2^63 - 1)
    """

    split = str(
        experiment_split
    ).strip().lower()

    interaction = str(
        interaction_id
    )

    require(
        split in {
            "validation",
            "test",
        },
        f"Unexpected experiment split: {split}",
    )

    require(
        len(interaction) > 0,
        "interaction_id cannot be empty",
    )

    material = (
        f"{SEED_NAMESPACE}|"
        f"{EVALUATION_BASE_SEED}|"
        f"{split}|"
        f"{interaction}"
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
        % (
            (2 ** 63)
            - 1
        )
    )


def write_json(
    path: Path,
    payload: dict,
) -> None:

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def get_metric(
    overlap_df: pd.DataFrame,
    metric: str,
) -> int:

    row = overlap_df.loc[
        overlap_df[
            "metric"
        ]
        == metric
    ]

    require(
        len(row) == 1,
        f"Missing overlap metric: {metric}",
    )

    return int(
        row.iloc[0][
            "value"
        ]
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.1.2b — "
        "FREEZE T60 EVALUATION "
        "CANDIDATE-GENERATION RUNTIME CONTRACT"
    )

    print(
        "Evaluation negatives generated:       0"
    )
    print(
        "RNG instantiated:                     NO"
    )
    print(
        "Model instantiated:                   NO"
    )
    print(
        "Optimizer created:                    NO"
    )
    print(
        "Training performed:                   NO"
    )
    print(
        "Evaluation performed:                 NO"
    )
    print(
        "Epoch count selected:                 NO"
    )
    print(
        "Early stopping selected:              NO"
    )
    print(
        "Weight decay selected:                NO"
    )

    # =========================================================================
    # Authoritative inputs
    # =========================================================================

    banner(
        "AUTHORITATIVE INPUT EXISTENCE"
    )

    required_paths = (
        TRAINING_NEGATIVE_CONTRACT,
        UNIVERSE_REGISTER,
        POLICY_REGISTER,
        COLLISION_SUMMARY,
        SPLIT_OVERLAP_AUDIT,
        AUDIT_MANIFEST,
    )

    for path in required_paths:

        require(
            path.exists(),
            f"Missing authoritative input: {path}",
        )

        print(
            f"FOUND  {path}"
        )

    # =========================================================================
    # Training negative contract remains frozen
    # =========================================================================

    banner(
        "PHASE 5.1.1d TRAINING CONTRACT RECHECK"
    )

    training_contract = json.loads(
        TRAINING_NEGATIVE_CONTRACT.read_text(
            encoding="utf-8"
        )
    )

    require(
        training_contract[
            "phase"
        ]
        == "5.1.1d",
        "Unexpected training contract phase",
    )

    require(
        training_contract[
            "status"
        ]
        == "FROZEN",
        "Training-negative contract not frozen",
    )

    require(
        training_contract[
            "paper_unspecified_reproduction_choices"
        ][
            "training_negative_positive_ratio"
        ]
        == 4,
        "Frozen training K drift",
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

    print(
        "Training negative contract:           FROZEN  PASS"
    )

    print(
        "Training K:                           4       PASS"
    )

    print(
        "Training candidate distribution:      uniform PASS"
    )

    # =========================================================================
    # Phase 5.1.2a audit status
    # =========================================================================

    banner(
        "PHASE 5.1.2a AUDIT RECHECK"
    )

    audit_manifest = json.loads(
        AUDIT_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    require(
        audit_manifest[
            "phase"
        ]
        == "5.1.2a",
        "Unexpected audit phase",
    )

    require(
        audit_manifest[
            "status"
        ]
        == "AUDIT_COMPLETE_NOT_FROZEN",
        "Unexpected 5.1.2a audit status",
    )

    require(
        audit_manifest[
            "evaluation_negative_samples_generated"
        ]
        is False,
        "5.1.2a unexpectedly generated negatives",
    )

    require(
        audit_manifest[
            "rng_instantiated"
        ]
        is False,
        "5.1.2a unexpectedly instantiated RNG",
    )

    require(
        audit_manifest[
            "training_performed"
        ]
        is False,
        "5.1.2a unexpectedly trained",
    )

    require(
        audit_manifest[
            "evaluation_performed"
        ]
        is False,
        "5.1.2a unexpectedly evaluated",
    )

    print(
        "5.1.2a audit complete:                YES  PASS"
    )

    print(
        "Evaluation negatives generated:       NO   PASS"
    )

    print(
        "RNG instantiated:                     NO   PASS"
    )

    print(
        "Training/evaluation performed:        NO   PASS"
    )

    # =========================================================================
    # Candidate universe freeze evidence
    # =========================================================================

    banner(
        "GLOBAL ROLE UNIVERSE FREEZE EVIDENCE"
    )

    universes = pd.read_csv(
        UNIVERSE_REGISTER
    )

    global_row = universes.loc[
        universes[
            "candidate_universe"
        ]
        == CANDIDATE_UNIVERSE
    ].copy()

    require(
        len(global_row) == 1,
        "Missing global-role-universe row",
    )

    global_row = (
        global_row.iloc[0]
    )

    require(
        int(
            global_row[
                "candidate_universe_size"
            ]
        )
        == EXPECTED[
            "startup_universe"
        ],
        "Global Startup universe count drift",
    )

    require(
        float(
            global_row[
                "t60_focal_outside_share"
            ]
        )
        == 0.0,
        "Global universe excludes T60 focal positives",
    )

    require(
        not bool(
            global_row[
                "uses_t60_labels_for_membership"
            ]
        ),
        "Global universe unexpectedly uses T60 labels",
    )

    pre_t60_row = universes.loc[
        universes[
            "candidate_universe"
        ]
        == "pre_t60_any"
    ]

    require(
        len(pre_t60_row) == 1,
        "Missing pre_t60_any comparison row",
    )

    pre_t60_outside_share = float(
        pre_t60_row.iloc[0][
            "t60_focal_outside_share"
        ]
    )

    require(
        pre_t60_outside_share > 0.40,
        (
            "Expected pre_t60_any to exclude "
            "a substantial T60 focal population"
        ),
    )

    print(
        "Frozen evaluation Startup universe:  "
        f"{EXPECTED['startup_universe']:,}"
    )

    print(
        "T60 focal positives outside universe:0"
    )

    print(
        "Universe membership uses T60 labels: NO"
    )

    print(
        "pre_t60_any focal exclusion share:   "
        f"{pre_t60_outside_share:.6%}"
    )

    # =========================================================================
    # Exclusion-policy freeze evidence
    # =========================================================================

    banner(
        "PRIOR-PLUS-FOCAL POLICY FREEZE EVIDENCE"
    )

    policies = pd.read_csv(
        POLICY_REGISTER
    )

    selected_policy = policies.loc[
        policies[
            "exclusion_policy"
        ]
        == EXCLUSION_POLICY
    ].copy()

    require(
        len(selected_policy) == 1,
        "Missing selected exclusion-policy row",
    )

    selected_policy = (
        selected_policy.iloc[0]
    )

    require(
        bool(
            selected_policy[
                "uses_pre_t60_history"
            ]
        ),
        "Selected policy does not use historical positives",
    )

    require(
        not bool(
            selected_policy[
                "uses_other_validation_labels"
            ]
        ),
        "Selected policy uses other validation labels",
    )

    require(
        not bool(
            selected_policy[
                "uses_other_test_labels"
            ]
        ),
        "Selected policy uses other test labels",
    )

    require(
        not bool(
            selected_policy[
                "historical_positive_collision_possible"
            ]
        ),
        "Selected policy permits historical-positive collisions",
    )

    require(
        bool(
            selected_policy[
                "other_t60_positive_collision_possible"
            ]
        ),
        (
            "Expected concurrent T60 positives to remain "
            "eligible under target-label-independent contract"
        ),
    )

    print(
        "Pre-T60 positives excluded:           YES  PASS"
    )

    print(
        "Focal T60 positive excluded:          YES  PASS"
    )

    print(
        "Other validation labels used:         NO   PASS"
    )

    print(
        "Other test labels used:               NO   PASS"
    )

    print(
        "Other T60 positives may remain:       YES  EXPECTED"
    )

    # =========================================================================
    # Collision evidence
    # =========================================================================

    banner(
        "SELECTED POLICY COLLISION EVIDENCE"
    )

    summary = pd.read_csv(
        COLLISION_SUMMARY
    )

    selected_summary = summary.loc[
        (
            summary[
                "scope"
            ]
            == "t60_overall"
        )
        & (
            summary[
                "candidate_universe"
            ]
            == CANDIDATE_UNIVERSE
        )
        & (
            summary[
                "exclusion_policy"
            ]
            == EXCLUSION_POLICY
        )
    ].copy()

    require(
        len(selected_summary) == 1,
        "Missing selected T60 collision-summary row",
    )

    selected_summary = (
        selected_summary.iloc[0]
    )

    positive_events = int(
        selected_summary[
            "positive_events"
        ]
    )

    require(
        positive_events
        == EXPECTED[
            "t60_events"
        ],
        "T60 event-count drift in collision summary",
    )

    negative_pool_min = int(
        selected_summary[
            "negative_pool_min"
        ]
    )

    negative_pool_mean = float(
        selected_summary[
            "negative_pool_mean"
        ]
    )

    historical_collision_probability = float(
        selected_summary[
            "mean_p_one_draw_hits_historical_positive"
        ]
    )

    concurrent_collision_probability_one_draw = float(
        selected_summary[
            "mean_p_one_draw_hits_other_t60_positive"
        ]
    )

    expected_concurrent_collisions_per_case = float(
        selected_summary[
            "mean_expected_other_t60_positive_collisions_in_99"
        ]
    )

    p_case_contains_concurrent_positive = float(
        selected_summary[
            "mean_p_99_contains_at_least_one_other_t60_positive"
        ]
    )

    require(
        negative_pool_min
        >= EVALUATION_NEGATIVES_PER_POSITIVE,
        "Selected evaluation pool cannot support 99 negatives",
    )

    require(
        historical_collision_probability
        == 0.0,
        "Historical positive collision remains",
    )

    require(
        concurrent_collision_probability_one_draw > 0.0,
        (
            "Expected other T60 positives to remain "
            "possible under selected policy"
        ),
    )

    require(
        p_case_contains_concurrent_positive < 0.005,
        (
            "Concurrent-positive list collision risk "
            "exceeds 0.5%; re-audit decision"
        ),
    )

    expected_cases_with_concurrent_positive = (
        positive_events
        * p_case_contains_concurrent_positive
    )

    expected_concurrent_negative_slots = (
        positive_events
        * expected_concurrent_collisions_per_case
    )

    print(
        f"T60 evaluation cases:                "
        f"{positive_events:,}"
    )

    print(
        f"Minimum eligible negative pool:      "
        f"{negative_pool_min:,}"
    )

    print(
        f"Mean eligible negative pool:         "
        f"{negative_pool_mean:,.6f}"
    )

    print(
        "Historical collision probability:    "
        f"{historical_collision_probability:.9f}"
    )

    print(
        "Other-T60 positive p / one draw:     "
        f"{concurrent_collision_probability_one_draw:.9f}"
    )

    print(
        "Expected T60 collisions / 99-list:   "
        f"{expected_concurrent_collisions_per_case:.9f}"
    )

    print(
        "P(99-list contains another T60 +):   "
        f"{p_case_contains_concurrent_positive:.9f}"
    )

    print(
        "Expected affected T60 lists:         "
        f"{expected_cases_with_concurrent_positive:,.3f}"
    )

    print(
        "Expected collided negative slots:    "
        f"{expected_concurrent_negative_slots:,.3f}"
    )

    # =========================================================================
    # Full-T60-label-aware comparison
    # =========================================================================

    banner(
        "HELD-OUT-LABEL-AWARE ALTERNATIVE REJECTION"
    )

    full_t60 = summary.loc[
        (
            summary[
                "scope"
            ]
            == "t60_overall"
        )
        & (
            summary[
                "candidate_universe"
            ]
            == CANDIDATE_UNIVERSE
        )
        & (
            summary[
                "exclusion_policy"
            ]
            == "exclude_prior_plus_all_t60_pairs"
        )
    ].copy()

    require(
        len(full_t60) == 1,
        "Missing full-T60 exclusion comparison row",
    )

    full_t60 = (
        full_t60.iloc[0]
    )

    require(
        float(
            full_t60[
                "mean_p_one_draw_hits_other_t60_positive"
            ]
        )
        == 0.0,
        "Full-T60 policy unexpectedly leaves T60 collisions",
    )

    require(
        float(
            full_t60[
                "mean_p_99_contains_at_least_one_other_t60_positive"
            ]
        )
        == 0.0,
        "Full-T60 policy unexpectedly leaves list collisions",
    )

    print(
        "All-T60 exclusion collision p:       0"
    )

    print(
        "All-T60 exclusion uses held-out labels:YES"
    )

    print(
        "Selected for reproduction:           NO"
    )

    print(
        "Reason: avoid undocumented dependence "
        "on complete T60 outcome labels."
    )

    # =========================================================================
    # Split-overlap recheck
    # =========================================================================

    banner(
        "FROZEN EVENT-LEVEL SPLIT RECHECK"
    )

    overlap = pd.read_csv(
        SPLIT_OVERLAP_AUDIT
    )

    pair_overlap = get_metric(
        overlap,
        "validation_test_pair_overlap",
    )

    funding_round_overlap = get_metric(
        overlap,
        "validation_test_funding_round_overlap",
    )

    require(
        pair_overlap
        == EXPECTED[
            "validation_test_pair_overlap"
        ],
        "Pair-overlap drift",
    )

    require(
        funding_round_overlap
        == EXPECTED[
            "validation_test_funding_round_overlap"
        ],
        "Funding-round-overlap drift",
    )

    print(
        "Evaluation case identity:            interaction_id"
    )

    print(
        "Deduplicate T60 events:              NO"
    )

    print(
        f"Validation/test pair overlap:         "
        f"{pair_overlap}  PASS"
    )

    print(
        f"Validation/test funding-round overlap:"
        f" {funding_round_overlap:,}  PASS"
    )

    # =========================================================================
    # Deterministic per-case seed contract — hashing only
    # =========================================================================

    banner(
        "DETERMINISTIC PER-CASE RNG CONTRACT"
    )

    # These are synthetic identifiers used ONLY to demonstrate
    # deterministic hashing. They are not data rows and no RNG
    # is instantiated.
    seed_examples = {
        "validation_example": (
            derive_case_seed_without_rng(
                "validation",
                "EXAMPLE_VALIDATION_INTERACTION",
            )
        ),
        "test_example": (
            derive_case_seed_without_rng(
                "test",
                "EXAMPLE_TEST_INTERACTION",
            )
        ),
    }

    print(
        f"Evaluation base seed:                "
        f"{EVALUATION_BASE_SEED}"
    )

    print(
        f"Seed namespace:                      "
        f"{SEED_NAMESPACE}"
    )

    print(
        "Case identity:                       "
        "experiment_split + interaction_id"
    )

    print(
        "Seed derivation:                     "
        "SHA256(namespace|42|split|interaction_id)"
    )

    print(
        f"Future BitGenerator:                 "
        f"{BIT_GENERATOR}"
    )

    print(
        "RNG instantiated by this script:     NO"
    )

    print()
    print(
        "Hash-only seed examples:"
    )

    for label, seed in seed_examples.items():
        print(
            f"  {label}: {seed}"
        )

    # =========================================================================
    # Frozen exact contract
    # =========================================================================

    banner(
        "EXACT FROZEN T60 EVALUATION CONTRACT"
    )

    mathematical_contract = (
        "For focal T60 event e=(o,b): "
        "N_eval(o,b) = all frozen Startup role nodes "
        "minus {Startups positive for investor o before T60} "
        "minus {focal Startup b}; "
        "draw exactly 99 distinct negatives uniformly once; "
        "do not use any other T60 labels for candidate exclusion; "
        "persist and reuse the resulting list."
    )

    print(
        mathematical_contract
    )

    print()

    print(
        "Positive candidates per case:         1"
    )

    print(
        "Negative candidates per case:         99"
    )

    print(
        "Total candidates per case:            100"
    )

    print(
        "Candidate universe:                   "
        "all 311,589 frozen Startup nodes"
    )

    print(
        "Historical positives excluded:        YES"
    )

    print(
        "Focal positive excluded from negatives:YES"
    )

    print(
        "Other T60 labels used for exclusion:  NO"
    )

    print(
        "Distribution:                         uniform"
    )

    print(
        "Replacement:                          without replacement"
    )

    print(
        "Regeneration across validation runs:  NO"
    )

    print(
        "Regeneration across checkpoints:      NO"
    )

    print(
        "Post-hoc collision resampling:         FORBIDDEN"
    )

    # =========================================================================
    # Decision register
    # =========================================================================

    decisions = pd.DataFrame(
        [
            {
                "decision_id": (
                    "evaluation_positive_count_per_case"
                ),
                "value": "1",
                "classification": (
                    "PAPER_SPECIFIED"
                ),
                "status": (
                    "FROZEN"
                ),
                "rationale": (
                    "ITRS explicitly reports one positive "
                    "instance in each validation/test case."
                ),
            },
            {
                "decision_id": (
                    "evaluation_negative_count_per_case"
                ),
                "value": "99",
                "classification": (
                    "PAPER_SPECIFIED"
                ),
                "status": (
                    "FROZEN"
                ),
                "rationale": (
                    "ITRS explicitly reports 99 randomly "
                    "sampled negatives per validation/test case."
                ),
            },
            {
                "decision_id": (
                    "evaluation_candidate_universe"
                ),
                "value": (
                    "all_311589_frozen_startup_role_nodes"
                ),
                "classification": (
                    "DATASET_ADAPTATION+"
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_1_2b"
                ),
                "rationale": (
                    "Preserves every frozen T60 focal positive "
                    "without using T60 labels to define membership."
                ),
            },
            {
                "decision_id": (
                    "evaluation_historical_positive_exclusion"
                ),
                "value": (
                    "exclude_all_pairs_positive_before_T60"
                ),
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_1_2b"
                ),
                "rationale": (
                    "Known historical positive pairs must not "
                    "receive negative labels."
                ),
            },
            {
                "decision_id": (
                    "evaluation_focal_positive_exclusion"
                ),
                "value": "exclude_focal_startup",
                "classification": (
                    "SEMANTIC_REQUIREMENT"
                ),
                "status": (
                    "FROZEN_PHASE_5_1_2b"
                ),
                "rationale": (
                    "The focal positive cannot simultaneously "
                    "occupy a negative slot."
                ),
            },
            {
                "decision_id": (
                    "evaluation_other_T60_positive_exclusion"
                ),
                "value": "do_not_exclude",
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_1_2b"
                ),
                "rationale": (
                    "Avoids undocumented candidate construction "
                    "using complete validation/test outcome labels; "
                    "audited 99-list collision probability is ~0.24%."
                ),
            },
            {
                "decision_id": (
                    "evaluation_sampling_distribution"
                ),
                "value": (
                    SAMPLING_DISTRIBUTION
                ),
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_1_2b"
                ),
                "rationale": (
                    "Minimal interpretation of random sampling."
                ),
            },
            {
                "decision_id": (
                    "evaluation_sampling_replacement"
                ),
                "value": (
                    REPLACEMENT_SEMANTICS
                ),
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_1_2b"
                ),
                "rationale": (
                    "Produces 99 distinct negatives; "
                    "all pools are vastly larger than 99."
                ),
            },
            {
                "decision_id": (
                    "evaluation_candidate_persistence"
                ),
                "value": (
                    PERSISTENCE
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_1_2b"
                ),
                "rationale": (
                    "Validation/test metrics must use an immutable "
                    "comparison set across checkpoints and reruns."
                ),
            },
            {
                "decision_id": (
                    "evaluation_case_identity"
                ),
                "value": (
                    CASE_IDENTITY
                ),
                "classification": (
                    "INHERITED_PHASE_2_CONSTRAINT"
                ),
                "status": (
                    "FROZEN"
                ),
                "rationale": (
                    "Phase 2 froze event-level validation/test "
                    "splits and explicitly retained repeated pairs."
                ),
            },
            {
                "decision_id": (
                    "evaluation_sampler_base_seed"
                ),
                "value": str(
                    EVALUATION_BASE_SEED
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_1_2b"
                ),
                "rationale": (
                    "Continues the reproduction seed convention "
                    "using an evaluation-specific RNG namespace."
                ),
            },
            {
                "decision_id": (
                    "evaluation_sampler_seed_derivation"
                ),
                "value": (
                    "SHA256("
                    "ITRS_PHASE5_EVAL_NEGATIVE|42|"
                    "experiment_split|interaction_id"
                    ") -> first_8_bytes_little_endian_mod_2^63_minus_1"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_1_2b"
                ),
                "rationale": (
                    "Makes each candidate list reproducible "
                    "independently of dataframe processing order."
                ),
            },
            {
                "decision_id": (
                    "evaluation_bit_generator"
                ),
                "value": BIT_GENERATOR,
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_1_2b"
                ),
                "rationale": (
                    "Explicit generator backend prevents hidden "
                    "dependence on an unspecified global RNG."
                ),
            },
            {
                "decision_id": (
                    "post_generation_T60_collision_handling"
                ),
                "value": (
                    "diagnose_only_never_resample"
                ),
                "classification": (
                    "TEMPORAL_LABEL_INDEPENDENCE_GUARD"
                ),
                "status": (
                    "FROZEN_PHASE_5_1_2b"
                ),
                "rationale": (
                    "Using full T60 labels after generation to "
                    "replace collisions would silently change the "
                    "frozen candidate eligibility contract."
                ),
            },
            {
                "decision_id": (
                    "training_epoch_count"
                ),
                "value": None,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": (
                    "DEFERRED"
                ),
                "rationale": (
                    "Requires training-control audit."
                ),
            },
            {
                "decision_id": (
                    "early_stopping"
                ),
                "value": None,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": (
                    "DEFERRED"
                ),
                "rationale": (
                    "Requires training-control audit."
                ),
            },
            {
                "decision_id": (
                    "weight_decay"
                ),
                "value": None,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": (
                    "DEFERRED"
                ),
                "rationale": (
                    "Requires optimizer audit."
                ),
            },
        ]
    )

    # =========================================================================
    # Freeze checks
    # =========================================================================

    checks = [
        (
            "training_negative_contract_still_frozen",
            training_contract[
                "status"
            ]
            == "FROZEN",
        ),
        (
            "phase_5_1_2a_audit_complete",
            audit_manifest[
                "status"
            ]
            == "AUDIT_COMPLETE_NOT_FROZEN",
        ),
        (
            "global_universe_size_311589",
            int(
                global_row[
                    "candidate_universe_size"
                ]
            )
            == EXPECTED[
                "startup_universe"
            ],
        ),
        (
            "global_universe_retains_all_focals",
            float(
                global_row[
                    "t60_focal_outside_share"
                ]
            )
            == 0.0,
        ),
        (
            "global_universe_not_T60_label_defined",
            not bool(
                global_row[
                    "uses_t60_labels_for_membership"
                ]
            ),
        ),
        (
            "historical_collision_zero",
            historical_collision_probability
            == 0.0,
        ),
        (
            "pool_supports_99",
            negative_pool_min
            >= EVALUATION_NEGATIVES_PER_POSITIVE,
        ),
        (
            "selected_policy_uses_no_other_validation_labels",
            not bool(
                selected_policy[
                    "uses_other_validation_labels"
                ]
            ),
        ),
        (
            "selected_policy_uses_no_other_test_labels",
            not bool(
                selected_policy[
                    "uses_other_test_labels"
                ]
            ),
        ),
        (
            "concurrent_positive_risk_audited_and_small",
            (
                p_case_contains_concurrent_positive
                > 0.0
            )
            and (
                p_case_contains_concurrent_positive
                < 0.005
            ),
        ),
        (
            "evaluation_negative_count_is_99",
            EVALUATION_NEGATIVES_PER_POSITIVE
            == 99,
        ),
        (
            "evaluation_sampling_is_uniform",
            SAMPLING_DISTRIBUTION
            == "uniform_over_eligible_startups",
        ),
        (
            "evaluation_without_replacement",
            REPLACEMENT_SEMANTICS
            == "without_replacement_within_case",
        ),
        (
            "evaluation_candidates_fixed_once",
            PERSISTENCE
            == "generate_once_and_reuse",
        ),
        (
            "event_level_case_identity_preserved",
            CASE_IDENTITY
            == "interaction_id",
        ),
        (
            "no_negative_rows_generated",
            True,
        ),
        (
            "no_rng_instantiated",
            True,
        ),
        (
            "no_training_performed",
            True,
        ),
        (
            "no_evaluation_performed",
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
            freeze_audit[
                "result"
            ]
            == "PASS"
        ).all(),
        (
            "One or more Phase-5.1.2b "
            "freeze checks failed"
        ),
    )

    # =========================================================================
    # Contract
    # =========================================================================

    contract = {
        "phase": "5.1.2b",
        "title": (
            "T60 Evaluation Candidate-Generation "
            "Runtime Contract Freeze"
        ),
        "status": "FROZEN",

        "evaluation_negative_samples_generated": False,
        "rng_instantiated": False,
        "model_instantiated": False,
        "optimizer_created": False,
        "training_performed": False,
        "evaluation_performed": False,

        "paper_specified": {
            "positive_candidates_per_case": 1,
            "random_negatives_per_case": (
                EVALUATION_NEGATIVES_PER_POSITIVE
            ),
            "candidate_count_per_case": 100,
            "metrics": [
                "HR@10",
                "NDCG@10",
            ],
        },

        "candidate_universe": {
            "name": CANDIDATE_UNIVERSE,
            "count": EXPECTED[
                "startup_universe"
            ],
            "definition": (
                "all frozen Phase-3 Startup role nodes"
            ),
            "classification": (
                "DATASET_ADAPTATION+"
                "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
            ),
            "uses_T60_labels_for_membership": False,
        },

        "negative_eligibility": {
            "policy": (
                EXCLUSION_POLICY
            ),
            "definition": (
                "For focal event (o,b,T60), eligible negatives are "
                "all frozen Startup role nodes excluding all Startups "
                "with a positive pair for investor o before T60 and "
                "excluding focal Startup b."
            ),
            "exclude_pre_T60_positive_pairs": True,
            "exclude_focal_positive": True,
            "exclude_other_validation_positives": False,
            "exclude_other_test_positives": False,
            "exclude_other_T60_positives": False,
            "classification": (
                "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
            ),
        },

        "sampling": {
            "K": (
                EVALUATION_NEGATIVES_PER_POSITIVE
            ),
            "distribution": "uniform",
            "without_replacement": True,
            "generate_once": True,
            "reuse_for_every_validation_pass": True,
            "reuse_for_every_test_pass": True,
            "regenerate_across_training_epochs": False,
            "regenerate_across_checkpoints": False,
        },

        "case_identity": {
            "unit": "interaction_event",
            "key": CASE_IDENTITY,
            "deduplicate_repeated_pairs": False,
            "classification": (
                "INHERITED_PHASE_2_CONSTRAINT"
            ),
        },

        "rng_runtime": {
            "base_seed": (
                EVALUATION_BASE_SEED
            ),
            "namespace": (
                SEED_NAMESPACE
            ),
            "case_seed_key": (
                "experiment_split + interaction_id"
            ),
            "seed_derivation": (
                "SHA256("
                "ITRS_PHASE5_EVAL_NEGATIVE|42|"
                "experiment_split|interaction_id"
                "); first 8 digest bytes interpreted "
                "little-endian unsigned; modulo (2^63 - 1)"
            ),
            "bit_generator": (
                BIT_GENERATOR
            ),
            "dedicated_evaluation_rng": True,
            "share_rng_with_training_negative_sampler": False,
            "share_rng_with_model": False,
            "share_rng_with_dataloader": False,
            "row_order_independent": True,
        },

        "collision_contract": {
            "historical_positive_collision_probability": (
                historical_collision_probability
            ),
            "expected_other_T60_positive_collisions_per_99_list": (
                expected_concurrent_collisions_per_case
            ),
            "probability_99_list_contains_other_T60_positive": (
                p_case_contains_concurrent_positive
            ),
            "expected_T60_lists_with_at_least_one_concurrent_positive": (
                expected_cases_with_concurrent_positive
            ),
            "expected_total_concurrent_positive_negative_slots": (
                expected_concurrent_negative_slots
            ),
            "full_T60_labels_may_be_used_for_post_generation_diagnostics": True,
            "full_T60_labels_may_modify_candidate_lists": False,
            "resample_detected_T60_positive_collision": False,
            "interpretation": (
                "Concurrent T60-positive collisions are accepted as "
                "the small cost of target-label-independent candidate "
                "generation. They are reported diagnostically only."
            ),
        },

        "rejected_alternatives": {
            "experiment_any_candidate_universe": (
                "Held-out T60 observations define membership."
            ),
            "pre_t60_any_candidate_universe": (
                "Excludes >40% of T60 focal positives and creates "
                "warm-negative/cold-positive asymmetry."
            ),
            "exclude_focal_only": (
                "Can relabel known historical positives as negatives."
            ),
            "exclude_prior_plus_same_split_t60_pairs": (
                "Uses split labels but still leaves opposite-split "
                "T60 positives."
            ),
            "exclude_prior_plus_all_t60_pairs": (
                "Positive-safe, but candidate construction depends on "
                "the complete held-out T60 outcome set."
            ),
        },

        "still_unresolved_original_phase_5_handoff_decisions": [
            "training epoch count",
            "early stopping",
            "weight decay",
        ],

        "authoritative_input_hashes": {
            str(
                TRAINING_NEGATIVE_CONTRACT
            ): sha256_file(
                TRAINING_NEGATIVE_CONTRACT
            ),
            str(
                UNIVERSE_REGISTER
            ): sha256_file(
                UNIVERSE_REGISTER
            ),
            str(
                POLICY_REGISTER
            ): sha256_file(
                POLICY_REGISTER
            ),
            str(
                COLLISION_SUMMARY
            ): sha256_file(
                COLLISION_SUMMARY
            ),
            str(
                SPLIT_OVERLAP_AUDIT
            ): sha256_file(
                SPLIT_OVERLAP_AUDIT
            ),
            str(
                AUDIT_MANIFEST
            ): sha256_file(
                AUDIT_MANIFEST
            ),
        },
    }

    # =========================================================================
    # Write frozen artifacts
    # =========================================================================

    banner(
        "WRITE FROZEN PHASE-5.1.2b CONTRACT"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        CONTRACT_PATH,
        contract,
    )

    decisions.to_csv(
        DECISION_REGISTER_PATH,
        index=False,
    )

    freeze_audit.to_csv(
        FREEZE_AUDIT_PATH,
        index=False,
    )

    for path in (
        CONTRACT_PATH,
        DECISION_REGISTER_PATH,
        FREEZE_AUDIT_PATH,
    ):
        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Final status
    # =========================================================================

    banner(
        "PHASE 5.1.2b FINAL STATUS"
    )

    print(
        "Evaluation positives / case:          FROZEN -> 1"
    )

    print(
        "Evaluation negatives / case:          FROZEN -> 99"
    )

    print(
        "Evaluation candidate universe:        "
        "FROZEN -> all 311,589 Startup nodes"
    )

    print(
        "Historical-positive exclusion:        FROZEN -> YES"
    )

    print(
        "Focal-positive exclusion:             FROZEN -> YES"
    )

    print(
        "Other T60-positive exclusion:         FROZEN -> NO"
    )

    print(
        "Evaluation distribution:              FROZEN -> uniform"
    )

    print(
        "Evaluation replacement:               "
        "FROZEN -> without replacement"
    )

    print(
        "Evaluation candidate persistence:     "
        "FROZEN -> generate once / reuse"
    )

    print(
        "Evaluation case identity:             "
        "FROZEN -> interaction_id"
    )

    print(
        "Evaluation sampler base seed:         FROZEN -> 42"
    )

    print(
        "Evaluation sampler namespace:         "
        f"FROZEN -> {SEED_NAMESPACE}"
    )

    print(
        "Post-hoc T60 collision resampling:    FROZEN -> FORBIDDEN"
    )

    print()

    print(
        "Evaluation negatives generated:       0"
    )

    print(
        "RNG instantiated:                     NO"
    )

    print(
        "Training performed:                   NO"
    )

    print(
        "Evaluation performed:                 NO"
    )

    print()

    print(
        "Original Phase-5 handoff decisions still unresolved:"
    )

    print(
        "  1. training epoch count"
    )

    print(
        "  2. early stopping"
    )

    print(
        "  3. weight decay"
    )

    banner(
        "PHASE 5.1.2b COMPLETE / "
        "EVALUATION CANDIDATE-GENERATION CONTRACT FULLY FROZEN"
    )


if __name__ == "__main__":
    main()