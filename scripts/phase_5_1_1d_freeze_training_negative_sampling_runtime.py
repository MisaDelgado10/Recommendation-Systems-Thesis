"""
Phase 5.1.1d — Freeze Training Negative-Sampling Runtime Contract

This phase freezes the COMPLETE TRAINING-SIDE negative-sampling contract.

It DOES NOT:
- generate negative samples;
- instantiate an RNG;
- instantiate the model;
- create an optimizer;
- choose the number of epochs;
- choose early stopping;
- choose weight decay;
- freeze evaluation candidate generation;
- train.

Frozen here
-----------
1. Training negative:positive ratio = 4:1.
2. Uniform sampling over the eligible set frozen in Phase 5.1.1b.
3. Without replacement within each focal positive event.
4. Negative sets regenerated once per epoch.
5. Dedicated deterministic epoch-indexed sampler seed contract.
6. Base sampler seed = 42.

The ratio is NOT reported by ITRS. It is an explicit
PAPER_UNSPECIFIED_REPRODUCTION_CHOICE, supported by:
- the NCF-style scoring architecture used by ITRS;
- the original NCF training precedent of four negatives per positive;
- the Phase-5.1.1c workload audit.

No negative row is produced by this script.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


# =============================================================================
# Paths
# =============================================================================

PHASE_5_1_1B_CONTRACT = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_1_1b_training_negative_semantics_contract.json"
)

PHASE_5_1_1C_DIR = Path(
    "data/experimental/phase_5/audits/phase_5_1_1c"
)

RATIO_AUDIT = (
    PHASE_5_1_1C_DIR
    / "training_negative_ratio_workload_audit.csv"
)

DISTRIBUTION_REGISTER = (
    PHASE_5_1_1C_DIR
    / "training_negative_distribution_option_register.csv"
)

REPLACEMENT_REGISTER = (
    PHASE_5_1_1C_DIR
    / "training_negative_replacement_option_register.csv"
)

RESAMPLING_REGISTER = (
    PHASE_5_1_1C_DIR
    / "training_negative_resampling_option_register.csv"
)

RNG_REGISTER = (
    PHASE_5_1_1C_DIR
    / "training_negative_rng_option_register.csv"
)

AUDIT_5_1_1C_MANIFEST = (
    PHASE_5_1_1C_DIR
    / "phase_5_1_1c_audit_manifest.json"
)

OUT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

CONTRACT_PATH = (
    OUT_DIR
    / "phase_5_1_1d_training_negative_sampling_runtime_contract.json"
)

DECISION_REGISTER_PATH = (
    OUT_DIR
    / "phase_5_1_1d_training_negative_sampling_decision_register.csv"
)

FREEZE_AUDIT_PATH = (
    OUT_DIR
    / "phase_5_1_1d_training_negative_sampling_freeze_audit.csv"
)


# =============================================================================
# Frozen values selected in Phase 5.1.1d
# =============================================================================

NEGATIVES_PER_POSITIVE = 4

SAMPLING_DISTRIBUTION = (
    "uniform_over_eligible_startups"
)

REPLACEMENT_SEMANTICS = (
    "without_replacement_within_each_positive"
)

RESAMPLING_CADENCE = (
    "regenerate_once_per_epoch"
)

RNG_STRATEGY = (
    "deterministic_epoch_indexed_sampler_seed"
)

SAMPLER_BASE_SEED = 42

SEED_NAMESPACE = (
    "ITRS_PHASE5_TRAIN_NEGATIVE"
)

EPOCH_INDEX_BASE = 0

PAPER_BATCH_SIZE = 512

EXPECTED_TRAINING_POSITIVES = 1_073_249
EXPECTED_NEGATIVES_PER_EPOCH = 4_292_996
EXPECTED_LABELED_EXAMPLES_PER_EPOCH = 5_366_245
EXPECTED_BATCHES_PER_EPOCH = 10_481

EXPECTED_STARTUP_UNIVERSE = 311_589


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


def derive_epoch_seed_without_rng(
    epoch_index: int,
) -> int:
    """
    Deterministically derive the epoch sampler seed.

    IMPORTANT:
    This performs hashing only. It does NOT instantiate an RNG.

    Contract:
        material =
            "ITRS_PHASE5_TRAIN_NEGATIVE|42|<epoch_index>"

        SHA256(material)

    The first 8 digest bytes are interpreted as an unsigned
    little-endian integer and restricted to signed 63-bit range.
    """

    require(
        epoch_index >= 0,
        "epoch_index must be nonnegative",
    )

    material = (
        f"{SEED_NAMESPACE}|"
        f"{SAMPLER_BASE_SEED}|"
        f"{epoch_index}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        material
    ).digest()

    seed = (
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

    return seed


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


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.1.1d — "
        "FREEZE TRAINING NEGATIVE-SAMPLING RUNTIME CONTRACT"
    )

    print("Negative rows generated:          0")
    print("RNG instantiated:                 NO")
    print("Model instantiated:               NO")
    print("Optimizer created:                NO")
    print("Training performed:               NO")
    print("Epoch count selected:             NO")
    print("Early stopping selected:          NO")
    print("Weight decay selected:            NO")
    print("Evaluation runtime frozen:        NO")

    # =========================================================================
    # Authoritative inputs
    # =========================================================================

    banner("AUTHORITATIVE INPUT EXISTENCE")

    required = (
        PHASE_5_1_1B_CONTRACT,
        RATIO_AUDIT,
        DISTRIBUTION_REGISTER,
        REPLACEMENT_REGISTER,
        RESAMPLING_REGISTER,
        RNG_REGISTER,
        AUDIT_5_1_1C_MANIFEST,
    )

    for path in required:

        require(
            path.exists(),
            f"Missing authoritative input: {path}",
        )

        print(f"FOUND  {path}")

    # =========================================================================
    # Recheck Phase 5.1.1b
    # =========================================================================

    banner("PHASE 5.1.1b SEMANTIC CONTRACT RECHECK")

    semantic_contract = json.loads(
        PHASE_5_1_1B_CONTRACT.read_text(
            encoding="utf-8"
        )
    )

    require(
        semantic_contract["phase"]
        == "5.1.1b",
        "Unexpected semantic-contract phase",
    )

    require(
        semantic_contract["status"]
        == "FROZEN",
        "Phase 5.1.1b is not frozen",
    )

    universe = semantic_contract[
        "training_negative_candidate_universe"
    ]

    exclusion = semantic_contract[
        "training_negative_exclusion"
    ]

    require(
        int(universe["count"])
        == EXPECTED_STARTUP_UNIVERSE,
        "Frozen Startup universe drift",
    )

    require(
        exclusion[
            "exclude_prior_positive_pairs"
        ]
        is True,
        "Prior-positive exclusion drift",
    )

    require(
        exclusion[
            "exclude_other_positive_pairs_in_target_segment"
        ]
        is True,
        "Target-positive exclusion drift",
    )

    require(
        exclusion[
            "exclude_future_positive_pairs"
        ]
        is False,
        "Future-positive exclusion drift",
    )

    require(
        exclusion[
            "use_T60_labels_for_T1_T59_sampling"
        ]
        is False,
        "T60 leakage guard drift",
    )

    print(
        "Startup universe:                 "
        f"{int(universe['count']):,}  PASS"
    )

    print(
        "Prior positives excluded:         YES  PASS"
    )

    print(
        "Same-target positives excluded:   YES  PASS"
    )

    print(
        "Future positives excluded:        NO   PASS"
    )

    print(
        "T60 labels used in training:      NO   PASS"
    )

    # =========================================================================
    # Recheck Phase 5.1.1c manifest
    # =========================================================================

    banner("PHASE 5.1.1c AUDIT RECHECK")

    audit_manifest = json.loads(
        AUDIT_5_1_1C_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    require(
        audit_manifest["phase"]
        == "5.1.1c",
        "Unexpected 5.1.1c audit phase",
    )

    require(
        audit_manifest["status"]
        == "AUDIT_COMPLETE_NOT_FROZEN",
        "Unexpected 5.1.1c audit status",
    )

    require(
        audit_manifest[
            "negative_samples_generated"
        ]
        is False,
        "5.1.1c unexpectedly generated negatives",
    )

    require(
        audit_manifest[
            "rng_instantiated"
        ]
        is False,
        "5.1.1c unexpectedly instantiated RNG",
    )

    require(
        audit_manifest[
            "training_performed"
        ]
        is False,
        "5.1.1c unexpectedly trained",
    )

    print(
        "5.1.1c audit status:              PASS"
    )

    print(
        "5.1.1c generated negatives:       NO   PASS"
    )

    print(
        "5.1.1c instantiated RNG:           NO   PASS"
    )

    print(
        "5.1.1c performed training:         NO   PASS"
    )

    # =========================================================================
    # Ratio freeze evidence
    # =========================================================================

    banner("4:1 RATIO FREEZE EVIDENCE")

    ratios = pd.read_csv(
        RATIO_AUDIT
    )

    row4 = ratios.loc[
        ratios[
            "negatives_per_positive"
        ]
        == NEGATIVES_PER_POSITIVE
    ].copy()

    require(
        len(row4) == 1,
        "Expected exactly one 4:1 audit row",
    )

    row4 = row4.iloc[0]

    require(
        int(
            row4[
                "training_positive_events"
            ]
        )
        == EXPECTED_TRAINING_POSITIVES,
        "Training-positive count drift",
    )

    require(
        int(
            row4[
                "negative_examples_per_epoch"
            ]
        )
        == EXPECTED_NEGATIVES_PER_EPOCH,
        "4:1 negative-count arithmetic drift",
    )

    require(
        int(
            row4[
                "labeled_examples_per_epoch"
            ]
        )
        == EXPECTED_LABELED_EXAMPLES_PER_EPOCH,
        "4:1 labeled-count arithmetic drift",
    )

    require(
        int(
            row4[
                "batches_per_epoch_batch512"
            ]
        )
        == EXPECTED_BATCHES_PER_EPOCH,
        "4:1 batch-count arithmetic drift",
    )

    require(
        bool(
            row4[
                "without_replacement_feasible_for_all_events"
            ]
        ),
        "4:1 is not feasible without replacement",
    )

    print(
        f"Training positives:               "
        f"{EXPECTED_TRAINING_POSITIVES:,}"
    )

    print(
        f"Negatives / positive:             "
        f"{NEGATIVES_PER_POSITIVE}"
    )

    print(
        f"Negatives / epoch:                "
        f"{EXPECTED_NEGATIVES_PER_EPOCH:,}"
    )

    print(
        f"Labeled examples / epoch:         "
        f"{EXPECTED_LABELED_EXAMPLES_PER_EPOCH:,}"
    )

    print(
        f"Batches / epoch @ 512:            "
        f"{EXPECTED_BATCHES_PER_EPOCH:,}"
    )

    print(
        "Without-replacement feasible:      YES  PASS"
    )

    print(
        "Future-positive diagnostic draws:  "
        f"{float(row4['uniform_reference_expected_future_positive_draws_per_epoch']):,.6f}"
    )

    # =========================================================================
    # Distribution freeze evidence
    # =========================================================================

    banner("UNIFORM DISTRIBUTION FREEZE EVIDENCE")

    distributions = pd.read_csv(
        DISTRIBUTION_REGISTER
    )

    uniform = distributions.loc[
        distributions["option"]
        == SAMPLING_DISTRIBUTION
    ].copy()

    require(
        len(uniform) == 1,
        "Missing uniform-distribution audit row",
    )

    uniform = uniform.iloc[0]

    require(
        not bool(
            uniform[
                "uses_future_information"
            ]
        ),
        "Uniform sampler unexpectedly marked future-aware",
    )

    require(
        uniform[
            "audit_assessment"
        ]
        == "STRONG_CANDIDATE_FOR_STRICT_REPRODUCTION",
        "Unexpected uniform-sampling audit assessment",
    )

    print(
        "Distribution:                     "
        "uniform over frozen eligible set"
    )

    print(
        "Uses future information:           NO   PASS"
    )

    # =========================================================================
    # Replacement freeze evidence
    # =========================================================================

    banner("WITHOUT-REPLACEMENT FREEZE EVIDENCE")

    replacements = pd.read_csv(
        REPLACEMENT_REGISTER
    )

    without_replacement = replacements.loc[
        replacements["option"]
        == REPLACEMENT_SEMANTICS
    ].copy()

    require(
        len(without_replacement) == 1,
        "Missing without-replacement audit row",
    )

    without_replacement = (
        without_replacement.iloc[0]
    )

    require(
        not bool(
            without_replacement[
                "duplicate_negative_within_positive_possible"
            ]
        ),
        "Selected replacement contract allows duplicates",
    )

    require(
        bool(
            without_replacement[
                "feasible_at_K99_for_all_training_events"
            ]
        ),
        "Without-replacement feasibility drift",
    )

    print(
        "Replacement:                      "
        "without replacement within focal event"
    )

    print(
        "Duplicate startup within focal set:NO   PASS"
    )

    # =========================================================================
    # Resampling freeze evidence
    # =========================================================================

    banner("PER-EPOCH RESAMPLING FREEZE EVIDENCE")

    resampling = pd.read_csv(
        RESAMPLING_REGISTER
    )

    per_epoch = resampling.loc[
        resampling["option"]
        == RESAMPLING_CADENCE
    ].copy()

    require(
        len(per_epoch) == 1,
        "Missing per-epoch resampling audit row",
    )

    per_epoch = per_epoch.iloc[0]

    require(
        per_epoch[
            "audit_assessment"
        ]
        == "STRONG_CANDIDATE",
        "Unexpected per-epoch resampling assessment",
    )

    require(
        not bool(
            per_epoch[
                "depends_on_batch_order"
            ]
        ),
        "Per-epoch regeneration unexpectedly batch-order dependent",
    )

    print(
        "Resampling cadence:                "
        "once per epoch"
    )

    print(
        "Depends on mini-batch order:        NO   PASS"
    )

    # =========================================================================
    # RNG freeze evidence
    # =========================================================================

    banner("DETERMINISTIC EPOCH-INDEXED RNG CONTRACT")

    rng_options = pd.read_csv(
        RNG_REGISTER
    )

    epoch_rng = rng_options.loc[
        rng_options["option"]
        == RNG_STRATEGY
    ].copy()

    require(
        len(epoch_rng) == 1,
        "Missing epoch-indexed RNG audit row",
    )

    epoch_rng = epoch_rng.iloc[0]

    require(
        bool(
            epoch_rng[
                "exact_epoch_reconstruction"
            ]
        ),
        "Selected RNG strategy cannot reconstruct epochs",
    )

    require(
        not bool(
            epoch_rng[
                "coupled_to_unrelated_rng_consumption"
            ]
        ),
        "Selected RNG remains coupled to other RNG operations",
    )

    seed_examples = {
        str(epoch): derive_epoch_seed_without_rng(
            epoch
        )
        for epoch in range(5)
    }

    print(
        f"Sampler base seed:                 "
        f"{SAMPLER_BASE_SEED}"
    )

    print(
        f"Seed namespace:                    "
        f"{SEED_NAMESPACE}"
    )

    print(
        "Epoch index base:                   0"
    )

    print(
        "Seed derivation:                    "
        "SHA256(namespace|base_seed|epoch)"
    )

    print(
        "RNG instantiated by this script:    NO"
    )

    print()
    print(
        "First five derived epoch seeds "
        "(hashing only):"
    )

    for epoch, seed in seed_examples.items():
        print(
            f"  epoch {epoch}: {seed}"
        )

    # =========================================================================
    # Exact frozen runtime semantics
    # =========================================================================

    banner("EXACT FROZEN TRAINING SAMPLING CONTRACT")

    sampling_expression = (
        "For each positive event e=(o,b,h), h in 1..59: "
        "draw K=4 distinct Startup candidates uniformly from "
        "N(o,h) = {all frozen Startup role nodes} minus "
        "{Startup b': first_positive_segment(o,b') <= h}; "
        "regenerate the four negatives once per epoch; "
        "future-positive pairs remain eligible."
    )

    print(sampling_expression)

    print()
    print(
        "Negative ratio:                     4:1"
    )
    print(
        "Distribution:                       uniform"
    )
    print(
        "Within-positive replacement:        none"
    )
    print(
        "Regeneration:                       every epoch"
    )
    print(
        "Future positives excluded:          NO"
    )
    print(
        "T60 labels used:                    NO"
    )
    print(
        "Negative sampler shares global RNG: NO"
    )

    # =========================================================================
    # Decision register
    # =========================================================================

    decisions = pd.DataFrame(
        [
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
                "status": "INHERITED_FROZEN_5_1_1b",
                "rationale": (
                    "Preserve full frozen Startup role universe."
                ),
            },
            {
                "decision_id": (
                    "training_positive_pair_exclusion"
                ),
                "value": (
                    "exclude_first_positive_segment_le_target"
                ),
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": "INHERITED_FROZEN_5_1_1b",
                "rationale": (
                    "Prevents historical and same-target positive "
                    "pairs from receiving contradictory negative labels."
                ),
            },
            {
                "decision_id": (
                    "training_negative_positive_ratio"
                ),
                "value": "4:1",
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_1_1d",
                "rationale": (
                    "ITRS does not report the ratio. Four negatives "
                    "per positive follows the original NCF training "
                    "precedent used as the closest scoring-family "
                    "reference and remains computationally tractable."
                ),
            },
            {
                "decision_id": (
                    "training_negative_distribution"
                ),
                "value": (
                    "uniform_over_eligible_startups"
                ),
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_1_1d",
                "rationale": (
                    "Minimal interpretation of random sampling "
                    "without adding popularity or model-score weighting."
                ),
            },
            {
                "decision_id": (
                    "training_negative_replacement"
                ),
                "value": (
                    "without_replacement_within_each_positive"
                ),
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_1_1d",
                "rationale": (
                    "Produces four distinct Startup negatives per "
                    "positive; all candidate pools are vastly larger "
                    "than K=4."
                ),
            },
            {
                "decision_id": (
                    "training_negative_resampling_cadence"
                ),
                "value": (
                    "regenerate_once_per_epoch"
                ),
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_1_1d",
                "rationale": (
                    "Provides high negative diversity without coupling "
                    "sampling to mini-batch execution order."
                ),
            },
            {
                "decision_id": (
                    "training_negative_sampler_base_seed"
                ),
                "value": str(
                    SAMPLER_BASE_SEED
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_1_1d",
                "rationale": (
                    "Continues the reproduction seed convention while "
                    "remaining independent of neural/dataloader RNG."
                ),
            },
            {
                "decision_id": (
                    "training_negative_sampler_seed_derivation"
                ),
                "value": (
                    "SHA256("
                    "ITRS_PHASE5_TRAIN_NEGATIVE|42|epoch_index"
                    ") -> first_8_bytes_little_endian_mod_2^63_minus_1"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_1_1d",
                "rationale": (
                    "Allows deterministic independent reconstruction "
                    "of any epoch without replaying prior epochs."
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
                "status": "DEFERRED",
                "rationale": (
                    "Not part of the negative-sampling contract."
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
                "status": "DEFERRED",
                "rationale": (
                    "Requires separate training-control audit."
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
                "status": "DEFERRED",
                "rationale": (
                    "Requires separate optimizer audit."
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
                    "Must be resolved before validation/test evaluation."
                ),
            },
        ]
    )

    # =========================================================================
    # Freeze audit
    # =========================================================================

    checks = [
        (
            "phase_5_1_1b_contract_frozen",
            semantic_contract["status"]
            == "FROZEN",
        ),
        (
            "phase_5_1_1c_audit_complete",
            audit_manifest["status"]
            == "AUDIT_COMPLETE_NOT_FROZEN",
        ),
        (
            "startup_universe_311589",
            int(universe["count"])
            == EXPECTED_STARTUP_UNIVERSE,
        ),
        (
            "ratio_is_4",
            NEGATIVES_PER_POSITIVE
            == 4,
        ),
        (
            "4_to_1_negative_count",
            int(
                row4[
                    "negative_examples_per_epoch"
                ]
            )
            == EXPECTED_NEGATIVES_PER_EPOCH,
        ),
        (
            "4_to_1_labeled_count",
            int(
                row4[
                    "labeled_examples_per_epoch"
                ]
            )
            == EXPECTED_LABELED_EXAMPLES_PER_EPOCH,
        ),
        (
            "uniform_is_future_safe",
            not bool(
                uniform[
                    "uses_future_information"
                ]
            ),
        ),
        (
            "without_replacement_feasible",
            bool(
                row4[
                    "without_replacement_feasible_for_all_events"
                ]
            ),
        ),
        (
            "per_epoch_not_batch_order_dependent",
            not bool(
                per_epoch[
                    "depends_on_batch_order"
                ]
            ),
        ),
        (
            "epoch_seed_reconstructable",
            bool(
                epoch_rng[
                    "exact_epoch_reconstruction"
                ]
            ),
        ),
        (
            "future_positive_exclusion_remains_false",
            exclusion[
                "exclude_future_positive_pairs"
            ]
            is False,
        ),
        (
            "t60_training_leakage_remains_false",
            exclusion[
                "use_T60_labels_for_T1_T59_sampling"
            ]
            is False,
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
        "One or more 5.1.1d freeze checks failed",
    )

    # =========================================================================
    # Frozen contract
    # =========================================================================

    contract = {
        "phase": "5.1.1d",
        "title": (
            "Training Negative-Sampling Runtime Contract Freeze"
        ),
        "status": "FROZEN",

        "negative_samples_generated": False,
        "rng_instantiated": False,
        "model_instantiated": False,
        "optimizer_created": False,
        "training_performed": False,

        "paper_specified": {
            "training_negative_sampling_method": "random",
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": PAPER_BATCH_SIZE,
            "evaluation_negative_count": 99,
            "evaluation_metrics": [
                "HR@10",
                "NDCG@10",
            ],
        },

        "paper_unspecified_reproduction_choices": {
            "training_negative_positive_ratio": (
                NEGATIVES_PER_POSITIVE
            ),
            "training_distribution": (
                SAMPLING_DISTRIBUTION
            ),
            "replacement": (
                REPLACEMENT_SEMANTICS
            ),
            "resampling_cadence": (
                RESAMPLING_CADENCE
            ),
        },

        "inherited_semantics": {
            "candidate_universe_count": (
                EXPECTED_STARTUP_UNIVERSE
            ),
            "candidate_universe": (
                "all frozen Phase-3 Startup role nodes"
            ),
            "eligibility_rule": (
                "eligible iff first_positive_segment(investor,startup) "
                "> target_segment or pair is never positive"
            ),
            "exclude_prior_positive_pairs": True,
            "exclude_same_target_positive_pairs": True,
            "exclude_future_positive_pairs": False,
            "use_T60_labels_during_T1_T59_sampling": False,
        },

        "sampling_runtime": {
            "K": NEGATIVES_PER_POSITIVE,
            "distribution": "uniform",
            "without_replacement_within_focal_positive": True,
            "independent_across_positive_events": True,
            "same_negative_can_recur_across_different_positive_events": True,
            "regenerate_each_epoch": True,
            "epoch_indexing": "zero_based",
            "sampler_base_seed": SAMPLER_BASE_SEED,
            "seed_namespace": SEED_NAMESPACE,
            "seed_derivation": (
                "SHA256("
                "ITRS_PHASE5_TRAIN_NEGATIVE|42|epoch_index"
                "); first 8 digest bytes interpreted little-endian "
                "unsigned; modulo (2^63 - 1)"
            ),
            "dedicated_sampler_rng_required": True,
            "share_rng_with_model_initialization": False,
            "share_rng_with_dataloader_shuffle": False,
            "share_rng_with_other_stochastic_operations": False,
        },

        "per_epoch_workload": {
            "positive_examples": (
                EXPECTED_TRAINING_POSITIVES
            ),
            "negative_examples": (
                EXPECTED_NEGATIVES_PER_EPOCH
            ),
            "total_labeled_examples": (
                EXPECTED_LABELED_EXAMPLES_PER_EPOCH
            ),
            "batches_at_512": (
                EXPECTED_BATCHES_PER_EPOCH
            ),
            "negative_fraction": float(
                row4[
                    "negative_fraction"
                ]
            ),
        },

        "diagnostics_not_filters": {
            "future_positive_expected_draws_per_epoch_under_uniform": float(
                row4[
                    "uniform_reference_expected_future_positive_draws_per_epoch"
                ]
            ),
            "interpretation": (
                "A candidate becoming positive in a later segment "
                "does not invalidate its historical negative status "
                "and MUST NOT be removed using future labels."
            ),
        },

        "reference_seed_examples_hash_only": (
            seed_examples
        ),

        "still_unresolved_original_phase_5_handoff_decisions": [
            "training epoch count",
            "early stopping",
            "weight decay",
            (
                "evaluation candidate-generation "
                "runtime contract"
            ),
        ],

        "authoritative_input_hashes": {
            str(PHASE_5_1_1B_CONTRACT): (
                sha256_file(
                    PHASE_5_1_1B_CONTRACT
                )
            ),
            str(RATIO_AUDIT): (
                sha256_file(
                    RATIO_AUDIT
                )
            ),
            str(DISTRIBUTION_REGISTER): (
                sha256_file(
                    DISTRIBUTION_REGISTER
                )
            ),
            str(REPLACEMENT_REGISTER): (
                sha256_file(
                    REPLACEMENT_REGISTER
                )
            ),
            str(RESAMPLING_REGISTER): (
                sha256_file(
                    RESAMPLING_REGISTER
                )
            ),
            str(RNG_REGISTER): (
                sha256_file(
                    RNG_REGISTER
                )
            ),
            str(AUDIT_5_1_1C_MANIFEST): (
                sha256_file(
                    AUDIT_5_1_1C_MANIFEST
                )
            ),
        },
    }

    # =========================================================================
    # Write
    # =========================================================================

    banner("WRITE FROZEN PHASE-5.1.1d CONTRACT")

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

    banner("PHASE 5.1.1d FINAL STATUS")

    print(
        "Training candidate eligibility:       FROZEN (5.1.1b)"
    )

    print(
        "Training positive-pair exclusion:     FROZEN (5.1.1b)"
    )

    print(
        "Training negative:positive ratio:     FROZEN -> 4:1"
    )

    print(
        "Training sampling distribution:       FROZEN -> uniform"
    )

    print(
        "Training replacement semantics:       FROZEN -> without replacement"
    )

    print(
        "Training resampling cadence:          FROZEN -> once per epoch"
    )

    print(
        "Training sampler base seed:           FROZEN -> 42"
    )

    print(
        "Training sampler seed derivation:     FROZEN -> epoch-indexed SHA256"
    )

    print()
    print(
        "Negative rows generated:              0"
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

    print(
        "  4. evaluation candidate-generation runtime contract"
    )

    banner(
        "PHASE 5.1.1d COMPLETE / "
        "TRAINING NEGATIVE-SAMPLING CONTRACT FULLY FROZEN"
    )


if __name__ == "__main__":
    main()