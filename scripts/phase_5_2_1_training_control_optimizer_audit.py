"""
Phase 5.2.1 — Training-Control and Optimizer Audit

AUDIT ONLY.

This phase examines the final unresolved Phase-5 training decisions:

1. training epoch count;
2. early stopping;
3. weight decay.

It also audits implementation-equivalent optimizer and training-order
details that must be explicit before the first optimizer step.

THIS SCRIPT DOES NOT:
- generate training negatives;
- instantiate a training RNG;
- instantiate the ITRS model;
- instantiate an optimizer;
- modify the frozen evaluation candidates;
- train;
- evaluate a trained model;
- freeze any of the remaining three Phase-5 handoff decisions.

Paper-specified
---------------
- optimizer: Adam
- learning rate: 0.001
- mini-batch size: 512
- validation set used to obtain best parameters
- evaluation metrics: HR@10 and NDCG@10

Paper-unspecified
-----------------
- epoch count
- early stopping
- weight decay
- validation checkpoint-selection metric/tie break

External methodological precedent
---------------------------------
The official NCF implementation, whose model family is explicitly used
as the basis of ITRS's recommendation component, commonly runs 20
epochs with Adam and zero explicit regularization settings.

This is supporting evidence only, NOT an ITRS paper parameter.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import platform
import sys
from pathlib import Path

import pandas as pd
import torch


# =============================================================================
# Paths
# =============================================================================

TRAIN_NEGATIVE_CONTRACT = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_1_1d_training_negative_sampling_runtime_contract.json"
)

EVAL_CONTRACT = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_1_2b_t60_evaluation_candidate_runtime_contract.json"
)

EVAL_GENERATION_MANIFEST = Path(
    "data/experimental/phase_5/model_ready/evaluation/"
    "phase_5_1_2c_generation_manifest.json"
)

EVAL_NEGATIVE_MATRIX = Path(
    "data/experimental/phase_5/model_ready/evaluation/"
    "t60_evaluation_negative_node_indices.npy"
)

EVAL_CASE_MANIFEST = Path(
    "data/experimental/phase_5/model_ready/evaluation/"
    "t60_evaluation_case_manifest.parquet"
)

OUT_DIR = Path(
    "data/experimental/phase_5/audits/phase_5_2_1"
)

EPOCH_AUDIT_PATH = (
    OUT_DIR
    / "training_epoch_budget_workload_audit.csv"
)

TRAINING_CONTROL_REGISTER_PATH = (
    OUT_DIR
    / "training_control_option_register.csv"
)

CHECKPOINT_REGISTER_PATH = (
    OUT_DIR
    / "validation_checkpoint_selection_option_register.csv"
)

WEIGHT_DECAY_REGISTER_PATH = (
    OUT_DIR
    / "weight_decay_option_register.csv"
)

ADAM_DEFAULTS_PATH = (
    OUT_DIR
    / "torch_adam_runtime_defaults.csv"
)

TRAIN_ORDER_REGISTER_PATH = (
    OUT_DIR
    / "training_order_runtime_option_register.csv"
)

AUDIT_MANIFEST_PATH = (
    OUT_DIR
    / "phase_5_2_1_audit_manifest.json"
)


# =============================================================================
# Frozen values inherited from prior phases
# =============================================================================

EXPECTED = {
    "training_positive_events": 1_073_249,
    "training_negatives_per_positive": 4,

    "training_negatives_per_epoch": 4_292_996,
    "training_labeled_examples_per_epoch": 5_366_245,
    "training_batches_per_epoch": 10_481,

    "batch_size": 512,
    "learning_rate": 0.001,

    "evaluation_cases": 22_515,
    "validation_cases": 2_251,
    "test_cases": 20_264,
    "evaluation_negatives_per_case": 99,

    # Correct authoritative arithmetic:
    # 22,515 * 99
    "evaluation_negative_slots": 2_228_985,

    # 22,515 * 100
    "evaluation_total_candidate_slots": 2_251_500,

    "validation_candidate_slots": 225_100,
    "test_candidate_slots": 2_026_400,
}


# Diagnostic probes only.
EPOCH_PROBES = (
    5,
    10,
    20,
    50,
)


# Candidate weight-decay values for methodological comparison only.
# No tuning or optimizer creation occurs.
WEIGHT_DECAY_PROBES = (
    0.0,
    1e-6,
    1e-5,
    1e-4,
)


TRAIN_ORDER_BASE_SEED = 42
TRAIN_ORDER_NAMESPACE = (
    "ITRS_PHASE5_TRAIN_ORDER"
)


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


def derive_train_order_seed_without_rng(
    epoch_index: int,
) -> int:
    """
    Candidate deterministic training-order seed.

    HASHING ONLY.
    No RNG is instantiated.

    material:
        ITRS_PHASE5_TRAIN_ORDER|42|<epoch_index>
    """

    require(
        epoch_index >= 0,
        "epoch_index must be nonnegative",
    )

    material = (
        f"{TRAIN_ORDER_NAMESPACE}|"
        f"{TRAIN_ORDER_BASE_SEED}|"
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
        % (
            (2 ** 63)
            - 1
        )
    )


def normalize_default(value):
    """
    Convert an inspect.signature default to a CSV/JSON-friendly value.
    """

    if value is inspect._empty:
        return "<REQUIRED>"

    return repr(
        value
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.2.1 — "
        "TRAINING-CONTROL AND OPTIMIZER AUDIT"
    )

    print(
        "Training negatives generated:         NO"
    )

    print(
        "Training RNG instantiated:            NO"
    )

    print(
        "Model instantiated:                   NO"
    )

    print(
        "Optimizer instantiated:               NO"
    )

    print(
        "Training performed:                   NO"
    )

    print(
        "Epoch count frozen:                   NO"
    )

    print(
        "Early stopping frozen:                NO"
    )

    print(
        "Weight decay frozen:                  NO"
    )

    # =========================================================================
    # Authoritative inputs
    # =========================================================================

    banner(
        "AUTHORITATIVE INPUT EXISTENCE"
    )

    required_paths = (
        TRAIN_NEGATIVE_CONTRACT,
        EVAL_CONTRACT,
        EVAL_GENERATION_MANIFEST,
        EVAL_NEGATIVE_MATRIX,
        EVAL_CASE_MANIFEST,
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
    # Recheck training-negative contract
    # =========================================================================

    banner(
        "PHASE 5.1.1d TRAINING-NEGATIVE CONTRACT RECHECK"
    )

    train_contract = json.loads(
        TRAIN_NEGATIVE_CONTRACT.read_text(
            encoding="utf-8"
        )
    )

    require(
        train_contract[
            "phase"
        ]
        == "5.1.1d",
        "Unexpected training-negative contract phase",
    )

    require(
        train_contract[
            "status"
        ]
        == "FROZEN",
        "Training-negative contract is not frozen",
    )

    require(
        int(
            train_contract[
                "paper_unspecified_reproduction_choices"
            ][
                "training_negative_positive_ratio"
            ]
        )
        == EXPECTED[
            "training_negatives_per_positive"
        ],
        "Frozen training K drift",
    )

    per_epoch = train_contract[
        "per_epoch_workload"
    ]

    require(
        int(
            per_epoch[
                "positive_examples"
            ]
        )
        == EXPECTED[
            "training_positive_events"
        ],
        "Training-positive count drift",
    )

    require(
        int(
            per_epoch[
                "negative_examples"
            ]
        )
        == EXPECTED[
            "training_negatives_per_epoch"
        ],
        "Training-negative count drift",
    )

    require(
        int(
            per_epoch[
                "total_labeled_examples"
            ]
        )
        == EXPECTED[
            "training_labeled_examples_per_epoch"
        ],
        "Training labeled-example count drift",
    )

    require(
        int(
            per_epoch[
                "batches_at_512"
            ]
        )
        == EXPECTED[
            "training_batches_per_epoch"
        ],
        "Training batch-count drift",
    )

    print(
        f"Training positives / epoch:         "
        f"{EXPECTED['training_positive_events']:,}  PASS"
    )

    print(
        f"Training negatives / epoch:         "
        f"{EXPECTED['training_negatives_per_epoch']:,}  PASS"
    )

    print(
        f"Labeled examples / epoch:           "
        f"{EXPECTED['training_labeled_examples_per_epoch']:,}  PASS"
    )

    print(
        f"Batches / epoch @512:               "
        f"{EXPECTED['training_batches_per_epoch']:,}  PASS"
    )

    # =========================================================================
    # Recheck frozen/generated evaluation artifact
    # =========================================================================

    banner(
        "PHASE 5.1.2b/c EVALUATION CONTRACT RECHECK"
    )

    eval_contract = json.loads(
        EVAL_CONTRACT.read_text(
            encoding="utf-8"
        )
    )

    eval_manifest = json.loads(
        EVAL_GENERATION_MANIFEST.read_text(
            encoding="utf-8"
        )
    )

    require(
        eval_contract[
            "status"
        ]
        == "FROZEN",
        "Evaluation runtime contract not frozen",
    )

    require(
        eval_manifest[
            "status"
        ]
        == "GENERATED_AND_AUDITED",
        "Evaluation candidates not generated/audited",
    )

    require(
        eval_manifest[
            "evaluation_negative_samples_generated"
        ]
        is True,
        "Evaluation-negative artifact not generated",
    )

    require(
        eval_manifest[
            "training_performed"
        ]
        is False,
        "Training unexpectedly occurred",
    )

    candidate_contract = eval_manifest[
        "candidate_contract"
    ]

    require(
        int(
            candidate_contract[
                "negative_slots"
            ]
        )
        == EXPECTED[
            "evaluation_negative_slots"
        ],
        (
            "Evaluation negative-slot count drift. "
            "Expected 2,228,985 = 22,515 * 99."
        ),
    )

    require(
        int(
            candidate_contract[
                "total_candidate_slots"
            ]
        )
        == EXPECTED[
            "evaluation_total_candidate_slots"
        ],
        "Evaluation total-candidate count drift",
    )

    print(
        f"Evaluation cases:                   "
        f"{EXPECTED['evaluation_cases']:,}  PASS"
    )

    print(
        f"Evaluation negative slots:          "
        f"{EXPECTED['evaluation_negative_slots']:,}  PASS"
    )

    print(
        f"Evaluation total candidate slots:   "
        f"{EXPECTED['evaluation_total_candidate_slots']:,}  PASS"
    )

    print(
        "Evaluation candidates immutable:      YES  PASS"
    )

    # =========================================================================
    # Paper-specified optimizer facts
    # =========================================================================

    banner(
        "PAPER-SPECIFIED OPTIMIZER FACTS"
    )

    print(
        "Optimizer:                           Adam"
    )

    print(
        f"Learning rate:                       "
        f"{EXPECTED['learning_rate']}"
    )

    print(
        f"Mini-batch size:                     "
        f"{EXPECTED['batch_size']}"
    )

    print(
        "Validation used for parameter choice: YES"
    )

    print(
        "Paper reports epoch count:           NO"
    )

    print(
        "Paper reports early stopping:        NO"
    )

    print(
        "Paper reports weight decay:          NO"
    )

    # =========================================================================
    # Epoch budget workload
    # =========================================================================

    banner(
        "EPOCH-BUDGET WORKLOAD AUDIT"
    )

    epoch_rows = []

    for epochs in EPOCH_PROBES:

        training_examples = (
            EXPECTED[
                "training_labeled_examples_per_epoch"
            ]
            * epochs
        )

        positive_presentations = (
            EXPECTED[
                "training_positive_events"
            ]
            * epochs
        )

        negative_presentations = (
            EXPECTED[
                "training_negatives_per_epoch"
            ]
            * epochs
        )

        optimizer_batches = (
            EXPECTED[
                "training_batches_per_epoch"
            ]
            * epochs
        )

        validation_candidate_scores = (
            EXPECTED[
                "validation_candidate_slots"
            ]
            * epochs
        )

        # Test should be evaluated once AFTER checkpoint selection.
        test_candidate_scores = (
            EXPECTED[
                "test_candidate_slots"
            ]
        )

        epoch_rows.append(
            {
                "epochs": epochs,
                "classification": (
                    "AUDIT_PROBE_NOT_FROZEN"
                ),
                "positive_presentations": (
                    positive_presentations
                ),
                "negative_presentations": (
                    negative_presentations
                ),
                "labeled_training_presentations": (
                    training_examples
                ),
                "optimizer_batches": (
                    optimizer_batches
                ),
                "validation_candidate_scores_if_checked_each_epoch": (
                    validation_candidate_scores
                ),
                "test_candidate_scores_if_tested_once": (
                    test_candidate_scores
                ),
                "total_ranking_candidate_scores_validation_plus_one_test": (
                    validation_candidate_scores
                    + test_candidate_scores
                ),
                "workload_multiplier_vs_20_epochs": (
                    epochs
                    / 20.0
                ),
                "external_NCF_20_epoch_reference": (
                    epochs == 20
                ),
            }
        )

    epoch_df = pd.DataFrame(
        epoch_rows
    )

    print(
        epoch_df[
            [
                "epochs",
                "labeled_training_presentations",
                "optimizer_batches",
                (
                    "validation_candidate_scores_"
                    "if_checked_each_epoch"
                ),
                "external_NCF_20_epoch_reference",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # Training-control options
    # =========================================================================

    banner(
        "TRAINING-CONTROL OPTION REGISTER"
    )

    training_control_df = pd.DataFrame(
        [
            {
                "option": (
                    "fixed_20_epochs_no_early_stopping"
                ),
                "maximum_epochs": 20,
                "early_stopping": False,
                "validation_each_epoch": True,
                "best_validation_checkpoint_retained": True,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "external_precedent": (
                    "Official NCF reference implementation uses "
                    "20 epochs in standard commands."
                ),
                "main_advantage": (
                    "No arbitrary patience/min_delta; exact fixed "
                    "optimization budget; validation still selects "
                    "best checkpoint."
                ),
                "main_risk": (
                    "ITRS itself does not report 20 epochs."
                ),
                "audit_assessment": (
                    "STRONG_CANDIDATE"
                ),
            },
            {
                "option": (
                    "fixed_20_epochs_final_epoch_only"
                ),
                "maximum_epochs": 20,
                "early_stopping": False,
                "validation_each_epoch": False,
                "best_validation_checkpoint_retained": False,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "external_precedent": (
                    "20-epoch NCF precedent only"
                ),
                "main_advantage": (
                    "Simplest fixed-budget training."
                ),
                "main_risk": (
                    "Underuses the validation set that ITRS "
                    "explicitly says was used to obtain best parameters."
                ),
                "audit_assessment": (
                    "NOT_PREFERRED"
                ),
            },
            {
                "option": (
                    "20_epoch_cap_early_stop_patience_3"
                ),
                "maximum_epochs": 20,
                "early_stopping": True,
                "validation_each_epoch": True,
                "best_validation_checkpoint_retained": True,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "external_precedent": None,
                "main_advantage": (
                    "Can reduce unnecessary training."
                ),
                "main_risk": (
                    "Patience=3 and stopping criterion are not "
                    "reported by ITRS."
                ),
                "audit_assessment": (
                    "NOT_PREFERRED_ARBITRARY_PATIENCE"
                ),
            },
            {
                "option": (
                    "20_epoch_cap_early_stop_patience_5"
                ),
                "maximum_epochs": 20,
                "early_stopping": True,
                "validation_each_epoch": True,
                "best_validation_checkpoint_retained": True,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "external_precedent": None,
                "main_advantage": (
                    "Can reduce unnecessary training."
                ),
                "main_risk": (
                    "Patience=5 and stopping criterion are not "
                    "reported by ITRS."
                ),
                "audit_assessment": (
                    "NOT_PREFERRED_ARBITRARY_PATIENCE"
                ),
            },
            {
                "option": (
                    "fixed_50_epochs_best_validation_checkpoint"
                ),
                "maximum_epochs": 50,
                "early_stopping": False,
                "validation_each_epoch": True,
                "best_validation_checkpoint_retained": True,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "external_precedent": None,
                "main_advantage": (
                    "Larger optimization budget."
                ),
                "main_risk": (
                    "2.5x the 20-epoch workload with no direct "
                    "ITRS or NCF reference justification."
                ),
                "audit_assessment": (
                    "NOT_PREFERRED"
                ),
            },
        ]
    )

    print(
        training_control_df[
            [
                "option",
                "maximum_epochs",
                "early_stopping",
                "best_validation_checkpoint_retained",
                "audit_assessment",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # Checkpoint-selection options
    # =========================================================================

    banner(
        "VALIDATION CHECKPOINT-SELECTION OPTION REGISTER"
    )

    checkpoint_df = pd.DataFrame(
        [
            {
                "option": (
                    "max_validation_NDCG10_then_HR10_then_earliest_epoch"
                ),
                "uses_validation": True,
                "primary_metric": "NDCG@10",
                "secondary_metric": "HR@10",
                "final_tie_break": "earliest_epoch",
                "test_labels_used": False,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "main_advantage": (
                    "Uses the paper's ranking metrics; NDCG captures "
                    "rank position, HR resolves remaining ties."
                ),
                "audit_assessment": (
                    "STRONG_CANDIDATE"
                ),
            },
            {
                "option": (
                    "max_validation_HR10_then_NDCG10_then_earliest_epoch"
                ),
                "uses_validation": True,
                "primary_metric": "HR@10",
                "secondary_metric": "NDCG@10",
                "final_tie_break": "earliest_epoch",
                "test_labels_used": False,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "main_advantage": (
                    "Directly prioritizes hit rate."
                ),
                "audit_assessment": (
                    "VALID_BUT_LESS_RANK_SENSITIVE"
                ),
            },
            {
                "option": (
                    "final_epoch_only"
                ),
                "uses_validation": False,
                "primary_metric": None,
                "secondary_metric": None,
                "final_tie_break": None,
                "test_labels_used": False,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "main_advantage": (
                    "No checkpoint-selection rule required."
                ),
                "audit_assessment": (
                    "NOT_PREFERRED_VALIDATION_UNDERUSED"
                ),
            },
        ]
    )

    print(
        checkpoint_df[
            [
                "option",
                "primary_metric",
                "secondary_metric",
                "audit_assessment",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # Weight decay options
    # =========================================================================

    banner(
        "WEIGHT-DECAY OPTION AUDIT"
    )

    weight_rows = []

    for weight_decay in WEIGHT_DECAY_PROBES:

        if weight_decay == 0.0:

            assessment = (
                "STRONG_CANDIDATE_MINIMAL_ASSUMPTION"
            )

            rationale = (
                "ITRS reports Adam but no weight decay. "
                "Zero introduces no additional optimizer regularizer; "
                "official NCF examples also use zero explicit "
                "regularization settings."
            )

        else:

            assessment = (
                "NOT_PREFERRED_UNREPORTED_REGULARIZATION"
            )

            rationale = (
                "Adds nonzero L2-style optimizer regularization "
                "not reported by ITRS; no evidence selects this value."
            )

        weight_rows.append(
            {
                "weight_decay": weight_decay,
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_CANDIDATE"
                ),
                "introduces_nonzero_optimizer_regularization": (
                    weight_decay != 0.0
                ),
                "audit_assessment": (
                    assessment
                ),
                "rationale": (
                    rationale
                ),
            }
        )

    weight_decay_df = pd.DataFrame(
        weight_rows
    )

    print(
        weight_decay_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Actual torch.optim.Adam defaults — inspection only
    # =========================================================================

    banner(
        "PYTORCH ADAM RUNTIME DEFAULT INSPECTION — NO OPTIMIZER CREATED"
    )

    adam_signature = inspect.signature(
        torch.optim.Adam.__init__
    )

    parameters = (
        adam_signature.parameters
    )

    parameter_names = [
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "foreach",
        "maximize",
        "capturable",
        "differentiable",
        "fused",
    ]

    adam_rows = []

    for name in parameter_names:

        if name not in parameters:

            adam_rows.append(
                {
                    "parameter": name,
                    "available_in_runtime": False,
                    "runtime_default": None,
                    "paper_specified": (
                        name == "lr"
                    ),
                    "future_classification": (
                        "NOT_APPLICABLE_RUNTIME"
                    ),
                }
            )

            continue

        default = parameters[
            name
        ].default

        adam_rows.append(
            {
                "parameter": name,
                "available_in_runtime": True,
                "runtime_default": (
                    normalize_default(
                        default
                    )
                ),
                "paper_specified": (
                    name == "lr"
                ),
                "future_classification": (
                    "PAPER_SPECIFIED"
                    if name == "lr"
                    else (
                        "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                        if name == "weight_decay"
                        else "IMPLEMENTATION_EQUIVALENT_CHOICE_CANDIDATE"
                    )
                ),
            }
        )

    adam_df = pd.DataFrame(
        adam_rows
    )

    print(
        f"torch version: {torch.__version__}"
    )

    print()

    print(
        adam_df.to_string(
            index=False
        )
    )

    # The paper specifies lr=0.001 rather than relying on
    # whatever the PyTorch default happens to be.
    print()
    print(
        "Paper override:"
    )
    print(
        "  lr = 0.001"
    )

    print(
        "Weight decay remains NOT frozen in this audit."
    )

    # =========================================================================
    # Training-order runtime options
    # =========================================================================

    banner(
        "TRAINING-ORDER RUNTIME OPTION AUDIT"
    )

    train_order_df = pd.DataFrame(
        [
            {
                "option": (
                    "deterministic_shuffle_all_labeled_examples_each_epoch"
                ),
                "shuffle": True,
                "epoch_specific": True,
                "dedicated_rng_stream": True,
                "depends_on_negative_sampler_rng_state": False,
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE_CANDIDATE"
                ),
                "main_advantage": (
                    "Avoids fixed positive/negative/temporal ordering "
                    "while remaining exactly reproducible."
                ),
                "audit_assessment": (
                    "STRONG_CANDIDATE"
                ),
            },
            {
                "option": (
                    "fixed_serialized_training_order"
                ),
                "shuffle": False,
                "epoch_specific": False,
                "dedicated_rng_stream": False,
                "depends_on_negative_sampler_rng_state": False,
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE_CANDIDATE"
                ),
                "main_advantage": (
                    "Simplest ordering."
                ),
                "main_risk": (
                    "Optimizer trajectory depends on arbitrary "
                    "serialization/grouping order."
                ),
                "audit_assessment": (
                    "NOT_PREFERRED"
                ),
            },
            {
                "option": (
                    "shuffle_using_shared_global_rng"
                ),
                "shuffle": True,
                "epoch_specific": True,
                "dedicated_rng_stream": False,
                "depends_on_negative_sampler_rng_state": True,
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE_CANDIDATE"
                ),
                "main_advantage": (
                    "Simple."
                ),
                "main_risk": (
                    "Batch order changes when unrelated RNG "
                    "consumption changes."
                ),
                "audit_assessment": (
                    "NOT_PREFERRED"
                ),
            },
        ]
    )

    print(
        train_order_df[
            [
                "option",
                "shuffle",
                "dedicated_rng_stream",
                "audit_assessment",
            ]
        ].to_string(
            index=False
        )
    )

    # Hash-only examples.
    print()
    print(
        "Candidate epoch-indexed training-order seeds "
        "(hashing only; no RNG):"
    )

    for epoch in range(5):

        print(
            f"  epoch {epoch}: "
            f"{derive_train_order_seed_without_rng(epoch)}"
        )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL AUDIT INVARIANTS"
    )

    require(
        EXPECTED[
            "training_positive_events"
        ]
        * (
            1
            + EXPECTED[
                "training_negatives_per_positive"
            ]
        )
        == EXPECTED[
            "training_labeled_examples_per_epoch"
        ],
        "Training per-epoch arithmetic failed",
    )

    require(
        math.ceil(
            EXPECTED[
                "training_labeled_examples_per_epoch"
            ]
            / EXPECTED[
                "batch_size"
            ]
        )
        == EXPECTED[
            "training_batches_per_epoch"
        ],
        "Batch arithmetic failed",
    )

    require(
        EXPECTED[
            "evaluation_cases"
        ]
        * EXPECTED[
            "evaluation_negatives_per_case"
        ]
        == EXPECTED[
            "evaluation_negative_slots"
        ],
        "Evaluation negative-slot arithmetic failed",
    )

    require(
        EXPECTED[
            "validation_cases"
        ]
        * 100
        == EXPECTED[
            "validation_candidate_slots"
        ],
        "Validation candidate-slot arithmetic failed",
    )

    require(
        EXPECTED[
            "test_cases"
        ]
        * 100
        == EXPECTED[
            "test_candidate_slots"
        ],
        "Test candidate-slot arithmetic failed",
    )

    print(
        "Training-negative contract frozen:        PASS"
    )

    print(
        "Evaluation candidate contract frozen:     PASS"
    )

    print(
        "Evaluation candidates generated/audited:  PASS"
    )

    print(
        "Per-epoch workload arithmetic:            PASS"
    )

    print(
        "Evaluation slot arithmetic:               PASS"
    )

    print(
        "Adam runtime defaults inspected only:    PASS"
    )

    print(
        "Training negatives generated:             0"
    )

    print(
        "Training RNG instantiated:                NO"
    )

    print(
        "Model instantiated:                       NO"
    )

    print(
        "Optimizer instantiated:                   NO"
    )

    print(
        "Training performed:                       NO"
    )

    print(
        "Remaining decisions frozen:               NO"
    )

    # =========================================================================
    # Write audit artifacts
    # =========================================================================

    banner(
        "WRITE AUDIT-ONLY OUTPUTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    epoch_df.to_csv(
        EPOCH_AUDIT_PATH,
        index=False,
    )

    training_control_df.to_csv(
        TRAINING_CONTROL_REGISTER_PATH,
        index=False,
    )

    checkpoint_df.to_csv(
        CHECKPOINT_REGISTER_PATH,
        index=False,
    )

    weight_decay_df.to_csv(
        WEIGHT_DECAY_REGISTER_PATH,
        index=False,
    )

    adam_df.to_csv(
        ADAM_DEFAULTS_PATH,
        index=False,
    )

    train_order_df.to_csv(
        TRAIN_ORDER_REGISTER_PATH,
        index=False,
    )

    manifest = {
        "phase": "5.2.1",
        "title": (
            "Training-Control and Optimizer Audit"
        ),
        "status": (
            "AUDIT_COMPLETE_NOT_FROZEN"
        ),

        "training_negative_samples_generated": False,
        "training_rng_instantiated": False,
        "model_instantiated": False,
        "optimizer_instantiated": False,
        "training_performed": False,

        "paper_specified": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "batch_size": 512,
            "validation_used_for_best_parameters": True,
            "evaluation_metrics": [
                "HR@10",
                "NDCG@10",
            ],
        },

        "paper_unspecified": [
            "training epoch count",
            "early stopping",
            "weight decay",
            "checkpoint-selection metric",
        ],

        "external_NCF_reference": {
            "role": (
                "Methodological precedent only; "
                "not an ITRS specification."
            ),
            "standard_epoch_example": 20,
            "standard_optimizer_example": "Adam",
            "standard_learning_rate_example": 0.001,
            "standard_negative_ratio_example": 4,
            "standard_regularization_example": (
                "zero explicit regularization in reference commands"
            ),
        },

        "leading_candidates_not_frozen": {
            "training_epoch_count": 20,
            "early_stopping": False,
            "evaluate_validation_each_epoch": True,
            "checkpoint_selection": (
                "max NDCG@10, then HR@10, "
                "then earliest epoch"
            ),
            "weight_decay": 0.0,
            "training_order": (
                "deterministically shuffle all labeled examples "
                "once per epoch with dedicated epoch-indexed RNG"
            ),
        },

        "authoritative_workload": {
            "training_positive_examples_per_epoch": (
                EXPECTED[
                    "training_positive_events"
                ]
            ),
            "training_negative_examples_per_epoch": (
                EXPECTED[
                    "training_negatives_per_epoch"
                ]
            ),
            "training_labeled_examples_per_epoch": (
                EXPECTED[
                    "training_labeled_examples_per_epoch"
                ]
            ),
            "training_batches_per_epoch": (
                EXPECTED[
                    "training_batches_per_epoch"
                ]
            ),
            "evaluation_negative_slots": (
                EXPECTED[
                    "evaluation_negative_slots"
                ]
            ),
            "evaluation_total_candidate_slots": (
                EXPECTED[
                    "evaluation_total_candidate_slots"
                ]
            ),
        },

        "torch_runtime": {
            "torch_version": (
                torch.__version__
            ),
            "adam_signature": str(
                adam_signature
            ),
        },

        "still_unresolved_original_phase_5_handoff_decisions": [
            "training epoch count",
            "early stopping",
            "weight decay",
        ],

        "authoritative_input_hashes": {
            str(
                TRAIN_NEGATIVE_CONTRACT
            ): sha256_file(
                TRAIN_NEGATIVE_CONTRACT
            ),
            str(
                EVAL_CONTRACT
            ): sha256_file(
                EVAL_CONTRACT
            ),
            str(
                EVAL_GENERATION_MANIFEST
            ): sha256_file(
                EVAL_GENERATION_MANIFEST
            ),
            str(
                EVAL_NEGATIVE_MATRIX
            ): sha256_file(
                EVAL_NEGATIVE_MATRIX
            ),
            str(
                EVAL_CASE_MANIFEST
            ): sha256_file(
                EVAL_CASE_MANIFEST
            ),
        },

        "outputs": [
            str(
                EPOCH_AUDIT_PATH
            ),
            str(
                TRAINING_CONTROL_REGISTER_PATH
            ),
            str(
                CHECKPOINT_REGISTER_PATH
            ),
            str(
                WEIGHT_DECAY_REGISTER_PATH
            ),
            str(
                ADAM_DEFAULTS_PATH
            ),
            str(
                TRAIN_ORDER_REGISTER_PATH
            ),
        ],

        "environment": {
            "python": (
                sys.version
            ),
            "torch": (
                torch.__version__
            ),
            "pandas": (
                pd.__version__
            ),
            "platform": (
                platform.platform()
            ),
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
        EPOCH_AUDIT_PATH,
        TRAINING_CONTROL_REGISTER_PATH,
        CHECKPOINT_REGISTER_PATH,
        WEIGHT_DECAY_REGISTER_PATH,
        ADAM_DEFAULTS_PATH,
        TRAIN_ORDER_REGISTER_PATH,
        AUDIT_MANIFEST_PATH,
    ):

        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Decision-facing summary
    # =========================================================================

    banner(
        "DECISION-FACING SUMMARY — NOTHING NEW FROZEN"
    )

    epoch20 = epoch_df.loc[
        epoch_df[
            "epochs"
        ]
        == 20
    ].iloc[0]

    print(
        "Leading epoch candidate:"
    )

    print(
        "  fixed epochs:                    20"
    )

    print(
        f"  labeled presentations:           "
        f"{int(epoch20['labeled_training_presentations']):,}"
    )

    print(
        f"  optimizer batches:               "
        f"{int(epoch20['optimizer_batches']):,}"
    )

    print(
        f"  validation candidate scores:     "
        f"{int(epoch20['validation_candidate_scores_if_checked_each_epoch']):,}"
    )

    print()

    print(
        "Leading early-stopping candidate:"
    )

    print(
        "  early stopping:                  disabled"
    )

    print(
        "  validation evaluation:           after every epoch"
    )

    print(
        "  retain best checkpoint:          yes"
    )

    print(
        "  proposed ranking:                "
        "NDCG@10 -> HR@10 -> earliest epoch"
    )

    print()

    print(
        "Leading weight-decay candidate:"
    )

    print(
        "  weight_decay:                    0.0"
    )

    print()

    print(
        "IMPORTANT: These remain candidates only."
    )

    print(
        "No optimizer or training step has been executed."
    )

    banner(
        "PHASE 5.2.1 COMPLETE"
    )

    print(
        "AUDIT COMPLETE — "
        "FINAL THREE HANDOFF DECISIONS STILL NOT FROZEN"
    )


if __name__ == "__main__":
    main()