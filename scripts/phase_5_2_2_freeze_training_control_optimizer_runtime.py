"""
Phase 5.2.2 — Final Training-Control and Optimizer Runtime Freeze

This phase closes the LAST THREE original Phase-5 handoff decisions:

1. training epoch count;
2. early stopping;
3. weight decay.

It also freezes implementation-equivalent runtime details that must be
explicit before optimizer step #1.

NO TRAINING OCCURS HERE.

Frozen
------
PAPER_SPECIFIED:
- optimizer = Adam
- learning_rate = 0.001
- batch_size = 512
- validation is used for best-parameter/model selection

PAPER_UNSPECIFIED_REPRODUCTION_CHOICE:
- fixed epoch count = 20
- early stopping = disabled
- weight_decay = 0.0
- validation checkpoint rule:
      NDCG@10 descending
      HR@10 descending
      epoch ascending
- learning-rate scheduler = none

IMPLEMENTATION_EQUIVALENT_CHOICE:
- validation after every completed epoch
- final partial training batch retained (drop_last=False)
- deterministic per-epoch full-example shuffle
- dedicated training-order PCG64 RNG
- training-negative sampler BitGenerator = PCG64
- explicit single-tensor Adam implementation path
  (foreach=False, fused=False)

This script DOES NOT:
- generate training negatives;
- instantiate a training RNG;
- instantiate the ITRS model;
- instantiate Adam;
- perform forward/backward passes;
- modify evaluation candidates;
- compute HR/NDCG from a trained model.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd
import torch


# =============================================================================
# Inputs
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

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/phase_5_2_1"
)

EPOCH_AUDIT = (
    AUDIT_DIR
    / "training_epoch_budget_workload_audit.csv"
)

TRAINING_CONTROL_REGISTER = (
    AUDIT_DIR
    / "training_control_option_register.csv"
)

CHECKPOINT_REGISTER = (
    AUDIT_DIR
    / "validation_checkpoint_selection_option_register.csv"
)

WEIGHT_DECAY_REGISTER = (
    AUDIT_DIR
    / "weight_decay_option_register.csv"
)

ADAM_DEFAULTS = (
    AUDIT_DIR
    / "torch_adam_runtime_defaults.csv"
)

TRAIN_ORDER_REGISTER = (
    AUDIT_DIR
    / "training_order_runtime_option_register.csv"
)

AUDIT_MANIFEST = (
    AUDIT_DIR
    / "phase_5_2_1_audit_manifest.json"
)


# =============================================================================
# Outputs
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

CONTRACT_PATH = (
    OUT_DIR
    / "phase_5_2_2_training_control_optimizer_runtime_contract.json"
)

DECISION_REGISTER_PATH = (
    OUT_DIR
    / "phase_5_2_2_training_control_optimizer_decision_register.csv"
)

FREEZE_AUDIT_PATH = (
    OUT_DIR
    / "phase_5_2_2_training_control_optimizer_freeze_audit.csv"
)


# =============================================================================
# Frozen choices
# =============================================================================

FIXED_EPOCHS = 20

EARLY_STOPPING = False

VALIDATE_AFTER_EVERY_EPOCH = True

CHECKPOINT_PRIMARY_METRIC = "NDCG@10"
CHECKPOINT_SECONDARY_METRIC = "HR@10"
CHECKPOINT_FINAL_TIEBREAK = "earliest_epoch"

OPTIMIZER_NAME = "Adam"

LEARNING_RATE = 0.001

BATCH_SIZE = 512

WEIGHT_DECAY = 0.0

ADAM_BETAS = (
    0.9,
    0.999,
)

ADAM_EPS = 1e-8

ADAM_AMSGRAD = False
ADAM_MAXIMIZE = False
ADAM_CAPTURABLE = False
ADAM_DIFFERENTIABLE = False

# Explicit implementation path instead of runtime-dependent None.
ADAM_FOREACH = False
ADAM_FUSED = False

LEARNING_RATE_SCHEDULER = None

DROP_LAST = False

TRAIN_ORDER_NAMESPACE = (
    "ITRS_PHASE5_TRAIN_ORDER"
)

TRAIN_ORDER_BASE_SEED = 42

TRAIN_ORDER_BIT_GENERATOR = (
    "numpy.random.PCG64"
)

TRAIN_NEGATIVE_BIT_GENERATOR = (
    "numpy.random.PCG64"
)


# =============================================================================
# Expected frozen workload
# =============================================================================

EXPECTED = {
    "training_positive_examples_per_epoch": 1_073_249,
    "training_negative_examples_per_epoch": 4_292_996,
    "training_examples_per_epoch": 5_366_245,
    "batches_per_epoch": 10_481,

    "full_batches_per_epoch": 10_480,
    "final_batch_size": 485,

    "epochs": 20,

    "total_training_presentations": 107_324_900,
    "total_optimizer_batches": 209_620,

    "validation_cases": 2_251,
    "test_cases": 20_264,

    "validation_candidates_per_epoch": 225_100,
    "validation_candidate_scores_20_epochs": 4_502_000,

    "evaluation_negative_slots": 2_228_985,
    "evaluation_total_candidate_slots": 2_251_500,
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


def derive_train_order_seed_without_rng(
    epoch_index: int,
) -> int:
    """
    Frozen deterministic training-order seed derivation.

    HASHING ONLY.

    No RNG is instantiated by this function.
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


def get_runtime_default(
    parameter_name: str,
):
    """
    Inspect torch.optim.Adam.__init__ WITHOUT instantiating Adam.
    """

    signature = inspect.signature(
        torch.optim.Adam.__init__
    )

    require(
        parameter_name
        in signature.parameters,
        (
            "torch.optim.Adam runtime missing "
            f"parameter: {parameter_name}"
        ),
    )

    return (
        signature.parameters[
            parameter_name
        ].default
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.2.2 — "
        "FINAL TRAINING-CONTROL AND OPTIMIZER RUNTIME FREEZE"
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
        "HR/NDCG computed from trained model:  NO"
    )

    # =========================================================================
    # Inputs
    # =========================================================================

    banner(
        "AUTHORITATIVE INPUT EXISTENCE"
    )

    required_paths = (
        TRAIN_NEGATIVE_CONTRACT,
        EVAL_CONTRACT,
        EVAL_GENERATION_MANIFEST,
        EPOCH_AUDIT,
        TRAINING_CONTROL_REGISTER,
        CHECKPOINT_REGISTER,
        WEIGHT_DECAY_REGISTER,
        ADAM_DEFAULTS,
        TRAIN_ORDER_REGISTER,
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
    # Phase-5.1 contracts remain frozen
    # =========================================================================

    banner(
        "PHASE 5.1 CONTRACT RECHECK"
    )

    train_negative_contract = json.loads(
        TRAIN_NEGATIVE_CONTRACT.read_text(
            encoding="utf-8"
        )
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
        train_negative_contract[
            "status"
        ]
        == "FROZEN",
        "Training-negative contract not frozen",
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
        "Evaluation candidate artifact not generated/audited",
    )

    require(
        eval_manifest[
            "training_performed"
        ]
        is False,
        "Training already performed unexpectedly",
    )

    require(
        int(
            eval_manifest[
                "candidate_contract"
            ][
                "negative_slots"
            ]
        )
        == EXPECTED[
            "evaluation_negative_slots"
        ],
        "Evaluation negative-slot count drift",
    )

    require(
        int(
            eval_manifest[
                "candidate_contract"
            ][
                "total_candidate_slots"
            ]
        )
        == EXPECTED[
            "evaluation_total_candidate_slots"
        ],
        "Evaluation total candidate-slot drift",
    )

    print(
        "Training-negative contract:           FROZEN  PASS"
    )

    print(
        "Evaluation runtime contract:          FROZEN  PASS"
    )

    print(
        "Evaluation candidates generated:      YES     PASS"
    )

    print(
        f"Evaluation negative slots:            "
        f"{EXPECTED['evaluation_negative_slots']:,}  PASS"
    )

    # =========================================================================
    # 5.2.1 audit status
    # =========================================================================

    banner(
        "PHASE 5.2.1 AUDIT RECHECK"
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
        == "5.2.1",
        "Unexpected Phase-5.2.1 audit phase",
    )

    require(
        audit_manifest[
            "status"
        ]
        == "AUDIT_COMPLETE_NOT_FROZEN",
        "Unexpected Phase-5.2.1 audit status",
    )

    require(
        audit_manifest[
            "training_negative_samples_generated"
        ]
        is False,
        "5.2.1 generated training negatives unexpectedly",
    )

    require(
        audit_manifest[
            "model_instantiated"
        ]
        is False,
        "5.2.1 instantiated model unexpectedly",
    )

    require(
        audit_manifest[
            "optimizer_instantiated"
        ]
        is False,
        "5.2.1 instantiated optimizer unexpectedly",
    )

    require(
        audit_manifest[
            "training_performed"
        ]
        is False,
        "5.2.1 performed training unexpectedly",
    )

    print(
        "5.2.1 audit complete:                 YES  PASS"
    )

    print(
        "5.2.1 model instantiated:             NO   PASS"
    )

    print(
        "5.2.1 optimizer instantiated:         NO   PASS"
    )

    print(
        "5.2.1 training performed:             NO   PASS"
    )

    # =========================================================================
    # Epoch freeze evidence
    # =========================================================================

    banner(
        "20-EPOCH FREEZE EVIDENCE"
    )

    epoch_df = pd.read_csv(
        EPOCH_AUDIT
    )

    epoch20 = epoch_df.loc[
        epoch_df[
            "epochs"
        ]
        == FIXED_EPOCHS
    ].copy()

    require(
        len(epoch20) == 1,
        "Expected exactly one 20-epoch audit row",
    )

    epoch20 = (
        epoch20.iloc[0]
    )

    require(
        int(
            epoch20[
                "labeled_training_presentations"
            ]
        )
        == EXPECTED[
            "total_training_presentations"
        ],
        "20-epoch training-presentation arithmetic drift",
    )

    require(
        int(
            epoch20[
                "optimizer_batches"
            ]
        )
        == EXPECTED[
            "total_optimizer_batches"
        ],
        "20-epoch optimizer-batch arithmetic drift",
    )

    require(
        bool(
            epoch20[
                "external_NCF_20_epoch_reference"
            ]
        ),
        "20-epoch external-reference audit flag missing",
    )

    print(
        f"Fixed training epochs:               "
        f"{FIXED_EPOCHS}"
    )

    print(
        f"Labeled presentations:               "
        f"{EXPECTED['total_training_presentations']:,}"
    )

    print(
        f"Optimizer batches:                   "
        f"{EXPECTED['total_optimizer_batches']:,}"
    )

    # =========================================================================
    # Training-control evidence
    # =========================================================================

    banner(
        "NO-EARLY-STOPPING FREEZE EVIDENCE"
    )

    training_control = pd.read_csv(
        TRAINING_CONTROL_REGISTER
    )

    selected_control = training_control.loc[
        training_control[
            "option"
        ]
        == "fixed_20_epochs_no_early_stopping"
    ].copy()

    require(
        len(selected_control) == 1,
        "Missing fixed-20/no-early-stopping audit row",
    )

    selected_control = (
        selected_control.iloc[0]
    )

    require(
        selected_control[
            "audit_assessment"
        ]
        == "STRONG_CANDIDATE",
        "Unexpected training-control audit assessment",
    )

    require(
        not bool(
            selected_control[
                "early_stopping"
            ]
        ),
        "Selected training control unexpectedly enables early stopping",
    )

    require(
        bool(
            selected_control[
                "best_validation_checkpoint_retained"
            ]
        ),
        "Selected training control does not retain best checkpoint",
    )

    print(
        "Early stopping:                       DISABLED"
    )

    print(
        "Validation after each epoch:           YES"
    )

    print(
        "Best validation checkpoint retained:  YES"
    )

    # =========================================================================
    # Checkpoint-selection evidence
    # =========================================================================

    banner(
        "CHECKPOINT-SELECTION FREEZE EVIDENCE"
    )

    checkpoint_df = pd.read_csv(
        CHECKPOINT_REGISTER
    )

    selected_checkpoint = checkpoint_df.loc[
        checkpoint_df[
            "option"
        ]
        == (
            "max_validation_NDCG10_then_HR10_"
            "then_earliest_epoch"
        )
    ].copy()

    require(
        len(selected_checkpoint) == 1,
        "Missing checkpoint-selection audit row",
    )

    selected_checkpoint = (
        selected_checkpoint.iloc[0]
    )

    require(
        selected_checkpoint[
            "audit_assessment"
        ]
        == "STRONG_CANDIDATE",
        "Unexpected checkpoint-selection assessment",
    )

    require(
        selected_checkpoint[
            "primary_metric"
        ]
        == CHECKPOINT_PRIMARY_METRIC,
        "Primary checkpoint metric drift",
    )

    require(
        selected_checkpoint[
            "secondary_metric"
        ]
        == CHECKPOINT_SECONDARY_METRIC,
        "Secondary checkpoint metric drift",
    )

    require(
        not bool(
            selected_checkpoint[
                "test_labels_used"
            ]
        ),
        "Checkpoint selection unexpectedly uses test labels",
    )

    print(
        "Primary checkpoint metric:            NDCG@10"
    )

    print(
        "Secondary checkpoint metric:          HR@10"
    )

    print(
        "Final tie-break:                      earliest epoch"
    )

    print(
        "Test labels used during selection:    NO"
    )

    # =========================================================================
    # Weight-decay evidence
    # =========================================================================

    banner(
        "ZERO WEIGHT-DECAY FREEZE EVIDENCE"
    )

    weight_decay_df = pd.read_csv(
        WEIGHT_DECAY_REGISTER
    )

    zero_decay = weight_decay_df.loc[
        weight_decay_df[
            "weight_decay"
        ]
        == WEIGHT_DECAY
    ].copy()

    require(
        len(zero_decay) == 1,
        "Missing zero-weight-decay audit row",
    )

    zero_decay = (
        zero_decay.iloc[0]
    )

    require(
        zero_decay[
            "audit_assessment"
        ]
        == "STRONG_CANDIDATE_MINIMAL_ASSUMPTION",
        "Unexpected zero-weight-decay assessment",
    )

    require(
        not bool(
            zero_decay[
                "introduces_nonzero_optimizer_regularization"
            ]
        ),
        "Zero weight decay marked as nonzero regularization",
    )

    print(
        "Adam weight_decay:                    0.0"
    )

    print(
        "Additional optimizer L2 regularizer:  NONE"
    )

    # =========================================================================
    # PyTorch Adam runtime inspection / exact implementation freeze
    # =========================================================================

    banner(
        "EXACT ADAM RUNTIME FREEZE"
    )

    require(
        torch.__version__.startswith(
            "2.7.0"
        ),
        (
            "Reference training contract expects PyTorch 2.7.0; "
            f"current runtime is {torch.__version__}"
        ),
    )

    runtime_betas = (
        get_runtime_default(
            "betas"
        )
    )

    runtime_eps = (
        get_runtime_default(
            "eps"
        )
    )

    runtime_weight_decay = (
        get_runtime_default(
            "weight_decay"
        )
    )

    runtime_amsgrad = (
        get_runtime_default(
            "amsgrad"
        )
    )

    require(
        tuple(
            runtime_betas
        )
        == ADAM_BETAS,
        "PyTorch Adam beta defaults drift",
    )

    require(
        float(
            runtime_eps
        )
        == ADAM_EPS,
        "PyTorch Adam epsilon default drift",
    )

    require(
        float(
            runtime_weight_decay
        )
        == 0.0,
        "PyTorch Adam runtime default weight decay drift",
    )

    require(
        bool(
            runtime_amsgrad
        )
        is ADAM_AMSGRAD,
        "PyTorch Adam AMSGrad default drift",
    )

    adam_signature = inspect.signature(
        torch.optim.Adam.__init__
    )

    has_fused = (
        "fused"
        in adam_signature.parameters
    )

    has_foreach = (
        "foreach"
        in adam_signature.parameters
    )

    require(
        has_fused,
        "PyTorch Adam runtime does not expose fused parameter",
    )

    require(
        has_foreach,
        "PyTorch Adam runtime does not expose foreach parameter",
    )

    print(
        f"PyTorch version:                      "
        f"{torch.__version__}"
    )

    print(
        f"Optimizer:                            "
        f"{OPTIMIZER_NAME}"
    )

    print(
        f"lr:                                   "
        f"{LEARNING_RATE}"
    )

    print(
        f"betas:                                "
        f"{ADAM_BETAS}"
    )

    print(
        f"eps:                                  "
        f"{ADAM_EPS}"
    )

    print(
        f"weight_decay:                         "
        f"{WEIGHT_DECAY}"
    )

    print(
        f"amsgrad:                              "
        f"{ADAM_AMSGRAD}"
    )

    print(
        f"foreach:                              "
        f"{ADAM_FOREACH}"
    )

    print(
        f"fused:                                "
        f"{ADAM_FUSED}"
    )

    print(
        f"maximize:                             "
        f"{ADAM_MAXIMIZE}"
    )

    print(
        f"capturable:                           "
        f"{ADAM_CAPTURABLE}"
    )

    print(
        f"differentiable:                       "
        f"{ADAM_DIFFERENTIABLE}"
    )

    print(
        "Learning-rate scheduler:              NONE"
    )

    print(
        "Optimizer instantiated by this script:NO"
    )

    # =========================================================================
    # Batch semantics
    # =========================================================================

    banner(
        "MINI-BATCH COMPLETENESS CONTRACT"
    )

    examples_per_epoch = EXPECTED[
        "training_examples_per_epoch"
    ]

    full_batches = (
        examples_per_epoch
        // BATCH_SIZE
    )

    remainder = (
        examples_per_epoch
        % BATCH_SIZE
    )

    total_batches = (
        full_batches
        + (
            1
            if remainder > 0
            else 0
        )
    )

    require(
        full_batches
        == EXPECTED[
            "full_batches_per_epoch"
        ],
        "Full-batch count drift",
    )

    require(
        remainder
        == EXPECTED[
            "final_batch_size"
        ],
        "Final partial-batch size drift",
    )

    require(
        total_batches
        == EXPECTED[
            "batches_per_epoch"
        ],
        "Total per-epoch batch count drift",
    )

    print(
        f"Examples / epoch:                    "
        f"{examples_per_epoch:,}"
    )

    print(
        f"Full 512-example batches:             "
        f"{full_batches:,}"
    )

    print(
        f"Final partial batch size:             "
        f"{remainder}"
    )

    print(
        f"Total batches / epoch:                "
        f"{total_batches:,}"
    )

    print(
        "drop_last:                           FALSE"
    )

    # =========================================================================
    # Training-order runtime freeze
    # =========================================================================

    banner(
        "DETERMINISTIC TRAINING-ORDER RUNTIME FREEZE"
    )

    train_order_df = pd.read_csv(
        TRAIN_ORDER_REGISTER
    )

    selected_order = train_order_df.loc[
        train_order_df[
            "option"
        ]
        == (
            "deterministic_shuffle_all_"
            "labeled_examples_each_epoch"
        )
    ].copy()

    require(
        len(selected_order) == 1,
        "Missing deterministic training-order audit row",
    )

    selected_order = (
        selected_order.iloc[0]
    )

    require(
        selected_order[
            "audit_assessment"
        ]
        == "STRONG_CANDIDATE",
        "Unexpected training-order assessment",
    )

    require(
        bool(
            selected_order[
                "shuffle"
            ]
        ),
        "Selected training-order strategy does not shuffle",
    )

    require(
        bool(
            selected_order[
                "dedicated_rng_stream"
            ]
        ),
        "Selected training-order strategy lacks dedicated RNG",
    )

    train_order_seed_examples = {
        str(epoch): (
            derive_train_order_seed_without_rng(
                epoch
            )
        )
        for epoch in range(5)
    }

    print(
        "Shuffle all labeled examples:         YES"
    )

    print(
        "Shuffle frequency:                    once per epoch"
    )

    print(
        f"Base seed:                            "
        f"{TRAIN_ORDER_BASE_SEED}"
    )

    print(
        f"Namespace:                            "
        f"{TRAIN_ORDER_NAMESPACE}"
    )

    print(
        f"BitGenerator:                         "
        f"{TRAIN_ORDER_BIT_GENERATOR}"
    )

    print(
        "Shared with negative sampler RNG:     NO"
    )

    print(
        "Training-order RNG instantiated here: NO"
    )

    print()

    print(
        "First five frozen train-order seeds "
        "(hashing only):"
    )

    for epoch, seed in (
        train_order_seed_examples.items()
    ):

        print(
            f"  epoch {epoch}: {seed}"
        )

    # =========================================================================
    # Training-negative RNG backend completion
    # =========================================================================

    banner(
        "TRAINING-NEGATIVE RNG BACKEND COMPLETION"
    )

    require(
        train_negative_contract[
            "sampling_runtime"
        ][
            "dedicated_sampler_rng_required"
        ]
        is True,
        "Frozen 5.1.1d contract does not require dedicated sampler RNG",
    )

    require(
        train_negative_contract[
            "sampling_runtime"
        ][
            "regenerate_each_epoch"
        ]
        is True,
        "Frozen negative regeneration cadence drift",
    )

    require(
        int(
            train_negative_contract[
                "sampling_runtime"
            ][
                "sampler_base_seed"
            ]
        )
        == 42,
        "Frozen training-negative base seed drift",
    )

    print(
        "5.1.1d sampling semantics modified:   NO"
    )

    print(
        "Negative ratio:                       inherited -> 4"
    )

    print(
        "Distribution:                         inherited -> uniform"
    )

    print(
        "Replacement:                          inherited -> without replacement"
    )

    print(
        "Regeneration:                         inherited -> each epoch"
    )

    print(
        f"Concrete BitGenerator:                "
        f"{TRAIN_NEGATIVE_BIT_GENERATOR}"
    )

    print(
        "Classification:                       "
        "IMPLEMENTATION_EQUIVALENT_CHOICE"
    )

    # =========================================================================
    # Exact frozen pre-training contract
    # =========================================================================

    banner(
        "EXACT FROZEN PRE-TRAINING CONTRACT"
    )

    print(
        "Epochs:                               20"
    )

    print(
        "Early stopping:                       DISABLED"
    )

    print(
        "Validation:                           after every completed epoch"
    )

    print(
        "Checkpoint selection:                 "
        "max NDCG@10 -> max HR@10 -> earliest epoch"
    )

    print(
        "Test evaluation during training:       FORBIDDEN"
    )

    print(
        "Test evaluation after selection:       ONCE"
    )

    print(
        "Optimizer:                            Adam"
    )

    print(
        "Learning rate:                        0.001 constant"
    )

    print(
        "Weight decay:                         0.0"
    )

    print(
        "LR scheduler:                         NONE"
    )

    print(
        "Batch size:                           512"
    )

    print(
        "Final partial batch:                  KEEP (485 examples)"
    )

    print(
        "Training shuffle:                     deterministic per epoch"
    )

    print(
        "Training negative BitGenerator:       numpy.random.PCG64"
    )

    print(
        "Training order BitGenerator:          numpy.random.PCG64"
    )

    # =========================================================================
    # Decision register
    # =========================================================================

    decisions = pd.DataFrame(
        [
            {
                "decision_id": "optimizer_family",
                "value": "Adam",
                "classification": "PAPER_SPECIFIED",
                "status": "FROZEN",
                "rationale": (
                    "ITRS explicitly reports Adam."
                ),
            },
            {
                "decision_id": "learning_rate",
                "value": "0.001",
                "classification": "PAPER_SPECIFIED",
                "status": "FROZEN",
                "rationale": (
                    "ITRS explicitly reports learning rate 0.001."
                ),
            },
            {
                "decision_id": "mini_batch_size",
                "value": "512",
                "classification": "PAPER_SPECIFIED",
                "status": "FROZEN",
                "rationale": (
                    "ITRS explicitly reports mini-batch size 512."
                ),
            },
            {
                "decision_id": "training_epoch_count",
                "value": "20",
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_2_2",
                "rationale": (
                    "ITRS does not report epoch count. "
                    "20 follows the closest NCF-family methodological "
                    "precedent audited in Phase 5.2.1."
                ),
            },
            {
                "decision_id": "early_stopping",
                "value": "disabled",
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_2_2",
                "rationale": (
                    "Avoids inventing undocumented patience/min_delta."
                ),
            },
            {
                "decision_id": "validation_frequency",
                "value": "after_every_completed_epoch",
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_2_2",
                "rationale": (
                    "Required for deterministic best-checkpoint selection."
                ),
            },
            {
                "decision_id": "checkpoint_selection",
                "value": (
                    "max_NDCG10_then_max_HR10_then_earliest_epoch"
                ),
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_2_2",
                "rationale": (
                    "Uses ITRS ranking metrics and validation set; "
                    "test remains untouched."
                ),
            },
            {
                "decision_id": "weight_decay",
                "value": "0.0",
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_2_2",
                "rationale": (
                    "Adds no unreported optimizer regularization."
                ),
            },
            {
                "decision_id": "learning_rate_scheduler",
                "value": "none",
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_2_2",
                "rationale": (
                    "ITRS reports lr=0.001 and no scheduler."
                ),
            },
            {
                "decision_id": "adam_betas",
                "value": "(0.9,0.999)",
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_2_2",
                "rationale": (
                    "Audited PyTorch 2.7.0 Adam defaults."
                ),
            },
            {
                "decision_id": "adam_eps",
                "value": "1e-8",
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_2_2",
                "rationale": (
                    "Audited PyTorch 2.7.0 Adam default."
                ),
            },
            {
                "decision_id": "adam_amsgrad",
                "value": "false",
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_2_2",
                "rationale": (
                    "Audited PyTorch 2.7.0 default."
                ),
            },
            {
                "decision_id": "adam_foreach",
                "value": "false",
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_2_2",
                "rationale": (
                    "Explicit single-tensor optimizer path prevents "
                    "runtime-dependent foreach auto-selection."
                ),
            },
            {
                "decision_id": "adam_fused",
                "value": "false",
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_2_2",
                "rationale": (
                    "Explicitly disables runtime-dependent fused path."
                ),
            },
            {
                "decision_id": "drop_last",
                "value": "false",
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_2_2",
                "rationale": (
                    "Preserves all 5,366,245 labeled examples; "
                    "final batch contains 485 examples."
                ),
            },
            {
                "decision_id": "training_order",
                "value": (
                    "deterministic_shuffle_all_examples_once_per_epoch"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_2_2",
                "rationale": (
                    "Avoids arbitrary serialization-order effects."
                ),
            },
            {
                "decision_id": "training_order_rng",
                "value": (
                    "PCG64 + SHA256("
                    "ITRS_PHASE5_TRAIN_ORDER|42|epoch_index)"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_2_2",
                "rationale": (
                    "Independent reproducible epoch ordering."
                ),
            },
            {
                "decision_id": "training_negative_bit_generator",
                "value": "numpy.random.PCG64",
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": "FROZEN_PHASE_5_2_2",
                "rationale": (
                    "Concrete RNG backend for the already-frozen "
                    "Phase-5.1.1d sampling contract."
                ),
            },
            {
                "decision_id": "test_usage_during_training",
                "value": "forbidden",
                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),
                "status": "FROZEN_PHASE_5_2_2",
                "rationale": (
                    "Test labels/results cannot affect checkpoint choice."
                ),
            },
        ]
    )

    # =========================================================================
    # Freeze audit
    # =========================================================================

    checks = [
        (
            "training_negative_contract_frozen",
            train_negative_contract["status"]
            == "FROZEN",
        ),
        (
            "evaluation_contract_frozen",
            eval_contract["status"]
            == "FROZEN",
        ),
        (
            "evaluation_candidates_generated",
            eval_manifest["status"]
            == "GENERATED_AND_AUDITED",
        ),
        (
            "phase_5_2_1_audit_complete",
            audit_manifest["status"]
            == "AUDIT_COMPLETE_NOT_FROZEN",
        ),
        (
            "epoch_count_20",
            FIXED_EPOCHS == 20,
        ),
        (
            "early_stopping_disabled",
            EARLY_STOPPING is False,
        ),
        (
            "validation_every_epoch",
            VALIDATE_AFTER_EVERY_EPOCH is True,
        ),
        (
            "checkpoint_primary_NDCG10",
            CHECKPOINT_PRIMARY_METRIC
            == "NDCG@10",
        ),
        (
            "checkpoint_secondary_HR10",
            CHECKPOINT_SECONDARY_METRIC
            == "HR@10",
        ),
        (
            "weight_decay_zero",
            WEIGHT_DECAY == 0.0,
        ),
        (
            "constant_learning_rate_0_001",
            LEARNING_RATE == 0.001,
        ),
        (
            "no_lr_scheduler",
            LEARNING_RATE_SCHEDULER is None,
        ),
        (
            "batch_size_512",
            BATCH_SIZE == 512,
        ),
        (
            "final_batch_size_485",
            remainder == 485,
        ),
        (
            "drop_last_false",
            DROP_LAST is False,
        ),
        (
            "20_epoch_training_presentations",
            EXPECTED[
                "training_examples_per_epoch"
            ]
            * FIXED_EPOCHS
            == EXPECTED[
                "total_training_presentations"
            ],
        ),
        (
            "20_epoch_optimizer_batches",
            EXPECTED[
                "batches_per_epoch"
            ]
            * FIXED_EPOCHS
            == EXPECTED[
                "total_optimizer_batches"
            ],
        ),
        (
            "adam_betas_match_runtime",
            tuple(runtime_betas)
            == ADAM_BETAS,
        ),
        (
            "adam_eps_matches_runtime",
            float(runtime_eps)
            == ADAM_EPS,
        ),
        (
            "adam_weight_decay_runtime_default_zero",
            float(runtime_weight_decay)
            == 0.0,
        ),
        (
            "no_model_instantiated",
            True,
        ),
        (
            "no_optimizer_instantiated",
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
            for name, passed
            in checks
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
            "One or more Phase-5.2.2 "
            "freeze checks failed"
        ),
    )

    # =========================================================================
    # Frozen contract
    # =========================================================================

    contract = {
        "phase": "5.2.2",
        "title": (
            "Final Training-Control and "
            "Optimizer Runtime Freeze"
        ),
        "status": "FROZEN",

        "training_negative_samples_generated": False,
        "training_rng_instantiated": False,
        "model_instantiated": False,
        "optimizer_instantiated": False,
        "training_performed": False,

        "paper_specified": {
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "validation_used_for_parameter_selection": True,
        },

        "paper_unspecified_reproduction_choices": {
            "fixed_epochs": FIXED_EPOCHS,
            "early_stopping": EARLY_STOPPING,
            "weight_decay": WEIGHT_DECAY,
            "learning_rate_scheduler": (
                LEARNING_RATE_SCHEDULER
            ),
            "checkpoint_selection": {
                "primary": (
                    CHECKPOINT_PRIMARY_METRIC
                ),
                "secondary": (
                    CHECKPOINT_SECONDARY_METRIC
                ),
                "final_tiebreak": (
                    CHECKPOINT_FINAL_TIEBREAK
                ),
            },
        },

        "validation_runtime": {
            "evaluate_after_every_completed_epoch": True,
            "epochs_evaluated": list(
                range(
                    1,
                    FIXED_EPOCHS + 1,
                )
            ),
            "immutable_candidate_artifact": True,
            "validation_cases": (
                EXPECTED[
                    "validation_cases"
                ]
            ),
            "candidate_scores_per_epoch": (
                EXPECTED[
                    "validation_candidates_per_epoch"
                ]
            ),
            "candidate_scores_across_20_epochs": (
                EXPECTED[
                    "validation_candidate_scores_20_epochs"
                ]
            ),
            "test_used_for_checkpoint_selection": False,
            "test_evaluations_before_checkpoint_selection": 0,
            "final_test_evaluations": 1,
        },

        "optimizer_runtime": {
            "torch_reference_version": (
                torch.__version__
            ),
            "optimizer": OPTIMIZER_NAME,
            "lr": LEARNING_RATE,
            "betas": list(
                ADAM_BETAS
            ),
            "eps": ADAM_EPS,
            "weight_decay": WEIGHT_DECAY,
            "amsgrad": ADAM_AMSGRAD,
            "foreach": ADAM_FOREACH,
            "fused": ADAM_FUSED,
            "maximize": ADAM_MAXIMIZE,
            "capturable": ADAM_CAPTURABLE,
            "differentiable": (
                ADAM_DIFFERENTIABLE
            ),
            "learning_rate_scheduler": None,
        },

        "batch_runtime": {
            "examples_per_epoch": (
                EXPECTED[
                    "training_examples_per_epoch"
                ]
            ),
            "batch_size": BATCH_SIZE,
            "full_batches_per_epoch": (
                full_batches
            ),
            "final_partial_batch_size": (
                remainder
            ),
            "batches_per_epoch": (
                total_batches
            ),
            "drop_last": DROP_LAST,
        },

        "training_order_runtime": {
            "shuffle": True,
            "shuffle_frequency": (
                "once_per_epoch_after_negative_generation"
            ),
            "base_seed": (
                TRAIN_ORDER_BASE_SEED
            ),
            "namespace": (
                TRAIN_ORDER_NAMESPACE
            ),
            "seed_derivation": (
                "SHA256("
                "ITRS_PHASE5_TRAIN_ORDER|42|epoch_index"
                "); first 8 bytes little-endian unsigned; "
                "modulo (2^63 - 1)"
            ),
            "bit_generator": (
                TRAIN_ORDER_BIT_GENERATOR
            ),
            "share_rng_with_negative_sampler": False,
            "share_rng_with_model": False,
            "share_rng_with_evaluation": False,
            "epoch_indexing": "zero_based",
            "reference_seed_examples_hash_only": (
                train_order_seed_examples
            ),
        },

        "training_negative_runtime_completion": {
            "semantic_contract_source": (
                str(
                    TRAIN_NEGATIVE_CONTRACT
                )
            ),
            "semantic_contract_modified": False,
            "bit_generator": (
                TRAIN_NEGATIVE_BIT_GENERATOR
            ),
            "classification": (
                "IMPLEMENTATION_EQUIVALENT_CHOICE"
            ),
        },

        "training_workload": {
            "fixed_epochs": (
                FIXED_EPOCHS
            ),
            "positive_examples_per_epoch": (
                EXPECTED[
                    "training_positive_examples_per_epoch"
                ]
            ),
            "negative_examples_per_epoch": (
                EXPECTED[
                    "training_negative_examples_per_epoch"
                ]
            ),
            "labeled_examples_per_epoch": (
                EXPECTED[
                    "training_examples_per_epoch"
                ]
            ),
            "total_labeled_presentations": (
                EXPECTED[
                    "total_training_presentations"
                ]
            ),
            "optimizer_batches_per_epoch": (
                EXPECTED[
                    "batches_per_epoch"
                ]
            ),
            "total_optimizer_batches": (
                EXPECTED[
                    "total_optimizer_batches"
                ]
            ),
        },

        "test_integrity_guard": {
            "test_labels_may_select_checkpoint": False,
            "test_metrics_may_select_checkpoint": False,
            "test_may_trigger_early_stopping": False,
            "test_evaluate_once_after_best_checkpoint_selected": True,
        },

        "original_phase_5_handoff_decisions_remaining": [],

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
                EPOCH_AUDIT
            ): sha256_file(
                EPOCH_AUDIT
            ),
            str(
                TRAINING_CONTROL_REGISTER
            ): sha256_file(
                TRAINING_CONTROL_REGISTER
            ),
            str(
                CHECKPOINT_REGISTER
            ): sha256_file(
                CHECKPOINT_REGISTER
            ),
            str(
                WEIGHT_DECAY_REGISTER
            ): sha256_file(
                WEIGHT_DECAY_REGISTER
            ),
            str(
                ADAM_DEFAULTS
            ): sha256_file(
                ADAM_DEFAULTS
            ),
            str(
                TRAIN_ORDER_REGISTER
            ): sha256_file(
                TRAIN_ORDER_REGISTER
            ),
            str(
                AUDIT_MANIFEST
            ): sha256_file(
                AUDIT_MANIFEST
            ),
        },
    }

    # =========================================================================
    # Write
    # =========================================================================

    banner(
        "WRITE FROZEN PHASE-5.2.2 CONTRACT"
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
    # Final
    # =========================================================================

    banner(
        "PHASE 5.2.2 FINAL STATUS"
    )

    print(
        "Training epoch count:                 FROZEN -> 20"
    )

    print(
        "Early stopping:                       FROZEN -> DISABLED"
    )

    print(
        "Weight decay:                         FROZEN -> 0.0"
    )

    print(
        "Validation frequency:                 FROZEN -> every epoch"
    )

    print(
        "Checkpoint primary metric:            FROZEN -> NDCG@10"
    )

    print(
        "Checkpoint secondary metric:          FROZEN -> HR@10"
    )

    print(
        "Checkpoint tie-break:                 FROZEN -> earliest epoch"
    )

    print(
        "Learning-rate scheduler:              FROZEN -> NONE"
    )

    print(
        "Final partial batch:                  FROZEN -> KEEP (485)"
    )

    print(
        "Training order:                       FROZEN -> deterministic shuffle"
    )

    print(
        "Training-negative BitGenerator:       FROZEN -> PCG64"
    )

    print(
        "Training-order BitGenerator:          FROZEN -> PCG64"
    )

    print(
        "Test use during checkpoint selection: FROZEN -> FORBIDDEN"
    )

    print()

    print(
        "Original Phase-5 handoff decisions remaining: 0"
    )

    print()

    print(
        "Training negatives generated:         0"
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

    banner(
        "PHASE 5.2.2 COMPLETE / "
        "ALL PRE-TRAINING DECISIONS FROZEN"
    )


if __name__ == "__main__":
    main()