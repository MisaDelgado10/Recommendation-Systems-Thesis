#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json
import math


# =============================================================================
# PHASE 8.2.7 — AUTOGRAD / OPTIMIZER PREFLIGHT FORMAL CLOSURE V1
# =============================================================================


ROOT = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_7_autograd_optimizer_preflight"
)

INTERPRETATION = (
    ROOT
    / "phase_8_2_7_autograd_optimizer_interpretation_closure_V2.json"
)

INTERPRETATION_HASHES = (
    ROOT
    / "phase_8_2_7_autograd_optimizer_interpretation_closure_sha256_V2.json"
)

EXECUTION_RESULT = (
    ROOT
    / "phase_8_2_7_dev_t57_frozen_batch_one_step_results_V1.json"
)

EXECUTION_HASHES = (
    ROOT
    / "phase_8_2_7_dev_t57_frozen_batch_one_step_sha256_V1.json"
)

EXECUTION_SCRIPT = Path(
    "scripts/"
    "phase_8_2_7_dev_t57_frozen_batch_one_step_preflight_V1.py"
)

OUTPUT_MANIFEST = (
    ROOT
    / "phase_8_2_7_autograd_optimizer_preflight_closure_manifest_V1.json"
)

OUTPUT_HASHES = (
    ROOT
    / "phase_8_2_7_autograd_optimizer_preflight_closure_sha256_V1.json"
)


EXPECTED_FAMILIES = {
    "B1a": {
        "trend_active": True,
        "graph_active": False,
        "structured_mode": "ZERO",
        "parameter_tensors": 29,
        "parameter_elements": 6_751_789,
    },

    "B1b": {
        "trend_active": True,
        "graph_active": False,
        "structured_mode": "BASE",
        "parameter_tensors": 33,
        "parameter_elements": 6_875_797,
    },

    "M1": {
        "trend_active": False,
        "graph_active": True,
        "structured_mode": "EXTENDED",
        "parameter_tensors": 31,
        "parameter_elements": 6_920_129,
    },

    "M2": {
        "trend_active": True,
        "graph_active": True,
        "structured_mode": "EXTENDED",
        "parameter_tensors": 41,
        "parameter_elements": 6_952_609,
    },
}


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


print("=" * 120)
print("PHASE 8.2.7 — AUTOGRAD / OPTIMIZER PREFLIGHT FORMAL CLOSURE V1")
print("=" * 120)


# =============================================================================
# 1. REQUIRED ARTIFACTS
# =============================================================================

print()
print("1. REQUIRED ARTIFACTS")

required = [
    INTERPRETATION,
    INTERPRETATION_HASHES,
    EXECUTION_RESULT,
    EXECUTION_HASHES,
    EXECUTION_SCRIPT,
]

for path in required:
    require(
        path.exists(),
        f"Missing required artifact: {path}",
    )

print(
    f"  Required artifacts: {len(required)} / {len(required)} PASS"
)


# =============================================================================
# 2. INTERPRETATION CONTRACT
# =============================================================================

print()
print("2. FROZEN INTERPRETATION CONTRACT")

interpretation = json.loads(
    INTERPRETATION.read_text(
        encoding="utf-8"
    )
)

require(
    interpretation["status"]
    == "FROZEN_BEFORE_MODEL_EXECUTION",
    "8.2.7 interpretation contract status changed.",
)

require(
    interpretation["phase"]
    == "8.2.7",
    "Interpretation phase mismatch.",
)

require(
    interpretation[
        "scope"
    ][
        "randomness_scope"
    ]
    == "DEV_T57",
    "Interpretation scope changed.",
)

require(
    interpretation[
        "scope"
    ][
        "epoch_index"
    ]
    == 0,
    "Interpretation epoch changed.",
)

require(
    interpretation[
        "scope"
    ][
        "batch_size"
    ]
    == 512,
    "Interpretation batch size changed.",
)

require(
    interpretation[
        "candidate_families"
    ]
    == [
        "B1a",
        "B1b",
        "M1",
        "M2",
    ],
    "Candidate-family order changed.",
)

require(
    interpretation[
        "training_allowed_now"
    ]
    is False,
    "Interpretation contract unexpectedly permits training.",
)

require(
    interpretation[
        "optimizer_step_policy"
    ][
        "optimizer_step_calls_per_family"
    ]
    == 1,
    "Optimizer-step contract changed.",
)

require(
    interpretation[
        "optimizer_step_policy"
    ][
        "updated_state_retained"
    ]
    is False,
    "Interpretation unexpectedly permits updated-state retention.",
)

print("  Interpretation contract: PASS")


# =============================================================================
# 3. INTERPRETATION HASH REGISTRY
# =============================================================================

print()
print("3. INTERPRETATION ARTIFACT INTEGRITY")

interpretation_hashes = json.loads(
    INTERPRETATION_HASHES.read_text(
        encoding="utf-8"
    )
)

require(
    str(INTERPRETATION)
    in interpretation_hashes,
    "Interpretation hash registry missing contract.",
)

require(
    sha256_file(INTERPRETATION)
    == interpretation_hashes[
        str(INTERPRETATION)
    ],
    "Interpretation contract hash mismatch.",
)

print("  Interpretation SHA256: PASS")


# =============================================================================
# 4. EXECUTION RESULT HASH REGISTRY
# =============================================================================

print()
print("4. EXECUTION ARTIFACT INTEGRITY")

execution_hashes = json.loads(
    EXECUTION_HASHES.read_text(
        encoding="utf-8"
    )
)

require(
    str(EXECUTION_RESULT)
    in execution_hashes,
    "Execution hash registry missing result.",
)

require(
    sha256_file(EXECUTION_RESULT)
    == execution_hashes[
        str(EXECUTION_RESULT)
    ],
    "Execution result SHA256 mismatch.",
)

print("  Execution-result SHA256: PASS")


# =============================================================================
# 5. EXECUTION RESULT CONTRACT
# =============================================================================

print()
print("5. REAL ONE-BATCH EXECUTION CONTRACT")

result = json.loads(
    EXECUTION_RESULT.read_text(
        encoding="utf-8"
    )
)

require(
    result["status"]
    == "ONE_STEP_AUTOGRAD_OPTIMIZER_PREFLIGHT_EXECUTED_AND_AUDITED",
    "Unexpected 8.2.7 execution status.",
)

require(
    result["phase"]
    == "8.2.7",
    "Execution phase mismatch.",
)

require(
    result["scope"]
    == "DEV_T57",
    "Execution scope changed.",
)

require(
    result["origin"]
    == "T57",
    "Execution origin changed.",
)

require(
    result["epoch_index"]
    == 0,
    "Execution epoch changed.",
)


batch = result["batch"]

require(
    batch["rows"] == 512,
    "Batch row count changed.",
)

require(
    batch["positives"] == 98,
    "Batch positive count changed.",
)

require(
    batch["negatives"] == 414,
    "Batch negative count changed.",
)

require(
    batch["same_batch_all_families"]
    is True,
    "Families did not share the same frozen batch.",
)

require(
    batch["negative_regeneration"]
    is False,
    "Negatives were unexpectedly regenerated.",
)

require(
    batch["order_regeneration"]
    is False,
    "Epoch order was unexpectedly regenerated.",
)

print("  Frozen batch identity: PASS")


# =============================================================================
# 6. REAL RUNTIME INPUTS
# =============================================================================

print()
print("6. REAL RUNTIME INPUT AUDIT")

runtime = result[
    "real_runtime_inputs"
]

require(
    runtime[
        "unique_batch_investors"
    ]
    == 489,
    "Batch Investor count changed.",
)

require(
    runtime[
        "unique_batch_startups"
    ]
    == 512,
    "Batch Startup count changed.",
)

require(
    runtime[
        "strict_history_memberships"
    ]
    == 49_602,
    "Strict history membership count changed.",
)

require(
    runtime[
        "strict_history_unique_startups"
    ]
    == 34_239,
    "Strict history Startup count changed.",
)

require(
    runtime[
        "graph_closure_nodes"
    ]
    == 2_484,
    "Graph closure node count changed.",
)

require(
    runtime[
        "graph_closure_edges"
    ]
    == 2_483,
    "Graph closure edge count changed.",
)

require(
    runtime[
        "graph_relations_present"
    ]
    == list(range(10)),
    "Not all ten frozen graph relations were present.",
)

require(
    runtime[
        "startup_feature_union_rows"
    ]
    == 35_074,
    "One-batch Startup feature union changed.",
)

print("  Frozen real-data wiring: PASS")


# =============================================================================
# 7. OPTIMIZER CONTRACT
# =============================================================================

print()
print("7. OPTIMIZER CONTRACT")

optimizer = result[
    "optimizer"
]

require(
    optimizer["class"]
    == "torch.optim.Adam",
    "Optimizer class changed.",
)

require(
    math.isclose(
        float(
            optimizer["lr"]
        ),
        1e-3,
        rel_tol=0.0,
        abs_tol=0.0,
    ),
    "Adam learning rate changed.",
)

require(
    optimizer["betas"]
    == [
        0.9,
        0.999,
    ],
    "Adam betas changed.",
)

require(
    math.isclose(
        float(
            optimizer["eps"]
        ),
        1e-8,
        rel_tol=0.0,
        abs_tol=0.0,
    ),
    "Adam epsilon changed.",
)

require(
    float(
        optimizer[
            "weight_decay"
        ]
    )
    == 0.0,
    "Adam weight decay changed.",
)

require(
    optimizer["amsgrad"]
    is False,
    "Adam amsgrad changed.",
)

require(
    optimizer["foreach"]
    is False,
    "Adam foreach changed.",
)

require(
    optimizer["fused"]
    is False,
    "Adam fused changed.",
)

print("  Adam configuration: PASS")


# =============================================================================
# 8. FAMILY-BY-FAMILY AUTOGRAD / UPDATE AUDIT
# =============================================================================

print()
print("8. FAMILY-BY-FAMILY AUTOGRAD / UPDATE AUDIT")

families = result[
    "families"
]

require(
    set(families.keys())
    == set(EXPECTED_FAMILIES.keys()),
    "Family result set changed.",
)


for family, expected in EXPECTED_FAMILIES.items():

    observed = families[
        family
    ]

    architecture = observed[
        "architecture"
    ]

    numerical = observed[
        "numerical_audit"
    ]

    state = observed[
        "state_audit"
    ]

    post_step = observed[
        "post_step_policy"
    ]

    # -------------------------------------------------------------------------
    # Frozen topology.
    # -------------------------------------------------------------------------

    require(
        architecture[
            "trend_active"
        ]
        is expected[
            "trend_active"
        ],
        f"{family}: trend branch drift.",
    )

    require(
        architecture[
            "graph_active"
        ]
        is expected[
            "graph_active"
        ],
        f"{family}: graph branch drift.",
    )

    require(
        architecture[
            "structured_mode"
        ]
        == expected[
            "structured_mode"
        ],
        f"{family}: structured mode drift.",
    )

    require(
        architecture[
            "parameter_tensors"
        ]
        == expected[
            "parameter_tensors"
        ],
        f"{family}: parameter tensor count drift.",
    )

    require(
        architecture[
            "parameter_elements"
        ]
        == expected[
            "parameter_elements"
        ],
        f"{family}: parameter element count drift.",
    )

    # -------------------------------------------------------------------------
    # Numerical integrity.
    # -------------------------------------------------------------------------

    require(
        math.isfinite(
            float(
                numerical[
                    "loss_bce_with_logits"
                ]
            )
        ),
        f"{family}: BCE is non-finite.",
    )

    require(
        numerical[
            "logits_finite"
        ]
        is True,
        f"{family}: non-finite logits.",
    )

    require(
        numerical[
            "loss_finite"
        ]
        is True,
        f"{family}: non-finite loss.",
    )

    require(
        numerical[
            "all_gradients_present"
        ]
        is True,
        f"{family}: missing gradient.",
    )

    require(
        numerical[
            "all_gradients_finite"
        ]
        is True,
        f"{family}: non-finite gradient.",
    )

    require(
        numerical[
            "all_active_components_nonzero_gradient"
        ]
        is True,
        f"{family}: zero-gradient active component.",
    )

    require(
        numerical[
            "gradient_tensors"
        ]
        == expected[
            "parameter_tensors"
        ],
        (
            f"{family}: not every parameter tensor "
            "received a gradient."
        ),
    )

    for component, grad_abs_sum in (
        numerical[
            "gradient_components"
        ].items()
    ):

        require(
            math.isfinite(
                float(
                    grad_abs_sum
                )
            )
            and float(
                grad_abs_sum
            ) > 0.0,
            (
                f"{family}: component {component} "
                "has invalid aggregate gradient."
            ),
        )

    # -------------------------------------------------------------------------
    # State mutation boundary.
    # -------------------------------------------------------------------------

    require(
        state[
            "forward_parameter_neutrality"
        ]
        is True,
        f"{family}: forward mutated parameters.",
    )

    require(
        state[
            "backward_parameter_neutrality"
        ]
        is True,
        f"{family}: backward mutated parameters.",
    )

    require(
        state[
            "optimizer_step_calls"
        ]
        == 1,
        f"{family}: optimizer.step count != 1.",
    )

    require(
        state[
            "post_step_hash_differs"
        ]
        is True,
        f"{family}: optimizer step produced no state change.",
    )

    require(
        state[
            "initial_parameter_sha256"
        ]
        != state[
            "post_step_parameter_sha256"
        ],
        f"{family}: pre/post hashes are equal.",
    )

    # Stronger than the minimum gate:
    # every trainable parameter tensor changed.
    require(
        state[
            "changed_parameter_tensors"
        ]
        == expected[
            "parameter_tensors"
        ],
        (
            f"{family}: changed parameter tensor count "
            "does not equal total parameter tensor count."
        ),
    )

    require(
        state[
            "optimizer_state_entries_after_step"
        ]
        == expected[
            "parameter_tensors"
        ],
        (
            f"{family}: Adam state was not created "
            "for every parameter tensor."
        ),
    )

    # -------------------------------------------------------------------------
    # No post-step use/persistence.
    # -------------------------------------------------------------------------

    require(
        post_step[
            "post_step_forward_executed"
        ]
        is False,
        f"{family}: post-step forward occurred.",
    )

    require(
        post_step[
            "post_step_scoring_executed"
        ]
        is False,
        f"{family}: post-step scoring occurred.",
    )

    require(
        post_step[
            "checkpoint_written"
        ]
        is False,
        f"{family}: checkpoint was written.",
    )

    require(
        post_step[
            "updated_state_reused"
        ]
        is False,
        f"{family}: updated state was reused.",
    )

    require(
        post_step[
            "updated_state_retained"
        ]
        is False,
        f"{family}: updated state was retained.",
    )

    print(
        f"  {family}: "
        f"{expected['parameter_tensors']}/"
        f"{expected['parameter_tensors']} tensors "
        "gradient-bearing + updated — PASS"
    )


# =============================================================================
# 9. GLOBAL EXECUTION BOUNDARY
# =============================================================================

print()
print("9. GLOBAL EXECUTION BOUNDARY")

execution = result[
    "execution_summary"
]

required_true = [
    "models_instantiated",
    "canonical_initialization_applied",
    "real_forward_executed",
    "BCEWithLogitsLoss_computed",
    "backward_executed",
    "optimizer_created",
    "optimizer_step_executed",
]

for key in required_true:

    require(
        execution[key]
        is True,
        f"Expected execution flag not true: {key}",
    )


required_false = [
    "full_epoch_training_executed",
    "ranking_metrics_computed",
    "model_selection_performed",
    "post_step_forward_executed",
    "checkpoint_written",
    "updated_state_retained",
    "T60_scored",
    "April_analyzed",
]

for key in required_false:

    require(
        execution[key]
        is False,
        f"Forbidden execution flag true: {key}",
    )


require(
    execution[
        "optimizer_step_calls_per_family"
    ]
    == 1,
    "Global optimizer-step count changed.",
)

require(
    result[
        "training_allowed_now"
    ]
    is False,
    "8.2.7 result incorrectly authorizes training.",
)

print("  Execution boundary: PASS")


# =============================================================================
# 10. CHECKPOINT FILESYSTEM AUDIT
# =============================================================================

print()
print("10. CHECKPOINT PERSISTENCE AUDIT")

checkpoint_suffixes = {
    ".pt",
    ".pth",
    ".ckpt",
    ".bin",
}

checkpoint_files = [
    path
    for path in ROOT.rglob("*")
    if (
        path.is_file()
        and path.suffix.lower()
        in checkpoint_suffixes
    )
]

require(
    len(checkpoint_files) == 0,
    (
        "Unexpected checkpoint-like file(s) "
        f"inside Phase 8.2.7 root: {checkpoint_files}"
    ),
)

print("  Checkpoint-like files under 8.2.7 root: 0 PASS")


# =============================================================================
# 11. FORMAL CLOSURE
# =============================================================================

print()
print("11. WRITE FORMAL PHASE 8.2.7 CLOSURE")

closure = {
    "schema_version":
        "PHASE_8_2_7_AUTOGRAD_OPTIMIZER_PREFLIGHT_CLOSURE_V1",

    "phase":
        "8.2.7",

    "status":
        "AUTOGRAD_OPTIMIZER_PREFLIGHT_CLOSED",

    "scientific_role":
        "FINAL_ONE_BATCH_DIFFERENTIABILITY_AND_UPDATE_GATE",

    "authoritative_interpretation":
        str(INTERPRETATION),

    "authoritative_execution_result":
        str(EXECUTION_RESULT),

    "scope": {
        "randomness_scope":
            "DEV_T57",

        "origin":
            "T57",

        "epoch_index":
            0,

        "display_epoch":
            1,

        "batch_rows":
            512,

        "positives":
            98,

        "negatives":
            414,

        "same_frozen_batch_all_families":
            True,
    },

    "families": {
        family: {
            "parameter_tensors":
                expected[
                    "parameter_tensors"
                ],

            "parameter_elements":
                expected[
                    "parameter_elements"
                ],

            "all_parameter_tensors_received_gradient":
                True,

            "all_gradients_finite":
                True,

            "all_active_components_nonzero_gradient":
                True,

            "forward_parameter_neutral":
                True,

            "backward_parameter_neutral":
                True,

            "optimizer_step_calls":
                1,

            "all_parameter_tensors_changed_after_step":
                True,

            "Adam_state_created_for_all_parameter_tensors":
                True,

            "post_step_forward":
                False,

            "post_step_scoring":
                False,

            "checkpoint_written":
                False,

            "updated_state_retained":
                False,
        }
        for family, expected
        in EXPECTED_FAMILIES.items()
    },

    "verified_gates": {
        "canonical_initialization":
            True,

        "real_frozen_batch_forward":
            True,

        "raw_logits_finite":
            True,

        "BCEWithLogitsLoss_finite":
            True,

        "backward":
            True,

        "gradient_integrity":
            True,

        "forward_state_neutrality":
            True,

        "backward_state_neutrality":
            True,

        "exactly_one_Adam_step_per_family":
            True,

        "post_step_parameter_change":
            True,

        "Adam_state_creation":
            True,

        "updated_state_discarded":
            True,

        "checkpoint_files_absent":
            True,
    },

    "forbidden_actions_confirmed_absent": {
        "full_epoch_training":
            True,

        "ranking_metric_computation":
            True,

        "epoch_selection":
            True,

        "model_selection":
            True,

        "post_step_scoring":
            True,

        "checkpoint_persistence":
            True,

        "updated_state_reuse":
            True,

        "T60_scoring":
            True,

        "April_analysis":
            True,
    },

    "execution_state": {
        "model_instantiated_for_preflight":
            True,

        "forward_executed":
            True,

        "BCE_computed":
            True,

        "backward_executed":
            True,

        "optimizer_created":
            True,

        "optimizer_step_executed":
            True,

        "optimizer_step_calls_per_family":
            1,

        "post_step_forward_executed":
            False,

        "full_training_epoch_executed":
            False,

        "performance_inspected":
            False,

        "checkpoint_written":
            False,

        "updated_state_retained":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,
    },

    "training_allowed_now":
        False,

    "next_phase":
        "8.2.8",

    "next_step":
        (
            "Final Phase 8.2.8 integrated pre-training gate. "
            "Comparative development training remains forbidden "
            "until Phase 8.2.8 itself passes and formally authorizes launch."
        ),
}


OUTPUT_MANIFEST.write_text(
    json.dumps(
        closure,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)


hash_targets = [
    INTERPRETATION,
    INTERPRETATION_HASHES,
    EXECUTION_RESULT,
    EXECUTION_HASHES,
    EXECUTION_SCRIPT,
    OUTPUT_MANIFEST,
]

OUTPUT_HASHES.write_text(
    json.dumps(
        {
            str(path):
                sha256_file(path)
            for path
            in hash_targets
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)


# =============================================================================
# FINAL STATUS
# =============================================================================

print(
    f"  Closure manifest: {OUTPUT_MANIFEST}"
)

print(
    f"  SHA256 registry:  {OUTPUT_HASHES}"
)

print()
print("=" * 120)
print("PHASE 8.2.7 — CLOSED")
print("=" * 120)

print("Families:                           4 / 4")
print("Frozen batch:                       512 / 512")
print("Canonical initialization:           PASS")
print("Real forward:                       PASS")
print("BCEWithLogitsLoss:                  PASS")
print("Backward:                           PASS")
print("All parameter gradients present:    PASS")
print("All gradients finite:               PASS")
print("Active-component gradients:         PASS")
print("Forward parameter neutrality:       PASS")
print("Backward parameter neutrality:      PASS")
print("Exactly one Adam step / family:     PASS")
print("All parameter tensors updated:      PASS")
print("Adam state creation:                PASS")
print("Updated states discarded:           PASS")
print("Checkpoint persistence:             NONE")
print("Post-step scoring:                  NO")
print("Full epoch training:                NO")
print("Performance inspected:              NO")
print("T60 scored:                         NO")
print("April analyzed:                     NO")
print("Training allowed now:               NO")

print()
print("STATUS: AUTOGRAD_OPTIMIZER_PREFLIGHT_CLOSED")
print("NEXT:   Phase 8.2.8 — final integrated pre-training gate")
print("=" * 120)
