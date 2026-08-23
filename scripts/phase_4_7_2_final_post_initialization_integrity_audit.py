from pathlib import Path
import hashlib
import json
import sys

import pandas as pd


# =============================================================================
# PHASE 4.7.2 — FINAL POST-INITIALIZATION MODEL INTEGRITY AUDIT
#
# PURPOSE
# -------
# Verify that the newly frozen Phase-4.7.1b initialization/seed contract
# integrates cleanly with the already frozen Phase-4 model reconstruction.
#
# THIS SCRIPT DOES NOT:
#
#   - instantiate the 19M-parameter model,
#   - run neural initialization again,
#   - run the R-GCN,
#   - run a forward pass,
#   - run BCE,
#   - run backward,
#   - train,
#   - create an optimizer,
#   - generate negatives,
#   - modify any frozen artifact.
#
#
# IMPORTANT PROVENANCE RULE
# -------------------------
# The older Phase-4.6.1b topology contract correctly records that, at the time
# it was frozen, the exact Kaiming variant was NOT_YET_FROZEN.
#
# We DO NOT rewrite that historical contract.
#
# Instead:
#
#   Phase 4.6.1b
#       records the architecture before initialization was resolved.
#
#   Phase 4.7.1b
#       is the authoritative later contract resolving initialization + seed.
#
# Phase 4.7.2 verifies that these contracts compose consistently.
# =============================================================================


# =============================================================================
# ROOT
# =============================================================================

PHASE_4_ROOT = Path(
    "data/experimental/phase_4"
)


# =============================================================================
# PHASE 4.7.1a INPUTS
# =============================================================================

PHASE_4_7_1A_DIR = (
    PHASE_4_ROOT
    / "model_integrity_audit"
)


PHASE_4_7_1A_METADATA_PATH = (
    PHASE_4_7_1A_DIR
    / "phase_4_7_1a_integrity_metadata.json"
)

PHASE_4_7_1A_GLOBAL_AUDIT_PATH = (
    PHASE_4_7_1A_DIR
    / "phase_4_7_1a_global_integrity_audit.csv"
)

PHASE_4_7_1A_DECISION_CLASSIFICATION_PATH = (
    PHASE_4_7_1A_DIR
    / "phase_4_7_1a_open_decision_classification.csv"
)


# =============================================================================
# PHASE 4.7.1b INPUTS
# =============================================================================

INITIALIZATION_DIR = (
    PHASE_4_ROOT
    / "initialization_contract"
)


INITIALIZATION_DECISION_AUDIT_PATH = (
    INITIALIZATION_DIR
    / "phase_4_7_1b_initialization_decision_audit.csv"
)

PARAMETER_INITIALIZATION_AUDIT_PATH = (
    INITIALIZATION_DIR
    / "phase_4_7_1b_parameter_initialization_audit.csv"
)

SEED_REPRODUCIBILITY_AUDIT_PATH = (
    INITIALIZATION_DIR
    / "phase_4_7_1b_seed_reproducibility_audit.csv"
)

INITIALIZATION_STATE_HASH_PATH = (
    INITIALIZATION_DIR
    / "phase_4_7_1b_initialization_state_hash.json"
)

NEURAL_INITIALIZATION_CONTRACT_PATH = (
    INITIALIZATION_DIR
    / "phase_4_7_1b_neural_initialization_contract.json"
)

PHASE_4_7_1B_ARTIFACT_HASHES_PATH = (
    INITIALIZATION_DIR
    / "phase_4_7_1b_artifact_hashes.csv"
)


# =============================================================================
# FROZEN FULL MODEL INPUTS
# =============================================================================

FULL_MODEL_CONTRACT_DIR = (
    PHASE_4_ROOT
    / "full_model_contract"
)


FULL_MODEL_TOPOLOGY_CONTRACT_PATH = (
    FULL_MODEL_CONTRACT_DIR
    / "full_itrs_model_topology_contract.json"
)

PARAMETER_NAMESPACE_PATH = (
    FULL_MODEL_CONTRACT_DIR
    / "full_model_parameter_namespace.csv"
)


# =============================================================================
# PHASE 4.6 CLOSURE
# =============================================================================

PHASE_4_6_CLOSURE_PATH = (
    PHASE_4_ROOT
    / "full_model_forward_audit"
    / "phase_4_6_closure_manifest.json"
)


# =============================================================================
# OUTPUTS
# =============================================================================

OUT_DIR = (
    PHASE_4_ROOT
    / "model_integrity_audit"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FROZEN EXPECTATIONS
# =============================================================================

EXPECTED_FULL_PARAMETERS = 19_217_929

EXPECTED_PARAMETER_TENSORS = 32

EXPECTED_KAIMING_TENSORS = 21

EXPECTED_ZERO_BIAS_TENSORS = 11

EXPECTED_GLOBAL_NEURAL_SEED = 42

EXPECTED_REFERENCE_TORCH_VERSION = "2.7.0"

EXPECTED_INITIALIZATION_FUNCTION = (
    "torch.nn.init.kaiming_normal_"
)

EXPECTED_KAIMING_DISTRIBUTION = "normal"

EXPECTED_KAIMING_A = 0.0

EXPECTED_KAIMING_MODE = "fan_in"

EXPECTED_KAIMING_NONLINEARITY = "relu"


# =============================================================================
# HISTORICAL PHASE-4.7.1a BLOCKERS
# =============================================================================

EXPECTED_HISTORICAL_BLOCKERS = {

    "exact global Kaiming initialization variant",

    "global neural seed policy",
}


# =============================================================================
# DECISIONS THAT MUST REMAIN DEFERRED
# =============================================================================

EXPECTED_DEFERRED_DECISIONS = {

    "training negative:positive ratio",

    "training negative candidate eligibility",

    "training historical negative exclusion",

    "training epoch count",

    "early stopping",

    "weight decay",

    "evaluation candidate-generation runtime contract",
}


# =============================================================================
# HELPERS
# =============================================================================

def banner(title):

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def require(
    condition,
    message,
):

    if not condition:

        raise AssertionError(
            message
        )


def load_json(path):

    require(
        path.exists(),
        f"Missing required JSON: {path}",
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def sha256_file(path):

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        for chunk in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


def parse_bool(value):

    if isinstance(
        value,
        bool,
    ):

        return value


    text = str(
        value
    ).strip().casefold()


    if text in {
        "true",
        "1",
        "yes",
    }:

        return True


    if text in {
        "false",
        "0",
        "no",
    }:

        return False


    raise AssertionError(
        f"Could not parse boolean value: {value}"
    )


# =============================================================================
# START
# =============================================================================

banner(
    "PHASE 4.7.2 — "
    "FINAL POST-INITIALIZATION MODEL INTEGRITY AUDIT"
)


# =============================================================================
# 1. ENVIRONMENT
# =============================================================================

banner(
    "ENVIRONMENT"
)


print(
    f"Python: "
    f"{sys.version.splitlines()[0]}"
)


# =============================================================================
# 2. REQUIRED ARTIFACT EXISTENCE
# =============================================================================

banner(
    "REQUIRED ARTIFACT EXISTENCE"
)


required_paths = [

    PHASE_4_7_1A_METADATA_PATH,
    PHASE_4_7_1A_GLOBAL_AUDIT_PATH,
    PHASE_4_7_1A_DECISION_CLASSIFICATION_PATH,

    INITIALIZATION_DECISION_AUDIT_PATH,
    PARAMETER_INITIALIZATION_AUDIT_PATH,
    SEED_REPRODUCIBILITY_AUDIT_PATH,
    INITIALIZATION_STATE_HASH_PATH,
    NEURAL_INITIALIZATION_CONTRACT_PATH,
    PHASE_4_7_1B_ARTIFACT_HASHES_PATH,

    FULL_MODEL_TOPOLOGY_CONTRACT_PATH,
    PARAMETER_NAMESPACE_PATH,

    PHASE_4_6_CLOSURE_PATH,
]


for path in required_paths:

    exists = path.exists()


    print(
        f"{str(path):<110} "
        f"{'FOUND' if exists else 'MISSING'}"
    )


    require(
        exists,
        f"Missing required Phase-4 artifact: {path}",
    )


# =============================================================================
# 3. PHASE 4.7.1a INTEGRITY MUST STILL PASS
# =============================================================================

banner(
    "PHASE 4.7.1a HISTORICAL INTEGRITY"
)


integrity_1a = load_json(
    PHASE_4_7_1A_METADATA_PATH
)


require(
    integrity_1a.get(
        "status"
    )
    == "COMPLETE_AUDIT_ONLY",
    "Phase 4.7.1a status changed.",
)


global_audit_1a = pd.read_csv(
    PHASE_4_7_1A_GLOBAL_AUDIT_PATH
)


require(
    {
        "check",
        "status",
    }
    .issubset(
        global_audit_1a.columns
    ),
    "Phase 4.7.1a global-audit schema changed.",
)


failed_1a_checks = global_audit_1a[
    global_audit_1a[
        "status"
    ]
    .astype(str)
    .str.upper()
    != "PASS"
]


print(
    f"Phase 4.7.1a global checks: "
    f"{len(global_audit_1a)}"
)

print(
    f"Failures:                  "
    f"{len(failed_1a_checks)}"
)


require(
    len(
        failed_1a_checks
    )
    == 0,
    "A Phase 4.7.1a integrity check no longer passes.",
)


historical_blockers = set(
    integrity_1a[
        "phase_4_closure_blockers"
    ]
)


print()
print(
    "Historical blockers recorded by 4.7.1a:"
)


for blocker in sorted(
    historical_blockers
):

    print(
        f"  - {blocker}"
    )


require(
    historical_blockers
    == EXPECTED_HISTORICAL_BLOCKERS,
    "Historical Phase-4 blocker set changed.",
)


# =============================================================================
# 4. HISTORICAL OPEN-DECISION CLASSIFICATION
# =============================================================================

banner(
    "HISTORICAL OPEN-DECISION CLASSIFICATION"
)


decision_1a = pd.read_csv(
    PHASE_4_7_1A_DECISION_CLASSIFICATION_PATH
)


require(
    {
        "decision",
        "classification",
        "next_action",
    }
    .issubset(
        decision_1a.columns
    ),
    "Phase 4.7.1a decision schema changed.",
)


blocker_rows = decision_1a[
    decision_1a[
        "classification"
    ]
    == "PHASE_4_CLOSURE_BLOCKER"
]


deferred_rows_1a = decision_1a[
    decision_1a[
        "classification"
    ]
    == "DEFER_TO_TRAINING_OR_EVALUATION"
]


blocker_decisions_1a = set(
    blocker_rows[
        "decision"
    ]
    .astype(str)
)


deferred_decisions_1a = set(
    deferred_rows_1a[
        "decision"
    ]
    .astype(str)
)


print(
    f"Historical Phase-4 blockers: "
    f"{len(blocker_decisions_1a)}"
)

print(
    f"Historical deferred choices: "
    f"{len(deferred_decisions_1a)}"
)


require(
    blocker_decisions_1a
    == EXPECTED_HISTORICAL_BLOCKERS,
    "4.7.1a blocker classification changed.",
)


require(
    deferred_decisions_1a
    == EXPECTED_DEFERRED_DECISIONS,
    "4.7.1a deferred decision set changed.",
)


# =============================================================================
# 5. PHASE 4.7.1b CONTRACT
# =============================================================================

banner(
    "PHASE 4.7.1b INITIALIZATION CONTRACT"
)


init_contract = load_json(
    NEURAL_INITIALIZATION_CONTRACT_PATH
)


require(
    init_contract.get(
        "status"
    )
    == "FROZEN",
    "Phase 4.7.1b initialization contract is not frozen.",
)


require(
    init_contract[
        "global_neural_seed"
    ][
        "value"
    ]
    == EXPECTED_GLOBAL_NEURAL_SEED,
    "Frozen global neural seed changed.",
)


require(
    init_contract[
        "kaiming"
    ][
        "function"
    ]
    == EXPECTED_INITIALIZATION_FUNCTION,
    "Frozen Kaiming function changed.",
)


require(
    init_contract[
        "kaiming"
    ][
        "distribution"
    ]
    == EXPECTED_KAIMING_DISTRIBUTION,
    "Frozen Kaiming distribution changed.",
)


require(
    float(
        init_contract[
            "kaiming"
        ][
            "a"
        ]
    )
    == EXPECTED_KAIMING_A,
    "Frozen Kaiming a changed.",
)


require(
    init_contract[
        "kaiming"
    ][
        "mode"
    ]
    == EXPECTED_KAIMING_MODE,
    "Frozen Kaiming mode changed.",
)


require(
    init_contract[
        "kaiming"
    ][
        "nonlinearity"
    ]
    == EXPECTED_KAIMING_NONLINEARITY,
    "Frozen Kaiming nonlinearity changed.",
)


require(
    init_contract[
        "bias"
    ][
        "initialization"
    ]
    == "zeros",
    "Frozen bias initialization changed.",
)


require(
    float(
        init_contract[
            "bias"
        ][
            "value"
        ]
    )
    == 0.0,
    "Frozen bias value changed.",
)


require(
    init_contract[
        "reference_environment"
    ][
        "pytorch"
    ]
    == EXPECTED_REFERENCE_TORCH_VERSION,
    "Frozen PyTorch initialization environment changed.",
)


require(
    init_contract[
        "reference_environment"
    ][
        "canonical_initialization_device"
    ]
    == "cpu",
    "Canonical initialization device changed.",
)


print(
    f"Global neural seed:     "
    f"{EXPECTED_GLOBAL_NEURAL_SEED} PASS"
)

print(
    "Kaiming function:       "
    "kaiming_normal_ PASS"
)

print(
    "Kaiming mode:           "
    "fan_in PASS"
)

print(
    "Kaiming nonlinearity:   "
    "relu PASS"
)

print(
    "Bias initialization:    "
    "zero PASS"
)

print(
    "Reference PyTorch:      "
    "2.7.0 PASS"
)

print(
    "Canonical device:       "
    "CPU PASS"
)


# =============================================================================
# 6. INITIALIZATION COVERAGE CONTRACT
# =============================================================================

banner(
    "INITIALIZATION COVERAGE CONTRACT"
)


coverage = init_contract[
    "coverage"
]


require(
    coverage[
        "parameter_tensors"
    ]
    == EXPECTED_PARAMETER_TENSORS,
    "Initialization parameter-tensor count changed.",
)


require(
    coverage[
        "kaiming_parameter_tensors"
    ]
    == EXPECTED_KAIMING_TENSORS,
    "Kaiming tensor count changed.",
)


require(
    coverage[
        "zero_bias_tensors"
    ]
    == EXPECTED_ZERO_BIAS_TENSORS,
    "Zero-bias tensor count changed.",
)


require(
    coverage[
        "missing_parameters"
    ]
    == 0,
    "Initialization contract now misses parameters.",
)


require(
    coverage[
        "unexpected_parameters"
    ]
    == 0,
    "Initialization contract contains unexpected parameters.",
)


require(
    coverage[
        "duplicate_initialization_specs"
    ]
    == 0,
    "Duplicate initialization specification detected.",
)


print(
    f"Total parameter tensors: "
    f"{coverage['parameter_tensors']}"
)

print(
    f"Kaiming tensors:         "
    f"{coverage['kaiming_parameter_tensors']}"
)

print(
    f"Zero-bias tensors:       "
    f"{coverage['zero_bias_tensors']}"
)

print(
    "Missing specs:           0"
)

print(
    "Unexpected specs:        0"
)

print(
    "Duplicate specs:         0"
)


# =============================================================================
# 7. PARAMETER NAMESPACE VS INITIALIZATION NAMESPACE
# =============================================================================

banner(
    "PARAMETER NAMESPACE / INITIALIZATION NAMESPACE"
)


namespace = pd.read_csv(
    PARAMETER_NAMESPACE_PATH
)


parameter_init_audit = pd.read_csv(
    PARAMETER_INITIALIZATION_AUDIT_PATH
)


require(
    {
        "parameter",
        "numel",
    }
    .issubset(
        namespace.columns
    ),
    "Frozen parameter namespace schema changed.",
)


require(
    {
        "parameter",
        "numel",
        "initialization_kind",
        "finite",
        "sha256",
        "status",
    }
    .issubset(
        parameter_init_audit.columns
    ),
    "Initialization parameter-audit schema changed.",
)


frozen_parameter_names = set(
    namespace[
        "parameter"
    ]
    .astype(str)
)


initialization_parameter_names = set(
    parameter_init_audit[
        "parameter"
    ]
    .astype(str)
)


print(
    f"Frozen topology parameter tensors: "
    f"{len(frozen_parameter_names)}"
)

print(
    f"Initialized parameter tensors:     "
    f"{len(initialization_parameter_names)}"
)


require(
    frozen_parameter_names
    == initialization_parameter_names,
    (
        "Initialization namespace differs from "
        "frozen full-model parameter namespace."
    ),
)


namespace_total = int(
    namespace[
        "numel"
    ].sum()
)


initialization_total = int(
    parameter_init_audit[
        "numel"
    ].sum()
)


print(
    f"Frozen topology parameters: "
    f"{namespace_total:,}"
)

print(
    f"Initialization parameters:  "
    f"{initialization_total:,}"
)


require(
    namespace_total
    == EXPECTED_FULL_PARAMETERS,
    "Frozen topology parameter total changed.",
)


require(
    initialization_total
    == EXPECTED_FULL_PARAMETERS,
    "Initialization parameter total changed.",
)


# =============================================================================
# 8. PER-PARAMETER INITIALIZATION AUDIT RECHECK
# =============================================================================

banner(
    "PER-PARAMETER INITIALIZATION AUDIT RECHECK"
)


failed_init_parameters = parameter_init_audit[
    parameter_init_audit[
        "status"
    ]
    .astype(str)
    .str.upper()
    != "PASS"
]


print(
    f"Parameter audit rows: "
    f"{len(parameter_init_audit)}"
)

print(
    f"Failed rows:          "
    f"{len(failed_init_parameters)}"
)


require(
    len(
        parameter_init_audit
    )
    == EXPECTED_PARAMETER_TENSORS,
    "Initialization audit no longer contains 32 parameters.",
)


require(
    len(
        failed_init_parameters
    )
    == 0,
    "At least one initialization parameter audit failed.",
)


finite_values = [
    parse_bool(
        value
    )
    for value
    in parameter_init_audit[
        "finite"
    ]
]


require(
    all(
        finite_values
    ),
    "At least one initialized parameter is recorded non-finite.",
)


kaiming_rows = parameter_init_audit[
    parameter_init_audit[
        "initialization_kind"
    ]
    .isin(
        [
            "kaiming",
            "kaiming_basis_stack",
        ]
    )
]


zero_bias_rows = parameter_init_audit[
    parameter_init_audit[
        "initialization_kind"
    ]
    == "zero_bias"
]


print(
    f"Kaiming rows:   "
    f"{len(kaiming_rows)}"
)

print(
    f"Zero-bias rows: "
    f"{len(zero_bias_rows)}"
)


require(
    len(
        kaiming_rows
    )
    == EXPECTED_KAIMING_TENSORS,
    "Per-parameter Kaiming count changed.",
)


require(
    len(
        zero_bias_rows
    )
    == EXPECTED_ZERO_BIAS_TENSORS,
    "Per-parameter zero-bias count changed.",
)


bias_zero_results = [
    parse_bool(
        value
    )
    for value
    in zero_bias_rows[
        "bias_exact_zero"
    ]
]


require(
    all(
        bias_zero_results
    ),
    "At least one frozen bias is not exactly zero.",
)


# =============================================================================
# 9. SEED REPRODUCIBILITY AUDIT RECHECK
# =============================================================================

banner(
    "SEED REPRODUCIBILITY AUDIT RECHECK"
)


seed_audit = pd.read_csv(
    SEED_REPRODUCIBILITY_AUDIT_PATH
)


require(
    {
        "check",
        "expected",
        "actual",
        "status",
    }
    .issubset(
        seed_audit.columns
    ),
    "Seed reproducibility audit schema changed.",
)


seed_failures = seed_audit[
    seed_audit[
        "status"
    ]
    .astype(str)
    .str.upper()
    != "PASS"
]


print(
    f"Seed audit checks: "
    f"{len(seed_audit)}"
)

print(
    f"Failures:          "
    f"{len(seed_failures)}"
)


require(
    len(
        seed_failures
    )
    == 0,
    "A seed reproducibility audit check failed.",
)


# =============================================================================
# 10. INITIALIZATION STATE-HASH CONSISTENCY
# =============================================================================

banner(
    "CANONICAL INITIALIZATION STATE-HASH CONSISTENCY"
)


state_hash = load_json(
    INITIALIZATION_STATE_HASH_PATH
)


require(
    state_hash.get(
        "status"
    )
    == "FROZEN",
    "Initialization state-hash manifest is not frozen.",
)


canonical_hash_state = state_hash[
    "canonical_state_sha256"
]


canonical_hash_contract = (
    init_contract[
        "reproducibility"
    ][
        "canonical_state_sha256"
    ]
)


repeat_hash_state = state_hash[
    "repeat_same_seed_sha256"
]


repeat_hash_contract = (
    init_contract[
        "reproducibility"
    ][
        "repeat_state_sha256"
    ]
)


control_hash_state = state_hash[
    "control_state_sha256"
]


control_hash_contract = (
    init_contract[
        "reproducibility"
    ][
        "different_seed_state_sha256"
    ]
)


print(
    "Canonical state SHA256:"
)

print(
    f"  {canonical_hash_state}"
)


print()
print(
    "Repeated seed-42 SHA256:"
)

print(
    f"  {repeat_hash_state}"
)


print()
print(
    "Seed-43 control SHA256:"
)

print(
    f"  {control_hash_state}"
)


require(
    canonical_hash_state
    == canonical_hash_contract,
    (
        "Canonical initialization hash differs "
        "between state manifest and contract."
    ),
)


require(
    repeat_hash_state
    == repeat_hash_contract,
    (
        "Repeated-state hash differs between "
        "state manifest and contract."
    ),
)


require(
    control_hash_state
    == control_hash_contract,
    (
        "Control-state hash differs between "
        "state manifest and contract."
    ),
)


require(
    canonical_hash_state
    == repeat_hash_state,
    "Same-seed initialization hashes are no longer identical.",
)


require(
    canonical_hash_state
    != control_hash_state,
    "Different seed no longer changes initialized state.",
)


require(
    state_hash[
        "global_neural_seed"
    ]
    == EXPECTED_GLOBAL_NEURAL_SEED,
    "State-hash manifest neural seed changed.",
)


require(
    state_hash[
        "same_seed_exact"
    ]
    is True,
    "Same-seed exact flag changed.",
)


require(
    state_hash[
        "different_seed_changes_state"
    ]
    is True,
    "Different-seed sensitivity flag changed.",
)


print()
print(
    "Cross-artifact canonical hash: PASS"
)

print(
    "Same seed exact:              PASS"
)

print(
    "Different seed sensitivity:   PASS"
)


# =============================================================================
# 11. VERIFY PHASE 4.7.1b ARTIFACT HASH MANIFEST
# =============================================================================

banner(
    "PHASE 4.7.1b ARTIFACT HASH RECHECK"
)


artifact_hashes = pd.read_csv(
    PHASE_4_7_1B_ARTIFACT_HASHES_PATH
)


require(
    {
        "artifact",
        "path",
        "sha256",
    }
    .issubset(
        artifact_hashes.columns
    ),
    "Phase 4.7.1b artifact-hash schema changed.",
)


artifact_hash_recheck_records = []


for _, row in artifact_hashes.iterrows():

    artifact = str(
        row[
            "artifact"
        ]
    )

    path = Path(
        row[
            "path"
        ]
    )

    frozen_hash = str(
        row[
            "sha256"
        ]
    ).strip().lower()


    require(
        path.exists(),
        (
            "Frozen Phase-4.7.1b artifact "
            f"is missing: {path}"
        ),
    )


    current_hash = sha256_file(
        path
    )


    exact = (
        current_hash
        == frozen_hash
    )


    print(
        f"{artifact:<58} "
        f"{'PASS' if exact else 'FAIL'}"
    )


    require(
        exact,
        (
            "Phase-4.7.1b artifact hash changed: "
            f"{artifact}"
        ),
    )


    artifact_hash_recheck_records.append(
        {

            "artifact":
                artifact,

            "path":
                str(
                    path
                ),

            "frozen_sha256":
                frozen_hash,

            "current_sha256":
                current_hash,

            "status":
                "PASS",
        }
    )


# =============================================================================
# 12. PAPER / REPRODUCTION CLASSIFICATION RECHECK
# =============================================================================

banner(
    "INITIALIZATION DECISION CLASSIFICATION RECHECK"
)


initialization_decisions = pd.read_csv(
    INITIALIZATION_DECISION_AUDIT_PATH
)


require(
    {
        "decision",
        "value",
        "classification",
        "reason",
    }
    .issubset(
        initialization_decisions.columns
    ),
    "Initialization decision-audit schema changed.",
)


require(
    len(
        initialization_decisions
    )
    > 0,
    "Initialization decision audit is empty.",
)


missing_classification = (
    initialization_decisions[
        "classification"
    ]
    .isna()
    .sum()
)


print(
    f"Initialization decisions: "
    f"{len(initialization_decisions)}"
)

print(
    f"Missing classifications:  "
    f"{missing_classification}"
)


require(
    missing_classification
    == 0,
    "At least one initialization decision lacks classification.",
)


# =============================================================================
# 13. HISTORICAL TOPOLOGY CONTRACT PROVENANCE
# =============================================================================

banner(
    "HISTORICAL TOPOLOGY / NEW INITIALIZATION COMPOSITION"
)


topology_contract = load_json(
    FULL_MODEL_TOPOLOGY_CONTRACT_PATH
)


require(
    topology_contract.get(
        "status"
    )
    == "FROZEN",
    "Full topology contract is no longer frozen.",
)


require(
    topology_contract[
        "parameter_budget"
    ][
        "total"
    ]
    == EXPECTED_FULL_PARAMETERS,
    "Full-model parameter total changed.",
)


historical_topology_init_value = (
    topology_contract[
        "initialization"
    ][
        "exact_variant"
    ]
)


print(
    "Phase 4.6.1b historical initialization field:"
)

print(
    f"  {historical_topology_init_value}"
)


require(
    historical_topology_init_value
    == "NOT_YET_FROZEN",
    (
        "Historical topology provenance was "
        "unexpectedly rewritten."
    ),
)


print()
print(
    "Authoritative later resolution:"
)

print(
    "  Phase 4.7.1b neural initialization contract"
)

print(
    "  status = FROZEN"
)


print()
print(
    "Historical contract rewritten: NO"
)

print(
    "Later contract resolves open field: YES"
)


# =============================================================================
# 14. DEFERRED DECISIONS MUST REMAIN DEFERRED
# =============================================================================

banner(
    "DEFERRED TRAINING / EVALUATION DECISIONS"
)


deferred_from_init_contract = set(
    init_contract[
        "still_deferred_to_training_evaluation"
    ]
)


print(
    f"Expected deferred decisions: "
    f"{len(EXPECTED_DEFERRED_DECISIONS)}"
)

print(
    f"Actual deferred decisions:   "
    f"{len(deferred_from_init_contract)}"
)


require(
    deferred_from_init_contract
    == EXPECTED_DEFERRED_DECISIONS,
    (
        "Training/evaluation deferred-decision "
        "set changed during initialization freeze."
    ),
)


for decision in sorted(
    deferred_from_init_contract
):

    print(
        f"  - {decision}"
    )


# =============================================================================
# 15. MODEL-RECONSTRUCTION BLOCKERS MUST NOW BE EMPTY
# =============================================================================

banner(
    "PHASE-4 MODEL-RECONSTRUCTION BLOCKER RESOLUTION"
)


remaining_blockers = (
    init_contract[
        "phase_4_model_reconstruction_blockers_remaining"
    ]
)


print(
    f"Historical blockers entering 4.7.1b: "
    f"{len(historical_blockers)}"
)

print(
    f"Blockers after 4.7.1b:             "
    f"{len(remaining_blockers)}"
)


require(
    len(
        remaining_blockers
    )
    == 0,
    (
        "Phase-4 model-reconstruction blockers "
        "remain after initialization freeze."
    ),
)


print()
print(
    "Resolved:"
)

print(
    "  exact global Kaiming initialization variant"
)

print(
    "  global neural seed policy"
)


# =============================================================================
# 16. TRAINING BOUNDARY
# =============================================================================

banner(
    "TRAINING BOUNDARY INTEGRITY"
)


require(
    init_contract[
        "training"
    ][
        "performed"
    ]
    is False,
    "Initialization contract unexpectedly records training.",
)


require(
    init_contract[
        "training"
    ][
        "optimizer_created"
    ]
    is False,
    "Initialization contract unexpectedly created optimizer.",
)


require(
    init_contract[
        "training"
    ][
        "optimizer_step"
    ]
    is False,
    "Initialization contract unexpectedly records optimizer.step.",
)


require(
    init_contract[
        "training"
    ][
        "model_state_persisted"
    ]
    is False,
    "Initialization audit unexpectedly persisted model state.",
)


phase_4_6_closure = load_json(
    PHASE_4_6_CLOSURE_PATH
)


require(
    phase_4_6_closure.get(
        "status"
    )
    == "COMPLETE",
    "Phase 4.6 closure status changed.",
)


require(
    phase_4_6_closure[
        "training_performed"
    ]
    is False,
    "Phase 4.6 unexpectedly records training.",
)


print(
    "Phase 4.6 training performed:   NO PASS"
)

print(
    "Phase 4.7.1b training performed:NO PASS"
)

print(
    "Optimizer created:              NO PASS"
)

print(
    "Optimizer step:                 NO PASS"
)

print(
    "Learned state persisted:        NO PASS"
)


# =============================================================================
# 17. GLOBAL POST-INITIALIZATION INTEGRITY SUMMARY
# =============================================================================

banner(
    "GLOBAL POST-INITIALIZATION INTEGRITY SUMMARY"
)


summary_checks = [

    (
        "Phase 4.7.1a integrity still PASS",
        len(
            failed_1a_checks
        )
        == 0,
    ),

    (
        "Historical blocker set preserved",
        historical_blockers
        == EXPECTED_HISTORICAL_BLOCKERS,
    ),

    (
        "Initialization contract frozen",
        init_contract.get(
            "status"
        )
        == "FROZEN",
    ),

    (
        "Global neural seed = 42",
        init_contract[
            "global_neural_seed"
        ][
            "value"
        ]
        == EXPECTED_GLOBAL_NEURAL_SEED,
    ),

    (
        "Kaiming normal",
        init_contract[
            "kaiming"
        ][
            "distribution"
        ]
        == "normal",
    ),

    (
        "Kaiming mode fan_in",
        init_contract[
            "kaiming"
        ][
            "mode"
        ]
        == "fan_in",
    ),

    (
        "Kaiming nonlinearity ReLU",
        init_contract[
            "kaiming"
        ][
            "nonlinearity"
        ]
        == "relu",
    ),

    (
        "Bias initialization zero",
        init_contract[
            "bias"
        ][
            "value"
        ]
        == 0.0,
    ),

    (
        "Initialization covers 32 tensors",
        len(
            initialization_parameter_names
        )
        == EXPECTED_PARAMETER_TENSORS,
    ),

    (
        "Initialization namespace exact",
        frozen_parameter_names
        == initialization_parameter_names,
    ),

    (
        "Initialization total 19,217,929",
        initialization_total
        == EXPECTED_FULL_PARAMETERS,
    ),

    (
        "21 Kaiming tensors",
        len(
            kaiming_rows
        )
        == EXPECTED_KAIMING_TENSORS,
    ),

    (
        "11 zero-bias tensors",
        len(
            zero_bias_rows
        )
        == EXPECTED_ZERO_BIAS_TENSORS,
    ),

    (
        "Every parameter initialization PASS",
        len(
            failed_init_parameters
        )
        == 0,
    ),

    (
        "Every initialized parameter finite",
        all(
            finite_values
        ),
    ),

    (
        "Every bias exactly zero",
        all(
            bias_zero_results
        ),
    ),

    (
        "Seed audit has zero failures",
        len(
            seed_failures
        )
        == 0,
    ),

    (
        "Same-seed state byte exact",
        canonical_hash_state
        == repeat_hash_state,
    ),

    (
        "Different seed changes state",
        canonical_hash_state
        != control_hash_state,
    ),

    (
        "Initialization artifacts unchanged",
        True,
    ),

    (
        "Historical topology provenance preserved",
        historical_topology_init_value
        == "NOT_YET_FROZEN",
    ),

    (
        "Seven training/evaluation choices deferred",
        deferred_from_init_contract
        == EXPECTED_DEFERRED_DECISIONS,
    ),

    (
        "Phase-4 reconstruction blockers = 0",
        len(
            remaining_blockers
        )
        == 0,
    ),

    (
        "No training performed",
        (
            init_contract[
                "training"
            ][
                "performed"
            ]
            is False
        ),
    ),
]


global_integrity_records = []


for (
    check,
    passed,
) in summary_checks:

    status = (
        "PASS"
        if passed
        else "FAIL"
    )


    print(
        f"{check:<58} "
        f"{status}"
    )


    require(
        passed,
        (
            "Post-initialization integrity failure: "
            f"{check}"
        ),
    )


    global_integrity_records.append(
        {

            "check":
                check,

            "status":
                status,
        }
    )


# =============================================================================
# 18. SAVE ARTIFACT HASH RECHECK
# =============================================================================

artifact_hash_recheck_df = pd.DataFrame(
    artifact_hash_recheck_records
)


artifact_hash_recheck_path = (
    OUT_DIR
    / "phase_4_7_2_initialization_artifact_hash_recheck.csv"
)


artifact_hash_recheck_df.to_csv(
    artifact_hash_recheck_path,
    index=False,
)


# =============================================================================
# 19. SAVE GLOBAL POST-INITIALIZATION INTEGRITY TABLE
# =============================================================================

global_integrity_df = pd.DataFrame(
    global_integrity_records
)


global_integrity_path = (
    OUT_DIR
    / "phase_4_7_2_post_initialization_integrity_audit.csv"
)


global_integrity_df.to_csv(
    global_integrity_path,
    index=False,
)


# =============================================================================
# 20. SAVE PHASE 4.7.2 METADATA
# =============================================================================

metadata = {

    "phase":
        "4.7.2",

    "status":
        "COMPLETE_AUDIT_ONLY",

    "component":
        (
            "Final post-initialization "
            "model integrity audit"
        ),

    "model":
        {

            "trainable_parameters":
                EXPECTED_FULL_PARAMETERS,

            "parameter_tensors":
                EXPECTED_PARAMETER_TENSORS,
        },

    "initialization":
        {

            "family":
                "Kaiming",

            "function":
                EXPECTED_INITIALIZATION_FUNCTION,

            "distribution":
                EXPECTED_KAIMING_DISTRIBUTION,

            "a":
                EXPECTED_KAIMING_A,

            "mode":
                EXPECTED_KAIMING_MODE,

            "nonlinearity":
                EXPECTED_KAIMING_NONLINEARITY,

            "bias":
                "zero",

            "global_neural_seed":
                EXPECTED_GLOBAL_NEURAL_SEED,

            "reference_pytorch":
                EXPECTED_REFERENCE_TORCH_VERSION,

            "canonical_initialization_device":
                "cpu",

            "kaiming_parameter_tensors":
                EXPECTED_KAIMING_TENSORS,

            "zero_bias_tensors":
                EXPECTED_ZERO_BIAS_TENSORS,

            "canonical_state_sha256":
                canonical_hash_state,

            "same_seed_exact":
                True,

            "different_seed_changes_state":
                True,
        },

    "historical_provenance":
        {

            "phase_4_6_1b_initialization_field":
                historical_topology_init_value,

            "historical_contract_rewritten":
                False,

            "resolved_by_later_contract":
                "phase_4_7_1b_neural_initialization_contract.json",
        },

    "phase_4_model_reconstruction_blockers_remaining":
        [],

    "deferred_to_training_evaluation":
        sorted(
            EXPECTED_DEFERRED_DECISIONS
        ),

    "training_performed":
        False,

    "frozen_architecture_changed":
        False,

    "phase_2_reopened":
        False,

    "phase_3_reopened":
        False,

    "next_phase":
        {

            "phase":
                "4.8",

            "name":
                "Phase-4 Closure and Handoff",
        },
}


metadata_path = (
    OUT_DIR
    / "phase_4_7_2_post_initialization_integrity_metadata.json"
)


with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 21. CLOSE PHASE 4.7
# =============================================================================

banner(
    "FREEZING PHASE 4.7 CLOSURE"
)


phase_4_7_closure = {

    "phase":
        "4.7",

    "name":
        "Complete Model Integrity Audit",

    "status":
        "COMPLETE",

    "subphases":
        {

            "4.7.1a":
                "COMPLETE_AUDIT_ONLY",

            "4.7.1b":
                "FROZEN",

            "4.7.2":
                "COMPLETE_AUDIT_ONLY",
        },

    "verified":
        {

            "pre_initialization_global_integrity":
                True,

            "exact_kaiming_variant_frozen":
                True,

            "global_neural_seed_frozen":
                True,

            "full_initialization_namespace_coverage":
                True,

            "same_seed_byte_reproducibility":
                True,

            "different_seed_sensitivity":
                True,

            "initialization_artifact_hash_integrity":
                True,

            "historical_topology_provenance_preserved":
                True,

            "training_boundary_preserved":
                True,
        },

    "model_reconstruction_blockers_remaining":
        [],

    "deferred_to_training_evaluation":
        sorted(
            EXPECTED_DEFERRED_DECISIONS
        ),

    "training_performed":
        False,

    "next_phase":
        {

            "phase":
                "4.8",

            "name":
                "Phase-4 Closure and Handoff",
        },
}


phase_4_7_closure_path = (
    OUT_DIR
    / "phase_4_7_closure_manifest.json"
)


with open(
    phase_4_7_closure_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        phase_4_7_closure,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 22. OUTPUT ARTIFACT HASHES
# =============================================================================

output_paths = [

    artifact_hash_recheck_path,

    global_integrity_path,

    metadata_path,

    phase_4_7_closure_path,
]


output_hash_records = []


for path in output_paths:

    output_hash_records.append(
        {

            "artifact":
                path.stem,

            "path":
                str(
                    path
                ),

            "sha256":
                sha256_file(
                    path
                ),

            "bytes":
                int(
                    path.stat().st_size
                ),
        }
    )


output_hash_df = pd.DataFrame(
    output_hash_records
)


output_hash_path = (
    OUT_DIR
    / "phase_4_7_2_artifact_hashes.csv"
)


output_hash_df.to_csv(
    output_hash_path,
    index=False,
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

banner(
    "PHASE 4.7.2 FINAL SUMMARY"
)


print(
    "Pre-initialization integrity:"
)

print(
    "  Phase 4.7.1a                  PASS"
)

print(
    "  historical global checks      PASS"
)


print()
print(
    "Initialization / seed:"
)

print(
    "  Kaiming family                 PASS"
)

print(
    "  distribution = normal          PASS"
)

print(
    "  mode = fan_in                  PASS"
)

print(
    "  nonlinearity = ReLU            PASS"
)

print(
    "  a = 0                          PASS"
)

print(
    "  bias = zero                    PASS"
)

print(
    "  global neural seed = 42        PASS"
)

print(
    "  reference PyTorch = 2.7.0      PASS"
)

print(
    "  canonical init device = CPU    PASS"
)


print()
print(
    "Initialization coverage:"
)

print(
    "  parameter tensors              32"
)

print(
    "  Kaiming tensors                21"
)

print(
    "  zero-bias tensors              11"
)

print(
    "  namespace exact                PASS"
)

print(
    f"  parameter total                "
    f"{initialization_total:,}"
)


print()
print(
    "Initialization reproducibility:"
)

print(
    "  same-seed whole state          PASS"
)

print(
    "  different-seed sensitivity     PASS"
)

print(
    "  initialization artifact hashes PASS"
)


print()
print(
    "Canonical initial-state SHA256:"
)

print(
    f"  {canonical_hash_state}"
)


print()
print(
    "Historical provenance:"
)

print(
    "  4.6.1b topology rewritten      NO"
)

print(
    "  4.7.1b resolves initialization YES"
)


print()
print(
    "Phase-4 model-reconstruction blockers:"
)

print(
    "  NONE"
)


print()
print(
    "Deferred to training/evaluation:"
)


for decision in sorted(
    EXPECTED_DEFERRED_DECISIONS
):

    print(
        f"  - {decision}"
    )


print()
print(
    "Training performed:              NO"
)

print(
    "Frozen architecture changed:     NO"
)

print(
    "Phase 2 reopened:                NO"
)

print(
    "Phase 3 reopened:                NO"
)


print()
print(
    "Outputs:"
)


for path in [

    artifact_hash_recheck_path,

    global_integrity_path,

    metadata_path,

    phase_4_7_closure_path,

    output_hash_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.7.2 STATUS: COMPLETE — "
    "POST-INITIALIZATION MODEL INTEGRITY AUDITED"
)


print()
print("=" * 120)

print(
    "PHASE 4.7 STATUS: COMPLETE — "
    "MODEL INTEGRITY AUDIT CLOSED"
)

print("=" * 120)


print()
print(
    "NEXT:"
)

print(
    "PHASE 4.8 — "
    "PHASE-4 CLOSURE AND HANDOFF"
)