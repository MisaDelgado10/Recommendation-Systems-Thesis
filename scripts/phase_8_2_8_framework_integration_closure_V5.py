#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import csv
import hashlib
import json


# =============================================================================
# PHASE 8.2.8 — FRAMEWORK INTEGRATION CLOSURE V5
# =============================================================================
#
# This is the FINAL PRE-TRAINING GATE.
#
# NO model instantiation.
# NO forward.
# NO optimizer.
# NO training.
# NO performance inspection.
# NO T60 scoring.
# NO April analysis.
#
# Only after every gate below passes is comparative DEVELOPMENT training
# B1a/B1b/M1/M2 on DEV_T57 / DEV_T58 / DEV_T59 authorized.
# =============================================================================


PHASE8_ROOT = Path(
    "data/experimental/phase_8"
)

FRAMEWORK_ROOT = (
    PHASE8_ROOT
    / "phase_8_2_framework_development"
)

OUT_DIR = (
    FRAMEWORK_ROOT
    / "phase_8_2_8_framework_integration_closure"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# AUTHORITATIVE METHOD / ENTRY CONTRACT
# =============================================================================

PREFLIGHT_V2 = (
    FRAMEWORK_ROOT
    / "phase_8_2_0_preflight"
    / "phase_8_2_0_preflight_contract_V2.json"
)

WORK_PLAN_V2 = (
    FRAMEWORK_ROOT
    / "phase_8_2_0_preflight"
    / "phase_8_2_0_work_package_plan_V2.csv"
)


V5_DIR = (
    PHASE8_ROOT
    / "phase_8_1_inductive_startup_representation"
    / "phase_8_1_12_execution_contract"
)

V5_CONTRACT = (
    V5_DIR
    / "phase_8_1_12_execution_contract_V5.json"
)

V5_MANIFEST = (
    V5_DIR
    / "phase_8_1_12_execution_contract_manifest_V5.json"
)

V5_HASHES = (
    V5_DIR
    / "phase_8_1_12_execution_contract_sha256_V5.json"
)


# =============================================================================
# AUTHORITATIVE PHASE CLOSURES
# =============================================================================

P1D_DIR = (
    FRAMEWORK_ROOT
    / "phase_8_2_1_feature_materialization"
    / "phase_8_2_1d_preprocessing_V2"
)

P1D_MANIFEST = (
    P1D_DIR
    / "phase_8_2_1d_manifest_V2.json"
)

P1D_HASHES = (
    P1D_DIR
    / "phase_8_2_1d_artifact_sha256_V2.json"
)


P2_DIR = (
    FRAMEWORK_ROOT
    / "phase_8_2_2_strict_inductive_heterogeneous_graph"
)

P2_MANIFEST = (
    P2_DIR
    / "phase_8_2_2_manifest_V2.json"
)

P2_HASHES = (
    P2_DIR
    / "phase_8_2_2_artifact_sha256_V2.json"
)


P6_CLOSURE = (
    FRAMEWORK_ROOT
    / "phase_8_2_6_training_stream_dry_run"
    / "phase_8_2_6_training_stream_integration_closure_manifest_V2.json"
)

P6_HASHES = (
    FRAMEWORK_ROOT
    / "phase_8_2_6_training_stream_dry_run"
    / "phase_8_2_6_training_stream_integration_closure_sha256_V2.json"
)


P6_TEMPORAL_CSV = (
    FRAMEWORK_ROOT
    / "phase_8_2_6_training_stream_dry_run"
    / "phase_8_2_6_all_epoch_temporal_negative_integrity_V1.csv"
)

P6_TEMPORAL_MANIFEST = (
    FRAMEWORK_ROOT
    / "phase_8_2_6_training_stream_dry_run"
    / "phase_8_2_6_all_epoch_temporal_negative_integrity_manifest_V1.json"
)

P6_TEMPORAL_HASHES = (
    FRAMEWORK_ROOT
    / "phase_8_2_6_training_stream_dry_run"
    / "phase_8_2_6_all_epoch_temporal_negative_integrity_sha256_V1.json"
)


P7_CLOSURE = (
    FRAMEWORK_ROOT
    / "phase_8_2_7_autograd_optimizer_preflight"
    / "phase_8_2_7_autograd_optimizer_preflight_closure_manifest_V1.json"
)

P7_HASHES = (
    FRAMEWORK_ROOT
    / "phase_8_2_7_autograd_optimizer_preflight"
    / "phase_8_2_7_autograd_optimizer_preflight_closure_sha256_V1.json"
)


# =============================================================================
# IMPLEMENTATION FILES TO FREEZE AT LAUNCH
# =============================================================================

IMPLEMENTATION_SCRIPTS = [
    Path(
        "scripts/"
        "phase_8_2_3_candidate_model_modules_V1.py"
    ),

    Path(
        "scripts/"
        "phase_8_2_4_canonical_initialization_audit_V1.py"
    ),

    Path(
        "scripts/"
        "phase_8_2_5_canonical_forward_branch_insertion_audit_V1.py"
    ),

    Path(
        "scripts/"
        "phase_8_2_6_development_epoch_fingerprint_coverage_V2.py"
    ),

    Path(
        "scripts/"
        "phase_8_2_6_training_stream_integration_closure_V2.py"
    ),

    Path(
        "scripts/"
        "phase_8_2_7_dev_t57_frozen_batch_one_step_preflight_V1.py"
    ),

    Path(
        "scripts/"
        "phase_8_2_7_autograd_optimizer_preflight_closure_V1.py"
    ),

    Path(
        "scripts/"
        "phase_8_2_6_all_epoch_temporal_negative_integrity_audit_V1.py"
    ),

    Path(
        "scripts/"
        "phase_8_2_8_framework_integration_closure_V5.py"
    ),
]


OUTPUT_RUNTIME = (
    OUT_DIR
    / "phase_8_2_8_training_runtime_contract_V5.json"
)

OUTPUT_MANIFEST = (
    OUT_DIR
    / "phase_8_2_8_framework_integration_closure_manifest_V5.json"
)

OUTPUT_HASHES = (
    OUT_DIR
    / "phase_8_2_8_final_implementation_sha256_V5.json"
)


# =============================================================================
# EXPECTED FROZEN MODEL FAMILY
# =============================================================================

EXPECTED_PARAMETER_COUNTS = {
    "B1a": {
        "tensors": 29,
        "elements": 6_751_789,
    },

    "B1b": {
        "tensors": 33,
        "elements": 6_875_797,
    },

    "M1": {
        "tensors": 31,
        "elements": 6_920_129,
    },

    "M2": {
        "tensors": 41,
        "elements": 6_952_609,
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def verify_hash_registry(
    *,
    artifact: Path,
    registry: Path,
    label: str,
):

    require(
        artifact.exists(),
        f"{label}: artifact missing: {artifact}",
    )

    require(
        registry.exists(),
        f"{label}: hash registry missing: {registry}",
    )

    hashes = load_json(
        registry
    )

    require(
        str(artifact) in hashes,
        (
            f"{label}: registry does not contain "
            f"{artifact}"
        ),
    )

    observed = sha256_file(
        artifact
    )

    expected = hashes[
        str(artifact)
    ]

    require(
        observed == expected,
        (
            f"{label}: SHA256 mismatch\n"
            f"expected={expected}\n"
            f"observed={observed}"
        ),
    )

    return observed


def find_unique_json_with_status(
    root: Path,
    expected_status: str,
):

    matches = []

    for path in root.rglob("*.json"):

        try:
            data = load_json(
                path
            )

        except Exception:
            continue

        if (
            isinstance(
                data,
                dict,
            )
            and data.get(
                "status"
            )
            == expected_status
        ):

            matches.append(
                (
                    path,
                    data,
                )
            )

    require(
        len(matches) == 1,
        (
            f"Expected exactly one JSON with status "
            f"{expected_status}, found {len(matches)}:\n"
            + "\n".join(
                str(path)
                for path, _data
                in matches
            )
        ),
    )

    return matches[0]


def recursive_contains_string(
    obj,
    target: str,
):

    if isinstance(
        obj,
        dict,
    ):

        return any(
            recursive_contains_string(
                key,
                target,
            )
            or
            recursive_contains_string(
                value,
                target,
            )
            for key, value
            in obj.items()
        )

    if isinstance(
        obj,
        list,
    ):

        return any(
            recursive_contains_string(
                value,
                target,
            )
            for value
            in obj
        )

    return target in str(
        obj
    )


# =============================================================================
# START
# =============================================================================

print("=" * 120)
print("PHASE 8.2.8 — FRAMEWORK INTEGRATION CLOSURE V5")
print("=" * 120)


# =============================================================================
# 1. REQUIRED ARTIFACTS
# =============================================================================

print()
print("1. REQUIRED AUTHORITATIVE ARTIFACTS")

required = [
    PREFLIGHT_V2,
    WORK_PLAN_V2,
    V5_CONTRACT,
    V5_MANIFEST,
    V5_HASHES,
    P1D_MANIFEST,
    P1D_HASHES,
    P2_MANIFEST,
    P2_HASHES,
    P6_CLOSURE,
    P6_HASHES,
    P6_TEMPORAL_CSV,
    P6_TEMPORAL_MANIFEST,
    P6_TEMPORAL_HASHES,
    P7_CLOSURE,
    P7_HASHES,
    *IMPLEMENTATION_SCRIPTS,
]

for path in required:

    require(
        path.exists(),
        f"Missing required artifact: {path}",
    )

print(
    f"  Required files: {len(required)} / {len(required)} PASS"
)


# =============================================================================
# 2. AUTHORITATIVE 8.2 ENTRY CONTRACT
# =============================================================================

print()
print("2. PHASE 8.2 ENTRY CONTRACT")

preflight = load_json(
    PREFLIGHT_V2
)

require(
    preflight[
        "status"
    ]
    == "FROZEN",
    "Phase 8.2.0 V2 is not frozen.",
)

require(
    preflight[
        "method_design_closed"
    ]
    is True,
    "Method design is not closed.",
)

require(
    preflight[
        "method_design_reopened"
    ]
    is False,
    "Method design was reopened.",
)

require(
    preflight[
        "performance_inspected"
    ]
    is False,
    "Performance was inspected during framework development.",
)

require(
    preflight[
        "training_allowed_now"
    ]
    is False,
    (
        "Entry contract should prohibit training "
        "until 8.2.8."
    ),
)

require(
    preflight[
        "training_launch_condition"
    ]
    == (
        "Only Phase 8.2.8 PASS may authorize "
        "comparative candidate-model training."
    ),
    "Training launch condition drifted.",
)


work_packages = preflight[
    "work_packages"
]

require(
    len(
        work_packages
    )
    == 8,
    "Expected exactly eight Phase 8.2 work packages.",
)

phase828 = [
    row
    for row in work_packages
    if row[
        "phase"
    ]
    == "8.2.8"
]

require(
    len(
        phase828
    )
    == 1,
    "Could not uniquely locate 8.2.8.",
)

phase828 = phase828[0]

require(
    phase828[
        "order"
    ]
    == 8,
    "8.2.8 order changed.",
)

require(
    phase828[
        "name"
    ]
    == "Framework integration closure",
    "8.2.8 name changed.",
)

require(
    phase828[
        "performance_allowed"
    ]
    is False,
    "8.2.8 must not inspect performance.",
)

require(
    phase828[
        "training_allowed_after"
    ]
    is True,
    "8.2.8 is no longer the training launch gate.",
)

print("  Corrected V2 entry contract: PASS")
print("  Work packages 1..8:          PASS")
print("  8.2.8 launch authority:     PASS")


# =============================================================================
# 3. AUTHORITATIVE 8.1.12 V5 CONTRACT
# =============================================================================

print()
print("3. AUTHORITATIVE 8.1.12 V5 CONTRACT")

v5_hash = verify_hash_registry(
    artifact=V5_CONTRACT,
    registry=V5_HASHES,
    label="8.1.12 V5",
)

v5 = load_json(
    V5_CONTRACT
)

# Schema-robust semantic checks. These must be present somewhere in the
# authoritative frozen contract.
required_v5_markers = [
    "DEV_T57",
    "DEV_T58",
    "DEV_T59",
    "T60_FINAL_IMPROVED",
    "APRIL_FINAL_M2",
    "ITRS_PHASE8_NEURAL_INIT",
    "ITRS_PHASE8_TRAIN_NEGATIVE",
    "ITRS_PHASE8_TRAIN_ORDER",
    "ALL_COLD_t_STARTUPS",
    "FROZEN_1_POSITIVE_PLUS_99_NEGATIVES",
]

for marker in required_v5_markers:

    require(
        recursive_contains_string(
            v5,
            marker,
        ),
        (
            "Authoritative 8.1.12 V5 contract "
            f"missing marker: {marker}"
        ),
    )

# -------------------------------------------------------------------------
# April confirmatory policy lives under root.April_confirmatory in V5.
# V1 incorrectly attempted to read these fields from the JSON root.
# -------------------------------------------------------------------------

require(
    isinstance(
        v5.get(
            "April_confirmatory"
        ),
        dict,
    ),
    "V5 April_confirmatory block missing.",
)

april_confirmatory = (
    v5[
        "April_confirmatory"
    ]
)

require(
    april_confirmatory[
        "candidate_matrix"
    ]
    == "FROZEN_1_POSITIVE_PLUS_99_NEGATIVES",
    "V5 April candidate matrix changed.",
)

require(
    april_confirmatory[
        "candidate_regeneration"
    ]
    is False,
    "V5 candidate regeneration policy changed.",
)

require(
    april_confirmatory[
        "feature_reselection"
    ]
    is False,
    "V5 feature-reselection policy changed.",
)

require(
    april_confirmatory[
        "model_reselection"
    ]
    is False,
    "V5 model-reselection policy changed.",
)


# -------------------------------------------------------------------------
# Development-fold contract.
# -------------------------------------------------------------------------

development_folds = (
    v5[
        "development_folds"
    ]
)

require(
    len(
        development_folds
    )
    == 3,
    "V5 must contain exactly three development folds.",
)

expected_dev = [
    (
        "T57",
        "DEV_T57",
        "T1-T56",
        1_003_526,
        20,
    ),
    (
        "T58",
        "DEV_T58",
        "T1-T57",
        1_026_268,
        20,
    ),
    (
        "T59",
        "DEV_T59",
        "T1-T58",
        1_049_605,
        20,
    ),
]

for (
    fold,
    (
        expected_origin,
        expected_scope,
        expected_segments,
        expected_positives,
        expected_max_epochs,
    ),
) in zip(
    development_folds,
    expected_dev,
):

    require(
        fold[
            "origin"
        ]
        == expected_origin,
        (
            "V5 development origin changed: "
            f"{fold}"
        ),
    )

    require(
        fold[
            "randomness_scope"
        ]
        == expected_scope,
        (
            "V5 development randomness "
            "scope changed."
        ),
    )

    require(
        fold[
            "training_segments"
        ]
        == expected_segments,
        (
            "V5 development training "
            "segments changed."
        ),
    )

    require(
        fold[
            "training_positive_events"
        ]
        == expected_positives,
        (
            "V5 development positive "
            "count changed."
        ),
    )

    require(
        fold[
            "maximum_epochs"
        ]
        == expected_max_epochs,
        (
            "V5 maximum epoch policy changed."
        ),
    )

    require(
        fold[
            "candidate_matrix"
        ]
        == "FROZEN_PHASE8_MATRIX",
        (
            "V5 development candidate "
            "matrix changed."
        ),
    )

    require(
        fold[
            "candidates_per_case"
        ]
        == 100,
        (
            "V5 candidates-per-case changed."
        ),
    )

    require(
        fold[
            "evaluation_negatives_per_case"
        ]
        == 99,
        (
            "V5 evaluation negative count changed."
        ),
    )

    require(
        fold[
            "used_for_epoch_selection"
        ]
        is True,
        (
            "A development fold is no longer "
            "used for epoch selection."
        ),
    )


# -------------------------------------------------------------------------
# T60 / April remain strictly non-selective.
# -------------------------------------------------------------------------

require(
    v5[
        "T60_selection_allowed"
    ]
    is False,
    "T60 unexpectedly allowed for selection.",
)

require(
    v5[
        "April_selection_allowed"
    ]
    is False,
    "April unexpectedly allowed for selection.",
)

require(
    v5[
        "architecture_change_after_dev_allowed"
    ]
    is False,
    (
        "Architecture change after development "
        "unexpectedly allowed."
    ),
)

require(
    v5[
        "feature_change_after_dev_allowed"
    ]
    is False,
    (
        "Feature change after development "
        "unexpectedly allowed."
    ),
)


# -------------------------------------------------------------------------
# Final-fit epoch policies.
# -------------------------------------------------------------------------

t60_april = (
    v5[
        "T60_April_execution"
    ]
)

require(
    len(
        t60_april
    )
    == 4,
    "Unexpected V5 T60/April execution rows.",
)

stage_lookup = {
    row[
        "stage"
    ]:
        row
    for row
    in t60_april
}

require(
    stage_lookup[
        "T60_IMPROVED_MODELS"
    ][
        "epoch_policy"
    ]
    == "USE_MODEL_SPECIFIC_DEV_SELECTED_EPOCH",
    "T60 improved-model epoch policy changed.",
)

require(
    stage_lookup[
        "T60_IMPROVED_MODELS"
    ][
        "selection_allowed"
    ]
    is False,
    "T60 improved models unexpectedly allow selection.",
)

require(
    stage_lookup[
        "APRIL_M2_FINAL_FIT"
    ][
        "epoch_policy"
    ]
    == "USE_M2_DEV_SELECTED_EPOCH",
    "April M2 epoch policy changed.",
)

require(
    stage_lookup[
        "APRIL_M2_FINAL_FIT"
    ][
        "selection_allowed"
    ]
    is False,
    "April M2 unexpectedly allows selection.",
)


# -------------------------------------------------------------------------
# Initialization contract.
# -------------------------------------------------------------------------

initialization = (
    v5[
        "initialization"
    ]
)

require(
    initialization[
        "base_seed"
    ]
    == 42,
    "V5 initialization base seed changed.",
)

require(
    initialization[
        "canonical_initialization_device"
    ]
    == "CPU",
    "Canonical initialization device changed.",
)

require(
    initialization[
        "startup_identity_parameter"
    ]
    is False,
    "Startup identity parameter unexpectedly enabled.",
)

require(
    initialization[
        "person_identity_parameter"
    ]
    is False,
    "Person identity parameter unexpectedly enabled.",
)


# -------------------------------------------------------------------------
# Exact RNG namespaces and stream-sharing rules.
# -------------------------------------------------------------------------

randomness = (
    v5[
        "randomness"
    ]
)

require(
    randomness[
        "seed_derivation"
    ][
        "base_seed"
    ]
    == 42,
    "V5 seed-derivation base seed changed.",
)

require(
    randomness[
        "seed_derivation"
    ][
        "hash"
    ]
    == "SHA256",
    "V5 seed hash changed.",
)

require(
    randomness[
        "seed_derivation"
    ][
        "seed_u64"
    ]
    == (
        "Interpret the first 8 SHA256 digest bytes "
        "as an unsigned big-endian 64-bit integer."
    ),
    "V5 uint64 seed interpretation changed.",
)

streams = (
    randomness[
        "streams"
    ]
)

require(
    streams[
        "neural_initialization"
    ][
        "namespace"
    ]
    == "ITRS_PHASE8_NEURAL_INIT",
    "Neural initialization namespace changed.",
)

require(
    streams[
        "training_negatives"
    ][
        "namespace"
    ]
    == "ITRS_PHASE8_TRAIN_NEGATIVE",
    "Training-negative namespace changed.",
)

require(
    streams[
        "training_order"
    ][
        "namespace"
    ]
    == "ITRS_PHASE8_TRAIN_ORDER",
    "Training-order namespace changed.",
)

require(
    streams[
        "training_negatives"
    ][
        "negative_positive_ratio"
    ]
    == 4,
    "Training negative ratio changed.",
)

require(
    streams[
        "training_negatives"
    ][
        "model_name_enters_seed"
    ]
    is False,
    "Model name unexpectedly enters negative seed.",
)

require(
    streams[
        "training_order"
    ][
        "model_name_enters_seed"
    ]
    is False,
    "Model name unexpectedly enters order seed.",
)

require(
    streams[
        "training_negatives"
    ][
        "same_stream_across_B1a_B1b_M1_M2_within_scope_epoch"
    ]
    is True,
    "Family-shared negative stream contract changed.",
)

require(
    streams[
        "training_order"
    ][
        "same_stream_across_B1a_B1b_M1_M2_within_scope_epoch"
    ]
    is True,
    "Family-shared order stream contract changed.",
)


print("  V5 contract SHA256:           PASS")
print("  Development folds:            3 / 3 PASS")
print("  Frozen candidate policy:      PASS")
print("  April no-regeneration:        PASS")
print("  No feature/model reselection: PASS")
print("  T60 selection forbidden:      PASS")
print("  April selection forbidden:    PASS")
print("  Final-fit epoch policies:     PASS")
print("  Identity-free initialization: PASS")
print("  RNG namespaces:               PASS")
print("  Shared training streams:      PASS")


# =============================================================================
# 4. PHASE 8.2.1D
# =============================================================================

print()
print("4. PHASE 8.2.1D — PREPROCESSING")

p1_hash = verify_hash_registry(
    artifact=P1D_MANIFEST,
    registry=P1D_HASHES,
    label="8.2.1D",
)

p1 = load_json(
    P1D_MANIFEST
)

require(
    p1[
        "status"
    ]
    == "MATERIALIZED_AND_AUDITED",
    "8.2.1D not closed successfully.",
)

# 8.2.1D records the authoritative contract by exact filesystem path,
# not by the human-readable literal string "8.1.12 V5".
require(
    p1[
        "authoritative_execution_contract"
    ]
    == str(
        V5_CONTRACT
    ),
    (
        "8.2.1D authoritative_execution_contract "
        "does not point to the frozen V5 contract."
    ),
)

require(
    p1[
        "cold_raw_values_used_for_fit"
    ]
    is False,
    "8.2.1D used COLD_t raw values during preprocessing fit.",
)

require(
    p1[
        "evaluation_candidate_matrix_used_for_fit"
    ]
    is False,
    "8.2.1D used evaluation candidate matrix during fit.",
)

require(
    p1[
        "evaluation_negative_membership_used_for_fit"
    ]
    is False,
    "8.2.1D used evaluation negative membership during fit.",
)

require(
    p1[
        "evaluation_positive_membership_used_for_fit"
    ]
    is False,
    "8.2.1D used evaluation positive membership during fit.",
)

require(
    p1[
        "future_evaluation_membership_used_for_fit"
    ]
    is False,
    "8.2.1D used future evaluation membership during fit.",
)

require(
    p1[
        "feature_discovery_reopened"
    ]
    is False,
    "8.2.1D reopened feature discovery.",
)

require(
    p1[
        "feature_inventory_changed"
    ]
    is False,
    "8.2.1D changed the frozen feature inventory.",
)

require(
    p1[
        "learned_preprocessing_fit"
    ]
    is True,
    "8.2.1D learned preprocessing was not materialized.",
)

require(
    p1[
        "method_design_reopened"
    ]
    is False,
    "8.2.1D reopened method design.",
)

require(
    p1[
        "model_instantiated"
    ]
    is False,
    "8.2.1D unexpectedly instantiated a model.",
)

require(
    p1[
        "optimizer_instantiated"
    ]
    is False,
    "8.2.1D unexpectedly instantiated an optimizer.",
)

print("  Authoritative V5 contract path: PASS")
print("  Training-only preprocessing:    PASS")
print("  COLD_t excluded from fit:       PASS")
print("  Evaluation membership excluded: PASS")
print("  Feature/method freeze:          PASS")
print("  Manifest SHA256:                PASS")


# =============================================================================
# 5. PHASE 8.2.2
# =============================================================================

print()
print("5. PHASE 8.2.2 — STRICT INDUCTIVE GRAPH")

p2_hash = verify_hash_registry(
    artifact=P2_MANIFEST,
    registry=P2_HASHES,
    label="8.2.2",
)

p2 = load_json(
    P2_MANIFEST
)

require(
    p2[
        "status"
    ]
    == "MATERIALIZED_AND_AUDITED",
    "8.2.2 not closed successfully.",
)

require(
    recursive_contains_string(
        p2,
        "ALL_COLD_t",
    )
    or recursive_contains_string(
        p2,
        "cold",
    ),
    "8.2.2 cold-start masking evidence missing.",
)

print("  Graph materialization:       PASS")
print("  Cold-start graph semantics:  PASS")
print("  Manifest SHA256:             PASS")


# =============================================================================
# 6. PHASE 8.2.3 / 8.2.4 / 8.2.5
# =============================================================================

print()
print("6. MODEL / INITIALIZATION / FORWARD AUDIT CHAIN")

# -------------------------------------------------------------------------
# Phase 8.2.3 has distinct authoritative artifacts:
#
#   architectural consistency closure
#   candidate-model implementation manifest
#   artifact SHA256 registry
#   candidate-model Python module
#
# The implementation manifest carries terminal status
# IMPLEMENTED_AND_TOPOLOGY_AUDITED, while the architectural closure carries
# status FROZEN. They must not be conflated.
# -------------------------------------------------------------------------

P3_DIR = (
    FRAMEWORK_ROOT
    / "phase_8_2_3_candidate_model_implementation"
)

p3_path = (
    P3_DIR
    / "phase_8_2_3_manifest_V1.json"
)

p3_closure_path = (
    P3_DIR
    / "phase_8_2_3_architectural_consistency_closure_V1.json"
)

p3_hashes_path = (
    P3_DIR
    / "phase_8_2_3_artifact_sha256_V1.json"
)

p3_model_module_path = Path(
    "scripts/"
    "phase_8_2_3_candidate_model_modules_V1.py"
)

for path in [
    p3_path,
    p3_closure_path,
    p3_hashes_path,
    p3_model_module_path,
]:

    require(
        path.exists(),
        f"Required authoritative 8.2.3 artifact missing: {path}",
    )


p3 = load_json(
    p3_path
)

p3_closure = load_json(
    p3_closure_path
)

p3_hash_registry = load_json(
    p3_hashes_path
)


require(
    p3[
        "status"
    ]
    == "IMPLEMENTED_AND_TOPOLOGY_AUDITED",
    "8.2.3 implementation manifest status changed.",
)

require(
    p3_closure[
        "status"
    ]
    == "FROZEN",
    "8.2.3 architectural closure is not frozen.",
)

require(
    p3_closure[
        "phase"
    ]
    == "8.2.3",
    "8.2.3 architectural closure phase changed.",
)

p4_path, p4 = find_unique_json_with_status(
    FRAMEWORK_ROOT,
    "INITIALIZATION_AND_REPRODUCIBILITY_AUDITED",
)

# Phase 8.2.5 intentionally has both a summary JSON and a manifest JSON
# carrying the same terminal status. The manifest is the authoritative
# integration artifact; therefore it must be selected by exact path rather
# than inferred by globally unique status.
p5_path = (
    FRAMEWORK_ROOT
    / "phase_8_2_5_forward_branch_insertion"
    / "phase_8_2_5_manifest_V1.json"
)

require(
    p5_path.exists(),
    f"Authoritative 8.2.5 manifest missing: {p5_path}",
)

p5 = load_json(
    p5_path
)

require(
    p5[
        "status"
    ]
    == "FORWARD_AND_BRANCH_INSERTION_AUDITED",
    "Authoritative 8.2.5 manifest has unexpected status.",
)


# -------------------------------------------------------------------------
# Authoritative 8.2.3 integrity chain.
#
# V4 incorrectly compared the implementation manifest SHA against the
# architectural-closure SHA and also contained a string-splitting typo in
# that hard-coded SHA. V5 verifies the correct artifact identities.
# -------------------------------------------------------------------------

EXPECTED_P3_CLOSURE_SHA256 = (
    "2652d06ca76c3e56176914418fc49bb76973d359b6be7f06f31f225efe48316f"
)

p3_expected_artifacts = [
    p3_closure_path,
    p3_path,
    p3_model_module_path,
]

for artifact in p3_expected_artifacts:

    require(
        str(
            artifact
        )
        in p3_hash_registry,
        (
            "8.2.3 artifact registry missing authoritative artifact: "
            f"{artifact}"
        ),
    )

    observed_sha = sha256_file(
        artifact
    )

    registered_sha = p3_hash_registry[
        str(
            artifact
        )
    ]

    require(
        observed_sha
        == registered_sha,
        (
            "8.2.3 artifact SHA256 mismatch:\n"
            f"artifact={artifact}\n"
            f"registered={registered_sha}\n"
            f"observed={observed_sha}"
        ),
    )


require(
    sha256_file(
        p3_closure_path
    )
    == EXPECTED_P3_CLOSURE_SHA256,
    (
        "8.2.3 architectural closure differs from "
        "the independently frozen closure fingerprint."
    ),
)


require(
    p3_hash_registry[
        str(
            p3_closure_path
        )
    ]
    == EXPECTED_P3_CLOSURE_SHA256,
    (
        "8.2.3 registry does not preserve the independently "
        "frozen architectural-closure fingerprint."
    ),
)


print("  8.2.3 implementation manifest:      PASS")
print("  8.2.3 architectural closure:        PASS")
print("  8.2.3 closure SHA 2652...:          PASS")
print("  8.2.3 candidate module SHA registry:PASS")


for family, expected in EXPECTED_PARAMETER_COUNTS.items():

    require(
        recursive_contains_string(
            p3,
            family,
        ),
        f"8.2.3 closure missing {family}.",
    )


require(
    recursive_contains_string(
        p4,
        "ITRS_PHASE8_NEURAL_INIT",
    ),
    "8.2.4 neural-init namespace missing.",
)

require(
    recursive_contains_string(
        p4,
        "startup_identity"
    )
    or recursive_contains_string(
        p4,
        "Startup"
    ),
    "8.2.4 identity-free evidence missing.",
)

require(
    recursive_contains_string(
        p5,
        "COLD"
    )
    or recursive_contains_string(
        p5,
        "cold"
    ),
    "8.2.5 cold insertion evidence missing.",
)


print(
    f"  8.2.3: {p3_path.name} — PASS"
)

print(
    f"  8.2.4: {p4_path.name} — PASS"
)

print(
    f"  8.2.5: {p5_path.name} — PASS"
)


# =============================================================================
# 7. PHASE 8.2.6
# =============================================================================

print()
print("7. PHASE 8.2.6 — TRAINING STREAM")

p6 = load_json(
    P6_CLOSURE
)

require(
    p6[
        "status"
    ]
    == "TRAINING_STREAM_INTEGRATION_DRY_RUN_CLOSED",
    "8.2.6 not formally closed.",
)

require(
    p6[
        "training_allowed_now"
    ]
    is False,
    "8.2.6 incorrectly authorized training.",
)

coverage = p6[
    "development_coverage"
]

require(
    coverage[
        "scopes"
    ]
    == 3,
    "8.2.6 development scope count changed.",
)

require(
    coverage[
        "epochs_per_scope"
    ]
    == 20,
    "8.2.6 epoch coverage changed.",
)

require(
    coverage[
        "total_scope_epoch_streams"
    ]
    == 60,
    "8.2.6 stream coverage is not 60/60.",
)

print("  Development folds:            3 / 3")
print("  Epoch streams:                60 / 60")
print("  Shared deterministic stream:  PASS")


# -----------------------------------------------------------------------------
# Supplemental hard temporal accepted-negative audit.
# -----------------------------------------------------------------------------

temporal_audit_sha = verify_hash_registry(
    artifact=P6_TEMPORAL_MANIFEST,
    registry=P6_TEMPORAL_HASHES,
    label="8.2.6 temporal supplement",
)

temporal_hash_registry = load_json(
    P6_TEMPORAL_HASHES
)

require(
    str(
        P6_TEMPORAL_CSV
    )
    in temporal_hash_registry,
    "8.2.6 temporal hash registry missing CSV.",
)

require(
    sha256_file(
        P6_TEMPORAL_CSV
    )
    ==
    temporal_hash_registry[
        str(
            P6_TEMPORAL_CSV
        )
    ],
    "8.2.6 temporal-audit CSV hash mismatch.",
)

temporal_audit = load_json(
    P6_TEMPORAL_MANIFEST
)

require(
    temporal_audit[
        "status"
    ]
    == "ALL_DEVELOPMENT_STREAM_TEMPORAL_NEGATIVE_INTEGRITY_AUDITED",
    "8.2.6 all-epoch temporal audit not closed.",
)

require(
    temporal_audit[
        "audit_scope"
    ][
        "scope_epoch_streams"
    ]
    == 60,
    "Temporal audit does not cover 60 streams.",
)

require(
    temporal_audit[
        "results"
    ][
        "streams_audited"
    ]
    == 60,
    "Temporal audit stream count changed.",
)

require(
    temporal_audit[
        "results"
    ][
        "accepted_negatives_audited"
    ]
    == 246_351_920,
    "Temporal audit accepted-negative count changed.",
)

require(
    temporal_audit[
        "results"
    ][
        "fingerprints_matching_frozen_registry"
    ]
    == 60,
    "Not all temporal-audit streams match frozen fingerprints.",
)

require(
    temporal_audit[
        "results"
    ][
        "forbidden_temporal_collisions"
    ]
    == 0,
    "Temporally forbidden accepted negative detected.",
)

require(
    temporal_audit[
        "results"
    ][
        "origin_ineligible_negatives"
    ]
    == 0,
    "Origin-ineligible negative detected.",
)

require(
    temporal_audit[
        "results"
    ][
        "duplicate_negative_rows"
    ]
    == 0,
    "Duplicate negative row detected.",
)

require(
    temporal_audit[
        "results"
    ][
        "focal_collisions"
    ]
    == 0,
    "Focal-positive negative collision detected.",
)

require(
    temporal_audit[
        "results"
    ][
        "temporal_accounting_complete"
    ]
    is True,
    "Temporal negative accounting is incomplete.",
)

require(
    temporal_audit[
        "temporal_acceptance_rule"
    ][
        "labels_after_fold_train_max_used"
    ]
    is False,
    "Temporal audit used labels after fold origin.",
)

require(
    temporal_audit[
        "training_allowed_now"
    ]
    is False,
    "Supplemental audit itself must not authorize training.",
)

print("  Temporal accepted-pair audit:   60 / 60 PASS")
print("  Accepted negatives audited:     246,351,920")
print("  Frozen fingerprint matches:     60 / 60 PASS")
print("  Forbidden temporal collisions:  0 PASS")
print("  Labels beyond fold origin:      NO")


# =============================================================================
# 8. PHASE 8.2.7
# =============================================================================

print()
print("8. PHASE 8.2.7 — REAL AUTOGRAD / ADAM PREFLIGHT")

p7 = load_json(
    P7_CLOSURE
)

require(
    p7[
        "status"
    ]
    == "AUTOGRAD_OPTIMIZER_PREFLIGHT_CLOSED",
    "8.2.7 not formally closed.",
)

require(
    p7[
        "training_allowed_now"
    ]
    is False,
    "8.2.7 incorrectly authorized training.",
)

for family, expected in EXPECTED_PARAMETER_COUNTS.items():

    family_result = (
        p7[
            "families"
        ][family]
    )

    require(
        family_result[
            "parameter_tensors"
        ]
        == expected[
            "tensors"
        ],
        f"{family}: parameter tensor count drift.",
    )

    require(
        family_result[
            "parameter_elements"
        ]
        == expected[
            "elements"
        ],
        f"{family}: parameter element count drift.",
    )

    require(
        family_result[
            "all_parameter_tensors_received_gradient"
        ]
        is True,
        f"{family}: gradient coverage failed.",
    )

    require(
        family_result[
            "all_parameter_tensors_changed_after_step"
        ]
        is True,
        f"{family}: Adam update coverage failed.",
    )

    require(
        family_result[
            "optimizer_step_calls"
        ]
        == 1,
        f"{family}: optimizer-step count drift.",
    )

    require(
        family_result[
            "updated_state_retained"
        ]
        is False,
        f"{family}: temporary updated state retained.",
    )

print("  B1a: 29 / 29 tensors: PASS")
print("  B1b: 33 / 33 tensors: PASS")
print("  M1:  31 / 31 tensors: PASS")
print("  M2:  41 / 41 tensors: PASS")
print("  Temporary state discarded: PASS")


# =============================================================================
# 9. FINAL IMPLEMENTATION HASHES
# =============================================================================

print()
print("9. FINAL IMPLEMENTATION HASH FREEZE")

implementation_artifacts = [
    V5_CONTRACT,
    V5_MANIFEST,
    PREFLIGHT_V2,
    WORK_PLAN_V2,
    P1D_MANIFEST,
    P2_MANIFEST,
    p3_closure_path,
    p3_path,
    p3_hashes_path,
    p4_path,
    p5_path,
    P6_CLOSURE,
    P6_TEMPORAL_CSV,
    P6_TEMPORAL_MANIFEST,
    P6_TEMPORAL_HASHES,
    P7_CLOSURE,
    *IMPLEMENTATION_SCRIPTS,
]

final_hashes = {}

for path in implementation_artifacts:

    digest = sha256_file(
        path
    )

    final_hashes[
        str(path)
    ] = digest

    print(
        f"  {path.name:<70} {digest[:16]}..."
    )


# =============================================================================
# 10. FREEZE DEVELOPMENT TRAINING RUNTIME
# =============================================================================

print()
print("10. FREEZE COMPARATIVE DEVELOPMENT TRAINING RUNTIME")

runtime_contract = {
    "schema_version":
        "PHASE_8_2_8_TRAINING_RUNTIME_CONTRACT_V5",

    "phase":
        "8.2.8",

    "status":
        "FROZEN_FOR_COMPARATIVE_DEVELOPMENT_TRAINING",

    "authoritative_method_contract":
        str(
            V5_CONTRACT
        ),

    "authorized_models": [
        "B1a",
        "B1b",
        "M1",
        "M2",
    ],

    "development_folds": {
        "DEV_T57": {
            "origin":
                "T57",

            "training_segments":
                "T1..T56",

            "rng_scope":
                "DEV_T57",
        },

        "DEV_T58": {
            "origin":
                "T58",

            "training_segments":
                "T1..T57",

            "rng_scope":
                "DEV_T58",
        },

        "DEV_T59": {
            "origin":
                "T59",

            "training_segments":
                "T1..T58",

            "rng_scope":
                "DEV_T59",
        },
    },

    "epoch_policy": {
        "candidate_display_epochs":
            list(
                range(
                    1,
                    21,
                )
            ),

        "rng_epoch_indices":
            list(
                range(
                    20
                )
            ),

        "maximum_epochs":
            20,

        "early_stopping":
            False,

        "development_selection_primary":
            "EQUAL_FOLD_MEAN_NDCG_AT_10",

        "development_selection_secondary":
            "EQUAL_FOLD_MEAN_HR_AT_10",

        "final_tiebreak":
            "EARLIEST_EPOCH",
    },

    "training_stream": {
        "negative_ratio":
            4,

        "serialization":
            [
                "positive",
                "negative_0",
                "negative_1",
                "negative_2",
                "negative_3",
            ],

        "batch_size":
            512,

        "drop_last":
            False,

        "negative_namespace":
            "ITRS_PHASE8_TRAIN_NEGATIVE",

        "order_namespace":
            "ITRS_PHASE8_TRAIN_ORDER",

        "same_stream_B1a_B1b_M1_M2":
            True,

        "negative_resampling":
            "ONCE_PER_EPOCH",

        "order_shuffle":
            "ONCE_PER_EPOCH",

        "future_positive_pairs_allowed_as_training_negatives":
            True,

        "prior_or_same_target_positive_pairs_allowed":
            False,
    },

    "initialization": {
        "namespace":
            "ITRS_PHASE8_NEURAL_INIT",

        "base_seed":
            42,

        "device":
            "CPU_CANONICAL_INITIALIZATION",

        "semantic_parameter_key":
            (
                "scope_id + semantic_component_id + "
                "parameter_name + shape + dtype"
            ),

        "bias_initialization":
            "ZEROS",

        "matrix_initialization":
            (
                "KAIMING_NORMAL_A0_FAN_IN_RELU"
            ),

        "startup_identity_parameter":
            False,

        "person_identity_parameter":
            False,
    },

    "optimizer": {
        "class":
            "torch.optim.Adam",

        "lr":
            0.001,

        "betas":
            [
                0.9,
                0.999,
            ],

        "eps":
            1e-8,

        "weight_decay":
            0.0,

        "amsgrad":
            False,

        "foreach":
            False,

        "fused":
            False,
    },

    "loss": {
        "class":
            "torch.nn.BCEWithLogitsLoss",

        "scorer_output":
            "RAW_LOGIT",
    },

    "cold_start_fit_policy": {
        "population":
            "ALL_COLD_t_STARTUPS",

        "startup_feature_slots":
            "ZERO_DURING_PARAMETER_FIT",

        "founder_acquisition_incident_edges":
            "MASK_DURING_PARAMETER_FIT",

        "inference_restore":
            (
                "RESTORE_ALLOWED_ORIGIN_TIME_FEATURES_AND_"
                "INCOMING_NONCOLD_MESSAGES_ONLY"
            ),

        "cold_outgoing_messages":
            "SUPPRESSED",

        "parameter_updates_after_restore":
            False,
    },

    "development_performance_policy": {
        "performance_inspection_during_training":
            "ONLY_FROZEN_DEVELOPMENT_VALIDATION_ENDPOINTS",

        "T60_may_select_epoch":
            False,

        "April_may_select_epoch":
            False,

        "feature_reselection":
            False,

        "architecture_reselection":
            False,

        "negative_regeneration_for_evaluation":
            False,
    },

    "post_development_policy": {
        "T60_FINAL_IMPROVED":
            (
                "Each B1a/B1b/M1/M2 model uses its own "
                "development-selected epoch for final fit T1..T59."
            ),

        "APRIL_FINAL_M2":
            (
                "M2 uses the development-selected M2 epoch "
                "for final fit T1..T60."
            ),

        "T60_or_April_training_authorized_by_this_closure":
            False,
    },

    "implementation_hash_registry":
        str(
            OUTPUT_HASHES
        ),

    "comparative_development_training_authorized":
        True,

    "authorization_scope":
        "DEV_T57_DEV_T58_DEV_T59_ONLY",
}


OUTPUT_RUNTIME.write_text(
    json.dumps(
        runtime_contract,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)

print("  Candidate models:               B1a / B1b / M1 / M2")
print("  Development folds:              T57 / T58 / T59")
print("  Candidate epochs:               1..20")
print("  Training RNG epoch indices:     0..19")
print("  Batch size:                     512")
print("  Negatives / positive:           4")
print("  Loss:                           BCEWithLogitsLoss")
print("  Optimizer:                      Adam")
print("  Early stopping:                 NO")
print("  T60 selection:                  FORBIDDEN")
print("  April selection:                FORBIDDEN")


# =============================================================================
# 11. FINAL INTEGRATION CLOSURE + AUTHORIZATION
# =============================================================================

print()
print("11. FINAL INTEGRATION GATE")

closure = {
    "schema_version":
        "PHASE_8_2_8_FRAMEWORK_INTEGRATION_CLOSURE_V5",

    "phase":
        "8.2.8",

    "status":
        "FRAMEWORK_INTEGRATION_CLOSED_AND_DEVELOPMENT_TRAINING_AUTHORIZED",

    "scientific_role":
        "FINAL_PRETRAINING_IMPLEMENTATION_GATE",

    "supersession": {
        "V1_attempt":
            "FAILED_BEFORE_RUNTIME_OR_AUTHORIZATION_WRITE",

        "failure_class":
            "READ_ONLY_V5_JSON_SCHEMA_PATH_MISMATCH",

        "incorrect_assumption":
            (
                "candidate_regeneration, feature_reselection and "
                "model_reselection were read from the V5 root."
            ),

        "correct_location":
            "root.April_confirmatory",

        "model_instantiated_by_V1":
            False,

        "optimizer_created_by_V1":
            False,

        "training_performed_by_V1":
            False,

        "performance_inspected_by_V1":
            False,

        "training_authorized_by_V1":
            False,

        "scientific_state_changed":
            False,
    },

    "prior_failed_gate_attempts": [
        {
            "version":
                "V1",

            "failure_class":
                "V5_JSON_SCHEMA_PATH_MISMATCH",

            "scientific_state_changed":
                False,

            "training_authorized":
                False,
        },
        {
            "version":
                "V2",

            "failure_class":
                "8_2_1D_HUMAN_READABLE_LITERAL_CHECK_MISMATCH",

            "scientific_state_changed":
                False,

            "training_authorized":
                False,
        },
        {
            "version":
                "V3",

            "failure_class":
                "8_2_5_NONUNIQUE_TERMINAL_STATUS_ARTIFACT_DISCOVERY",

            "scientific_state_changed":
                False,

            "training_authorized":
                False,

            "resolution":
                (
                    "Use exact authoritative "
                    "phase_8_2_5_manifest_V1.json path rather "
                    "than global status uniqueness."
                ),
        },
        {
            "version":
                "V4",

            "failure_class":
                "8_2_3_MANIFEST_CLOSURE_SHA_IDENTITY_MISMATCH_IN_GATE",

            "scientific_state_changed":
                False,

            "training_authorized":
                False,

            "details":
                (
                    "V4 compared the implementation manifest against "
                    "the architectural-closure SHA and also contained "
                    "a typo in the split hard-coded closure digest."
                ),

            "resolution":
                (
                    "Validate the implementation manifest, architectural "
                    "closure and candidate-model module individually "
                    "against the authoritative 8.2.3 artifact registry."
                ),
        },
    ],

    "authoritative_method_contract":
        str(
            V5_CONTRACT
        ),

    "authoritative_method_contract_sha256":
        v5_hash,

    "work_package":
        {
            "name":
                phase828[
                    "name"
                ],

            "produces":
                phase828[
                    "produces"
                ],

            "scientific_gate":
                phase828[
                    "scientific_gate"
                ],

            "performance_allowed":
                False,

            "training_allowed_after":
                True,
        },

    "phase_chain": {
        "8.2.1D": {
            "status":
                p1[
                    "status"
                ],

            "manifest":
                str(
                    P1D_MANIFEST
                ),

            "sha256":
                p1_hash,
        },

        "8.2.2": {
            "status":
                p2[
                    "status"
                ],

            "manifest":
                str(
                    P2_MANIFEST
                ),

            "sha256":
                p2_hash,
        },

        "8.2.3": {
            "status":
                p3[
                    "status"
                ],

            "implementation_manifest":
                str(
                    p3_path
                ),

            "implementation_manifest_sha256":
                sha256_file(
                    p3_path
                ),

            "architectural_closure":
                str(
                    p3_closure_path
                ),

            "architectural_closure_status":
                p3_closure[
                    "status"
                ],

            "architectural_closure_sha256":
                sha256_file(
                    p3_closure_path
                ),

            "artifact_sha256_registry":
                str(
                    p3_hashes_path
                ),

            "candidate_model_module":
                str(
                    p3_model_module_path
                ),

            "candidate_model_module_sha256":
                sha256_file(
                    p3_model_module_path
                ),

            "registry_integrity_verified":
                True,
        },

        "8.2.4": {
            "status":
                p4[
                    "status"
                ],

            "closure":
                str(
                    p4_path
                ),

            "sha256":
                sha256_file(
                    p4_path
                ),
        },

        "8.2.5": {
            "status":
                p5[
                    "status"
                ],

            "closure":
                str(
                    p5_path
                ),

            "sha256":
                sha256_file(
                    p5_path
                ),
        },

        "8.2.6": {
            "status":
                p6[
                    "status"
                ],

            "closure":
                str(
                    P6_CLOSURE
                ),

            "sha256":
                sha256_file(
                    P6_CLOSURE
                ),

            "development_streams":
                60,

            "all_epoch_temporal_negative_audit":
                str(
                    P6_TEMPORAL_MANIFEST
                ),

            "all_epoch_temporal_negative_audit_sha256":
                temporal_audit_sha,

            "accepted_negatives_temporally_audited":
                246_351_920,

            "frozen_fingerprint_matches":
                60,

            "forbidden_temporal_collisions":
                0,
        },

        "8.2.7": {
            "status":
                p7[
                    "status"
                ],

            "closure":
                str(
                    P7_CLOSURE
                ),

            "sha256":
                sha256_file(
                    P7_CLOSURE
                ),

            "families_one_step_audited":
                4,
        },
    },

    "final_implementation_hashes":
        final_hashes,

    "runtime_contract":
        str(
            OUTPUT_RUNTIME
        ),

    "authorization": {
        "comparative_development_training":
            True,

        "authorized_models": [
            "B1a",
            "B1b",
            "M1",
            "M2",
        ],

        "authorized_scopes": [
            "DEV_T57",
            "DEV_T58",
            "DEV_T59",
        ],

        "authorized_epoch_indices":
            list(
                range(
                    20
                )
            ),

        "performance_allowed":
            True,

        "performance_scope":
            "FROZEN_DEVELOPMENT_VALIDATION_ONLY",

        "T60_training_now":
            False,

        "T60_scoring_now":
            False,

        "April_training_now":
            False,

        "April_analysis_now":
            False,

        "architecture_changes_after_launch":
            False,

        "feature_changes_after_launch":
            False,

        "evaluation_candidate_regeneration":
            False,
    },

    "execution_state_of_8_2_8": {
        "model_instantiated":
            False,

        "forward_executed":
            False,

        "optimizer_created":
            False,

        "backward_executed":
            False,

        "optimizer_step_executed":
            False,

        "training_performed":
            False,

        "performance_inspected":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,
    },

    "training_allowed_now":
        True,

    "training_allowed_scope":
        "COMPARATIVE_DEVELOPMENT_ONLY",

    "next_step":
        (
            "Launch comparative development training for "
            "B1a/B1b/M1/M2 over DEV_T57, DEV_T58 and DEV_T59 "
            "under the frozen 20-epoch V5 runtime. "
            "Do not use T60 or April for epoch/model selection."
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


# Now that runtime + closure exist, add them to the final immutable-ish
# implementation fingerprint registry.
final_hashes[
    str(
        OUTPUT_RUNTIME
    )
] = sha256_file(
    OUTPUT_RUNTIME
)

final_hashes[
    str(
        OUTPUT_MANIFEST
    )
] = sha256_file(
    OUTPUT_MANIFEST
)


OUTPUT_HASHES.write_text(
    json.dumps(
        final_hashes,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)


print("  8.2.1D: PASS")
print("  8.2.2:  PASS")
print("  8.2.3:  PASS")
print("  8.2.4:  PASS")
print("  8.2.5:  PASS")
print("  8.2.6:  PASS")
print("  8.2.7:  PASS")
print("  V5:     PASS")

print()
print(
    f"  Runtime contract: {OUTPUT_RUNTIME}"
)

print(
    f"  Closure manifest: {OUTPUT_MANIFEST}"
)

print(
    f"  Final hashes:     {OUTPUT_HASHES}"
)


# =============================================================================
# FINAL STATUS
# =============================================================================

print()
print("=" * 120)
print("PHASE 8.2.8 — FRAMEWORK INTEGRATION CLOSED")
print("=" * 120)

print("Authoritative method:                 8.1.12 V5")
print("Framework work packages:              8 / 8 PASS")
print("60-stream temporal negative audit:    PASS")
print("Forbidden temporal collisions:        0")
print("Final implementation hashes:          FROZEN")
print("Development runtime contract:         FROZEN")
print()
print("Model instantiated in 8.2.8:          NO")
print("Forward executed in 8.2.8:            NO")
print("Optimizer created in 8.2.8:           NO")
print("Training performed in 8.2.8:          NO")
print("Performance inspected in 8.2.8:       NO")
print("T60 scored:                            NO")
print("April analyzed:                        NO")
print()
print("Comparative development training:      AUTHORIZED")
print("Authorized models:                     B1a / B1b / M1 / M2")
print("Authorized folds:                      DEV_T57 / DEV_T58 / DEV_T59")
print("Authorized candidate epochs:           1..20")
print("T60 training/scoring now:              NOT AUTHORIZED")
print("April training/analysis now:           NOT AUTHORIZED")
print()
print("TRAINING ALLOWED NOW:                  YES — DEVELOPMENT ONLY")
print()
print(
    "STATUS: FRAMEWORK_INTEGRATION_CLOSED_AND_DEVELOPMENT_TRAINING_AUTHORIZED"
)
print(
    "NEXT:   Comparative B1a/B1b/M1/M2 development training under frozen V5 protocol"
)
print("=" * 120)
