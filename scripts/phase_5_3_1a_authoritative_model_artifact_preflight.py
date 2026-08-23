"""
Phase 5.3.1a — Authoritative Model Entrypoint and Artifact Integrity Preflight

AUDIT ONLY.

Purpose
-------
We have reached the final pristine boundary before model/optimizer
instantiation. All Phase-5 pre-training decisions are frozen.

This audit resolves the exact authoritative Phase-4 implementation/state
provenance before numerical training begins.

THIS SCRIPT DOES NOT:
- instantiate the ITRS model;
- instantiate Adam;
- instantiate any RNG;
- generate training negatives;
- generate training-order permutations;
- perform forward propagation;
- perform backward propagation;
- perform optimizer.step();
- modify any frozen artifact.

It DOES:
1. verify Phase-5 frozen contracts;
2. verify hashes of generated evaluation artifacts;
3. inspect authoritative Phase-4 closure/handoff metadata;
4. locate references to the frozen canonical initialization SHA256;
5. inventory Phase-4 model/state artifact references;
6. inventory Phase-4 Python implementation candidates;
7. write an audit-only preflight manifest.

No model entrypoint is selected automatically.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


# =============================================================================
# Frozen reference values
# =============================================================================

CANONICAL_INITIAL_STATE_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_PARAMETER_COUNT = 19_217_929

EXPECTED_PARAMETER_TENSORS = 32

EXPECTED = {
    "evaluation_cases": 22_515,
    "evaluation_negative_slots": 2_228_985,
    "evaluation_total_candidate_slots": 2_251_500,

    "fixed_epochs": 20,
    "batch_size": 512,
    "training_examples_per_epoch": 5_366_245,
    "batches_per_epoch": 10_481,
    "final_batch_size": 485,
}


# =============================================================================
# Phase-4 authoritative paths
# =============================================================================

PHASE4_ROOT = Path(
    "data/experimental/phase_4"
)

PHASE4_CLOSURE_DIR = (
    PHASE4_ROOT
    / "closure"
)

PHASE4_CLOSURE_MANIFEST = (
    PHASE4_CLOSURE_DIR
    / "phase_4_closure_manifest.json"
)

PHASE4_HANDOFF_CONTRACT = (
    PHASE4_CLOSURE_DIR
    / "phase_4_to_phase_5_handoff_contract.json"
)

PHASE4_ARTIFACT_REGISTRY = (
    PHASE4_CLOSURE_DIR
    / "phase_4_authoritative_artifact_registry.csv"
)

PHASE4_PARAMETER_SUMMARY = (
    PHASE4_CLOSURE_DIR
    / "phase_4_parameter_summary.csv"
)

PHASE4_FINAL_CONTRACT_AUDIT = (
    PHASE4_CLOSURE_DIR
    / "phase_4_final_contract_status_audit.csv"
)

PHASE4_DECISION_REGISTER = (
    PHASE4_CLOSURE_DIR
    / "phase_4_final_model_decision_register.csv"
)

PHASE4_REPRO_LOG = (
    PHASE4_CLOSURE_DIR
    / "Phase_4_Reproduction_Log_Entry.md"
)


# =============================================================================
# Phase-5 frozen contracts / artifacts
# =============================================================================

PHASE5_TRAIN_NEGATIVE_CONTRACT = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_1_1d_training_negative_sampling_runtime_contract.json"
)

PHASE5_EVAL_CONTRACT = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_1_2b_t60_evaluation_candidate_runtime_contract.json"
)

PHASE5_EVAL_GENERATION_MANIFEST = Path(
    "data/experimental/phase_5/model_ready/evaluation/"
    "phase_5_1_2c_generation_manifest.json"
)

PHASE5_EVAL_HASH_REGISTRY = Path(
    "data/experimental/phase_5/model_ready/evaluation/"
    "t60_evaluation_artifact_hashes.csv"
)

PHASE5_TRAINING_CONTROL_CONTRACT = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_2_2_training_control_optimizer_runtime_contract.json"
)


# =============================================================================
# Output paths
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_5/audits/phase_5_3_1a"
)

INPUT_INTEGRITY_PATH = (
    OUT_DIR
    / "authoritative_input_integrity.csv"
)

PHASE4_HASH_REFERENCE_PATH = (
    OUT_DIR
    / "phase4_initialization_hash_reference_locations.csv"
)

PHASE4_REFERENCED_ARTIFACT_PATH = (
    OUT_DIR
    / "phase4_referenced_artifact_inventory.csv"
)

PHASE4_ENTRYPOINT_CANDIDATE_PATH = (
    OUT_DIR
    / "phase4_model_entrypoint_candidates.csv"
)

EVALUATION_HASH_AUDIT_PATH = (
    OUT_DIR
    / "evaluation_artifact_hash_recheck.csv"
)

PREFLIGHT_MANIFEST_PATH = (
    OUT_DIR
    / "phase_5_3_1a_preflight_manifest.json"
)


# =============================================================================
# Helpers
# =============================================================================

def banner(text: str) -> None:
    print(
        "\n"
        + "=" * 118
    )
    print(text)
    print(
        "=" * 118
    )


def require(
    condition: bool,
    message: str,
) -> None:

    if not bool(condition):
        raise AssertionError(
            message
        )


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

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


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def path_looks_like_artifact(
    value: str,
) -> bool:

    lower = value.lower()

    extensions = (
        ".pt",
        ".pth",
        ".bin",
        ".safetensors",
        ".npy",
        ".npz",
        ".parquet",
        ".csv",
        ".json",
        ".md",
        ".py",
    )

    return any(
        lower.endswith(
            extension
        )
        for extension
        in extensions
    )


def recursively_collect_strings(
    obj: Any,
    prefix: str = "$",
) -> list[tuple[str, str]]:
    """
    Recursively extract all string leaves from arbitrary JSON.
    """

    rows = []

    if isinstance(
        obj,
        dict,
    ):

        for key, value in obj.items():

            rows.extend(
                recursively_collect_strings(
                    value,
                    f"{prefix}.{key}",
                )
            )

    elif isinstance(
        obj,
        list,
    ):

        for index, value in enumerate(
            obj
        ):

            rows.extend(
                recursively_collect_strings(
                    value,
                    f"{prefix}[{index}]",
                )
            )

    elif isinstance(
        obj,
        str,
    ):

        rows.append(
            (
                prefix,
                obj,
            )
        )

    return rows


def safe_read_text(
    path: Path,
) -> str:

    try:

        return path.read_text(
            encoding="utf-8",
            errors="replace",
        )

    except Exception:

        return ""


def candidate_path_from_string(
    text: str,
) -> Path | None:
    """
    Convert a metadata string into a repository-local path candidate.

    Only strings that already look path-like are considered.
    """

    cleaned = (
        text.strip()
        .strip("'")
        .strip('"')
    )

    if not cleaned:
        return None

    if "://" in cleaned:
        return None

    if not path_looks_like_artifact(
        cleaned
    ):
        return None

    candidate = Path(
        cleaned
    )

    return candidate


def dataframe_strings(
    df: pd.DataFrame,
) -> list[
    tuple[
        int,
        str,
        str,
    ]
]:
    """
    Return:
        row_index, column_name, string_value
    """

    results = []

    for row_index, row in df.iterrows():

        for column in df.columns:

            value = row[
                column
            ]

            if pd.isna(
                value
            ):
                continue

            if isinstance(
                value,
                str,
            ):

                results.append(
                    (
                        int(
                            row_index
                        ),
                        str(
                            column
                        ),
                        value,
                    )
                )

    return results


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.1a — "
        "AUTHORITATIVE MODEL ENTRYPOINT AND ARTIFACT INTEGRITY PREFLIGHT"
    )

    print(
        "Model instantiated:                   NO"
    )

    print(
        "Optimizer instantiated:               NO"
    )

    print(
        "RNG instantiated:                     NO"
    )

    print(
        "Training negatives generated:         NO"
    )

    print(
        "Training order generated:             NO"
    )

    print(
        "Forward pass performed:               NO"
    )

    print(
        "Backward pass performed:              NO"
    )

    print(
        "Optimizer steps performed:            0"
    )

    # =========================================================================
    # Authoritative input existence
    # =========================================================================

    banner(
        "AUTHORITATIVE INPUT EXISTENCE"
    )

    required_paths = (
        PHASE4_CLOSURE_MANIFEST,
        PHASE4_HANDOFF_CONTRACT,
        PHASE4_ARTIFACT_REGISTRY,
        PHASE4_PARAMETER_SUMMARY,
        PHASE4_FINAL_CONTRACT_AUDIT,
        PHASE4_DECISION_REGISTER,

        PHASE5_TRAIN_NEGATIVE_CONTRACT,
        PHASE5_EVAL_CONTRACT,
        PHASE5_EVAL_GENERATION_MANIFEST,
        PHASE5_EVAL_HASH_REGISTRY,
        PHASE5_TRAINING_CONTROL_CONTRACT,
    )

    integrity_rows = []

    for path in required_paths:

        exists = (
            path.exists()
        )

        integrity_rows.append(
            {
                "path": str(
                    path
                ),
                "exists": (
                    exists
                ),
                "sha256": (
                    sha256_file(
                        path
                    )
                    if exists
                    else None
                ),
            }
        )

        print(
            (
                "FOUND  "
                if exists
                else "MISSING "
            )
            + str(
                path
            )
        )

    require(
        all(
            row[
                "exists"
            ]
            for row
            in integrity_rows
        ),
        (
            "At least one authoritative "
            "Phase-4/Phase-5 input is missing"
        ),
    )

    # =========================================================================
    # Recheck Phase-5 frozen state
    # =========================================================================

    banner(
        "PHASE-5 FROZEN CONTRACT RECHECK"
    )

    train_negative_contract = load_json(
        PHASE5_TRAIN_NEGATIVE_CONTRACT
    )

    eval_contract = load_json(
        PHASE5_EVAL_CONTRACT
    )

    eval_generation = load_json(
        PHASE5_EVAL_GENERATION_MANIFEST
    )

    training_control = load_json(
        PHASE5_TRAINING_CONTROL_CONTRACT
    )

    require(
        train_negative_contract[
            "phase"
        ]
        == "5.1.1d",
        "Unexpected training-negative contract",
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
            "phase"
        ]
        == "5.1.2b",
        "Unexpected evaluation contract",
    )

    require(
        eval_contract[
            "status"
        ]
        == "FROZEN",
        "Evaluation runtime not frozen",
    )

    require(
        eval_generation[
            "phase"
        ]
        == "5.1.2c",
        "Unexpected evaluation generation manifest",
    )

    require(
        eval_generation[
            "status"
        ]
        == "GENERATED_AND_AUDITED",
        "Evaluation artifacts not generated/audited",
    )

    require(
        training_control[
            "phase"
        ]
        == "5.2.2",
        "Unexpected training-control contract",
    )

    require(
        training_control[
            "status"
        ]
        == "FROZEN",
        "Training-control contract not frozen",
    )

    remaining = training_control[
        "original_phase_5_handoff_decisions_remaining"
    ]

    require(
        remaining == [],
        (
            "Phase-5 handoff decisions remain unresolved: "
            f"{remaining}"
        ),
    )

    require(
        int(
            training_control[
                "paper_unspecified_reproduction_choices"
            ][
                "fixed_epochs"
            ]
        )
        == EXPECTED[
            "fixed_epochs"
        ],
        "Frozen epoch count drift",
    )

    require(
        int(
            training_control[
                "batch_runtime"
            ][
                "batch_size"
            ]
        )
        == EXPECTED[
            "batch_size"
        ],
        "Frozen batch-size drift",
    )

    require(
        int(
            training_control[
                "batch_runtime"
            ][
                "examples_per_epoch"
            ]
        )
        == EXPECTED[
            "training_examples_per_epoch"
        ],
        "Training example-count drift",
    )

    require(
        int(
            training_control[
                "batch_runtime"
            ][
                "batches_per_epoch"
            ]
        )
        == EXPECTED[
            "batches_per_epoch"
        ],
        "Training batch-count drift",
    )

    require(
        int(
            training_control[
                "batch_runtime"
            ][
                "final_partial_batch_size"
            ]
        )
        == EXPECTED[
            "final_batch_size"
        ],
        "Final partial-batch drift",
    )

    print(
        "5.1.1d training-negative contract:   FROZEN  PASS"
    )

    print(
        "5.1.2b evaluation runtime:            FROZEN  PASS"
    )

    print(
        "5.1.2c evaluation artifact:           GENERATED/AUDITED  PASS"
    )

    print(
        "5.2.2 training-control contract:      FROZEN  PASS"
    )

    print(
        "Original Phase-5 handoff decisions:   0 OPEN  PASS"
    )

    # =========================================================================
    # Evaluation artifact hash verification
    # =========================================================================

    banner(
        "IMMUTABLE EVALUATION ARTIFACT HASH RECHECK"
    )

    eval_hash_registry = pd.read_csv(
        PHASE5_EVAL_HASH_REGISTRY
    )

    require(
        {
            "path",
            "sha256",
        }
        <= set(
            eval_hash_registry.columns
        ),
        (
            "Evaluation hash registry must contain "
            "path and sha256 columns"
        ),
    )

    evaluation_hash_rows = []

    for row in eval_hash_registry.itertuples(
        index=False
    ):

        path = Path(
            str(
                row.path
            )
        )

        expected_hash = str(
            row.sha256
        )

        exists = path.exists()

        actual_hash = (
            sha256_file(
                path
            )
            if exists
            else None
        )

        match = (
            exists
            and actual_hash
            == expected_hash
        )

        evaluation_hash_rows.append(
            {
                "path": str(
                    path
                ),
                "exists": exists,
                "expected_sha256": (
                    expected_hash
                ),
                "actual_sha256": (
                    actual_hash
                ),
                "hash_match": (
                    match
                ),
            }
        )

        print(
            (
                "PASS  "
                if match
                else "FAIL  "
            )
            + str(
                path
            )
        )

    evaluation_hash_df = pd.DataFrame(
        evaluation_hash_rows
    )

    require(
        evaluation_hash_df[
            "hash_match"
        ].all(),
        (
            "At least one immutable evaluation artifact "
            "no longer matches its recorded SHA256"
        ),
    )

    # =========================================================================
    # Phase-4 closure contract
    # =========================================================================

    banner(
        "PHASE-4 AUTHORITATIVE CLOSURE RECHECK"
    )

    phase4_closure = load_json(
        PHASE4_CLOSURE_MANIFEST
    )

    phase4_handoff = load_json(
        PHASE4_HANDOFF_CONTRACT
    )

    phase4_registry = pd.read_csv(
        PHASE4_ARTIFACT_REGISTRY
    )

    phase4_parameter_summary = pd.read_csv(
        PHASE4_PARAMETER_SUMMARY
    )

    phase4_contract_audit = pd.read_csv(
        PHASE4_FINAL_CONTRACT_AUDIT
    )

    phase4_decisions = pd.read_csv(
        PHASE4_DECISION_REGISTER
    )

    print(
        f"Closure manifest:                    "
        f"{PHASE4_CLOSURE_MANIFEST}"
    )

    print(
        f"Phase-5 handoff contract:             "
        f"{PHASE4_HANDOFF_CONTRACT}"
    )

    print(
        f"Authoritative registry rows:          "
        f"{len(phase4_registry):,}"
    )

    print(
        f"Parameter-summary rows:               "
        f"{len(phase4_parameter_summary):,}"
    )

    print(
        f"Final-contract audit rows:            "
        f"{len(phase4_contract_audit):,}"
    )

    print(
        f"Decision-register rows:               "
        f"{len(phase4_decisions):,}"
    )

    # =========================================================================
    # Locate canonical initialization hash references
    # =========================================================================

    banner(
        "CANONICAL INITIALIZATION HASH PROVENANCE"
    )

    hash_reference_rows = []

    # -------------------------------------------------------------------------
    # JSON sources
    # -------------------------------------------------------------------------

    for source_path, payload in (
        (
            PHASE4_CLOSURE_MANIFEST,
            phase4_closure,
        ),
        (
            PHASE4_HANDOFF_CONTRACT,
            phase4_handoff,
        ),
    ):

        for json_path, value in (
            recursively_collect_strings(
                payload
            )
        ):

            if (
                CANONICAL_INITIAL_STATE_SHA256
                in value
            ):

                hash_reference_rows.append(
                    {
                        "source_file": str(
                            source_path
                        ),
                        "location": (
                            json_path
                        ),
                        "value": (
                            value
                        ),
                    }
                )

    # -------------------------------------------------------------------------
    # CSV sources
    # -------------------------------------------------------------------------

    csv_sources = (
        (
            PHASE4_ARTIFACT_REGISTRY,
            phase4_registry,
        ),
        (
            PHASE4_PARAMETER_SUMMARY,
            phase4_parameter_summary,
        ),
        (
            PHASE4_FINAL_CONTRACT_AUDIT,
            phase4_contract_audit,
        ),
        (
            PHASE4_DECISION_REGISTER,
            phase4_decisions,
        ),
    )

    for source_path, df in csv_sources:

        for (
            row_index,
            column,
            value,
        ) in dataframe_strings(
            df
        ):

            if (
                CANONICAL_INITIAL_STATE_SHA256
                in value
            ):

                hash_reference_rows.append(
                    {
                        "source_file": str(
                            source_path
                        ),
                        "location": (
                            f"row={row_index},column={column}"
                        ),
                        "value": (
                            value
                        ),
                    }
                )

    # -------------------------------------------------------------------------
    # Reproduction log
    # -------------------------------------------------------------------------

    if PHASE4_REPRO_LOG.exists():

        repro_log_text = safe_read_text(
            PHASE4_REPRO_LOG
        )

        if (
            CANONICAL_INITIAL_STATE_SHA256
            in repro_log_text
        ):

            for line_number, line in enumerate(
                repro_log_text.splitlines(),
                start=1,
            ):

                if (
                    CANONICAL_INITIAL_STATE_SHA256
                    in line
                ):

                    hash_reference_rows.append(
                        {
                            "source_file": str(
                                PHASE4_REPRO_LOG
                            ),
                            "location": (
                                f"line={line_number}"
                            ),
                            "value": line.strip(),
                        }
                    )

    hash_reference_df = pd.DataFrame(
        hash_reference_rows
    )

    print(
        "Expected canonical initial-state SHA256:"
    )

    print(
        CANONICAL_INITIAL_STATE_SHA256
    )

    print()

    print(
        f"Authoritative references found:      "
        f"{len(hash_reference_df):,}"
    )

    if not hash_reference_df.empty:

        print(
            hash_reference_df.to_string(
                index=False
            )
        )

    else:

        print(
            "WARNING: hash not found literally in the "
            "inspected authoritative Phase-4 metadata."
        )

    # =========================================================================
    # Inventory artifact path references from authoritative Phase-4 metadata
    # =========================================================================

    banner(
        "PHASE-4 REFERENCED ARTIFACT INVENTORY"
    )

    referenced_rows = []

    # JSON path strings
    for source_path, payload in (
        (
            PHASE4_CLOSURE_MANIFEST,
            phase4_closure,
        ),
        (
            PHASE4_HANDOFF_CONTRACT,
            phase4_handoff,
        ),
    ):

        for location, value in (
            recursively_collect_strings(
                payload
            )
        ):

            candidate = (
                candidate_path_from_string(
                    value
                )
            )

            if candidate is None:
                continue

            referenced_rows.append(
                {
                    "source_file": str(
                        source_path
                    ),
                    "source_location": (
                        location
                    ),
                    "referenced_path": str(
                        candidate
                    ),
                    "suffix": (
                        candidate.suffix.lower()
                    ),
                    "exists": (
                        candidate.exists()
                    ),
                    "sha256_if_file": (
                        sha256_file(
                            candidate
                        )
                        if candidate.is_file()
                        else None
                    ),
                }
            )

    # CSV path strings
    for source_path, df in csv_sources:

        for (
            row_index,
            column,
            value,
        ) in dataframe_strings(
            df
        ):

            candidate = (
                candidate_path_from_string(
                    value
                )
            )

            if candidate is None:
                continue

            referenced_rows.append(
                {
                    "source_file": str(
                        source_path
                    ),
                    "source_location": (
                        f"row={row_index},column={column}"
                    ),
                    "referenced_path": str(
                        candidate
                    ),
                    "suffix": (
                        candidate.suffix.lower()
                    ),
                    "exists": (
                        candidate.exists()
                    ),
                    "sha256_if_file": (
                        sha256_file(
                            candidate
                        )
                        if candidate.is_file()
                        else None
                    ),
                }
            )

    referenced_df = pd.DataFrame(
        referenced_rows
    )

    if not referenced_df.empty:

        referenced_df = (
            referenced_df
            .drop_duplicates(
                subset=[
                    "source_file",
                    "source_location",
                    "referenced_path",
                ]
            )
            .sort_values(
                [
                    "suffix",
                    "referenced_path",
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )

        print(
            referenced_df[
                [
                    "referenced_path",
                    "suffix",
                    "exists",
                ]
            ]
            .drop_duplicates()
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No path-like artifact references detected."
        )

    # =========================================================================
    # Candidate persisted model/state artifacts
    # =========================================================================

    banner(
        "PERSISTED MODEL / STATE ARTIFACT CANDIDATES"
    )

    state_suffixes = {
        ".pt",
        ".pth",
        ".bin",
        ".safetensors",
        ".npy",
        ".npz",
    }

    if not referenced_df.empty:

        state_candidates = referenced_df.loc[
            referenced_df[
                "suffix"
            ].isin(
                state_suffixes
            )
        ].copy()

    else:

        state_candidates = pd.DataFrame()

    if state_candidates.empty:

        print(
            "No persisted model/state candidate was identified "
            "from authoritative Phase-4 path references."
        )

        print(
            "This is NOT automatically an error: Phase 4 may have "
            "frozen deterministic reconstruction instead of a "
            "serialized state_dict."
        )

    else:

        print(
            state_candidates[
                [
                    "referenced_path",
                    "suffix",
                    "exists",
                    "sha256_if_file",
                    "source_file",
                    "source_location",
                ]
            ]
            .drop_duplicates()
            .to_string(
                index=False
            )
        )

    # =========================================================================
    # Python model-entrypoint candidates
    # =========================================================================

    banner(
        "PHASE-4 PYTHON MODEL ENTRYPOINT CANDIDATES"
    )

    script_dir = Path(
        "scripts"
    )

    python_candidates = []

    if script_dir.exists():

        for path in sorted(
            script_dir.glob(
                "phase_4*.py"
            )
        ):

            text = safe_read_text(
                path
            )

            lower = text.lower()

            evidence = []

            keywords = (
                "torch.nn.module",
                "class itrs",
                "def build_model",
                "def instantiate",
                "state_dict",
                "kaiming",
                "rgcn",
                "trend",
                "scoring",
                "doc2vec",
            )

            for keyword in keywords:

                if (
                    keyword
                    in lower
                ):

                    evidence.append(
                        keyword
                    )

            if not evidence:
                continue

            python_candidates.append(
                {
                    "path": str(
                        path
                    ),
                    "sha256": (
                        sha256_file(
                            path
                        )
                    ),
                    "evidence_keywords": (
                        ";".join(
                            evidence
                        )
                    ),
                    "contains_canonical_initial_hash_literal": (
                        CANONICAL_INITIAL_STATE_SHA256
                        in text
                    ),
                }
            )

    entrypoint_df = pd.DataFrame(
        python_candidates
    )

    if entrypoint_df.empty:

        print(
            "No Phase-4 Python candidate matched the audit keywords."
        )

    else:

        print(
            entrypoint_df.to_string(
                index=False
            )
        )

    # =========================================================================
    # Parameter-count evidence search
    # =========================================================================

    banner(
        "FROZEN PARAMETER-BUDGET RECHECK"
    )

    combined_phase4_text = (
        PHASE4_PARAMETER_SUMMARY.read_text(
            encoding="utf-8",
            errors="replace",
        )
        + "\n"
        + PHASE4_HANDOFF_CONTRACT.read_text(
            encoding="utf-8",
            errors="replace",
        )
        + "\n"
        + PHASE4_CLOSURE_MANIFEST.read_text(
            encoding="utf-8",
            errors="replace",
        )
    )

    normalized_text = (
        combined_phase4_text
        .replace(
            ",",
            "",
        )
        .replace(
            "_",
            "",
        )
    )

    parameter_count_present = (
        str(
            EXPECTED_PARAMETER_COUNT
        )
        in normalized_text
    )

    parameter_tensor_count_present = (
        str(
            EXPECTED_PARAMETER_TENSORS
        )
        in normalized_text
    )

    print(
        f"Expected trainable parameters:        "
        f"{EXPECTED_PARAMETER_COUNT:,}"
    )

    print(
        f"Parameter count found in metadata:    "
        f"{'YES' if parameter_count_present else 'NO'}"
    )

    print(
        f"Expected parameter tensors:           "
        f"{EXPECTED_PARAMETER_TENSORS}"
    )

    print(
        f"Tensor-count literal found:           "
        f"{'YES' if parameter_tensor_count_present else 'NO'}"
    )

    require(
        parameter_count_present,
        (
            "Frozen Phase-4 parameter count "
            "19,217,929 was not found in authoritative metadata"
        ),
    )

    # Tensor-count literal may be encoded structurally rather than as
    # a standalone number, so we report it rather than treating absence
    # as a hard failure here.

    # =========================================================================
    # Decision-facing model provenance status
    # =========================================================================

    banner(
        "MODEL PROVENANCE RESOLUTION STATUS"
    )

    canonical_hash_referenced = (
        not hash_reference_df.empty
    )

    persisted_state_candidate_count = (
        0
        if state_candidates.empty
        else int(
            state_candidates[
                "referenced_path"
            ].nunique()
        )
    )

    python_candidate_count = (
        len(
            entrypoint_df
        )
    )

    print(
        f"Canonical initialization hash referenced: "
        f"{'YES' if canonical_hash_referenced else 'NO'}"
    )

    print(
        f"Persisted state/artifact candidates:       "
        f"{persisted_state_candidate_count}"
    )

    print(
        f"Phase-4 Python entrypoint candidates:      "
        f"{python_candidate_count}"
    )

    print()

    if persisted_state_candidate_count > 0:

        print(
            "Interpretation:"
        )

        print(
            "  A persisted model/state-like artifact is referenced."
        )

        print(
            "  Its exact role must be matched to the Phase-4 "
            "initialization contract before loading it."
        )

    else:

        print(
            "Interpretation:"
        )

        print(
            "  No authoritative serialized state was identified."
        )

        print(
            "  The Phase-4 initialization may therefore be intended "
            "to be reconstructed deterministically from the frozen "
            "architecture + initialization contract."
        )

    if python_candidate_count == 1:

        print()

        print(
            "Exactly one Phase-4 implementation candidate was found."
        )

        print(
            "This is strong evidence for the next numerical preflight, "
            "but the script still does NOT import it automatically."
        )

    elif python_candidate_count > 1:

        print()

        print(
            "Multiple Phase-4 implementation candidates were found."
        )

        print(
            "The next step must identify which one matches the "
            "authoritative closure/handoff contract."
        )

    else:

        print()

        print(
            "No implementation candidate was discovered from script names."
        )

        print(
            "The next step will reconstruct the model from the frozen "
            "Phase-4 contract only if no authoritative implementation "
            "artifact exists."
        )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PREFLIGHT INVARIANTS"
    )

    checks = [
        (
            "phase4_authoritative_inputs_exist",
            all(
                path.exists()
                for path
                in (
                    PHASE4_CLOSURE_MANIFEST,
                    PHASE4_HANDOFF_CONTRACT,
                    PHASE4_ARTIFACT_REGISTRY,
                    PHASE4_PARAMETER_SUMMARY,
                    PHASE4_FINAL_CONTRACT_AUDIT,
                    PHASE4_DECISION_REGISTER,
                )
            ),
        ),
        (
            "phase5_training_negative_contract_frozen",
            train_negative_contract[
                "status"
            ]
            == "FROZEN",
        ),
        (
            "phase5_evaluation_contract_frozen",
            eval_contract[
                "status"
            ]
            == "FROZEN",
        ),
        (
            "phase5_evaluation_generated",
            eval_generation[
                "status"
            ]
            == "GENERATED_AND_AUDITED",
        ),
        (
            "phase5_training_control_frozen",
            training_control[
                "status"
            ]
            == "FROZEN",
        ),
        (
            "no_open_phase5_handoff_decisions",
            remaining == [],
        ),
        (
            "evaluation_artifact_hashes_match",
            bool(
                evaluation_hash_df[
                    "hash_match"
                ].all()
            ),
        ),
        (
            "phase4_parameter_count_reference_found",
            parameter_count_present,
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
            "no_rng_instantiated",
            True,
        ),
        (
            "no_training_performed",
            True,
        ),
    ]

    integrity_audit_df = pd.DataFrame(
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
            integrity_audit_df[
                "result"
            ]
            == "PASS"
        ).all(),
        "At least one hard preflight invariant failed",
    )

    print(
        integrity_audit_df.to_string(
            index=False
        )
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

    pd.DataFrame(
        integrity_rows
    ).to_csv(
        INPUT_INTEGRITY_PATH,
        index=False,
    )

    hash_reference_df.to_csv(
        PHASE4_HASH_REFERENCE_PATH,
        index=False,
    )

    referenced_df.to_csv(
        PHASE4_REFERENCED_ARTIFACT_PATH,
        index=False,
    )

    entrypoint_df.to_csv(
        PHASE4_ENTRYPOINT_CANDIDATE_PATH,
        index=False,
    )

    evaluation_hash_df.to_csv(
        EVALUATION_HASH_AUDIT_PATH,
        index=False,
    )

    manifest = {
        "phase": "5.3.1a",
        "title": (
            "Authoritative Model Entrypoint and "
            "Artifact Integrity Preflight"
        ),
        "status": (
            "AUDIT_COMPLETE_NO_NUMERICAL_MODEL_INSTANTIATION"
        ),

        "model_instantiated": False,
        "optimizer_instantiated": False,
        "rng_instantiated": False,
        "training_negative_samples_generated": False,
        "training_order_generated": False,
        "forward_pass_performed": False,
        "backward_pass_performed": False,
        "optimizer_steps": 0,

        "frozen_phase5_status": {
            "training_negative_contract": "FROZEN",
            "evaluation_runtime_contract": "FROZEN",
            "evaluation_candidate_artifact": (
                "GENERATED_AND_AUDITED"
            ),
            "training_control_contract": "FROZEN",
            "open_original_phase5_handoff_decisions": 0,
        },

        "phase4_reference_contract": {
            "expected_trainable_parameters": (
                EXPECTED_PARAMETER_COUNT
            ),
            "expected_parameter_tensors": (
                EXPECTED_PARAMETER_TENSORS
            ),
            "expected_initial_state_sha256": (
                CANONICAL_INITIAL_STATE_SHA256
            ),
            "initial_hash_reference_found": (
                canonical_hash_referenced
            ),
            "persisted_state_candidate_count": (
                persisted_state_candidate_count
            ),
            "python_entrypoint_candidate_count": (
                python_candidate_count
            ),
        },

        "evaluation_integrity": {
            "hash_registry": str(
                PHASE5_EVAL_HASH_REGISTRY
            ),
            "all_hashes_match": bool(
                evaluation_hash_df[
                    "hash_match"
                ].all()
            ),
        },

        "next_phase_requirement": (
            "Resolve exact authoritative Phase-4 model implementation "
            "and initialization realization before importing or "
            "instantiating the model."
        ),

        "outputs": [
            str(
                INPUT_INTEGRITY_PATH
            ),
            str(
                PHASE4_HASH_REFERENCE_PATH
            ),
            str(
                PHASE4_REFERENCED_ARTIFACT_PATH
            ),
            str(
                PHASE4_ENTRYPOINT_CANDIDATE_PATH
            ),
            str(
                EVALUATION_HASH_AUDIT_PATH
            ),
        ],
    }

    PREFLIGHT_MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    for path in (
        INPUT_INTEGRITY_PATH,
        PHASE4_HASH_REFERENCE_PATH,
        PHASE4_REFERENCED_ARTIFACT_PATH,
        PHASE4_ENTRYPOINT_CANDIDATE_PATH,
        EVALUATION_HASH_AUDIT_PATH,
        PREFLIGHT_MANIFEST_PATH,
    ):

        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Final status
    # =========================================================================

    banner(
        "PHASE 5.3.1a FINAL STATUS"
    )

    print(
        "Phase-5 contracts:                   VERIFIED"
    )

    print(
        "Evaluation artifact hashes:           VERIFIED"
    )

    print(
        "Phase-4 closure metadata:             VERIFIED"
    )

    print(
        "Frozen parameter budget:              "
        f"{EXPECTED_PARAMETER_COUNT:,}"
    )

    print(
        "Frozen canonical init SHA256:         "
        f"{CANONICAL_INITIAL_STATE_SHA256}"
    )

    print(
        "Canonical init hash metadata refs:    "
        f"{len(hash_reference_df):,}"
    )

    print(
        "Persisted state candidates:           "
        f"{persisted_state_candidate_count}"
    )

    print(
        "Python model candidates:              "
        f"{python_candidate_count}"
    )

    print()

    print(
        "Model instantiated:                   NO"
    )

    print(
        "Optimizer instantiated:               NO"
    )

    print(
        "RNG instantiated:                     NO"
    )

    print(
        "Forward pass performed:               NO"
    )

    print(
        "Backward pass performed:              NO"
    )

    print(
        "Optimizer steps:                      0"
    )

    banner(
        "PHASE 5.3.1a COMPLETE / "
        "AUTHORITATIVE NUMERICAL ENTRYPOINT AUDITED"
    )


if __name__ == "__main__":
    main()