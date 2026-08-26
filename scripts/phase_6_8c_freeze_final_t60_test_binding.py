#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TEST_CASES = 20_264
CANDIDATES_PER_CASE = 100
NEGATIVES_PER_CASE = 99

EXPECTED_CHECKPOINT_SHA = (
    "a86d1e3ecb8058be747d2f289414dbf0"
    "bddf779feb2fb7fdbc1cba8cfd3bd4b2"
)

CHECKPOINT_PATH = Path(
    "/workspace/phase6_10pct_best_epoch2.pt"
)

SELECTION_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_8a_final_training_budget_selection.json"
)

RANKING_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_3a_validation_ranking_metric_contract.json"
)

PREFLIGHT_RUNTIME_PATH = Path(
    "scripts/"
    "phase_5_3_3b_"
    "canonical_real_validation_scoring_preflight.py"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_6/final_test"
)

TEST_BINDING_PATH = (
    OUTPUT_DIR
    / "final_t60_test_case_binding.parquet"
)

TEST_CANDIDATE_PATH = (
    OUTPUT_DIR
    / "final_t60_test_candidate_startup_local.npy"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_8c_final_t60_test_binding_contract.json"
)


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def file_sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def load_json(path):
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    require(
        spec is not None
        and spec.loader is not None,
        f"Could not load {path}",
    )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module

    spec.loader.exec_module(
        module
    )

    return module


def main():

    print("=" * 100)
    print(
        "PHASE 6.8c — "
        "FREEZE FINAL T60 TEST CASE BINDING"
    )
    print("=" * 100)

    for path in (
        CHECKPOINT_PATH,
        SELECTION_CONTRACT_PATH,
        RANKING_CONTRACT_PATH,
        PREFLIGHT_RUNTIME_PATH,
    ):
        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

    # ------------------------------------------------------------------
    # Final selected model identity
    # ------------------------------------------------------------------

    selection = load_json(
        SELECTION_CONTRACT_PATH
    )

    require(
        selection["status"]
        == "FINAL_TRAINING_BUDGET_SELECTION_FROZEN",
        "Final budget selection is not frozen.",
    )

    selected = (
        selection[
            "selected_configuration"
        ]
    )

    require(
        selected[
            "training_budget"
        ]
        == "10pct",
        "Selected budget drift.",
    )

    require(
        int(
            selected[
                "best_epoch"
            ]
        )
        == 2,
        "Selected epoch drift.",
    )

    checkpoint_sha = file_sha256(
        CHECKPOINT_PATH
    )

    require(
        checkpoint_sha
        == EXPECTED_CHECKPOINT_SHA,
        "Frozen final checkpoint SHA drift.",
    )

    require(
        selected[
            "checkpoint_sha256"
        ]
        == EXPECTED_CHECKPOINT_SHA,
        "Selection contract checkpoint SHA drift.",
    )

    # ------------------------------------------------------------------
    # Frozen Phase-5 evaluation artifacts
    # ------------------------------------------------------------------

    ranking = load_json(
        RANKING_CONTRACT_PATH
    )

    require(
        ranking["status"] == "FROZEN",
        "Phase-5 ranking contract not frozen.",
    )

    negative_artifact = (
        ranking[
            "frozen_artifacts"
        ][
            "negative_matrix"
        ]
    )

    case_artifact = (
        ranking[
            "frozen_artifacts"
        ][
            "case_manifest"
        ]
    )

    negative_path = Path(
        negative_artifact[
            "path"
        ]
    )

    case_path = Path(
        case_artifact[
            "path"
        ]
    )

    require(
        file_sha256(
            negative_path
        )
        == negative_artifact[
            "file_sha256"
        ],
        "Frozen negative-matrix hash drift.",
    )

    require(
        file_sha256(
            case_path
        )
        == case_artifact[
            "file_sha256"
        ],
        "Frozen case-manifest hash drift.",
    )

    # ------------------------------------------------------------------
    # Load frozen binding runtime
    # ------------------------------------------------------------------

    preflight = load_module(
        PREFLIGHT_RUNTIME_PATH,
        "_phase6_8c_preflight",
    )

    negative_raw = np.load(
        negative_path,
        mmap_mode="r",
    )

    require(
        negative_raw.shape
        == (
            22_515,
            NEGATIVES_PER_CASE,
        ),
        "Frozen negative matrix shape drift.",
    )

    coordinate = (
        preflight
        .infer_negative_matrix_coordinate(
            negative_raw
        )
    )

    negative_local = (
        preflight
        .normalize_negative_matrix_to_local(
            negative_raw,
            coordinate,
        )
    )

    case_manifest = pd.read_parquet(
        case_path
    )

    require(
        len(case_manifest) == 22_515,
        "Case manifest row-count drift.",
    )

    (
        resolved_cases,
        binding_metadata,
    ) = (
        preflight
        .resolve_case_bindings(
            case_manifest,
            ranking,
        )
    )

    require(
        int(
            (
                resolved_cases[
                    "split"
                ]
                == "validation"
            ).sum()
        )
        == 2_251,
        "Resolved validation count drift.",
    )

    test_cases = (
        resolved_cases.loc[
            resolved_cases[
                "split"
            ]
            == "test"
        ]
        .sort_values(
            "matrix_row_index",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    require(
        len(test_cases)
        == TEST_CASES,
        "Resolved test count drift.",
    )

    require(
        test_cases[
            "matrix_row_index"
        ].is_unique,
        "Test matrix_row_index not unique.",
    )

    # ------------------------------------------------------------------
    # Construct final immutable test candidate sets
    # Positive is position 0, followed by frozen 99 negatives.
    # ------------------------------------------------------------------

    candidate_matrix = np.empty(
        (
            TEST_CASES,
            CANDIDATES_PER_CASE,
        ),
        dtype=np.int64,
    )

    binding_rows = []

    for test_position, row in (
        test_cases.iterrows()
    ):

        matrix_row_index = int(
            row[
                "matrix_row_index"
            ]
        )

        positive_local = int(
            row[
                "positive_startup_local"
            ]
        )

        negatives_local = np.asarray(
            negative_local[
                matrix_row_index
            ],
            dtype=np.int64,
        )

        require(
            negatives_local.shape
            == (
                NEGATIVES_PER_CASE,
            ),
            "Test case negative count drift.",
        )

        require(
            not bool(
                np.any(
                    negatives_local
                    == positive_local
                )
            ),
            (
                "Positive startup appears "
                "inside frozen negatives."
            ),
        )

        require(
            len(
                np.unique(
                    negatives_local
                )
            )
            == NEGATIVES_PER_CASE,
            "Duplicate frozen negatives in test case.",
        )

        candidates = np.concatenate(
            [
                np.asarray(
                    [
                        positive_local,
                    ],
                    dtype=np.int64,
                ),
                negatives_local,
            ]
        )

        require(
            len(
                np.unique(
                    candidates
                )
            )
            == CANDIDATES_PER_CASE,
            (
                "Final test candidate set "
                "does not contain 100 "
                "unique startups."
            ),
        )

        candidate_matrix[
            test_position
        ] = candidates

        binding_rows.append(
            {
                "test_case_position":
                    int(test_position),

                "matrix_row_index":
                    matrix_row_index,

                "interaction_id":
                    str(
                        row[
                            "interaction_id"
                        ]
                    ),

                "investor_global":
                    int(
                        row[
                            "investor_global"
                        ]
                    ),

                "positive_startup_local":
                    positive_local,

                "candidate_count":
                    CANDIDATES_PER_CASE,

                "negative_count":
                    NEGATIVES_PER_CASE,

                "candidate_position_of_positive":
                    0,

                "split":
                    "test",
            }
        )

    binding_df = pd.DataFrame(
        binding_rows
    )

    require(
        len(binding_df)
        == TEST_CASES,
        "Final binding count drift.",
    )

    require(
        candidate_matrix.shape
        == (
            TEST_CASES,
            CANDIDATES_PER_CASE,
        ),
        "Final candidate-matrix shape drift.",
    )

    # ------------------------------------------------------------------
    # Logical fingerprints
    # ------------------------------------------------------------------

    binding_sha = (
        preflight
        .dataframe_logical_sha256(
            binding_df,
            columns=[
                "test_case_position",
                "matrix_row_index",
                "interaction_id",
                "investor_global",
                "positive_startup_local",
                "candidate_count",
                "negative_count",
            ],
        )
    )

    candidate_sha = (
        preflight
        .array_logical_sha256(
            candidate_matrix
        )
    )

    # ------------------------------------------------------------------
    # Persist frozen final-test inputs
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    binding_df.to_parquet(
        TEST_BINDING_PATH,
        index=False,
    )

    np.save(
        TEST_CANDIDATE_PATH,
        candidate_matrix,
        allow_pickle=False,
    )

    contract = {
        "phase": "6.8c",
        "status":
            "FINAL_T60_TEST_BINDING_FROZEN",

        "selected_model": {
            "training_budget":
                "10pct",
            "best_epoch":
                2,
            "checkpoint_path":
                str(CHECKPOINT_PATH),
            "checkpoint_sha256":
                checkpoint_sha,
        },

        "source_artifacts": {
            "case_manifest": {
                "path":
                    str(case_path),
                "file_sha256":
                    case_artifact[
                        "file_sha256"
                    ],
            },

            "negative_matrix": {
                "path":
                    str(negative_path),
                "file_sha256":
                    negative_artifact[
                        "file_sha256"
                    ],
                "coordinate_detected":
                    coordinate,
            },
        },

        "final_test_binding": {
            "test_cases":
                TEST_CASES,
            "candidates_per_case":
                CANDIDATES_PER_CASE,
            "negatives_per_case":
                NEGATIVES_PER_CASE,

            "positive_candidate_position":
                0,

            "binding_path":
                str(TEST_BINDING_PATH),

            "candidate_matrix_path":
                str(TEST_CANDIDATE_PATH),

            "binding_logical_sha256":
                binding_sha,

            "candidate_matrix_logical_sha256":
                candidate_sha,

            "binding_file_sha256":
                file_sha256(
                    TEST_BINDING_PATH
                ),

            "candidate_matrix_file_sha256":
                file_sha256(
                    TEST_CANDIDATE_PATH
                ),
        },

        "evaluation_policy": {
            "ranking_score":
                "raw_model_logit",

            "ranking_order":
                "descending_score",

            "tie_break":
                "startup_local_ascending",

            "metrics": [
                "HR@10",
                "NDCG@10",
            ],

            "model_inference_performed":
                False,

            "test_metrics_computed":
                False,

            "optimizer_instantiated":
                False,

            "backward_performed":
                False,
        },

        "binding_metadata":
            binding_metadata,
    }

    CONTRACT_PATH.write_text(
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    print()
    print(
        f"Resolved test cases:             "
        f"{len(binding_df):,}"
    )

    print(
        f"Candidate matrix shape:          "
        f"{candidate_matrix.shape}"
    )

    print(
        f"Candidates / case:               "
        f"{CANDIDATES_PER_CASE}"
    )

    print(
        f"Negative matrix coordinate:      "
        f"{coordinate}"
    )

    print(
        "Positive position:               0"
    )

    print(
        "Duplicate candidate sets:        "
        "NONE WITHIN CASE"
    )

    print(
        "Checkpoint SHA:                  PASS"
    )

    print(
        "Frozen source hashes:            PASS"
    )

    print()
    print(
        "Binding logical SHA256:"
    )
    print(binding_sha)

    print()
    print(
        "Candidate matrix logical SHA256:"
    )
    print(candidate_sha)

    print()
    print(
        f"WROTE  {TEST_BINDING_PATH}"
    )
    print(
        f"WROTE  {TEST_CANDIDATE_PATH}"
    )
    print(
        f"WROTE  {CONTRACT_PATH}"
    )

    print()
    print(
        "MODEL INFERENCE PERFORMED:       NO"
    )
    print(
        "TEST METRICS COMPUTED:           NO"
    )

    print()
    print(
        "PHASE 6.8c: PASS / "
        "FINAL T60 TEST BINDING FROZEN / "
        "ZERO MODEL SCORING"
    )


if __name__ == "__main__":
    main()
