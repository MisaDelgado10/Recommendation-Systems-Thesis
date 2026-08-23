"""
Phase 5.3.3c — Full Validation Split Runtime Dry-Run and Freeze

Purpose
-------
Execute one complete REAL validation pass over all 2,251 frozen validation
events using the canonical seed-42 initial model, while scoring ZERO test cases.

This phase freezes the production validation runtime before it is integrated
into the 20-epoch training driver.

Inherited semantics
-------------------
Candidate set per event:
    1 true startup + 99 frozen negatives = 100 candidates

Ranking:
    raw model logit descending
    tie-break startup_local ascending
    positive rank is 1-based

Metrics:
    HR@10
    NDCG@10
    arithmetic mean over event-level cases

T60 trend:
    original TrendExtractor.encode_sequence
    exactly T0..T59
    NO training adapter
    NO post-T59 padding

Bounded validation execution
----------------------------
To preserve the exact Phase-5.3.3b regression anchor while bounding dense
description-feature memory:

    chunk 0:
        validation cases 0..15
        size = 16

    remaining chunks:
        size <= 64

The first 16 cases MUST reproduce exactly the frozen Phase-5.3.3b:
    selected-case SHA256
    candidate-matrix SHA256
    raw-logit matrix SHA256
    case-metric SHA256

Structural F_s for all 477,564 nodes is computed ONCE for the full validation
pass. Description/trend features are materialized only for the current chunk.

Safety boundary
---------------
- canonical INITIAL seed-42 model only
- model.eval()
- torch.no_grad()
- optimizer instantiated: NO
- backward: NO
- optimizer.step(): 0
- test cases scored: 0
- validation metrics used for checkpoint selection: NO
- canonical parameter state must remain byte-identical
- Python / NumPy / Torch CPU RNG states must remain unchanged

The full-split HR@10/NDCG@10 produced here are validation-runtime regression
fingerprints at the untrained canonical initialization. They are NOT reported
test performance and are NOT used to select a checkpoint.
"""

from __future__ import annotations

import ast
import gc
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Frozen sources / contracts
# =============================================================================

PREFLIGHT_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_3b_canonical_real_validation_scoring_preflight.py"
)

RANKING_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_3a_validation_ranking_metric_semantics_audit.py"
)

ROUNDTRIP_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_2b_checkpoint_resume_roundtrip_proof.py"
)

PHASE_5_3_3A_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_3a_validation_ranking_metric_contract.json"
)

PHASE_5_3_3B_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_3b_canonical_validation_scoring_preflight_contract.json"
)

PHASE_5_3_3B_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_3b/"
    "phase_5_3_3b_canonical_validation_scoring_preflight_manifest.json"
)


# =============================================================================
# Frozen dimensions / counts
# =============================================================================

NUM_INVESTORS = 165_975
NUM_STARTUPS = 311_589
NUM_NODES = 477_564
NUM_HISTORY_PERIODS = 60

DESCRIPTION_DIM = 40
TREND_ITEM_DIM = 80
TREND_QUERY_DIM = 80
TREND_DIM = 40
STRUCTURAL_DIM = 40

VALIDATION_CASES = 2_251
TEST_CASES = 20_264
CANDIDATES_PER_CASE = 100
NEGATIVES_PER_CASE = 99

PREFIX_CASES = 16
MAIN_CHUNK_SIZE = 64

EXPECTED_INITIAL_MODEL_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

REFERENCE_TORCH_VERSION_PREFIX = "2.7.0"


# =============================================================================
# Frozen Phase-5.3.3b prefix fingerprints
# =============================================================================

EXPECTED_PREFIX_SELECTED_CASE_SHA256 = (
    "72331e17299e61fe94757eb5f4c00129"
    "dbfaffcd92b1424013443523cca37f96"
)

EXPECTED_PREFIX_CANDIDATE_MATRIX_SHA256 = (
    "a6a1b63954f3065d7c748fab55886af2"
    "8e440d0029626ed9db1809a389665514"
)

EXPECTED_PREFIX_LOGIT_MATRIX_SHA256 = (
    "bf4f986f96cf2e4557ca1e360a19b974"
    "9fadf452f1c6669cae83099a8905ea0c"
)

EXPECTED_PREFIX_CASE_METRIC_SHA256 = (
    "7b6b4a19b19052af84a14ee631020f79"
    "6fd6bfd2dc89f7c484b568983b6b2606"
)

EXPECTED_PREFIX_HR10 = 0.125
EXPECTED_PREFIX_NDCG10 = 0.040329500839


# =============================================================================
# Outputs
# =============================================================================

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_3c"
)

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

VALIDATION_CASES_PATH = (
    AUDIT_DIR
    / "full_validation_case_binding.parquet"
)

CANDIDATE_MATRIX_PATH = (
    AUDIT_DIR
    / "full_validation_candidate_startup_local.npy"
)

LOGIT_MATRIX_PATH = (
    AUDIT_DIR
    / "full_validation_raw_logit_matrix.npy"
)

CASE_METRICS_PATH = (
    AUDIT_DIR
    / "full_validation_case_metrics.parquet"
)

CHUNK_REGISTRY_PATH = (
    AUDIT_DIR
    / "full_validation_chunk_registry.csv"
)

CHUNK_TREND_AUDIT_PATH = (
    AUDIT_DIR
    / "full_validation_chunk_trend_audit.csv"
)

PREFIX_REGRESSION_PATH = (
    AUDIT_DIR
    / "phase_5_3_3b_prefix_regression_audit.csv"
)

FULL_METRIC_SUMMARY_PATH = (
    AUDIT_DIR
    / "full_validation_metric_summary.csv"
)

STATE_NEUTRALITY_PATH = (
    AUDIT_DIR
    / "full_validation_state_neutrality.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_3c_final_invariants.csv"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_3c_full_validation_runtime_decision_register.csv"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_3c_full_validation_runtime_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_3c_full_validation_runtime_manifest.json"
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


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def file_sha256(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def is_main_guard(
    node: ast.AST,
) -> bool:
    if not isinstance(
        node,
        ast.If,
    ):
        return False

    test = node.test

    return (
        isinstance(
            test,
            ast.Compare,
        )
        and isinstance(
            test.left,
            ast.Name,
        )
        and test.left.id == "__name__"
        and len(
            test.ops
        )
        == 1
        and isinstance(
            test.ops[0],
            ast.Eq,
        )
        and len(
            test.comparators
        )
        == 1
        and isinstance(
            test.comparators[0],
            ast.Constant,
        )
        and test.comparators[0].value
        == "__main__"
    )


def load_guarded_module(
    path: Path,
    module_name: str,
):
    require(
        path.exists(),
        f"Missing source: {path}",
    )

    source = path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(path),
    )

    guards = [
        node
        for node in tree.body
        if is_main_guard(node)
    ]

    require(
        len(guards) == 1,
        (
            "Expected exactly one __main__ "
            f"guard in {path}."
        ),
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    require(
        spec is not None
        and spec.loader is not None,
        (
            "Could not construct import spec "
            f"for {path}."
        ),
    )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def dataframe_logical_sha256(
    frame: pd.DataFrame,
    columns: list[str],
) -> str:
    digest = hashlib.sha256()

    for column in columns:
        require(
            column in frame.columns,
            (
                "Missing dataframe hash column: "
                f"{column}"
            ),
        )

        digest.update(
            column.encode("utf-8")
        )
        digest.update(b"\0")

        series = frame[
            column
        ]

        if pd.api.types.is_integer_dtype(
            series.dtype
        ):
            values = np.ascontiguousarray(
                series.to_numpy(
                    dtype=np.int64
                )
            )

            digest.update(
                values.tobytes(
                    order="C"
                )
            )

        elif pd.api.types.is_float_dtype(
            series.dtype
        ):
            values = np.ascontiguousarray(
                series.to_numpy(
                    dtype=np.float64
                )
            )

            digest.update(
                values.tobytes(
                    order="C"
                )
            )

        else:
            for value in (
                series
                .astype(str)
                .tolist()
            ):
                digest.update(
                    value.encode("utf-8")
                )
                digest.update(b"\0")

        digest.update(b"\0")

    return digest.hexdigest()


def array_logical_sha256(
    array: np.ndarray,
) -> str:
    value = np.ascontiguousarray(
        array
    )

    digest = hashlib.sha256()

    digest.update(
        str(
            value.dtype
        ).encode("utf-8")
    )

    digest.update(
        str(
            tuple(
                value.shape
            )
        ).encode("utf-8")
    )

    digest.update(
        value.tobytes(
            order="C"
        )
    )

    return digest.hexdigest()


def build_chunk_schedule() -> list[tuple[int, int, int]]:
    """
    Returns:
        [(chunk_index, start_case, end_case_exclusive), ...]
    """

    chunks = []

    # Exact preflight prefix chunk.
    chunks.append(
        (
            0,
            0,
            PREFIX_CASES,
        )
    )

    chunk_index = 1
    start = PREFIX_CASES

    while start < VALIDATION_CASES:
        end = min(
            start + MAIN_CHUNK_SIZE,
            VALIDATION_CASES,
        )

        chunks.append(
            (
                chunk_index,
                start,
                end,
            )
        )

        chunk_index += 1
        start = end

    require(
        chunks[0]
        == (
            0,
            0,
            16,
        ),
        (
            "Validation prefix chunk "
            "must be cases 0..15."
        ),
    )

    require(
        chunks[-1][2]
        == VALIDATION_CASES,
        (
            "Validation chunk schedule "
            "does not cover all cases."
        ),
    )

    covered = []

    for _, start, end in chunks:
        covered.extend(
            range(
                start,
                end,
            )
        )

    require(
        covered
        == list(
            range(
                VALIDATION_CASES
            )
        ),
        (
            "Validation chunks contain "
            "gap/overlap/reordering."
        ),
    )

    return chunks


# =============================================================================
# Chunk non-structural features
# =============================================================================

def compute_chunk_features(
    *,
    preflight_runtime,
    model: torch.nn.Module,
    chunk_cases: pd.DataFrame,
    chunk_candidate_matrix_local: np.ndarray,
    shared: dict,
    F_s_all: torch.Tensor,
) -> dict:
    selected_investors = (
        chunk_cases[
            "investor_global"
        ]
        .astype(int)
        .tolist()
    )

    (
        history_by_investor,
        historical_nodes,
    ) = (
        preflight_runtime
        .collect_history_by_investor(
            selected_investors,
            shared,
        )
    )

    candidate_globals = (
        chunk_candidate_matrix_local
        + NUM_INVESTORS
    )

    node_parts = [
        np.asarray(
            selected_investors,
            dtype=np.int64,
        ),
        np.asarray(
            candidate_globals.reshape(
                -1
            ),
            dtype=np.int64,
        ),
    ]

    node_parts.extend(
        historical_nodes
    )

    required_nodes = np.unique(
        np.concatenate(
            node_parts
        )
    )

    global_to_subset = np.full(
        NUM_NODES,
        -1,
        dtype=np.int64,
    )

    global_to_subset[
        required_nodes
    ] = np.arange(
        len(
            required_nodes
        ),
        dtype=np.int64,
    )

    doc_subset_np = np.array(
        shared[
            "doc2vec_all"
        ][
            required_nodes
        ],
        dtype=np.float32,
        copy=True,
    )

    label_subset_np = (
        shared[
            "labels_sparse"
        ][
            required_nodes
        ]
        .toarray()
        .astype(
            np.float32,
            copy=False,
        )
    )

    require(
        bool(
            np.isfinite(
                doc_subset_np
            ).all()
        ),
        (
            "Validation chunk Doc2Vec "
            "contains non-finite values."
        ),
    )

    require(
        bool(
            np.isfinite(
                label_subset_np
            ).all()
        ),
        (
            "Validation chunk category matrix "
            "contains non-finite values."
        ),
    )

    doc_subset = torch.from_numpy(
        doc_subset_np
    )

    label_subset = torch.from_numpy(
        label_subset_np
    )

    description_subset = (
        model.description_encoder(
            doc_subset,
            label_subset,
        )
    )

    require(
        description_subset.shape
        == (
            len(
                required_nodes
            ),
            DESCRIPTION_DIM,
        ),
        (
            "Chunk description representation "
            "shape drift."
        ),
    )

    F_t_by_investor = {}
    trend_rows = []

    for investor_global in sorted(
        history_by_investor
    ):
        investor_tensor = torch.tensor(
            [
                investor_global,
            ],
            dtype=torch.int64,
        )

        L_o = (
            model.investor_embedding(
                investor_tensor
            )[0]
        )

        investor_subset_position = int(
            global_to_subset[
                investor_global
            ]
        )

        require(
            investor_subset_position >= 0,
            (
                "Chunk investor description "
                "row missing."
            ),
        )

        F_d_o = description_subset[
            investor_subset_position
        ]

        query = torch.cat(
            [
                L_o,
                F_d_o,
            ],
            dim=0,
        )

        require(
            query.shape
            == (
                TREND_QUERY_DIM,
            ),
            (
                "Chunk trend query shape drift."
            ),
        )

        period_vectors = []

        nonempty_periods = 0
        multi_item_periods = 0
        maximum_items = 0

        for startups_global in (
            history_by_investor[
                investor_global
            ]
        ):
            item_count = int(
                len(
                    startups_global
                )
            )

            maximum_items = max(
                maximum_items,
                item_count,
            )

            if item_count == 0:
                period_vector = torch.zeros(
                    TREND_ITEM_DIM,
                    dtype=query.dtype,
                    device=query.device,
                )

            else:
                nonempty_periods += 1

                if item_count >= 2:
                    multi_item_periods += 1

                startup_local = (
                    startups_global
                    - NUM_INVESTORS
                )

                startup_tensor = torch.from_numpy(
                    np.array(
                        startup_local,
                        dtype=np.int64,
                        copy=True,
                    )
                )

                history_latent = (
                    model.startup_embedding(
                        startup_tensor
                    )
                )

                subset_positions_np = (
                    global_to_subset[
                        startups_global
                    ]
                )

                require(
                    bool(
                        (
                            subset_positions_np
                            >= 0
                        ).all()
                    ),
                    (
                        "Chunk historical startup "
                        "description row missing."
                    ),
                )

                subset_positions = torch.from_numpy(
                    np.array(
                        subset_positions_np,
                        dtype=np.int64,
                        copy=True,
                    )
                )

                history_description = (
                    description_subset[
                        subset_positions
                    ]
                )

                items = torch.cat(
                    [
                        history_latent,
                        history_description,
                    ],
                    dim=1,
                )

                require(
                    items.shape
                    == (
                        item_count,
                        TREND_ITEM_DIM,
                    ),
                    (
                        "Chunk trend item "
                        "shape drift."
                    ),
                )

                (
                    period_vector,
                    alpha,
                ) = (
                    model
                    .trend_extractor
                    .attend_period(
                        query,
                        items,
                    )
                )

                require(
                    bool(
                        torch.isfinite(
                            alpha
                        ).all()
                    ),
                    (
                        "Chunk trend attention "
                        "contains non-finite values."
                    ),
                )

                require(
                    abs(
                        float(
                            alpha
                            .detach()
                            .sum()
                        )
                        - 1.0
                    )
                    <= 1e-6,
                    (
                        "Chunk trend attention "
                        "does not sum to one."
                    ),
                )

            period_vectors.append(
                period_vector
            )

        require(
            len(
                period_vectors
            )
            == 60,
            (
                "T60 validation chunk history "
                "must contain exactly 60 periods."
            ),
        )

        sequence = torch.stack(
            period_vectors,
            dim=0,
        ).unsqueeze(
            0
        )

        require(
            sequence.shape
            == (
                1,
                60,
                80,
            ),
            (
                "T60 chunk sequence "
                "shape drift."
            ),
        )

        (
            F_t,
            gru_output,
        ) = (
            model
            .trend_extractor
            .encode_sequence(
                sequence
            )
        )

        require(
            F_t.shape
            == (
                1,
                40,
            ),
            (
                "T60 chunk F_t shape drift."
            ),
        )

        require(
            gru_output.shape
            == (
                1,
                60,
                40,
            ),
            (
                "T60 chunk GRU shape drift."
            ),
        )

        require(
            bool(
                torch.isfinite(
                    F_t
                ).all()
            ),
            (
                "Chunk F_t contains "
                "non-finite values."
            ),
        )

        F_t_by_investor[
            investor_global
        ] = F_t[
            0
        ]

        trend_rows.append(
            {
                "investor_global": (
                    investor_global
                ),
                "history_periods": (
                    60
                ),
                "nonempty_periods": (
                    nonempty_periods
                ),
                "multi_item_periods": (
                    multi_item_periods
                ),
                "maximum_items_in_period": (
                    maximum_items
                ),
                "runtime_method": (
                    "TrendExtractor.encode_sequence"
                ),
                "training_adapter_used": (
                    False
                ),
                "post_T59_padding": (
                    False
                ),
                "status": (
                    "PASS"
                ),
            }
        )

    return {
        "required_nodes": (
            required_nodes
        ),
        "global_to_subset": (
            global_to_subset
        ),
        "description_subset": (
            description_subset
        ),
        "F_s_all": (
            F_s_all
        ),
        "F_t_by_investor": (
            F_t_by_investor
        ),
        "trend_audit": pd.DataFrame(
            trend_rows
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.3c — "
        "FULL VALIDATION SPLIT RUNTIME DRY-RUN AND FREEZE"
    )

    print(
        "Validation cases to score:            2,251"
    )
    print(
        "Candidates / case:                    100"
    )
    print(
        "Test cases to score:                  0"
    )
    print(
        "Canonical model state:                INITIAL SEED-42"
    )
    print(
        "Optimizer instantiated:               NO"
    )
    print(
        "Backward computation:                 NO"
    )
    print(
        "Checkpoint selection performed:       NO"
    )

    # =========================================================================
    # Authoritative prerequisites
    # =========================================================================

    banner(
        "AUTHORITATIVE FULL-VALIDATION GATE RECHECK"
    )

    for path in (
        PREFLIGHT_SOURCE_PATH,
        RANKING_SOURCE_PATH,
        ROUNDTRIP_SOURCE_PATH,
        PHASE_5_3_3A_CONTRACT_PATH,
        PHASE_5_3_3B_CONTRACT_PATH,
        PHASE_5_3_3B_MANIFEST_PATH,
    ):
        require(
            path.exists(),
            (
                "Missing authoritative input: "
                f"{path}"
            ),
        )

        print(
            f"FOUND  {path}"
        )

    ranking_contract = load_json(
        PHASE_5_3_3A_CONTRACT_PATH
    )

    preflight_contract = load_json(
        PHASE_5_3_3B_CONTRACT_PATH
    )

    preflight_manifest = load_json(
        PHASE_5_3_3B_MANIFEST_PATH
    )

    require(
        ranking_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.3a contract "
            "is not frozen."
        ),
    )

    require(
        preflight_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.3b contract "
            "is not frozen."
        ),
    )

    require(
        preflight_manifest[
            "status"
        ]
        == (
            "CANONICAL_REAL_VALIDATION_SCORING_"
            "PREFLIGHT_PASSED_AND_FROZEN"
        ),
        (
            "Unexpected Phase-5.3.3b "
            "manifest status."
        ),
    )

    require(
        preflight_manifest[
            "candidate_matrix_sha256"
        ]
        == EXPECTED_PREFIX_CANDIDATE_MATRIX_SHA256,
        (
            "Phase-5.3.3b candidate prefix "
            "fingerprint drift."
        ),
    )

    require(
        preflight_manifest[
            "raw_logit_matrix_sha256"
        ]
        == EXPECTED_PREFIX_LOGIT_MATRIX_SHA256,
        (
            "Phase-5.3.3b logit prefix "
            "fingerprint drift."
        ),
    )

    require(
        preflight_manifest[
            "case_metric_sha256"
        ]
        == EXPECTED_PREFIX_CASE_METRIC_SHA256,
        (
            "Phase-5.3.3b metric prefix "
            "fingerprint drift."
        ),
    )

    require(
        torch.__version__.startswith(
            REFERENCE_TORCH_VERSION_PREFIX
        ),
        (
            "Reference runtime requires "
            "PyTorch 2.7.0."
        ),
    )

    print(
        "Phase-5.3.3a ranking contract:        FROZEN / PASS"
    )
    print(
        "Phase-5.3.3b real-model preflight:    FROZEN / PASS"
    )

    # =========================================================================
    # Load frozen runtime modules
    # =========================================================================

    banner(
        "LOAD FROZEN VALIDATION RUNTIMES"
    )

    preflight_runtime = load_guarded_module(
        PREFLIGHT_SOURCE_PATH,
        "_itrs_phase5_3_3c_preflight",
    )

    ranking_runtime = load_guarded_module(
        RANKING_SOURCE_PATH,
        "_itrs_phase5_3_3c_ranking",
    )

    runtime_2b = load_guarded_module(
        ROUNDTRIP_SOURCE_PATH,
        "_itrs_phase5_3_3c_runtime2b",
    )

    required_preflight_symbols = (
        "construct_canonical_validation_model",
        "infer_negative_matrix_coordinate",
        "normalize_negative_matrix_to_local",
        "resolve_case_bindings",
        "collect_history_by_investor",
        "score_validation_case",
    )

    for symbol in required_preflight_symbols:
        require(
            hasattr(
                preflight_runtime,
                symbol,
            ),
            (
                "Phase-5.3.3b runtime missing "
                f"{symbol}."
            ),
        )

    for symbol in (
        "rank_candidates",
        "metrics_from_positive_rank",
        "aggregate_event_level_metrics",
    ):
        require(
            hasattr(
                ranking_runtime,
                symbol,
            ),
            (
                "Phase-5.3.3a runtime missing "
                f"{symbol}."
            ),
        )

    # =========================================================================
    # Bind frozen evaluation artifacts
    # =========================================================================

    banner(
        "BIND ALL 2,251 FROZEN VALIDATION CASES"
    )

    negative_artifact = (
        ranking_contract[
            "frozen_artifacts"
        ][
            "negative_matrix"
        ]
    )

    case_artifact = (
        ranking_contract[
            "frozen_artifacts"
        ][
            "case_manifest"
        ]
    )

    negative_matrix_path = Path(
        negative_artifact[
            "path"
        ]
    )

    case_manifest_path = Path(
        case_artifact[
            "path"
        ]
    )

    require(
        file_sha256(
            negative_matrix_path
        )
        == negative_artifact[
            "file_sha256"
        ],
        (
            "Evaluation negative-matrix "
            "physical hash drift."
        ),
    )

    require(
        file_sha256(
            case_manifest_path
        )
        == case_artifact[
            "file_sha256"
        ],
        (
            "Evaluation case-manifest "
            "physical hash drift."
        ),
    )

    negative_matrix_raw = np.load(
        negative_matrix_path,
        mmap_mode="r",
    )

    negative_coordinate = (
        preflight_runtime
        .infer_negative_matrix_coordinate(
            negative_matrix_raw
        )
    )

    negative_matrix_local = (
        preflight_runtime
        .normalize_negative_matrix_to_local(
            negative_matrix_raw,
            negative_coordinate,
        )
    )

    case_manifest = pd.read_parquet(
        case_manifest_path
    )

    (
        resolved_cases,
        binding_metadata,
    ) = (
        preflight_runtime
        .resolve_case_bindings(
            case_manifest,
            ranking_contract,
        )
    )

    validation_cases = (
        resolved_cases.loc[
            resolved_cases[
                "split"
            ]
            == "validation"
        ]
        .sort_values(
            [
                "matrix_row_index",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    require(
        len(
            validation_cases
        )
        == VALIDATION_CASES,
        (
            "Resolved full validation "
            "case count drift."
        ),
    )

    require(
        int(
            (
                resolved_cases[
                    "split"
                ]
                == "test"
            ).sum()
        )
        == TEST_CASES,
        (
            "Resolved test case count drift."
        ),
    )

    candidate_matrix_local = np.empty(
        (
            VALIDATION_CASES,
            CANDIDATES_PER_CASE,
        ),
        dtype=np.int64,
    )

    case_rows = []

    for case_position, row in (
        validation_cases
        .iterrows()
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
            negative_matrix_local[
                matrix_row_index
            ],
            dtype=np.int64,
        )

        require(
            negatives_local.shape
            == (
                NEGATIVES_PER_CASE,
            ),
            (
                "Validation case does not "
                "contain 99 negatives."
            ),
        )

        require(
            positive_local
            not in set(
                int(value)
                for value
                in negatives_local.tolist()
            ),
            (
                "Positive startup appears "
                "inside frozen negatives."
            ),
        )

        candidates = np.concatenate(
            [
                np.array(
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
                "Validation candidate set "
                "does not contain 100 unique startups."
            ),
        )

        candidate_matrix_local[
            case_position
        ] = candidates

        case_rows.append(
            {
                "validation_case_position": (
                    int(
                        case_position
                    )
                ),
                "matrix_row_index": (
                    matrix_row_index
                ),
                "interaction_id": str(
                    row[
                        "interaction_id"
                    ]
                ),
                "investor_global": int(
                    row[
                        "investor_global"
                    ]
                ),
                "positive_startup_local": (
                    positive_local
                ),
                "candidate_count": (
                    100
                ),
                "negative_count": (
                    99
                ),
                "candidate_position_of_positive": (
                    0
                ),
                "split": (
                    "validation"
                ),
            }
        )

    validation_binding_df = pd.DataFrame(
        case_rows
    )

    full_case_binding_sha = (
        dataframe_logical_sha256(
            validation_binding_df,
            columns=[
                "validation_case_position",
                "matrix_row_index",
                "interaction_id",
                "investor_global",
                "positive_startup_local",
                "candidate_count",
                "negative_count",
            ],
        )
    )

    full_candidate_matrix_sha = (
        array_logical_sha256(
            candidate_matrix_local
        )
    )

    # -------------------------------------------------------------------------
    # Prefix binding must exactly reproduce Phase 5.3.3b.
    # -------------------------------------------------------------------------

    prefix_selection = (
        validation_binding_df.iloc[
            :PREFIX_CASES
        ]
        .copy()
        .rename(
            columns={
                "validation_case_position":
                    "preflight_case_position",
            }
        )
        .reset_index(
            drop=True
        )
    )

    prefix_selected_case_sha = (
        dataframe_logical_sha256(
            prefix_selection,
            columns=[
                "preflight_case_position",
                "matrix_row_index",
                "interaction_id",
                "investor_global",
                "positive_startup_local",
                "candidate_count",
                "negative_count",
            ],
        )
    )

    prefix_candidate_sha = (
        array_logical_sha256(
            candidate_matrix_local[
                :PREFIX_CASES
            ]
        )
    )

    require(
        prefix_selected_case_sha
        == EXPECTED_PREFIX_SELECTED_CASE_SHA256,
        (
            "Full-validation prefix case "
            "binding does not reproduce 5.3.3b."
        ),
    )

    require(
        prefix_candidate_sha
        == EXPECTED_PREFIX_CANDIDATE_MATRIX_SHA256,
        (
            "Full-validation prefix candidate "
            "matrix does not reproduce 5.3.3b."
        ),
    )

    print(
        f"Validation cases bound:                "
        f"{len(validation_binding_df):,}"
    )
    print(
        f"Test cases scored/bound for scoring:   0"
    )
    print()
    print(
        "Full validation case-binding SHA256:"
    )
    print(
        full_case_binding_sha
    )
    print()
    print(
        "Full validation candidate-matrix SHA256:"
    )
    print(
        full_candidate_matrix_sha
    )

    # =========================================================================
    # Chunk schedule
    # =========================================================================

    banner(
        "FREEZE MEMORY-BOUNDED VALIDATION CHUNK SCHEDULE"
    )

    chunks = build_chunk_schedule()

    chunk_rows = []

    for (
        chunk_index,
        start,
        end,
    ) in chunks:
        chunk_rows.append(
            {
                "chunk_index": (
                    chunk_index
                ),
                "start_case_position": (
                    start
                ),
                "end_case_position_exclusive": (
                    end
                ),
                "case_count": (
                    end
                    - start
                ),
                "is_exact_phase_5_3_3b_prefix": (
                    chunk_index
                    == 0
                ),
            }
        )

    chunk_registry_df = pd.DataFrame(
        chunk_rows
    )

    require(
        int(
            chunk_registry_df[
                "case_count"
            ].sum()
        )
        == VALIDATION_CASES,
        (
            "Validation chunk schedule "
            "does not cover 2,251 cases."
        ),
    )

    require(
        int(
            chunk_registry_df.iloc[
                0
            ][
                "case_count"
            ]
        )
        == 16,
        (
            "First validation chunk "
            "must contain 16 cases."
        ),
    )

    require(
        int(
            chunk_registry_df.iloc[
                1:
            ][
                "case_count"
            ].max()
        )
        <= MAIN_CHUNK_SIZE,
        (
            "Post-prefix validation chunk "
            "exceeds size 64."
        ),
    )

    print(
        chunk_registry_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Canonical model / static structural representation
    # =========================================================================

    banner(
        "RECONSTRUCT CANONICAL INITIAL MODEL"
    )

    (
        preflight_numerical_runtime,
        model,
        canonical_hash_fn,
        runtime_ast_sha,
        adapter_sha,
        removed_guard_sha,
    ) = (
        preflight_runtime
        .construct_canonical_validation_model(
            runtime_2b
        )
    )

    shared = (
        runtime_2b
        .load_shared_inputs(
            preflight_numerical_runtime
        )
    )

    parameter_hash_before = (
        canonical_hash_fn(
            model
        )
    )

    require(
        parameter_hash_before
        == EXPECTED_INITIAL_MODEL_SHA256,
        (
            "Canonical initial validation "
            "parameter hash drift."
        ),
    )

    require(
        model.training
        is False,
        (
            "Validation model must be eval()."
        ),
    )

    require(
        all(
            parameter.grad
            is None
            for parameter
            in model.parameters()
        ),
        (
            "Validation model has "
            "pre-existing gradients."
        ),
    )

    rng_before = (
        preflight_runtime
        .rng_snapshot()
    )

    banner(
        "COMPUTE FULL-GRAPH STRUCTURAL F_s ONCE"
    )

    with torch.no_grad():
        latent_all = torch.cat(
            [
                model.investor_embedding.weight,
                model.startup_embedding.weight,
            ],
            dim=0,
        )

        structural = (
            model.preference_propagation(
                latent_all,
                shared[
                    "edge_index"
                ],
                shared[
                    "edge_type"
                ],
            )
        )

        require(
            isinstance(
                structural,
                dict,
            )
            and "F_s"
            in structural,
            (
                "Full validation structural "
                "output invalid."
            ),
        )

        F_s_all = structural[
            "F_s"
        ]

        require(
            F_s_all.shape
            == (
                NUM_NODES,
                STRUCTURAL_DIM,
            ),
            (
                "Full validation F_s "
                "shape drift."
            ),
        )

        require(
            bool(
                torch.isfinite(
                    F_s_all
                ).all()
            ),
            (
                "Full validation F_s "
                "contains non-finite values."
            ),
        )

    print(
        "Structural representation shape:      "
        f"{tuple(F_s_all.shape)}"
    )
    print(
        "Structural recomputations:             1"
    )

    # =========================================================================
    # Full validation forward
    # =========================================================================

    banner(
        "SCORE ALL 2,251 VALIDATION CASES"
    )

    score_matrix = np.empty(
        (
            VALIDATION_CASES,
            CANDIDATES_PER_CASE,
        ),
        dtype=np.float32,
    )

    metric_rows = []
    trend_frames = []

    with torch.no_grad():
        for (
            chunk_index,
            start,
            end,
        ) in chunks:
            chunk_cases = (
                validation_cases.iloc[
                    start:end
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            chunk_candidates = (
                candidate_matrix_local[
                    start:end
                ]
            )

            features = compute_chunk_features(
                preflight_runtime=(
                    preflight_runtime
                ),
                model=model,
                chunk_cases=(
                    chunk_cases
                ),
                chunk_candidate_matrix_local=(
                    chunk_candidates
                ),
                shared=shared,
                F_s_all=F_s_all,
            )

            chunk_trend_df = (
                features[
                    "trend_audit"
                ].copy()
            )

            chunk_trend_df.insert(
                0,
                "chunk_index",
                chunk_index,
            )

            trend_frames.append(
                chunk_trend_df
            )

            for local_case_position, row in (
                chunk_cases.iterrows()
            ):
                global_case_position = (
                    start
                    + int(
                        local_case_position
                    )
                )

                investor_global = int(
                    row[
                        "investor_global"
                    ]
                )

                positive_local = int(
                    row[
                        "positive_startup_local"
                    ]
                )

                candidates_local = (
                    chunk_candidates[
                        local_case_position
                    ]
                )

                logits = (
                    preflight_runtime
                    .score_validation_case(
                        model,
                        investor_global,
                        candidates_local,
                        features,
                    )
                )

                logits_np = (
                    logits
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

                require(
                    np.isfinite(
                        logits_np
                    ).all(),
                    (
                        "Full validation logits "
                        "contain non-finite values."
                    ),
                )

                score_matrix[
                    global_case_position
                ] = logits_np

                positive_rank = (
                    ranking_runtime
                    .rank_candidates(
                        logits_np,
                        candidates_local,
                        positive_local,
                    )
                )

                (
                    hr10,
                    ndcg10,
                ) = (
                    ranking_runtime
                    .metrics_from_positive_rank(
                        positive_rank,
                        k=10,
                    )
                )

                metric_rows.append(
                    {
                        "validation_case_position": (
                            global_case_position
                        ),
                        "matrix_row_index": int(
                            row[
                                "matrix_row_index"
                            ]
                        ),
                        "interaction_id": str(
                            row[
                                "interaction_id"
                            ]
                        ),
                        "investor_global": (
                            investor_global
                        ),
                        "positive_startup_local": (
                            positive_local
                        ),
                        "positive_logit": float(
                            logits_np[
                                0
                            ]
                        ),
                        "minimum_logit": float(
                            np.min(
                                logits_np
                            )
                        ),
                        "maximum_logit": float(
                            np.max(
                                logits_np
                            )
                        ),
                        "mean_logit": float(
                            np.mean(
                                logits_np,
                                dtype=np.float64,
                            )
                        ),
                        "positive_rank": (
                            int(
                                positive_rank
                            )
                        ),
                        "HR@10": (
                            float(
                                hr10
                            )
                        ),
                        "NDCG@10": (
                            float(
                                ndcg10
                            )
                        ),
                        "split": (
                            "validation"
                        ),
                        "chunk_index": (
                            chunk_index
                        ),
                    }
                )

            print(
                f"Chunk {chunk_index:02d}: "
                f"cases {start:4d}..{end - 1:4d} "
                f"({end - start:2d}) PASS"
            )

            del features
            gc.collect()

    metric_df = pd.DataFrame(
        metric_rows
    ).sort_values(
        [
            "validation_case_position",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    trend_df = pd.concat(
        trend_frames,
        ignore_index=True,
    )

    require(
        len(
            metric_df
        )
        == VALIDATION_CASES,
        (
            "Full validation metric row "
            "count drift."
        ),
    )

    require(
        metric_df[
            "validation_case_position"
        ].tolist()
        == list(
            range(
                VALIDATION_CASES
            )
        ),
        (
            "Full validation metric rows "
            "are not in exact case order."
        ),
    )

    require(
        bool(
            metric_df[
                "positive_rank"
            ]
            .between(
                1,
                100,
            )
            .all()
        ),
        (
            "Full validation positive rank "
            "outside 1..100."
        ),
    )

    require(
        bool(
            np.isfinite(
                score_matrix
            ).all()
        ),
        (
            "Full validation score matrix "
            "contains non-finite values."
        ),
    )

    # =========================================================================
    # Exact prefix regression
    # =========================================================================

    banner(
        "PHASE-5.3.3b 16-CASE PREFIX REGRESSION"
    )

    prefix_score_sha = (
        array_logical_sha256(
            score_matrix[
                :PREFIX_CASES
            ]
        )
    )

    prefix_metric = (
        metric_df.iloc[
            :PREFIX_CASES
        ]
        .copy()
        .rename(
            columns={
                "validation_case_position":
                    "preflight_case_position",
            }
        )
        .reset_index(
            drop=True
        )
    )

    prefix_metric_sha = (
        dataframe_logical_sha256(
            prefix_metric,
            columns=[
                "preflight_case_position",
                "matrix_row_index",
                "interaction_id",
                "investor_global",
                "positive_startup_local",
                "positive_logit",
                "minimum_logit",
                "maximum_logit",
                "mean_logit",
                "positive_rank",
                "HR@10",
                "NDCG@10",
            ],
        )
    )

    prefix_hr10 = float(
        prefix_metric[
            "HR@10"
        ].mean()
    )

    prefix_ndcg10 = float(
        prefix_metric[
            "NDCG@10"
        ].mean()
    )

    require(
        prefix_score_sha
        == EXPECTED_PREFIX_LOGIT_MATRIX_SHA256,
        (
            "Full validation first-16 logits "
            "do not reproduce Phase-5.3.3b."
        ),
    )

    require(
        prefix_metric_sha
        == EXPECTED_PREFIX_CASE_METRIC_SHA256,
        (
            "Full validation first-16 metrics "
            "do not reproduce Phase-5.3.3b."
        ),
    )

    require(
        prefix_hr10
        == EXPECTED_PREFIX_HR10,
        (
            "Full validation first-16 HR@10 "
            "does not reproduce 5.3.3b."
        ),
    )

    require(
        abs(
            prefix_ndcg10
            - EXPECTED_PREFIX_NDCG10
        )
        <= 5e-13,
        (
            "Full validation first-16 NDCG@10 "
            "does not reproduce 5.3.3b."
        ),
    )

    prefix_regression_df = pd.DataFrame(
        [
            {
                "check": (
                    "selected_case_sha256"
                ),
                "actual": (
                    prefix_selected_case_sha
                ),
                "expected": (
                    EXPECTED_PREFIX_SELECTED_CASE_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "candidate_matrix_sha256"
                ),
                "actual": (
                    prefix_candidate_sha
                ),
                "expected": (
                    EXPECTED_PREFIX_CANDIDATE_MATRIX_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "raw_logit_matrix_sha256"
                ),
                "actual": (
                    prefix_score_sha
                ),
                "expected": (
                    EXPECTED_PREFIX_LOGIT_MATRIX_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "case_metric_sha256"
                ),
                "actual": (
                    prefix_metric_sha
                ),
                "expected": (
                    EXPECTED_PREFIX_CASE_METRIC_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "diagnostic_HR@10"
                ),
                "actual": (
                    prefix_hr10
                ),
                "expected": (
                    EXPECTED_PREFIX_HR10
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "diagnostic_NDCG@10"
                ),
                "actual": (
                    prefix_ndcg10
                ),
                "expected": (
                    EXPECTED_PREFIX_NDCG10
                ),
                "status": (
                    "PASS"
                ),
            },
        ]
    )

    print(
        prefix_regression_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Full validation metrics / fingerprints
    # =========================================================================

    banner(
        "FULL VALIDATION METRICS / FINGERPRINTS"
    )

    (
        validation_hr10,
        validation_ndcg10,
    ) = (
        ranking_runtime
        .aggregate_event_level_metrics(
            metric_df
        )
    )

    full_logit_matrix_sha = (
        array_logical_sha256(
            score_matrix
        )
    )

    full_metric_sha = (
        dataframe_logical_sha256(
            metric_df,
            columns=[
                "validation_case_position",
                "matrix_row_index",
                "interaction_id",
                "investor_global",
                "positive_startup_local",
                "positive_logit",
                "minimum_logit",
                "maximum_logit",
                "mean_logit",
                "positive_rank",
                "HR@10",
                "NDCG@10",
            ],
        )
    )

    positive_rank_array = (
        metric_df[
            "positive_rank"
        ].to_numpy(
            dtype=np.int64
        )
    )

    positive_rank_sha = (
        array_logical_sha256(
            positive_rank_array
        )
    )

    hit_count = int(
        metric_df[
            "HR@10"
        ].sum()
    )

    metric_summary_df = pd.DataFrame(
        [
            {
                "metric": (
                    "validation_cases"
                ),
                "value": (
                    VALIDATION_CASES
                ),
            },
            {
                "metric": (
                    "test_cases_scored"
                ),
                "value": (
                    0
                ),
            },
            {
                "metric": (
                    "HR@10"
                ),
                "value": (
                    validation_hr10
                ),
            },
            {
                "metric": (
                    "NDCG@10"
                ),
                "value": (
                    validation_ndcg10
                ),
            },
            {
                "metric": (
                    "HR@10_hit_count"
                ),
                "value": (
                    hit_count
                ),
            },
            {
                "metric": (
                    "mean_positive_rank"
                ),
                "value": float(
                    metric_df[
                        "positive_rank"
                    ].mean()
                ),
            },
            {
                "metric": (
                    "median_positive_rank"
                ),
                "value": float(
                    metric_df[
                        "positive_rank"
                    ].median()
                ),
            },
        ]
    )

    print(
        metric_summary_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Full raw-logit matrix logical SHA256:"
    )
    print(
        full_logit_matrix_sha
    )
    print()
    print(
        "Full case-metric logical SHA256:"
    )
    print(
        full_metric_sha
    )
    print()
    print(
        "Positive-rank vector logical SHA256:"
    )
    print(
        positive_rank_sha
    )

    # =========================================================================
    # State / RNG neutrality
    # =========================================================================

    banner(
        "FULL VALIDATION STATE / RNG NEUTRALITY"
    )

    parameter_hash_after = (
        canonical_hash_fn(
            model
        )
    )

    rng_after = (
        preflight_runtime
        .rng_snapshot()
    )

    require(
        parameter_hash_after
        == parameter_hash_before
        == EXPECTED_INITIAL_MODEL_SHA256,
        (
            "Full validation changed "
            "canonical parameter values."
        ),
    )

    require(
        all(
            parameter.grad
            is None
            for parameter
            in model.parameters()
        ),
        (
            "Full validation created gradients."
        ),
    )

    require(
        preflight_runtime
        .rng_equal(
            rng_before,
            rng_after,
        ),
        (
            "Full validation changed "
            "global RNG state."
        ),
    )

    state_df = pd.DataFrame(
        [
            {
                "check": (
                    "canonical_parameter_sha_before"
                ),
                "actual": (
                    parameter_hash_before
                ),
                "expected": (
                    EXPECTED_INITIAL_MODEL_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "canonical_parameter_sha_after"
                ),
                "actual": (
                    parameter_hash_after
                ),
                "expected": (
                    EXPECTED_INITIAL_MODEL_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "gradients_created"
                ),
                "actual": (
                    "False"
                ),
                "expected": (
                    "False"
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "global_RNG_changed"
                ),
                "actual": (
                    "False"
                ),
                "expected": (
                    "False"
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "optimizer_instantiated"
                ),
                "actual": (
                    "False"
                ),
                "expected": (
                    "False"
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "test_cases_scored"
                ),
                "actual": (
                    0
                ),
                "expected": (
                    0
                ),
                "status": (
                    "PASS"
                ),
            },
        ]
    )

    print(
        state_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.3c INVARIANTS"
    )

    checks = [
        (
            "phase_5_3_3a_contract_frozen",
            (
                ranking_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),
        (
            "phase_5_3_3b_contract_frozen",
            (
                preflight_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),
        (
            "validation_cases_exactly_2251",
            (
                len(
                    validation_cases
                )
                == VALIDATION_CASES
            ),
        ),
        (
            "test_cases_scored_zero",
            True,
        ),
        (
            "candidate_matrix_2251_by_100",
            (
                candidate_matrix_local.shape
                == (
                    2251,
                    100,
                )
            ),
        ),
        (
            "all_candidate_sets_unique_100",
            all(
                len(
                    np.unique(
                        candidate_matrix_local[
                            index
                        ]
                    )
                )
                == 100
                for index in range(
                    VALIDATION_CASES
                )
            ),
        ),
        (
            "prefix_selected_case_sha_exact",
            (
                prefix_selected_case_sha
                == EXPECTED_PREFIX_SELECTED_CASE_SHA256
            ),
        ),
        (
            "prefix_candidate_matrix_sha_exact",
            (
                prefix_candidate_sha
                == EXPECTED_PREFIX_CANDIDATE_MATRIX_SHA256
            ),
        ),
        (
            "prefix_logit_matrix_sha_exact",
            (
                prefix_score_sha
                == EXPECTED_PREFIX_LOGIT_MATRIX_SHA256
            ),
        ),
        (
            "prefix_metric_sha_exact",
            (
                prefix_metric_sha
                == EXPECTED_PREFIX_CASE_METRIC_SHA256
            ),
        ),
        (
            "prefix_HR10_exact",
            (
                prefix_hr10
                == EXPECTED_PREFIX_HR10
            ),
        ),
        (
            "prefix_NDCG10_exact",
            (
                abs(
                    prefix_ndcg10
                    - EXPECTED_PREFIX_NDCG10
                )
                <= 5e-13
            ),
        ),
        (
            "chunk_schedule_covers_all_validation_cases",
            (
                int(
                    chunk_registry_df[
                        "case_count"
                    ].sum()
                )
                == VALIDATION_CASES
            ),
        ),
        (
            "prefix_chunk_size_16",
            (
                int(
                    chunk_registry_df.iloc[
                        0
                    ][
                        "case_count"
                    ]
                )
                == 16
            ),
        ),
        (
            "remaining_chunks_at_most_64",
            (
                int(
                    chunk_registry_df.iloc[
                        1:
                    ][
                        "case_count"
                    ].max()
                )
                <= 64
            ),
        ),
        (
            "structural_F_s_computed_once",
            True,
        ),
        (
            "T60_original_encode_sequence_used",
            (
                set(
                    trend_df[
                        "runtime_method"
                    ]
                )
                == {
                    "TrendExtractor.encode_sequence"
                }
            ),
        ),
        (
            "all_T60_histories_exactly_60_periods",
            bool(
                (
                    trend_df[
                        "history_periods"
                    ]
                    == 60
                ).all()
            ),
        ),
        (
            "training_adapter_not_used",
            bool(
                (
                    trend_df[
                        "training_adapter_used"
                    ]
                    == False
                ).all()
            ),
        ),
        (
            "full_logit_matrix_2251_by_100",
            (
                score_matrix.shape
                == (
                    2251,
                    100,
                )
            ),
        ),
        (
            "all_validation_logits_finite",
            bool(
                np.isfinite(
                    score_matrix
                ).all()
            ),
        ),
        (
            "all_positive_ranks_1_to_100",
            bool(
                metric_df[
                    "positive_rank"
                ]
                .between(
                    1,
                    100,
                )
                .all()
            ),
        ),
        (
            "HR10_finite_and_bounded",
            (
                math.isfinite(
                    validation_hr10
                )
                and 0.0
                <= validation_hr10
                <= 1.0
            ),
        ),
        (
            "NDCG10_finite_and_bounded",
            (
                math.isfinite(
                    validation_ndcg10
                )
                and 0.0
                <= validation_ndcg10
                <= 1.0
            ),
        ),
        (
            "event_level_mean_aggregation",
            True,
        ),
        (
            "checkpoint_selection_not_performed",
            True,
        ),
        (
            "canonical_parameter_hash_unchanged",
            (
                parameter_hash_after
                == parameter_hash_before
                == EXPECTED_INITIAL_MODEL_SHA256
            ),
        ),
        (
            "gradients_not_created",
            all(
                parameter.grad
                is None
                for parameter
                in model.parameters()
            ),
        ),
        (
            "global_RNG_unchanged",
            (
                preflight_runtime
                .rng_equal(
                    rng_before,
                    rng_after,
                )
            ),
        ),
        (
            "optimizer_not_instantiated",
            True,
        ),
        (
            "backward_not_performed",
            True,
        ),
        (
            "optimizer_steps_zero",
            True,
        ),
    ]

    invariant_df = pd.DataFrame(
        [
            {
                "check": name,
                "result": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
            }
            for (
                name,
                passed,
            ) in checks
        ]
    )

    require(
        bool(
            (
                invariant_df[
                    "result"
                ]
                == "PASS"
            ).all()
        ),
        (
            "At least one Phase-5.3.3c "
            "full-validation invariant failed."
        ),
    )

    print(
        invariant_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Write outputs
    # =========================================================================

    banner(
        "WRITE PHASE-5.3.3c OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_binding_df.to_parquet(
        VALIDATION_CASES_PATH,
        index=False,
    )

    np.save(
        CANDIDATE_MATRIX_PATH,
        candidate_matrix_local,
        allow_pickle=False,
    )

    np.save(
        LOGIT_MATRIX_PATH,
        score_matrix,
        allow_pickle=False,
    )

    metric_df.to_parquet(
        CASE_METRICS_PATH,
        index=False,
    )

    chunk_registry_df.to_csv(
        CHUNK_REGISTRY_PATH,
        index=False,
    )

    trend_df.to_csv(
        CHUNK_TREND_AUDIT_PATH,
        index=False,
    )

    prefix_regression_df.to_csv(
        PREFIX_REGRESSION_PATH,
        index=False,
    )

    metric_summary_df.to_csv(
        FULL_METRIC_SUMMARY_PATH,
        index=False,
    )

    state_df.to_csv(
        STATE_NEUTRALITY_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    decision_df = pd.DataFrame(
        [
            {
                "decision": (
                    "full_validation_split"
                ),
                "value": (
                    "ALL_2251_EVENT_LEVEL_VALIDATION_CASES"
                ),
                "classification": (
                    "INHERITED_FROZEN_PHASE_2_AND_PHASE_5_1"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3c"
                ),
            },
            {
                "decision": (
                    "validation_chunk_schedule"
                ),
                "value": (
                    "FIRST_16_EXACT_PREFIX_THEN_CHUNKS_OF_AT_MOST_64"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3c"
                ),
            },
            {
                "decision": (
                    "validation_structural_runtime"
                ),
                "value": (
                    "COMPUTE_FULL_F_S_ONCE_PER_VALIDATION_PASS"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3c"
                ),
            },
            {
                "decision": (
                    "validation_description_runtime"
                ),
                "value": (
                    "MATERIALIZE_REQUIRED_NODES_PER_CHUNK"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3c"
                ),
            },
            {
                "decision": (
                    "validation_trend_runtime"
                ),
                "value": (
                    "ORIGINAL_ENCODE_SEQUENCE_EXACT_T0_TO_T59"
                ),
                "classification": (
                    "INHERITED_FROZEN_PHASE_4"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3c"
                ),
            },
            {
                "decision": (
                    "canonical_initial_full_validation_metrics"
                ),
                "value": (
                    "REGRESSION_FINGERPRINT_ONLY_NOT_MODEL_SELECTION"
                ),
                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3c"
                ),
            },
            {
                "decision": (
                    "test_access"
                ),
                "value": (
                    "ZERO_TEST_CASES_SCORED"
                ),
                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3c"
                ),
            },
        ]
    )

    decision_df.to_csv(
        DECISION_REGISTER_PATH,
        index=False,
    )

    contract = {
        "phase": (
            "5.3.3c"
        ),
        "title": (
            "Full Validation Split Runtime Contract"
        ),
        "status": (
            "FROZEN"
        ),
        "validation_split": {
            "cases": (
                VALIDATION_CASES
            ),
            "candidates_per_case": (
                CANDIDATES_PER_CASE
            ),
            "test_cases_scored": (
                0
            ),
            "case_binding_logical_sha256": (
                full_case_binding_sha
            ),
            "candidate_matrix_logical_sha256": (
                full_candidate_matrix_sha
            ),
        },
        "chunk_runtime": {
            "first_chunk_cases": (
                16
            ),
            "first_chunk_purpose": (
                "exact Phase-5.3.3b regression prefix"
            ),
            "remaining_max_chunk_size": (
                64
            ),
            "chunk_count": (
                len(
                    chunk_registry_df
                )
            ),
            "structural_F_s_recomputations": (
                1
            ),
            "description_scope": (
                "required nodes per chunk"
            ),
        },
        "T60_trend": {
            "history": (
                "T0..T59"
            ),
            "periods": (
                60
            ),
            "runtime_method": (
                "TrendExtractor.encode_sequence"
            ),
            "training_adapter_used": (
                False
            ),
            "post_T59_padding": (
                False
            ),
        },
        "ranking": {
            "score": (
                "raw model logit"
            ),
            "primary_order": (
                "logit descending"
            ),
            "tie_break": (
                "startup_local ascending"
            ),
            "positive_rank": (
                "1-based"
            ),
            "HR@10": (
                "1 if rank <= 10 else 0"
            ),
            "NDCG@10": (
                "1/log2(rank+1) if rank <= 10 else 0"
            ),
            "aggregation": (
                "arithmetic event-level mean"
            ),
        },
        "phase_5_3_3b_prefix_regression": {
            "selected_case_sha256": (
                prefix_selected_case_sha
            ),
            "candidate_matrix_sha256": (
                prefix_candidate_sha
            ),
            "raw_logit_matrix_sha256": (
                prefix_score_sha
            ),
            "case_metric_sha256": (
                prefix_metric_sha
            ),
            "HR@10": (
                prefix_hr10
            ),
            "NDCG@10": (
                prefix_ndcg10
            ),
            "exact": (
                True
            ),
        },
        "canonical_initial_validation": {
            "HR@10": (
                validation_hr10
            ),
            "NDCG@10": (
                validation_ndcg10
            ),
            "HR@10_hit_count": (
                hit_count
            ),
            "raw_logit_matrix_logical_sha256": (
                full_logit_matrix_sha
            ),
            "case_metric_logical_sha256": (
                full_metric_sha
            ),
            "positive_rank_vector_logical_sha256": (
                positive_rank_sha
            ),
            "purpose": (
                "validation runtime regression fingerprint only"
            ),
            "used_for_checkpoint_selection": (
                False
            ),
        },
        "state_neutrality": {
            "parameter_sha256_before": (
                parameter_hash_before
            ),
            "parameter_sha256_after": (
                parameter_hash_after
            ),
            "parameters_changed": (
                False
            ),
            "gradients_created": (
                False
            ),
            "global_RNG_changed": (
                False
            ),
            "optimizer_instantiated": (
                False
            ),
            "optimizer_steps": (
                0
            ),
        },
        "boundary": {
            "test_cases_scored": (
                0
            ),
            "test_used_for_selection": (
                False
            ),
            "checkpoint_selection_performed": (
                False
            ),
            "training_performed": (
                False
            ),
        },
        "next_phase": {
            "id": (
                "5.3.3d"
            ),
            "title": (
                "Validation-to-Checkpoint Selection Integration Proof"
            ),
            "requirement": (
                "Integrate the frozen full-validation runtime with the "
                "Phase-5.3.2 production controller and best-checkpoint "
                "comparison rule. Use controlled validation metric inputs "
                "plus one real canonical full-validation pass to prove "
                "epoch validation commit, best replacement/tie behavior, "
                "checkpoint payload fields, epoch advancement, and continued "
                "test isolation before full training."
            ),
        },
    }

    CONTRACT_PATH.write_text(
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    manifest = {
        "phase": (
            "5.3.3c"
        ),
        "status": (
            "FULL_VALIDATION_SPLIT_RUNTIME_"
            "PASSED_AND_FROZEN"
        ),
        "validation_cases": (
            VALIDATION_CASES
        ),
        "test_cases_scored": (
            0
        ),
        "chunk_count": (
            len(
                chunk_registry_df
            )
        ),
        "full_validation_HR@10": (
            validation_hr10
        ),
        "full_validation_NDCG@10": (
            validation_ndcg10
        ),
        "full_logit_matrix_sha256": (
            full_logit_matrix_sha
        ),
        "full_case_metric_sha256": (
            full_metric_sha
        ),
        "positive_rank_vector_sha256": (
            positive_rank_sha
        ),
        "canonical_parameter_sha256_after": (
            parameter_hash_after
        ),
        "optimizer_instantiated": (
            False
        ),
        "optimizer_steps": (
            0
        ),
        "contract": str(
            CONTRACT_PATH
        ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    for path in (
        VALIDATION_CASES_PATH,
        CANDIDATE_MATRIX_PATH,
        LOGIT_MATRIX_PATH,
        CASE_METRICS_PATH,
        CHUNK_REGISTRY_PATH,
        CHUNK_TREND_AUDIT_PATH,
        PREFIX_REGRESSION_PATH,
        FULL_METRIC_SUMMARY_PATH,
        STATE_NEUTRALITY_PATH,
        FINAL_INVARIANT_PATH,
        DECISION_REGISTER_PATH,
        CONTRACT_PATH,
        MANIFEST_PATH,
    ):
        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Final
    # =========================================================================

    banner(
        "PHASE 5.3.3c FINAL STATUS"
    )

    print(
        "Validation cases scored:              2,251"
    )
    print(
        "Candidates / case:                    100"
    )
    print(
        "Test cases scored:                    0"
    )
    print()

    print(
        "Chunking:"
    )
    print(
        "  first chunk:                        16 cases"
    )
    print(
        "  remaining max chunk:                64 cases"
    )
    print(
        f"  total chunks:                       "
        f"{len(chunk_registry_df)}"
    )
    print()

    print(
        "Phase-5.3.3b first-16 regression:     EXACT PASS"
    )
    print()

    print(
        f"Canonical initial validation HR@10:   "
        f"{validation_hr10:.12f}"
    )
    print(
        f"Canonical initial validation NDCG@10: "
        f"{validation_ndcg10:.12f}"
    )
    print(
        "Used for checkpoint selection:        NO"
    )
    print()

    print(
        "Full raw-logit matrix SHA256:"
    )
    print(
        full_logit_matrix_sha
    )
    print()

    print(
        "Full case-metric SHA256:"
    )
    print(
        full_metric_sha
    )
    print()

    print(
        "Positive-rank vector SHA256:"
    )
    print(
        positive_rank_sha
    )
    print()

    print(
        "Canonical parameter SHA256 after:"
    )
    print(
        parameter_hash_after
    )
    print()

    print(
        "Parameter values changed:             NO"
    )
    print(
        "Gradients created:                    NO"
    )
    print(
        "Global RNG changed:                   NO"
    )
    print(
        "Optimizer instantiated:               NO"
    )
    print(
        "optimizer.step():                     0"
    )
    print(
        "Test used for model selection:        NO"
    )

    banner(
        "PHASE 5.3.3c COMPLETE / "
        "FULL VALIDATION SPLIT RUNTIME PASSED AND FROZEN"
    )


if __name__ == "__main__":
    main()