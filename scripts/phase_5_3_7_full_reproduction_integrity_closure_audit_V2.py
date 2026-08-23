#!/usr/bin/env python3
"""
Phase 5.3.7 — Full Reproduction Integrity Closure Audit V2

Purpose
-------
Run one READ-ONLY, cross-phase pre-launch audit before any Phase-5.4
production training is allowed.

This script intentionally does NOT reconstruct scientific behavior. It checks
the already-frozen artifacts/contracts against the key numerical and structural
anchors established in Phases 1–5.3.

Design principle
----------------
Unlike earlier fail-fast audit scripts, this closure audit COLLECTS all check
results first, writes a complete report and explicit launch authorization, and
only then exits PASS or BLOCKED.

No model construction.
No optimizer construction.
No negative generation.
No forward/backward.
No optimizer.step().
No validation scoring.
No test scoring.

If every critical check passes:
    PHASE 5.4 PRODUCTION TRAINING LAUNCH = ALLOWED

Otherwise:
    PHASE 5.4 PRODUCTION TRAINING LAUNCH = BLOCKED
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# =============================================================================
# Frozen paths
# =============================================================================

# Phase 1
PHASE1_INTERACTIONS_PATH = Path(
    "data/processed/interactions.parquet"
)

# Phase 2
PHASE2_TEMPORAL_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "interactions_itrs_temporal_split.parquet"
)

# Phase 3
PHASE3_NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/model_ready/"
    "node_index.parquet"
)

PHASE3_EDGE_INDEX_PATH = Path(
    "data/experimental/phase_3/model_ready/"
    "edge_index.npy"
)

PHASE3_EDGE_TYPE_PATH = Path(
    "data/experimental/phase_3/model_ready/"
    "edge_type.npy"
)

PHASE3_RELATION_INDEX_PATH = Path(
    "data/experimental/phase_3/model_ready/"
    "relation_index.csv"
)

# Phase 4
PHASE4_DOC2VEC_PATH = Path(
    "data/experimental/phase_4/doc2vec/vectors/"
    "doc2vec_vectors_all.npy"
)

PHASE4_LABEL_MULTIHOT_PATH = Path(
    "data/experimental/phase_4/description_labels/"
    "description_label_multihot.npz"
)

PHASE4_TREND_PTR_PATH = Path(
    "data/experimental/phase_4/trend_runtime/"
    "trend_period_ptr.npy"
)

PHASE4_TREND_STARTUP_PATH = Path(
    "data/experimental/phase_4/trend_runtime/"
    "trend_startup_node_indices.npy"
)

PHASE4_TREND_COUNTS_PATH = Path(
    "data/experimental/phase_4/trend_runtime/"
    "trend_period_startup_counts.npy"
)

# Phase 5 evaluation
PHASE5_EVAL_NEGATIVE_PATH = Path(
    "data/experimental/phase_5/model_ready/evaluation/"
    "t60_evaluation_negative_node_indices.npy"
)

PHASE5_EVAL_CASE_PATH = Path(
    "data/experimental/phase_5/model_ready/evaluation/"
    "t60_evaluation_case_manifest.parquet"
)

# Phase 5 epoch-0 training stream
PHASE5_EPOCH0_DIR = Path(
    "data/experimental/phase_5/training_runtime/epoch_0"
)

PHASE5_POSITIVE_ORDER_PATH = (
    PHASE5_EPOCH0_DIR
    / "canonical_training_positive_event_order.parquet"
)

PHASE5_EPOCH0_NEGATIVE_PATH = (
    PHASE5_EPOCH0_DIR
    / "epoch_0_training_negative_startup_local.npy"
)

PHASE5_EPOCH0_ORDER_PATH = (
    PHASE5_EPOCH0_DIR
    / "epoch_0_training_example_order.npy"
)

# Phase 5 contracts
CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

PHASE5_CONTRACT_PATHS = {
    "5.3.2a_training_execution": (
        CONTRACT_DIR
        / "phase_5_3_2a_training_execution_state_contract.json"
    ),
    "5.3.2c_production_controller": (
        CONTRACT_DIR
        / "phase_5_3_2c_production_driver_controller_contract.json"
    ),
    "5.3.3a_ranking": (
        CONTRACT_DIR
        / "phase_5_3_3a_validation_ranking_metric_contract.json"
    ),
    "5.3.3b_validation_preflight": (
        CONTRACT_DIR
        / "phase_5_3_3b_canonical_validation_scoring_preflight_contract.json"
    ),
    "5.3.3c_full_validation": (
        CONTRACT_DIR
        / "phase_5_3_3c_full_validation_runtime_contract.json"
    ),
    "5.3.3d_validation_checkpoint": (
        CONTRACT_DIR
        / "phase_5_3_3d_validation_checkpoint_integration_contract.json"
    ),
    "5.3.4_production_assembly": (
        CONTRACT_DIR
        / "phase_5_3_4_production_trainer_assembly_contract.json"
    ),
    "5.3.5a_representation": (
        CONTRACT_DIR
        / "phase_5_3_5a_epoch0_stream_representation_audit.json"
    ),
    "5.3.5_generalized_stream": (
        CONTRACT_DIR
        / "phase_5_3_5_generalized_training_stream_generator_contract.json"
    ),
    "5.3.6_launch_contract": (
        CONTRACT_DIR
        / "phase_5_3_6_production_training_launch_contract.json"
    ),
}

PHASE5_LAUNCH_CONFIG_PATH = (
    CONTRACT_DIR
    / "phase_5_4_production_training_launch_config.json"
)

# Outputs
AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_7"
)

CHECK_TABLE_PATH = (
    AUDIT_DIR
    / "full_reproduction_integrity_check_table.csv"
)

SECTION_SUMMARY_PATH = (
    AUDIT_DIR
    / "full_reproduction_integrity_section_summary.csv"
)

AUTHORIZATION_PATH = (
    CONTRACT_DIR
    / "phase_5_4_production_training_launch_authorization.json"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_7_full_reproduction_integrity_closure_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_7_full_reproduction_integrity_closure_manifest.json"
)

V1_ARCHIVE_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_7_v1_blocked"
)


# =============================================================================
# Frozen numerical anchors
# =============================================================================

# Phase 1 / 2
EXPECTED_PHASE1_ROWS = 1_208_051

# Canonical Phase-2 model-ready temporal artifact includes ONLY the
# T0..T60 experiment universe. Post-endpoint rows remain outside it.
EXPECTED_PHASE2_ROWS = 1_195_937
EXPECTED_TEMPORAL_EXPERIMENT_ROWS = 1_195_937
EXPECTED_POST_ENDPOINT_ROWS = 12_114

EXPECTED_T0_ROWS = 100_173
EXPECTED_T1_T59_ROWS = 1_073_249
EXPECTED_T60_ROWS = 22_515

EXPECTED_T60_VALIDATION_ROWS = 2_251
EXPECTED_T60_TEST_ROWS = 20_264

# Phase 3
EXPECTED_ROLE_NODES = 477_564
EXPECTED_INVESTOR_NODES = 165_975
EXPECTED_STARTUP_NODES = 311_589

EXPECTED_STRUCTURAL_EDGES = 158_818
EXPECTED_RELATION_CHANNELS = 12

# Phase 4
EXPECTED_DOC2VEC_SHAPE = (
    477_564,
    32,
)

EXPECTED_DOC2VEC_DTYPE = np.dtype(
    np.float32
)

EXPECTED_DOC2VEC_ZERO_ROWS = 2_670

EXPECTED_LABEL_SHAPE = (
    477_564,
    802,
)

EXPECTED_LABEL_NNZ = 1_230_068
EXPECTED_LABEL_ZERO_ROWS = 111_919

EXPECTED_TREND_PERIODS = (
    165_975
    * 60
)

EXPECTED_TREND_PTR_LENGTH = (
    EXPECTED_TREND_PERIODS
    + 1
)

EXPECTED_TREND_MENTIONS = 1_145_364

# Phase 4/5 model
EXPECTED_INITIAL_MODEL_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_TWO_STEP_MODEL_SHA256 = (
    "c41702cda99092a7fb63bb0a8227e658"
    "851b3ac4cbc373d90cdd6816eccdd196"
)

EXPECTED_TWO_STEP_OPTIMIZER_SHA256 = (
    "569a6691424ac32d0f252728750281cff"
    "d175a2b6b6c6ea1913f5f497200b00d"
)

# Phase 5 evaluation physical artifacts
EXPECTED_EVAL_NEGATIVE_FILE_SHA256 = (
    "7f98269d0291382dacfc783ffd66ad6d"
    "2c2f8775877d57dc0d83504e40d8716d"
)

EXPECTED_EVAL_CASE_FILE_SHA256 = (
    "44b4b7e1ec1b1978249318a080df02d0"
    "d9f617845263513594de64e71b969e0c"
)

EXPECTED_EVAL_NEGATIVE_SHAPE = (
    22_515,
    99,
)

# Phase 5 initial validation diagnostic
EXPECTED_INITIAL_VALIDATION_HR10 = (
    0.091514882275
)

EXPECTED_INITIAL_VALIDATION_NDCG10 = (
    0.040193099163
)

# Phase 5 training stream
EXPECTED_POSITIVE_ORDER_ROWS = 1_073_249

EXPECTED_EPOCH0_NEGATIVE_SHAPE = (
    1_073_249,
    4,
)

EXPECTED_EPOCH0_NEGATIVE_DTYPE = np.dtype(
    np.int32
)

EXPECTED_EPOCH0_ORDER_SHAPE = (
    5_366_245,
)

EXPECTED_EPOCH0_ORDER_DTYPE = np.dtype(
    np.int64
)

EXPECTED_POSITIVE_ORDER_SHA256 = (
    "73b074a80675793b811fbdc8a0609883"
    "c857fb2a687a2e01c31865ade5b509d1"
)

EXPECTED_EPOCH0_NEGATIVE_LOGICAL_SHA256 = (
    "47015b147b1949562c0f6737a6f3a3f2"
    "d7cabd2d2202e4e57456d884a1e23fe6"
)

EXPECTED_EPOCH0_ORDER_LOGICAL_SHA256 = (
    "0156be3ee623ade1ae696557337bfb324"
    "e9011adb7df8be9648ecb0a426c134e"
)

EXPECTED_EPOCH1_NEGATIVE_SHA256 = (
    "f7b415e0f305e049cc94c7e4261b6838"
    "00085f8fa7f7b2df11ff3658dea9d850"
)

EXPECTED_EPOCH1_ORDER_SHA256 = (
    "2da43d28e540ed48cb557ca889190b5d"
    "febc7c1207cbd3882df2fa14ca2a28d8"
)

EXPECTED_EPOCH19_NEGATIVE_SHA256 = (
    "06f9a11d8986ba9b7e0242fc41423478"
    "9ccc7adba91cab6da68ebc21e44682b7"
)

EXPECTED_EPOCH19_ORDER_SHA256 = (
    "c6236cb081ddba7f72eb0d36199500ac"
    "6b7695646e21a824fb84b85fae769329"
)

# Phase 5 production schedule
EXPECTED_EPOCHS = 20
EXPECTED_EXAMPLES_PER_EPOCH = 5_366_245
EXPECTED_BATCHES_PER_EPOCH = 10_481
EXPECTED_FINAL_BATCH_SIZE = 485
EXPECTED_TOTAL_OPTIMIZER_STEPS = 209_620


# =============================================================================
# Generic helpers
# =============================================================================

def banner(
    text: str,
) -> None:
    print(
        "\n"
        + "=" * 118
    )

    print(
        text
    )

    print(
        "=" * 118
    )


def load_json(
    path: Path,
) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(
            handle
        )


def safe_nested_get(
    payload: dict,
    keys: list[str],
    default=None,
):
    current = payload

    for key in keys:
        if (
            not isinstance(
                current,
                dict,
            )
            or key
            not in current
        ):
            return default

        current = current[
            key
        ]

    return current


def file_sha256(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:
        while True:
            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def exact_float(
    actual: Any,
    expected: float,
    tolerance: float = 5e-13,
) -> bool:
    try:
        value = float(
            actual
        )
    except Exception:
        return False

    return (
        math.isfinite(
            value
        )
        and abs(
            value
            - expected
        )
        <= tolerance
    )


# =============================================================================
# Audit collector
# =============================================================================

rows: list[dict] = []


def record(
    *,
    section: str,
    check: str,
    passed: bool,
    actual: Any,
    expected: Any,
    critical: bool = True,
    note: str = "",
) -> None:
    row = {
        "section": (
            section
        ),
        "check": (
            check
        ),
        "critical": (
            bool(
                critical
            )
        ),
        "result": (
            "PASS"
            if bool(
                passed
            )
            else "FAIL"
        ),
        "actual": (
            str(
                actual
            )
        ),
        "expected": (
            str(
                expected
            )
        ),
        "note": (
            note
        ),
    }

    rows.append(
        row
    )

    print(
        f"[{row['result']}] "
        f"{section:18s} "
        f"{check}"
    )

    if not passed:
        print(
            f"       actual:   {actual}"
        )

        print(
            f"       expected: {expected}"
        )

        if note:
            print(
                f"       note:     {note}"
            )


def guarded(
    *,
    section: str,
    check: str,
    expected: Any,
    fn: Callable[[], tuple[bool, Any]],
    critical: bool = True,
    note: str = "",
) -> None:
    try:
        (
            passed,
            actual,
        ) = fn()

        record(
            section=section,
            check=check,
            passed=(
                bool(
                    passed
                )
            ),
            actual=actual,
            expected=expected,
            critical=critical,
            note=note,
        )

    except Exception as exc:
        record(
            section=section,
            check=check,
            passed=False,
            actual=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            expected=expected,
            critical=critical,
            note=(
                (
                    note
                    + " | "
                )
                if note
                else ""
            )
            + (
                "Audit exception collected; "
                "closure continues."
            ),
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    banner(
        "PHASE 5.3.7 — "
        "FULL REPRODUCTION INTEGRITY CLOSURE AUDIT"
    )

    print(
        "Model instantiated:                   NO"
    )
    print(
        "Optimizer instantiated:               NO"
    )
    print(
        "Training batches executed:            0"
    )
    print(
        "Validation cases scored:              0"
    )
    print(
        "Test cases scored:                    0"
    )

    # =========================================================================
    # PHASE 1
    # =========================================================================

    banner(
        "PHASE 1 — CANONICAL INTERACTION INTEGRITY"
    )

    guarded(
        section="Phase 1",
        check="canonical interactions file exists",
        expected=True,
        fn=lambda: (
            PHASE1_INTERACTIONS_PATH.exists(),
            PHASE1_INTERACTIONS_PATH.exists(),
        ),
    )

    guarded(
        section="Phase 1",
        check="canonical interaction row count",
        expected=EXPECTED_PHASE1_ROWS,
        fn=lambda: (
            pq.ParquetFile(
                PHASE1_INTERACTIONS_PATH
            ).metadata.num_rows
            == EXPECTED_PHASE1_ROWS,
            pq.ParquetFile(
                PHASE1_INTERACTIONS_PATH
            ).metadata.num_rows,
        ),
    )

    # =========================================================================
    # PHASE 2
    # =========================================================================

    banner(
        "PHASE 2 — TEMPORAL SPLIT INTEGRITY"
    )

    guarded(
        section="Phase 2",
        check="temporal split file exists",
        expected=True,
        fn=lambda: (
            PHASE2_TEMPORAL_PATH.exists(),
            PHASE2_TEMPORAL_PATH.exists(),
        ),
    )

    guarded(
        section="Phase 2",
        check="temporal split total rows",
        expected=EXPECTED_PHASE2_ROWS,
        fn=lambda: (
            pq.ParquetFile(
                PHASE2_TEMPORAL_PATH
            ).metadata.num_rows
            == EXPECTED_PHASE2_ROWS,
            pq.ParquetFile(
                PHASE2_TEMPORAL_PATH
            ).metadata.num_rows,
        ),
    )

    temporal_df = None

    try:
        temporal_df = pd.read_parquet(
            PHASE2_TEMPORAL_PATH,
            columns=[
                "segment_number",
                "experiment_split",
            ],
        )

        segment_numeric = pd.to_numeric(
            temporal_df[
                "segment_number"
            ],
            errors="coerce",
        )

        nonnull_segment_count = int(
            segment_numeric.notna().sum()
        )

        # The canonical Phase-2 model-ready artifact intentionally contains
        # ONLY T0..T60 experimental rows. The 12,114 post-endpoint Phase-1
        # interactions are excluded from this file rather than retained with
        # null segment_number values.
        phase2_model_ready_rows = int(
            len(
                temporal_df
            )
        )

        excluded_post_endpoint_rows = int(
            EXPECTED_PHASE1_ROWS
            - phase2_model_ready_rows
        )

        t0_count = int(
            (
                segment_numeric
                == 0
            ).sum()
        )

        t1_t59_count = int(
            (
                segment_numeric
                .between(
                    1,
                    59,
                    inclusive="both",
                )
            ).sum()
        )

        t60_mask = (
            segment_numeric
            == 60
        )

        t60_count = int(
            t60_mask.sum()
        )

        split_norm = (
            temporal_df[
                "experiment_split"
            ]
            .astype(
                str
            )
            .str.strip()
            .str.casefold()
        )

        t60_validation_count = int(
            (
                t60_mask
                & (
                    split_norm
                    == "validation"
                )
            ).sum()
        )

        t60_test_count = int(
            (
                t60_mask
                & (
                    split_norm
                    == "test"
                )
            ).sum()
        )

        record(
            section="Phase 2",
            check="temporal experiment rows T0..T60",
            passed=(
                nonnull_segment_count
                == EXPECTED_TEMPORAL_EXPERIMENT_ROWS
            ),
            actual=nonnull_segment_count,
            expected=EXPECTED_TEMPORAL_EXPERIMENT_ROWS,
        )

        record(
            section="Phase 2",
            check="post-endpoint rows excluded from model-ready artifact",
            passed=(
                excluded_post_endpoint_rows
                == EXPECTED_POST_ENDPOINT_ROWS
            ),
            actual=excluded_post_endpoint_rows,
            expected=EXPECTED_POST_ENDPOINT_ROWS,
            note=(
                "Computed as Phase-1 canonical rows minus "
                "Phase-2 model-ready rows; post-endpoint rows "
                "are intentionally absent from the Phase-2 artifact."
            ),
        )

        record(
            section="Phase 2",
            check="T0 rows",
            passed=(
                t0_count
                == EXPECTED_T0_ROWS
            ),
            actual=t0_count,
            expected=EXPECTED_T0_ROWS,
        )

        record(
            section="Phase 2",
            check="T1..T59 training rows",
            passed=(
                t1_t59_count
                == EXPECTED_T1_T59_ROWS
            ),
            actual=t1_t59_count,
            expected=EXPECTED_T1_T59_ROWS,
        )

        record(
            section="Phase 2",
            check="T60 holdout rows",
            passed=(
                t60_count
                == EXPECTED_T60_ROWS
            ),
            actual=t60_count,
            expected=EXPECTED_T60_ROWS,
        )

        record(
            section="Phase 2",
            check="T60 validation rows",
            passed=(
                t60_validation_count
                == EXPECTED_T60_VALIDATION_ROWS
            ),
            actual=t60_validation_count,
            expected=EXPECTED_T60_VALIDATION_ROWS,
        )

        record(
            section="Phase 2",
            check="T60 test rows",
            passed=(
                t60_test_count
                == EXPECTED_T60_TEST_ROWS
            ),
            actual=t60_test_count,
            expected=EXPECTED_T60_TEST_ROWS,
        )

    except Exception as exc:
        for check, expected in (
            (
                "temporal experiment rows T0..T60",
                EXPECTED_TEMPORAL_EXPERIMENT_ROWS,
            ),
            (
                "post-endpoint rows excluded from model-ready artifact",
                EXPECTED_POST_ENDPOINT_ROWS,
            ),
            (
                "T0 rows",
                EXPECTED_T0_ROWS,
            ),
            (
                "T1..T59 training rows",
                EXPECTED_T1_T59_ROWS,
            ),
            (
                "T60 holdout rows",
                EXPECTED_T60_ROWS,
            ),
            (
                "T60 validation rows",
                EXPECTED_T60_VALIDATION_ROWS,
            ),
            (
                "T60 test rows",
                EXPECTED_T60_TEST_ROWS,
            ),
        ):
            record(
                section="Phase 2",
                check=check,
                passed=False,
                actual=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                expected=expected,
                note=(
                    "Temporal-column audit could not run."
                ),
            )

    del temporal_df

    # =========================================================================
    # PHASE 3
    # =========================================================================

    banner(
        "PHASE 3 — STRUCTURAL GRAPH INTEGRITY"
    )

    node_df = None

    try:
        node_df = pd.read_parquet(
            PHASE3_NODE_INDEX_PATH,
            columns=[
                "node_index",
                "node_type",
                "raw_entity_id",
            ],
        )

        node_numeric = pd.to_numeric(
            node_df[
                "node_index"
            ],
            errors="raise",
        ).to_numpy(
            dtype=np.int64
        )

        total_nodes = int(
            len(
                node_df
            )
        )

        contiguous = bool(
            np.array_equal(
                np.sort(
                    node_numeric
                ),
                np.arange(
                    EXPECTED_ROLE_NODES,
                    dtype=np.int64,
                ),
            )
        )

        investor_count = int(
            np.count_nonzero(
                (
                    node_numeric
                    >= 0
                )
                & (
                    node_numeric
                    < EXPECTED_INVESTOR_NODES
                )
            )
        )

        startup_count = int(
            np.count_nonzero(
                (
                    node_numeric
                    >= EXPECTED_INVESTOR_NODES
                )
                & (
                    node_numeric
                    < EXPECTED_ROLE_NODES
                )
            )
        )

        record(
            section="Phase 3",
            check="role-node count",
            passed=(
                total_nodes
                == EXPECTED_ROLE_NODES
            ),
            actual=total_nodes,
            expected=EXPECTED_ROLE_NODES,
        )

        record(
            section="Phase 3",
            check="node_index contiguous 0..477563",
            passed=contiguous,
            actual=contiguous,
            expected=True,
        )

        record(
            section="Phase 3",
            check="Investor numeric slice count",
            passed=(
                investor_count
                == EXPECTED_INVESTOR_NODES
            ),
            actual=investor_count,
            expected=EXPECTED_INVESTOR_NODES,
        )

        record(
            section="Phase 3",
            check="Startup numeric slice count",
            passed=(
                startup_count
                == EXPECTED_STARTUP_NODES
            ),
            actual=startup_count,
            expected=EXPECTED_STARTUP_NODES,
        )

    except Exception as exc:
        for check, expected in (
            (
                "role-node count",
                EXPECTED_ROLE_NODES,
            ),
            (
                "node_index contiguous 0..477563",
                True,
            ),
            (
                "Investor numeric slice count",
                EXPECTED_INVESTOR_NODES,
            ),
            (
                "Startup numeric slice count",
                EXPECTED_STARTUP_NODES,
            ),
        ):
            record(
                section="Phase 3",
                check=check,
                passed=False,
                actual=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                expected=expected,
            )

    del node_df

    try:
        edge_index = np.load(
            PHASE3_EDGE_INDEX_PATH,
            mmap_mode="r",
        )

        edge_type = np.load(
            PHASE3_EDGE_TYPE_PATH,
            mmap_mode="r",
        )

        edge_index_ok = (
            edge_index.ndim
            == 2
            and (
                (
                    edge_index.shape[
                        0
                    ]
                    == 2
                    and edge_index.shape[
                        1
                    ]
                    == EXPECTED_STRUCTURAL_EDGES
                )
                or (
                    edge_index.shape[
                        1
                    ]
                    == 2
                    and edge_index.shape[
                        0
                    ]
                    == EXPECTED_STRUCTURAL_EDGES
                )
            )
        )

        edge_count = (
            int(
                edge_index.shape[
                    1
                ]
            )
            if (
                edge_index.ndim
                == 2
                and edge_index.shape[
                    0
                ]
                == 2
            )
            else (
                int(
                    edge_index.shape[
                        0
                    ]
                )
                if edge_index.ndim == 2
                else -1
            )
        )

        record(
            section="Phase 3",
            check="structural edge count",
            passed=(
                edge_index_ok
                and edge_count
                == EXPECTED_STRUCTURAL_EDGES
            ),
            actual=(
                tuple(
                    edge_index.shape
                )
            ),
            expected=(
                f"2 x {EXPECTED_STRUCTURAL_EDGES} "
                "or transpose"
            ),
        )

        record(
            section="Phase 3",
            check="edge_type count",
            passed=(
                edge_type.ndim
                == 1
                and len(
                    edge_type
                )
                == EXPECTED_STRUCTURAL_EDGES
            ),
            actual=(
                tuple(
                    edge_type.shape
                )
            ),
            expected=(
                EXPECTED_STRUCTURAL_EDGES,
            ),
        )

    except Exception as exc:
        record(
            section="Phase 3",
            check="structural edge count",
            passed=False,
            actual=(
                f"{type(exc).__name__}: {exc}"
            ),
            expected=EXPECTED_STRUCTURAL_EDGES,
        )

        record(
            section="Phase 3",
            check="edge_type count",
            passed=False,
            actual=(
                f"{type(exc).__name__}: {exc}"
            ),
            expected=EXPECTED_STRUCTURAL_EDGES,
        )

    guarded(
        section="Phase 3",
        check="relation channel count",
        expected=EXPECTED_RELATION_CHANNELS,
        fn=lambda: (
            len(
                pd.read_csv(
                    PHASE3_RELATION_INDEX_PATH
                )
            )
            == EXPECTED_RELATION_CHANNELS,
            len(
                pd.read_csv(
                    PHASE3_RELATION_INDEX_PATH
                )
            ),
        ),
    )

    # =========================================================================
    # PHASE 4
    # =========================================================================

    banner(
        "PHASE 4 — MODEL-INPUT ARTIFACT INTEGRITY"
    )

    try:
        doc2vec = np.load(
            PHASE4_DOC2VEC_PATH,
            mmap_mode="r",
        )

        doc_shape = tuple(
            int(
                value
            )
            for value
            in doc2vec.shape
        )

        doc_dtype = np.dtype(
            doc2vec.dtype
        )

        zero_rows = int(
            np.count_nonzero(
                np.all(
                    np.asarray(
                        doc2vec
                    )
                    == 0,
                    axis=1,
                )
            )
        )

        record(
            section="Phase 4",
            check="Doc2Vec shape",
            passed=(
                doc_shape
                == EXPECTED_DOC2VEC_SHAPE
            ),
            actual=doc_shape,
            expected=EXPECTED_DOC2VEC_SHAPE,
        )

        record(
            section="Phase 4",
            check="Doc2Vec dtype float32",
            passed=(
                doc_dtype
                == EXPECTED_DOC2VEC_DTYPE
            ),
            actual=doc_dtype,
            expected=EXPECTED_DOC2VEC_DTYPE,
        )

        record(
            section="Phase 4",
            check="Doc2Vec zero-vector rows",
            passed=(
                zero_rows
                == EXPECTED_DOC2VEC_ZERO_ROWS
            ),
            actual=zero_rows,
            expected=EXPECTED_DOC2VEC_ZERO_ROWS,
        )

    except Exception as exc:
        for check, expected in (
            (
                "Doc2Vec shape",
                EXPECTED_DOC2VEC_SHAPE,
            ),
            (
                "Doc2Vec dtype float32",
                EXPECTED_DOC2VEC_DTYPE,
            ),
            (
                "Doc2Vec zero-vector rows",
                EXPECTED_DOC2VEC_ZERO_ROWS,
            ),
        ):
            record(
                section="Phase 4",
                check=check,
                passed=False,
                actual=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                expected=expected,
            )

    try:
        from scipy import sparse

        label_matrix = sparse.load_npz(
            PHASE4_LABEL_MULTIHOT_PATH
        ).tocsr()

        label_shape = tuple(
            int(
                value
            )
            for value
            in label_matrix.shape
        )

        label_nnz = int(
            label_matrix.nnz
        )

        label_zero_rows = int(
            np.count_nonzero(
                np.diff(
                    label_matrix.indptr
                )
                == 0
            )
        )

        record(
            section="Phase 4",
            check="description-label matrix shape",
            passed=(
                label_shape
                == EXPECTED_LABEL_SHAPE
            ),
            actual=label_shape,
            expected=EXPECTED_LABEL_SHAPE,
        )

        record(
            section="Phase 4",
            check="description-label nnz",
            passed=(
                label_nnz
                == EXPECTED_LABEL_NNZ
            ),
            actual=label_nnz,
            expected=EXPECTED_LABEL_NNZ,
        )

        record(
            section="Phase 4",
            check="description-label zero rows",
            passed=(
                label_zero_rows
                == EXPECTED_LABEL_ZERO_ROWS
            ),
            actual=label_zero_rows,
            expected=EXPECTED_LABEL_ZERO_ROWS,
        )

    except Exception as exc:
        for check, expected in (
            (
                "description-label matrix shape",
                EXPECTED_LABEL_SHAPE,
            ),
            (
                "description-label nnz",
                EXPECTED_LABEL_NNZ,
            ),
            (
                "description-label zero rows",
                EXPECTED_LABEL_ZERO_ROWS,
            ),
        ):
            record(
                section="Phase 4",
                check=check,
                passed=False,
                actual=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                expected=expected,
            )

    try:
        trend_ptr = np.load(
            PHASE4_TREND_PTR_PATH,
            mmap_mode="r",
        )

        trend_startup = np.load(
            PHASE4_TREND_STARTUP_PATH,
            mmap_mode="r",
        )

        trend_counts = np.load(
            PHASE4_TREND_COUNTS_PATH,
            mmap_mode="r",
        )

        ptr_length = int(
            len(
                trend_ptr
            )
        )

        startup_mentions = int(
            len(
                trend_startup
            )
        )

        counts_length = int(
            len(
                trend_counts
            )
        )

        counts_sum = int(
            np.asarray(
                trend_counts,
                dtype=np.int64,
            ).sum()
        )

        ptr_terminal = int(
            trend_ptr[
                -1
            ]
        )

        record(
            section="Phase 4",
            check="trend period pointer length",
            passed=(
                ptr_length
                == EXPECTED_TREND_PTR_LENGTH
            ),
            actual=ptr_length,
            expected=EXPECTED_TREND_PTR_LENGTH,
        )

        record(
            section="Phase 4",
            check="trend period-count length",
            passed=(
                counts_length
                == EXPECTED_TREND_PERIODS
            ),
            actual=counts_length,
            expected=EXPECTED_TREND_PERIODS,
        )

        record(
            section="Phase 4",
            check="trend startup mentions",
            passed=(
                startup_mentions
                == EXPECTED_TREND_MENTIONS
            ),
            actual=startup_mentions,
            expected=EXPECTED_TREND_MENTIONS,
        )

        record(
            section="Phase 4",
            check="trend counts sum == startup mentions",
            passed=(
                counts_sum
                == EXPECTED_TREND_MENTIONS
            ),
            actual=counts_sum,
            expected=EXPECTED_TREND_MENTIONS,
        )

        record(
            section="Phase 4",
            check="trend pointer terminal == startup mentions",
            passed=(
                ptr_terminal
                == EXPECTED_TREND_MENTIONS
            ),
            actual=ptr_terminal,
            expected=EXPECTED_TREND_MENTIONS,
        )

    except Exception as exc:
        for check, expected in (
            (
                "trend period pointer length",
                EXPECTED_TREND_PTR_LENGTH,
            ),
            (
                "trend period-count length",
                EXPECTED_TREND_PERIODS,
            ),
            (
                "trend startup mentions",
                EXPECTED_TREND_MENTIONS,
            ),
            (
                "trend counts sum == startup mentions",
                EXPECTED_TREND_MENTIONS,
            ),
            (
                "trend pointer terminal == startup mentions",
                EXPECTED_TREND_MENTIONS,
            ),
        ):
            record(
                section="Phase 4",
                check=check,
                passed=False,
                actual=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                expected=expected,
            )

    # =========================================================================
    # PHASE 5 — EVALUATION ARTIFACTS
    # =========================================================================

    banner(
        "PHASE 5 — EVALUATION ARTIFACT INTEGRITY"
    )

    guarded(
        section="Phase 5 Eval",
        check="evaluation negative physical SHA256",
        expected=EXPECTED_EVAL_NEGATIVE_FILE_SHA256,
        fn=lambda: (
            file_sha256(
                PHASE5_EVAL_NEGATIVE_PATH
            )
            == EXPECTED_EVAL_NEGATIVE_FILE_SHA256,
            file_sha256(
                PHASE5_EVAL_NEGATIVE_PATH
            ),
        ),
    )

    guarded(
        section="Phase 5 Eval",
        check="evaluation case-manifest physical SHA256",
        expected=EXPECTED_EVAL_CASE_FILE_SHA256,
        fn=lambda: (
            file_sha256(
                PHASE5_EVAL_CASE_PATH
            )
            == EXPECTED_EVAL_CASE_FILE_SHA256,
            file_sha256(
                PHASE5_EVAL_CASE_PATH
            ),
        ),
    )

    try:
        eval_negative = np.load(
            PHASE5_EVAL_NEGATIVE_PATH,
            mmap_mode="r",
        )

        eval_case_rows = (
            pq.ParquetFile(
                PHASE5_EVAL_CASE_PATH
            ).metadata.num_rows
        )

        record(
            section="Phase 5 Eval",
            check="evaluation negative matrix shape",
            passed=(
                tuple(
                    eval_negative.shape
                )
                == EXPECTED_EVAL_NEGATIVE_SHAPE
            ),
            actual=(
                tuple(
                    eval_negative.shape
                )
            ),
            expected=EXPECTED_EVAL_NEGATIVE_SHAPE,
        )

        record(
            section="Phase 5 Eval",
            check="evaluation case count",
            passed=(
                int(
                    eval_case_rows
                )
                == EXPECTED_T60_ROWS
            ),
            actual=eval_case_rows,
            expected=EXPECTED_T60_ROWS,
        )

    except Exception as exc:
        record(
            section="Phase 5 Eval",
            check="evaluation negative matrix shape",
            passed=False,
            actual=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            expected=EXPECTED_EVAL_NEGATIVE_SHAPE,
        )

        record(
            section="Phase 5 Eval",
            check="evaluation case count",
            passed=False,
            actual=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            expected=EXPECTED_T60_ROWS,
        )

    # =========================================================================
    # PHASE 5 — EPOCH-0 STREAM PHYSICAL REPRESENTATION
    # =========================================================================

    banner(
        "PHASE 5 — TRAINING STREAM PHYSICAL INTEGRITY"
    )

    guarded(
        section="Phase 5 Stream",
        check="positive-order row count",
        expected=EXPECTED_POSITIVE_ORDER_ROWS,
        fn=lambda: (
            pq.ParquetFile(
                PHASE5_POSITIVE_ORDER_PATH
            ).metadata.num_rows
            == EXPECTED_POSITIVE_ORDER_ROWS,
            pq.ParquetFile(
                PHASE5_POSITIVE_ORDER_PATH
            ).metadata.num_rows,
        ),
    )

    try:
        epoch0_negative = np.load(
            PHASE5_EPOCH0_NEGATIVE_PATH,
            mmap_mode="r",
        )

        epoch0_order = np.load(
            PHASE5_EPOCH0_ORDER_PATH,
            mmap_mode="r",
        )

        record(
            section="Phase 5 Stream",
            check="epoch-0 negative shape",
            passed=(
                tuple(
                    epoch0_negative.shape
                )
                == EXPECTED_EPOCH0_NEGATIVE_SHAPE
            ),
            actual=(
                tuple(
                    epoch0_negative.shape
                )
            ),
            expected=EXPECTED_EPOCH0_NEGATIVE_SHAPE,
        )

        record(
            section="Phase 5 Stream",
            check="epoch-0 negative dtype",
            passed=(
                np.dtype(
                    epoch0_negative.dtype
                )
                == EXPECTED_EPOCH0_NEGATIVE_DTYPE
            ),
            actual=(
                np.dtype(
                    epoch0_negative.dtype
                )
            ),
            expected=EXPECTED_EPOCH0_NEGATIVE_DTYPE,
        )

        record(
            section="Phase 5 Stream",
            check="epoch-0 negative C-contiguous",
            passed=(
                bool(
                    epoch0_negative.flags.c_contiguous
                )
            ),
            actual=(
                bool(
                    epoch0_negative.flags.c_contiguous
                )
            ),
            expected=True,
        )

        record(
            section="Phase 5 Stream",
            check="epoch-0 order shape",
            passed=(
                tuple(
                    epoch0_order.shape
                )
                == EXPECTED_EPOCH0_ORDER_SHAPE
            ),
            actual=(
                tuple(
                    epoch0_order.shape
                )
            ),
            expected=EXPECTED_EPOCH0_ORDER_SHAPE,
        )

        record(
            section="Phase 5 Stream",
            check="epoch-0 order dtype",
            passed=(
                np.dtype(
                    epoch0_order.dtype
                )
                == EXPECTED_EPOCH0_ORDER_DTYPE
            ),
            actual=(
                np.dtype(
                    epoch0_order.dtype
                )
            ),
            expected=EXPECTED_EPOCH0_ORDER_DTYPE,
        )

    except Exception as exc:
        for check, expected in (
            (
                "epoch-0 negative shape",
                EXPECTED_EPOCH0_NEGATIVE_SHAPE,
            ),
            (
                "epoch-0 negative dtype",
                EXPECTED_EPOCH0_NEGATIVE_DTYPE,
            ),
            (
                "epoch-0 negative C-contiguous",
                True,
            ),
            (
                "epoch-0 order shape",
                EXPECTED_EPOCH0_ORDER_SHAPE,
            ),
            (
                "epoch-0 order dtype",
                EXPECTED_EPOCH0_ORDER_DTYPE,
            ),
        ):
            record(
                section="Phase 5 Stream",
                check=check,
                passed=False,
                actual=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                expected=expected,
            )

    # =========================================================================
    # PHASE 5 — CONTRACT STATUS CLOSURE
    # =========================================================================

    banner(
        "PHASE 5 — FROZEN CONTRACT STATUS CLOSURE"
    )

    contracts: dict[str, dict] = {}

    for (
        name,
        path,
    ) in PHASE5_CONTRACT_PATHS.items():
        try:
            payload = load_json(
                path
            )

            contracts[
                name
            ] = payload

            status = str(
                payload.get(
                    "status",
                    "",
                )
            )

            expected_status = (
                "DIAGNOSIS_COMPLETE"
                if name
                == "5.3.5a_representation"
                else "FROZEN"
            )

            record(
                section="Phase 5 Contracts",
                check=(
                    f"{name} status"
                ),
                passed=(
                    status
                    == expected_status
                ),
                actual=status,
                expected=expected_status,
            )

        except Exception as exc:
            record(
                section="Phase 5 Contracts",
                check=(
                    f"{name} status"
                ),
                passed=False,
                actual=(
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                expected=(
                    "DIAGNOSIS_COMPLETE"
                    if name
                    == "5.3.5a_representation"
                    else "FROZEN"
                ),
            )

    # =========================================================================
    # PHASE 5 — KEY CONTRACT CROSS-CHECKS
    # =========================================================================

    banner(
        "PHASE 5 — CROSS-CONTRACT NUMERICAL CONSISTENCY"
    )

    assembly = contracts.get(
        "5.3.4_production_assembly",
        {},
    )

    stream_contract = contracts.get(
        "5.3.5_generalized_stream",
        {},
    )

    validation_contract = contracts.get(
        "5.3.3c_full_validation",
        {},
    )

    launch_contract = contracts.get(
        "5.3.6_launch_contract",
        {},
    )

    # Initial model
    # Authoritative owner: frozen Phase-5.3.2a execution contract.
    # This avoids requiring the same fingerprint to be redundantly embedded
    # inside the later 5.3.4 assembly contract.
    execution_contract = contracts.get(
        "5.3.2a_training_execution",
        {},
    )

    canonical_initial_sha = safe_nested_get(
        execution_contract,
        [
            "frozen_numerical_fingerprints",
            "canonical_initial_model_sha256",
        ],
    )

    record(
        section="Phase 5 Cross",
        check="canonical initial model SHA256",
        passed=(
            canonical_initial_sha
            == EXPECTED_INITIAL_MODEL_SHA256
        ),
        actual=canonical_initial_sha,
        expected=EXPECTED_INITIAL_MODEL_SHA256,
        note=(
            "Authoritative source: "
            "Phase-5.3.2a frozen_numerical_fingerprints."
        ),
    )

    # Two-step anchors
    two_step_model = safe_nested_get(
        assembly,
        [
            "real_training_dry_run",
            "post_two_step_model_sha256",
        ],
    )

    two_step_optimizer = safe_nested_get(
        assembly,
        [
            "real_training_dry_run",
            "post_two_step_optimizer_sha256",
        ],
    )

    record(
        section="Phase 5 Cross",
        check="two-step model SHA256",
        passed=(
            two_step_model
            == EXPECTED_TWO_STEP_MODEL_SHA256
        ),
        actual=two_step_model,
        expected=EXPECTED_TWO_STEP_MODEL_SHA256,
    )

    record(
        section="Phase 5 Cross",
        check="two-step optimizer SHA256",
        passed=(
            two_step_optimizer
            == EXPECTED_TWO_STEP_OPTIMIZER_SHA256
        ),
        actual=two_step_optimizer,
        expected=EXPECTED_TWO_STEP_OPTIMIZER_SHA256,
    )

    # Full initial validation
    initial_validation = safe_nested_get(
        validation_contract,
        [
            "canonical_initial_validation",
        ],
        default={},
    )

    initial_hr = (
        initial_validation.get(
            "HR@10"
        )
        if isinstance(
            initial_validation,
            dict,
        )
        else None
    )

    initial_ndcg = (
        initial_validation.get(
            "NDCG@10"
        )
        if isinstance(
            initial_validation,
            dict,
        )
        else None
    )

    record(
        section="Phase 5 Cross",
        check="canonical-initial validation HR@10",
        passed=exact_float(
            initial_hr,
            EXPECTED_INITIAL_VALIDATION_HR10,
        ),
        actual=initial_hr,
        expected=EXPECTED_INITIAL_VALIDATION_HR10,
    )

    record(
        section="Phase 5 Cross",
        check="canonical-initial validation NDCG@10",
        passed=exact_float(
            initial_ndcg,
            EXPECTED_INITIAL_VALIDATION_NDCG10,
        ),
        actual=initial_ndcg,
        expected=EXPECTED_INITIAL_VALIDATION_NDCG10,
    )

    # Stream contract
    stream_checks = [
        (
            "positive-order logical SHA256",
            safe_nested_get(
                stream_contract,
                [
                    "positive_stream",
                    "logical_sha256",
                ],
            ),
            EXPECTED_POSITIVE_ORDER_SHA256,
        ),
        (
            "epoch-0 negative logical SHA256",
            safe_nested_get(
                stream_contract,
                [
                    "epoch0_exact_regression",
                    "negative_sha256",
                ],
            ),
            EXPECTED_EPOCH0_NEGATIVE_LOGICAL_SHA256,
        ),
        (
            "epoch-0 order logical SHA256",
            safe_nested_get(
                stream_contract,
                [
                    "epoch0_exact_regression",
                    "order_sha256",
                ],
            ),
            EXPECTED_EPOCH0_ORDER_LOGICAL_SHA256,
        ),
        (
            "epoch-1 negative logical SHA256",
            safe_nested_get(
                stream_contract,
                [
                    "epoch1_frozen_fingerprint",
                    "negative_sha256",
                ],
            ),
            EXPECTED_EPOCH1_NEGATIVE_SHA256,
        ),
        (
            "epoch-1 order logical SHA256",
            safe_nested_get(
                stream_contract,
                [
                    "epoch1_frozen_fingerprint",
                    "order_sha256",
                ],
            ),
            EXPECTED_EPOCH1_ORDER_SHA256,
        ),
        (
            "epoch-19 negative logical SHA256",
            safe_nested_get(
                stream_contract,
                [
                    "epoch19_frozen_fingerprint",
                    "negative_sha256",
                ],
            ),
            EXPECTED_EPOCH19_NEGATIVE_SHA256,
        ),
        (
            "epoch-19 order logical SHA256",
            safe_nested_get(
                stream_contract,
                [
                    "epoch19_frozen_fingerprint",
                    "order_sha256",
                ],
            ),
            EXPECTED_EPOCH19_ORDER_SHA256,
        ),
    ]

    for (
        name,
        actual,
        expected,
    ) in stream_checks:
        record(
            section="Phase 5 Cross",
            check=name,
            passed=(
                actual
                == expected
            ),
            actual=actual,
            expected=expected,
        )

    # Launch schedule
    schedule = safe_nested_get(
        launch_contract,
        [
            "production_schedule",
        ],
        default={},
    )

    schedule_checks = [
        (
            "production epochs",
            schedule.get(
                "epochs"
            ),
            EXPECTED_EPOCHS,
        ),
        (
            "production examples per epoch",
            schedule.get(
                "examples_per_epoch"
            ),
            EXPECTED_EXAMPLES_PER_EPOCH,
        ),
        (
            "production batches per epoch",
            schedule.get(
                "batches_per_epoch"
            ),
            EXPECTED_BATCHES_PER_EPOCH,
        ),
        (
            "production total optimizer steps",
            schedule.get(
                "total_optimizer_steps"
            ),
            EXPECTED_TOTAL_OPTIMIZER_STEPS,
        ),
    ]

    for (
        name,
        actual,
        expected,
    ) in schedule_checks:
        record(
            section="Phase 5 Cross",
            check=name,
            passed=(
                actual
                == expected
            ),
            actual=actual,
            expected=expected,
        )

    # =========================================================================
    # LAUNCH CONFIG
    # =========================================================================

    banner(
        "PHASE 5.4 — LOCKED LAUNCH CONFIG CONSISTENCY"
    )

    launch_config = {}

    try:
        launch_config = load_json(
            PHASE5_LAUNCH_CONFIG_PATH
        )

        record(
            section="Launch Config",
            check="launch config status is locked pending 5.3.7",
            passed=(
                launch_config.get(
                    "status"
                )
                == (
                    "LOCKED_PENDING_PHASE_5_3_7_"
                    "INTEGRITY_CLOSURE"
                )
            ),
            actual=(
                launch_config.get(
                    "status"
                )
            ),
            expected=(
                "LOCKED_PENDING_PHASE_5_3_7_"
                "INTEGRITY_CLOSURE"
            ),
        )

        record(
            section="Launch Config",
            check="reference device CPU",
            passed=(
                launch_config.get(
                    "reference_device"
                )
                == "CPU"
            ),
            actual=(
                launch_config.get(
                    "reference_device"
                )
            ),
            expected="CPU",
        )

        record(
            section="Launch Config",
            check="initial model SHA256",
            passed=(
                launch_config.get(
                    "initial_model_sha256"
                )
                == EXPECTED_INITIAL_MODEL_SHA256
            ),
            actual=(
                launch_config.get(
                    "initial_model_sha256"
                )
            ),
            expected=EXPECTED_INITIAL_MODEL_SHA256,
        )

        record(
            section="Launch Config",
            check="epochs 20",
            passed=(
                launch_config.get(
                    "epochs"
                )
                == EXPECTED_EPOCHS
            ),
            actual=(
                launch_config.get(
                    "epochs"
                )
            ),
            expected=EXPECTED_EPOCHS,
        )

        record(
            section="Launch Config",
            check="total optimizer steps 209620",
            passed=(
                launch_config.get(
                    "total_optimizer_steps"
                )
                == EXPECTED_TOTAL_OPTIMIZER_STEPS
            ),
            actual=(
                launch_config.get(
                    "total_optimizer_steps"
                )
            ),
            expected=EXPECTED_TOTAL_OPTIMIZER_STEPS,
        )

        record(
            section="Launch Config",
            check="pre-5.3.7 training_allowed_now false",
            passed=(
                safe_nested_get(
                    launch_config,
                    [
                        "launch_gate",
                        "training_allowed_now",
                    ],
                )
                is False
            ),
            actual=(
                safe_nested_get(
                    launch_config,
                    [
                        "launch_gate",
                        "training_allowed_now",
                    ],
                )
            ),
            expected=False,
        )

    except Exception as exc:
        record(
            section="Launch Config",
            check="launch config readable",
            passed=False,
            actual=(
                f"{type(exc).__name__}: "
                f"{exc}"
            ),
            expected=True,
        )

    # =========================================================================
    # GLOBAL CLOSURE
    # =========================================================================

    banner(
        "GLOBAL REPRODUCTION CLOSURE"
    )

    check_df = pd.DataFrame(
        rows
    )

    critical_df = check_df.loc[
        check_df[
            "critical"
        ]
        == True
    ]

    failed_critical = critical_df.loc[
        critical_df[
            "result"
        ]
        != "PASS"
    ]

    all_critical_pass = (
        len(
            failed_critical
        )
        == 0
    )

    section_summary_df = (
        check_df.groupby(
            [
                "section",
            ],
            sort=False,
        )
        .agg(
            checks=(
                "check",
                "count",
            ),
            passed=(
                "result",
                lambda series: int(
                    (
                        series
                        == "PASS"
                    ).sum()
                ),
            ),
            failed=(
                "result",
                lambda series: int(
                    (
                        series
                        != "PASS"
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    section_summary_df[
        "status"
    ] = np.where(
        section_summary_df[
            "failed"
        ]
        == 0,
        "PASS",
        "FAIL",
    )

    print()
    print(
        section_summary_df.to_string(
            index=False
        )
    )

    print()

    if all_critical_pass:
        print(
            "PHASE 5.4 PRODUCTION TRAINING LAUNCH: ALLOWED"
        )
    else:
        print(
            "PHASE 5.4 PRODUCTION TRAINING LAUNCH: BLOCKED"
        )

        print()
        print(
            "Failed critical checks:"
        )

        print(
            failed_critical[
                [
                    "section",
                    "check",
                    "actual",
                    "expected",
                ]
            ].to_string(
                index=False
            )
        )

    # =========================================================================
    # WRITE OUTPUTS EVEN IF BLOCKED
    # =========================================================================

    banner(
        "ARCHIVE PRIOR V1 BLOCKED CLOSURE OUTPUTS"
    )

    V1_ARCHIVE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Preserve the prior V1 blocked result before V2 updates canonical
    # Phase-5.3.7 closure artifacts.
    archive_pairs = [
        (
            CHECK_TABLE_PATH,
            V1_ARCHIVE_DIR
            / "full_reproduction_integrity_check_table_V1_blocked.csv",
        ),
        (
            SECTION_SUMMARY_PATH,
            V1_ARCHIVE_DIR
            / "full_reproduction_integrity_section_summary_V1_blocked.csv",
        ),
        (
            AUTHORIZATION_PATH,
            V1_ARCHIVE_DIR
            / "phase_5_4_production_training_launch_authorization_V1_blocked.json",
        ),
        (
            CONTRACT_PATH,
            V1_ARCHIVE_DIR
            / "phase_5_3_7_full_reproduction_integrity_closure_contract_V1_blocked.json",
        ),
        (
            MANIFEST_PATH,
            V1_ARCHIVE_DIR
            / "phase_5_3_7_full_reproduction_integrity_closure_manifest_V1_blocked.json",
        ),
    ]

    for source_path, archive_path in archive_pairs:
        if source_path.exists():
            archive_path.write_bytes(
                source_path.read_bytes()
            )

            print(
                f"ARCHIVED  {source_path} -> {archive_path}"
            )

    banner(
        "WRITE PHASE-5.3.7 V2 CLOSURE OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    check_df.to_csv(
        CHECK_TABLE_PATH,
        index=False,
    )

    section_summary_df.to_csv(
        SECTION_SUMMARY_PATH,
        index=False,
    )

    authorization = {
        "schema_version": (
            "ITRS_PHASE5_PRODUCTION_LAUNCH_AUTHORIZATION_V1"
        ),
        "phase": (
            "5.4"
        ),
        "status": (
            "ALLOWED"
            if all_critical_pass
            else "BLOCKED"
        ),
        "authorized_by_phase": (
            "5.3.7"
        ),
        "closure_script_version": (
            "V2"
        ),
        "v1_blocked_result_preserved": (
            True
        ),
        "v2_audit_corrections": [
            (
                "Phase-2 model-ready row count is 1,195,937; "
                "12,114 post-endpoint rows are excluded, not null rows."
            ),
            (
                "Canonical initial model SHA is read from the explicit "
                "Phase-5.3.2a frozen_numerical_fingerprints field."
            ),
        ],
        "critical_checks": int(
            len(
                critical_df
            )
        ),
        "critical_checks_passed": int(
            (
                critical_df[
                    "result"
                ]
                == "PASS"
            ).sum()
        ),
        "critical_checks_failed": int(
            len(
                failed_critical
            )
        ),
        "training_allowed": (
            bool(
                all_critical_pass
            )
        ),
        "test_access_during_training": (
            False
        ),
        "first_real_trained_metrics_after_display_epoch": (
            1
        ),
        "final_test_after_all_20_epochs": (
            True
        ),
        "failed_critical_checks": (
            failed_critical[
                [
                    "section",
                    "check",
                    "actual",
                    "expected",
                ]
            ].to_dict(
                orient="records"
            )
        ),
    }

    AUTHORIZATION_PATH.write_text(
        json.dumps(
            authorization,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    contract = {
        "phase": (
            "5.3.7"
        ),
        "script_version": (
            "V2"
        ),
        "title": (
            "Full Reproduction Integrity Closure Audit"
        ),
        "status": (
            "FROZEN"
            if all_critical_pass
            else "BLOCKED"
        ),
        "read_only": (
            True
        ),
        "model_instantiated": (
            False
        ),
        "optimizer_instantiated": (
            False
        ),
        "optimizer_steps": (
            0
        ),
        "validation_cases_scored": (
            0
        ),
        "test_cases_scored": (
            0
        ),
        "critical_checks": int(
            len(
                critical_df
            )
        ),
        "critical_checks_passed": int(
            (
                critical_df[
                    "result"
                ]
                == "PASS"
            ).sum()
        ),
        "critical_checks_failed": int(
            len(
                failed_critical
            )
        ),
        "launch_authorization": (
            "ALLOWED"
            if all_critical_pass
            else "BLOCKED"
        ),
        "next_phase": (
            "5.4_actual_20_epoch_training"
            if all_critical_pass
            else "resolve_integrity_failures"
        ),
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
            "5.3.7"
        ),
        "script_version": (
            "V2"
        ),
        "status": (
            "FULL_REPRODUCTION_INTEGRITY_CLOSURE_"
            + (
                "PASSED_AND_FROZEN"
                if all_critical_pass
                else "BLOCKED"
            )
        ),
        "critical_checks": int(
            len(
                critical_df
            )
        ),
        "critical_checks_failed": int(
            len(
                failed_critical
            )
        ),
        "phase_5_4_training_allowed": (
            bool(
                all_critical_pass
            )
        ),
        "authorization": str(
            AUTHORIZATION_PATH
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
        CHECK_TABLE_PATH,
        SECTION_SUMMARY_PATH,
        AUTHORIZATION_PATH,
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
        "PHASE 5.3.7 FINAL STATUS"
    )

    print(
        f"Critical checks:                      "
        f"{len(critical_df):,}"
    )

    print(
        f"Critical checks passed:               "
        f"{int((critical_df['result'] == 'PASS').sum()):,}"
    )

    print(
        f"Critical checks failed:               "
        f"{len(failed_critical):,}"
    )

    print()

    print(
        "Model instantiated:                   NO"
    )

    print(
        "Optimizer instantiated:               NO"
    )

    print(
        "Training steps executed:              0"
    )

    print(
        "Validation cases scored:              0"
    )

    print(
        "Test cases scored:                    0"
    )

    print()

    print(
        "PHASE 5.4 PRODUCTION TRAINING LAUNCH: "
        + (
            "ALLOWED"
            if all_critical_pass
            else "BLOCKED"
        )
    )

    if all_critical_pass:
        banner(
            "PHASE 5.3.7 COMPLETE / "
            "FULL REPRODUCTION INTEGRITY CLOSURE PASSED AND FROZEN"
        )

        return

    banner(
        "PHASE 5.3.7 BLOCKED / "
        "RESOLVE FAILED CRITICAL CHECKS BEFORE TRAINING"
    )

    sys.exit(
        1
    )


if __name__ == "__main__":
    main()