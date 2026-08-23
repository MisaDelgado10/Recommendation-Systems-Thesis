"""
Phase 5.3.3b — Canonical Real-Model Validation Scoring Preflight

Purpose
-------
Exercise the complete REAL T60 validation-scoring path on a deterministic,
bounded subset of frozen validation cases before integrating validation into
the production 20-epoch trainer.

This phase uses:
    - the canonical seed-42 initial model state
    - exact Phase-4 numerical methods
    - real Phase-3 structural graph
    - real Phase-4 Doc2Vec/category features
    - real T0..T59 trend histories
    - frozen Phase-5.1.2c 1-positive + 99-negative candidate lists
    - frozen Phase-5.3.3a ranking / HR@10 / NDCG@10 semantics

Diagnostic subset
-----------------
Exactly 16 validation cases are selected deterministically as the first
16 validation cases in the frozen evaluation-case order.

This is PREFLIGHT_ONLY.
The resulting subset HR@10/NDCG@10 values are diagnostic fingerprints,
NOT reported model performance and NOT used for checkpoint selection.

Candidate binding
-----------------
For each selected validation event:
    candidate 0    = the true focal startup
    candidates 1..99 = frozen negatives in their frozen matrix order

Candidate POSITION has no ranking privilege because Phase 5.3.3a freezes:
    primary:  raw logit descending
    tie-break: startup_local ascending

T60 trend semantics
-------------------
Validation target is T60, so TrendExtractor.encode_sequence receives exactly
60 periods:
    T0..T59

The original frozen T60 encode_sequence method is used.
The training-only variable-h adapter is NOT used for validation.

Safety boundary
---------------
No optimizer is instantiated.
No backward occurs.
No optimizer.step() occurs.
No checkpoint is written.
No test case is scored.
No validation result is used for model selection.

The canonical initial parameter SHA256 must remain unchanged before/after.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Frozen dependencies
# =============================================================================

ROUNDTRIP_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_2b_checkpoint_resume_roundtrip_proof.py"
)

RANKING_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_3a_validation_ranking_metric_semantics_audit.py"
)

PHASE_5_3_3A_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_3a_validation_ranking_metric_contract.json"
)

PHASE_3_NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/model_ready/"
    "node_index.parquet"
)


# =============================================================================
# Frozen model/runtime dimensions
# =============================================================================

NUM_INVESTORS = 165_975
NUM_STARTUPS = 311_589
NUM_NODES = 477_564
NUM_HISTORY_PERIODS = 60

DOC2VEC_DIM = 32
LABEL_DIM = 802

LATENT_DIM = 40
DESCRIPTION_DIM = 40
TREND_ITEM_DIM = 80
TREND_QUERY_DIM = 80
TREND_DIM = 40
STRUCTURAL_DIM = 40

INVESTOR_SCORING_DIM = 160
STARTUP_SCORING_DIM = 120
PAIR_DIM = 280

NUM_PREFLIGHT_CASES = 16
CANDIDATES_PER_CASE = 100
NEGATIVES_PER_CASE = 99

EXPECTED_INITIAL_MODEL_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

REFERENCE_TORCH_VERSION_PREFIX = "2.7.0"


# =============================================================================
# Outputs
# =============================================================================

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_3b"
)

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

CASE_BINDING_PATH = (
    AUDIT_DIR
    / "validation_preflight_case_binding.csv"
)

SELECTED_CASES_PATH = (
    AUDIT_DIR
    / "validation_preflight_selected_cases.parquet"
)

CANDIDATE_MATRIX_PATH = (
    AUDIT_DIR
    / "validation_preflight_candidate_startup_local.npy"
)

SCORE_MATRIX_PATH = (
    AUDIT_DIR
    / "validation_preflight_raw_logit_matrix.npy"
)

CASE_METRICS_PATH = (
    AUDIT_DIR
    / "validation_preflight_case_metrics.csv"
)

TREND_AUDIT_PATH = (
    AUDIT_DIR
    / "validation_preflight_trend_audit.csv"
)

FORWARD_AUDIT_PATH = (
    AUDIT_DIR
    / "validation_preflight_forward_audit.csv"
)

STATE_NEUTRALITY_PATH = (
    AUDIT_DIR
    / "validation_preflight_state_neutrality.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_3b_final_invariants.csv"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_3b_validation_scoring_decision_register.csv"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_3b_canonical_validation_scoring_preflight_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_3b_canonical_validation_scoring_preflight_manifest.json"
)


# =============================================================================
# Generic helpers
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


def numpy_rng_state_equal(
    left,
    right,
) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(
            left[1],
            right[1],
        )
        and left[2:] == right[2:]
    )


def rng_snapshot() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }


def rng_equal(
    left: dict,
    right: dict,
) -> bool:
    return (
        left["python"]
        == right["python"]
        and numpy_rng_state_equal(
            left["numpy"],
            right["numpy"],
        )
        and torch.equal(
            left["torch"],
            right["torch"],
        )
    )


def choose_existing_column(
    columns: list[str],
    candidates: list[str],
) -> str | None:
    lower_to_actual = {
        str(column).lower(): str(column)
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[
                candidate.lower()
            ]

    return None


# =============================================================================
# Import-safe module loader
# =============================================================================

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


# =============================================================================
# Canonical model reconstruction WITHOUT Adam
# =============================================================================

def construct_canonical_validation_model(
    runtime_2b,
):
    preflight = (
        runtime_2b
        .load_preflight_runtime()
    )

    canonical_source = (
        preflight
        .CANONICAL_SOURCE_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    forward_source = (
        preflight
        .FORWARD_SOURCE_PATH
        .read_text(
            encoding="utf-8"
        )
    )

    canonical_tree = ast.parse(
        canonical_source,
        filename=str(
            preflight
            .CANONICAL_SOURCE_PATH
        ),
    )

    forward_tree = ast.parse(
        forward_source,
        filename=str(
            preflight
            .FORWARD_SOURCE_PATH
        ),
    )

    (
        canonical_runtime,
        runtime_ast_sha,
    ) = (
        preflight
        .build_canonical_runtime(
            canonical_tree
        )
    )

    (
        _,
        exact_methods,
        training_adapter,
        adapter_sha,
        removed_guard_sha,
    ) = (
        preflight
        .build_forward_runtime(
            forward_tree
        )
    )

    (
        model,
        canonical_hash_fn,
    ) = (
        preflight
        .compose_canonical_model(
            canonical_runtime,
            exact_methods,
            training_adapter,
        )
    )

    require(
        canonical_hash_fn(
            model
        )
        == EXPECTED_INITIAL_MODEL_SHA256,
        (
            "Canonical validation model "
            "initial SHA256 drift."
        ),
    )

    model.eval()

    return (
        preflight,
        model,
        canonical_hash_fn,
        runtime_ast_sha,
        adapter_sha,
        removed_guard_sha,
    )


# =============================================================================
# Evaluation artifact binding
# =============================================================================

def normalize_split_value(
    value,
) -> str:
    text = str(
        value
    ).strip().lower()

    if text in {
        "validation",
        "valid",
        "val",
    }:
        return "validation"

    if text == "test":
        return "test"

    raise AssertionError(
        f"Unexpected evaluation split: {value!r}"
    )


def infer_negative_matrix_coordinate(
    negative_matrix: np.ndarray,
) -> str:
    minimum = int(
        np.min(
            negative_matrix
        )
    )

    maximum = int(
        np.max(
            negative_matrix
        )
    )

    local_possible = (
        minimum >= 0
        and maximum < NUM_STARTUPS
    )

    global_possible = (
        minimum >= NUM_INVESTORS
        and maximum < NUM_NODES
    )

    # With >2M frozen draws, the actual matrix should make this unambiguous.
    require(
        local_possible
        != global_possible,
        (
            "Could not unambiguously infer whether the "
            "frozen negative matrix stores startup-local "
            "or global node indices. "
            f"min={minimum}, max={maximum}, "
            f"local_possible={local_possible}, "
            f"global_possible={global_possible}"
        ),
    )

    return (
        "startup_local"
        if local_possible
        else "startup_global"
    )


def normalize_negative_matrix_to_local(
    negative_matrix: np.ndarray,
    coordinate: str,
) -> np.ndarray:
    value = np.asarray(
        negative_matrix
    )

    if coordinate == "startup_local":
        local = value.astype(
            np.int64,
            copy=False,
        )

    elif coordinate == "startup_global":
        local = (
            value.astype(
                np.int64,
                copy=False,
            )
            - NUM_INVESTORS
        )

    else:
        raise AssertionError(
            f"Unexpected coordinate: {coordinate}"
        )

    require(
        bool(
            (
                local >= 0
            ).all()
        )
        and bool(
            (
                local < NUM_STARTUPS
            ).all()
        ),
        (
            "Normalized negative startup-local "
            "indices outside role universe."
        ),
    )

    return local


def build_node_id_maps():
    require(
        PHASE_3_NODE_INDEX_PATH.exists(),
        (
            "Missing Phase-3 node index "
            "for ID fallback."
        ),
    )

    node_index = pd.read_parquet(
        PHASE_3_NODE_INDEX_PATH,
        columns=[
            "node_index",
            "node_type",
            "raw_entity_id",
        ],
    )

    investor_rows = node_index.loc[
        node_index[
            "node_type"
        ]
        == "Investor"
    ]

    startup_rows = node_index.loc[
        node_index[
            "node_type"
        ]
        == "Startup"
    ]

    require(
        len(
            investor_rows
        )
        == NUM_INVESTORS,
        (
            "Investor node registry count drift."
        ),
    )

    require(
        len(
            startup_rows
        )
        == NUM_STARTUPS,
        (
            "Startup node registry count drift."
        ),
    )

    investor_map = dict(
        zip(
            investor_rows[
                "raw_entity_id"
            ].astype(str),
            investor_rows[
                "node_index"
            ].astype(int),
        )
    )

    startup_global_map = dict(
        zip(
            startup_rows[
                "raw_entity_id"
            ].astype(str),
            startup_rows[
                "node_index"
            ].astype(int),
        )
    )

    return (
        investor_map,
        startup_global_map,
    )


def resolve_case_bindings(
    case_manifest: pd.DataFrame,
    contract: dict,
) -> tuple[pd.DataFrame, dict]:
    columns = list(
        case_manifest.columns
    )

    frozen_artifact = (
        contract[
            "frozen_artifacts"
        ][
            "case_manifest"
        ]
    )

    split_column = frozen_artifact[
        "split_column"
    ]

    interaction_column = frozen_artifact[
        "interaction_column"
    ]

    require(
        split_column in columns,
        (
            "Frozen split column missing "
            "from case manifest."
        ),
    )

    require(
        interaction_column in columns,
        (
            "Frozen interaction column missing "
            "from case manifest."
        ),
    )

    explicit_case_index_column = (
        choose_existing_column(
            columns,
            [
                "case_index",
                "evaluation_case_index",
                "eval_case_index",
                "case_row_index",
            ],
        )
    )

    investor_index_column = (
        choose_existing_column(
            columns,
            [
                "investor_global",
                "investor_node_index",
                "investor_global_index",
                "investor_index",
            ],
        )
    )

    startup_local_column = (
        choose_existing_column(
            columns,
            [
                "startup_local",
                "positive_startup_local",
                "focal_startup_local",
                "startup_local_index",
                "positive_startup_local_index",
            ],
        )
    )

    startup_global_column = (
        choose_existing_column(
            columns,
            [
                "startup_global",
                "positive_startup_global",
                "focal_startup_global",
                "startup_node_index",
                "positive_startup_node_index",
                "focal_startup_node_index",
                "startup_global_index",
            ],
        )
    )

    investor_id_column = (
        choose_existing_column(
            columns,
            [
                "investor_id",
                "raw_investor_id",
            ],
        )
    )

    startup_id_column = (
        choose_existing_column(
            columns,
            [
                "startup_id",
                "positive_startup_id",
                "focal_startup_id",
                "raw_startup_id",
            ],
        )
    )

    need_id_maps = (
        investor_index_column
        is None
        or (
            startup_local_column
            is None
            and startup_global_column
            is None
        )
    )

    investor_map = None
    startup_global_map = None

    if need_id_maps:
        (
            investor_map,
            startup_global_map,
        ) = build_node_id_maps()

    rows = []

    for physical_row_index, row in (
        case_manifest
        .reset_index(
            drop=True
        )
        .iterrows()
    ):
        split = normalize_split_value(
            row[
                split_column
            ]
        )

        if explicit_case_index_column is not None:
            matrix_row_index = int(
                row[
                    explicit_case_index_column
                ]
            )
            binding_method = (
                "explicit_case_index_column"
            )
        else:
            matrix_row_index = int(
                physical_row_index
            )
            binding_method = (
                "frozen_manifest_physical_row_order"
            )

        require(
            0
            <= matrix_row_index
            < len(
                case_manifest
            ),
            (
                "Evaluation matrix row index "
                "outside case universe."
            ),
        )

        if investor_index_column is not None:
            investor_global = int(
                row[
                    investor_index_column
                ]
            )
            investor_binding = (
                investor_index_column
            )
        else:
            require(
                investor_id_column
                is not None,
                (
                    "Could not resolve investor: "
                    "no node-index column or investor_id."
                ),
            )

            investor_id = str(
                row[
                    investor_id_column
                ]
            )

            require(
                investor_id
                in investor_map,
                (
                    "Investor ID absent from "
                    "Phase-3 investor registry: "
                    f"{investor_id}"
                ),
            )

            investor_global = int(
                investor_map[
                    investor_id
                ]
            )

            investor_binding = (
                f"{investor_id_column}"
                "->phase3_raw_entity_id"
            )

        require(
            0
            <= investor_global
            < NUM_INVESTORS,
            (
                "Resolved investor global index "
                "outside Investor role universe."
            ),
        )

        if startup_local_column is not None:
            positive_startup_local = int(
                row[
                    startup_local_column
                ]
            )

            startup_binding = (
                startup_local_column
            )

        elif startup_global_column is not None:
            startup_global = int(
                row[
                    startup_global_column
                ]
            )

            require(
                NUM_INVESTORS
                <= startup_global
                < NUM_NODES,
                (
                    "Resolved positive startup global "
                    "index outside Startup role universe."
                ),
            )

            positive_startup_local = (
                startup_global
                - NUM_INVESTORS
            )

            startup_binding = (
                startup_global_column
            )

        else:
            require(
                startup_id_column
                is not None,
                (
                    "Could not resolve positive startup: "
                    "no startup index column or startup_id."
                ),
            )

            startup_id = str(
                row[
                    startup_id_column
                ]
            )

            require(
                startup_id
                in startup_global_map,
                (
                    "Startup ID absent from "
                    "Phase-3 Startup registry: "
                    f"{startup_id}"
                ),
            )

            startup_global = int(
                startup_global_map[
                    startup_id
                ]
            )

            positive_startup_local = (
                startup_global
                - NUM_INVESTORS
            )

            startup_binding = (
                f"{startup_id_column}"
                "->phase3_raw_entity_id"
            )

        require(
            0
            <= positive_startup_local
            < NUM_STARTUPS,
            (
                "Resolved positive startup local "
                "index outside Startup role universe."
            ),
        )

        rows.append(
            {
                "physical_manifest_row": (
                    int(
                        physical_row_index
                    )
                ),
                "matrix_row_index": (
                    matrix_row_index
                ),
                "split": (
                    split
                ),
                "interaction_id": str(
                    row[
                        interaction_column
                    ]
                ),
                "investor_global": (
                    investor_global
                ),
                "positive_startup_local": (
                    positive_startup_local
                ),
                "positive_startup_global": (
                    NUM_INVESTORS
                    + positive_startup_local
                ),
                "case_index_binding_method": (
                    binding_method
                ),
                "investor_binding_source": (
                    investor_binding
                ),
                "startup_binding_source": (
                    startup_binding
                ),
            }
        )

    resolved = pd.DataFrame(
        rows
    )

    require(
        resolved[
            "matrix_row_index"
        ].is_unique,
        (
            "Resolved evaluation matrix-row "
            "indices are not unique."
        ),
    )

    require(
        resolved[
            "interaction_id"
        ].is_unique,
        (
            "Resolved interaction IDs "
            "are not unique."
        ),
    )

    metadata = {
        "split_column": (
            split_column
        ),
        "interaction_column": (
            interaction_column
        ),
        "explicit_case_index_column": (
            explicit_case_index_column
        ),
        "investor_index_column": (
            investor_index_column
        ),
        "startup_local_column": (
            startup_local_column
        ),
        "startup_global_column": (
            startup_global_column
        ),
        "investor_id_column": (
            investor_id_column
        ),
        "startup_id_column": (
            startup_id_column
        ),
    }

    return (
        resolved,
        metadata,
    )


# =============================================================================
# Real T60 feature construction
# =============================================================================

def collect_history_by_investor(
    selected_investors: list[int],
    shared: dict,
) -> tuple[
    dict[int, list[np.ndarray]],
    list[np.ndarray],
]:
    history_by_investor = {}
    all_historical_nodes = []

    for investor_global in sorted(
        set(
            int(value)
            for value in selected_investors
        )
    ):
        periods = []

        for period in range(
            NUM_HISTORY_PERIODS
        ):
            flattened = (
                investor_global
                * NUM_HISTORY_PERIODS
                + period
            )

            start = int(
                shared[
                    "trend_period_ptr"
                ][
                    flattened
                ]
            )

            end = int(
                shared[
                    "trend_period_ptr"
                ][
                    flattened
                    + 1
                ]
            )

            count = int(
                shared[
                    "trend_period_counts"
                ][
                    flattened
                ]
            )

            require(
                end - start
                == count,
                (
                    "Trend CSR mismatch: "
                    f"investor={investor_global}, "
                    f"period={period}"
                ),
            )

            startups_global = np.array(
                shared[
                    "trend_startup_indices"
                ][
                    start:end
                ],
                dtype=np.int64,
                copy=True,
            )

            require(
                bool(
                    (
                        startups_global
                        >= NUM_INVESTORS
                    ).all()
                )
                and bool(
                    (
                        startups_global
                        < NUM_NODES
                    ).all()
                ),
                (
                    "Trend history contains "
                    "non-Startup global node index."
                ),
            )

            if len(
                startups_global
            ) > 0:
                all_historical_nodes.append(
                    startups_global
                )

            periods.append(
                startups_global
            )

        require(
            len(
                periods
            )
            == 60,
            (
                "T60 validation history "
                "does not contain exactly T0..T59."
            ),
        )

        history_by_investor[
            investor_global
        ] = periods

    return (
        history_by_investor,
        all_historical_nodes,
    )


def compute_validation_features(
    model: torch.nn.Module,
    selected_cases: pd.DataFrame,
    candidate_matrix_local: np.ndarray,
    shared: dict,
) -> dict:
    selected_investors = (
        selected_cases[
            "investor_global"
        ]
        .astype(int)
        .tolist()
    )

    (
        history_by_investor,
        historical_nodes,
    ) = collect_history_by_investor(
        selected_investors,
        shared,
    )

    candidate_globals = (
        candidate_matrix_local
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
            "Validation Doc2Vec subset "
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
            "Validation label subset "
            "contains non-finite values."
        ),
    )

    doc_subset = torch.from_numpy(
        doc_subset_np
    )

    label_subset = torch.from_numpy(
        label_subset_np
    )

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
            "Invalid structural output."
        ),
    )

    F_s_all = structural[
        "F_s"
    ]

    description_subset = (
        model.description_encoder(
            doc_subset,
            label_subset,
        )
    )

    require(
        F_s_all.shape
        == (
            NUM_NODES,
            STRUCTURAL_DIM,
        ),
        (
            "Validation structural "
            "representation shape drift."
        ),
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
            "Validation description "
            "representation shape drift."
        ),
    )

    # -------------------------------------------------------------------------
    # Exact T60 trend: original encode_sequence, exactly 60 periods.
    # -------------------------------------------------------------------------

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
            investor_subset_position
            >= 0,
            (
                "Investor description row "
                "missing from validation subset."
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
                "Validation trend query "
                "shape drift."
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
                        "Historical startup description "
                        "row missing."
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
                        "Validation trend item "
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
                        "Validation attention "
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
                        "Validation attention "
                        "weights do not sum to one."
                    ),
                )

            period_vectors.append(
                period_vector
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
                "T60 validation sequence "
                "must be exactly (1,60,80)."
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
                "T60 validation F_t "
                "shape drift."
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
                "T60 validation GRU "
                "output shape drift."
            ),
        )

        require(
            bool(
                torch.isfinite(
                    F_t
                ).all()
            ),
            (
                "Validation F_t "
                "contains non-finite values."
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
# Case scoring
# =============================================================================

def score_validation_case(
    model: torch.nn.Module,
    investor_global: int,
    candidate_startup_local: np.ndarray,
    features: dict,
) -> torch.Tensor:
    candidate_startup_local = np.asarray(
        candidate_startup_local,
        dtype=np.int64,
    )

    require(
        candidate_startup_local.shape
        == (
            100,
        ),
        (
            "Validation case must contain "
            "exactly 100 candidate startups."
        ),
    )

    require(
        len(
            np.unique(
                candidate_startup_local
            )
        )
        == 100,
        (
            "Validation candidate startups "
            "must be unique within a case."
        ),
    )

    candidate_global = (
        candidate_startup_local
        + NUM_INVESTORS
    )

    investor_tensor = torch.tensor(
        [
            investor_global,
        ],
        dtype=torch.int64,
    )

    startup_local_tensor = torch.from_numpy(
        np.array(
            candidate_startup_local,
            dtype=np.int64,
            copy=True,
        )
    )

    startup_global_tensor = torch.from_numpy(
        np.array(
            candidate_global,
            dtype=np.int64,
            copy=True,
        )
    )

    investor_subset_position = int(
        features[
            "global_to_subset"
        ][
            investor_global
        ]
    )

    startup_subset_positions_np = (
        features[
            "global_to_subset"
        ][
            candidate_global
        ]
    )

    require(
        investor_subset_position
        >= 0,
        (
            "Validation investor description "
            "row missing."
        ),
    )

    require(
        bool(
            (
                startup_subset_positions_np
                >= 0
            ).all()
        ),
        (
            "Validation candidate description "
            "row missing."
        ),
    )

    startup_subset_positions = torch.from_numpy(
        np.array(
            startup_subset_positions_np,
            dtype=np.int64,
            copy=True,
        )
    )

    L_o = (
        model.investor_embedding(
            investor_tensor
        )[0]
    )

    F_d_o = (
        features[
            "description_subset"
        ][
            investor_subset_position
        ]
    )

    F_s_o = (
        features[
            "F_s_all"
        ][
            investor_global
        ]
    )

    F_t = (
        features[
            "F_t_by_investor"
        ][
            investor_global
        ]
    )

    investor_representation_single = (
        torch.cat(
            [
                F_t,
                L_o,
                F_d_o,
                F_s_o,
            ],
            dim=0,
        )
    )

    require(
        investor_representation_single.shape
        == (
            INVESTOR_SCORING_DIM,
        ),
        (
            "Validation investor scoring "
            "representation shape drift."
        ),
    )

    investor_representation = (
        investor_representation_single
        .unsqueeze(
            0
        )
        .expand(
            CANDIDATES_PER_CASE,
            -1,
        )
    )

    L_b = (
        model.startup_embedding(
            startup_local_tensor
        )
    )

    F_d_b = (
        features[
            "description_subset"
        ][
            startup_subset_positions
        ]
    )

    F_s_b = (
        features[
            "F_s_all"
        ][
            startup_global_tensor
        ]
    )

    startup_representation = torch.cat(
        [
            L_b,
            F_d_b,
            F_s_b,
        ],
        dim=1,
    )

    pair_representation = torch.cat(
        [
            investor_representation,
            startup_representation,
        ],
        dim=1,
    )

    require(
        startup_representation.shape
        == (
            100,
            STARTUP_SCORING_DIM,
        ),
        (
            "Validation startup scoring "
            "representation shape drift."
        ),
    )

    require(
        pair_representation.shape
        == (
            100,
            PAIR_DIM,
        ),
        (
            "Validation pair representation "
            "shape drift."
        ),
    )

    scoring = model.scoring_mlp(
        pair_representation
    )

    require(
        isinstance(
            scoring,
            dict,
        )
        and "logit"
        in scoring,
        (
            "Validation scoring output invalid."
        ),
    )

    logits = scoring[
        "logit"
    ].reshape(
        -1
    )

    require(
        logits.shape
        == (
            100,
        ),
        (
            "Validation logits must "
            "have shape (100,)."
        ),
    )

    require(
        bool(
            torch.isfinite(
                logits
            ).all()
        ),
        (
            "Validation logits contain "
            "non-finite values."
        ),
    )

    return logits


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.3b — "
        "CANONICAL REAL-MODEL VALIDATION SCORING PREFLIGHT"
    )

    print(
        "Diagnostic validation cases:          16"
    )
    print(
        "Candidates / case:                    100"
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
        "Test cases scored:                    0"
    )
    print(
        "Subset metric used for selection:     NO"
    )

    # =========================================================================
    # Contracts
    # =========================================================================

    banner(
        "AUTHORITATIVE VALIDATION CONTRACT RECHECK"
    )

    for path in (
        ROUNDTRIP_SOURCE_PATH,
        RANKING_SOURCE_PATH,
        PHASE_5_3_3A_CONTRACT_PATH,
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

    require(
        ranking_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.3a ranking "
            "contract is not frozen."
        ),
    )

    require(
        ranking_contract[
            "ranking"
        ][
            "score"
        ]
        == "raw model logit",
        (
            "Validation ranking score "
            "contract drift."
        ),
    )

    require(
        ranking_contract[
            "ranking"
        ][
            "primary_order"
        ]
        == "logit descending",
        (
            "Validation primary ranking "
            "order drift."
        ),
    )

    require(
        ranking_contract[
            "ranking"
        ][
            "tie_break"
        ]
        == "startup_local ascending",
        (
            "Validation tie-break "
            "contract drift."
        ),
    )

    require(
        int(
            ranking_contract[
                "candidate_runtime"
            ][
                "validation_cases"
            ]
        )
        == 2251,
        (
            "Validation case count drift."
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

    # =========================================================================
    # Load exact prior runtime modules
    # =========================================================================

    banner(
        "LOAD FROZEN NUMERICAL / RANKING RUNTIMES"
    )

    runtime_2b = load_guarded_module(
        ROUNDTRIP_SOURCE_PATH,
        "_itrs_phase5_3_3b_runtime2b",
    )

    ranking_runtime = load_guarded_module(
        RANKING_SOURCE_PATH,
        "_itrs_phase5_3_3b_ranking",
    )

    for symbol in (
        "load_preflight_runtime",
        "load_shared_inputs",
    ):
        require(
            hasattr(
                runtime_2b,
                symbol,
            ),
            (
                "Phase-5.3.2b runtime missing "
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
    # Frozen candidate artifacts
    # =========================================================================

    banner(
        "BIND FROZEN EVALUATION ARTIFACTS"
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
        negative_matrix_path.exists(),
        (
            "Frozen evaluation negative "
            "matrix path missing."
        ),
    )

    require(
        case_manifest_path.exists(),
        (
            "Frozen evaluation case "
            "manifest path missing."
        ),
    )

    require(
        file_sha256(
            negative_matrix_path
        )
        == negative_artifact[
            "file_sha256"
        ],
        (
            "Evaluation negative matrix "
            "physical SHA drift."
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
            "Evaluation case manifest "
            "physical SHA drift."
        ),
    )

    negative_matrix_raw = np.load(
        negative_matrix_path,
        mmap_mode="r",
    )

    require(
        negative_matrix_raw.shape
        == (
            22515,
            99,
        ),
        (
            "Evaluation negative matrix "
            "shape drift."
        ),
    )

    negative_coordinate = (
        infer_negative_matrix_coordinate(
            negative_matrix_raw
        )
    )

    negative_matrix_local = (
        normalize_negative_matrix_to_local(
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
    ) = resolve_case_bindings(
        case_manifest,
        ranking_contract,
    )

    require(
        len(
            resolved_cases
        )
        == 22515,
        (
            "Resolved evaluation case "
            "count drift."
        ),
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
        == 2251,
        (
            "Resolved validation case "
            "count drift."
        ),
    )

    # =========================================================================
    # Select deterministic validation subset
    # =========================================================================

    banner(
        "SELECT DETERMINISTIC 16-CASE VALIDATION PREFLIGHT SUBSET"
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

    selected_cases = (
        validation_cases.iloc[
            :NUM_PREFLIGHT_CASES
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    require(
        len(
            selected_cases
        )
        == NUM_PREFLIGHT_CASES,
        (
            "Could not select 16 validation "
            "preflight cases."
        ),
    )

    require(
        bool(
            (
                selected_cases[
                    "split"
                ]
                == "validation"
            ).all()
        ),
        (
            "Preflight selection contains "
            "non-validation case."
        ),
    )

    candidate_matrix_local = np.empty(
        (
            NUM_PREFLIGHT_CASES,
            CANDIDATES_PER_CASE,
        ),
        dtype=np.int64,
    )

    candidate_rows = []

    for case_position, row in (
        selected_cases
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
                99,
            ),
            (
                "Frozen validation negative "
                "case does not contain 99 rows."
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
                "Focal positive startup appears "
                "inside frozen negative list."
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
            == 100,
            (
                "Real validation candidate set "
                "does not contain 100 unique startups."
            ),
        )

        candidate_matrix_local[
            case_position
        ] = candidates

        candidate_rows.append(
            {
                "preflight_case_position": (
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

    selection_df = pd.DataFrame(
        candidate_rows
    )

    selection_sha = (
        dataframe_logical_sha256(
            selection_df,
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

    candidate_matrix_sha = (
        array_logical_sha256(
            candidate_matrix_local
        )
    )

    print(
        selection_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Selected-case logical SHA256:"
    )
    print(
        selection_sha
    )

    print()
    print(
        "Candidate-matrix logical SHA256:"
    )
    print(
        candidate_matrix_sha
    )

    # =========================================================================
    # Canonical initial model / real features
    # =========================================================================

    banner(
        "RECONSTRUCT CANONICAL INITIAL VALIDATION MODEL"
    )

    (
        preflight_runtime,
        model,
        canonical_hash_fn,
        runtime_ast_sha,
        adapter_sha,
        removed_guard_sha,
    ) = construct_canonical_validation_model(
        runtime_2b
    )

    shared = (
        runtime_2b
        .load_shared_inputs(
            preflight_runtime
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
            "state mismatch."
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
            "Validation model has pre-existing "
            "gradients."
        ),
    )

    rng_before = rng_snapshot()

    # =========================================================================
    # Full real feature forward under no_grad
    # =========================================================================

    banner(
        "REAL T60 FEATURE FORWARD"
    )

    with torch.no_grad():
        features = (
            compute_validation_features(
                model,
                selected_cases,
                candidate_matrix_local,
                shared,
            )
        )

    trend_df = features[
        "trend_audit"
    ]

    require(
        bool(
            (
                trend_df[
                    "history_periods"
                ]
                == 60
            ).all()
        ),
        (
            "At least one validation investor "
            "did not consume T0..T59."
        ),
    )

    require(
        set(
            trend_df[
                "runtime_method"
            ]
        )
        == {
            "TrendExtractor.encode_sequence"
        },
        (
            "Validation used wrong trend "
            "sequence runtime."
        ),
    )

    print(
        trend_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Score real frozen candidate lists
    # =========================================================================

    banner(
        "SCORE 16 REAL VALIDATION 1+99 CANDIDATE LISTS"
    )

    score_matrix = np.empty(
        (
            NUM_PREFLIGHT_CASES,
            CANDIDATES_PER_CASE,
        ),
        dtype=np.float32,
    )

    metric_rows = []

    with torch.no_grad():
        for case_position, row in (
            selected_cases
            .iterrows()
        ):
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
                candidate_matrix_local[
                    case_position
                ]
            )

            logits = score_validation_case(
                model,
                investor_global,
                candidates_local,
                features,
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
                    "Real validation logits "
                    "contain non-finite values."
                ),
            )

            score_matrix[
                case_position
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
                    "preflight_case_position": (
                        int(
                            case_position
                        )
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
                }
            )

    metric_df = pd.DataFrame(
        metric_rows
    )

    require(
        len(
            metric_df
        )
        == NUM_PREFLIGHT_CASES,
        (
            "Real validation metric row "
            "count drift."
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
            "Real validation positive rank "
            "outside 1..100."
        ),
    )

    (
        diagnostic_hr10,
        diagnostic_ndcg10,
    ) = (
        ranking_runtime
        .aggregate_event_level_metrics(
            metric_df
        )
    )

    score_matrix_sha = (
        array_logical_sha256(
            score_matrix
        )
    )

    metric_sha = (
        dataframe_logical_sha256(
            metric_df,
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

    print(
        metric_df.to_string(
            index=False
        )
    )

    print()
    print(
        f"DIAGNOSTIC subset HR@10:              "
        f"{diagnostic_hr10:.12f}"
    )
    print(
        f"DIAGNOSTIC subset NDCG@10:            "
        f"{diagnostic_ndcg10:.12f}"
    )
    print(
        "Used for checkpoint selection:        NO"
    )

    print()
    print(
        "Raw-logit matrix logical SHA256:"
    )
    print(
        score_matrix_sha
    )

    print()
    print(
        "Case-metric logical SHA256:"
    )
    print(
        metric_sha
    )

    # =========================================================================
    # State/RNG neutrality
    # =========================================================================

    banner(
        "VALIDATION STATE / RNG NEUTRALITY"
    )

    parameter_hash_after = (
        canonical_hash_fn(
            model
        )
    )

    rng_after = rng_snapshot()

    require(
        parameter_hash_after
        == EXPECTED_INITIAL_MODEL_SHA256,
        (
            "Validation scoring modified "
            "canonical parameter values."
        ),
    )

    require(
        parameter_hash_after
        == parameter_hash_before,
        (
            "Validation parameter state "
            "changed during no_grad forward."
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
            "Validation scoring created gradients."
        ),
    )

    require(
        rng_equal(
            rng_before,
            rng_after,
        ),
        (
            "Validation scoring changed "
            "Python/NumPy/Torch global RNG."
        ),
    )

    state_df = pd.DataFrame(
        [
            {
                "check": (
                    "canonical_hash_before"
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
                    "canonical_hash_after"
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
        ]
    )

    print(
        state_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Forward/binding audit
    # =========================================================================

    binding_df = pd.DataFrame(
        [
            {
                "item": (
                    "negative_matrix_coordinate"
                ),
                "value": (
                    negative_coordinate
                ),
            },
            {
                "item": (
                    "case_index_binding"
                ),
                "value": (
                    selected_cases[
                        "case_index_binding_method"
                    ].iloc[
                        0
                    ]
                ),
            },
            {
                "item": (
                    "split_column"
                ),
                "value": (
                    binding_metadata[
                        "split_column"
                    ]
                ),
            },
            {
                "item": (
                    "interaction_column"
                ),
                "value": (
                    binding_metadata[
                        "interaction_column"
                    ]
                ),
            },
            {
                "item": (
                    "investor_index_column"
                ),
                "value": (
                    str(
                        binding_metadata[
                            "investor_index_column"
                        ]
                    )
                ),
            },
            {
                "item": (
                    "startup_local_column"
                ),
                "value": (
                    str(
                        binding_metadata[
                            "startup_local_column"
                        ]
                    )
                ),
            },
            {
                "item": (
                    "startup_global_column"
                ),
                "value": (
                    str(
                        binding_metadata[
                            "startup_global_column"
                        ]
                    )
                ),
            },
            {
                "item": (
                    "investor_id_column"
                ),
                "value": (
                    str(
                        binding_metadata[
                            "investor_id_column"
                        ]
                    )
                ),
            },
            {
                "item": (
                    "startup_id_column"
                ),
                "value": (
                    str(
                        binding_metadata[
                            "startup_id_column"
                        ]
                    )
                ),
            },
        ]
    )

    forward_df = pd.DataFrame(
        [
            {
                "feature": (
                    "required_description_nodes"
                ),
                "value": (
                    len(
                        features[
                            "required_nodes"
                        ]
                    )
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "feature": (
                    "structural_F_s_shape"
                ),
                "value": (
                    str(
                        tuple(
                            features[
                                "F_s_all"
                            ].shape
                        )
                    )
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "feature": (
                    "validation_investors_with_F_t"
                ),
                "value": (
                    len(
                        features[
                            "F_t_by_investor"
                        ]
                    )
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "feature": (
                    "candidate_score_matrix_shape"
                ),
                "value": (
                    str(
                        tuple(
                            score_matrix.shape
                        )
                    )
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "feature": (
                    "T60_history_periods"
                ),
                "value": (
                    60
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "feature": (
                    "trend_runtime_method"
                ),
                "value": (
                    "TrendExtractor.encode_sequence"
                ),
                "status": (
                    "PASS"
                ),
            },
        ]
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.3b INVARIANTS"
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
            "negative_matrix_physical_hash_exact",
            (
                file_sha256(
                    negative_matrix_path
                )
                == negative_artifact[
                    "file_sha256"
                ]
            ),
        ),
        (
            "case_manifest_physical_hash_exact",
            (
                file_sha256(
                    case_manifest_path
                )
                == case_artifact[
                    "file_sha256"
                ]
            ),
        ),
        (
            "preflight_cases_exactly_16",
            (
                len(
                    selected_cases
                )
                == 16
            ),
        ),
        (
            "preflight_cases_validation_only",
            bool(
                (
                    selected_cases[
                        "split"
                    ]
                    == "validation"
                ).all()
            ),
        ),
        (
            "candidate_matrix_16_by_100",
            (
                candidate_matrix_local.shape
                == (
                    16,
                    100,
                )
            ),
        ),
        (
            "candidate_startups_unique_per_case",
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
                    16
                )
            ),
        ),
        (
            "positive_not_in_frozen_negatives",
            all(
                int(
                    selected_cases.iloc[
                        index
                    ][
                        "positive_startup_local"
                    ]
                )
                not in set(
                    int(value)
                    for value
                    in candidate_matrix_local[
                        index,
                        1:
                    ].tolist()
                )
                for index in range(
                    16
                )
            ),
        ),
        (
            "canonical_initial_model_hash_exact",
            (
                parameter_hash_before
                == EXPECTED_INITIAL_MODEL_SHA256
            ),
        ),
        (
            "validation_model_eval_mode",
            (
                model.training
                is False
            ),
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
            "raw_logit_matrix_16_by_100",
            (
                score_matrix.shape
                == (
                    16,
                    100,
                )
            ),
        ),
        (
            "all_real_validation_logits_finite",
            bool(
                np.isfinite(
                    score_matrix
                ).all()
            ),
        ),
        (
            "positive_ranks_all_1_to_100",
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
            "ranking_uses_frozen_raw_logit_semantics",
            (
                ranking_contract[
                    "ranking"
                ][
                    "score"
                ]
                == "raw model logit"
            ),
        ),
        (
            "tie_break_uses_startup_local_ascending",
            (
                ranking_contract[
                    "ranking"
                ][
                    "tie_break"
                ]
                == "startup_local ascending"
            ),
        ),
        (
            "subset_metrics_event_level_mean",
            True,
        ),
        (
            "subset_metrics_diagnostic_only",
            True,
        ),
        (
            "test_cases_scored_zero",
            True,
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
            "parameter_hash_unchanged",
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
            "validation_forward_RNG_neutral",
            rng_equal(
                rng_before,
                rng_after,
            ),
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
            "At least one Phase-5.3.3b "
            "validation-scoring invariant failed."
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
        "WRITE PHASE-5.3.3b OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    binding_df.to_csv(
        CASE_BINDING_PATH,
        index=False,
    )

    selected_cases.to_parquet(
        SELECTED_CASES_PATH,
        index=False,
    )

    np.save(
        CANDIDATE_MATRIX_PATH,
        candidate_matrix_local,
        allow_pickle=False,
    )

    np.save(
        SCORE_MATRIX_PATH,
        score_matrix,
        allow_pickle=False,
    )

    metric_df.to_csv(
        CASE_METRICS_PATH,
        index=False,
    )

    trend_df.to_csv(
        TREND_AUDIT_PATH,
        index=False,
    )

    forward_df.to_csv(
        FORWARD_AUDIT_PATH,
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
                    "validation_real_model_preflight_state"
                ),
                "value": (
                    "CANONICAL_INITIAL_SEED42_MODEL"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE_PREFLIGHT_ONLY"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3b"
                ),
            },
            {
                "decision": (
                    "validation_preflight_subset"
                ),
                "value": (
                    "FIRST_16_VALIDATION_CASES_IN_FROZEN_CASE_ORDER"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE_PREFLIGHT_ONLY"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3b"
                ),
            },
            {
                "decision": (
                    "validation_candidate_serialization"
                ),
                "value": (
                    "POSITIVE_SLOT0_THEN_99_FROZEN_NEGATIVES_IN_MATRIX_ORDER"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3b"
                ),
            },
            {
                "decision": (
                    "validation_trend_runtime"
                ),
                "value": (
                    "ORIGINAL_T60_ENCODE_SEQUENCE_T0_TO_T59"
                ),
                "classification": (
                    "INHERITED_FROZEN_PHASE_4_AND_PHASE_5_3_3a"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3b"
                ),
            },
            {
                "decision": (
                    "validation_preflight_metric_use"
                ),
                "value": (
                    "DIAGNOSTIC_ONLY_NOT_CHECKPOINT_SELECTION"
                ),
                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3b"
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
                    "FROZEN_PHASE_5_3_3b"
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
            "5.3.3b"
        ),
        "title": (
            "Canonical Real-Model Validation Scoring Preflight Contract"
        ),
        "status": (
            "FROZEN"
        ),
        "model_state": {
            "purpose": (
                "preflight only"
            ),
            "canonical_initial_sha256": (
                EXPECTED_INITIAL_MODEL_SHA256
            ),
            "model_mode": (
                "eval"
            ),
            "gradient_mode": (
                "torch.no_grad"
            ),
            "optimizer_instantiated": (
                False
            ),
        },
        "case_selection": {
            "split": (
                "validation"
            ),
            "cases": (
                NUM_PREFLIGHT_CASES
            ),
            "selection_rule": (
                "first 16 validation cases by frozen matrix row order"
            ),
            "selected_case_logical_sha256": (
                selection_sha
            ),
            "test_cases_scored": (
                0
            ),
        },
        "candidate_binding": {
            "negative_matrix_coordinate": (
                negative_coordinate
            ),
            "case_index_binding_method": (
                selected_cases[
                    "case_index_binding_method"
                ].iloc[
                    0
                ]
            ),
            "serialization": (
                "positive slot0 then 99 frozen negatives"
            ),
            "candidate_matrix_logical_sha256": (
                candidate_matrix_sha
            ),
            "candidates_per_case": (
                100
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
            "inherited_contract": (
                str(
                    PHASE_5_3_3A_CONTRACT_PATH
                )
            ),
            "score": (
                "raw model logit"
            ),
            "primary_order": (
                "logit descending"
            ),
            "tie_break": (
                "startup_local ascending"
            ),
        },
        "numerical_fingerprints": {
            "raw_logit_matrix_logical_sha256": (
                score_matrix_sha
            ),
            "case_metric_logical_sha256": (
                metric_sha
            ),
            "diagnostic_subset_HR@10": (
                diagnostic_hr10
            ),
            "diagnostic_subset_NDCG@10": (
                diagnostic_ndcg10
            ),
        },
        "state_neutrality": {
            "parameter_sha256_before": (
                parameter_hash_before
            ),
            "parameter_sha256_after": (
                parameter_hash_after
            ),
            "parameter_values_changed": (
                False
            ),
            "gradients_created": (
                False
            ),
            "global_RNG_changed": (
                False
            ),
        },
        "boundary": {
            "full_validation_split_scored": (
                False
            ),
            "diagnostic_metric_used_for_selection": (
                False
            ),
            "test_case_scored": (
                False
            ),
            "test_used_for_selection": (
                False
            ),
        },
        "next_phase": {
            "id": (
                "5.3.3c"
            ),
            "title": (
                "Full Validation Split Runtime Dry-Run and Freeze"
            ),
            "requirement": (
                "Generalize the proven real-model scorer to all 2,251 "
                "validation cases using bounded candidate batching. "
                "Run one full validation pass on the canonical initial "
                "model, freeze full-split HR@10/NDCG@10 and score/rank "
                "fingerprints, prove state/RNG neutrality, and still "
                "score zero test cases."
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
            "5.3.3b"
        ),
        "status": (
            "CANONICAL_REAL_VALIDATION_SCORING_"
            "PREFLIGHT_PASSED_AND_FROZEN"
        ),
        "preflight_cases": (
            NUM_PREFLIGHT_CASES
        ),
        "candidate_matrix_sha256": (
            candidate_matrix_sha
        ),
        "raw_logit_matrix_sha256": (
            score_matrix_sha
        ),
        "case_metric_sha256": (
            metric_sha
        ),
        "diagnostic_HR@10": (
            diagnostic_hr10
        ),
        "diagnostic_NDCG@10": (
            diagnostic_ndcg10
        ),
        "canonical_parameter_sha256_after": (
            parameter_hash_after
        ),
        "optimizer_instantiated": (
            False
        ),
        "test_cases_scored": (
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
        CASE_BINDING_PATH,
        SELECTED_CASES_PATH,
        CANDIDATE_MATRIX_PATH,
        SCORE_MATRIX_PATH,
        CASE_METRICS_PATH,
        TREND_AUDIT_PATH,
        FORWARD_AUDIT_PATH,
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
        "PHASE 5.3.3b FINAL STATUS"
    )

    print(
        "Real validation cases scored:         16"
    )
    print(
        "Candidates / case:                    100"
    )
    print(
        "Test cases scored:                    0"
    )
    print()

    print(
        "Validation model:"
    )
    print(
        "  canonical initial seed-42 state"
    )
    print(
        "  model.eval()"
    )
    print(
        "  torch.no_grad()"
    )
    print(
        "  optimizer: NONE"
    )
    print()

    print(
        "T60 trend:"
    )
    print(
        "  history:                            T0..T59"
    )
    print(
        "  periods:                            60"
    )
    print(
        "  runtime:                            TrendExtractor.encode_sequence"
    )
    print(
        "  training adapter used:              NO"
    )
    print()

    print(
        "Selected-case logical SHA256:"
    )
    print(
        selection_sha
    )
    print()

    print(
        "Candidate-matrix logical SHA256:"
    )
    print(
        candidate_matrix_sha
    )
    print()

    print(
        "Raw-logit matrix logical SHA256:"
    )
    print(
        score_matrix_sha
    )
    print()

    print(
        "Case-metric logical SHA256:"
    )
    print(
        metric_sha
    )
    print()

    print(
        f"DIAGNOSTIC subset HR@10:              "
        f"{diagnostic_hr10:.12f}"
    )
    print(
        f"DIAGNOSTIC subset NDCG@10:            "
        f"{diagnostic_ndcg10:.12f}"
    )
    print(
        "Used for checkpoint selection:        NO"
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
        "Test used for model selection:        NO"
    )

    banner(
        "PHASE 5.3.3b COMPLETE / "
        "CANONICAL REAL VALIDATION SCORING PREFLIGHT PASSED AND FROZEN"
    )


if __name__ == "__main__":
    main()