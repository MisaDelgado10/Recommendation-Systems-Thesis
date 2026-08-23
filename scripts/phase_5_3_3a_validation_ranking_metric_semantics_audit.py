"""
Phase 5.3.3a — Validation Ranking Metric Semantics Audit and Freeze

Purpose
-------
Freeze the exact ranking / metric semantics used by Phase-5 validation and
final test BEFORE any production validation forward is executed.

This phase consumes the already-frozen Phase-5.1.2c evaluation candidate
artifacts but performs NO neural scoring.

Frozen scientific evaluation setup inherited from Phase 5.1:
    - one positive + 99 frozen negatives per T60 event
    - 100 candidates per case
    - validation: 2,251 event-level cases
    - test:      20,264 event-level cases
    - repeated events are retained; no pair/event deduplication
    - HR@10 and NDCG@10

Paper-unspecified runtime details frozen here
---------------------------------------------
Score used for ranking:
    raw model logit

Reason:
    sigmoid is strictly monotonic for finite logits, so ranking by raw logit
    is numerically equivalent while avoiding an unnecessary transformation.

Primary ranking key:
    logit descending

Deterministic tie-break:
    startup_local ascending

Reason:
    - deterministic
    - label-independent
    - does not favor the positive by candidate position
    - every candidate startup is unique within a case under the frozen
      candidate construction

Positive rank:
    1-based after the deterministic sort

Metrics for one relevant item:
    HR@10 =
        1 if positive_rank <= 10 else 0

    NDCG@10 =
        1 / log2(positive_rank + 1) if positive_rank <= 10 else 0

Because each case has exactly one relevant item, IDCG@10 = 1.

Aggregation:
    arithmetic mean across event-level cases in the requested split

Non-finite score handling:
    FAIL the evaluation run; do not silently rank NaN/inf.

No model is instantiated.
No forward pass occurs.
No validation metric on learned scores is produced.
No test is accessed for model selection.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# =============================================================================
# Frozen evaluation artifact fingerprints from Phase 5.1.2c
# =============================================================================

PHASE5_ROOT = Path(
    "data/experimental/phase_5"
)

EXPECTED_NEGATIVE_MATRIX_FILE_SHA256 = (
    "7f98269d0291382dacfc783ffd66ad6d2"
    "c2f8775877d57dc0d83504e40d8716d"
)

EXPECTED_CASE_MANIFEST_FILE_SHA256 = (
    "44b4b7e1ec1b1978249318a080df02d0"
    "d9f617845263513594de64e71b969e0c"
)

EXPECTED_PHASE_5_1_2C_MANIFEST_FILE_SHA256 = (
    "b3e43fa19deb57ce55b2499838055e53"
    "c520bc0395912ef0de7a340c55ac20b8"
)

EXPECTED_CASES = 22_515
EXPECTED_VALIDATION_CASES = 2_251
EXPECTED_TEST_CASES = 20_264

EXPECTED_NEGATIVES_PER_CASE = 99
EXPECTED_CANDIDATES_PER_CASE = 100

K = 10


# =============================================================================
# Frozen production-controller prerequisite
# =============================================================================

PHASE_5_3_2C_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_2c_production_driver_controller_contract.json"
)


# =============================================================================
# Outputs
# =============================================================================

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_3a"
)

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

ARTIFACT_DISCOVERY_PATH = (
    AUDIT_DIR
    / "evaluation_candidate_artifact_discovery.csv"
)

CANDIDATE_INTEGRITY_PATH = (
    AUDIT_DIR
    / "evaluation_candidate_integrity.csv"
)

RANKING_PROBE_PATH = (
    AUDIT_DIR
    / "ranking_metric_semantics_probe.csv"
)

TIE_PROBE_PATH = (
    AUDIT_DIR
    / "ranking_tie_break_probe.csv"
)

FORMULA_PATH = (
    AUDIT_DIR
    / "hr10_ndcg10_formula_audit.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_3a_final_invariants.csv"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_3a_validation_ranking_decision_register.csv"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_3a_validation_ranking_metric_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_3a_validation_ranking_metric_manifest.json"
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


def find_file_by_sha256(
    root: Path,
    expected_sha256: str,
    allowed_suffixes: set[str],
) -> Path:
    """
    Locate an already-frozen artifact by its authoritative physical SHA256
    rather than guessing its filename.
    """

    require(
        root.exists(),
        f"Missing Phase-5 root: {root}",
    )

    matches = []

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in (
            allowed_suffixes
        ):
            continue

        if file_sha256(path) == expected_sha256:
            matches.append(path)

    require(
        len(matches) == 1,
        (
            "Expected exactly one artifact with SHA256 "
            f"{expected_sha256}; found {len(matches)}: "
            f"{[str(p) for p in matches]}"
        ),
    )

    return matches[0]


def normalize_split_value(
    value,
) -> str:
    text = str(value).strip().lower()

    mapping = {
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "test": "test",
    }

    require(
        text in mapping,
        (
            "Unexpected evaluation split value: "
            f"{value!r}"
        ),
    )

    return mapping[text]


def choose_column(
    columns: list[str],
    candidates: list[str],
    label: str,
) -> str:
    lower_to_actual = {
        str(column).lower(): str(column)
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[
                candidate.lower()
            ]

    raise AssertionError(
        f"Could not identify {label} column. "
        f"Available columns: {columns}"
    )


# =============================================================================
# Frozen ranking / metric implementation
# =============================================================================

def rank_candidates(
    logits: np.ndarray,
    startup_local: np.ndarray,
    positive_startup_local: int,
) -> int:
    """
    Return the positive candidate's deterministic 1-based rank.

    Sort:
        1. logit descending
        2. startup_local ascending
    """

    logits = np.asarray(
        logits,
        dtype=np.float64,
    )

    startup_local = np.asarray(
        startup_local,
        dtype=np.int64,
    )

    require(
        logits.ndim == 1,
        "Logits must be one-dimensional.",
    )

    require(
        startup_local.ndim == 1,
        "startup_local must be one-dimensional.",
    )

    require(
        len(logits)
        == len(startup_local),
        "Score/candidate length mismatch.",
    )

    require(
        len(logits)
        == EXPECTED_CANDIDATES_PER_CASE,
        (
            "Each evaluation case must contain "
            "exactly 100 candidates."
        ),
    )

    require(
        np.isfinite(
            logits
        ).all(),
        (
            "Evaluation logits contain NaN/inf. "
            "Evaluation must fail rather than rank "
            "non-finite values."
        ),
    )

    require(
        len(
            np.unique(
                startup_local
            )
        )
        == EXPECTED_CANDIDATES_PER_CASE,
        (
            "Evaluation candidate startup IDs "
            "must be unique within one case."
        ),
    )

    positive_matches = np.flatnonzero(
        startup_local
        == int(
            positive_startup_local
        )
    )

    require(
        len(
            positive_matches
        )
        == 1,
        (
            "Positive startup must occur exactly "
            "once in candidate set."
        ),
    )

    # np.lexsort uses the LAST key as primary.
    # Primary: -logit ascending == logit descending
    # Secondary: startup_local ascending
    order = np.lexsort(
        (
            startup_local,
            -logits,
        )
    )

    ranked_startups = startup_local[
        order
    ]

    positive_rank_positions = np.flatnonzero(
        ranked_startups
        == int(
            positive_startup_local
        )
    )

    require(
        len(
            positive_rank_positions
        )
        == 1,
        (
            "Positive startup disappeared "
            "during ranking."
        ),
    )

    return int(
        positive_rank_positions[0]
        + 1
    )


def metrics_from_positive_rank(
    positive_rank: int,
    k: int = K,
) -> tuple[float, float]:
    require(
        isinstance(
            positive_rank,
            int,
        ),
        "Positive rank must be Python int.",
    )

    require(
        1
        <= positive_rank
        <= EXPECTED_CANDIDATES_PER_CASE,
        (
            "Positive rank outside 1..100."
        ),
    )

    require(
        k == 10,
        "Frozen evaluation cutoff is k=10.",
    )

    if positive_rank <= k:
        hr = 1.0
        ndcg = (
            1.0
            / math.log2(
                positive_rank
                + 1
            )
        )
    else:
        hr = 0.0
        ndcg = 0.0

    return (
        float(hr),
        float(ndcg),
    )


def evaluate_case(
    logits: np.ndarray,
    startup_local: np.ndarray,
    positive_startup_local: int,
) -> dict:
    rank = rank_candidates(
        logits,
        startup_local,
        positive_startup_local,
    )

    hr10, ndcg10 = (
        metrics_from_positive_rank(
            rank,
            k=10,
        )
    )

    return {
        "positive_rank": (
            rank
        ),
        "HR@10": (
            hr10
        ),
        "NDCG@10": (
            ndcg10
        ),
    }


def aggregate_event_level_metrics(
    case_metrics: pd.DataFrame,
) -> tuple[float, float]:
    require(
        len(
            case_metrics
        )
        > 0,
        "Cannot aggregate empty case set.",
    )

    require(
        {
            "HR@10",
            "NDCG@10",
        }.issubset(
            case_metrics.columns
        ),
        (
            "Case-metric frame missing "
            "HR@10/NDCG@10."
        ),
    )

    return (
        float(
            case_metrics[
                "HR@10"
            ].mean()
        ),
        float(
            case_metrics[
                "NDCG@10"
            ].mean()
        ),
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.3a — "
        "VALIDATION RANKING METRIC SEMANTICS AUDIT AND FREEZE"
    )

    print(
        "Neural model instantiated:            NO"
    )
    print(
        "Validation forward executed:          NO"
    )
    print(
        "Validation learned scores consumed:   NO"
    )
    print(
        "Test used for model selection:        NO"
    )

    # =========================================================================
    # Prerequisite controller
    # =========================================================================

    banner(
        "AUTHORITATIVE PRE-VALIDATION GATE RECHECK"
    )

    require(
        PHASE_5_3_2C_CONTRACT_PATH.exists(),
        (
            "Missing Phase-5.3.2c production "
            "controller contract."
        ),
    )

    controller_contract = load_json(
        PHASE_5_3_2C_CONTRACT_PATH
    )

    require(
        controller_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.2c contract "
            "is not frozen."
        ),
    )

    require(
        controller_contract[
            "end_epoch_controller"
        ][
            "next_action"
        ]
        == "VALIDATE",
        (
            "Production controller does not "
            "enter VALIDATE after epoch completion."
        ),
    )

    require(
        controller_contract[
            "boundary"
        ][
            "test_accessed"
        ]
        is False,
        (
            "Test was accessed before "
            "validation runtime freeze."
        ),
    )

    print(
        "Phase-5.3.2c controller:              FROZEN / PASS"
    )

    # =========================================================================
    # Discover exact frozen Phase-5.1.2c candidate artifacts by SHA
    # =========================================================================

    banner(
        "DISCOVER FROZEN PHASE-5.1.2c EVALUATION ARTIFACTS BY SHA256"
    )

    negative_matrix_path = (
        find_file_by_sha256(
            PHASE5_ROOT,
            EXPECTED_NEGATIVE_MATRIX_FILE_SHA256,
            {
                ".npy",
            },
        )
    )

    case_manifest_path = (
        find_file_by_sha256(
            PHASE5_ROOT,
            EXPECTED_CASE_MANIFEST_FILE_SHA256,
            {
                ".parquet",
            },
        )
    )

    phase_5_1_2c_manifest_path = (
        find_file_by_sha256(
            PHASE5_ROOT,
            EXPECTED_PHASE_5_1_2C_MANIFEST_FILE_SHA256,
            {
                ".json",
            },
        )
    )

    discovery_df = pd.DataFrame(
        [
            {
                "artifact": (
                    "negative_matrix"
                ),
                "path": str(
                    negative_matrix_path
                ),
                "sha256": (
                    file_sha256(
                        negative_matrix_path
                    )
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "artifact": (
                    "case_manifest"
                ),
                "path": str(
                    case_manifest_path
                ),
                "sha256": (
                    file_sha256(
                        case_manifest_path
                    )
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "artifact": (
                    "phase_5_1_2c_manifest"
                ),
                "path": str(
                    phase_5_1_2c_manifest_path
                ),
                "sha256": (
                    file_sha256(
                        phase_5_1_2c_manifest_path
                    )
                ),
                "status": (
                    "PASS"
                ),
            },
        ]
    )

    print(
        discovery_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Candidate integrity
    # =========================================================================

    banner(
        "FROZEN CANDIDATE-SET INTEGRITY"
    )

    negative_matrix = np.load(
        negative_matrix_path,
        mmap_mode="r",
    )

    require(
        negative_matrix.shape
        == (
            EXPECTED_CASES,
            EXPECTED_NEGATIVES_PER_CASE,
        ),
        (
            "Frozen evaluation negative matrix "
            "shape is not (22515, 99)."
        ),
    )

    require(
        np.issubdtype(
            negative_matrix.dtype,
            np.integer,
        ),
        (
            "Evaluation negative matrix "
            "must contain integer startup indices."
        ),
    )

    # All 99 negatives must be unique inside every case.
    # Vectorized sort avoids Python iteration over 22,515 rows.
    sorted_negatives = np.sort(
        np.asarray(
            negative_matrix
        ),
        axis=1,
    )

    duplicate_inside_case = bool(
        np.any(
            sorted_negatives[
                :,
                1:
            ]
            == sorted_negatives[
                :,
                :-1
            ]
        )
    )

    require(
        duplicate_inside_case
        is False,
        (
            "At least one frozen evaluation case "
            "contains duplicate negative startups."
        ),
    )

    parquet_file = pq.ParquetFile(
        case_manifest_path
    )

    manifest_rows = int(
        parquet_file.metadata.num_rows
    )

    manifest_columns = list(
        parquet_file.schema.names
    )

    require(
        manifest_rows
        == EXPECTED_CASES,
        (
            "Evaluation case manifest row "
            "count drift."
        ),
    )

    split_column = choose_column(
        manifest_columns,
        [
            "split",
            "experiment_split",
            "evaluation_split",
            "holdout_split",
        ],
        "evaluation split",
    )

    interaction_column = choose_column(
        manifest_columns,
        [
            "interaction_id",
            "case_interaction_id",
            "evaluation_interaction_id",
        ],
        "interaction identity",
    )

    case_frame = pd.read_parquet(
        case_manifest_path,
        columns=[
            split_column,
            interaction_column,
        ],
    )

    normalized_split = (
        case_frame[
            split_column
        ]
        .map(
            normalize_split_value
        )
    )

    validation_cases = int(
        (
            normalized_split
            == "validation"
        ).sum()
    )

    test_cases = int(
        (
            normalized_split
            == "test"
        ).sum()
    )

    require(
        validation_cases
        == EXPECTED_VALIDATION_CASES,
        (
            "Validation case count drift."
        ),
    )

    require(
        test_cases
        == EXPECTED_TEST_CASES,
        (
            "Test case count drift."
        ),
    )

    require(
        validation_cases
        + test_cases
        == EXPECTED_CASES,
        (
            "Validation/test case counts "
            "do not cover all T60 cases."
        ),
    )

    require(
        case_frame[
            interaction_column
        ].astype(
            str
        ).is_unique,
        (
            "Evaluation event identity "
            "must be unique by interaction_id."
        ),
    )

    candidate_integrity_df = pd.DataFrame(
        [
            {
                "check": (
                    "evaluation_cases"
                ),
                "actual": (
                    EXPECTED_CASES
                ),
                "expected": (
                    EXPECTED_CASES
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "negatives_per_case"
                ),
                "actual": (
                    negative_matrix.shape[
                        1
                    ]
                ),
                "expected": (
                    99
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "candidates_per_case"
                ),
                "actual": (
                    1
                    + negative_matrix.shape[
                        1
                    ]
                ),
                "expected": (
                    100
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "validation_cases"
                ),
                "actual": (
                    validation_cases
                ),
                "expected": (
                    2251
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "test_cases"
                ),
                "actual": (
                    test_cases
                ),
                "expected": (
                    20264
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "99_negatives_unique_per_case"
                ),
                "actual": (
                    str(
                        not duplicate_inside_case
                    )
                ),
                "expected": (
                    "True"
                ),
                "status": (
                    "PASS"
                ),
            },
            {
                "check": (
                    "event_identity_unique"
                ),
                "actual": (
                    str(
                        case_frame[
                            interaction_column
                        ].astype(
                            str
                        ).is_unique
                    )
                ),
                "expected": (
                    "True"
                ),
                "status": (
                    "PASS"
                ),
            },
        ]
    )

    print(
        candidate_integrity_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Formula probes
    # =========================================================================

    banner(
        "HR@10 / NDCG@10 FORMULA PROBES"
    )

    formula_rows = []

    probe_ranks = [
        1,
        2,
        5,
        10,
        11,
        100,
    ]

    for rank in probe_ranks:
        hr10, ndcg10 = (
            metrics_from_positive_rank(
                rank
            )
        )

        expected_hr = (
            1.0
            if rank <= 10
            else 0.0
        )

        expected_ndcg = (
            1.0
            / math.log2(
                rank
                + 1
            )
            if rank <= 10
            else 0.0
        )

        require(
            hr10 == expected_hr,
            (
                f"HR@10 formula failed "
                f"at rank {rank}."
            ),
        )

        require(
            ndcg10 == expected_ndcg,
            (
                f"NDCG@10 formula failed "
                f"at rank {rank}."
            ),
        )

        formula_rows.append(
            {
                "positive_rank": (
                    rank
                ),
                "HR@10": (
                    hr10
                ),
                "NDCG@10": (
                    ndcg10
                ),
                "expected_HR@10": (
                    expected_hr
                ),
                "expected_NDCG@10": (
                    expected_ndcg
                ),
                "status": (
                    "PASS"
                ),
            }
        )

    formula_df = pd.DataFrame(
        formula_rows
    )

    print(
        formula_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Ranking probes with 100 candidates
    # =========================================================================

    banner(
        "DETERMINISTIC 100-CANDIDATE RANKING PROBES"
    )

    candidate_startups = np.arange(
        10_000,
        10_100,
        dtype=np.int64,
    )

    ranking_rows = []

    desired_ranks = [
        1,
        10,
        11,
        100,
    ]

    for desired_rank in desired_ranks:
        positive_startup = int(
            candidate_startups[
                50
            ]
        )

        # Strictly decreasing unique scores.
        logits = np.linspace(
            1.0,
            0.0,
            num=100,
            dtype=np.float64,
        )

        # Move positive score to exact desired rank while keeping all
        # scores unique.
        sorted_scores = np.sort(
            logits
        )[
            ::-1
        ]

        positive_score = (
            sorted_scores[
                desired_rank
                - 1
            ]
        )

        positive_index = int(
            np.flatnonzero(
                candidate_startups
                == positive_startup
            )[0]
        )

        swap_index = int(
            np.flatnonzero(
                logits
                == positive_score
            )[0]
        )

        temporary = float(
            logits[
                positive_index
            ]
        )

        logits[
            positive_index
        ] = logits[
            swap_index
        ]

        logits[
            swap_index
        ] = temporary

        result = evaluate_case(
            logits,
            candidate_startups,
            positive_startup,
        )

        require(
            result[
                "positive_rank"
            ]
            == desired_rank,
            (
                "Ranking probe failed: "
                f"expected rank {desired_rank}, "
                f"got {result['positive_rank']}."
            ),
        )

        ranking_rows.append(
            {
                "probe": (
                    f"strict_rank_{desired_rank}"
                ),
                "positive_rank": (
                    result[
                        "positive_rank"
                    ]
                ),
                "HR@10": (
                    result[
                        "HR@10"
                    ]
                ),
                "NDCG@10": (
                    result[
                        "NDCG@10"
                    ]
                ),
                "status": (
                    "PASS"
                ),
            }
        )

    ranking_df = pd.DataFrame(
        ranking_rows
    )

    print(
        ranking_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Tie-break probes
    # =========================================================================

    banner(
        "LABEL-INDEPENDENT TIE-BREAK PROBES"
    )

    tie_rows = []

    # All candidates use unique startup IDs 20000..20099.
    startups = np.arange(
        20_000,
        20_100,
        dtype=np.int64,
    )

    # Probe A:
    # startup 20050 (positive) ties with 20040.
    # Lower startup_local=20040 must rank ahead.
    logits = np.linspace(
        2.0,
        -2.0,
        num=100,
        dtype=np.float64,
    )

    positive = 20_050
    tied_other = 20_040

    positive_index = int(
        positive
        - 20_000
    )

    other_index = int(
        tied_other
        - 20_000
    )

    shared_score = 3.0

    logits[
        positive_index
    ] = shared_score

    logits[
        other_index
    ] = shared_score

    rank_positive = rank_candidates(
        logits,
        startups,
        positive,
    )

    rank_other = rank_candidates(
        logits,
        startups,
        tied_other,
    )

    require(
        rank_other
        < rank_positive,
        (
            "Tie-break failed: lower "
            "startup_local did not rank first."
        ),
    )

    tie_rows.append(
        {
            "probe": (
                "lower_startup_local_wins_equal_logit"
            ),
            "startup_A": (
                tied_other
            ),
            "startup_B": (
                positive
            ),
            "equal_logit": (
                shared_score
            ),
            "rank_A": (
                rank_other
            ),
            "rank_B": (
                rank_positive
            ),
            "expected": (
                "startup_A_before_startup_B"
            ),
            "status": (
                "PASS"
            ),
        }
    )

    # Probe B:
    # Reverse which tied startup is the "positive". Ordering must NOT change
    # because the tie-break is label-independent.
    result_when_lower_is_positive = (
        evaluate_case(
            logits,
            startups,
            tied_other,
        )
    )

    result_when_higher_is_positive = (
        evaluate_case(
            logits,
            startups,
            positive,
        )
    )

    require(
        result_when_lower_is_positive[
            "positive_rank"
        ]
        == rank_other,
        (
            "Tie ordering changed when lower "
            "startup became positive."
        ),
    )

    require(
        result_when_higher_is_positive[
            "positive_rank"
        ]
        == rank_positive,
        (
            "Tie ordering changed when higher "
            "startup became positive."
        ),
    )

    tie_rows.append(
        {
            "probe": (
                "tie_break_independent_of_positive_label"
            ),
            "startup_A": (
                tied_other
            ),
            "startup_B": (
                positive
            ),
            "equal_logit": (
                shared_score
            ),
            "rank_A": (
                result_when_lower_is_positive[
                    "positive_rank"
                ]
            ),
            "rank_B": (
                result_when_higher_is_positive[
                    "positive_rank"
                ]
            ),
            "expected": (
                "same_order_regardless_of_label"
            ),
            "status": (
                "PASS"
            ),
        }
    )

    tie_df = pd.DataFrame(
        tie_rows
    )

    print(
        tie_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Raw-logit versus sigmoid ranking equivalence
    # =========================================================================

    banner(
        "RAW-LOGIT vs SIGMOID RANKING EQUIVALENCE"
    )

    logits = np.linspace(
        -8.0,
        8.0,
        num=100,
        dtype=np.float64,
    )

    startups = np.arange(
        30_000,
        30_100,
        dtype=np.int64,
    )

    positive = 30_037

    raw_rank = rank_candidates(
        logits,
        startups,
        positive,
    )

    sigmoid_scores = (
        1.0
        / (
            1.0
            + np.exp(
                -logits
            )
        )
    )

    sigmoid_rank = rank_candidates(
        sigmoid_scores,
        startups,
        positive,
    )

    require(
        raw_rank == sigmoid_rank,
        (
            "Finite raw-logit and sigmoid "
            "rankings are not equivalent."
        ),
    )

    print(
        f"Raw-logit positive rank:              {raw_rank}"
    )
    print(
        f"Sigmoid positive rank:                {sigmoid_rank}"
    )
    print(
        "Ranking equivalence:                  PASS"
    )

    # =========================================================================
    # Event-level aggregation proof
    # =========================================================================

    banner(
        "EVENT-LEVEL AGGREGATION PROOF"
    )

    synthetic_cases = pd.DataFrame(
        [
            {
                "HR@10": 1.0,
                "NDCG@10": 1.0,
            },
            {
                "HR@10": 1.0,
                "NDCG@10": (
                    1.0
                    / math.log2(
                        11
                    )
                ),
            },
            {
                "HR@10": 0.0,
                "NDCG@10": 0.0,
            },
        ]
    )

    aggregate_hr, aggregate_ndcg = (
        aggregate_event_level_metrics(
            synthetic_cases
        )
    )

    expected_aggregate_hr = (
        2.0
        / 3.0
    )

    expected_aggregate_ndcg = (
        (
            1.0
            + (
                1.0
                / math.log2(
                    11
                )
            )
            + 0.0
        )
        / 3.0
    )

    require(
        aggregate_hr
        == expected_aggregate_hr,
        (
            "Event-level HR aggregation failed."
        ),
    )

    require(
        aggregate_ndcg
        == expected_aggregate_ndcg,
        (
            "Event-level NDCG aggregation failed."
        ),
    )

    print(
        f"Synthetic event-level mean HR@10:     "
        f"{aggregate_hr:.12f}"
    )
    print(
        f"Synthetic event-level mean NDCG@10:   "
        f"{aggregate_ndcg:.12f}"
    )

    # =========================================================================
    # Non-finite guard proof
    # =========================================================================

    banner(
        "NON-FINITE SCORE FAIL-CLOSED PROOF"
    )

    finite_guard_passed = False

    bad_logits = np.zeros(
        100,
        dtype=np.float64,
    )

    bad_logits[
        17
    ] = np.nan

    startups = np.arange(
        40_000,
        40_100,
        dtype=np.int64,
    )

    try:
        rank_candidates(
            bad_logits,
            startups,
            40_000,
        )
    except AssertionError:
        finite_guard_passed = True

    require(
        finite_guard_passed,
        (
            "Non-finite score guard "
            "did not fail closed."
        ),
    )

    print(
        "NaN ranking attempt rejected:         PASS"
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.3a INVARIANTS"
    )

    checks = [
        (
            "phase_5_3_2c_controller_frozen",
            (
                controller_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),
        (
            "negative_matrix_file_hash_exact",
            (
                file_sha256(
                    negative_matrix_path
                )
                == EXPECTED_NEGATIVE_MATRIX_FILE_SHA256
            ),
        ),
        (
            "case_manifest_file_hash_exact",
            (
                file_sha256(
                    case_manifest_path
                )
                == EXPECTED_CASE_MANIFEST_FILE_SHA256
            ),
        ),
        (
            "phase_5_1_2c_manifest_file_hash_exact",
            (
                file_sha256(
                    phase_5_1_2c_manifest_path
                )
                == EXPECTED_PHASE_5_1_2C_MANIFEST_FILE_SHA256
            ),
        ),
        (
            "evaluation_cases_22515",
            (
                manifest_rows
                == EXPECTED_CASES
            ),
        ),
        (
            "validation_cases_2251",
            (
                validation_cases
                == EXPECTED_VALIDATION_CASES
            ),
        ),
        (
            "test_cases_20264",
            (
                test_cases
                == EXPECTED_TEST_CASES
            ),
        ),
        (
            "negative_matrix_22515_by_99",
            (
                negative_matrix.shape
                == (
                    22515,
                    99,
                )
            ),
        ),
        (
            "99_negatives_unique_per_case",
            (
                duplicate_inside_case
                is False
            ),
        ),
        (
            "ranking_score_raw_logit",
            True,
        ),
        (
            "primary_sort_logit_descending",
            True,
        ),
        (
            "tie_break_startup_local_ascending",
            True,
        ),
        (
            "tie_break_label_independent",
            True,
        ),
        (
            "positive_rank_one_based",
            True,
        ),
        (
            "HR10_formula_exact",
            bool(
                (
                    formula_df[
                        "HR@10"
                    ]
                    == formula_df[
                        "expected_HR@10"
                    ]
                ).all()
            ),
        ),
        (
            "NDCG10_formula_exact",
            bool(
                (
                    formula_df[
                        "NDCG@10"
                    ]
                    == formula_df[
                        "expected_NDCG@10"
                    ]
                ).all()
            ),
        ),
        (
            "raw_logit_sigmoid_rank_equivalent",
            (
                raw_rank
                == sigmoid_rank
            ),
        ),
        (
            "event_level_mean_aggregation",
            (
                aggregate_hr
                == expected_aggregate_hr
                and aggregate_ndcg
                == expected_aggregate_ndcg
            ),
        ),
        (
            "nonfinite_scores_fail_closed",
            (
                finite_guard_passed
            ),
        ),
        (
            "model_not_instantiated",
            True,
        ),
        (
            "validation_forward_not_executed",
            True,
        ),
        (
            "test_not_used_for_model_selection",
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
            "At least one Phase-5.3.3a "
            "validation-ranking invariant failed."
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
        "WRITE PHASE-5.3.3a OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    discovery_df.to_csv(
        ARTIFACT_DISCOVERY_PATH,
        index=False,
    )

    candidate_integrity_df.to_csv(
        CANDIDATE_INTEGRITY_PATH,
        index=False,
    )

    ranking_df.to_csv(
        RANKING_PROBE_PATH,
        index=False,
    )

    tie_df.to_csv(
        TIE_PROBE_PATH,
        index=False,
    )

    formula_df.to_csv(
        FORMULA_PATH,
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
                    "evaluation_score_for_ranking"
                ),
                "value": (
                    "RAW_MODEL_LOGIT"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3a"
                ),
            },
            {
                "decision": (
                    "evaluation_primary_sort"
                ),
                "value": (
                    "LOGIT_DESCENDING"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3a"
                ),
            },
            {
                "decision": (
                    "evaluation_tie_break"
                ),
                "value": (
                    "STARTUP_LOCAL_ASCENDING"
                ),
                "classification": (
                    "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3a"
                ),
            },
            {
                "decision": (
                    "positive_rank_indexing"
                ),
                "value": (
                    "ONE_BASED"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3a"
                ),
            },
            {
                "decision": (
                    "HR10"
                ),
                "value": (
                    "1_IF_POSITIVE_RANK_LE_10_ELSE_0"
                ),
                "classification": (
                    "PAPER_SPECIFIED"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3a"
                ),
            },
            {
                "decision": (
                    "NDCG10"
                ),
                "value": (
                    "1_OVER_LOG2_RANK_PLUS_1_IF_RANK_LE_10_ELSE_0"
                ),
                "classification": (
                    "PAPER_SPECIFIED_METRIC_STANDARD_SINGLE_RELEVANT_ITEM"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3a"
                ),
            },
            {
                "decision": (
                    "split_metric_aggregation"
                ),
                "value": (
                    "ARITHMETIC_MEAN_OVER_EVENT_LEVEL_CASES"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3a"
                ),
            },
            {
                "decision": (
                    "nonfinite_evaluation_scores"
                ),
                "value": (
                    "FAIL_CLOSED"
                ),
                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_3a"
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
            "5.3.3a"
        ),
        "title": (
            "Validation Ranking Metric Semantics Contract"
        ),
        "status": (
            "FROZEN"
        ),
        "candidate_runtime": {
            "cases": (
                EXPECTED_CASES
            ),
            "validation_cases": (
                EXPECTED_VALIDATION_CASES
            ),
            "test_cases": (
                EXPECTED_TEST_CASES
            ),
            "positive_candidates_per_case": (
                1
            ),
            "negative_candidates_per_case": (
                EXPECTED_NEGATIVES_PER_CASE
            ),
            "candidates_per_case": (
                EXPECTED_CANDIDATES_PER_CASE
            ),
            "case_identity": (
                "interaction_id"
            ),
            "aggregation_unit": (
                "event-level case"
            ),
            "deduplication": (
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
            "tie_break_label_independent": (
                True
            ),
            "positive_rank": (
                "1-based"
            ),
            "nonfinite_score_policy": (
                "fail closed"
            ),
        },
        "metrics": {
            "cutoff_k": (
                10
            ),
            "HR@10": (
                "1 if positive_rank <= 10 else 0"
            ),
            "NDCG@10": (
                "1/log2(positive_rank+1) "
                "if positive_rank <= 10 else 0"
            ),
            "IDCG@10_single_relevant_item": (
                1.0
            ),
            "split_aggregation": (
                "arithmetic mean over event-level cases"
            ),
        },
        "frozen_artifacts": {
            "negative_matrix": {
                "path": str(
                    negative_matrix_path
                ),
                "file_sha256": (
                    EXPECTED_NEGATIVE_MATRIX_FILE_SHA256
                ),
            },
            "case_manifest": {
                "path": str(
                    case_manifest_path
                ),
                "file_sha256": (
                    EXPECTED_CASE_MANIFEST_FILE_SHA256
                ),
                "split_column": (
                    split_column
                ),
                "interaction_column": (
                    interaction_column
                ),
            },
            "phase_5_1_2c_manifest": {
                "path": str(
                    phase_5_1_2c_manifest_path
                ),
                "file_sha256": (
                    EXPECTED_PHASE_5_1_2C_MANIFEST_FILE_SHA256
                ),
            },
        },
        "boundary": {
            "neural_model_instantiated": (
                False
            ),
            "validation_forward_executed": (
                False
            ),
            "learned_validation_metric_computed": (
                False
            ),
            "test_used_for_model_selection": (
                False
            ),
        },
        "next_phase": {
            "id": (
                "5.3.3b"
            ),
            "title": (
                "Canonical Real-Model Validation Scoring Preflight"
            ),
            "requirement": (
                "Use the canonical model and frozen validation cases to "
                "score real 1+99 candidate lists, verify exact candidate "
                "binding, raw-logit ranking, HR@10/NDCG@10 calculation, "
                "and validation-only access before integrating validation "
                "into the production trainer."
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
            "5.3.3a"
        ),
        "status": (
            "VALIDATION_RANKING_METRIC_SEMANTICS_"
            "FROZEN"
        ),
        "validation_cases": (
            validation_cases
        ),
        "test_cases": (
            test_cases
        ),
        "candidates_per_case": (
            EXPECTED_CANDIDATES_PER_CASE
        ),
        "ranking_score": (
            "raw_logit"
        ),
        "tie_break": (
            "startup_local_ascending"
        ),
        "cutoff": (
            K
        ),
        "model_instantiated": (
            False
        ),
        "validation_forward_executed": (
            False
        ),
        "test_used_for_model_selection": (
            False
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
        ARTIFACT_DISCOVERY_PATH,
        CANDIDATE_INTEGRITY_PATH,
        RANKING_PROBE_PATH,
        TIE_PROBE_PATH,
        FORMULA_PATH,
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
        "PHASE 5.3.3a FINAL STATUS"
    )

    print(
        "Evaluation cases:                     22,515"
    )
    print(
        "Validation cases:                     2,251"
    )
    print(
        "Test cases:                           20,264"
    )
    print(
        "Candidates / case:                    100"
    )
    print()

    print(
        "Ranking score:                        RAW LOGIT"
    )
    print(
        "Primary sort:                         LOGIT DESCENDING"
    )
    print(
        "Tie break:                            STARTUP_LOCAL ASCENDING"
    )
    print(
        "Tie break uses positive label:        NO"
    )
    print(
        "Positive rank indexing:               1-BASED"
    )
    print()

    print(
        "HR@10:"
    )
    print(
        "  1 if positive_rank <= 10 else 0"
    )
    print()

    print(
        "NDCG@10:"
    )
    print(
        "  1/log2(positive_rank+1) if rank <= 10 else 0"
    )
    print()

    print(
        "Split aggregation:                    EVENT-LEVEL MEAN"
    )
    print(
        "Non-finite score policy:              FAIL CLOSED"
    )
    print()

    print(
        "Neural model instantiated:            NO"
    )
    print(
        "Validation forward executed:          NO"
    )
    print(
        "Test used for model selection:        NO"
    )

    banner(
        "PHASE 5.3.3a COMPLETE / "
        "VALIDATION RANKING METRIC SEMANTICS FROZEN"
    )


if __name__ == "__main__":
    main()