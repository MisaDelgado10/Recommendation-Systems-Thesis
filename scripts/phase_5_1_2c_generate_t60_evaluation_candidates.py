"""
Phase 5.1.2c — Deterministic T60 Evaluation Candidate Generation
and Realized-Collision Audit

THIS SCRIPT GENERATES THE FIRST REAL NEGATIVE EXAMPLES IN PHASE 5.

It DOES:
- load the frozen Phase-5.1.2b evaluation contract;
- instantiate deterministic per-case NumPy PCG64 RNGs;
- generate exactly 99 distinct negatives for each T60 event;
- preserve the frozen event-level validation/test split;
- persist the immutable evaluation candidate matrix;
- audit realized collisions against other T60 positives AFTER generation;
- verify deterministic regeneration exactly.

It DOES NOT:
- alter Phase-2 splits;
- alter the Phase-3 node universe;
- alter the Phase-4 model;
- generate training negatives;
- instantiate the ITRS model;
- create an optimizer;
- train;
- compute HR@10/NDCG@10;
- resample negatives that collide with other T60 positives;
- choose epoch count;
- choose early stopping;
- choose weight decay.

Frozen evaluation rule
----------------------
For focal T60 event e=(o,b):

    N_eval(o,b)
        = all frozen Phase-3 Startup role nodes
          minus Startups positive for investor o before T60
          minus focal Startup b

Then:

    draw exactly 99 distinct candidates
    uniformly without replacement
    once and persist them forever.

Other T60 labels MUST NOT influence generation.

They may be consulted only AFTER generation for diagnostics.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

TEMPORAL_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "interactions_itrs_temporal_split.parquet"
)

NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/model_ready/"
    "node_index.parquet"
)

EVALUATION_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_1_2b_t60_evaluation_candidate_runtime_contract.json"
)

AUDIT_5_1_2A_SUMMARY_PATH = Path(
    "data/experimental/phase_5/audits/phase_5_1_2a/"
    "t60_evaluation_pool_collision_summary.csv"
)

OUT_DIR = Path(
    "data/experimental/phase_5/model_ready/evaluation"
)

NEGATIVE_MATRIX_PATH = (
    OUT_DIR
    / "t60_evaluation_negative_node_indices.npy"
)

CASE_MANIFEST_PATH = (
    OUT_DIR
    / "t60_evaluation_case_manifest.parquet"
)

COLLISION_DETAIL_PATH = (
    OUT_DIR
    / "t60_evaluation_realized_t60_collision_detail.parquet"
)

COLLISION_SUMMARY_PATH = (
    OUT_DIR
    / "t60_evaluation_realized_collision_summary.csv"
)

GENERATION_AUDIT_PATH = (
    OUT_DIR
    / "t60_evaluation_generation_integrity_audit.csv"
)

ARTIFACT_HASH_PATH = (
    OUT_DIR
    / "t60_evaluation_artifact_hashes.csv"
)

MANIFEST_PATH = (
    OUT_DIR
    / "phase_5_1_2c_generation_manifest.json"
)


# =============================================================================
# Frozen values
# =============================================================================

EXPECTED = {
    "temporal_rows": 1_195_937,
    "t0_t59_rows": 1_173_422,
    "t60_rows": 22_515,
    "validation_rows": 2_251,
    "test_rows": 20_264,

    "node_rows": 477_564,
    "investor_nodes": 165_975,
    "startup_nodes": 311_589,

    "evaluation_negatives_per_case": 99,

    "t60_unique_pairs": 22_327,
    "validation_test_pair_overlap": 33,
    "validation_test_funding_round_overlap": 1_315,
}

EXPECTED_TOTAL_NEGATIVE_SLOTS = (
    EXPECTED["t60_rows"]
    * EXPECTED[
        "evaluation_negatives_per_case"
    ]
)

EXPECTED_TOTAL_CANDIDATE_SLOTS = (
    EXPECTED["t60_rows"]
    * (
        1
        + EXPECTED[
            "evaluation_negatives_per_case"
        ]
    )
)

EVALUATION_BASE_SEED = 42

SEED_NAMESPACE = (
    "ITRS_PHASE5_EVAL_NEGATIVE"
)

NEGATIVES_PER_CASE = 99

PROPOSAL_BATCH_SIZE = 256


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


def sha256_int64_vector(
    values: np.ndarray,
) -> str:
    """
    Logical hash of an int64 vector.

    This is used per evaluation case so the 99-candidate
    list can be checked independently later.
    """

    array = np.asarray(
        values,
        dtype="<i8",
    )

    return hashlib.sha256(
        array.tobytes(
            order="C"
        )
    ).hexdigest()


def derive_case_seed(
    experiment_split: str,
    interaction_id: str,
) -> int:
    """
    Frozen Phase-5.1.2b seed derivation.

    material:
        ITRS_PHASE5_EVAL_NEGATIVE
        |42
        |<split>
        |<interaction_id>

    SHA256 -> first 8 bytes little-endian
    -> modulo (2^63 - 1)
    """

    split = str(
        experiment_split
    ).strip().lower()

    interaction = str(
        interaction_id
    )

    require(
        split in {
            "validation",
            "test",
        },
        f"Unexpected evaluation split: {split}",
    )

    require(
        len(interaction) > 0,
        "interaction_id cannot be empty",
    )

    material = (
        f"{SEED_NAMESPACE}|"
        f"{EVALUATION_BASE_SEED}|"
        f"{split}|"
        f"{interaction}"
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


def get_grouped_set_mapping(
    df: pd.DataFrame,
    key_col: str,
    value_col: str,
) -> dict:
    """
    Return:
        key -> Python set(values)
    """

    return (
        df.groupby(
            key_col,
            sort=False,
            observed=True,
        )[value_col]
        .agg(
            lambda values:
            set(
                values.tolist()
            )
        )
        .to_dict()
    )


def get_set(
    mapping: dict,
    key,
) -> set:

    return mapping.get(
        key,
        set(),
    )


def generate_uniform_negatives_by_rejection(
    *,
    universe_size: int,
    excluded_indices: set,
    focal_index: int,
    k: int,
    seed: int,
) -> np.ndarray:
    """
    Draw K distinct candidate positions uniformly from:

        {0, ..., universe_size-1}
        minus excluded_indices
        minus focal_index

    Method
    ------
    Uniform independent proposals are generated over the full
    universe.

    A proposal is rejected when:
    - historically excluded;
    - equal to focal Startup;
    - already accepted in this case.

    Sequential rejection of invalid/duplicate proposals gives
    a uniform sample without replacement from the eligible set.

    A dedicated per-case PCG64 generator is instantiated here.
    """

    require(
        universe_size > 0,
        "universe_size must be positive",
    )

    require(
        k > 0,
        "k must be positive",
    )

    require(
        0 <= focal_index < universe_size,
        "focal_index outside Startup universe",
    )

    focal_already_historical = (
        focal_index
        in excluded_indices
    )

    excluded_count = (
        len(
            excluded_indices
        )
        + (
            0
            if focal_already_historical
            else 1
        )
    )

    eligible_count = (
        universe_size
        - excluded_count
    )

    require(
        eligible_count >= k,
        (
            "Eligible evaluation pool cannot "
            f"support k={k}: {eligible_count}"
        ),
    )

    rng = np.random.Generator(
        np.random.PCG64(
            seed
        )
    )

    accepted = np.empty(
        k,
        dtype=np.int64,
    )

    accepted_set = set()

    cursor = 0

    while cursor < k:

        proposals = rng.integers(
            low=0,
            high=universe_size,
            size=PROPOSAL_BATCH_SIZE,
            dtype=np.int64,
        )

        for proposal_np in proposals:

            proposal = int(
                proposal_np
            )

            if proposal == focal_index:
                continue

            if proposal in excluded_indices:
                continue

            if proposal in accepted_set:
                continue

            accepted[
                cursor
            ] = proposal

            accepted_set.add(
                proposal
            )

            cursor += 1

            if cursor == k:
                break

    require(
        len(
            accepted_set
        )
        == k,
        "Generated evaluation negatives are not unique",
    )

    return accepted


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.1.2c — "
        "DETERMINISTIC T60 EVALUATION CANDIDATE GENERATION "
        "AND REALIZED-COLLISION AUDIT"
    )

    print(
        "Evaluation negatives WILL be generated in this phase."
    )

    print(
        "Evaluation RNG WILL be instantiated."
    )

    print(
        "Training negatives generated:          NO"
    )

    print(
        "Model instantiated:                    NO"
    )

    print(
        "Optimizer created:                     NO"
    )

    print(
        "Training performed:                    NO"
    )

    print(
        "HR/NDCG evaluation performed:          NO"
    )

    print(
        "Detected T60 collisions resampled:     NEVER"
    )

    # =========================================================================
    # Inputs
    # =========================================================================

    banner(
        "AUTHORITATIVE INPUT EXISTENCE"
    )

    for path in (
        TEMPORAL_PATH,
        NODE_INDEX_PATH,
        EVALUATION_CONTRACT_PATH,
        AUDIT_5_1_2A_SUMMARY_PATH,
    ):

        require(
            path.exists(),
            f"Missing authoritative input: {path}",
        )

        print(
            f"FOUND  {path}"
        )

    # =========================================================================
    # Contract recheck
    # =========================================================================

    banner(
        "PHASE 5.1.2b FROZEN CONTRACT RECHECK"
    )

    contract = json.loads(
        EVALUATION_CONTRACT_PATH.read_text(
            encoding="utf-8"
        )
    )

    require(
        contract[
            "phase"
        ]
        == "5.1.2b",
        "Unexpected evaluation-contract phase",
    )

    require(
        contract[
            "status"
        ]
        == "FROZEN",
        "Evaluation contract is not frozen",
    )

    require(
        int(
            contract[
                "paper_specified"
            ][
                "random_negatives_per_case"
            ]
        )
        == NEGATIVES_PER_CASE,
        "Frozen evaluation K drift",
    )

    require(
        contract[
            "candidate_universe"
        ][
            "name"
        ]
        == "global_role_universe",
        "Candidate-universe contract drift",
    )

    require(
        int(
            contract[
                "candidate_universe"
            ][
                "count"
            ]
        )
        == EXPECTED[
            "startup_nodes"
        ],
        "Startup candidate count drift",
    )

    require(
        contract[
            "negative_eligibility"
        ][
            "exclude_pre_T60_positive_pairs"
        ]
        is True,
        "Historical-positive exclusion drift",
    )

    require(
        contract[
            "negative_eligibility"
        ][
            "exclude_focal_positive"
        ]
        is True,
        "Focal-positive exclusion drift",
    )

    require(
        contract[
            "negative_eligibility"
        ][
            "exclude_other_T60_positives"
        ]
        is False,
        "Other-T60-positive exclusion drift",
    )

    require(
        contract[
            "sampling"
        ][
            "without_replacement"
        ]
        is True,
        "Replacement semantics drift",
    )

    require(
        contract[
            "sampling"
        ][
            "generate_once"
        ]
        is True,
        "Candidate persistence drift",
    )

    require(
        int(
            contract[
                "rng_runtime"
            ][
                "base_seed"
            ]
        )
        == EVALUATION_BASE_SEED,
        "Evaluation base-seed drift",
    )

    require(
        contract[
            "rng_runtime"
        ][
            "namespace"
        ]
        == SEED_NAMESPACE,
        "Evaluation RNG namespace drift",
    )

    require(
        contract[
            "rng_runtime"
        ][
            "bit_generator"
        ]
        == "numpy.random.PCG64",
        "Evaluation BitGenerator drift",
    )

    require(
        contract[
            "collision_contract"
        ][
            "full_T60_labels_may_modify_candidate_lists"
        ]
        is False,
        "Post-generation collision mutation unexpectedly allowed",
    )

    require(
        contract[
            "collision_contract"
        ][
            "resample_detected_T60_positive_collision"
        ]
        is False,
        "Collision-resampling guard drift",
    )

    print(
        "Evaluation contract:                  FROZEN  PASS"
    )

    print(
        "Candidates per case:                  1 + 99 PASS"
    )

    print(
        "Candidate universe:                   global role universe PASS"
    )

    print(
        "Historical positive exclusion:        YES  PASS"
    )

    print(
        "Other T60 labels used in generation:  NO   PASS"
    )

    print(
        "Sampling:                             uniform / without replacement"
    )

    print(
        "Persistence:                          generate once / reuse"
    )

    print(
        "Evaluation RNG:                       PCG64 / per-case deterministic"
    )

    # =========================================================================
    # Load frozen data
    # =========================================================================

    banner(
        "FROZEN PHASE-2 / PHASE-3 INPUT INTEGRITY"
    )

    temporal = pd.read_parquet(
        TEMPORAL_PATH,
        columns=[
            "interaction_id",
            "funding_round_id",
            "investor_id",
            "startup_id",
            "segment_number",
            "experiment_split",
        ],
    )

    nodes = pd.read_parquet(
        NODE_INDEX_PATH,
        columns=[
            "node_index",
            "node_type",
            "raw_entity_id",
        ],
    )

    require(
        len(temporal)
        == EXPECTED[
            "temporal_rows"
        ],
        "Temporal row-count drift",
    )

    require(
        len(nodes)
        == EXPECTED[
            "node_rows"
        ],
        "Node row-count drift",
    )

    require(
        temporal[
            "interaction_id"
        ].notna().all(),
        "Null interaction_id found",
    )

    require(
        not temporal[
            "interaction_id"
        ].duplicated().any(),
        "Duplicate interaction_id found",
    )

    require(
        temporal[
            "investor_id"
        ].notna().all(),
        "Null investor_id found",
    )

    require(
        temporal[
            "startup_id"
        ].notna().all(),
        "Null startup_id found",
    )

    # Normalize UUID-like identifiers consistently.
    temporal[
        "interaction_id"
    ] = temporal[
        "interaction_id"
    ].astype(str)

    temporal[
        "funding_round_id"
    ] = temporal[
        "funding_round_id"
    ].astype(str)

    temporal[
        "investor_id"
    ] = temporal[
        "investor_id"
    ].astype(str)

    temporal[
        "startup_id"
    ] = temporal[
        "startup_id"
    ].astype(str)

    temporal[
        "segment_number"
    ] = temporal[
        "segment_number"
    ].astype(
        np.int16
    )

    temporal[
        "experiment_split"
    ] = (
        temporal[
            "experiment_split"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    nodes[
        "raw_entity_id"
    ] = nodes[
        "raw_entity_id"
    ].astype(str)

    pre_t60 = temporal.loc[
        temporal[
            "segment_number"
        ]
        <= 59
    ].copy()

    t60 = temporal.loc[
        temporal[
            "segment_number"
        ]
        == 60
    ].copy()

    require(
        len(pre_t60)
        == EXPECTED[
            "t0_t59_rows"
        ],
        "T0-T59 count drift",
    )

    require(
        len(t60)
        == EXPECTED[
            "t60_rows"
        ],
        "T60 count drift",
    )

    validation_count = int(
        (
            t60[
                "experiment_split"
            ]
            == "validation"
        ).sum()
    )

    test_count = int(
        (
            t60[
                "experiment_split"
            ]
            == "test"
        ).sum()
    )

    require(
        validation_count
        == EXPECTED[
            "validation_rows"
        ],
        "Validation count drift",
    )

    require(
        test_count
        == EXPECTED[
            "test_rows"
        ],
        "Test count drift",
    )

    print(
        f"Temporal rows:                     "
        f"{len(temporal):,}  PASS"
    )

    print(
        f"T0-T59 rows:                      "
        f"{len(pre_t60):,}  PASS"
    )

    print(
        f"T60 events:                       "
        f"{len(t60):,}  PASS"
    )

    print(
        f"Validation events:                "
        f"{validation_count:,}  PASS"
    )

    print(
        f"Test events:                      "
        f"{test_count:,}  PASS"
    )

    # =========================================================================
    # Node mappings
    # =========================================================================

    banner(
        "FROZEN NODE-INDEX MAPPINGS"
    )

    investor_nodes = (
        nodes.loc[
            nodes[
                "node_type"
            ]
            == "investor",
            [
                "node_index",
                "raw_entity_id",
            ],
        ]
        .copy()
    )

    startup_nodes = (
        nodes.loc[
            nodes[
                "node_type"
            ]
            == "startup",
            [
                "node_index",
                "raw_entity_id",
            ],
        ]
        .sort_values(
            "node_index",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    require(
        len(investor_nodes)
        == EXPECTED[
            "investor_nodes"
        ],
        "Investor node count drift",
    )

    require(
        len(startup_nodes)
        == EXPECTED[
            "startup_nodes"
        ],
        "Startup node count drift",
    )

    require(
        not investor_nodes[
            "raw_entity_id"
        ].duplicated().any(),
        "Duplicate Investor raw_entity_id",
    )

    require(
        not startup_nodes[
            "raw_entity_id"
        ].duplicated().any(),
        "Duplicate Startup raw_entity_id",
    )

    investor_id_to_node = dict(
        zip(
            investor_nodes[
                "raw_entity_id"
            ],
            investor_nodes[
                "node_index"
            ].astype(
                np.int64
            ),
        )
    )

    startup_id_to_local = {
        raw_id: local_index
        for local_index, raw_id
        in enumerate(
            startup_nodes[
                "raw_entity_id"
            ].tolist()
        )
    }

    startup_local_to_global = (
        startup_nodes[
            "node_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    startup_id_by_local = (
        startup_nodes[
            "raw_entity_id"
        ]
        .to_numpy(
            dtype=object
        )
    )

    require(
        t60[
            "investor_id"
        ].isin(
            investor_id_to_node
        ).all(),
        "T60 investor missing from frozen node index",
    )

    require(
        t60[
            "startup_id"
        ].isin(
            startup_id_to_local
        ).all(),
        "T60 startup missing from frozen node index",
    )

    print(
        f"Investor nodes:                   "
        f"{len(investor_nodes):,}  PASS"
    )

    print(
        f"Startup nodes:                    "
        f"{len(startup_nodes):,}  PASS"
    )

    # =========================================================================
    # Build PRE-T60 exclusion sets ONLY
    # =========================================================================

    banner(
        "BUILD PRE-T60 EVALUATION EXCLUSION SETS"
    )

    # IMPORTANT:
    # These are the ONLY non-focal labels permitted to affect
    # candidate generation.
    #
    # No T60 diagnostic sets are constructed until AFTER the
    # candidate matrix has already been generated.

    pre_t60 = pre_t60[
        [
            "investor_id",
            "startup_id",
        ]
    ].drop_duplicates()

    pre_t60[
        "startup_local_index"
    ] = pre_t60[
        "startup_id"
    ].map(
        startup_id_to_local
    )

    require(
        pre_t60[
            "startup_local_index"
        ].notna().all(),
        (
            "Pre-T60 Startup missing from "
            "frozen Startup universe"
        ),
    )

    pre_t60[
        "startup_local_index"
    ] = pre_t60[
        "startup_local_index"
    ].astype(
        np.int64
    )

    historical_local_by_investor = (
        get_grouped_set_mapping(
            pre_t60,
            "investor_id",
            "startup_local_index",
        )
    )

    history_sizes = np.array(
        [
            len(
                values
            )
            for values
            in historical_local_by_investor.values()
        ],
        dtype=np.int64,
    )

    print(
        f"Investors with pre-T60 positives:  "
        f"{len(historical_local_by_investor):,}"
    )

    print(
        f"Maximum historical Startup set:    "
        f"{int(history_sizes.max()):,}"
    )

    # =========================================================================
    # Canonical event serialization order
    # =========================================================================

    banner(
        "CANONICAL T60 CASE ORDER"
    )

    split_order = {
        "validation": 0,
        "test": 1,
    }

    require(
        t60[
            "experiment_split"
        ].isin(
            split_order
        ).all(),
        "Unexpected T60 experiment_split",
    )

    cases = t60.copy()

    cases[
        "_split_order"
    ] = cases[
        "experiment_split"
    ].map(
        split_order
    )

    # IMPORTANT:
    # The random candidates do NOT depend on this row order because
    # each case has its own interaction-specific seed.
    #
    # The sort only gives the matrix a stable serialization order.
    cases = (
        cases.sort_values(
            [
                "_split_order",
                "interaction_id",
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    cases[
        "case_index"
    ] = np.arange(
        len(cases),
        dtype=np.int64,
    )

    cases[
        "investor_node_index"
    ] = (
        cases[
            "investor_id"
        ]
        .map(
            investor_id_to_node
        )
        .astype(
            np.int64
        )
    )

    cases[
        "positive_startup_local_index"
    ] = (
        cases[
            "startup_id"
        ]
        .map(
            startup_id_to_local
        )
        .astype(
            np.int64
        )
    )

    cases[
        "positive_startup_node_index"
    ] = startup_local_to_global[
        cases[
            "positive_startup_local_index"
        ].to_numpy(
            dtype=np.int64
        )
    ]

    cases[
        "evaluation_seed"
    ] = [
        derive_case_seed(
            split,
            interaction_id,
        )
        for split, interaction_id
        in zip(
            cases[
                "experiment_split"
            ],
            cases[
                "interaction_id"
            ],
        )
    ]

    require(
        cases[
            "evaluation_seed"
        ].nunique()
        == len(cases),
        (
            "Per-case evaluation seed collision detected. "
            "This should be extraordinarily unlikely; "
            "do not continue without inspection."
        ),
    )

    print(
        f"Evaluation cases:                  "
        f"{len(cases):,}"
    )

    print(
        "Serialization order:                "
        "validation -> test, then interaction_id"
    )

    print(
        "Sampling depends on row order:       NO"
    )

    print(
        "Unique per-case seeds:               PASS"
    )

    # =========================================================================
    # Generate immutable evaluation negatives
    # =========================================================================

    banner(
        "GENERATE 99 NEGATIVES PER T60 EVENT"
    )

    N_STARTUPS = len(
        startup_nodes
    )

    negative_local_matrix = np.empty(
        (
            len(cases),
            NEGATIVES_PER_CASE,
        ),
        dtype=np.int64,
    )

    negative_pool_sizes = np.empty(
        len(cases),
        dtype=np.int64,
    )

    for row in cases.itertuples(
        index=False
    ):

        case_index = int(
            row.case_index
        )

        investor_id = (
            row.investor_id
        )

        focal_local = int(
            row.positive_startup_local_index
        )

        historical = get_set(
            historical_local_by_investor,
            investor_id,
        )

        focal_already_historical = (
            focal_local
            in historical
        )

        negative_pool_size = (
            N_STARTUPS
            - len(
                historical
            )
            - (
                0
                if focal_already_historical
                else 1
            )
        )

        require(
            negative_pool_size
            >= NEGATIVES_PER_CASE,
            (
                f"Case {case_index} has only "
                f"{negative_pool_size} eligible negatives"
            ),
        )

        negative_pool_sizes[
            case_index
        ] = negative_pool_size

        negative_local_matrix[
            case_index,
            :,
        ] = (
            generate_uniform_negatives_by_rejection(
                universe_size=N_STARTUPS,
                excluded_indices=historical,
                focal_index=focal_local,
                k=NEGATIVES_PER_CASE,
                seed=int(
                    row.evaluation_seed
                ),
            )
        )

        if (
            case_index + 1
        ) % 2500 == 0:

            print(
                f"Generated "
                f"{case_index + 1:>6,} / "
                f"{len(cases):,} cases"
            )

    print(
        f"Generated {len(cases):,} / "
        f"{len(cases):,} cases"
    )

    require(
        negative_local_matrix.shape
        == (
            EXPECTED[
                "t60_rows"
            ],
            NEGATIVES_PER_CASE,
        ),
        "Negative matrix shape mismatch",
    )

    # Convert Startup-local positions to frozen Phase-3 global node indices.
    negative_global_matrix = (
        startup_local_to_global[
            negative_local_matrix
        ]
    )

    require(
        negative_global_matrix.dtype
        == np.int64,
        "Negative node-index matrix dtype drift",
    )

    print(
        f"Negative matrix shape:             "
        f"{negative_global_matrix.shape}"
    )

    print(
        f"Negative slots generated:          "
        f"{negative_global_matrix.size:,}"
    )

    require(
        negative_global_matrix.size
        == EXPECTED_TOTAL_NEGATIVE_SLOTS,
        "Total negative-slot count mismatch",
    )

    # =========================================================================
    # Immediate integrity checks — NO T60 diagnostics yet
    # =========================================================================

    banner(
        "GENERATION INTEGRITY AUDIT"
    )

    unique_count_failures = 0
    focal_collision_failures = 0
    historical_collision_failures = 0
    startup_range_failures = 0

    startup_global_set = set(
        startup_local_to_global.tolist()
    )

    for row in cases.itertuples(
        index=False
    ):

        i = int(
            row.case_index
        )

        local_negatives = (
            negative_local_matrix[
                i
            ]
        )

        global_negatives = (
            negative_global_matrix[
                i
            ]
        )

        if (
            np.unique(
                local_negatives
            ).size
            != NEGATIVES_PER_CASE
        ):
            unique_count_failures += 1

        focal_local = int(
            row.positive_startup_local_index
        )

        if np.any(
            local_negatives
            == focal_local
        ):
            focal_collision_failures += 1

        historical = get_set(
            historical_local_by_investor,
            row.investor_id,
        )

        if any(
            int(candidate)
            in historical
            for candidate
            in local_negatives
        ):
            historical_collision_failures += 1

        if any(
            int(candidate)
            not in startup_global_set
            for candidate
            in global_negatives
        ):
            startup_range_failures += 1

    require(
        unique_count_failures == 0,
        (
            "At least one case does not contain "
            "99 distinct negatives"
        ),
    )

    require(
        focal_collision_failures == 0,
        "Focal positive appears in a negative list",
    )

    require(
        historical_collision_failures == 0,
        (
            "Historical Investor-Startup positive "
            "appears in a negative list"
        ),
    )

    require(
        startup_range_failures == 0,
        "Non-Startup node appears as evaluation negative",
    )

    require(
        int(
            negative_pool_sizes.min()
        )
        == 305_410,
        (
            "Minimum realized evaluation pool differs "
            "from Phase-5.1.2a/5.1.2b evidence"
        ),
    )

    print(
        "99 unique negatives per case:        PASS"
    )

    print(
        "Focal positives excluded:            PASS"
    )

    print(
        "Pre-T60 positives excluded:          PASS"
    )

    print(
        "Every negative is a Startup node:    PASS"
    )

    print(
        f"Minimum realized pool:              "
        f"{int(negative_pool_sizes.min()):,}  PASS"
    )

    print(
        f"Mean realized pool:                 "
        f"{float(negative_pool_sizes.mean()):,.6f}"
    )

    # =========================================================================
    # Exact deterministic regeneration audit
    # =========================================================================

    banner(
        "EXACT DETERMINISTIC REGENERATION AUDIT"
    )

    regenerated_local_matrix = np.empty_like(
        negative_local_matrix
    )

    for row in cases.itertuples(
        index=False
    ):

        i = int(
            row.case_index
        )

        regenerated_local_matrix[
            i,
            :,
        ] = (
            generate_uniform_negatives_by_rejection(
                universe_size=N_STARTUPS,
                excluded_indices=get_set(
                    historical_local_by_investor,
                    row.investor_id,
                ),
                focal_index=int(
                    row.positive_startup_local_index
                ),
                k=NEGATIVES_PER_CASE,
                seed=int(
                    row.evaluation_seed
                ),
            )
        )

    deterministic_regeneration_pass = (
        np.array_equal(
            negative_local_matrix,
            regenerated_local_matrix,
        )
    )

    require(
        deterministic_regeneration_pass,
        (
            "Full deterministic regeneration "
            "did not reproduce the same candidate matrix"
        ),
    )

    del regenerated_local_matrix

    print(
        "Full 22,515-case regeneration:       PASS"
    )

    print(
        "Candidate matrix exactly identical:  PASS"
    )

    # =========================================================================
    # Freeze matrix logically before consulting T60 labels
    # =========================================================================

    banner(
        "LOCK GENERATED MATRIX BEFORE T60 COLLISION DIAGNOSTICS"
    )

    matrix_logical_sha256 = hashlib.sha256(
        np.asarray(
            negative_global_matrix,
            dtype="<i8",
        ).tobytes(
            order="C"
        )
    ).hexdigest()

    print(
        "Generated candidate matrix logical SHA256:"
    )

    print(
        matrix_logical_sha256
    )

    print()

    print(
        "From this point onward, T60 labels may be used "
        "for DIAGNOSTICS ONLY."
    )

    print(
        "Candidate mutation/resampling after this point: FORBIDDEN"
    )

    # =========================================================================
    # Build T60 diagnostic sets ONLY AFTER generation
    # =========================================================================

    banner(
        "POST-GENERATION REALIZED T60-POSITIVE COLLISION AUDIT"
    )

    # Convert generated negatives back to raw Startup IDs for
    # comparison with T60 positive outcomes.
    #
    # This cannot affect generation because generation has already
    # completed and the candidate-matrix hash above has been fixed.

    t60_unique_pairs = (
        t60[
            [
                "investor_id",
                "startup_id",
            ]
        ]
        .drop_duplicates()
    )

    t60_by_investor = (
        get_grouped_set_mapping(
            t60_unique_pairs,
            "investor_id",
            "startup_id",
        )
    )

    validation_pairs = (
        t60.loc[
            t60[
                "experiment_split"
            ]
            == "validation",
            [
                "investor_id",
                "startup_id",
            ],
        ]
        .drop_duplicates()
    )

    test_pairs = (
        t60.loc[
            t60[
                "experiment_split"
            ]
            == "test",
            [
                "investor_id",
                "startup_id",
            ],
        ]
        .drop_duplicates()
    )

    validation_by_investor = (
        get_grouped_set_mapping(
            validation_pairs,
            "investor_id",
            "startup_id",
        )
    )

    test_by_investor = (
        get_grouped_set_mapping(
            test_pairs,
            "investor_id",
            "startup_id",
        )
    )

    collision_rows = []

    case_has_any_t60_collision = np.zeros(
        len(cases),
        dtype=bool,
    )

    total_collision_slots = 0
    total_same_split_collision_slots = 0
    total_opposite_split_collision_slots = 0

    for row in cases.itertuples(
        index=False
    ):

        i = int(
            row.case_index
        )

        split = (
            row.experiment_split
        )

        focal_startup = (
            row.startup_id
        )

        all_t60_positive_startups = set(
            get_set(
                t60_by_investor,
                row.investor_id,
            )
        )

        # The focal positive itself can never occur because generation
        # explicitly excluded it. Remove it anyway so the diagnostic
        # semantics are explicit.
        all_t60_positive_startups.discard(
            focal_startup
        )

        if split == "validation":

            same_split_positive_startups = set(
                get_set(
                    validation_by_investor,
                    row.investor_id,
                )
            )

            opposite_split_positive_startups = set(
                get_set(
                    test_by_investor,
                    row.investor_id,
                )
            )

        else:

            same_split_positive_startups = set(
                get_set(
                    test_by_investor,
                    row.investor_id,
                )
            )

            opposite_split_positive_startups = set(
                get_set(
                    validation_by_investor,
                    row.investor_id,
                )
            )

        same_split_positive_startups.discard(
            focal_startup
        )

        opposite_split_positive_startups.discard(
            focal_startup
        )

        for negative_rank in range(
            NEGATIVES_PER_CASE
        ):

            local_index = int(
                negative_local_matrix[
                    i,
                    negative_rank,
                ]
            )

            startup_id = str(
                startup_id_by_local[
                    local_index
                ]
            )

            is_t60_positive = (
                startup_id
                in all_t60_positive_startups
            )

            if not is_t60_positive:
                continue

            is_same_split_positive = (
                startup_id
                in same_split_positive_startups
            )

            is_opposite_split_positive = (
                startup_id
                in opposite_split_positive_startups
            )

            case_has_any_t60_collision[
                i
            ] = True

            total_collision_slots += 1

            if is_same_split_positive:
                total_same_split_collision_slots += 1

            if is_opposite_split_positive:
                total_opposite_split_collision_slots += 1

            collision_rows.append(
                {
                    "case_index": i,
                    "interaction_id": (
                        row.interaction_id
                    ),
                    "experiment_split": (
                        split
                    ),
                    "investor_id": (
                        row.investor_id
                    ),
                    "positive_startup_id": (
                        focal_startup
                    ),
                    "negative_rank_0_based": (
                        negative_rank
                    ),
                    "negative_startup_id": (
                        startup_id
                    ),
                    "negative_startup_node_index": int(
                        negative_global_matrix[
                            i,
                            negative_rank,
                        ]
                    ),
                    "is_other_T60_positive": True,
                    "is_same_split_T60_positive": (
                        is_same_split_positive
                    ),
                    "is_opposite_split_T60_positive": (
                        is_opposite_split_positive
                    ),
                    "action": (
                        "DIAGNOSTIC_ONLY_KEEP_CANDIDATE"
                    ),
                }
            )

    collision_detail_df = pd.DataFrame(
        collision_rows
    )

    cases_with_collision = int(
        case_has_any_t60_collision.sum()
    )

    realized_case_collision_share = (
        cases_with_collision
        / len(cases)
    )

    realized_slot_collision_share = (
        total_collision_slots
        / EXPECTED_TOTAL_NEGATIVE_SLOTS
    )

    print(
        f"Realized T60-positive collision slots: "
        f"{total_collision_slots:,}"
    )

    print(
        f"Cases with >=1 T60-positive collision: "
        f"{cases_with_collision:,}"
    )

    print(
        f"Case collision share:                  "
        f"{realized_case_collision_share:.6%}"
    )

    print(
        f"Negative-slot collision share:         "
        f"{realized_slot_collision_share:.8%}"
    )

    print(
        f"Same-split collision memberships:      "
        f"{total_same_split_collision_slots:,}"
    )

    print(
        f"Opposite-split collision memberships:  "
        f"{total_opposite_split_collision_slots:,}"
    )

    print(
        "Collision candidates resampled:          0"
    )

    # =========================================================================
    # Compare realized collision count to analytical expectation
    # =========================================================================

    banner(
        "ANALYTICAL EXPECTATION VS REALIZED COLLISIONS"
    )

    audit_summary = pd.read_csv(
        AUDIT_5_1_2A_SUMMARY_PATH
    )

    analytical = audit_summary.loc[
        (
            audit_summary[
                "scope"
            ]
            == "t60_overall"
        )
        & (
            audit_summary[
                "candidate_universe"
            ]
            == "global_role_universe"
        )
        & (
            audit_summary[
                "exclusion_policy"
            ]
            == "exclude_prior_pairs_plus_focal"
        )
    ].copy()

    require(
        len(analytical) == 1,
        "Missing Phase-5.1.2a analytical reference row",
    )

    analytical = analytical.iloc[0]

    expected_collided_slots = (
        len(cases)
        * float(
            analytical[
                "mean_expected_other_t60_positive_collisions_in_99"
            ]
        )
    )

    expected_affected_cases = (
        len(cases)
        * float(
            analytical[
                "mean_p_99_contains_at_least_one_other_t60_positive"
            ]
        )
    )

    print(
        f"Analytical expected collided slots: "
        f"{expected_collided_slots:.3f}"
    )

    print(
        f"Realized collided slots:            "
        f"{total_collision_slots:,}"
    )

    print()

    print(
        f"Analytical expected affected cases: "
        f"{expected_affected_cases:.3f}"
    )

    print(
        f"Realized affected cases:            "
        f"{cases_with_collision:,}"
    )

    print()

    print(
        "NOTE: No equality is expected. The analytical values are "
        "expectations over random candidate generation."
    )

    # =========================================================================
    # Split-specific summaries
    # =========================================================================

    banner(
        "REALIZED COLLISION SUMMARY BY SPLIT"
    )

    collision_summary_rows = []

    for split in (
        "validation",
        "test",
        "t60_overall",
    ):

        if split == "t60_overall":

            mask = np.ones(
                len(cases),
                dtype=bool,
            )

        else:

            mask = (
                cases[
                    "experiment_split"
                ].to_numpy()
                == split
            )

        n_cases = int(
            mask.sum()
        )

        affected = int(
            case_has_any_t60_collision[
                mask
            ].sum()
        )

        if (
            collision_detail_df.empty
        ):

            slot_count = 0
            same_split_slot_count = 0
            opposite_split_slot_count = 0

        elif split == "t60_overall":

            slot_count = len(
                collision_detail_df
            )

            same_split_slot_count = int(
                collision_detail_df[
                    "is_same_split_T60_positive"
                ].sum()
            )

            opposite_split_slot_count = int(
                collision_detail_df[
                    "is_opposite_split_T60_positive"
                ].sum()
            )

        else:

            detail_part = (
                collision_detail_df.loc[
                    collision_detail_df[
                        "experiment_split"
                    ]
                    == split
                ]
            )

            slot_count = len(
                detail_part
            )

            same_split_slot_count = int(
                detail_part[
                    "is_same_split_T60_positive"
                ].sum()
            )

            opposite_split_slot_count = int(
                detail_part[
                    "is_opposite_split_T60_positive"
                ].sum()
            )

        total_slots = (
            n_cases
            * NEGATIVES_PER_CASE
        )

        collision_summary_rows.append(
            {
                "scope": split,
                "evaluation_cases": n_cases,
                "negative_slots": total_slots,
                "cases_with_any_other_T60_positive": (
                    affected
                ),
                "case_collision_share": (
                    affected
                    / n_cases
                ),
                "other_T60_positive_collision_slots": (
                    slot_count
                ),
                "slot_collision_share": (
                    slot_count
                    / total_slots
                ),
                "same_split_T60_positive_memberships": (
                    same_split_slot_count
                ),
                "opposite_split_T60_positive_memberships": (
                    opposite_split_slot_count
                ),
                "resampled_collision_slots": 0,
            }
        )

    collision_summary_df = pd.DataFrame(
        collision_summary_rows
    )

    print(
        collision_summary_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Build final case manifest
    # =========================================================================

    banner(
        "BUILD IMMUTABLE EVALUATION CASE MANIFEST"
    )

    cases[
        "negative_pool_size"
    ] = negative_pool_sizes

    cases[
        "negative_list_sha256"
    ] = [
        sha256_int64_vector(
            negative_global_matrix[
                i
            ]
        )
        for i
        in range(
            len(cases)
        )
    ]

    cases[
        "has_realized_other_T60_positive_collision"
    ] = (
        case_has_any_t60_collision
    )

    case_manifest = cases[
        [
            "case_index",
            "interaction_id",
            "funding_round_id",
            "experiment_split",
            "investor_id",
            "investor_node_index",
            "startup_id",
            "positive_startup_node_index",
            "evaluation_seed",
            "negative_pool_size",
            "negative_list_sha256",
            "has_realized_other_T60_positive_collision",
        ]
    ].rename(
        columns={
            "startup_id": (
                "positive_startup_id"
            )
        }
    )

    require(
        len(case_manifest)
        == EXPECTED[
            "t60_rows"
        ],
        "Case-manifest row count mismatch",
    )

    require(
        not case_manifest[
            "interaction_id"
        ].duplicated().any(),
        "Duplicate interaction_id in case manifest",
    )

    require(
        case_manifest[
            "negative_list_sha256"
        ].notna().all(),
        "Missing candidate-list hash",
    )

    print(
        f"Case manifest rows:                 "
        f"{len(case_manifest):,}  PASS"
    )

    print(
        "One row per interaction_id:          PASS"
    )

    # =========================================================================
    # Generation audit register
    # =========================================================================

    generation_checks = [
        (
            "evaluation_contract_frozen",
            contract[
                "status"
            ]
            == "FROZEN",
        ),
        (
            "t60_case_count_22515",
            len(cases)
            == EXPECTED[
                "t60_rows"
            ],
        ),
        (
            "validation_count_2251",
            int(
                (
                    cases[
                        "experiment_split"
                    ]
                    == "validation"
                ).sum()
            )
            == EXPECTED[
                "validation_rows"
            ],
        ),
        (
            "test_count_20264",
            int(
                (
                    cases[
                        "experiment_split"
                    ]
                    == "test"
                ).sum()
            )
            == EXPECTED[
                "test_rows"
            ],
        ),
        (
            "negative_matrix_shape",
            negative_global_matrix.shape
            == (
                EXPECTED[
                    "t60_rows"
                ],
                NEGATIVES_PER_CASE,
            ),
        ),
        (
            "negative_slot_count_2229985",
            negative_global_matrix.size
            == EXPECTED_TOTAL_NEGATIVE_SLOTS,
        ),
        (
            "all_cases_have_99_unique_negatives",
            unique_count_failures
            == 0,
        ),
        (
            "no_focal_positive_in_negative_lists",
            focal_collision_failures
            == 0,
        ),
        (
            "no_pre_t60_positive_in_negative_lists",
            historical_collision_failures
            == 0,
        ),
        (
            "all_negative_nodes_are_startups",
            startup_range_failures
            == 0,
        ),
        (
            "minimum_pool_305410",
            int(
                negative_pool_sizes.min()
            )
            == 305_410,
        ),
        (
            "full_deterministic_regeneration",
            deterministic_regeneration_pass,
        ),
        (
            "no_T60_collision_resampling",
            True,
        ),
        (
            "no_model_instantiated",
            True,
        ),
        (
            "no_optimizer_created",
            True,
        ),
        (
            "no_training_performed",
            True,
        ),
        (
            "no_HR_NDCG_evaluation_performed",
            True,
        ),
    ]

    generation_audit_df = pd.DataFrame(
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
            in generation_checks
        ]
    )

    require(
        (
            generation_audit_df[
                "result"
            ]
            == "PASS"
        ).all(),
        "At least one evaluation-generation integrity check failed",
    )

    # =========================================================================
    # Write model-ready artifacts
    # =========================================================================

    banner(
        "WRITE IMMUTABLE MODEL-READY EVALUATION ARTIFACTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        NEGATIVE_MATRIX_PATH,
        negative_global_matrix,
        allow_pickle=False,
    )

    case_manifest.to_parquet(
        CASE_MANIFEST_PATH,
        index=False,
    )

    collision_summary_df.to_csv(
        COLLISION_SUMMARY_PATH,
        index=False,
    )

    if collision_detail_df.empty:

        # Preserve a predictable schema even if no realized
        # T60 collision happens.
        collision_detail_df = pd.DataFrame(
            columns=[
                "case_index",
                "interaction_id",
                "experiment_split",
                "investor_id",
                "positive_startup_id",
                "negative_rank_0_based",
                "negative_startup_id",
                "negative_startup_node_index",
                "is_other_T60_positive",
                "is_same_split_T60_positive",
                "is_opposite_split_T60_positive",
                "action",
            ]
        )

    collision_detail_df.to_parquet(
        COLLISION_DETAIL_PATH,
        index=False,
    )

    generation_audit_df.to_csv(
        GENERATION_AUDIT_PATH,
        index=False,
    )

    # =========================================================================
    # Post-write reload check
    # =========================================================================

    banner(
        "POST-WRITE RELOAD / BYTE-LEVEL AUDIT"
    )

    reloaded_negatives = np.load(
        NEGATIVE_MATRIX_PATH,
        allow_pickle=False,
    )

    require(
        np.array_equal(
            reloaded_negatives,
            negative_global_matrix,
        ),
        "Reloaded .npy candidate matrix differs from generated matrix",
    )

    reloaded_case_manifest = pd.read_parquet(
        CASE_MANIFEST_PATH
    )

    require(
        len(
            reloaded_case_manifest
        )
        == len(
            case_manifest
        ),
        "Reloaded case-manifest row-count mismatch",
    )

    require(
        reloaded_case_manifest[
            "interaction_id"
        ].tolist()
        == case_manifest[
            "interaction_id"
        ].tolist(),
        "Reloaded case-manifest ordering mismatch",
    )

    print(
        "Reloaded candidate matrix identical: PASS"
    )

    print(
        "Reloaded case manifest identical order: PASS"
    )

    # =========================================================================
    # Artifact hashes
    # =========================================================================

    banner(
        "AUTHORITATIVE ARTIFACT HASHES"
    )

    artifact_paths = [
        NEGATIVE_MATRIX_PATH,
        CASE_MANIFEST_PATH,
        COLLISION_DETAIL_PATH,
        COLLISION_SUMMARY_PATH,
        GENERATION_AUDIT_PATH,
    ]

    hash_rows = []

    for path in artifact_paths:

        digest = sha256_file(
            path
        )

        hash_rows.append(
            {
                "path": str(
                    path
                ),
                "sha256": digest,
            }
        )

        print(
            f"{digest}  {path}"
        )

    artifact_hash_df = pd.DataFrame(
        hash_rows
    )

    artifact_hash_df.to_csv(
        ARTIFACT_HASH_PATH,
        index=False,
    )

    # =========================================================================
    # Generation manifest
    # =========================================================================

    manifest = {
        "phase": "5.1.2c",
        "title": (
            "Deterministic T60 Evaluation Candidate Generation "
            "and Realized-Collision Audit"
        ),
        "status": (
            "GENERATED_AND_AUDITED"
        ),

        "evaluation_negative_samples_generated": True,
        "evaluation_rng_instantiated": True,

        "training_negative_samples_generated": False,
        "model_instantiated": False,
        "optimizer_created": False,
        "training_performed": False,
        "ranking_evaluation_performed": False,

        "frozen_source_contract": (
            str(
                EVALUATION_CONTRACT_PATH
            )
        ),

        "case_contract": {
            "case_count": (
                EXPECTED[
                    "t60_rows"
                ]
            ),
            "validation_cases": (
                EXPECTED[
                    "validation_rows"
                ]
            ),
            "test_cases": (
                EXPECTED[
                    "test_rows"
                ]
            ),
            "case_identity": (
                "interaction_id"
            ),
            "deduplicate_pairs": False,
            "serialization_order": (
                "validation then test; interaction_id ascending "
                "within split"
            ),
        },

        "candidate_contract": {
            "positive_per_case": 1,
            "negative_per_case": (
                NEGATIVES_PER_CASE
            ),
            "total_candidates_per_case": 100,
            "startup_universe_size": (
                EXPECTED[
                    "startup_nodes"
                ]
            ),
            "negative_slots": (
                EXPECTED_TOTAL_NEGATIVE_SLOTS
            ),
            "total_candidate_slots": (
                EXPECTED_TOTAL_CANDIDATE_SLOTS
            ),
            "distribution": "uniform",
            "without_replacement": True,
            "exclude_pre_T60_positive_pairs": True,
            "exclude_focal_positive": True,
            "exclude_other_T60_positive_pairs": False,
        },

        "generator_runtime": {
            "base_seed": (
                EVALUATION_BASE_SEED
            ),
            "namespace": (
                SEED_NAMESPACE
            ),
            "seed_key": (
                "experiment_split + interaction_id"
            ),
            "bit_generator": (
                "numpy.random.PCG64"
            ),
            "sampling_algorithm": (
                "uniform full-universe proposal rejection; "
                "reject pre-T60 positives, focal positive, "
                "and within-case duplicates"
            ),
            "proposal_batch_size": (
                PROPOSAL_BATCH_SIZE
            ),
            "case_order_affects_candidate_list": False,
        },

        "integrity": {
            "minimum_negative_pool": int(
                negative_pool_sizes.min()
            ),
            "mean_negative_pool": float(
                negative_pool_sizes.mean()
            ),
            "all_cases_have_99_unique_negatives": True,
            "focal_positive_collisions": 0,
            "historical_positive_collisions": 0,
            "non_startup_negative_nodes": 0,
            "full_deterministic_regeneration_passed": (
                deterministic_regeneration_pass
            ),
            "logical_candidate_matrix_sha256": (
                matrix_logical_sha256
            ),
        },

        "realized_T60_collision_diagnostic": {
            "other_T60_positive_collision_slots": (
                total_collision_slots
            ),
            "cases_with_any_other_T60_positive_collision": (
                cases_with_collision
            ),
            "case_collision_share": (
                realized_case_collision_share
            ),
            "slot_collision_share": (
                realized_slot_collision_share
            ),
            "analytical_expected_collision_slots": (
                expected_collided_slots
            ),
            "analytical_expected_affected_cases": (
                expected_affected_cases
            ),
            "resampled_collision_slots": 0,
            "collision_action": (
                "diagnose_only_keep_candidate"
            ),
        },

        "model_ready_outputs": [
            str(
                NEGATIVE_MATRIX_PATH
            ),
            str(
                CASE_MANIFEST_PATH
            ),
        ],

        "diagnostic_outputs": [
            str(
                COLLISION_DETAIL_PATH
            ),
            str(
                COLLISION_SUMMARY_PATH
            ),
            str(
                GENERATION_AUDIT_PATH
            ),
            str(
                ARTIFACT_HASH_PATH
            ),
        ],

        "still_unresolved_original_phase_5_handoff_decisions": [
            "training epoch count",
            "early stopping",
            "weight decay",
        ],
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # Add manifest itself to hash registry after it exists.
    manifest_hash = sha256_file(
        MANIFEST_PATH
    )

    artifact_hash_df = pd.concat(
        [
            artifact_hash_df,
            pd.DataFrame(
                [
                    {
                        "path": str(
                            MANIFEST_PATH
                        ),
                        "sha256": (
                            manifest_hash
                        ),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    artifact_hash_df.to_csv(
        ARTIFACT_HASH_PATH,
        index=False,
    )

    print(
        f"{manifest_hash}  {MANIFEST_PATH}"
    )

    print(
        f"WROTE  {ARTIFACT_HASH_PATH}"
    )

    # =========================================================================
    # Final status
    # =========================================================================

    banner(
        "PHASE 5.1.2c FINAL STATUS"
    )

    print(
        f"T60 evaluation cases:                "
        f"{len(cases):,}"
    )

    print(
        f"Validation cases:                    "
        f"{validation_count:,}"
    )

    print(
        f"Test cases:                          "
        f"{test_count:,}"
    )

    print(
        f"Negatives per case:                  "
        f"{NEGATIVES_PER_CASE}"
    )

    print(
        f"Total generated evaluation negatives:"
        f" {negative_global_matrix.size:,}"
    )

    print(
        f"Total candidate slots incl. positive:"
        f" {EXPECTED_TOTAL_CANDIDATE_SLOTS:,}"
    )

    print()

    print(
        "All candidate lists deterministic:     PASS"
    )

    print(
        "All candidate lists have 99 unique negatives: PASS"
    )

    print(
        "Historical-positive negative collisions: 0"
    )

    print(
        "Focal-positive negative collisions:      0"
    )

    print(
        f"Realized other-T60-positive slots:       "
        f"{total_collision_slots:,}"
    )

    print(
        f"Realized affected evaluation cases:      "
        f"{cases_with_collision:,}"
    )

    print(
        "Detected T60 collisions resampled:       0"
    )

    print()

    print(
        "Evaluation candidate artifact:           FROZEN / GENERATED"
    )

    print(
        "Model instantiated:                      NO"
    )

    print(
        "Optimizer created:                       NO"
    )

    print(
        "Training performed:                      NO"
    )

    print(
        "HR@10/NDCG@10 computed:                  NO"
    )

    print()

    print(
        "Original Phase-5 handoff decisions still unresolved:"
    )

    print(
        "  1. training epoch count"
    )

    print(
        "  2. early stopping"
    )

    print(
        "  3. weight decay"
    )

    banner(
        "PHASE 5.1.2c COMPLETE / "
        "T60 EVALUATION CANDIDATES GENERATED AND AUDITED"
    )


if __name__ == "__main__":
    main()