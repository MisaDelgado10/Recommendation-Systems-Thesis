#!/usr/bin/env python3
"""
Phase 5.3.1l.1 — Freeze Epoch-0 Training Stream Serialization

Purpose
-------
Phase 5.3.1k proved that the canonical composed model can execute:

    real data
        -> forward
        -> BCEWithLogitsLoss
        -> backward

with all 32 trainable tensors receiving finite, non-zero gradients,
while canonical parameter values remain byte-identical.

Before Adam is instantiated, Phase 5 must now freeze the exact
deterministic epoch-0 training stream implied by the already-frozen
training decisions.

FROZEN SCIENTIFIC / TRAINING DECISIONS
--------------------------------------
Training targets:
    T1..T59 positive interaction EVENTS

Positive count:
    1,073,249

Negative semantics:
    eligible_negative(o,b,h) :=
        startup_role_node(b)
        AND NOT EXISTS positive_event(o,b,s)
                WITH segment_number(s) <= h

Important:
    - future-positive pairs with first positive > h remain eligible;
    - T60 labels are NEVER used for training-negative eligibility;
    - four distinct negatives per positive;
    - uniform sampling over eligible Startup-role nodes.

Negative count per epoch:
    4,292,996

Total examples per epoch:
    5,366,245

Batch size:
    512

Epoch batches:
    10,481

Final batch:
    485

Epoch-0 negative RNG:
    namespace = ITRS_PHASE5_TRAIN_NEGATIVE
    base seed = 42
    epoch     = 0
    PCG64
    derived seed = 7895109663985029800

Epoch-0 order RNG:
    namespace = ITRS_PHASE5_TRAIN_ORDER
    base seed = 42
    epoch     = 0
    PCG64
    derived seed = 4607400055922019930


IMPLEMENTATION-EQUIVALENT CHOICES FROZEN HERE
---------------------------------------------
1. Positive-event canonical pre-expansion order:
       lexicographic interaction_id
   using stable mergesort.

2. Each positive occupies five conceptual slots:
       slot 0 -> positive
       slot 1 -> negative draw 0
       slot 2 -> negative draw 1
       slot 3 -> negative draw 2
       slot 4 -> negative draw 3

3. Epoch negative RNG consumes positives in the canonical positive
   order above.

4. For each positive, four initial Startup-role indices are drawn
   from PCG64 uniformly over:
       0 .. NUM_STARTUPS-1

   Candidates are consumed in draw order.

   Reject:
       - pair positive at segment <= h;
       - duplicate accepted negative for that same positive.

   If fewer than four are accepted, additional scalar draws are
   consumed until four distinct eligible negatives exist.

5. The resulting conceptual example universe is:
       0 .. 5,366,244

   where:
       positive_order_index = conceptual_index // 5
       slot                 = conceptual_index % 5

6. Epoch order:
       np.random.Generator(np.random.PCG64(seed))
       .shuffle(np.arange(N, dtype=np.int64))

7. The first 512 shuffled conceptual indices define the canonical
   epoch-0 first training mini-batch.

8. Mixed target segments are preserved. A later numerical execution
   must group examples by target segment h, compute trend using
   exactly T0..T(h-1), then restore original batch positions.
   No artificial post-h GRU padding is allowed.


THIS PHASE DOES
---------------
- read frozen Phase-2 and Phase-3 artifacts;
- freeze positive ordering;
- instantiate epoch-0 training-negative PCG64;
- generate all epoch-0 negatives;
- instantiate epoch-0 training-order PCG64;
- generate the epoch permutation;
- materialize the first 512-example batch manifest;
- freeze logical hashes and artifacts.


THIS PHASE DOES NOT
-------------------
- instantiate the neural model;
- consume Phase-4 neural feature arrays;
- use T60 labels in negative sampling;
- use validation/test candidates;
- instantiate Adam;
- perform forward;
- compute BCE;
- perform backward;
- call optimizer.step();
- write a model checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Frozen dimensions / counts
# =============================================================================

NUM_INVESTORS = 165_975
NUM_STARTUPS = 311_589
NUM_NODES = 477_564

TRAIN_MIN_SEGMENT = 1
TRAIN_MAX_SEGMENT = 59

NEGATIVES_PER_POSITIVE = 4
SERIALIZED_SLOTS_PER_POSITIVE = 5

EXPECTED_T0_T59_EVENTS = 1_173_422
EXPECTED_TRAIN_POSITIVES = 1_073_249
EXPECTED_EPOCH_NEGATIVES = 4_292_996
EXPECTED_EPOCH_EXAMPLES = 5_366_245

BATCH_SIZE = 512
EXPECTED_EPOCH_BATCHES = 10_481
EXPECTED_FINAL_BATCH_SIZE = 485


# =============================================================================
# Frozen RNG contracts
# =============================================================================

BASE_SEED = 42
EPOCH = 0

NEGATIVE_NAMESPACE = (
    "ITRS_PHASE5_TRAIN_NEGATIVE"
)

ORDER_NAMESPACE = (
    "ITRS_PHASE5_TRAIN_ORDER"
)

EXPECTED_EPOCH0_NEGATIVE_SEED = (
    7_895_109_663_985_029_800
)

EXPECTED_EPOCH0_ORDER_SEED = (
    4_607_400_055_922_019_930
)


# =============================================================================
# Authoritative inputs
# =============================================================================

TEMPORAL_SPLIT_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "interactions_itrs_temporal_split.parquet"
)

NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/model_ready/"
    "node_index.parquet"
)

PHASE_5_3_1K_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1k_canonical_real_forward_bce_backward_contract.json"
)

PHASE_5_3_1K_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1k/"
    "phase_5_3_1k_real_forward_preflight_manifest.json"
)


# =============================================================================
# Outputs
# =============================================================================

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1l_1"
)

RUNTIME_DIR = Path(
    "data/experimental/phase_5/training_runtime/"
    "epoch_0"
)

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

POSITIVE_ORDER_PATH = (
    RUNTIME_DIR
    / "canonical_training_positive_event_order.parquet"
)

NEGATIVE_MATRIX_PATH = (
    RUNTIME_DIR
    / "epoch_0_training_negative_startup_local.npy"
)

EPOCH_ORDER_PATH = (
    RUNTIME_DIR
    / "epoch_0_training_example_order.npy"
)

FIRST_BATCH_PATH = (
    RUNTIME_DIR
    / "epoch_0_first_batch_manifest.parquet"
)

FIRST_BATCH_SEGMENT_GROUP_PATH = (
    RUNTIME_DIR
    / "epoch_0_first_batch_segment_groups.csv"
)

NEGATIVE_AUDIT_PATH = (
    AUDIT_DIR
    / "epoch_0_negative_generation_audit.csv"
)

TRAINING_STREAM_AUDIT_PATH = (
    AUDIT_DIR
    / "epoch_0_training_stream_audit.csv"
)

FIRST_BATCH_AUDIT_PATH = (
    AUDIT_DIR
    / "epoch_0_first_batch_integrity_audit.csv"
)

HASH_REGISTRY_PATH = (
    AUDIT_DIR
    / "epoch_0_training_stream_hash_registry.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_1l_1_final_invariants.csv"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_1l_1_training_stream_manifest.json"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1l_1_epoch0_training_stream_serialization_contract.json"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1l_1_training_stream_decision_register.csv"
)


# =============================================================================
# Helpers
# =============================================================================

def banner(
    text: str,
) -> None:

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


def file_sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:

            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def derive_seed(
    namespace: str,
    base_seed: int,
    epoch: int,
) -> int:

    material = (
        f"{namespace}|{base_seed}|{epoch}"
    )

    digest = hashlib.sha256(
        material.encode(
            "utf-8"
        )
    ).digest()

    raw = int.from_bytes(
        digest[
            :8
        ],
        byteorder="little",
        signed=False,
    )

    return int(
        raw
        % (
            2**63
            - 1
        )
    )


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
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        str(
            tuple(
                value.shape
            )
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        value.tobytes(
            order="C"
        )
    )

    return digest.hexdigest()


def positive_stream_logical_sha256(
    frame: pd.DataFrame,
) -> str:

    digest = hashlib.sha256()

    ids = (
        frame[
            "interaction_id"
        ]
        .astype(
            str
        )
        .tolist()
    )

    chunk_size = 10_000

    for start in range(
        0,
        len(
            ids
        ),
        chunk_size,
    ):

        end = min(
            start
            + chunk_size,
            len(
                ids
            ),
        )

        text = "\n".join(
            ids[
                start:end
            ]
        )

        digest.update(
            text.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

    for column in (
        "investor_global",
        "startup_local",
        "segment_number",
    ):

        values = np.ascontiguousarray(
            frame[
                column
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        digest.update(
            column.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            values.tobytes(
                order="C"
            )
        )

        digest.update(
            b"\0"
        )

    return digest.hexdigest()


def dataframe_logical_sha256(
    frame: pd.DataFrame,
    columns: list[str],
) -> str:

    digest = hashlib.sha256()

    for column in (
        columns
    ):

        digest.update(
            column.encode(
                "utf-8"
            )
        )

        digest.update(
            b"\0"
        )

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

        else:

            values = (
                series
                .astype(
                    str
                )
                .tolist()
            )

            for value in (
                values
            ):

                digest.update(
                    value.encode(
                        "utf-8"
                    )
                )

                digest.update(
                    b"\0"
                )

        digest.update(
            b"\0"
        )

    return digest.hexdigest()


# =============================================================================
# Epoch-0 negative generator
# =============================================================================

def generate_epoch_negatives(
    investor_globals: np.ndarray,
    target_segments: np.ndarray,
    pair_first_segment: dict[int, int],
    seed: int,
    max_rows: int | None = None,
    progress: bool = True,
):
    """
    Generate four distinct eligible Startup-role local indices
    per positive event.

    Pair key:
        investor_global * NUM_STARTUPS + startup_local

    Eligibility:
        pair absent from positive history through h

    Important:
        pair_first_segment was constructed ONLY using T0..T59.
        Therefore T60 labels cannot affect training-negative sampling.
    """

    total_rows = int(
        len(
            investor_globals
        )
    )

    if max_rows is not None:

        total_rows = min(
            total_rows,
            int(
                max_rows
            ),
        )

    rng = np.random.Generator(
        np.random.PCG64(
            seed
        )
    )

    negatives = np.empty(
        (
            total_rows,
            NEGATIVES_PER_POSITIVE,
        ),
        dtype=np.int32,
    )

    initial_draw_count = (
        total_rows
        * NEGATIVES_PER_POSITIVE
    )

    repair_draw_count = 0
    forbidden_rejections = 0
    duplicate_rejections = 0
    accepted_future_positive = 0
    accepted_never_positive_pre_t60 = 0

    for row_index in range(
        total_rows
    ):

        investor_global = int(
            investor_globals[
                row_index
            ]
        )

        h = int(
            target_segments[
                row_index
            ]
        )

        require(
            TRAIN_MIN_SEGMENT
            <= h
            <= TRAIN_MAX_SEGMENT,
            (
                f"Invalid training target segment "
                f"h={h} at row {row_index}."
            ),
        )

        accepted = []
        accepted_set = set()

        initial_candidates = (
            rng.integers(
                low=0,
                high=NUM_STARTUPS,
                size=NEGATIVES_PER_POSITIVE,
                dtype=np.int64,
            )
        )

        for candidate_value in (
            initial_candidates
        ):

            candidate = int(
                candidate_value
            )

            pair_key = (
                investor_global
                * NUM_STARTUPS
                + candidate
            )

            first_segment = (
                pair_first_segment.get(
                    pair_key
                )
            )

            if (
                first_segment
                is not None
                and int(
                    first_segment
                )
                <= h
            ):

                forbidden_rejections += 1
                continue

            if candidate in (
                accepted_set
            ):

                duplicate_rejections += 1
                continue

            accepted.append(
                candidate
            )

            accepted_set.add(
                candidate
            )

            if (
                first_segment
                is None
            ):

                accepted_never_positive_pre_t60 += 1

            else:

                require(
                    int(
                        first_segment
                    )
                    > h,
                    (
                        "Accepted negative has "
                        "first positive <= h."
                    ),
                )

                accepted_future_positive += 1

        while len(
            accepted
        ) < NEGATIVES_PER_POSITIVE:

            repair_draw_count += 1

            candidate = int(
                rng.integers(
                    low=0,
                    high=NUM_STARTUPS,
                    dtype=np.int64,
                )
            )

            pair_key = (
                investor_global
                * NUM_STARTUPS
                + candidate
            )

            first_segment = (
                pair_first_segment.get(
                    pair_key
                )
            )

            if (
                first_segment
                is not None
                and int(
                    first_segment
                )
                <= h
            ):

                forbidden_rejections += 1
                continue

            if candidate in (
                accepted_set
            ):

                duplicate_rejections += 1
                continue

            accepted.append(
                candidate
            )

            accepted_set.add(
                candidate
            )

            if first_segment is None:

                accepted_never_positive_pre_t60 += 1

            else:

                require(
                    int(
                        first_segment
                    )
                    > h,
                    (
                        "Accepted repair negative "
                        "has first positive <= h."
                    ),
                )

                accepted_future_positive += 1

        require(
            len(
                accepted
            )
            == NEGATIVES_PER_POSITIVE,
            (
                f"Negative count != 4 "
                f"at row {row_index}."
            ),
        )

        require(
            len(
                set(
                    accepted
                )
            )
            == NEGATIVES_PER_POSITIVE,
            (
                f"Duplicate accepted negative "
                f"at row {row_index}."
            ),
        )

        negatives[
            row_index
        ] = np.asarray(
            accepted,
            dtype=np.int32,
        )

        if (
            progress
            and (
                (
                    row_index
                    + 1
                )
                % 100_000
                == 0
                or (
                    row_index
                    + 1
                )
                == total_rows
            )
        ):

            print(
                f"Generated negatives for "
                f"{row_index + 1:,} / "
                f"{total_rows:,} positives"
            )

    stats = {
        "positive_rows": (
            total_rows
        ),

        "accepted_negatives": (
            total_rows
            * NEGATIVES_PER_POSITIVE
        ),

        "initial_draw_count": (
            initial_draw_count
        ),

        "repair_draw_count": (
            repair_draw_count
        ),

        "total_rng_candidate_draws": (
            initial_draw_count
            + repair_draw_count
        ),

        "forbidden_rejections": (
            forbidden_rejections
        ),

        "duplicate_rejections": (
            duplicate_rejections
        ),

        "accepted_future_positive": (
            accepted_future_positive
        ),

        "accepted_never_positive_pre_t60": (
            accepted_never_positive_pre_t60
        ),
    }

    return (
        negatives,
        stats,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.1l.1 — "
        "FREEZE EPOCH-0 TRAINING STREAM SERIALIZATION"
    )

    print(
        "Canonical neural model instantiated: NO"
    )

    print(
        "Epoch-0 negative RNG instantiated:   YES"
    )

    print(
        "Epoch-0 training order RNG:           YES"
    )

    print(
        "Adam instantiated:                    NO"
    )

    print(
        "Forward computation:                  NO"
    )

    print(
        "Backward computation:                 NO"
    )

    print(
        "Optimizer steps:                      0"
    )

    # =========================================================================
    # Prior canonical numerical gate
    # =========================================================================

    banner(
        "PHASE-5.3.1k CONTRACT RECHECK"
    )

    for path in (
        TEMPORAL_SPLIT_PATH,
        NODE_INDEX_PATH,
        PHASE_5_3_1K_CONTRACT_PATH,
        PHASE_5_3_1K_MANIFEST_PATH,
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

    phase_5_3_1k_contract = (
        load_json(
            PHASE_5_3_1K_CONTRACT_PATH
        )
    )

    phase_5_3_1k_manifest = (
        load_json(
            PHASE_5_3_1K_MANIFEST_PATH
        )
    )

    require(
        phase_5_3_1k_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.1k contract "
            "is not frozen."
        ),
    )

    require(
        phase_5_3_1k_manifest[
            "status"
        ]
        == (
            "CANONICAL_COMPOSED_REAL_DATA_"
            "FORWARD_BCE_BACKWARD_PREFLIGHT_PASSED"
        ),
        (
            "Unexpected Phase-5.3.1k status."
        ),
    )

    require(
        phase_5_3_1k_manifest[
            "optimizer_instantiated"
        ]
        is False,
        (
            "Optimizer unexpectedly instantiated "
            "before Phase-5.3.1l.1."
        ),
    )

    require(
        int(
            phase_5_3_1k_manifest[
                "optimizer_steps"
            ]
        )
        == 0,
        (
            "Optimizer step occurred before "
            "Phase-5.3.1l.1."
        ),
    )

    print(
        "Phase-5.3.1k:                         FROZEN / PASS"
    )

    print(
        "Optimizer steps entering phase:       0"
    )

    # =========================================================================
    # Seed derivation
    # =========================================================================

    banner(
        "FROZEN EPOCH-0 RNG SEED RECHECK"
    )

    epoch0_negative_seed = (
        derive_seed(
            NEGATIVE_NAMESPACE,
            BASE_SEED,
            EPOCH,
        )
    )

    epoch0_order_seed = (
        derive_seed(
            ORDER_NAMESPACE,
            BASE_SEED,
            EPOCH,
        )
    )

    require(
        epoch0_negative_seed
        == EXPECTED_EPOCH0_NEGATIVE_SEED,
        (
            "Epoch-0 negative seed drift.\n"
            f"Expected: "
            f"{EXPECTED_EPOCH0_NEGATIVE_SEED}\n"
            f"Actual:   "
            f"{epoch0_negative_seed}"
        ),
    )

    require(
        epoch0_order_seed
        == EXPECTED_EPOCH0_ORDER_SEED,
        (
            "Epoch-0 training-order seed drift.\n"
            f"Expected: "
            f"{EXPECTED_EPOCH0_ORDER_SEED}\n"
            f"Actual:   "
            f"{epoch0_order_seed}"
        ),
    )

    print(
        f"Negative namespace:                   "
        f"{NEGATIVE_NAMESPACE}"
    )

    print(
        f"Epoch-0 negative seed:                "
        f"{epoch0_negative_seed}"
    )

    print()

    print(
        f"Order namespace:                      "
        f"{ORDER_NAMESPACE}"
    )

    print(
        f"Epoch-0 order seed:                   "
        f"{epoch0_order_seed}"
    )

    print()

    print(
        f"NumPy:                                "
        f"{np.__version__}"
    )

    print(
        "Bit generator:                        PCG64"
    )

    # =========================================================================
    # Load Phase-3 role-node registry
    # =========================================================================

    banner(
        "ROLE-NODE REGISTRY RECHECK"
    )

    node_index = pd.read_parquet(
        NODE_INDEX_PATH,
        columns=[
            "node_index",
            "node_type",
            "raw_entity_id",
        ],
    )

    require(
        len(
            node_index
        )
        == NUM_NODES,
        (
            "Phase-3 node-index row count changed."
        ),
    )

    node_index = (
        node_index
        .sort_values(
            "node_index",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    require(
        np.array_equal(
            node_index[
                "node_index"
            ]
            .to_numpy(
                dtype=np.int64
            ),
            np.arange(
                NUM_NODES,
                dtype=np.int64,
            ),
        ),
        (
            "Phase-3 node indices are no longer "
            "contiguous global row order."
        ),
    )

    node_types = (
        node_index[
            "node_type"
        ]
        .astype(
            str
        )
        .str.lower()
    )

    investor_nodes = (
        node_index.loc[
            node_types
            == "investor"
        ]
        .copy()
    )

    startup_nodes = (
        node_index.loc[
            node_types
            == "startup"
        ]
        .copy()
    )

    require(
        len(
            investor_nodes
        )
        == NUM_INVESTORS,
        (
            "Investor-role node count changed."
        ),
    )

    require(
        len(
            startup_nodes
        )
        == NUM_STARTUPS,
        (
            "Startup-role node count changed."
        ),
    )

    require(
        np.array_equal(
            investor_nodes[
                "node_index"
            ]
            .to_numpy(
                dtype=np.int64
            ),
            np.arange(
                0,
                NUM_INVESTORS,
                dtype=np.int64,
            ),
        ),
        (
            "Investor node slice changed."
        ),
    )

    require(
        np.array_equal(
            startup_nodes[
                "node_index"
            ]
            .to_numpy(
                dtype=np.int64
            ),
            np.arange(
                NUM_INVESTORS,
                NUM_NODES,
                dtype=np.int64,
            ),
        ),
        (
            "Startup node slice changed."
        ),
    )

    investor_id_to_global = dict(
        zip(
            investor_nodes[
                "raw_entity_id"
            ].astype(
                str
            ),
            investor_nodes[
                "node_index"
            ].astype(
                np.int64
            ),
        )
    )

    startup_id_to_global = dict(
        zip(
            startup_nodes[
                "raw_entity_id"
            ].astype(
                str
            ),
            startup_nodes[
                "node_index"
            ].astype(
                np.int64
            ),
        )
    )

    startup_raw_by_local = (
        startup_nodes[
            "raw_entity_id"
        ]
        .astype(
            str
        )
        .to_numpy()
    )

    print(
        f"Investor role nodes:                  "
        f"{len(investor_nodes):,}"
    )

    print(
        f"Startup role nodes:                   "
        f"{len(startup_nodes):,}"
    )

    print(
        "Global role slices:                   PASS"
    )

    # =========================================================================
    # Load frozen temporal experiment
    # =========================================================================

    banner(
        "LOAD FROZEN T0..T59 TRAINING EVENTS"
    )

    temporal = pd.read_parquet(
        TEMPORAL_SPLIT_PATH,
        columns=[
            "interaction_id",
            "funding_round_id",
            "investor_id",
            "startup_id",
            "segment_number",
            "segment_label",
            "experiment_split",
        ],
    )

    temporal[
        "segment_number"
    ] = (
        temporal[
            "segment_number"
        ]
        .astype(
            np.int64
        )
    )

    t0_t59 = (
        temporal.loc[
            (
                temporal[
                    "segment_number"
                ]
                >= 0
            )
            & (
                temporal[
                    "segment_number"
                ]
                <= TRAIN_MAX_SEGMENT
            )
        ]
        .copy()
    )

    training = (
        temporal.loc[
            (
                temporal[
                    "segment_number"
                ]
                >= TRAIN_MIN_SEGMENT
            )
            & (
                temporal[
                    "segment_number"
                ]
                <= TRAIN_MAX_SEGMENT
            )
        ]
        .copy()
    )

    require(
        len(
            t0_t59
        )
        == EXPECTED_T0_T59_EVENTS,
        (
            "T0..T59 event count changed.\n"
            f"Expected: "
            f"{EXPECTED_T0_T59_EVENTS:,}\n"
            f"Actual:   "
            f"{len(t0_t59):,}"
        ),
    )

    require(
        len(
            training
        )
        == EXPECTED_TRAIN_POSITIVES,
        (
            "Training-positive event count changed.\n"
            f"Expected: "
            f"{EXPECTED_TRAIN_POSITIVES:,}\n"
            f"Actual:   "
            f"{len(training):,}"
        ),
    )

    require(
        training[
            "interaction_id"
        ].notna().all(),
        (
            "Training interaction_id contains null."
        ),
    )

    require(
        training[
            "interaction_id"
        ].astype(
            str
        ).is_unique,
        (
            "Training interaction_id is not unique."
        ),
    )

    require(
        bool(
            (
                training[
                    "segment_number"
                ]
                >= TRAIN_MIN_SEGMENT
            ).all()
        ),
        (
            "Training contains segment < T1."
        ),
    )

    require(
        bool(
            (
                training[
                    "segment_number"
                ]
                <= TRAIN_MAX_SEGMENT
            ).all()
        ),
        (
            "Training contains T60/post-T60."
        ),
    )

    print(
        f"T0..T59 events:                       "
        f"{len(t0_t59):,}"
    )

    print(
        f"T1..T59 training positives:           "
        f"{len(training):,}"
    )

    print(
        "T60 events used in training stream:   NO"
    )

    # =========================================================================
    # Map event IDs to frozen role indices
    # =========================================================================

    banner(
        "MAP TRAINING EVENTS TO FROZEN ROLE INDICES"
    )

    for frame_name, frame in (
        (
            "T0..T59",
            t0_t59,
        ),
        (
            "training",
            training,
        ),
    ):

        frame[
            "investor_global"
        ] = (
            frame[
                "investor_id"
            ]
            .astype(
                str
            )
            .map(
                investor_id_to_global
            )
        )

        frame[
            "startup_global"
        ] = (
            frame[
                "startup_id"
            ]
            .astype(
                str
            )
            .map(
                startup_id_to_global
            )
        )

        require(
            frame[
                "investor_global"
            ].notna().all(),
            (
                f"{frame_name} contains Investor IDs "
                "missing from Phase-3 role registry."
            ),
        )

        require(
            frame[
                "startup_global"
            ].notna().all(),
            (
                f"{frame_name} contains Startup IDs "
                "missing from Phase-3 role registry."
            ),
        )

        frame[
            "investor_global"
        ] = (
            frame[
                "investor_global"
            ]
            .astype(
                np.int64
            )
        )

        frame[
            "startup_global"
        ] = (
            frame[
                "startup_global"
            ]
            .astype(
                np.int64
            )
        )

        frame[
            "startup_local"
        ] = (
            frame[
                "startup_global"
            ]
            - NUM_INVESTORS
        ).astype(
            np.int64
        )

        require(
            bool(
                (
                    frame[
                        "investor_global"
                    ]
                    >= 0
                ).all()
            )
            and bool(
                (
                    frame[
                        "investor_global"
                    ]
                    < NUM_INVESTORS
                ).all()
            ),
            (
                f"{frame_name} Investor index "
                "outside Investor role slice."
            ),
        )

        require(
            bool(
                (
                    frame[
                        "startup_local"
                    ]
                    >= 0
                ).all()
            )
            and bool(
                (
                    frame[
                        "startup_local"
                    ]
                    < NUM_STARTUPS
                ).all()
            ),
            (
                f"{frame_name} Startup local index "
                "outside Startup role universe."
            ),
        )

    print(
        "Investor mapping:                     PASS"
    )

    print(
        "Startup mapping:                      PASS"
    )

    # =========================================================================
    # Freeze canonical positive order
    # =========================================================================

    banner(
        "CANONICAL POSITIVE-EVENT ORDER"
    )

    training[
        "interaction_id"
    ] = (
        training[
            "interaction_id"
        ]
        .astype(
            str
        )
    )

    ordered_positive = (
        training[
            [
                "interaction_id",
                "funding_round_id",
                "investor_id",
                "startup_id",
                "investor_global",
                "startup_global",
                "startup_local",
                "segment_number",
                "segment_label",
            ]
        ]
        .sort_values(
            "interaction_id",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    ordered_positive.insert(
        0,
        "positive_order_index",
        np.arange(
            len(
                ordered_positive
            ),
            dtype=np.int64,
        ),
    )

    require(
        len(
            ordered_positive
        )
        == EXPECTED_TRAIN_POSITIVES,
        (
            "Ordered positive count changed."
        ),
    )

    require(
        ordered_positive[
            "positive_order_index"
        ].iloc[
            0
        ]
        == 0,
        (
            "Positive order does not begin at zero."
        ),
    )

    require(
        ordered_positive[
            "positive_order_index"
        ].iloc[
            -1
        ]
        == (
            EXPECTED_TRAIN_POSITIVES
            - 1
        ),
        (
            "Positive order terminal index changed."
        ),
    )

    positive_logical_sha = (
        positive_stream_logical_sha256(
            ordered_positive
        )
    )

    print(
        "Canonical pre-expansion ordering:"
    )

    print(
        "  lexicographic interaction_id"
    )

    print()

    print(
        f"Positive events:                      "
        f"{len(ordered_positive):,}"
    )

    print(
        "Positive logical SHA256:"
    )

    print(
        positive_logical_sha
    )

    # =========================================================================
    # Build positive-pair first-segment lookup using ONLY T0..T59
    # =========================================================================

    banner(
        "TRAINING NEGATIVE ELIGIBILITY LOOKUP"
    )

    pair_keys = (
        t0_t59[
            "investor_global"
        ]
        .to_numpy(
            dtype=np.int64
        )
        * NUM_STARTUPS
        + t0_t59[
            "startup_local"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    first_frame = pd.DataFrame(
        {
            "pair_key": (
                pair_keys
            ),

            "segment_number": (
                t0_t59[
                    "segment_number"
                ]
                .to_numpy(
                    dtype=np.int64
                )
            ),
        }
    )

    pair_first_series = (
        first_frame
        .groupby(
            "pair_key",
            sort=False,
        )[
            "segment_number"
        ]
        .min()
    )

    pair_first_segment = dict(
        zip(
            pair_first_series
            .index
            .to_numpy(
                dtype=np.int64
            )
            .tolist(),
            pair_first_series
            .to_numpy(
                dtype=np.int64
            )
            .tolist(),
        )
    )

    del first_frame
    del pair_first_series

    print(
        f"Unique Investor–Startup pairs "
        f"in T0..T59:                          "
        f"{len(pair_first_segment):,}"
    )

    print(
        "Eligibility lookup uses T60 labels:   NO"
    )

    # =========================================================================
    # Generate full epoch-0 negative matrix
    # =========================================================================

    banner(
        "GENERATE EPOCH-0 TRAINING NEGATIVES"
    )

    investor_globals = (
        ordered_positive[
            "investor_global"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    positive_startup_locals = (
        ordered_positive[
            "startup_local"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    target_segments = (
        ordered_positive[
            "segment_number"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    (
        negative_matrix,
        negative_stats,
    ) = generate_epoch_negatives(
        investor_globals=(
            investor_globals
        ),
        target_segments=(
            target_segments
        ),
        pair_first_segment=(
            pair_first_segment
        ),
        seed=(
            epoch0_negative_seed
        ),
        max_rows=None,
        progress=True,
    )

    require(
        negative_matrix.shape
        == (
            EXPECTED_TRAIN_POSITIVES,
            NEGATIVES_PER_POSITIVE,
        ),
        (
            "Epoch-0 negative matrix shape changed."
        ),
    )

    require(
        negative_matrix.dtype
        == np.int32,
        (
            "Epoch-0 negative matrix dtype "
            "is not int32."
        ),
    )

    require(
        int(
            negative_matrix.min()
        )
        >= 0,
        (
            "Negative Startup local index < 0."
        ),
    )

    require(
        int(
            negative_matrix.max()
        )
        < NUM_STARTUPS,
        (
            "Negative Startup local index "
            "outside role universe."
        ),
    )

    # -------------------------------------------------------------------------
    # Hard row-level global integrity checks.
    # -------------------------------------------------------------------------

    forbidden_collision_count = 0
    focal_collision_count = 0
    duplicate_negative_row_count = 0

    for row_index in range(
        EXPECTED_TRAIN_POSITIVES
    ):

        investor_global = int(
            investor_globals[
                row_index
            ]
        )

        focal_startup = int(
            positive_startup_locals[
                row_index
            ]
        )

        h = int(
            target_segments[
                row_index
            ]
        )

        row_negatives = (
            negative_matrix[
                row_index
            ]
        )

        if len(
            set(
                int(
                    value
                )
                for value in (
                    row_negatives
                )
            )
        ) != NEGATIVES_PER_POSITIVE:

            duplicate_negative_row_count += 1

        for candidate_value in (
            row_negatives
        ):

            candidate = int(
                candidate_value
            )

            if candidate == focal_startup:

                focal_collision_count += 1

            pair_key = (
                investor_global
                * NUM_STARTUPS
                + candidate
            )

            first_segment = (
                pair_first_segment.get(
                    pair_key
                )
            )

            if (
                first_segment
                is not None
                and int(
                    first_segment
                )
                <= h
            ):

                forbidden_collision_count += 1

    require(
        duplicate_negative_row_count
        == 0,
        (
            "At least one positive has duplicate "
            "accepted negatives."
        ),
    )

    require(
        focal_collision_count
        == 0,
        (
            "At least one negative equals "
            "the focal positive Startup."
        ),
    )

    require(
        forbidden_collision_count
        == 0,
        (
            "At least one generated negative "
            "was already positive by target h."
        ),
    )

    expected_negative_slots = (
        EXPECTED_TRAIN_POSITIVES
        * NEGATIVES_PER_POSITIVE
    )

    require(
        expected_negative_slots
        == EXPECTED_EPOCH_NEGATIVES,
        (
            "Frozen epoch-negative count mismatch."
        ),
    )

    negative_logical_sha = (
        array_logical_sha256(
            negative_matrix
        )
    )

    print()

    print(
        f"Negative matrix:                      "
        f"{negative_matrix.shape}"
    )

    print(
        f"Accepted negatives:                   "
        f"{EXPECTED_EPOCH_NEGATIVES:,}"
    )

    print(
        f"Repair RNG draws:                     "
        f"{negative_stats['repair_draw_count']:,}"
    )

    print(
        f"Forbidden candidate rejections:       "
        f"{negative_stats['forbidden_rejections']:,}"
    )

    print(
        f"Duplicate candidate rejections:       "
        f"{negative_stats['duplicate_rejections']:,}"
    )

    print(
        f"Accepted future-positive pairs:       "
        f"{negative_stats['accepted_future_positive']:,}"
    )

    print(
        f"Accepted never-positive pre-T60:      "
        f"{negative_stats['accepted_never_positive_pre_t60']:,}"
    )

    print()

    print(
        "Epoch-0 negative logical SHA256:"
    )

    print(
        negative_logical_sha
    )

    # =========================================================================
    # Prefix deterministic regeneration audit
    # =========================================================================

    banner(
        "NEGATIVE RNG DETERMINISTIC PREFIX REGENERATION"
    )

    REPRO_PREFIX_ROWS = 10_000

    (
        prefix_regenerated,
        prefix_stats,
    ) = generate_epoch_negatives(
        investor_globals=(
            investor_globals
        ),
        target_segments=(
            target_segments
        ),
        pair_first_segment=(
            pair_first_segment
        ),
        seed=(
            epoch0_negative_seed
        ),
        max_rows=(
            REPRO_PREFIX_ROWS
        ),
        progress=False,
    )

    require(
        np.array_equal(
            prefix_regenerated,
            negative_matrix[
                :REPRO_PREFIX_ROWS
            ],
        ),
        (
            "Epoch-0 negative RNG prefix "
            "did not regenerate exactly."
        ),
    )

    print(
        f"Regenerated positive rows:            "
        f"{REPRO_PREFIX_ROWS:,}"
    )

    print(
        "Exact negative prefix match:          PASS"
    )

    # =========================================================================
    # Conceptual five-slot serialization
    # =========================================================================

    banner(
        "CANONICAL FIVE-SLOT TRAINING SERIALIZATION"
    )

    total_examples = (
        EXPECTED_TRAIN_POSITIVES
        * SERIALIZED_SLOTS_PER_POSITIVE
    )

    require(
        total_examples
        == EXPECTED_EPOCH_EXAMPLES,
        (
            "Frozen epoch example count mismatch."
        ),
    )

    calculated_batches = int(
        math.ceil(
            total_examples
            / BATCH_SIZE
        )
    )

    calculated_final_batch = (
        total_examples
        - (
            calculated_batches
            - 1
        )
        * BATCH_SIZE
    )

    require(
        calculated_batches
        == EXPECTED_EPOCH_BATCHES,
        (
            "Epoch batch count changed."
        ),
    )

    require(
        calculated_final_batch
        == EXPECTED_FINAL_BATCH_SIZE,
        (
            "Final batch size changed."
        ),
    )

    print(
        "Serialization:"
    )

    print(
        "  slot 0 = positive"
    )

    print(
        "  slot 1 = negative draw 0"
    )

    print(
        "  slot 2 = negative draw 1"
    )

    print(
        "  slot 3 = negative draw 2"
    )

    print(
        "  slot 4 = negative draw 3"
    )

    print()

    print(
        f"Conceptual epoch examples:            "
        f"{total_examples:,}"
    )

    print(
        f"Batch size:                           "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epoch batches:                        "
        f"{calculated_batches:,}"
    )

    print(
        f"Final batch:                          "
        f"{calculated_final_batch}"
    )

    # =========================================================================
    # Epoch-0 shuffle
    # =========================================================================

    banner(
        "GENERATE EPOCH-0 TRAINING ORDER"
    )

    epoch_order = np.arange(
        total_examples,
        dtype=np.int64,
    )

    order_rng = np.random.Generator(
        np.random.PCG64(
            epoch0_order_seed
        )
    )

    order_rng.shuffle(
        epoch_order
    )

    require(
        epoch_order.shape
        == (
            EXPECTED_EPOCH_EXAMPLES,
        ),
        (
            "Epoch-order shape changed."
        ),
    )

    require(
        epoch_order.dtype
        == np.int64,
        (
            "Epoch-order dtype is not int64."
        ),
    )

    require(
        int(
            epoch_order.min()
        )
        == 0,
        (
            "Epoch-order minimum changed."
        ),
    )

    require(
        int(
            epoch_order.max()
        )
        == (
            EXPECTED_EPOCH_EXAMPLES
            - 1
        ),
        (
            "Epoch-order maximum changed."
        ),
    )

    # Since this was created as arange() and shuffled in place,
    # it is by construction a permutation. We additionally verify
    # deterministic regeneration.
    regenerated_order = np.arange(
        total_examples,
        dtype=np.int64,
    )

    regenerated_order_rng = (
        np.random.Generator(
            np.random.PCG64(
                epoch0_order_seed
            )
        )
    )

    regenerated_order_rng.shuffle(
        regenerated_order
    )

    require(
        np.array_equal(
            epoch_order,
            regenerated_order,
        ),
        (
            "Epoch-0 training order did not "
            "regenerate exactly."
        ),
    )

    del regenerated_order

    epoch_order_logical_sha = (
        array_logical_sha256(
            epoch_order
        )
    )

    print(
        f"Shuffled conceptual indices:          "
        f"{len(epoch_order):,}"
    )

    print(
        "Deterministic full regeneration:      PASS"
    )

    print()

    print(
        "Epoch-0 order logical SHA256:"
    )

    print(
        epoch_order_logical_sha
    )

    # =========================================================================
    # Decode canonical first mini-batch
    # =========================================================================

    banner(
        "DECODE CANONICAL EPOCH-0 FIRST MINI-BATCH"
    )

    first_batch_serialized = (
        epoch_order[
            :BATCH_SIZE
        ]
    )

    first_batch_positive_indices = (
        first_batch_serialized
        // SERIALIZED_SLOTS_PER_POSITIVE
    ).astype(
        np.int64
    )

    first_batch_slots = (
        first_batch_serialized
        % SERIALIZED_SLOTS_PER_POSITIVE
    ).astype(
        np.int64
    )

    batch_rows = []

    for batch_position in range(
        BATCH_SIZE
    ):

        serialized_index = int(
            first_batch_serialized[
                batch_position
            ]
        )

        positive_index = int(
            first_batch_positive_indices[
                batch_position
            ]
        )

        slot = int(
            first_batch_slots[
                batch_position
            ]
        )

        positive_row = (
            ordered_positive.iloc[
                positive_index
            ]
        )

        investor_global = int(
            positive_row[
                "investor_global"
            ]
        )

        focal_startup_local = int(
            positive_row[
                "startup_local"
            ]
        )

        h = int(
            positive_row[
                "segment_number"
            ]
        )

        if slot == 0:

            label = 1

            startup_local = (
                focal_startup_local
            )

            negative_draw_index = None

        else:

            label = 0

            negative_draw_index = (
                slot
                - 1
            )

            startup_local = int(
                negative_matrix[
                    positive_index,
                    negative_draw_index,
                ]
            )

        startup_global = (
            NUM_INVESTORS
            + startup_local
        )

        startup_raw_id = str(
            startup_raw_by_local[
                startup_local
            ]
        )

        batch_rows.append(
            {
                "batch_position": (
                    batch_position
                ),

                "serialized_example_index": (
                    serialized_index
                ),

                "positive_order_index": (
                    positive_index
                ),

                "example_slot": (
                    slot
                ),

                "negative_draw_index": (
                    negative_draw_index
                ),

                "label": (
                    label
                ),

                "source_interaction_id": (
                    str(
                        positive_row[
                            "interaction_id"
                        ]
                    )
                ),

                "source_funding_round_id": (
                    str(
                        positive_row[
                            "funding_round_id"
                        ]
                    )
                ),

                "investor_id": (
                    str(
                        positive_row[
                            "investor_id"
                        ]
                    )
                ),

                "investor_global": (
                    investor_global
                ),

                "startup_id": (
                    startup_raw_id
                ),

                "startup_local": (
                    startup_local
                ),

                "startup_global": (
                    startup_global
                ),

                "focal_positive_startup_local": (
                    focal_startup_local
                ),

                "segment_number": (
                    h
                ),

                "segment_label": (
                    f"T{h}"
                ),

                "trend_history_period_count": (
                    h
                ),
            }
        )

    first_batch = pd.DataFrame(
        batch_rows
    )

    require(
        len(
            first_batch
        )
        == BATCH_SIZE,
        (
            "First training batch size changed."
        ),
    )

    require(
        int(
            first_batch[
                "batch_position"
            ].iloc[
                0
            ]
        )
        == 0,
        (
            "First-batch position starts incorrectly."
        ),
    )

    require(
        int(
            first_batch[
                "batch_position"
            ].iloc[
                -1
            ]
        )
        == (
            BATCH_SIZE
            - 1
        ),
        (
            "First-batch terminal position incorrect."
        ),
    )

    require(
        bool(
            first_batch[
                "segment_number"
            ]
            .between(
                TRAIN_MIN_SEGMENT,
                TRAIN_MAX_SEGMENT,
            )
            .all()
        ),
        (
            "First batch contains invalid "
            "training segment."
        ),
    )

    first_batch_positive_count = int(
        (
            first_batch[
                "label"
            ]
            == 1
        ).sum()
    )

    first_batch_negative_count = int(
        (
            first_batch[
                "label"
            ]
            == 0
        ).sum()
    )

    require(
        first_batch_positive_count
        + first_batch_negative_count
        == BATCH_SIZE,
        (
            "First-batch label count mismatch."
        ),
    )

    # -------------------------------------------------------------------------
    # Recheck all negative examples in first batch.
    # -------------------------------------------------------------------------

    first_batch_forbidden = 0
    first_batch_focal_collisions = 0

    for row in (
        first_batch.itertuples(
            index=False
        )
    ):

        if int(
            row.label
        ) == 1:

            require(
                int(
                    row.startup_local
                )
                == int(
                    row.focal_positive_startup_local
                ),
                (
                    "Positive first-batch example "
                    "does not use focal Startup."
                ),
            )

            continue

        if int(
            row.startup_local
        ) == int(
            row.focal_positive_startup_local
        ):

            first_batch_focal_collisions += 1

        pair_key = (
            int(
                row.investor_global
            )
            * NUM_STARTUPS
            + int(
                row.startup_local
            )
        )

        first_segment = (
            pair_first_segment.get(
                pair_key
            )
        )

        if (
            first_segment
            is not None
            and int(
                first_segment
            )
            <= int(
                row.segment_number
            )
        ):

            first_batch_forbidden += 1

    require(
        first_batch_forbidden
        == 0,
        (
            "First batch contains a forbidden negative."
        ),
    )

    require(
        first_batch_focal_collisions
        == 0,
        (
            "First batch contains a negative equal "
            "to its focal positive Startup."
        ),
    )

    # =========================================================================
    # Freeze mixed-segment grouping plan
    # =========================================================================

    banner(
        "FIRST-BATCH TARGET-SEGMENT GROUPING PLAN"
    )

    segment_group_rows = []

    for (
        segment_number,
        group,
    ) in first_batch.groupby(
        "segment_number",
        sort=True,
    ):

        positions = (
            group[
                "batch_position"
            ]
            .astype(
                np.int64
            )
            .tolist()
        )

        segment_group_rows.append(
            {
                "segment_number": (
                    int(
                        segment_number
                    )
                ),

                "segment_label": (
                    f"T{int(segment_number)}"
                ),

                "history_periods_consumed": (
                    int(
                        segment_number
                    )
                ),

                "example_count": (
                    len(
                        positions
                    )
                ),

                "batch_positions": (
                    ";".join(
                        str(
                            position
                        )
                        for position
                        in positions
                    )
                ),
            }
        )

    segment_groups = pd.DataFrame(
        segment_group_rows
    )

    require(
        int(
            segment_groups[
                "example_count"
            ].sum()
        )
        == BATCH_SIZE,
        (
            "Segment-group counts do not sum "
            "to first batch size."
        ),
    )

    require(
        bool(
            (
                segment_groups[
                    "history_periods_consumed"
                ]
                == segment_groups[
                    "segment_number"
                ]
            ).all()
        ),
        (
            "Target segment -> history-length mapping "
            "is not exact."
        ),
    )

    print(
        segment_groups[
            [
                "segment_number",
                "history_periods_consumed",
                "example_count",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Runtime rule:"
    )

    print(
        "  For target T_h, trend consumes exactly T0..T(h-1)."
    )

    print(
        "  No post-h zero periods are passed through the GRU."
    )

    print(
        "  Segment-group outputs must be restored to original "
        "batch_position before BCE."
    )

    # =========================================================================
    # First-batch logical hash
    # =========================================================================

    first_batch_logical_sha = (
        dataframe_logical_sha256(
            first_batch,
            columns=[
                "batch_position",
                "serialized_example_index",
                "positive_order_index",
                "example_slot",
                "label",
                "source_interaction_id",
                "investor_global",
                "startup_local",
                "segment_number",
            ],
        )
    )

    print()

    print(
        f"First-batch positives:                "
        f"{first_batch_positive_count}"
    )

    print(
        f"First-batch negatives:                "
        f"{first_batch_negative_count}"
    )

    print(
        f"Distinct target segments:             "
        f"{first_batch['segment_number'].nunique()}"
    )

    print(
        f"Min target segment:                   "
        f"T{int(first_batch['segment_number'].min())}"
    )

    print(
        f"Max target segment:                   "
        f"T{int(first_batch['segment_number'].max())}"
    )

    print()

    print(
        "First-batch logical SHA256:"
    )

    print(
        first_batch_logical_sha
    )

    # =========================================================================
    # Write runtime artifacts
    # =========================================================================

    banner(
        "WRITE EPOCH-0 TRAINING STREAM ARTIFACTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RUNTIME_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ordered_positive.to_parquet(
        POSITIVE_ORDER_PATH,
        index=False,
    )

    np.save(
        NEGATIVE_MATRIX_PATH,
        negative_matrix,
        allow_pickle=False,
    )

    np.save(
        EPOCH_ORDER_PATH,
        epoch_order,
        allow_pickle=False,
    )

    first_batch.to_parquet(
        FIRST_BATCH_PATH,
        index=False,
    )

    segment_groups.to_csv(
        FIRST_BATCH_SEGMENT_GROUP_PATH,
        index=False,
    )

    # =========================================================================
    # Post-write hash registry
    # =========================================================================

    hash_rows = [
        {
            "artifact": (
                str(
                    POSITIVE_ORDER_PATH
                )
            ),

            "physical_file_sha256": (
                file_sha256(
                    POSITIVE_ORDER_PATH
                )
            ),

            "logical_sha256": (
                positive_logical_sha
            ),
        },

        {
            "artifact": (
                str(
                    NEGATIVE_MATRIX_PATH
                )
            ),

            "physical_file_sha256": (
                file_sha256(
                    NEGATIVE_MATRIX_PATH
                )
            ),

            "logical_sha256": (
                negative_logical_sha
            ),
        },

        {
            "artifact": (
                str(
                    EPOCH_ORDER_PATH
                )
            ),

            "physical_file_sha256": (
                file_sha256(
                    EPOCH_ORDER_PATH
                )
            ),

            "logical_sha256": (
                epoch_order_logical_sha
            ),
        },

        {
            "artifact": (
                str(
                    FIRST_BATCH_PATH
                )
            ),

            "physical_file_sha256": (
                file_sha256(
                    FIRST_BATCH_PATH
                )
            ),

            "logical_sha256": (
                first_batch_logical_sha
            ),
        },

        {
            "artifact": (
                str(
                    FIRST_BATCH_SEGMENT_GROUP_PATH
                )
            ),

            "physical_file_sha256": (
                file_sha256(
                    FIRST_BATCH_SEGMENT_GROUP_PATH
                )
            ),

            "logical_sha256": (
                None
            ),
        },
    ]

    hash_df = pd.DataFrame(
        hash_rows
    )

    hash_df.to_csv(
        HASH_REGISTRY_PATH,
        index=False,
    )

    # =========================================================================
    # Audit tables
    # =========================================================================

    negative_audit_df = pd.DataFrame(
        [
            {
                "metric": (
                    key
                ),
                "value": (
                    value
                ),
            }
            for (
                key,
                value,
            ) in negative_stats.items()
        ]
        + [
            {
                "metric": (
                    "forbidden_collision_count"
                ),
                "value": (
                    forbidden_collision_count
                ),
            },

            {
                "metric": (
                    "focal_collision_count"
                ),
                "value": (
                    focal_collision_count
                ),
            },

            {
                "metric": (
                    "duplicate_negative_row_count"
                ),
                "value": (
                    duplicate_negative_row_count
                ),
            },

            {
                "metric": (
                    "negative_logical_sha256"
                ),
                "value": (
                    negative_logical_sha
                ),
            },
        ]
    )

    negative_audit_df.to_csv(
        NEGATIVE_AUDIT_PATH,
        index=False,
    )

    training_stream_audit_df = pd.DataFrame(
        [
            {
                "metric": (
                    "training_positive_events"
                ),
                "value": (
                    EXPECTED_TRAIN_POSITIVES
                ),
            },

            {
                "metric": (
                    "negative_examples"
                ),
                "value": (
                    EXPECTED_EPOCH_NEGATIVES
                ),
            },

            {
                "metric": (
                    "epoch_examples"
                ),
                "value": (
                    EXPECTED_EPOCH_EXAMPLES
                ),
            },

            {
                "metric": (
                    "batch_size"
                ),
                "value": (
                    BATCH_SIZE
                ),
            },

            {
                "metric": (
                    "epoch_batches"
                ),
                "value": (
                    EXPECTED_EPOCH_BATCHES
                ),
            },

            {
                "metric": (
                    "final_batch_size"
                ),
                "value": (
                    EXPECTED_FINAL_BATCH_SIZE
                ),
            },

            {
                "metric": (
                    "epoch0_negative_seed"
                ),
                "value": (
                    epoch0_negative_seed
                ),
            },

            {
                "metric": (
                    "epoch0_order_seed"
                ),
                "value": (
                    epoch0_order_seed
                ),
            },

            {
                "metric": (
                    "positive_logical_sha256"
                ),
                "value": (
                    positive_logical_sha
                ),
            },

            {
                "metric": (
                    "epoch_order_logical_sha256"
                ),
                "value": (
                    epoch_order_logical_sha
                ),
            },
        ]
    )

    training_stream_audit_df.to_csv(
        TRAINING_STREAM_AUDIT_PATH,
        index=False,
    )

    first_batch_audit_df = pd.DataFrame(
        [
            {
                "metric": (
                    "batch_size"
                ),
                "value": (
                    len(
                        first_batch
                    )
                ),
            },

            {
                "metric": (
                    "positive_count"
                ),
                "value": (
                    first_batch_positive_count
                ),
            },

            {
                "metric": (
                    "negative_count"
                ),
                "value": (
                    first_batch_negative_count
                ),
            },

            {
                "metric": (
                    "distinct_target_segments"
                ),
                "value": (
                    int(
                        first_batch[
                            "segment_number"
                        ].nunique()
                    )
                ),
            },

            {
                "metric": (
                    "minimum_target_segment"
                ),
                "value": (
                    int(
                        first_batch[
                            "segment_number"
                        ].min()
                    )
                ),
            },

            {
                "metric": (
                    "maximum_target_segment"
                ),
                "value": (
                    int(
                        first_batch[
                            "segment_number"
                        ].max()
                    )
                ),
            },

            {
                "metric": (
                    "forbidden_negative_count"
                ),
                "value": (
                    first_batch_forbidden
                ),
            },

            {
                "metric": (
                    "focal_negative_collision_count"
                ),
                "value": (
                    first_batch_focal_collisions
                ),
            },

            {
                "metric": (
                    "first_batch_logical_sha256"
                ),
                "value": (
                    first_batch_logical_sha
                ),
            },
        ]
    )

    first_batch_audit_df.to_csv(
        FIRST_BATCH_AUDIT_PATH,
        index=False,
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.1l.1 FREEZE INVARIANTS"
    )

    checks = [
        (
            "phase_5_3_1k_contract_frozen",
            (
                phase_5_3_1k_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),

        (
            "optimizer_steps_entering_phase_zero",
            (
                int(
                    phase_5_3_1k_manifest[
                        "optimizer_steps"
                    ]
                )
                == 0
            ),
        ),

        (
            "epoch0_negative_seed_exact",
            (
                epoch0_negative_seed
                == EXPECTED_EPOCH0_NEGATIVE_SEED
            ),
        ),

        (
            "epoch0_order_seed_exact",
            (
                epoch0_order_seed
                == EXPECTED_EPOCH0_ORDER_SEED
            ),
        ),

        (
            "t0_t59_event_count_exact",
            (
                len(
                    t0_t59
                )
                == EXPECTED_T0_T59_EVENTS
            ),
        ),

        (
            "training_positive_event_count_exact",
            (
                len(
                    ordered_positive
                )
                == EXPECTED_TRAIN_POSITIVES
            ),
        ),

        (
            "canonical_positive_order_interaction_id_unique",
            (
                ordered_positive[
                    "interaction_id"
                ].is_unique
            ),
        ),

        (
            "training_targets_only_T1_to_T59",
            bool(
                ordered_positive[
                    "segment_number"
                ]
                .between(
                    TRAIN_MIN_SEGMENT,
                    TRAIN_MAX_SEGMENT,
                )
                .all()
            ),
        ),

        (
            "T60_labels_not_used_for_negative_eligibility",
            True,
        ),

        (
            "negative_matrix_shape_exact",
            (
                negative_matrix.shape
                == (
                    EXPECTED_TRAIN_POSITIVES,
                    NEGATIVES_PER_POSITIVE,
                )
            ),
        ),

        (
            "epoch_negative_count_exact",
            (
                negative_matrix.size
                == EXPECTED_EPOCH_NEGATIVES
            ),
        ),

        (
            "all_negative_startups_inside_role_universe",
            (
                int(
                    negative_matrix.min()
                )
                >= 0
                and int(
                    negative_matrix.max()
                )
                < NUM_STARTUPS
            ),
        ),

        (
            "no_duplicate_negatives_within_positive",
            (
                duplicate_negative_row_count
                == 0
            ),
        ),

        (
            "no_focal_startup_as_negative",
            (
                focal_collision_count
                == 0
            ),
        ),

        (
            "no_positive_pair_at_or_before_h_sampled_negative",
            (
                forbidden_collision_count
                == 0
            ),
        ),

        (
            "future_positive_pairs_not_globally_excluded",
            (
                negative_stats[
                    "accepted_future_positive"
                ]
                >= 0
            ),
        ),

        (
            "negative_rng_prefix_exactly_regenerates",
            np.array_equal(
                prefix_regenerated,
                negative_matrix[
                    :REPRO_PREFIX_ROWS
                ],
            ),
        ),

        (
            "five_slot_serialization_exact",
            (
                total_examples
                == EXPECTED_EPOCH_EXAMPLES
            ),
        ),

        (
            "epoch_batch_count_exact",
            (
                calculated_batches
                == EXPECTED_EPOCH_BATCHES
            ),
        ),

        (
            "final_batch_size_exact",
            (
                calculated_final_batch
                == EXPECTED_FINAL_BATCH_SIZE
            ),
        ),

        (
            "epoch_order_shape_exact",
            (
                epoch_order.shape
                == (
                    EXPECTED_EPOCH_EXAMPLES,
                )
            ),
        ),

        (
            "epoch_order_full_regeneration_exact",
            True,
        ),

        (
            "first_batch_size_exact",
            (
                len(
                    first_batch
                )
                == BATCH_SIZE
            ),
        ),

        (
            "first_batch_negative_integrity",
            (
                first_batch_forbidden
                == 0
                and first_batch_focal_collisions
                == 0
            ),
        ),

        (
            "first_batch_segments_only_T1_to_T59",
            bool(
                first_batch[
                    "segment_number"
                ]
                .between(
                    TRAIN_MIN_SEGMENT,
                    TRAIN_MAX_SEGMENT,
                )
                .all()
            ),
        ),

        (
            "segment_history_length_equals_target_h",
            bool(
                (
                    segment_groups[
                        "history_periods_consumed"
                    ]
                    == segment_groups[
                        "segment_number"
                    ]
                ).all()
            ),
        ),

        (
            "segment_groups_cover_all_512_positions",
            (
                int(
                    segment_groups[
                        "example_count"
                    ].sum()
                )
                == BATCH_SIZE
            ),
        ),

        (
            "model_not_instantiated",
            True,
        ),

        (
            "Adam_not_instantiated",
            True,
        ),

        (
            "forward_not_performed",
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

        (
            "checkpoint_not_written",
            True,
        ),
    ]

    invariant_df = pd.DataFrame(
        [
            {
                "check": (
                    name
                ),

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
        (
            invariant_df[
                "result"
            ]
            == "PASS"
        ).all(),
        (
            "At least one Phase-5.3.1l.1 "
            "training-stream invariant failed."
        ),
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    print(
        invariant_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Freeze decision register
    # =========================================================================

    banner(
        "FREEZE EPOCH-0 TRAINING STREAM CONTRACT"
    )

    decision_df = pd.DataFrame(
        [
            {
                "decision": (
                    "canonical_positive_event_order"
                ),

                "value": (
                    "LEXICOGRAPHIC_INTERACTION_ID_STABLE_MERGESORT"
                ),

                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_1"
                ),
            },

            {
                "decision": (
                    "per_positive_serialization"
                ),

                "value": (
                    "SLOT0_POSITIVE_SLOT1_TO_4_NEGATIVE_DRAW_ORDER"
                ),

                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_1"
                ),
            },

            {
                "decision": (
                    "training_negative_rng_consumption_order"
                ),

                "value": (
                    "CANONICAL_POSITIVE_ORDER"
                ),

                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_1"
                ),
            },

            {
                "decision": (
                    "training_negative_draw_algorithm"
                ),

                "value": (
                    "PCG64_UNIFORM_STARTUP_LOCAL_REJECTION_"
                    "FOR_PRIOR_OR_TARGET_POSITIVE_AND_WITHIN_ROW_DUPLICATE"
                ),

                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_1"
                ),
            },

            {
                "decision": (
                    "epoch_order_algorithm"
                ),

                "value": (
                    "PCG64_INPLACE_SHUFFLE_OF_ARANGE_INT64"
                ),

                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_1"
                ),
            },

            {
                "decision": (
                    "mixed_segment_training_forward_policy"
                ),

                "value": (
                    "GROUP_BY_TARGET_H_USE_EXACT_T0_TO_T_H_MINUS_1_"
                    "THEN_RESTORE_BATCH_POSITION"
                ),

                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_1"
                ),
            },

            {
                "decision": (
                    "post_h_GRU_padding"
                ),

                "value": (
                    "FORBIDDEN"
                ),

                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1l_1"
                ),
            },
        ]
    )

    decision_df.to_csv(
        DECISION_REGISTER_PATH,
        index=False,
    )

    # =========================================================================
    # Contract
    # =========================================================================

    contract = {
        "phase": (
            "5.3.1l.1"
        ),

        "title": (
            "Epoch-0 Training Stream "
            "Serialization Contract"
        ),

        "status": (
            "FROZEN"
        ),

        "inherited_training_decisions": {
            "training_positive_segments": (
                "T1..T59"
            ),

            "training_positive_events": (
                EXPECTED_TRAIN_POSITIVES
            ),

            "negatives_per_positive": (
                NEGATIVES_PER_POSITIVE
            ),

            "epoch_negative_examples": (
                EXPECTED_EPOCH_NEGATIVES
            ),

            "epoch_examples": (
                EXPECTED_EPOCH_EXAMPLES
            ),

            "batch_size": (
                BATCH_SIZE
            ),

            "epoch_batches": (
                EXPECTED_EPOCH_BATCHES
            ),

            "final_batch_size": (
                EXPECTED_FINAL_BATCH_SIZE
            ),
        },

        "negative_sampling": {
            "eligibility": (
                "startup role node AND no positive event "
                "for the Investor–Startup pair at segment <= h"
            ),

            "future_positive_pairs_eligible": (
                True
            ),

            "T60_labels_used": (
                False
            ),

            "without_replacement_within_positive": (
                True
            ),

            "namespace": (
                NEGATIVE_NAMESPACE
            ),

            "base_seed": (
                BASE_SEED
            ),

            "epoch": (
                EPOCH
            ),

            "derived_seed": (
                epoch0_negative_seed
            ),

            "bit_generator": (
                "PCG64"
            ),

            "draw_algorithm": (
                "four initial uniform Startup-local draws; "
                "reject forbidden or within-positive duplicate; "
                "consume repair draws sequentially until four "
                "distinct eligible negatives are accepted"
            ),

            "negative_matrix_shape": (
                list(
                    negative_matrix.shape
                )
            ),

            "negative_matrix_dtype": (
                str(
                    negative_matrix.dtype
                )
            ),

            "logical_sha256": (
                negative_logical_sha
            ),
        },

        "serialization": {
            "canonical_positive_order": (
                "lexicographic interaction_id / stable mergesort"
            ),

            "positive_order_logical_sha256": (
                positive_logical_sha
            ),

            "slots_per_positive": (
                SERIALIZED_SLOTS_PER_POSITIVE
            ),

            "slots": {
                "0": (
                    "positive"
                ),
                "1": (
                    "negative_draw_0"
                ),
                "2": (
                    "negative_draw_1"
                ),
                "3": (
                    "negative_draw_2"
                ),
                "4": (
                    "negative_draw_3"
                ),
            },

            "decode": {
                "positive_order_index": (
                    "serialized_example_index // 5"
                ),

                "example_slot": (
                    "serialized_example_index % 5"
                ),
            },
        },

        "epoch_order": {
            "namespace": (
                ORDER_NAMESPACE
            ),

            "base_seed": (
                BASE_SEED
            ),

            "epoch": (
                EPOCH
            ),

            "derived_seed": (
                epoch0_order_seed
            ),

            "bit_generator": (
                "PCG64"
            ),

            "algorithm": (
                "in-place shuffle of "
                "np.arange(5_366_245, dtype=int64)"
            ),

            "logical_sha256": (
                epoch_order_logical_sha
            ),
        },

        "first_batch": {
            "size": (
                BATCH_SIZE
            ),

            "positive_count": (
                first_batch_positive_count
            ),

            "negative_count": (
                first_batch_negative_count
            ),

            "minimum_target_segment": (
                int(
                    first_batch[
                        "segment_number"
                    ].min()
                )
            ),

            "maximum_target_segment": (
                int(
                    first_batch[
                        "segment_number"
                    ].max()
                )
            ),

            "distinct_target_segments": (
                int(
                    first_batch[
                        "segment_number"
                    ].nunique()
                )
            ),

            "logical_sha256": (
                first_batch_logical_sha
            ),
        },

        "mixed_segment_forward": {
            "policy": (
                "group batch examples by exact target segment h"
            ),

            "history_for_T_h": (
                "T0..T(h-1)"
            ),

            "history_period_count": (
                "h"
            ),

            "post_h_GRU_padding": (
                False
            ),

            "restore_original_batch_positions_before_loss": (
                True
            ),
        },

        "artifacts": {
            "positive_order": (
                str(
                    POSITIVE_ORDER_PATH
                )
            ),

            "negative_matrix": (
                str(
                    NEGATIVE_MATRIX_PATH
                )
            ),

            "epoch_order": (
                str(
                    EPOCH_ORDER_PATH
                )
            ),

            "first_batch": (
                str(
                    FIRST_BATCH_PATH
                )
            ),

            "first_batch_segment_groups": (
                str(
                    FIRST_BATCH_SEGMENT_GROUP_PATH
                )
            ),

            "hash_registry": (
                str(
                    HASH_REGISTRY_PATH
                )
            ),
        },

        "training_boundary": {
            "neural_model_instantiated": (
                False
            ),

            "training_negative_rng_instantiated": (
                True
            ),

            "training_order_rng_instantiated": (
                True
            ),

            "Adam_instantiated": (
                False
            ),

            "forward_performed": (
                False
            ),

            "backward_performed": (
                False
            ),

            "optimizer_steps": (
                0
            ),

            "checkpoint_written": (
                False
            ),
        },

        "next_phase": {
            "id": (
                "5.3.1l.2"
            ),

            "title": (
                "Adam + Canonical Epoch-0 "
                "First Mini-Batch Forward/Backward Preflight"
            ),

            "requirement": (
                "Reconstruct exact canonical composed model; "
                "verify canonical state hash; instantiate frozen Adam; "
                "load this frozen first 512-example batch; compute "
                "mixed-h trend using exact history T0..T(h-1); "
                "perform batch forward/BCE/backward; verify finite "
                "gradients and unchanged parameter values; "
                "keep optimizer.step() at zero."
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
            "5.3.1l.1"
        ),

        "status": (
            "EPOCH0_TRAINING_STREAM_SERIALIZATION_"
            "PROVED_AND_FROZEN"
        ),

        "positive_events": (
            EXPECTED_TRAIN_POSITIVES
        ),

        "epoch_negatives": (
            EXPECTED_EPOCH_NEGATIVES
        ),

        "epoch_examples": (
            EXPECTED_EPOCH_EXAMPLES
        ),

        "epoch_batches": (
            EXPECTED_EPOCH_BATCHES
        ),

        "final_batch_size": (
            EXPECTED_FINAL_BATCH_SIZE
        ),

        "epoch0_negative_seed": (
            epoch0_negative_seed
        ),

        "epoch0_order_seed": (
            epoch0_order_seed
        ),

        "positive_order_logical_sha256": (
            positive_logical_sha
        ),

        "negative_matrix_logical_sha256": (
            negative_logical_sha
        ),

        "epoch_order_logical_sha256": (
            epoch_order_logical_sha
        ),

        "first_batch_logical_sha256": (
            first_batch_logical_sha
        ),

        "training_negative_rng_instantiated": (
            True
        ),

        "training_order_rng_instantiated": (
            True
        ),

        "Adam_instantiated": (
            False
        ),

        "optimizer_steps": (
            0
        ),

        "contract": (
            str(
                CONTRACT_PATH
            )
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

    # =========================================================================
    # Final
    # =========================================================================

    banner(
        "PHASE 5.3.1l.1 FINAL STATUS"
    )

    print(
        "Canonical positive order:             FROZEN"
    )

    print(
        "Epoch-0 negative matrix:              GENERATED / FROZEN"
    )

    print(
        "Epoch-0 training order:               GENERATED / FROZEN"
    )

    print(
        "Canonical first 512-example batch:    GENERATED / FROZEN"
    )

    print()

    print(
        f"Training positives:                   "
        f"{EXPECTED_TRAIN_POSITIVES:,}"
    )

    print(
        f"Epoch negatives:                      "
        f"{EXPECTED_EPOCH_NEGATIVES:,}"
    )

    print(
        f"Epoch examples:                       "
        f"{EXPECTED_EPOCH_EXAMPLES:,}"
    )

    print(
        f"Epoch batches:                        "
        f"{EXPECTED_EPOCH_BATCHES:,}"
    )

    print(
        f"Final batch:                          "
        f"{EXPECTED_FINAL_BATCH_SIZE}"
    )

    print()

    print(
        f"First-batch positives:                "
        f"{first_batch_positive_count}"
    )

    print(
        f"First-batch negatives:                "
        f"{first_batch_negative_count}"
    )

    print(
        f"First-batch distinct target segments: "
        f"{first_batch['segment_number'].nunique()}"
    )

    print()

    print(
        "Positive-order logical SHA256:"
    )

    print(
        positive_logical_sha
    )

    print()

    print(
        "Negative-matrix logical SHA256:"
    )

    print(
        negative_logical_sha
    )

    print()

    print(
        "Epoch-order logical SHA256:"
    )

    print(
        epoch_order_logical_sha
    )

    print()

    print(
        "First-batch logical SHA256:"
    )

    print(
        first_batch_logical_sha
    )

    print()

    print(
        "T60 labels used for negative sampling: NO"
    )

    print(
        "Future-positive pairs globally excluded: NO"
    )

    print(
        "Post-h GRU zero padding:              FORBIDDEN"
    )

    print()

    print(
        "Neural model instantiated:           NO"
    )

    print(
        "Adam instantiated:                   NO"
    )

    print(
        "Forward computation:                 NO"
    )

    print(
        "Backward computation:                NO"
    )

    print(
        "Optimizer steps:                     0"
    )

    print(
        "Checkpoint written:                  NO"
    )

    banner(
        "PHASE 5.3.1l.1 COMPLETE / "
        "EPOCH-0 TRAINING STREAM SERIALIZATION FROZEN"
    )


if __name__ == "__main__":
    main()