#!/usr/bin/env python3
"""
Phase 5.3.5 — Generalized 20-Epoch Training Stream Generator Proof

Purpose
-------
Freeze one epoch-indexed production training-stream generator that can be used
for epochs 0..19.

The production stream for epoch e consists of:

    1. the single frozen canonical positive-event order
    2. a newly generated (N_positive, 4) negative-startup matrix using the
       frozen epoch-specific negative RNG seed
    3. a newly generated permutation of all 5*N_positive serialized examples
       using the frozen epoch-specific order RNG seed

Critical regression requirement
-------------------------------
The generalized generator MUST regenerate epoch 0 from scratch and reproduce
the already-frozen Phase-5.3.1l.1 artifacts exactly:

    positive-order logical SHA256
    negative-matrix logical SHA256
    epoch-order logical SHA256

It must also reproduce the frozen epoch-0 diagnostic counts:

    accepted negatives             4,292,996
    repair positions                  1,759
    forbidden-history rejections      1,743
    duplicate rejections                 16
    accepted future-positive pairs    1,264
    accepted never-positive pairs 4,291,732

Only after exact epoch-0 regression is established does this script generate
epochs 1 and 19. Those two epochs are generated twice independently and their
logical fingerprints must match exactly.

No neural model is instantiated.
No Adam optimizer is instantiated.
No forward/backward occurs.
No optimizer.step() occurs.
No validation is executed.
No test case is accessed.

Scientific negative semantics inherited unchanged
-------------------------------------------------
For positive event (o,b,h), h in T1..T59:

    eligible_negative(o,c,h) :=
        c is a Startup role node
        AND no positive event (o,c,s) exists with s <= h

Future-positive startups with first positive segment > h remain eligible.

Sampling:
    K = 4
    uniform over Startup role universe
    without replacement within one positive
    regenerated independently every epoch

RNG:
    NumPy PCG64
    base seed 42
    epoch seed =
        int.from_bytes(
            SHA256(namespace|42|epoch)[0:8],
            little,
            unsigned
        ) mod (2^63 - 1)

Namespaces:
    ITRS_PHASE5_TRAIN_NEGATIVE
    ITRS_PHASE5_TRAIN_ORDER

Implementation-equivalent repair traversal
------------------------------------------
The original epoch-0 serializer used a large vectorized initial draw followed
by repair of invalid slots. To avoid silently assuming the repair traversal,
this script reconstructs a small set of semantically equivalent candidate
repair traversals and selects the UNIQUE traversal whose regenerated epoch-0
negative matrix matches the frozen logical SHA256 exactly.

The selected traversal is then frozen and reused for epochs 1..19.
"""

from __future__ import annotations

import ast
import copy
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Frozen sources / prerequisites
# =============================================================================

ROUNDTRIP_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_2b_checkpoint_resume_roundtrip_proof.py"
)

PHASE_5_3_2A_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_2a_training_execution_state_contract.json"
)

PHASE_5_3_4_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_4_production_trainer_assembly_contract.json"
)


# =============================================================================
# Frozen stream dimensions / fingerprints
# =============================================================================

BASE_SEED = 42

NEGATIVE_NAMESPACE = (
    "ITRS_PHASE5_TRAIN_NEGATIVE"
)

ORDER_NAMESPACE = (
    "ITRS_PHASE5_TRAIN_ORDER"
)

NUM_EPOCHS = 20
NUM_INVESTORS = 165_975
NUM_STARTUPS = 311_589
NUM_HISTORY_PERIODS = 60

NEGATIVES_PER_POSITIVE = 4
SLOTS_PER_POSITIVE = 5

POSITIVE_COUNT = 1_073_249
NEGATIVE_COUNT_PER_EPOCH = 4_292_996
EXAMPLES_PER_EPOCH = 5_366_245
BATCHES_PER_EPOCH = 10_481

EXPECTED_POSITIVE_ORDER_SHA256 = (
    "73b074a80675793b811fbdc8a0609883"
    "c857fb2a687a2e01c31865ade5b509d1"
)

EXPECTED_EPOCH0_NEGATIVE_SHA256 = (
    "47015b147b1949562c0f6737a6f3a3f2"
    "d7cabd2d2202e4e57456d884a1e23fe6"
)

EXPECTED_EPOCH0_ORDER_SHA256 = (
    "0156be3ee623ade1ae696557337bfb324"
    "e9011adb7df8be9648ecb0a426c134e"
)

EXPECTED_EPOCH0_NEGATIVE_SEED = (
    7_895_109_663_985_029_800
)

EXPECTED_EPOCH0_ORDER_SEED = (
    4_607_400_055_922_019_930
)

EXPECTED_EPOCH1_NEGATIVE_SEED = (
    1_992_367_500_136_940_751
)

EXPECTED_EPOCH1_ORDER_SEED = (
    4_873_016_756_318_806_758
)

EXPECTED_EPOCH19_NEGATIVE_SEED = (
    2_216_336_465_962_932_710
)

EXPECTED_EPOCH19_ORDER_SEED = (
    3_404_391_167_791_184_776
)

EXPECTED_EPOCH_SEED_REGISTRY_SHA256 = (
    "96a4e2c52526ec7d7ca48d3d7cd1eee3"
    "893b0f8c35df9717df107a874583f956"
)

EXPECTED_EPOCH0_REPAIR_POSITIONS = 1_759
EXPECTED_EPOCH0_FORBIDDEN_REJECTIONS = 1_743
EXPECTED_EPOCH0_DUPLICATE_REJECTIONS = 16
EXPECTED_EPOCH0_FUTURE_POSITIVE_ACCEPTED = 1_264
EXPECTED_EPOCH0_NEVER_POSITIVE_ACCEPTED = 4_291_732

PREFIX_ROWS_TO_SAVE = 4_096
ORDER_PREFIX_TO_SAVE = 16_384


# =============================================================================
# Outputs
# =============================================================================

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_5"
)

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

PAIR_INDEX_AUDIT_PATH = (
    AUDIT_DIR
    / "negative_eligibility_pair_index_audit.csv"
)

EPOCH0_VARIANT_AUDIT_PATH = (
    AUDIT_DIR
    / "epoch0_repair_traversal_regression.csv"
)

EPOCH_REGISTRY_PATH = (
    AUDIT_DIR
    / "generalized_epoch_stream_registry.csv"
)

DETERMINISM_PATH = (
    AUDIT_DIR
    / "epoch1_epoch19_determinism_audit.csv"
)

EPOCH1_NEGATIVE_PREFIX_PATH = (
    AUDIT_DIR
    / "epoch_1_negative_startup_local_prefix.npy"
)

EPOCH19_NEGATIVE_PREFIX_PATH = (
    AUDIT_DIR
    / "epoch_19_negative_startup_local_prefix.npy"
)

EPOCH1_ORDER_PREFIX_PATH = (
    AUDIT_DIR
    / "epoch_1_example_order_prefix.npy"
)

EPOCH19_ORDER_PREFIX_PATH = (
    AUDIT_DIR
    / "epoch_19_example_order_prefix.npy"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_5_final_invariants.csv"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_5_generalized_stream_decision_register.csv"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_5_generalized_training_stream_generator_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_5_generalized_training_stream_generator_manifest.json"
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
        len(
            guards
        )
        == 1,
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


def derive_epoch_seed(
    namespace: str,
    epoch_index: int,
) -> int:
    require(
        0
        <= int(
            epoch_index
        )
        < NUM_EPOCHS,
        (
            "Epoch index outside 0..19."
        ),
    )

    material = (
        f"{namespace}|{BASE_SEED}|{int(epoch_index)}"
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


# =============================================================================
# Historical pair first-segment index
# =============================================================================

def build_first_positive_segment_index(
    *,
    trend_period_counts: np.ndarray,
    trend_startup_indices_global: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Build sorted encoded pair keys:

        key = investor_global * NUM_STARTUPS + startup_local

    and the first T0..T59 segment in which that pair occurs.

    This index is sufficient for the frozen negative rule:
        candidate is forbidden at target h iff first_segment <= h.
    """

    counts = np.asarray(
        trend_period_counts,
        dtype=np.int64,
    )

    startups_global = np.asarray(
        trend_startup_indices_global,
        dtype=np.int64,
    )

    require(
        counts.ndim == 1,
        (
            "Trend period counts must "
            "be one-dimensional."
        ),
    )

    require(
        len(
            counts
        )
        == (
            NUM_INVESTORS
            * NUM_HISTORY_PERIODS
        ),
        (
            "Trend period-count length drift."
        ),
    )

    require(
        int(
            counts.sum(
                dtype=np.int64
            )
        )
        == len(
            startups_global
        ),
        (
            "Trend period counts do not sum "
            "to startup-index array length."
        ),
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
                < (
                    NUM_INVESTORS
                    + NUM_STARTUPS
                )
            ).all()
        ),
        (
            "Trend startup index array contains "
            "non-Startup global node indices."
        ),
    )

    nonzero_flat = np.flatnonzero(
        counts
        > 0
    )

    repeated_flat = np.repeat(
        nonzero_flat.astype(
            np.int64,
            copy=False,
        ),
        counts[
            nonzero_flat
        ],
    )

    require(
        len(
            repeated_flat
        )
        == len(
            startups_global
        ),
        (
            "Repeated flattened period mapping "
            "length drift."
        ),
    )

    investor_global = (
        repeated_flat
        // NUM_HISTORY_PERIODS
    )

    segment_number = (
        repeated_flat
        % NUM_HISTORY_PERIODS
    ).astype(
        np.int16,
        copy=False,
    )

    startup_local = (
        startups_global
        - NUM_INVESTORS
    )

    pair_key = (
        investor_global
        * NUM_STARTUPS
        + startup_local
    ).astype(
        np.int64,
        copy=False,
    )

    # Stable sort preserves chronological occurrence order for repeated keys.
    order = np.argsort(
        pair_key,
        kind="mergesort",
    )

    sorted_key = pair_key[
        order
    ]

    sorted_segment = segment_number[
        order
    ]

    first_mask = np.empty(
        len(
            sorted_key
        ),
        dtype=bool,
    )

    first_mask[
        0
    ] = True

    first_mask[
        1:
    ] = (
        sorted_key[
            1:
        ]
        != sorted_key[
            :-1
        ]
    )

    unique_key = sorted_key[
        first_mask
    ]

    first_segment = sorted_segment[
        first_mask
    ].astype(
        np.int16,
        copy=False,
    )

    require(
        bool(
            (
                first_segment
                >= 0
            ).all()
        )
        and bool(
            (
                first_segment
                <= 59
            ).all()
        ),
        (
            "First positive segment outside T0..T59."
        ),
    )

    require(
        bool(
            (
                unique_key[
                    1:
                ]
                > unique_key[
                    :-1
                ]
            ).all()
        ),
        (
            "Encoded historical pair keys "
            "are not strictly increasing."
        ),
    )

    metadata = {
        "trend_mentions": int(
            len(
                startups_global
            )
        ),
        "unique_pre_t60_pairs": int(
            len(
                unique_key
            )
        ),
        "nonempty_investor_periods": int(
            len(
                nonzero_flat
            )
        ),
    }

    return (
        unique_key,
        first_segment,
        metadata,
    )


def lookup_first_segment(
    *,
    pair_keys_sorted: np.ndarray,
    first_segment_sorted: np.ndarray,
    investor_global: np.ndarray,
    startup_local: np.ndarray,
) -> np.ndarray:
    investor = np.asarray(
        investor_global,
        dtype=np.int64,
    )

    startup = np.asarray(
        startup_local,
        dtype=np.int64,
    )

    require(
        investor.shape
        == startup.shape,
        (
            "Pair lookup investor/startup "
            "shape mismatch."
        ),
    )

    encoded = (
        investor
        * NUM_STARTUPS
        + startup
    )

    position = np.searchsorted(
        pair_keys_sorted,
        encoded,
        side="left",
    )

    clipped = np.minimum(
        position,
        len(
            pair_keys_sorted
        )
        - 1,
    )

    found = (
        (
            position
            < len(
                pair_keys_sorted
            )
        )
        & (
            pair_keys_sorted[
                clipped
            ]
            == encoded
        )
    )

    result = np.full(
        encoded.shape,
        127,
        dtype=np.int16,
    )

    result[
        found
    ] = first_segment_sorted[
        clipped[
            found
        ]
    ]

    return result


# =============================================================================
# Initial vectorized negative draw and invalid-mask audit
# =============================================================================

def draw_initial_negative_matrix(
    seed: int,
) -> tuple[np.ndarray, dict]:
    rng = np.random.Generator(
        np.random.PCG64(
            int(
                seed
            )
        )
    )

    matrix = rng.integers(
        0,
        NUM_STARTUPS,
        size=(
            POSITIVE_COUNT,
            NEGATIVES_PER_POSITIVE,
        ),
        dtype=np.int64,
    )

    state_after_initial = copy.deepcopy(
        rng.bit_generator.state
    )

    return (
        matrix,
        state_after_initial,
    )


def compute_initial_invalid_masks(
    *,
    initial_matrix: np.ndarray,
    positive_investor: np.ndarray,
    positive_segment: np.ndarray,
    pair_keys_sorted: np.ndarray,
    first_segment_sorted: np.ndarray,
    chunk_rows: int = 250_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(
        initial_matrix.shape
        == (
            POSITIVE_COUNT,
            NEGATIVES_PER_POSITIVE,
        ),
        (
            "Initial negative matrix shape drift."
        ),
    )

    forbidden = np.zeros(
        initial_matrix.shape,
        dtype=bool,
    )

    for start in range(
        0,
        POSITIVE_COUNT,
        chunk_rows,
    ):
        end = min(
            start
            + chunk_rows,
            POSITIVE_COUNT,
        )

        investors = np.broadcast_to(
            positive_investor[
                start:end,
                None,
            ],
            (
                end - start,
                NEGATIVES_PER_POSITIVE,
            ),
        )

        startups = initial_matrix[
            start:end
        ]

        first = lookup_first_segment(
            pair_keys_sorted=(
                pair_keys_sorted
            ),
            first_segment_sorted=(
                first_segment_sorted
            ),
            investor_global=(
                investors
            ),
            startup_local=(
                startups
            ),
        )

        forbidden[
            start:end
        ] = (
            first
            <= positive_segment[
                start:end,
                None,
            ]
        )

    duplicate = np.zeros(
        initial_matrix.shape,
        dtype=bool,
    )

    for slot in range(
        1,
        NEGATIVES_PER_POSITIVE,
    ):
        duplicate[
            :,
            slot
        ] = np.any(
            initial_matrix[
                :,
                slot,
                None,
            ]
            == initial_matrix[
                :,
                :slot,
            ],
            axis=1,
        )

    invalid = (
        forbidden
        | duplicate
    )

    return (
        forbidden,
        duplicate,
        invalid,
    )


# =============================================================================
# Repair variants
# =============================================================================

def is_forbidden_scalar(
    *,
    pair_keys_sorted: np.ndarray,
    first_segment_sorted: np.ndarray,
    investor_global: int,
    target_segment: int,
    startup_local: int,
) -> bool:
    encoded = (
        int(
            investor_global
        )
        * NUM_STARTUPS
        + int(
            startup_local
        )
    )

    position = int(
        np.searchsorted(
            pair_keys_sorted,
            encoded,
            side="left",
        )
    )

    if position >= len(
        pair_keys_sorted
    ):
        return False

    if int(
        pair_keys_sorted[
            position
        ]
    ) != encoded:
        return False

    return (
        int(
            first_segment_sorted[
                position
            ]
        )
        <= int(
            target_segment
        )
    )


def candidate_duplicate(
    *,
    matrix: np.ndarray,
    row: int,
    slot: int,
    candidate: int,
    scope: str,
) -> bool:
    if scope == "prior":
        if slot == 0:
            return False

        return bool(
            np.any(
                matrix[
                    row,
                    :slot,
                ]
                == candidate
            )
        )

    if scope == "all_other":
        for other_slot in range(
            NEGATIVES_PER_POSITIVE
        ):
            if other_slot == slot:
                continue

            if int(
                matrix[
                    row,
                    other_slot
                ]
            ) == int(
                candidate
            ):
                return True

        return False

    raise AssertionError(
        f"Unknown duplicate scope: {scope}"
    )


def ordered_repair_positions(
    invalid_mask: np.ndarray,
    traversal: str,
) -> np.ndarray:
    positions = np.argwhere(
        invalid_mask
    )

    if traversal == "row_major":
        # np.argwhere is already lexicographic row,slot.
        return positions

    if traversal == "slot_major":
        order = np.lexsort(
            (
                positions[
                    :,
                    0
                ],
                positions[
                    :,
                    1
                ],
            )
        )

        return positions[
            order
        ]

    raise AssertionError(
        f"Unknown repair traversal: {traversal}"
    )


def repair_negative_matrix(
    *,
    initial_matrix: np.ndarray,
    rng_state_after_initial: dict,
    invalid_mask: np.ndarray,
    positive_investor: np.ndarray,
    positive_segment: np.ndarray,
    pair_keys_sorted: np.ndarray,
    first_segment_sorted: np.ndarray,
    traversal: str,
    duplicate_scope: str,
) -> tuple[np.ndarray, dict]:
    matrix = np.array(
        initial_matrix,
        dtype=np.int64,
        copy=True,
    )

    rng = np.random.Generator(
        np.random.PCG64()
    )

    rng.bit_generator.state = copy.deepcopy(
        rng_state_after_initial
    )

    positions = ordered_repair_positions(
        invalid_mask,
        traversal,
    )

    total_repair_rng_draws = 0
    extra_forbidden_repair_rejections = 0
    extra_duplicate_repair_rejections = 0

    for row_value, slot_value in positions:
        row = int(
            row_value
        )

        slot = int(
            slot_value
        )

        while True:
            candidate = int(
                rng.integers(
                    0,
                    NUM_STARTUPS,
                    dtype=np.int64,
                )
            )

            total_repair_rng_draws += 1

            if is_forbidden_scalar(
                pair_keys_sorted=(
                    pair_keys_sorted
                ),
                first_segment_sorted=(
                    first_segment_sorted
                ),
                investor_global=int(
                    positive_investor[
                        row
                    ]
                ),
                target_segment=int(
                    positive_segment[
                        row
                    ]
                ),
                startup_local=(
                    candidate
                ),
            ):
                extra_forbidden_repair_rejections += 1
                continue

            if candidate_duplicate(
                matrix=matrix,
                row=row,
                slot=slot,
                candidate=candidate,
                scope=duplicate_scope,
            ):
                extra_duplicate_repair_rejections += 1
                continue

            matrix[
                row,
                slot
            ] = candidate

            break

    metadata = {
        "repair_positions": int(
            len(
                positions
            )
        ),
        "repair_rng_draws": int(
            total_repair_rng_draws
        ),
        "extra_forbidden_repair_rejections": int(
            extra_forbidden_repair_rejections
        ),
        "extra_duplicate_repair_rejections": int(
            extra_duplicate_repair_rejections
        ),
    }

    return (
        matrix,
        metadata,
    )


# =============================================================================
# Final matrix semantic audit
# =============================================================================

def audit_final_negative_matrix(
    *,
    matrix: np.ndarray,
    positive_investor: np.ndarray,
    positive_segment: np.ndarray,
    pair_keys_sorted: np.ndarray,
    first_segment_sorted: np.ndarray,
    chunk_rows: int = 250_000,
) -> dict:
    require(
        matrix.shape
        == (
            POSITIVE_COUNT,
            NEGATIVES_PER_POSITIVE,
        ),
        (
            "Final negative matrix "
            "shape drift."
        ),
    )

    require(
        bool(
            (
                matrix
                >= 0
            ).all()
        )
        and bool(
            (
                matrix
                < NUM_STARTUPS
            ).all()
        ),
        (
            "Final negative matrix contains "
            "startup index outside role universe."
        ),
    )

    duplicate_rows = 0
    forbidden_count = 0
    future_positive_count = 0
    never_positive_count = 0

    for start in range(
        0,
        POSITIVE_COUNT,
        chunk_rows,
    ):
        end = min(
            start
            + chunk_rows,
            POSITIVE_COUNT,
        )

        chunk = matrix[
            start:end
        ]

        sorted_chunk = np.sort(
            chunk,
            axis=1,
        )

        duplicate_rows += int(
            np.count_nonzero(
                np.any(
                    sorted_chunk[
                        :,
                        1:
                    ]
                    == sorted_chunk[
                        :,
                        :-1
                    ],
                    axis=1,
                )
            )
        )

        investors = np.broadcast_to(
            positive_investor[
                start:end,
                None,
            ],
            chunk.shape,
        )

        first = lookup_first_segment(
            pair_keys_sorted=(
                pair_keys_sorted
            ),
            first_segment_sorted=(
                first_segment_sorted
            ),
            investor_global=(
                investors
            ),
            startup_local=(
                chunk
            ),
        )

        target = positive_segment[
            start:end,
            None,
        ]

        forbidden = (
            first
            <= target
        )

        forbidden_count += int(
            np.count_nonzero(
                forbidden
            )
        )

        future = (
            (first > target)
            & (first <= 59)
        )

        never = (
            first
            == 127
        )

        future_positive_count += int(
            np.count_nonzero(
                future
            )
        )

        never_positive_count += int(
            np.count_nonzero(
                never
            )
        )

    require(
        duplicate_rows == 0,
        (
            "Final negative matrix violates "
            "without-replacement rule."
        ),
    )

    require(
        forbidden_count == 0,
        (
            "Final negative matrix contains "
            "historically observed pair."
        ),
    )

    require(
        (
            future_positive_count
            + never_positive_count
        )
        == NEGATIVE_COUNT_PER_EPOCH,
        (
            "Accepted-negative semantic "
            "classification does not cover all slots."
        ),
    )

    return {
        "accepted_negatives": (
            NEGATIVE_COUNT_PER_EPOCH
        ),
        "duplicate_rows": (
            duplicate_rows
        ),
        "forbidden_final_slots": (
            forbidden_count
        ),
        "accepted_future_positive": (
            future_positive_count
        ),
        "accepted_never_positive": (
            never_positive_count
        ),
    }


# =============================================================================
# Generalized generator
# =============================================================================

def generate_epoch_negative_matrix(
    *,
    epoch_index: int,
    positive_investor: np.ndarray,
    positive_segment: np.ndarray,
    pair_keys_sorted: np.ndarray,
    first_segment_sorted: np.ndarray,
    selected_traversal: str,
    selected_duplicate_scope: str,
) -> tuple[np.ndarray, dict]:
    seed = derive_epoch_seed(
        NEGATIVE_NAMESPACE,
        epoch_index,
    )

    (
        initial,
        state_after_initial,
    ) = draw_initial_negative_matrix(
        seed
    )

    (
        forbidden,
        duplicate,
        invalid,
    ) = compute_initial_invalid_masks(
        initial_matrix=initial,
        positive_investor=(
            positive_investor
        ),
        positive_segment=(
            positive_segment
        ),
        pair_keys_sorted=(
            pair_keys_sorted
        ),
        first_segment_sorted=(
            first_segment_sorted
        ),
    )

    forbidden_count = int(
        np.count_nonzero(
            forbidden
        )
    )

    duplicate_count = int(
        np.count_nonzero(
            duplicate
        )
    )

    overlap_count = int(
        np.count_nonzero(
            forbidden
            & duplicate
        )
    )

    invalid_count = int(
        np.count_nonzero(
            invalid
        )
    )

    (
        final_matrix,
        repair_metadata,
    ) = repair_negative_matrix(
        initial_matrix=initial,
        rng_state_after_initial=(
            state_after_initial
        ),
        invalid_mask=invalid,
        positive_investor=(
            positive_investor
        ),
        positive_segment=(
            positive_segment
        ),
        pair_keys_sorted=(
            pair_keys_sorted
        ),
        first_segment_sorted=(
            first_segment_sorted
        ),
        traversal=(
            selected_traversal
        ),
        duplicate_scope=(
            selected_duplicate_scope
        ),
    )

    semantic = audit_final_negative_matrix(
        matrix=final_matrix,
        positive_investor=(
            positive_investor
        ),
        positive_segment=(
            positive_segment
        ),
        pair_keys_sorted=(
            pair_keys_sorted
        ),
        first_segment_sorted=(
            first_segment_sorted
        ),
    )

    metadata = {
        "epoch_index": (
            int(
                epoch_index
            )
        ),
        "negative_seed": (
            int(
                seed
            )
        ),
        "initial_forbidden_rejections": (
            forbidden_count
        ),
        "initial_duplicate_rejections": (
            duplicate_count
        ),
        "initial_overlap_forbidden_and_duplicate": (
            overlap_count
        ),
        "initial_invalid_slots": (
            invalid_count
        ),
        **repair_metadata,
        **semantic,
    }

    del initial
    del forbidden
    del duplicate
    del invalid
    gc.collect()

    return (
        final_matrix,
        metadata,
    )


def generate_epoch_order(
    epoch_index: int,
) -> tuple[np.ndarray, int]:
    seed = derive_epoch_seed(
        ORDER_NAMESPACE,
        epoch_index,
    )

    rng = np.random.Generator(
        np.random.PCG64(
            int(
                seed
            )
        )
    )

    order = rng.permutation(
        EXAMPLES_PER_EPOCH
    ).astype(
        np.int64,
        copy=False,
    )

    require(
        len(
            order
        )
        == EXAMPLES_PER_EPOCH,
        (
            "Epoch order length drift."
        ),
    )

    return (
        order,
        seed,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.5 — "
        "GENERALIZED 20-EPOCH TRAINING STREAM GENERATOR PROOF"
    )

    print(
        "Neural model instantiated:            NO"
    )
    print(
        "Adam instantiated:                    NO"
    )
    print(
        "Forward/backward:                     NO"
    )
    print(
        "optimizer.step():                     0"
    )
    print(
        "Validation executed:                  NO"
    )
    print(
        "Test accessed:                        NO"
    )

    # =========================================================================
    # Prerequisite contracts
    # =========================================================================

    banner(
        "AUTHORITATIVE STREAM-GENERATOR GATE RECHECK"
    )

    for path in (
        ROUNDTRIP_SOURCE_PATH,
        PHASE_5_3_2A_CONTRACT_PATH,
        PHASE_5_3_4_CONTRACT_PATH,
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

    execution_contract = load_json(
        PHASE_5_3_2A_CONTRACT_PATH
    )

    assembly_contract = load_json(
        PHASE_5_3_4_CONTRACT_PATH
    )

    require(
        execution_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.2a execution "
            "contract not frozen."
        ),
    )

    require(
        assembly_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.4 assembly "
            "contract not frozen."
        ),
    )

    require(
        assembly_contract[
            "launch_readiness"
        ][
            "generalized_epoch_stream_generator"
        ]
        == "REQUIRED_BEFORE_LAUNCH",
        (
            "Phase-5.3.4 no longer identifies "
            "stream generator as launch gate."
        ),
    )

    # =========================================================================
    # Load exact frozen epoch-0 stream/runtime
    # =========================================================================

    banner(
        "LOAD FROZEN EPOCH-0 STREAM / HASH RUNTIME"
    )

    runtime_2b = load_guarded_module(
        ROUNDTRIP_SOURCE_PATH,
        "_itrs_phase5_3_5_runtime2b",
    )

    preflight = (
        runtime_2b
        .load_preflight_runtime()
    )

    stream0 = (
        runtime_2b
        .load_epoch0_stream(
            preflight
        )
    )

    positive_order = (
        stream0[
            "positive_order"
        ]
    )

    frozen_negative0 = (
        stream0[
            "negative_matrix"
        ]
    )

    frozen_order0 = (
        stream0[
            "epoch_order"
        ]
    )

    require(
        len(
            positive_order
        )
        == POSITIVE_COUNT,
        (
            "Frozen positive count drift."
        ),
    )

    positive_sha = (
        preflight
        .positive_stream_logical_sha256(
            positive_order
        )
    )

    require(
        positive_sha
        == EXPECTED_POSITIVE_ORDER_SHA256,
        (
            "Canonical positive-order "
            "fingerprint drift."
        ),
    )

    require(
        preflight
        .array_logical_sha256(
            np.asarray(
                frozen_negative0
            )
        )
        == EXPECTED_EPOCH0_NEGATIVE_SHA256,
        (
            "Frozen epoch-0 negative matrix "
            "fingerprint drift."
        ),
    )

    require(
        preflight
        .array_logical_sha256(
            np.asarray(
                frozen_order0
            )
        )
        == EXPECTED_EPOCH0_ORDER_SHA256,
        (
            "Frozen epoch-0 order "
            "fingerprint drift."
        ),
    )

    required_positive_columns = {
        "interaction_id",
        "investor_global",
        "startup_local",
        "segment_number",
    }

    require(
        required_positive_columns.issubset(
            positive_order.columns
        ),
        (
            "Canonical positive-order artifact "
            "missing required columns."
        ),
    )

    positive_investor = (
        positive_order[
            "investor_global"
        ].to_numpy(
            dtype=np.int64
        )
    )

    positive_startup = (
        positive_order[
            "startup_local"
        ].to_numpy(
            dtype=np.int64
        )
    )

    positive_segment = (
        positive_order[
            "segment_number"
        ].to_numpy(
            dtype=np.int16
        )
    )

    require(
        bool(
            (
                positive_investor
                >= 0
            ).all()
        )
        and bool(
            (
                positive_investor
                < NUM_INVESTORS
            ).all()
        ),
        (
            "Positive investor index "
            "outside Investor role universe."
        ),
    )

    require(
        bool(
            (
                positive_startup
                >= 0
            ).all()
        )
        and bool(
            (
                positive_startup
                < NUM_STARTUPS
            ).all()
        ),
        (
            "Positive startup index "
            "outside Startup role universe."
        ),
    )

    require(
        bool(
            (
                positive_segment
                >= 1
            ).all()
        )
        and bool(
            (
                positive_segment
                <= 59
            ).all()
        ),
        (
            "Training positive target "
            "outside T1..T59."
        ),
    )

    # =========================================================================
    # Seed derivation regression
    # =========================================================================

    banner(
        "20-EPOCH RNG SEED DERIVATION RECHECK"
    )

    seed_rows = []

    for epoch in range(
        NUM_EPOCHS
    ):
        negative_seed = derive_epoch_seed(
            NEGATIVE_NAMESPACE,
            epoch,
        )

        order_seed = derive_epoch_seed(
            ORDER_NAMESPACE,
            epoch,
        )

        seed_rows.append(
            {
                "epoch_index": (
                    epoch
                ),
                "negative_seed": (
                    negative_seed
                ),
                "order_seed": (
                    order_seed
                ),
            }
        )

    seed_df = pd.DataFrame(
        seed_rows
    )

    require(
        int(
            seed_df.iloc[
                0
            ][
                "negative_seed"
            ]
        )
        == EXPECTED_EPOCH0_NEGATIVE_SEED,
        (
            "Epoch-0 negative seed drift."
        ),
    )

    require(
        int(
            seed_df.iloc[
                0
            ][
                "order_seed"
            ]
        )
        == EXPECTED_EPOCH0_ORDER_SEED,
        (
            "Epoch-0 order seed drift."
        ),
    )

    require(
        int(
            seed_df.iloc[
                1
            ][
                "negative_seed"
            ]
        )
        == EXPECTED_EPOCH1_NEGATIVE_SEED,
        (
            "Epoch-1 negative seed drift."
        ),
    )

    require(
        int(
            seed_df.iloc[
                1
            ][
                "order_seed"
            ]
        )
        == EXPECTED_EPOCH1_ORDER_SEED,
        (
            "Epoch-1 order seed drift."
        ),
    )

    require(
        int(
            seed_df.iloc[
                19
            ][
                "negative_seed"
            ]
        )
        == EXPECTED_EPOCH19_NEGATIVE_SEED,
        (
            "Epoch-19 negative seed drift."
        ),
    )

    require(
        int(
            seed_df.iloc[
                19
            ][
                "order_seed"
            ]
        )
        == EXPECTED_EPOCH19_ORDER_SEED,
        (
            "Epoch-19 order seed drift."
        ),
    )

    print(
        seed_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Build frozen historical first-positive index
    # =========================================================================

    banner(
        "BUILD PRE-T60 PAIR FIRST-SEGMENT ELIGIBILITY INDEX"
    )

    trend_period_counts = np.load(
        preflight.TREND_PERIOD_COUNTS_PATH,
        mmap_mode="r",
    )

    trend_startup_indices = np.load(
        preflight.TREND_STARTUP_INDICES_PATH,
        mmap_mode="r",
    )

    (
        pair_keys_sorted,
        first_segment_sorted,
        pair_index_metadata,
    ) = build_first_positive_segment_index(
        trend_period_counts=(
            trend_period_counts
        ),
        trend_startup_indices_global=(
            trend_startup_indices
        ),
    )

    # Every positive training event's focal pair must be historically observed
    # no later than its target segment.
    focal_first = lookup_first_segment(
        pair_keys_sorted=(
            pair_keys_sorted
        ),
        first_segment_sorted=(
            first_segment_sorted
        ),
        investor_global=(
            positive_investor
        ),
        startup_local=(
            positive_startup
        ),
    )

    require(
        bool(
            (
                focal_first
                <= positive_segment
            ).all()
        ),
        (
            "At least one positive training event "
            "is missing from the T0..T59 pair index."
        ),
    )

    pair_index_df = pd.DataFrame(
        [
            {
                "metric": (
                    "trend_mentions"
                ),
                "value": (
                    pair_index_metadata[
                        "trend_mentions"
                    ]
                ),
            },
            {
                "metric": (
                    "unique_pre_t60_pairs"
                ),
                "value": (
                    pair_index_metadata[
                        "unique_pre_t60_pairs"
                    ]
                ),
            },
            {
                "metric": (
                    "nonempty_investor_periods"
                ),
                "value": (
                    pair_index_metadata[
                        "nonempty_investor_periods"
                    ]
                ),
            },
            {
                "metric": (
                    "all_training_positives_found_by_h"
                ),
                "value": (
                    True
                ),
            },
        ]
    )

    print(
        pair_index_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Epoch-0 initial draw / diagnostic regression
    # =========================================================================

    banner(
        "REGENERATE EPOCH-0 INITIAL NEGATIVE DRAW"
    )

    (
        initial0,
        state_after_initial0,
    ) = draw_initial_negative_matrix(
        EXPECTED_EPOCH0_NEGATIVE_SEED
    )

    (
        forbidden0,
        duplicate0,
        invalid0,
    ) = compute_initial_invalid_masks(
        initial_matrix=(
            initial0
        ),
        positive_investor=(
            positive_investor
        ),
        positive_segment=(
            positive_segment
        ),
        pair_keys_sorted=(
            pair_keys_sorted
        ),
        first_segment_sorted=(
            first_segment_sorted
        ),
    )

    forbidden_count0 = int(
        np.count_nonzero(
            forbidden0
        )
    )

    duplicate_count0 = int(
        np.count_nonzero(
            duplicate0
        )
    )

    overlap_count0 = int(
        np.count_nonzero(
            forbidden0
            & duplicate0
        )
    )

    invalid_count0 = int(
        np.count_nonzero(
            invalid0
        )
    )

    print(
        f"Initial forbidden slots:              "
        f"{forbidden_count0:,}"
    )
    print(
        f"Initial duplicate slots:              "
        f"{duplicate_count0:,}"
    )
    print(
        f"Forbidden/duplicate overlap:          "
        f"{overlap_count0:,}"
    )
    print(
        f"Initial invalid slots:                "
        f"{invalid_count0:,}"
    )

    require(
        forbidden_count0
        == EXPECTED_EPOCH0_FORBIDDEN_REJECTIONS,
        (
            "Epoch-0 forbidden rejection "
            "count drift."
        ),
    )

    require(
        duplicate_count0
        == EXPECTED_EPOCH0_DUPLICATE_REJECTIONS,
        (
            "Epoch-0 duplicate rejection "
            "count drift."
        ),
    )

    require(
        overlap_count0
        == 0,
        (
            "Epoch-0 forbidden/duplicate "
            "rejection overlap unexpectedly nonzero."
        ),
    )

    require(
        invalid_count0
        == EXPECTED_EPOCH0_REPAIR_POSITIONS,
        (
            "Epoch-0 repair-position "
            "count drift."
        ),
    )

    # =========================================================================
    # Discover exact repair traversal from frozen epoch-0 matrix
    # =========================================================================

    banner(
        "DISCOVER UNIQUE EPOCH-0 REPAIR TRAVERSAL"
    )

    candidate_variants = [
        (
            "row_major",
            "prior",
        ),
        (
            "row_major",
            "all_other",
        ),
        (
            "slot_major",
            "prior",
        ),
        (
            "slot_major",
            "all_other",
        ),
    ]

    variant_rows = []
    matching_variants = []

    for (
        traversal,
        duplicate_scope,
    ) in candidate_variants:

        (
            candidate_matrix,
            repair_metadata,
        ) = repair_negative_matrix(
            initial_matrix=(
                initial0
            ),
            rng_state_after_initial=(
                state_after_initial0
            ),
            invalid_mask=(
                invalid0
            ),
            positive_investor=(
                positive_investor
            ),
            positive_segment=(
                positive_segment
            ),
            pair_keys_sorted=(
                pair_keys_sorted
            ),
            first_segment_sorted=(
                first_segment_sorted
            ),
            traversal=(
                traversal
            ),
            duplicate_scope=(
                duplicate_scope
            ),
        )

        candidate_sha = (
            preflight
            .array_logical_sha256(
                candidate_matrix
            )
        )

        exact_array = bool(
            np.array_equal(
                candidate_matrix,
                np.asarray(
                    frozen_negative0
                ),
            )
        )

        exact_hash = (
            candidate_sha
            == EXPECTED_EPOCH0_NEGATIVE_SHA256
        )

        require(
            exact_array
            == exact_hash,
            (
                "Epoch-0 array equality/hash "
                "agreement failed."
            ),
        )

        variant_rows.append(
            {
                "traversal": (
                    traversal
                ),
                "duplicate_scope": (
                    duplicate_scope
                ),
                "repair_positions": (
                    repair_metadata[
                        "repair_positions"
                    ]
                ),
                "repair_rng_draws": (
                    repair_metadata[
                        "repair_rng_draws"
                    ]
                ),
                "extra_forbidden_repair_rejections": (
                    repair_metadata[
                        "extra_forbidden_repair_rejections"
                    ]
                ),
                "extra_duplicate_repair_rejections": (
                    repair_metadata[
                        "extra_duplicate_repair_rejections"
                    ]
                ),
                "negative_matrix_sha256": (
                    candidate_sha
                ),
                "exact_frozen_epoch0_match": (
                    exact_hash
                ),
                "status": (
                    "MATCH"
                    if exact_hash
                    else "NO_MATCH"
                ),
            }
        )

        if exact_hash:
            matching_variants.append(
                (
                    traversal,
                    duplicate_scope,
                )
            )

        del candidate_matrix
        gc.collect()

    variant_df = pd.DataFrame(
        variant_rows
    )

    print(
        variant_df.to_string(
            index=False
        )
    )

    require(
        len(
            matching_variants
        )
        == 1,
        (
            "Expected exactly one repair traversal "
            "to reproduce frozen epoch 0; found "
            f"{matching_variants}."
        ),
    )

    (
        selected_traversal,
        selected_duplicate_scope,
    ) = matching_variants[
        0
    ]

    print()
    print(
        "SELECTED production repair traversal:"
    )
    print(
        f"  traversal:                          "
        f"{selected_traversal}"
    )
    print(
        f"  duplicate scope:                    "
        f"{selected_duplicate_scope}"
    )

    # =========================================================================
    # Re-run generalized epoch 0 through the selected production function
    # =========================================================================

    banner(
        "GENERALIZED GENERATOR -> EXACT EPOCH-0 REGRESSION"
    )

    (
        generated_negative0,
        metadata0,
    ) = generate_epoch_negative_matrix(
        epoch_index=0,
        positive_investor=(
            positive_investor
        ),
        positive_segment=(
            positive_segment
        ),
        pair_keys_sorted=(
            pair_keys_sorted
        ),
        first_segment_sorted=(
            first_segment_sorted
        ),
        selected_traversal=(
            selected_traversal
        ),
        selected_duplicate_scope=(
            selected_duplicate_scope
        ),
    )

    generated_negative0_sha = (
        preflight
        .array_logical_sha256(
            generated_negative0
        )
    )

    (
        generated_order0,
        generated_order0_seed,
    ) = generate_epoch_order(
        0
    )

    generated_order0_sha = (
        preflight
        .array_logical_sha256(
            generated_order0
        )
    )

    require(
        generated_negative0_sha
        == EXPECTED_EPOCH0_NEGATIVE_SHA256,
        (
            "Generalized epoch-0 negative "
            "matrix hash mismatch."
        ),
    )

    require(
        np.array_equal(
            generated_negative0,
            np.asarray(
                frozen_negative0
            ),
        ),
        (
            "Generalized epoch-0 negative "
            "matrix is not byte-for-byte frozen matrix."
        ),
    )

    require(
        generated_order0_sha
        == EXPECTED_EPOCH0_ORDER_SHA256,
        (
            "Generalized epoch-0 example-order "
            "hash mismatch."
        ),
    )

    require(
        np.array_equal(
            generated_order0,
            np.asarray(
                frozen_order0
            ),
        ),
        (
            "Generalized epoch-0 example order "
            "is not byte-for-byte frozen order."
        ),
    )

    require(
        metadata0[
            "initial_forbidden_rejections"
        ]
        == EXPECTED_EPOCH0_FORBIDDEN_REJECTIONS,
        (
            "Generalized epoch-0 forbidden "
            "diagnostic drift."
        ),
    )

    require(
        metadata0[
            "initial_duplicate_rejections"
        ]
        == EXPECTED_EPOCH0_DUPLICATE_REJECTIONS,
        (
            "Generalized epoch-0 duplicate "
            "diagnostic drift."
        ),
    )

    require(
        metadata0[
            "initial_invalid_slots"
        ]
        == EXPECTED_EPOCH0_REPAIR_POSITIONS,
        (
            "Generalized epoch-0 repair "
            "diagnostic drift."
        ),
    )

    require(
        metadata0[
            "accepted_future_positive"
        ]
        == EXPECTED_EPOCH0_FUTURE_POSITIVE_ACCEPTED,
        (
            "Generalized epoch-0 future-positive "
            "accepted count drift."
        ),
    )

    require(
        metadata0[
            "accepted_never_positive"
        ]
        == EXPECTED_EPOCH0_NEVER_POSITIVE_ACCEPTED,
        (
            "Generalized epoch-0 never-positive "
            "accepted count drift."
        ),
    )

    print(
        "Positive order SHA256:"
    )
    print(
        positive_sha
    )
    print()
    print(
        "Epoch-0 negative SHA256:"
    )
    print(
        generated_negative0_sha
    )
    print()
    print(
        "Epoch-0 order SHA256:"
    )
    print(
        generated_order0_sha
    )
    print()
    print(
        "Epoch-0 matrix equality:              EXACT"
    )
    print(
        "Epoch-0 order equality:               EXACT"
    )

    del generated_negative0
    del generated_order0
    del initial0
    del forbidden0
    del duplicate0
    del invalid0
    gc.collect()

    # =========================================================================
    # Epochs 1 and 19 deterministic generation proof
    # =========================================================================

    banner(
        "EPOCH-1 / EPOCH-19 INDEPENDENT DETERMINISM PROOF"
    )

    epoch_registry_rows = [
        {
            "epoch_index": (
                0
            ),
            "negative_seed": (
                EXPECTED_EPOCH0_NEGATIVE_SEED
            ),
            "order_seed": (
                EXPECTED_EPOCH0_ORDER_SEED
            ),
            "positive_order_sha256": (
                positive_sha
            ),
            "negative_matrix_sha256": (
                EXPECTED_EPOCH0_NEGATIVE_SHA256
            ),
            "epoch_order_sha256": (
                EXPECTED_EPOCH0_ORDER_SHA256
            ),
            "repair_positions": (
                metadata0[
                    "initial_invalid_slots"
                ]
            ),
            "accepted_future_positive": (
                metadata0[
                    "accepted_future_positive"
                ]
            ),
            "accepted_never_positive": (
                metadata0[
                    "accepted_never_positive"
                ]
            ),
            "generation_status": (
                "EXACT_FROZEN_REGRESSION"
            ),
        }
    ]

    determinism_rows = []

    saved_prefixes = {}

    for epoch in (
        1,
        19,
    ):
        (
            negative_a,
            metadata_a,
        ) = generate_epoch_negative_matrix(
            epoch_index=(
                epoch
            ),
            positive_investor=(
                positive_investor
            ),
            positive_segment=(
                positive_segment
            ),
            pair_keys_sorted=(
                pair_keys_sorted
            ),
            first_segment_sorted=(
                first_segment_sorted
            ),
            selected_traversal=(
                selected_traversal
            ),
            selected_duplicate_scope=(
                selected_duplicate_scope
            ),
        )

        (
            order_a,
            order_seed_a,
        ) = generate_epoch_order(
            epoch
        )

        negative_sha_a = (
            preflight
            .array_logical_sha256(
                negative_a
            )
        )

        order_sha_a = (
            preflight
            .array_logical_sha256(
                order_a
            )
        )

        negative_prefix = np.array(
            negative_a[
                :PREFIX_ROWS_TO_SAVE
            ],
            dtype=np.int64,
            copy=True,
        )

        order_prefix = np.array(
            order_a[
                :ORDER_PREFIX_TO_SAVE
            ],
            dtype=np.int64,
            copy=True,
        )

        del negative_a
        del order_a
        gc.collect()

        (
            negative_b,
            metadata_b,
        ) = generate_epoch_negative_matrix(
            epoch_index=(
                epoch
            ),
            positive_investor=(
                positive_investor
            ),
            positive_segment=(
                positive_segment
            ),
            pair_keys_sorted=(
                pair_keys_sorted
            ),
            first_segment_sorted=(
                first_segment_sorted
            ),
            selected_traversal=(
                selected_traversal
            ),
            selected_duplicate_scope=(
                selected_duplicate_scope
            ),
        )

        (
            order_b,
            order_seed_b,
        ) = generate_epoch_order(
            epoch
        )

        negative_sha_b = (
            preflight
            .array_logical_sha256(
                negative_b
            )
        )

        order_sha_b = (
            preflight
            .array_logical_sha256(
                order_b
            )
        )

        require(
            negative_sha_a
            == negative_sha_b,
            (
                f"Epoch {epoch} negative "
                "regeneration is nondeterministic."
            ),
        )

        require(
            order_sha_a
            == order_sha_b,
            (
                f"Epoch {epoch} order "
                "regeneration is nondeterministic."
            ),
        )

        require(
            metadata_a
            == metadata_b,
            (
                f"Epoch {epoch} negative "
                "metadata is nondeterministic."
            ),
        )

        require(
            order_seed_a
            == order_seed_b,
            (
                f"Epoch {epoch} order seed "
                "regeneration drift."
            ),
        )

        require(
            np.array_equal(
                negative_prefix,
                negative_b[
                    :PREFIX_ROWS_TO_SAVE
                ],
            ),
            (
                f"Epoch {epoch} negative prefix "
                "regeneration mismatch."
            ),
        )

        require(
            np.array_equal(
                order_prefix,
                order_b[
                    :ORDER_PREFIX_TO_SAVE
                ],
            ),
            (
                f"Epoch {epoch} order prefix "
                "regeneration mismatch."
            ),
        )

        determinism_rows.append(
            {
                "epoch_index": (
                    epoch
                ),
                "negative_sha_run_A": (
                    negative_sha_a
                ),
                "negative_sha_run_B": (
                    negative_sha_b
                ),
                "negative_exact": (
                    True
                ),
                "order_sha_run_A": (
                    order_sha_a
                ),
                "order_sha_run_B": (
                    order_sha_b
                ),
                "order_exact": (
                    True
                ),
                "metadata_exact": (
                    True
                ),
                "status": (
                    "PASS"
                ),
            }
        )

        epoch_registry_rows.append(
            {
                "epoch_index": (
                    epoch
                ),
                "negative_seed": int(
                    metadata_a[
                        "negative_seed"
                    ]
                ),
                "order_seed": int(
                    order_seed_a
                ),
                "positive_order_sha256": (
                    positive_sha
                ),
                "negative_matrix_sha256": (
                    negative_sha_a
                ),
                "epoch_order_sha256": (
                    order_sha_a
                ),
                "repair_positions": int(
                    metadata_a[
                        "initial_invalid_slots"
                    ]
                ),
                "accepted_future_positive": int(
                    metadata_a[
                        "accepted_future_positive"
                    ]
                ),
                "accepted_never_positive": int(
                    metadata_a[
                        "accepted_never_positive"
                    ]
                ),
                "generation_status": (
                    "DOUBLE_REGENERATION_EXACT"
                ),
            }
        )

        saved_prefixes[
            epoch
        ] = {
            "negative": (
                negative_prefix
            ),
            "order": (
                order_prefix
            ),
        }

        del negative_b
        del order_b
        gc.collect()

    determinism_df = pd.DataFrame(
        determinism_rows
    )

    epoch_registry_df = pd.DataFrame(
        epoch_registry_rows
    ).sort_values(
        [
            "epoch_index",
        ]
    ).reset_index(
        drop=True
    )

    print(
        determinism_df.to_string(
            index=False
        )
    )

    print()
    print(
        "Frozen generated stream fingerprints:"
    )
    print(
        epoch_registry_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Distinct-epoch sanity
    # =========================================================================

    banner(
        "DISTINCT-EPOCH STREAM SANITY"
    )

    epoch0_row = (
        epoch_registry_df.loc[
            epoch_registry_df[
                "epoch_index"
            ]
            == 0
        ].iloc[
            0
        ]
    )

    epoch1_row = (
        epoch_registry_df.loc[
            epoch_registry_df[
                "epoch_index"
            ]
            == 1
        ].iloc[
            0
        ]
    )

    epoch19_row = (
        epoch_registry_df.loc[
            epoch_registry_df[
                "epoch_index"
            ]
            == 19
        ].iloc[
            0
        ]
    )

    require(
        len(
            {
                epoch0_row[
                    "negative_matrix_sha256"
                ],
                epoch1_row[
                    "negative_matrix_sha256"
                ],
                epoch19_row[
                    "negative_matrix_sha256"
                ],
            }
        )
        == 3,
        (
            "Epoch-specific negative matrices "
            "are not distinct."
        ),
    )

    require(
        len(
            {
                epoch0_row[
                    "epoch_order_sha256"
                ],
                epoch1_row[
                    "epoch_order_sha256"
                ],
                epoch19_row[
                    "epoch_order_sha256"
                ],
            }
        )
        == 3,
        (
            "Epoch-specific example orders "
            "are not distinct."
        ),
    )

    print(
        "Epoch-0/1/19 negative hashes distinct: YES"
    )
    print(
        "Epoch-0/1/19 order hashes distinct:    YES"
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.5 INVARIANTS"
    )

    checks = [
        (
            "phase_5_3_2a_contract_frozen",
            (
                execution_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),
        (
            "phase_5_3_4_contract_frozen",
            (
                assembly_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),
        (
            "canonical_positive_order_sha_exact",
            (
                positive_sha
                == EXPECTED_POSITIVE_ORDER_SHA256
            ),
        ),
        (
            "epoch0_negative_seed_exact",
            (
                derive_epoch_seed(
                    NEGATIVE_NAMESPACE,
                    0,
                )
                == EXPECTED_EPOCH0_NEGATIVE_SEED
            ),
        ),
        (
            "epoch0_order_seed_exact",
            (
                derive_epoch_seed(
                    ORDER_NAMESPACE,
                    0,
                )
                == EXPECTED_EPOCH0_ORDER_SEED
            ),
        ),
        (
            "epoch1_negative_seed_exact",
            (
                derive_epoch_seed(
                    NEGATIVE_NAMESPACE,
                    1,
                )
                == EXPECTED_EPOCH1_NEGATIVE_SEED
            ),
        ),
        (
            "epoch19_negative_seed_exact",
            (
                derive_epoch_seed(
                    NEGATIVE_NAMESPACE,
                    19,
                )
                == EXPECTED_EPOCH19_NEGATIVE_SEED
            ),
        ),
        (
            "epoch0_forbidden_rejections_1743",
            (
                metadata0[
                    "initial_forbidden_rejections"
                ]
                == 1743
            ),
        ),
        (
            "epoch0_duplicate_rejections_16",
            (
                metadata0[
                    "initial_duplicate_rejections"
                ]
                == 16
            ),
        ),
        (
            "epoch0_repair_positions_1759",
            (
                metadata0[
                    "initial_invalid_slots"
                ]
                == 1759
            ),
        ),
        (
            "epoch0_future_positive_accepted_1264",
            (
                metadata0[
                    "accepted_future_positive"
                ]
                == 1264
            ),
        ),
        (
            "epoch0_never_positive_accepted_4291732",
            (
                metadata0[
                    "accepted_never_positive"
                ]
                == 4_291_732
            ),
        ),
        (
            "unique_repair_traversal_discovered",
            (
                len(
                    matching_variants
                )
                == 1
            ),
        ),
        (
            "generalized_epoch0_negative_hash_exact",
            (
                generated_negative0_sha
                == EXPECTED_EPOCH0_NEGATIVE_SHA256
            ),
        ),
        (
            "generalized_epoch0_order_hash_exact",
            (
                generated_order0_sha
                == EXPECTED_EPOCH0_ORDER_SHA256
            ),
        ),
        (
            "generalized_epoch0_negative_array_exact",
            True,
        ),
        (
            "generalized_epoch0_order_array_exact",
            True,
        ),
        (
            "epoch1_negative_double_regeneration_exact",
            bool(
                determinism_df.loc[
                    determinism_df[
                        "epoch_index"
                    ]
                    == 1,
                    "negative_exact",
                ].iloc[
                    0
                ]
            ),
        ),
        (
            "epoch1_order_double_regeneration_exact",
            bool(
                determinism_df.loc[
                    determinism_df[
                        "epoch_index"
                    ]
                    == 1,
                    "order_exact",
                ].iloc[
                    0
                ]
            ),
        ),
        (
            "epoch19_negative_double_regeneration_exact",
            bool(
                determinism_df.loc[
                    determinism_df[
                        "epoch_index"
                    ]
                    == 19,
                    "negative_exact",
                ].iloc[
                    0
                ]
            ),
        ),
        (
            "epoch19_order_double_regeneration_exact",
            bool(
                determinism_df.loc[
                    determinism_df[
                        "epoch_index"
                    ]
                    == 19,
                    "order_exact",
                ].iloc[
                    0
                ]
            ),
        ),
        (
            "epoch0_1_19_negative_streams_distinct",
            (
                len(
                    {
                        epoch0_row[
                            "negative_matrix_sha256"
                        ],
                        epoch1_row[
                            "negative_matrix_sha256"
                        ],
                        epoch19_row[
                            "negative_matrix_sha256"
                        ],
                    }
                )
                == 3
            ),
        ),
        (
            "epoch0_1_19_order_streams_distinct",
            (
                len(
                    {
                        epoch0_row[
                            "epoch_order_sha256"
                        ],
                        epoch1_row[
                            "epoch_order_sha256"
                        ],
                        epoch19_row[
                            "epoch_order_sha256"
                        ],
                    }
                )
                == 3
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
            "forward_backward_not_executed",
            True,
        ),
        (
            "optimizer_steps_zero",
            True,
        ),
        (
            "validation_not_executed",
            True,
        ),
        (
            "test_not_accessed",
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
        bool(
            (
                invariant_df[
                    "result"
                ]
                == "PASS"
            ).all()
        ),
        (
            "At least one Phase-5.3.5 "
            "generalized-stream invariant failed."
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
        "WRITE PHASE-5.3.5 OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_index_df.to_csv(
        PAIR_INDEX_AUDIT_PATH,
        index=False,
    )

    variant_df.to_csv(
        EPOCH0_VARIANT_AUDIT_PATH,
        index=False,
    )

    epoch_registry_df.to_csv(
        EPOCH_REGISTRY_PATH,
        index=False,
    )

    determinism_df.to_csv(
        DETERMINISM_PATH,
        index=False,
    )

    np.save(
        EPOCH1_NEGATIVE_PREFIX_PATH,
        saved_prefixes[
            1
        ][
            "negative"
        ],
        allow_pickle=False,
    )

    np.save(
        EPOCH19_NEGATIVE_PREFIX_PATH,
        saved_prefixes[
            19
        ][
            "negative"
        ],
        allow_pickle=False,
    )

    np.save(
        EPOCH1_ORDER_PREFIX_PATH,
        saved_prefixes[
            1
        ][
            "order"
        ],
        allow_pickle=False,
    )

    np.save(
        EPOCH19_ORDER_PREFIX_PATH,
        saved_prefixes[
            19
        ][
            "order"
        ],
        allow_pickle=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    decision_df = pd.DataFrame(
        [
            {
                "decision": (
                    "production_epoch_stream_generator"
                ),
                "value": (
                    "ONE_EPOCH_INDEXED_GENERATOR_FOR_EPOCHS_0_TO_19"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_5"
                ),
            },
            {
                "decision": (
                    "positive_event_order"
                ),
                "value": (
                    "SINGLE_FROZEN_CANONICAL_ORDER_SHARED_BY_ALL_EPOCHS"
                ),
                "classification": (
                    "INHERITED_FROZEN_PHASE_5_3_1l_1"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_5"
                ),
            },
            {
                "decision": (
                    "negative_initial_draw"
                ),
                "value": (
                    "PCG64_VECTOR_INTEGERS_N_POS_BY_4"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE_EXACT_EPOCH0_REGRESSION"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_5"
                ),
            },
            {
                "decision": (
                    "negative_repair_traversal"
                ),
                "value": (
                    f"{selected_traversal.upper()}_"
                    f"DUPLICATE_SCOPE_{selected_duplicate_scope.upper()}"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE_EXACT_EPOCH0_REGRESSION"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_5"
                ),
            },
            {
                "decision": (
                    "epoch_order_generation"
                ),
                "value": (
                    "PCG64_PERMUTATION_5366245"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE_EXACT_EPOCH0_REGRESSION"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_5"
                ),
            },
            {
                "decision": (
                    "future_positive_negative_semantics"
                ),
                "value": (
                    "FIRST_POSITIVE_SEGMENT_GREATER_THAN_H_IS_ELIGIBLE"
                ),
                "classification": (
                    "INHERITED_FROZEN_PHASE_5_1_1b"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_5"
                ),
            },
            {
                "decision": (
                    "launch_gate"
                ),
                "value": (
                    "GENERALIZED_EPOCH0_EXACT_REGRESSION_REQUIRED_BEFORE_LAUNCH"
                ),
                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),
                "status": (
                    "FROZEN_PHASE_5_3_5"
                ),
            },
        ]
    )

    decision_df.to_csv(
        DECISION_REGISTER_PATH,
        index=False,
    )

    epoch1 = (
        epoch_registry_df.loc[
            epoch_registry_df[
                "epoch_index"
            ]
            == 1
        ].iloc[
            0
        ]
    )

    epoch19 = (
        epoch_registry_df.loc[
            epoch_registry_df[
                "epoch_index"
            ]
            == 19
        ].iloc[
            0
        ]
    )

    contract = {
        "phase": (
            "5.3.5"
        ),
        "title": (
            "Generalized 20-Epoch Training Stream Generator Contract"
        ),
        "status": (
            "FROZEN"
        ),
        "positive_stream": {
            "rows": (
                POSITIVE_COUNT
            ),
            "logical_sha256": (
                positive_sha
            ),
            "shared_across_epochs": (
                True
            ),
        },
        "negative_generator": {
            "K": (
                NEGATIVES_PER_POSITIVE
            ),
            "startup_universe": (
                NUM_STARTUPS
            ),
            "RNG": (
                "NumPy PCG64"
            ),
            "namespace": (
                NEGATIVE_NAMESPACE
            ),
            "initial_draw_shape": [
                POSITIVE_COUNT,
                NEGATIVES_PER_POSITIVE,
            ],
            "repair_traversal": (
                selected_traversal
            ),
            "duplicate_scope": (
                selected_duplicate_scope
            ),
            "eligibility": (
                "candidate forbidden iff first positive segment <= target h"
            ),
            "future_positive_first_segment_gt_h_eligible": (
                True
            ),
        },
        "order_generator": {
            "RNG": (
                "NumPy PCG64"
            ),
            "namespace": (
                ORDER_NAMESPACE
            ),
            "serialized_examples": (
                EXAMPLES_PER_EPOCH
            ),
            "operation": (
                "permutation"
            ),
        },
        "epoch0_exact_regression": {
            "negative_seed": (
                EXPECTED_EPOCH0_NEGATIVE_SEED
            ),
            "order_seed": (
                EXPECTED_EPOCH0_ORDER_SEED
            ),
            "negative_sha256": (
                generated_negative0_sha
            ),
            "order_sha256": (
                generated_order0_sha
            ),
            "repair_positions": (
                metadata0[
                    "initial_invalid_slots"
                ]
            ),
            "forbidden_rejections": (
                metadata0[
                    "initial_forbidden_rejections"
                ]
            ),
            "duplicate_rejections": (
                metadata0[
                    "initial_duplicate_rejections"
                ]
            ),
            "future_positive_accepted": (
                metadata0[
                    "accepted_future_positive"
                ]
            ),
            "never_positive_accepted": (
                metadata0[
                    "accepted_never_positive"
                ]
            ),
            "byte_exact": (
                True
            ),
        },
        "epoch1_frozen_fingerprint": {
            "negative_seed": int(
                epoch1[
                    "negative_seed"
                ]
            ),
            "order_seed": int(
                epoch1[
                    "order_seed"
                ]
            ),
            "negative_sha256": str(
                epoch1[
                    "negative_matrix_sha256"
                ]
            ),
            "order_sha256": str(
                epoch1[
                    "epoch_order_sha256"
                ]
            ),
            "repair_positions": int(
                epoch1[
                    "repair_positions"
                ]
            ),
            "future_positive_accepted": int(
                epoch1[
                    "accepted_future_positive"
                ]
            ),
            "never_positive_accepted": int(
                epoch1[
                    "accepted_never_positive"
                ]
            ),
            "double_regeneration_exact": (
                True
            ),
        },
        "epoch19_frozen_fingerprint": {
            "negative_seed": int(
                epoch19[
                    "negative_seed"
                ]
            ),
            "order_seed": int(
                epoch19[
                    "order_seed"
                ]
            ),
            "negative_sha256": str(
                epoch19[
                    "negative_matrix_sha256"
                ]
            ),
            "order_sha256": str(
                epoch19[
                    "epoch_order_sha256"
                ]
            ),
            "repair_positions": int(
                epoch19[
                    "repair_positions"
                ]
            ),
            "future_positive_accepted": int(
                epoch19[
                    "accepted_future_positive"
                ]
            ),
            "never_positive_accepted": int(
                epoch19[
                    "accepted_never_positive"
                ]
            ),
            "double_regeneration_exact": (
                True
            ),
        },
        "seed_registry": {
            "epochs": (
                NUM_EPOCHS
            ),
            "logical_sha256": (
                EXPECTED_EPOCH_SEED_REGISTRY_SHA256
            ),
        },
        "boundary": {
            "model_instantiated": (
                False
            ),
            "Adam_instantiated": (
                False
            ),
            "forward_backward": (
                False
            ),
            "optimizer_steps": (
                0
            ),
            "validation_executed": (
                False
            ),
            "test_accessed": (
                False
            ),
        },
        "next_phase": {
            "id": (
                "5.3.6"
            ),
            "title": (
                "Launch-Ready 20-Epoch Production Driver Freeze"
            ),
            "requirement": (
                "Assemble the now-proven generalized epoch stream generator "
                "with the frozen production training runtime, checkpoint/"
                "resume controller, full validation runtime, best-checkpoint "
                "selection, and final test guard. Freeze the exact executable "
                "20-epoch driver and perform a no-training launch-contract "
                "audit before the actual Phase-5.4 training execution."
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
            "5.3.5"
        ),
        "status": (
            "GENERALIZED_20_EPOCH_TRAINING_STREAM_"
            "GENERATOR_PROVED_AND_FROZEN"
        ),
        "positive_order_sha256": (
            positive_sha
        ),
        "selected_repair_traversal": (
            selected_traversal
        ),
        "selected_duplicate_scope": (
            selected_duplicate_scope
        ),
        "epoch0_negative_sha256": (
            generated_negative0_sha
        ),
        "epoch0_order_sha256": (
            generated_order0_sha
        ),
        "epoch1_negative_sha256": str(
            epoch1[
                "negative_matrix_sha256"
            ]
        ),
        "epoch1_order_sha256": str(
            epoch1[
                "epoch_order_sha256"
            ]
        ),
        "epoch19_negative_sha256": str(
            epoch19[
                "negative_matrix_sha256"
            ]
        ),
        "epoch19_order_sha256": str(
            epoch19[
                "epoch_order_sha256"
            ]
        ),
        "model_instantiated": (
            False
        ),
        "optimizer_steps": (
            0
        ),
        "test_accessed": (
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
        PAIR_INDEX_AUDIT_PATH,
        EPOCH0_VARIANT_AUDIT_PATH,
        EPOCH_REGISTRY_PATH,
        DETERMINISM_PATH,
        EPOCH1_NEGATIVE_PREFIX_PATH,
        EPOCH19_NEGATIVE_PREFIX_PATH,
        EPOCH1_ORDER_PREFIX_PATH,
        EPOCH19_ORDER_PREFIX_PATH,
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
        "PHASE 5.3.5 FINAL STATUS"
    )

    print(
        "Canonical positive-order SHA256:"
    )
    print(
        positive_sha
    )
    print()

    print(
        "Selected repair traversal:"
    )
    print(
        f"  traversal:                          "
        f"{selected_traversal}"
    )
    print(
        f"  duplicate scope:                    "
        f"{selected_duplicate_scope}"
    )
    print()

    print(
        "Epoch 0 generalized regression:"
    )
    print(
        "  negative matrix:                    BYTE-EXACT"
    )
    print(
        "  example order:                      BYTE-EXACT"
    )
    print(
        "  repair positions:                   1,759"
    )
    print(
        "  forbidden rejections:               1,743"
    )
    print(
        "  duplicate rejections:               16"
    )
    print(
        "  accepted future-positive:           1,264"
    )
    print()

    print(
        "Epoch 1 frozen stream:"
    )
    print(
        f"  negative SHA256:                    "
        f"{epoch1['negative_matrix_sha256']}"
    )
    print(
        f"  order SHA256:                       "
        f"{epoch1['epoch_order_sha256']}"
    )
    print(
        "  independent regeneration:           EXACT"
    )
    print()

    print(
        "Epoch 19 frozen stream:"
    )
    print(
        f"  negative SHA256:                    "
        f"{epoch19['negative_matrix_sha256']}"
    )
    print(
        f"  order SHA256:                       "
        f"{epoch19['epoch_order_sha256']}"
    )
    print(
        "  independent regeneration:           EXACT"
    )
    print()

    print(
        "Neural model instantiated:            NO"
    )
    print(
        "Adam instantiated:                    NO"
    )
    print(
        "optimizer.step():                     0"
    )
    print(
        "Validation executed:                  NO"
    )
    print(
        "Test accessed:                        NO"
    )

    banner(
        "PHASE 5.3.5 COMPLETE / "
        "GENERALIZED 20-EPOCH TRAINING STREAM GENERATOR PROVED AND FROZEN"
    )


if __name__ == "__main__":
    main()