#!/usr/bin/env python3
"""
Phase 5.3.5 — Generalized 20-Epoch Training Stream Generator Proof V5

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
    repair / rejected RNG draws       1,759
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

Exact sequential RNG-consumption reconstruction
-------------------------------------------
The original Phase-5.3.1l.1 runtime generated negatives positive-by-positive.
For each positive event and each of its four negative slots, one scalar
candidate was drawn from the epoch PCG64 stream. Historically forbidden or
within-positive duplicate candidates were rejected immediately, so every
rejection advanced the SAME RNG before the next candidate/positive.

The 1,759 recorded "Repair RNG draws" are therefore rejected scalar draws:
    1,743 forbidden-history rejections
       16 within-positive duplicate rejections

There is no vectorized initial (N,4) draw and no separate repair pass.

Authoritative eligibility-source reconstruction
-----------------------------------------------
Phase-5.3.1l.1 built the historical eligibility lookup directly from the
audited Phase-2 T0..T59 temporal interaction table after role-index mapping.
This script therefore reconstructs first-positive segments from that same
Phase-2 table and Phase-3 role-node registry. Phase-4 trend-runtime arrays are
used only as a diagnostic cross-check and are NOT the authoritative sampling
lookup.

Phase-5.3.5a additionally proved that the frozen epoch-0 training-negative
matrix is C-contiguous int32 (<i4), while the V4 generalized matrix contained
the same values but was int64. V5 therefore freezes int32 as part of the
production stream representation contract.

The generalized generator must reproduce the frozen 10,000-positive prefix,
the complete epoch-0 negative VALUES, the int32 representation, and the exact
logical SHA256 before it is trusted for epochs 1..19. During epoch-0
regeneration, the frozen matrix is compared row-by-row so any value divergence
fails at the first mismatching positive event.
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

TEMPORAL_INTERACTIONS_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "interactions_itrs_temporal_split.parquet"
)

NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/model_ready/"
    "node_index.parquet"
)

PHASE_5_3_2A_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_2a_training_execution_state_contract.json"
)

PHASE_5_3_4_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_4_production_trainer_assembly_contract.json"
)

PHASE_5_3_5A_REPRESENTATION_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_5a_epoch0_stream_representation_audit.json"
)

FROZEN_NEGATIVE_DTYPE = np.dtype(np.int32)
FROZEN_NEGATIVE_DTYPE_STR = "<i4"

EXPECTED_V4_INT64_NEGATIVE_SHA256 = (
    "de40bb466ff979a382b23f13d6cff404"
    "57320dad833cbcd4c1d851c3cbe21d2d"
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
    / "epoch0_sequential_sampler_regression.csv"
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

def build_first_positive_segment_index_from_phase2(
    *,
    temporal_path: Path,
    node_index_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Reconstruct the authoritative Phase-5.3.1l.1 negative-eligibility lookup
    directly from Phase-2 T0..T59 interactions and the frozen Phase-3 role
    registry.

    IMPORTANT
    ---------
    Phase 3 froze role identity by NUMERIC node-index slices:

        Investor: 0 .. 165,974
        Startup:  165,975 .. 477,563

    Those numeric slices are authoritative here.

    The node_type text column is retained only as a diagnostic because literal
    label spellings/casing are not allowed to redefine a role boundary that
    Phase 3 already froze.

    Pair key:
        investor_global * NUM_STARTUPS + startup_local

    Stored value:
        earliest segment_number (T0..T59) for the pair.
    """

    require(
        temporal_path.exists(),
        f"Missing Phase-2 temporal interactions: {temporal_path}",
    )

    require(
        node_index_path.exists(),
        f"Missing Phase-3 node index: {node_index_path}",
    )

    # -------------------------------------------------------------------------
    # Authoritative Phase-2 T0..T59 event universe
    # -------------------------------------------------------------------------

    temporal = pd.read_parquet(
        temporal_path,
        columns=[
            "investor_id",
            "startup_id",
            "segment_number",
        ],
    )

    segment_numeric = pd.to_numeric(
        temporal[
            "segment_number"
        ],
        errors="coerce",
    )

    mask = (
        segment_numeric.notna()
        & (
            segment_numeric
            >= 0
        )
        & (
            segment_numeric
            <= 59
        )
    )

    historical = temporal.loc[
        mask,
        [
            "investor_id",
            "startup_id",
            "segment_number",
        ],
    ].copy()

    historical[
        "segment_number"
    ] = pd.to_numeric(
        historical[
            "segment_number"
        ],
        errors="raise",
    ).astype(
        np.int16
    )

    require(
        len(
            historical
        )
        == 1_173_422,
        (
            "Phase-2 T0..T59 historical-event count drift: "
            f"{len(historical):,} != 1,173,422."
        ),
    )

    # -------------------------------------------------------------------------
    # Authoritative Phase-3 role-node universe
    # -------------------------------------------------------------------------

    node_index = pd.read_parquet(
        node_index_path,
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
        == (
            NUM_INVESTORS
            + NUM_STARTUPS
        ),
        (
            "Phase-3 role-node total-count drift: "
            f"{len(node_index):,} != "
            f"{NUM_INVESTORS + NUM_STARTUPS:,}."
        ),
    )

    require(
        bool(
            node_index[
                "node_index"
            ].notna().all()
        ),
        (
            "Phase-3 node_index contains nulls."
        ),
    )

    node_index_numeric = pd.to_numeric(
        node_index[
            "node_index"
        ],
        errors="raise",
    ).astype(
        np.int64
    )

    require(
        bool(
            node_index_numeric.is_unique
        ),
        (
            "Phase-3 node_index is not unique."
        ),
    )

    sorted_numeric = np.sort(
        node_index_numeric.to_numpy(
            dtype=np.int64
        )
    )

    expected_numeric = np.arange(
        NUM_INVESTORS
        + NUM_STARTUPS,
        dtype=np.int64,
    )

    require(
        bool(
            np.array_equal(
                sorted_numeric,
                expected_numeric,
            )
        ),
        (
            "Phase-3 node_index is not the frozen contiguous "
            "0..477563 role-node universe."
        ),
    )

    node_index = node_index.copy()

    node_index[
        "__node_index_numeric"
    ] = node_index_numeric

    investor_nodes = node_index.loc[
        (
            node_index[
                "__node_index_numeric"
            ]
            >= 0
        )
        & (
            node_index[
                "__node_index_numeric"
            ]
            < NUM_INVESTORS
        ),
        [
            "raw_entity_id",
            "node_type",
            "__node_index_numeric",
        ],
    ].copy()

    startup_nodes = node_index.loc[
        (
            node_index[
                "__node_index_numeric"
            ]
            >= NUM_INVESTORS
        )
        & (
            node_index[
                "__node_index_numeric"
            ]
            < (
                NUM_INVESTORS
                + NUM_STARTUPS
            )
        ),
        [
            "raw_entity_id",
            "node_type",
            "__node_index_numeric",
        ],
    ].copy()

    require(
        len(
            investor_nodes
        )
        == NUM_INVESTORS,
        (
            "Frozen Investor numeric-slice count drift: "
            f"{len(investor_nodes):,} != {NUM_INVESTORS:,}."
        ),
    )

    require(
        len(
            startup_nodes
        )
        == NUM_STARTUPS,
        (
            "Frozen Startup numeric-slice count drift: "
            f"{len(startup_nodes):,} != {NUM_STARTUPS:,}."
        ),
    )

    require(
        int(
            investor_nodes[
                "__node_index_numeric"
            ].min()
        )
        == 0,
        (
            "Investor slice minimum node_index drift."
        ),
    )

    require(
        int(
            investor_nodes[
                "__node_index_numeric"
            ].max()
        )
        == (
            NUM_INVESTORS
            - 1
        ),
        (
            "Investor slice maximum node_index drift."
        ),
    )

    require(
        int(
            startup_nodes[
                "__node_index_numeric"
            ].min()
        )
        == NUM_INVESTORS,
        (
            "Startup slice minimum node_index drift."
        ),
    )

    require(
        int(
            startup_nodes[
                "__node_index_numeric"
            ].max()
        )
        == (
            NUM_INVESTORS
            + NUM_STARTUPS
            - 1
        ),
        (
            "Startup slice maximum node_index drift."
        ),
    )

    require(
        bool(
            investor_nodes[
                "raw_entity_id"
            ].notna().all()
        ),
        (
            "Investor numeric slice contains null raw_entity_id."
        ),
    )

    require(
        bool(
            startup_nodes[
                "raw_entity_id"
            ].notna().all()
        ),
        (
            "Startup numeric slice contains null raw_entity_id."
        ),
    )

    investor_raw = (
        investor_nodes[
            "raw_entity_id"
        ]
        .astype(
            str
        )
    )

    startup_raw = (
        startup_nodes[
            "raw_entity_id"
        ]
        .astype(
            str
        )
    )

    require(
        bool(
            investor_raw.is_unique
        ),
        (
            "Investor numeric slice raw_entity_id "
            "is not unique within the Investor role."
        ),
    )

    require(
        bool(
            startup_raw.is_unique
        ),
        (
            "Startup numeric slice raw_entity_id "
            "is not unique within the Startup role."
        ),
    )

    # node_type is DIAGNOSTIC ONLY.
    investor_node_type_counts = (
        investor_nodes[
            "node_type"
        ]
        .astype(
            str
        )
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    startup_node_type_counts = (
        startup_nodes[
            "node_type"
        ]
        .astype(
            str
        )
        .value_counts(
            dropna=False
        )
        .to_dict()
    )

    investor_normalized_role_matches = int(
        investor_nodes[
            "node_type"
        ]
        .astype(
            str
        )
        .str.strip()
        .str.casefold()
        .eq(
            "investor"
        )
        .sum()
    )

    startup_normalized_role_matches = int(
        startup_nodes[
            "node_type"
        ]
        .astype(
            str
        )
        .str.strip()
        .str.casefold()
        .eq(
            "startup"
        )
        .sum()
    )

    print(
        "Phase-3 numeric role slices:"
    )
    print(
        f"  Investor nodes:                     "
        f"{len(investor_nodes):,} "
        f"[0..{NUM_INVESTORS - 1:,}]"
    )
    print(
        f"  Startup nodes:                      "
        f"{len(startup_nodes):,} "
        f"[{NUM_INVESTORS:,}.."
        f"{NUM_INVESTORS + NUM_STARTUPS - 1:,}]"
    )
    print()
    print(
        "node_type diagnostic inside Investor slice:"
    )
    print(
        f"  {investor_node_type_counts}"
    )
    print(
        "node_type diagnostic inside Startup slice:"
    )
    print(
        f"  {startup_node_type_counts}"
    )
    print(
        "Normalized literal 'investor' matches: "
        f"{investor_normalized_role_matches:,}"
    )
    print(
        "Normalized literal 'startup' matches:  "
        f"{startup_normalized_role_matches:,}"
    )
    print(
        "Role boundary source used for mapping: NUMERIC NODE_INDEX SLICE"
    )

    # -------------------------------------------------------------------------
    # raw_entity_id -> frozen global role-node index
    # -------------------------------------------------------------------------

    investor_map = pd.Series(
        investor_nodes[
            "__node_index_numeric"
        ].to_numpy(
            dtype=np.int64
        ),
        index=(
            investor_raw
        ),
    ).to_dict()

    startup_map = pd.Series(
        startup_nodes[
            "__node_index_numeric"
        ].to_numpy(
            dtype=np.int64
        ),
        index=(
            startup_raw
        ),
    ).to_dict()

    investor_global = (
        historical[
            "investor_id"
        ]
        .astype(
            str
        )
        .map(
            investor_map
        )
    )

    startup_global = (
        historical[
            "startup_id"
        ]
        .astype(
            str
        )
        .map(
            startup_map
        )
    )

    investor_mapping_missing = int(
        investor_global.isna().sum()
    )

    startup_mapping_missing = int(
        startup_global.isna().sum()
    )

    if investor_mapping_missing > 0:
        sample_missing_investors = (
            historical.loc[
                investor_global.isna(),
                "investor_id",
            ]
            .astype(
                str
            )
            .drop_duplicates()
            .head(
                10
            )
            .tolist()
        )

        print(
            "Missing historical Investor role mappings:"
        )
        print(
            f"  rows:                               "
            f"{investor_mapping_missing:,}"
        )
        print(
            f"  sample raw IDs:                     "
            f"{sample_missing_investors}"
        )

    if startup_mapping_missing > 0:
        sample_missing_startups = (
            historical.loc[
                startup_global.isna(),
                "startup_id",
            ]
            .astype(
                str
            )
            .drop_duplicates()
            .head(
                10
            )
            .tolist()
        )

        print(
            "Missing historical Startup role mappings:"
        )
        print(
            f"  rows:                               "
            f"{startup_mapping_missing:,}"
        )
        print(
            f"  sample raw IDs:                     "
            f"{sample_missing_startups}"
        )

    require(
        investor_mapping_missing
        == 0,
        (
            "Historical investor role mapping incomplete "
            "against frozen numeric Investor slice."
        ),
    )

    require(
        startup_mapping_missing
        == 0,
        (
            "Historical startup role mapping incomplete "
            "against frozen numeric Startup slice."
        ),
    )

    investor_global_np = (
        investor_global
        .to_numpy(
            dtype=np.int64
        )
    )

    startup_global_np = (
        startup_global
        .to_numpy(
            dtype=np.int64
        )
    )

    require(
        bool(
            (
                (
                    investor_global_np
                    >= 0
                )
                & (
                    investor_global_np
                    < NUM_INVESTORS
                )
            ).all()
        ),
        (
            "Mapped historical investor index outside "
            "frozen Investor slice."
        ),
    )

    require(
        bool(
            (
                (
                    startup_global_np
                    >= NUM_INVESTORS
                )
                & (
                    startup_global_np
                    < (
                        NUM_INVESTORS
                        + NUM_STARTUPS
                    )
                )
            ).all()
        ),
        (
            "Mapped historical startup index outside "
            "frozen Startup slice."
        ),
    )

    startup_local_np = (
        startup_global_np
        - NUM_INVESTORS
    )

    segment_np = historical[
        "segment_number"
    ].to_numpy(
        dtype=np.int16
    )

    pair_key = (
        investor_global_np
        * NUM_STARTUPS
        + startup_local_np
    ).astype(
        np.int64,
        copy=False,
    )

    # Sort by pair, then segment ascending. The first occurrence of each
    # pair is therefore its earliest historical segment.
    order = np.lexsort(
        (
            segment_np.astype(
                np.int64,
                copy=False,
            ),
            pair_key,
        )
    )

    sorted_key = pair_key[
        order
    ]

    sorted_segment = segment_np[
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
        len(
            unique_key
        )
        == 963_374,
        (
            "Unique pre-T60 pair count drift: "
            f"{len(unique_key):,} != 963,374."
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
            "Encoded Phase-2 pair keys are not strictly increasing."
        ),
    )

    require(
        bool(
            (
                (
                    first_segment
                    >= 0
                )
                & (
                    first_segment
                    <= 59
                )
            ).all()
        ),
        (
            "Phase-2 first-positive segment outside T0..T59."
        ),
    )

    return (
        unique_key,
        first_segment,
        {
            "source": (
                "Phase-2 interactions_itrs_temporal_split.parquet"
            ),
            "role_boundary_source": (
                "Phase-3 frozen numeric node_index slices"
            ),
            "historical_events_T0_T59": int(
                len(
                    historical
                )
            ),
            "unique_pre_t60_pairs": int(
                len(
                    unique_key
                )
            ),
            "phase3_total_role_nodes": int(
                len(
                    node_index
                )
            ),
            "investor_numeric_slice_count": int(
                len(
                    investor_nodes
                )
            ),
            "startup_numeric_slice_count": int(
                len(
                    startup_nodes
                )
            ),
            "investor_node_type_counts": {
                str(
                    key
                ): int(
                    value
                )
                for (
                    key,
                    value,
                ) in (
                    investor_node_type_counts.items()
                )
            },
            "startup_node_type_counts": {
                str(
                    key
                ): int(
                    value
                )
                for (
                    key,
                    value,
                ) in (
                    startup_node_type_counts.items()
                )
            },
            "normalized_investor_label_matches": int(
                investor_normalized_role_matches
            ),
            "normalized_startup_label_matches": int(
                startup_normalized_role_matches
            ),
            "investor_mapping_complete": True,
            "startup_mapping_complete": True,
        },
    )


def compare_pair_first_segment_indices(
    *,
    authoritative_keys: np.ndarray,
    authoritative_segments: np.ndarray,
    diagnostic_keys: np.ndarray,
    diagnostic_segments: np.ndarray,
) -> dict:
    same_length = (
        len(authoritative_keys)
        == len(diagnostic_keys)
    )

    keys_equal = (
        same_length
        and bool(
            np.array_equal(
                authoritative_keys,
                diagnostic_keys,
            )
        )
    )

    if keys_equal:
        segment_difference_count = int(
            np.count_nonzero(
                authoritative_segments
                != diagnostic_segments
            )
        )
    else:
        segment_difference_count = -1

    return {
        "authoritative_pair_count": int(
            len(authoritative_keys)
        ),
        "diagnostic_pair_count": int(
            len(diagnostic_keys)
        ),
        "pair_key_arrays_equal": bool(
            keys_equal
        ),
        "first_segment_difference_count": int(
            segment_difference_count
        ),
    }


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
# Sequential positive-major negative sampling (exact Phase-5.3.1l.1 runtime)
# =============================================================================

_PAIR_FIRST_SEGMENT_CACHE: dict[int, int] | None = None


def is_forbidden_scalar(
    *,
    pair_first_segment_map: dict[int, int],
    investor_global: int,
    target_segment: int,
    startup_local: int,
) -> bool:
    encoded = (
        int(investor_global)
        * NUM_STARTUPS
        + int(startup_local)
    )

    first_segment = pair_first_segment_map.get(
        encoded
    )

    if first_segment is None:
        return False

    return (
        int(first_segment)
        <= int(target_segment)
    )


def build_pair_first_segment_map(
    pair_keys_sorted: np.ndarray,
    first_segment_sorted: np.ndarray,
) -> dict[int, int]:
    """
    Build the scalar O(1) lookup used by the exact sequential sampler.

    This changes only lookup implementation, not the frozen eligibility rule:
        candidate is forbidden at target h iff first positive segment <= h.
    """

    global _PAIR_FIRST_SEGMENT_CACHE

    if _PAIR_FIRST_SEGMENT_CACHE is None:
        require(
            len(pair_keys_sorted)
            == len(first_segment_sorted),
            (
                "Pair-key / first-segment "
                "length mismatch."
            ),
        )

        _PAIR_FIRST_SEGMENT_CACHE = {
            int(key): int(segment)
            for key, segment in zip(
                pair_keys_sorted,
                first_segment_sorted,
            )
        }

        require(
            len(_PAIR_FIRST_SEGMENT_CACHE)
            == len(pair_keys_sorted),
            (
                "Pair first-segment map "
                "lost unique keys."
            ),
        )

    return _PAIR_FIRST_SEGMENT_CACHE


def sample_negative_rows_sequential(
    *,
    seed: int,
    positive_investor: np.ndarray,
    positive_segment: np.ndarray,
    pair_first_segment_map: dict[int, int],
    num_rows: int,
    progress: bool = False,
    expected_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Exact Phase-5.3.1l.1 RNG-consumption pattern.

    For each positive event in canonical positive order:
        for negative slot 0..3:
            draw one scalar candidate from PCG64
            reject immediately if historically forbidden
            reject immediately if already accepted for this positive
            otherwise accept

    Therefore every rejection advances the SAME epoch RNG before the next
    candidate/positive is processed. There is no separate vectorized initial
    matrix and no later repair pass.
    """

    require(
        0
        < int(num_rows)
        <= POSITIVE_COUNT,
        (
            "Sequential sampler num_rows "
            "outside 1..POSITIVE_COUNT."
        ),
    )

    require(
        len(positive_investor)
        >= num_rows
        and len(positive_segment)
        >= num_rows,
        (
            "Sequential sampler positive "
            "arrays shorter than num_rows."
        ),
    )

    if expected_matrix is not None:
        require(
            expected_matrix.shape[0] >= int(num_rows)
            and expected_matrix.shape[1] == NEGATIVES_PER_POSITIVE,
            (
                "Expected frozen negative matrix "
                "has incompatible shape."
            ),
        )

    rng = np.random.Generator(
        np.random.PCG64(
            int(seed)
        )
    )

    matrix = np.empty(
        (
            int(num_rows),
            NEGATIVES_PER_POSITIVE,
        ),
        dtype=FROZEN_NEGATIVE_DTYPE,
    )

    forbidden_rejections = 0
    duplicate_rejections = 0
    accepted_future_positive = 0
    accepted_never_positive = 0

    for row_index in range(
        int(num_rows)
    ):
        investor_global = int(
            positive_investor[
                row_index
            ]
        )

        target_segment = int(
            positive_segment[
                row_index
            ]
        )

        for slot in range(
            NEGATIVES_PER_POSITIVE
        ):
            while True:
                # IMPORTANT:
                # Match the original scalar Generator.integers call.
                # Do not vectorize this call: rejection sampling must consume
                # the RNG inline before the next slot/positive.
                candidate = int(
                    rng.integers(
                        0,
                        NUM_STARTUPS,
                    )
                )

                encoded = (
                    investor_global
                    * NUM_STARTUPS
                    + candidate
                )

                first_segment = (
                    pair_first_segment_map.get(
                        encoded
                    )
                )

                if (
                    first_segment is not None
                    and int(first_segment)
                    <= target_segment
                ):
                    forbidden_rejections += 1
                    continue

                duplicate = False

                for previous_slot in range(
                    slot
                ):
                    if (
                        int(
                            matrix[
                                row_index,
                                previous_slot,
                            ]
                        )
                        == candidate
                    ):
                        duplicate = True
                        break

                if duplicate:
                    duplicate_rejections += 1
                    continue

                matrix[
                    row_index,
                    slot,
                ] = candidate

                if first_segment is None:
                    accepted_never_positive += 1
                else:
                    # The <= h case was rejected above.
                    require(
                        int(first_segment)
                        > target_segment,
                        (
                            "Accepted historical candidate "
                            "does not satisfy first_segment > h."
                        ),
                    )

                    accepted_future_positive += 1

                break

        if expected_matrix is not None:
            expected_row = np.asarray(
                expected_matrix[
                    row_index
                ],
                dtype=np.int64,
            )

            generated_row = matrix[
                row_index
            ]

            if not np.array_equal(
                generated_row,
                expected_row,
            ):
                print()
                print(
                    "FIRST FROZEN EPOCH-0 NEGATIVE ROW MISMATCH"
                )
                print(
                    f"  positive_order_index:              {row_index}"
                )
                print(
                    f"  investor_global:                   {investor_global}"
                )
                print(
                    f"  target_segment:                    {target_segment}"
                )
                print(
                    f"  generated negatives:               "
                    f"{generated_row.tolist()}"
                )
                print(
                    f"  frozen negatives:                  "
                    f"{expected_row.tolist()}"
                )

                for frozen_slot, value in enumerate(
                    expected_row.tolist()
                ):
                    candidate = int(
                        value
                    )

                    encoded_candidate = (
                        investor_global
                        * NUM_STARTUPS
                        + candidate
                    )

                    first_segment = (
                        pair_first_segment_map.get(
                            encoded_candidate
                        )
                    )

                    forbidden = (
                        first_segment is not None
                        and int(first_segment)
                        <= target_segment
                    )

                    print(
                        "  frozen slot "
                        f"{frozen_slot}: startup_local="
                        f"{candidate}, first_segment="
                        f"{first_segment}, "
                        "forbidden_by_authoritative_lookup="
                        f"{bool(forbidden)}"
                    )

                raise AssertionError(
                    "Sequential epoch-0 regeneration "
                    "diverged from the frozen matrix at "
                    f"positive row {row_index}."
                )

        if (
            progress
            and (
                (row_index + 1)
                % 100_000
                == 0
                or (
                    row_index + 1
                    == int(num_rows)
                )
            )
        ):
            print(
                "Generated negatives for "
                f"{row_index + 1:,} / "
                f"{int(num_rows):,} positives"
            )

    accepted_negatives = (
        int(num_rows)
        * NEGATIVES_PER_POSITIVE
    )

    repair_rng_draws = (
        forbidden_rejections
        + duplicate_rejections
    )

    metadata = {
        "negative_seed": (
            int(seed)
        ),
        "positive_rows": (
            int(num_rows)
        ),
        "accepted_negatives": (
            int(accepted_negatives)
        ),
        "repair_rng_draws": (
            int(repair_rng_draws)
        ),
        "forbidden_candidate_rejections": (
            int(forbidden_rejections)
        ),
        "duplicate_candidate_rejections": (
            int(duplicate_rejections)
        ),
        "total_rng_draws": (
            int(
                accepted_negatives
                + repair_rng_draws
            )
        ),
        "accepted_future_positive": (
            int(
                accepted_future_positive
            )
        ),
        "accepted_never_positive": (
            int(
                accepted_never_positive
            )
        ),

        # Backward-compatible audit aliases used by downstream assertions in
        # this script. These names are historical only; scientifically these
        # are immediate rejection counts, not a vectorized "initial" repair.
        "initial_forbidden_rejections": (
            int(forbidden_rejections)
        ),
        "initial_duplicate_rejections": (
            int(duplicate_rejections)
        ),
        "initial_overlap_forbidden_and_duplicate": (
            0
        ),
        "initial_invalid_slots": (
            int(repair_rng_draws)
        ),
        "repair_positions": (
            int(repair_rng_draws)
        ),
        "extra_forbidden_repair_rejections": (
            int(forbidden_rejections)
        ),
        "extra_duplicate_repair_rejections": (
            int(duplicate_rejections)
        ),
    }

    return (
        matrix,
        metadata,
    )


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
    progress: bool = False,
    expected_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Generalized epoch-indexed wrapper around the exact sequential
    Phase-5.3.1l.1 sampler.

    selected_traversal / selected_duplicate_scope are retained in the
    function signature because downstream production contracts serialize
    these implementation labels.
    """

    require(
        selected_traversal
        == "positive_major_immediate_rejection",
        (
            "Unexpected production negative "
            "sampling traversal."
        ),
    )

    require(
        selected_duplicate_scope
        == "prior_accepted_within_positive",
        (
            "Unexpected production duplicate "
            "sampling scope."
        ),
    )

    seed = derive_epoch_seed(
        NEGATIVE_NAMESPACE,
        epoch_index,
    )

    pair_first_segment_map = (
        build_pair_first_segment_map(
            pair_keys_sorted,
            first_segment_sorted,
        )
    )

    (
        final_matrix,
        metadata,
    ) = sample_negative_rows_sequential(
        seed=seed,
        positive_investor=(
            positive_investor
        ),
        positive_segment=(
            positive_segment
        ),
        pair_first_segment_map=(
            pair_first_segment_map
        ),
        num_rows=(
            POSITIVE_COUNT
        ),
        progress=(
            progress
        ),
        expected_matrix=(
            expected_matrix
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

    require(
        semantic[
            "accepted_future_positive"
        ]
        == metadata[
            "accepted_future_positive"
        ],
        (
            "Sequential sampler / independent vectorized audit "
            "future-positive audit disagree."
        ),
    )

    require(
        semantic[
            "accepted_never_positive"
        ]
        == metadata[
            "accepted_never_positive"
        ],
        (
            "Sequential sampler / independent vectorized audit "
            "never-positive audit disagree."
        ),
    )

    metadata = {
        "epoch_index": (
            int(epoch_index)
        ),
        **metadata,
        **semantic,
    }

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
        TEMPORAL_INTERACTIONS_PATH,
        NODE_INDEX_PATH,
        PHASE_5_3_2A_CONTRACT_PATH,
        PHASE_5_3_4_CONTRACT_PATH,
        PHASE_5_3_5A_REPRESENTATION_CONTRACT_PATH,
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

    representation_contract = load_json(
        PHASE_5_3_5A_REPRESENTATION_CONTRACT_PATH
    )

    require(
        representation_contract[
            "status"
        ]
        == "DIAGNOSIS_COMPLETE",
        (
            "Phase-5.3.5a representation audit "
            "is not complete."
        ),
    )

    require(
        representation_contract[
            "representation_explanation_found"
        ]
        is True,
        (
            "Phase-5.3.5a did not prove a "
            "representation-level explanation."
        ),
    )

    require(
        representation_contract[
            "frozen_negative"
        ][
            "dtype"
        ]
        == "int32",
        (
            "Frozen negative dtype is not int32."
        ),
    )

    require(
        representation_contract[
            "frozen_negative"
        ][
            "dtype_str"
        ]
        == FROZEN_NEGATIVE_DTYPE_STR,
        (
            "Frozen negative dtype.str is not <i4."
        ),
    )

    require(
        representation_contract[
            "frozen_negative"
        ][
            "logical_sha256"
        ]
        == EXPECTED_EPOCH0_NEGATIVE_SHA256,
        (
            "Phase-5.3.5a frozen negative hash drift."
        ),
    )

    require(
        representation_contract[
            "v4_observed_generated_sha256"
        ]
        == EXPECTED_V4_INT64_NEGATIVE_SHA256,
        (
            "Phase-5.3.5a V4 int64 hash anchor drift."
        ),
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
        np.asarray(
            frozen_negative0
        ).dtype
        == FROZEN_NEGATIVE_DTYPE,
        (
            "Frozen epoch-0 negative matrix "
            "dtype drift from int32."
        ),
    )

    require(
        np.asarray(
            frozen_negative0
        ).dtype.str
        == FROZEN_NEGATIVE_DTYPE_STR,
        (
            "Frozen epoch-0 negative matrix "
            "dtype.str drift from <i4."
        ),
    )

    require(
        bool(
            np.asarray(
                frozen_negative0
            ).flags.c_contiguous
        ),
        (
            "Frozen epoch-0 negative matrix "
            "is not C-contiguous."
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
    # Build authoritative Phase-2 historical first-positive index
    # =========================================================================

    banner(
        "BUILD AUTHORITATIVE PHASE-2 PRE-T60 ELIGIBILITY INDEX"
    )

    (
        pair_keys_sorted,
        first_segment_sorted,
        pair_index_metadata,
    ) = build_first_positive_segment_index_from_phase2(
        temporal_path=(
            TEMPORAL_INTERACTIONS_PATH
        ),
        node_index_path=(
            NODE_INDEX_PATH
        ),
    )

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
            "is missing from the authoritative "
            "Phase-2 T0..T59 pair index."
        ),
    )

    print(
        "Authoritative source:                 PHASE-2 TEMPORAL + PHASE-3 NUMERIC SLICES"
    )
    print(
        "T0..T59 historical events:           "
        f"{pair_index_metadata['historical_events_T0_T59']:,}"
    )
    print(
        "Unique Investor-Startup pairs:        "
        f"{pair_index_metadata['unique_pre_t60_pairs']:,}"
    )
    print(
        "All training focal pairs found by h:  PASS"
    )

    # Diagnostic only: compare V2's trend-derived lookup to Phase-2.
    banner(
        "DIAGNOSTIC CROSS-CHECK: PHASE-4 TREND INDEX vs PHASE-2"
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
        trend_pair_keys,
        trend_first_segments,
        trend_metadata,
    ) = build_first_positive_segment_index(
        trend_period_counts=(
            trend_period_counts
        ),
        trend_startup_indices_global=(
            trend_startup_indices
        ),
    )

    pair_index_comparison = compare_pair_first_segment_indices(
        authoritative_keys=(
            pair_keys_sorted
        ),
        authoritative_segments=(
            first_segment_sorted
        ),
        diagnostic_keys=(
            trend_pair_keys
        ),
        diagnostic_segments=(
            trend_first_segments
        ),
    )

    print(
        "Phase-2 pair count:                   "
        f"{pair_index_comparison['authoritative_pair_count']:,}"
    )
    print(
        "Phase-4 trend pair count:             "
        f"{pair_index_comparison['diagnostic_pair_count']:,}"
    )
    print(
        "Pair-key arrays equal:                "
        f"{pair_index_comparison['pair_key_arrays_equal']}"
    )
    print(
        "First-segment differences:            "
        f"{pair_index_comparison['first_segment_difference_count']}"
    )
    print(
        "Sampling lookup used below:           PHASE-2 ONLY"
    )

    pair_index_df = pd.DataFrame(
        [
            {
                "metric": "authoritative_source",
                "value": (
                    pair_index_metadata[
                        "source"
                    ]
                ),
            },
            {
                "metric": "role_boundary_source",
                "value": (
                    pair_index_metadata[
                        "role_boundary_source"
                    ]
                ),
            },
            {
                "metric": "historical_events_T0_T59",
                "value": (
                    pair_index_metadata[
                        "historical_events_T0_T59"
                    ]
                ),
            },
            {
                "metric": "unique_pre_t60_pairs",
                "value": (
                    pair_index_metadata[
                        "unique_pre_t60_pairs"
                    ]
                ),
            },
            {
                "metric": "all_training_positives_found_by_h",
                "value": True,
            },
            {
                "metric": "phase4_trend_pair_key_arrays_equal",
                "value": (
                    pair_index_comparison[
                        "pair_key_arrays_equal"
                    ]
                ),
            },
            {
                "metric": "phase4_trend_first_segment_difference_count",
                "value": (
                    pair_index_comparison[
                        "first_segment_difference_count"
                    ]
                ),
            },
            {
                "metric": "sampling_uses_phase4_trend_lookup",
                "value": False,
            },
        ]
    )

    # =========================================================================
    # Epoch-0 exact sequential-sampler regression
    # =========================================================================

    banner(
        "REGENERATE EPOCH-0 NEGATIVES WITH ORIGINAL SEQUENTIAL RNG CONSUMPTION"
    )

    selected_traversal = (
        "positive_major_immediate_rejection"
    )

    selected_duplicate_scope = (
        "prior_accepted_within_positive"
    )

    matching_variants = [
        (
            selected_traversal,
            selected_duplicate_scope,
        )
    ]

    pair_first_segment_map = (
        build_pair_first_segment_map(
            pair_keys_sorted,
            first_segment_sorted,
        )
    )

    # -------------------------------------------------------------------------
    # First reproduce the original Phase-5.3.1l.1 deterministic 10k prefix.
    # This is an early fail-fast check before generating the complete epoch.
    # -------------------------------------------------------------------------

    prefix_rows = 10_000

    (
        regenerated_prefix,
        prefix_metadata,
    ) = sample_negative_rows_sequential(
        seed=(
            EXPECTED_EPOCH0_NEGATIVE_SEED
        ),
        positive_investor=(
            positive_investor
        ),
        positive_segment=(
            positive_segment
        ),
        pair_first_segment_map=(
            pair_first_segment_map
        ),
        num_rows=(
            prefix_rows
        ),
        progress=False,
    )

    frozen_prefix = np.asarray(
        frozen_negative0[
            :prefix_rows
        ]
    )

    prefix_exact = bool(
        np.array_equal(
            regenerated_prefix,
            frozen_prefix,
        )
    )

    require(
        prefix_exact,
        (
            "Exact sequential epoch-0 sampler "
            "does not reproduce the frozen "
            "10,000-positive negative prefix."
        ),
    )

    print(
        "Regenerated positive rows:            10,000"
    )
    print(
        "Exact negative prefix match:          PASS"
    )

    del regenerated_prefix
    del frozen_prefix
    gc.collect()

    # -------------------------------------------------------------------------
    # Full exact epoch-0 regeneration.
    # -------------------------------------------------------------------------

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
        progress=True,
        expected_matrix=(
            np.asarray(
                frozen_negative0
            )
        ),
    )

    generated_negative0_sha = (
        preflight
        .array_logical_sha256(
            generated_negative0
        )
    )

    require(
        generated_negative0.dtype
        == FROZEN_NEGATIVE_DTYPE,
        (
            "Generalized epoch-0 negative matrix "
            "did not preserve frozen int32 dtype."
        ),
    )

    require(
        generated_negative0.dtype.str
        == FROZEN_NEGATIVE_DTYPE_STR,
        (
            "Generalized epoch-0 negative matrix "
            "dtype.str is not <i4."
        ),
    )

    require(
        bool(
            generated_negative0.flags.c_contiguous
        ),
        (
            "Generalized epoch-0 negative matrix "
            "is not C-contiguous."
        ),
    )

    v4_int64_projection_sha = (
        preflight
        .array_logical_sha256(
            np.ascontiguousarray(
                generated_negative0.astype(
                    np.int64,
                    copy=True,
                )
            )
        )
    )

    require(
        v4_int64_projection_sha
        == EXPECTED_V4_INT64_NEGATIVE_SHA256,
        (
            "The corrected int32 epoch-0 values "
            "do not reproduce the known V4 int64 "
            "representation hash when cast to int64."
        ),
    )

    exact_negative0_array = bool(
        np.array_equal(
            generated_negative0,
            np.asarray(
                frozen_negative0
            ),
        )
    )

    print()
    print(
        "Generalized epoch-0 negative SHA256:"
    )
    print(
        generated_negative0_sha
    )
    print(
        "Expected frozen negative SHA256:"
    )
    print(
        EXPECTED_EPOCH0_NEGATIVE_SHA256
    )
    print()
    print(
        "Epoch-0 representation regression:"
    )
    print(
        "  generated dtype:                   "
        f"{generated_negative0.dtype}"
    )
    print(
        "  generated dtype.str:               "
        f"{generated_negative0.dtype.str}"
    )
    print(
        "  frozen dtype:                      "
        f"{np.asarray(frozen_negative0).dtype}"
    )
    print(
        "  value equality:                    "
        f"{'PASS' if exact_negative0_array else 'FAIL'}"
    )
    print(
        "  V4 int64 projection SHA:           "
        f"{v4_int64_projection_sha}"
    )
    print()
    print(
        "Epoch-0 rejection diagnostics:"
    )
    print(
        "  repair / rejected RNG draws:       "
        f"{metadata0['repair_rng_draws']:,}"
    )
    print(
        "  forbidden rejections:              "
        f"{metadata0['forbidden_candidate_rejections']:,}"
    )
    print(
        "  duplicate rejections:              "
        f"{metadata0['duplicate_candidate_rejections']:,}"
    )
    print(
        "  accepted future-positive:          "
        f"{metadata0['accepted_future_positive']:,}"
    )
    print(
        "  accepted never-positive:           "
        f"{metadata0['accepted_never_positive']:,}"
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
        exact_negative0_array,
        (
            "Generalized epoch-0 negative "
            "matrix is not byte-for-byte "
            "the frozen matrix."
        ),
    )

    require(
        metadata0[
            "repair_rng_draws"
        ]
        == EXPECTED_EPOCH0_REPAIR_POSITIONS,
        (
            "Epoch-0 extra/rejected RNG draw "
            "count drift."
        ),
    )

    require(
        metadata0[
            "forbidden_candidate_rejections"
        ]
        == EXPECTED_EPOCH0_FORBIDDEN_REJECTIONS,
        (
            "Epoch-0 forbidden candidate "
            "rejection count drift."
        ),
    )

    require(
        metadata0[
            "duplicate_candidate_rejections"
        ]
        == EXPECTED_EPOCH0_DUPLICATE_REJECTIONS,
        (
            "Epoch-0 duplicate candidate "
            "rejection count drift."
        ),
    )

    require(
        metadata0[
            "accepted_future_positive"
        ]
        == EXPECTED_EPOCH0_FUTURE_POSITIVE_ACCEPTED,
        (
            "Epoch-0 accepted future-positive "
            "count drift."
        ),
    )

    require(
        metadata0[
            "accepted_never_positive"
        ]
        == EXPECTED_EPOCH0_NEVER_POSITIVE_ACCEPTED,
        (
            "Epoch-0 accepted never-positive "
            "count drift."
        ),
    )

    require(
        metadata0[
            "repair_rng_draws"
        ]
        == (
            metadata0[
                "forbidden_candidate_rejections"
            ]
            + metadata0[
                "duplicate_candidate_rejections"
            ]
        ),
        (
            "Epoch-0 repair RNG draw accounting "
            "does not equal forbidden + duplicate "
            "rejections."
        ),
    )

    print()
    print(
        f"Negative matrix:                      "
        f"{tuple(generated_negative0.shape)}"
    )
    print(
        f"Accepted negatives:                   "
        f"{metadata0['accepted_negatives']:,}"
    )
    print(
        f"Repair RNG draws:                     "
        f"{metadata0['repair_rng_draws']:,}"
    )
    print(
        f"Forbidden candidate rejections:       "
        f"{metadata0['forbidden_candidate_rejections']:,}"
    )
    print(
        f"Duplicate candidate rejections:       "
        f"{metadata0['duplicate_candidate_rejections']:,}"
    )
    print(
        f"Accepted future-positive pairs:       "
        f"{metadata0['accepted_future_positive']:,}"
    )
    print(
        f"Accepted never-positive pre-T60:      "
        f"{metadata0['accepted_never_positive']:,}"
    )

    print()
    print(
        "Epoch-0 negative logical SHA256:"
    )
    print(
        generated_negative0_sha
    )

    # The original run evidence showed generation positive-by-positive.
    # This audit row freezes that runtime interpretation; there is no
    # vectorized-initial-draw / later-repair variant search anymore.
    variant_df = pd.DataFrame(
        [
            {
                "traversal": (
                    selected_traversal
                ),
                "duplicate_scope": (
                    selected_duplicate_scope
                ),
                "runtime_evidence": (
                    "Phase-5.3.1l.1 generated "
                    "negatives positive-by-positive "
                    "with immediate rejected draws"
                ),
                "repair_rng_draws": (
                    metadata0[
                        "repair_rng_draws"
                    ]
                ),
                "forbidden_candidate_rejections": (
                    metadata0[
                        "forbidden_candidate_rejections"
                    ]
                ),
                "duplicate_candidate_rejections": (
                    metadata0[
                        "duplicate_candidate_rejections"
                    ]
                ),
                "negative_matrix_sha256": (
                    generated_negative0_sha
                ),
                "exact_frozen_epoch0_match": (
                    True
                ),
                "status": (
                    "MATCH"
                ),
            }
        ]
    )

    banner(
        "GENERALIZED GENERATOR -> EXACT EPOCH-0 ORDER REGRESSION"
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
        generated_order0_seed
        == EXPECTED_EPOCH0_ORDER_SEED,
        (
            "Generalized epoch-0 order seed drift."
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
                    "repair_rng_draws"
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
            dtype=FROZEN_NEGATIVE_DTYPE,
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
                        "repair_rng_draws"
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
            "phase_5_3_5a_representation_audit_complete",
            (
                representation_contract[
                    "status"
                ]
                == "DIAGNOSIS_COMPLETE"
            ),
        ),
        (
            "frozen_negative_dtype_int32",
            (
                np.asarray(
                    frozen_negative0
                ).dtype
                == FROZEN_NEGATIVE_DTYPE
            ),
        ),
        (
            "generated_epoch0_negative_dtype_int32",
            (
                generated_negative0.dtype
                == FROZEN_NEGATIVE_DTYPE
            ),
        ),
        (
            "generated_epoch0_negative_dtype_str_i4",
            (
                generated_negative0.dtype.str
                == FROZEN_NEGATIVE_DTYPE_STR
            ),
        ),
        (
            "v4_hash_explained_by_int64_projection",
            (
                v4_int64_projection_sha
                == EXPECTED_V4_INT64_NEGATIVE_SHA256
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
                    "forbidden_candidate_rejections"
                ]
                == 1743
            ),
        ),
        (
            "epoch0_duplicate_rejections_16",
            (
                metadata0[
                    "duplicate_candidate_rejections"
                ]
                == 16
            ),
        ),
        (
            "epoch0_repair_rng_draws_1759",
            (
                metadata0[
                    "repair_rng_draws"
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
            "sequential_sampler_runtime_frozen",
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
                    "negative_sampling_rng_consumption"
                ),
                "value": (
                    "POSITIVE_MAJOR_SCALAR_DRAWS_WITH_IMMEDIATE_REJECTION"
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
                    "negative_sampling_traversal"
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
                    "PHASE2_FIRST_POSITIVE_SEGMENT_GREATER_THAN_H_IS_ELIGIBLE"
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
                    "training_negative_matrix_dtype"
                ),
                "value": (
                    "INT32_LITTLE_ENDIAN_C_CONTIGUOUS"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE_"
                    "EXACT_FROZEN_ARTIFACT_REGRESSION"
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
            "matrix_representation": {
                "dtype": (
                    str(
                        FROZEN_NEGATIVE_DTYPE
                    )
                ),
                "dtype_str": (
                    FROZEN_NEGATIVE_DTYPE_STR
                ),
                "itemsize": (
                    int(
                        FROZEN_NEGATIVE_DTYPE.itemsize
                    )
                ),
                "layout": (
                    "C_CONTIGUOUS"
                ),
                "provenance": (
                    "Phase-5.3.5a frozen epoch-0 "
                    "representation audit"
                ),
            },
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
            "RNG_consumption": (
                "scalar draw per candidate in canonical positive-major order "
                "with immediate rejection"
            ),
            "matrix_shape": [
                POSITIVE_COUNT,
                NEGATIVES_PER_POSITIVE,
            ],
            "sampling_traversal": (
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
            "negative_dtype": (
                str(
                    generated_negative0.dtype
                )
            ),
            "negative_dtype_str": (
                generated_negative0.dtype.str
            ),
            "negative_c_contiguous": (
                bool(
                    generated_negative0.flags.c_contiguous
                )
            ),
            "v4_int64_projection_sha256": (
                v4_int64_projection_sha
            ),
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
            "repair_rng_draws": (
                metadata0[
                    "repair_rng_draws"
                ]
            ),
            "forbidden_rejections": (
                metadata0[
                    "forbidden_candidate_rejections"
                ]
            ),
            "duplicate_rejections": (
                metadata0[
                    "duplicate_candidate_rejections"
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
        "selected_sampling_traversal": (
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
        "Selected negative-sampling runtime:"
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
        "  negative values:                    EXACT"
    )
    print(
        "  negative dtype:                     int32 / <i4"
    )
    print(
        "  negative layout:                    C-CONTIGUOUS"
    )
    print(
        "  negative logical SHA:               EXACT"
    )
    print(
        "  negative matrix:                    BYTE/REPRESENTATION-EXACT"
    )
    print(
        "  example order:                      BYTE-EXACT"
    )
    print(
        "  rejected / repair RNG draws:        1,759"
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