#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math

import numpy as np
import pandas as pd


# =============================================================================
# FROZEN SCOPE
# =============================================================================

SCOPE_ID = "DEV_T58"
ORIGIN_LABEL = "T58"
EPOCH = 0

TRAIN_MIN_SEGMENT = 1
TRAIN_MAX_SEGMENT = 57
HISTORY_MIN_SEGMENT = 0

BASE_SEED = 42

NEGATIVE_NAMESPACE = (
    "ITRS_PHASE8_TRAIN_NEGATIVE"
)

ORDER_NAMESPACE = (
    "ITRS_PHASE8_TRAIN_ORDER"
)

NEGATIVES_PER_POSITIVE = 4
SERIALIZED_SLOTS_PER_POSITIVE = 5
BATCH_SIZE = 512

EXPECTED_NUM_INVESTORS = 165_975
EXPECTED_NUM_STARTUPS = 311_589
EXPECTED_NUM_NODES = (
    EXPECTED_NUM_INVESTORS
    + EXPECTED_NUM_STARTUPS
)

EXPECTED_T0_EVENTS = 100_173
EXPECTED_TRAIN_POSITIVES = 1_026_268
EXPECTED_HISTORY_EVENTS = (
    EXPECTED_T0_EVENTS
    + EXPECTED_TRAIN_POSITIVES
)

EXPECTED_ELIGIBLE_STARTUPS = 308_854


# =============================================================================
# AUTHORITATIVE INPUTS
# =============================================================================

TEMPORAL_SPLIT_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "interactions_itrs_temporal_split.parquet"
)

NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/model_ready/"
    "node_index.parquet"
)

ORIGIN_MASK_PATH = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_1_feature_materialization/"
    "phase_8_2_1c_origin_startup_masks_V2.parquet"
)

INTERPRETATION_CLOSURE = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_6_training_stream_dry_run/"
    "phase_8_2_6_training_stream_interpretation_closure_V1.json"
)


# =============================================================================
# OUTPUTS
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_6_training_stream_dry_run/"
    "DEV_T58/"
    "epoch_0"
)

POSITIVE_ORDER_PATH = (
    OUT_DIR
    / "canonical_training_positive_event_order.parquet"
)

ELIGIBLE_LOCAL_PATH = (
    OUT_DIR
    / "origin_eligible_startup_local.npy"
)

NEGATIVE_MATRIX_PATH = (
    OUT_DIR
    / "epoch_0_training_negative_startup_local.npy"
)

EPOCH_ORDER_PATH = (
    OUT_DIR
    / "epoch_0_training_example_order.npy"
)

FIRST_BATCH_PATH = (
    OUT_DIR
    / "epoch_0_first_batch_manifest.parquet"
)

SEGMENT_GROUP_PATH = (
    OUT_DIR
    / "epoch_0_first_batch_segment_groups.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "phase_8_2_6_dev_t58_epoch0_stream_summary_V1.json"
)

HASH_REGISTRY_PATH = (
    OUT_DIR
    / "phase_8_2_6_dev_t58_epoch0_artifact_sha256_V1.json"
)


# =============================================================================
# HELPERS
# =============================================================================

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def derive_seed(
    namespace: str,
    scope_id: str,
    epoch_index: int,
) -> tuple[int, str]:

    payload = (
        f"{namespace}|"
        f"base_seed={BASE_SEED}|"
        f"scope={scope_id}|"
        f"epoch={epoch_index}|"
        f"component=NA|"
        f"parameter=NA|"
        f"shape=NA|"
        f"dtype=NA|"
        f"replicate=NA|"
        f"attempt=0|"
        f"dimension=NA"
    )

    digest = hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).digest()

    seed = int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )

    return (
        seed,
        payload,
    )


def positive_stream_logical_sha256(
    frame: pd.DataFrame,
) -> str:

    digest = hashlib.sha256()

    ids = (
        frame[
            "interaction_id"
        ]
        .astype(str)
        .tolist()
    )

    chunk_size = 10_000

    for start in range(
        0,
        len(ids),
        chunk_size,
    ):

        end = min(
            start + chunk_size,
            len(ids),
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
            ].to_numpy(
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


def array_logical_sha256(
    value: np.ndarray,
) -> str:

    value = np.ascontiguousarray(
        value
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
        b"\0"
    )

    digest.update(
        "x".join(
            str(
                dim
            )
            for dim in value.shape
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        b"\0"
    )

    digest.update(
        value.tobytes(
            order="C"
        )
    )

    return digest.hexdigest()


def generate_epoch_negatives(
    *,
    investor_globals: np.ndarray,
    target_segments: np.ndarray,
    pair_first_segment: dict[int, int],
    eligible_startup_locals: np.ndarray,
    num_startups: int,
    seed: int,
    max_rows: int | None = None,
    progress: bool = True,
):

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

    candidate_count = int(
        len(
            eligible_startup_locals
        )
    )

    require(
        candidate_count > 0,
        "Eligible Startup candidate universe is empty.",
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
                f"Invalid target segment "
                f"T{h} at row {row_index}."
            ),
        )

        accepted = []
        accepted_set = set()

        initial_positions = (
            rng.integers(
                low=0,
                high=candidate_count,
                size=NEGATIVES_PER_POSITIVE,
                dtype=np.int64,
            )
        )

        for candidate_position in (
            initial_positions
        ):

            candidate = int(
                eligible_startup_locals[
                    int(
                        candidate_position
                    )
                ]
            )

            pair_key = (
                investor_global
                * num_startups
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

            if candidate in accepted_set:

                duplicate_rejections += 1
                continue

            accepted.append(
                candidate
            )

            accepted_set.add(
                candidate
            )

        while len(
            accepted
        ) < NEGATIVES_PER_POSITIVE:

            repair_draw_count += 1

            candidate_position = int(
                rng.integers(
                    low=0,
                    high=candidate_count,
                    dtype=np.int64,
                )
            )

            candidate = int(
                eligible_startup_locals[
                    candidate_position
                ]
            )

            pair_key = (
                investor_global
                * num_startups
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

            if candidate in accepted_set:

                duplicate_rejections += 1
                continue

            accepted.append(
                candidate
            )

            accepted_set.add(
                candidate
            )

        negatives[
            row_index,
            :
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
                "  negatives: "
                f"{row_index + 1:,}"
                " / "
                f"{total_rows:,}"
            )

    stats = {
        "positive_rows":
            total_rows,

        "accepted_negatives":
            int(
                negatives.size
            ),

        "initial_draw_count":
            int(
                initial_draw_count
            ),

        "repair_draw_count":
            int(
                repair_draw_count
            ),

        "total_rng_candidate_draws":
            int(
                initial_draw_count
                + repair_draw_count
            ),

        "forbidden_rejections":
            int(
                forbidden_rejections
            ),

        "duplicate_rejections":
            int(
                duplicate_rejections
            ),
    }

    return (
        negatives,
        stats,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 120)
    print("PHASE 8.2.6 — DEV_T58 EPOCH-0 TRAINING STREAM MATERIALIZATION V1")
    print("=" * 120)

    for path in (
        TEMPORAL_SPLIT_PATH,
        NODE_INDEX_PATH,
        ORIGIN_MASK_PATH,
        INTERPRETATION_CLOSURE,
    ):

        require(
            path.exists(),
            f"Missing required input: {path}",
        )

    closure = json.loads(
        INTERPRETATION_CLOSURE.read_text(
            encoding="utf-8"
        )
    )

    require(
        closure[
            "status"
        ]
        == "FROZEN_BEFORE_STREAM_GENERATION",
        "8.2.6 interpretation closure is not frozen.",
    )

    require(
        closure[
            "negative_candidate_universe"
        ][
            "expected_eligible_counts"
        ][
            SCOPE_ID
        ]
        == EXPECTED_ELIGIBLE_STARTUPS,
        "T58 eligible count changed in closure.",
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # ROLE REGISTRY
    # =========================================================================

    print()
    print("1. ROLE REGISTRY")

    node_index = pd.read_parquet(
        NODE_INDEX_PATH,
        columns=[
            "node_index",
            "node_type",
            "raw_entity_id",
        ],
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
        len(
            node_index
        )
        == EXPECTED_NUM_NODES,
        "Phase-3 node count changed.",
    )

    require(
        np.array_equal(
            node_index[
                "node_index"
            ].to_numpy(
                dtype=np.int64
            ),
            np.arange(
                EXPECTED_NUM_NODES,
                dtype=np.int64,
            ),
        ),
        "Phase-3 node indices are not contiguous.",
    )

    node_types = (
        node_index[
            "node_type"
        ]
        .astype(str)
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
        == EXPECTED_NUM_INVESTORS,
        "Investor role count changed.",
    )

    require(
        len(
            startup_nodes
        )
        == EXPECTED_NUM_STARTUPS,
        "Startup role count changed.",
    )

    require(
        np.array_equal(
            investor_nodes[
                "node_index"
            ].to_numpy(
                dtype=np.int64
            ),
            np.arange(
                0,
                EXPECTED_NUM_INVESTORS,
                dtype=np.int64,
            ),
        ),
        "Investor global slice changed.",
    )

    require(
        np.array_equal(
            startup_nodes[
                "node_index"
            ].to_numpy(
                dtype=np.int64
            ),
            np.arange(
                EXPECTED_NUM_INVESTORS,
                EXPECTED_NUM_NODES,
                dtype=np.int64,
            ),
        ),
        "Startup global slice changed.",
    )

    investor_id_to_global = dict(
        zip(
            investor_nodes[
                "raw_entity_id"
            ].astype(str),
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
            ].astype(str),
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
        .astype(str)
        .to_numpy()
    )

    print(
        "  Investor role slice: PASS"
    )

    print(
        "  Startup role slice:  PASS"
    )

    # =========================================================================
    # ORIGIN-ELIGIBLE NEGATIVE UNIVERSE
    # =========================================================================

    print()
    print("2. T58 ORIGIN-ELIGIBLE STARTUP UNIVERSE")

    masks = pd.read_parquet(
        ORIGIN_MASK_PATH,
        columns=[
            "origin_label",
            "startup_node_index",
            "startup_id",
            "origin_eligible",
        ],
    )

    t57_masks = (
        masks.loc[
            masks[
                "origin_label"
            ].astype(str)
            == ORIGIN_LABEL
        ]
        .copy()
    )

    require(
        len(
            t57_masks
        )
        == EXPECTED_NUM_STARTUPS,
        "T58 mask does not contain one row per Startup.",
    )

    eligible = (
        t57_masks.loc[
            t57_masks[
                "origin_eligible"
            ].astype(bool)
        ]
        .copy()
    )

    require(
        len(
            eligible
        )
        == EXPECTED_ELIGIBLE_STARTUPS,
        "T58 origin-eligible count changed.",
    )

    eligible[
        "startup_id"
    ] = (
        eligible[
            "startup_id"
        ]
        .astype(str)
    )

    eligible[
        "mapped_startup_global"
    ] = (
        eligible[
            "startup_id"
        ]
        .map(
            startup_id_to_global
        )
    )

    require(
        eligible[
            "mapped_startup_global"
        ].notna().all(),
        "Eligible Startup missing from Phase-3 role registry.",
    )

    eligible[
        "mapped_startup_global"
    ] = (
        eligible[
            "mapped_startup_global"
        ]
        .astype(
            np.int64
        )
    )

    eligible[
        "startup_node_index"
    ] = (
        eligible[
            "startup_node_index"
        ]
        .astype(
            np.int64
        )
    )

    require(
        np.array_equal(
            eligible[
                "startup_node_index"
            ].to_numpy(
                dtype=np.int64
            ),
            eligible[
                "mapped_startup_global"
            ].to_numpy(
                dtype=np.int64
            ),
        ),
        (
            "Phase-8 startup_node_index does not equal "
            "the frozen Phase-3 Startup global index."
        ),
    )

    eligible = (
        eligible
        .sort_values(
            "startup_node_index",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    eligible_startup_locals = (
        eligible[
            "mapped_startup_global"
        ].to_numpy(
            dtype=np.int64
        )
        - EXPECTED_NUM_INVESTORS
    )

    require(
        len(
            np.unique(
                eligible_startup_locals
            )
        )
        == EXPECTED_ELIGIBLE_STARTUPS,
        "Eligible Startup local indices are not unique.",
    )

    require(
        bool(
            (
                eligible_startup_locals
                >= 0
            ).all()
        )
        and bool(
            (
                eligible_startup_locals
                < EXPECTED_NUM_STARTUPS
            ).all()
        ),
        "Eligible Startup local index outside role universe.",
    )

    require(
        bool(
            (
                np.diff(
                    eligible_startup_locals
                )
                > 0
            ).all()
        ),
        "Eligible Startup local array is not strictly ascending.",
    )

    np.save(
        ELIGIBLE_LOCAL_PATH,
        eligible_startup_locals.astype(
            np.int32
        ),
        allow_pickle=False,
    )

    print(
        f"  Eligible Startups:    "
        f"{len(eligible_startup_locals):,}"
    )

    print(
        "  Phase-3 identity map: PASS"
    )

    # =========================================================================
    # TEMPORAL EVENTS + ROLE MAPPING
    # =========================================================================

    print()
    print("3. T0..T57 HISTORY / T1..T57 POSITIVES")

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

    history = (
        temporal.loc[
            (
                temporal[
                    "segment_number"
                ]
                >= HISTORY_MIN_SEGMENT
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

    t0_count = int(
        (
            history[
                "segment_number"
            ]
            == 0
        ).sum()
    )

    require(
        t0_count
        == EXPECTED_T0_EVENTS,
        "T0 event count changed.",
    )

    require(
        len(
            history
        )
        == EXPECTED_HISTORY_EVENTS,
        "T0..T57 event count changed.",
    )

    require(
        len(
            training
        )
        == EXPECTED_TRAIN_POSITIVES,
        "T1..T57 training positive count changed.",
    )

    require(
        training[
            "interaction_id"
        ].notna().all(),
        "Training interaction_id contains null.",
    )

    require(
        training[
            "interaction_id"
        ]
        .astype(str)
        .is_unique,
        "Training interaction_id is not unique.",
    )

    for frame_name, frame in (
        (
            "history",
            history,
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
            .astype(str)
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
            .astype(str)
            .map(
                startup_id_to_global
            )
        )

        require(
            frame[
                "investor_global"
            ].notna().all(),
            (
                f"{frame_name}: Investor missing "
                "from frozen role registry."
            ),
        )

        require(
            frame[
                "startup_global"
            ].notna().all(),
            (
                f"{frame_name}: Startup missing "
                "from frozen role registry."
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
            - EXPECTED_NUM_INVESTORS
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
                    < EXPECTED_NUM_INVESTORS
                ).all()
            ),
            f"{frame_name}: Investor global outside slice.",
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
                    < EXPECTED_NUM_STARTUPS
                ).all()
            ),
            f"{frame_name}: Startup local outside slice.",
        )

    print(
        f"  T0 events:             {t0_count:,}"
    )

    print(
        f"  T0..T57 events:        {len(history):,}"
    )

    print(
        f"  T1..T57 positives:     {len(training):,}"
    )

    # =========================================================================
    # CANONICAL POSITIVE ORDER
    # =========================================================================

    print()
    print("4. CANONICAL POSITIVE ORDER")

    training[
        "interaction_id"
    ] = (
        training[
            "interaction_id"
        ]
        .astype(str)
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
        "Ordered positive count changed.",
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

    positive_logical_sha = (
        positive_stream_logical_sha256(
            ordered_positive
        )
    )

    ordered_positive.to_parquet(
        POSITIVE_ORDER_PATH,
        index=False,
    )

    print(
        "  Order: interaction_id ASC / stable mergesort"
    )

    print(
        f"  Positive SHA256: {positive_logical_sha}"
    )

    # =========================================================================
    # PAIR FIRST-SEGMENT LOOKUP — T0..T57 ONLY
    # =========================================================================

    print()
    print("5. TEMPORAL NEGATIVE-ELIGIBILITY LOOKUP")

    pair_keys = (
        history[
            "investor_global"
        ].to_numpy(
            dtype=np.int64
        )
        * EXPECTED_NUM_STARTUPS
        + history[
            "startup_local"
        ].to_numpy(
            dtype=np.int64
        )
    )

    first_frame = pd.DataFrame(
        {
            "pair_key":
                pair_keys,

            "segment_number":
                history[
                    "segment_number"
                ].to_numpy(
                    dtype=np.int64
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

    audit_pair_keys = (
        pair_first_series
        .index
        .to_numpy(
            dtype=np.int64
        )
    )

    audit_first_segments = (
        pair_first_series
        .to_numpy(
            dtype=np.int64
        )
    )

    audit_sort = np.argsort(
        audit_pair_keys,
        kind="mergesort",
    )

    audit_pair_keys = (
        audit_pair_keys[
            audit_sort
        ]
    )

    audit_first_segments = (
        audit_first_segments[
            audit_sort
        ]
    )

    del first_frame
    del pair_first_series

    print(
        "  Lookup history: T0..T57 ONLY"
    )

    print(
        f"  Unique observed pairs: "
        f"{len(pair_first_segment):,}"
    )

    print(
        "  T58+ labels used: NO"
    )

    # =========================================================================
    # PHASE-8 SEEDS
    # =========================================================================

    print()
    print("6. PHASE-8 EPOCH-0 RNG SEEDS")

    (
        negative_seed,
        negative_payload,
    ) = derive_seed(
        NEGATIVE_NAMESPACE,
        SCOPE_ID,
        EPOCH,
    )

    (
        order_seed,
        order_payload,
    ) = derive_seed(
        ORDER_NAMESPACE,
        SCOPE_ID,
        EPOCH,
    )

    require(
        negative_seed
        != order_seed,
        "Negative/order seeds unexpectedly collide.",
    )

    print(
        f"  Negative seed: {negative_seed}"
    )

    print(
        f"  Order seed:    {order_seed}"
    )

    # =========================================================================
    # GENERATE EPOCH-0 NEGATIVES
    # =========================================================================

    print()
    print("7. GENERATE EPOCH-0 NEGATIVE MATRIX")

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
        eligible_startup_locals=(
            eligible_startup_locals
        ),
        num_startups=(
            EXPECTED_NUM_STARTUPS
        ),
        seed=(
            negative_seed
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
        "Negative matrix shape mismatch.",
    )

    require(
        negative_matrix.dtype
        == np.int32,
        "Negative matrix dtype changed.",
    )

    # Every sampled Startup must belong to origin_eligible.
    eligible_lookup = np.zeros(
        EXPECTED_NUM_STARTUPS,
        dtype=np.bool_,
    )

    eligible_lookup[
        eligible_startup_locals
    ] = True

    require(
        bool(
            eligible_lookup[
                negative_matrix
            ].all()
        ),
        "Negative matrix contains origin-ineligible Startup.",
    )

    # Four distinct negatives per positive.
    sorted_negatives = np.sort(
        negative_matrix,
        axis=1,
    )

    duplicate_rows = int(
        np.any(
            np.diff(
                sorted_negatives,
                axis=1,
            )
            == 0,
            axis=1,
        ).sum()
    )

    del sorted_negatives

    require(
        duplicate_rows
        == 0,
        "At least one positive has duplicate negatives.",
    )

    focal_collisions = int(
        (
            negative_matrix
            ==
            positive_startup_locals[
                :,
                None
            ]
        ).sum()
    )

    require(
        focal_collisions
        == 0,
        "At least one negative equals focal positive Startup.",
    )

    # Hard temporal collision audit in chunks.
    forbidden_collision_count = 0
    chunk_rows = 100_000

    for start in range(
        0,
        EXPECTED_TRAIN_POSITIVES,
        chunk_rows,
    ):

        end = min(
            start + chunk_rows,
            EXPECTED_TRAIN_POSITIVES,
        )

        chunk_keys = (
            investor_globals[
                start:end,
                None
            ]
            * EXPECTED_NUM_STARTUPS
            + negative_matrix[
                start:end
            ].astype(
                np.int64
            )
        )

        flat_keys = (
            chunk_keys
            .reshape(-1)
        )

        positions = np.searchsorted(
            audit_pair_keys,
            flat_keys,
        )

        exists = (
            positions
            < len(
                audit_pair_keys
            )
        )

        matched = np.zeros(
            len(
                flat_keys
            ),
            dtype=np.bool_,
        )

        valid_positions = positions[
            exists
        ]

        matched[
            exists
        ] = (
            audit_pair_keys[
                valid_positions
            ]
            == flat_keys[
                exists
            ]
        )

        first_segments = np.full(
            len(
                flat_keys
            ),
            fill_value=(
                TRAIN_MAX_SEGMENT
                + 1
            ),
            dtype=np.int64,
        )

        matched_positions = positions[
            matched
        ]

        first_segments[
            matched
        ] = (
            audit_first_segments[
                matched_positions
            ]
        )

        repeated_h = np.repeat(
            target_segments[
                start:end
            ],
            NEGATIVES_PER_POSITIVE,
        )

        forbidden_collision_count += int(
            (
                matched
                & (
                    first_segments
                    <= repeated_h
                )
            ).sum()
        )

    require(
        forbidden_collision_count
        == 0,
        "Negative matrix contains temporally forbidden pair.",
    )

    # Prefix deterministic regeneration, same pattern as Phase 5.
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
        eligible_startup_locals=(
            eligible_startup_locals
        ),
        num_startups=(
            EXPECTED_NUM_STARTUPS
        ),
        seed=(
            negative_seed
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
        "Negative RNG prefix regeneration failed.",
    )

    negative_logical_sha = (
        array_logical_sha256(
            negative_matrix
        )
    )

    np.save(
        NEGATIVE_MATRIX_PATH,
        negative_matrix,
        allow_pickle=False,
    )

    print(
        f"  Matrix:              {negative_matrix.shape}"
    )

    print(
        f"  Repair draws:        "
        f"{negative_stats['repair_draw_count']:,}"
    )

    print(
        f"  Forbidden collisions: {forbidden_collision_count}"
    )

    print(
        "  Prefix regeneration: PASS"
    )

    print(
        f"  Negative SHA256:     {negative_logical_sha}"
    )

    # =========================================================================
    # EPOCH ORDER
    # =========================================================================

    print()
    print("8. EPOCH-0 CONCEPTUAL EXAMPLE ORDER")

    total_examples = (
        EXPECTED_TRAIN_POSITIVES
        * SERIALIZED_SLOTS_PER_POSITIVE
    )

    epoch_order = np.arange(
        total_examples,
        dtype=np.int64,
    )

    order_rng = np.random.Generator(
        np.random.PCG64(
            order_seed
        )
    )

    order_rng.shuffle(
        epoch_order
    )

    require(
        int(
            epoch_order.min()
        )
        == 0,
        "Epoch order minimum changed.",
    )

    require(
        int(
            epoch_order.max()
        )
        == (
            total_examples
            - 1
        ),
        "Epoch order maximum changed.",
    )

    seen = np.zeros(
        total_examples,
        dtype=np.bool_,
    )

    seen[
        epoch_order
    ] = True

    require(
        bool(
            seen.all()
        ),
        "Epoch order is not a complete permutation.",
    )

    del seen

    regenerated_order = np.arange(
        total_examples,
        dtype=np.int64,
    )

    regenerated_rng = np.random.Generator(
        np.random.PCG64(
            order_seed
        )
    )

    regenerated_rng.shuffle(
        regenerated_order
    )

    require(
        np.array_equal(
            regenerated_order,
            epoch_order,
        ),
        "Epoch order full regeneration failed.",
    )

    del regenerated_order

    epoch_order_logical_sha = (
        array_logical_sha256(
            epoch_order
        )
    )

    np.save(
        EPOCH_ORDER_PATH,
        epoch_order,
        allow_pickle=False,
    )

    print(
        f"  Conceptual examples: {total_examples:,}"
    )

    print(
        "  Full regeneration:  PASS"
    )

    print(
        f"  Order SHA256:        {epoch_order_logical_sha}"
    )

    # =========================================================================
    # FIRST BATCH MANIFEST — STREAM ONLY
    # =========================================================================

    print()
    print("9. FIRST 512-EXAMPLE STREAM BATCH")

    first_serialized = (
        epoch_order[
            :BATCH_SIZE
        ]
    )

    first_positive_indices = (
        first_serialized
        // SERIALIZED_SLOTS_PER_POSITIVE
    )

    first_slots = (
        first_serialized
        % SERIALIZED_SLOTS_PER_POSITIVE
    )

    batch_rows = []

    for batch_position in range(
        BATCH_SIZE
    ):

        serialized_index = int(
            first_serialized[
                batch_position
            ]
        )

        positive_index = int(
            first_positive_indices[
                batch_position
            ]
        )

        slot = int(
            first_slots[
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
                slot - 1
            )

            startup_local = int(
                negative_matrix[
                    positive_index,
                    negative_draw_index,
                ]
            )

        startup_global = (
            EXPECTED_NUM_INVESTORS
            + startup_local
        )

        batch_rows.append(
            {
                "batch_position":
                    batch_position,

                "serialized_example_index":
                    serialized_index,

                "positive_order_index":
                    positive_index,

                "example_slot":
                    slot,

                "negative_draw_index":
                    negative_draw_index,

                "label":
                    label,

                "source_interaction_id":
                    str(
                        positive_row[
                            "interaction_id"
                        ]
                    ),

                "investor_id":
                    str(
                        positive_row[
                            "investor_id"
                        ]
                    ),

                "investor_global":
                    investor_global,

                "startup_id":
                    str(
                        startup_raw_by_local[
                            startup_local
                        ]
                    ),

                "startup_local":
                    startup_local,

                "startup_global":
                    startup_global,

                "focal_positive_startup_local":
                    focal_startup_local,

                "segment_number":
                    h,

                "segment_label":
                    f"T{h}",

                "trend_history_period_count":
                    h,
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
        "First batch size changed.",
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
        "First batch label count mismatch.",
    )

    first_batch.to_parquet(
        FIRST_BATCH_PATH,
        index=False,
    )

    segment_rows = []

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

        segment_rows.append(
            {
                "segment_number":
                    int(
                        segment_number
                    ),

                "segment_label":
                    f"T{int(segment_number)}",

                "history_periods_consumed":
                    int(
                        segment_number
                    ),

                "example_count":
                    len(
                        positions
                    ),

                "batch_positions":
                    ";".join(
                        str(
                            position
                        )
                        for position
                        in positions
                    ),
            }
        )

    segment_groups = pd.DataFrame(
        segment_rows
    )

    require(
        int(
            segment_groups[
                "example_count"
            ].sum()
        )
        == BATCH_SIZE,
        "Segment grouping does not cover first batch.",
    )

    require(
        bool(
            (
                segment_groups[
                    "history_periods_consumed"
                ]
                ==
                segment_groups[
                    "segment_number"
                ]
            ).all()
        ),
        "History period count != target segment.",
    )

    segment_groups.to_csv(
        SEGMENT_GROUP_PATH,
        index=False,
    )

    print(
        f"  Positives:            "
        f"{first_batch_positive_count}"
    )

    print(
        f"  Negatives:            "
        f"{first_batch_negative_count}"
    )

    print(
        f"  Distinct segments:    "
        f"{first_batch['segment_number'].nunique()}"
    )

    print(
        f"  Segment range:        "
        f"T{int(first_batch['segment_number'].min())}"
        " .. "
        f"T{int(first_batch['segment_number'].max())}"
    )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    epoch_batches = int(
        math.ceil(
            total_examples
            / BATCH_SIZE
        )
    )

    final_batch_size = int(
        total_examples
        % BATCH_SIZE
    )

    if final_batch_size == 0:
        final_batch_size = BATCH_SIZE

    summary = {
        "schema_version":
            "PHASE_8_2_6_DEV_T58_EPOCH0_TRAINING_STREAM_V1",

        "phase":
            "8.2.6",

        "status":
            "DEV_T58_EPOCH0_STREAM_MATERIALIZED_AND_AUDITED",

        "scope":
            SCOPE_ID,

        "origin_label":
            ORIGIN_LABEL,

        "epoch":
            EPOCH,

        "scientific_role":
            "PRE_TRAINING_STREAM_DRY_RUN",

        "role_universe": {
            "investors":
                EXPECTED_NUM_INVESTORS,

            "startups":
                EXPECTED_NUM_STARTUPS,

            "nodes":
                EXPECTED_NUM_NODES,
        },

        "training_stream": {
            "history_segments":
                "T0..T57",

            "positive_segments":
                "T1..T57",

            "history_events":
                len(
                    history
                ),

            "positive_events":
                len(
                    ordered_positive
                ),

            "negative_positive_ratio":
                NEGATIVES_PER_POSITIVE,

            "negative_examples":
                int(
                    negative_matrix.size
                ),

            "total_examples":
                total_examples,

            "batch_size":
                BATCH_SIZE,

            "drop_last":
                False,

            "epoch_batches":
                epoch_batches,

            "final_batch_size":
                final_batch_size,
        },

        "negative_candidate_universe": {
            "population":
                "T58 origin_eligible Startups",

            "count":
                len(
                    eligible_startup_locals
                ),

            "ordering":
                "startup_node_index ascending",

            "draw_domain":
                "uniform positions in ordered eligible array",

            "all_sampled_negatives_origin_eligible":
                True,
        },

        "negative_integrity": {
            "duplicate_negative_rows":
                duplicate_rows,

            "focal_collisions":
                focal_collisions,

            "forbidden_temporal_collisions":
                forbidden_collision_count,

            "prefix_regeneration_rows":
                REPRO_PREFIX_ROWS,

            "prefix_regeneration_exact":
                True,
        },

        "rng": {
            "negative": {
                "namespace":
                    NEGATIVE_NAMESPACE,

                "payload":
                    negative_payload,

                "seed_u64":
                    negative_seed,

                "bit_generator":
                    "PCG64",
            },

            "order": {
                "namespace":
                    ORDER_NAMESPACE,

                "payload":
                    order_payload,

                "seed_u64":
                    order_seed,

                "bit_generator":
                    "PCG64",
            },

            "model_name_enters_seed":
                False,
        },

        "fingerprints": {
            "positive_order_logical_sha256":
                positive_logical_sha,

            "negative_matrix_logical_sha256":
                negative_logical_sha,

            "epoch_order_logical_sha256":
                epoch_order_logical_sha,
        },

        "negative_generation_stats":
            negative_stats,

        "prefix_generation_stats":
            prefix_stats,

        "first_batch": {
            "size":
                BATCH_SIZE,

            "positives":
                first_batch_positive_count,

            "negatives":
                first_batch_negative_count,

            "distinct_target_segments":
                int(
                    first_batch[
                        "segment_number"
                    ].nunique()
                ),

            "minimum_target_segment":
                int(
                    first_batch[
                        "segment_number"
                    ].min()
                ),

            "maximum_target_segment":
                int(
                    first_batch[
                        "segment_number"
                    ].max()
                ),
        },

        "same_stream_for_models": [
            "B1a",
            "B1b",
            "M1",
            "M2",
        ],

        "execution_state": {
            "real_training_positives_loaded":
                True,

            "real_training_negatives_generated":
                True,

            "epoch_order_generated":
                True,

            "model_instantiated":
                False,

            "forward_executed":
                False,

            "BCE_computed":
                False,

            "optimizer_created":
                False,

            "backward_executed":
                False,

            "optimizer_step_executed":
                False,

            "training_performed":
                False,

            "performance_inspected":
                False,

            "T60_scored":
                False,

            "April_analyzed":
                False,
        },

        "training_allowed_now":
            False,

        "next_step":
            (
                "Audit DEV_T58 epoch-0 stream fingerprints "
                "before any model dry-run."
            ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    artifact_paths = [
        INTERPRETATION_CLOSURE,
        TEMPORAL_SPLIT_PATH,
        NODE_INDEX_PATH,
        ORIGIN_MASK_PATH,
        POSITIVE_ORDER_PATH,
        ELIGIBLE_LOCAL_PATH,
        NEGATIVE_MATRIX_PATH,
        EPOCH_ORDER_PATH,
        FIRST_BATCH_PATH,
        SEGMENT_GROUP_PATH,
        SUMMARY_PATH,
    ]

    hashes = {
        str(path):
            file_sha256(
                path
            )
        for path in artifact_paths
    }

    HASH_REGISTRY_PATH.write_text(
        json.dumps(
            hashes,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("DEV_T58 EPOCH-0 STREAM MATERIALIZATION READY")
    print("=" * 120)

    print(
        f"Positive events:                    "
        f"{len(ordered_positive):,}"
    )

    print(
        f"Eligible negative Startups:         "
        f"{len(eligible_startup_locals):,}"
    )

    print(
        f"Negative examples:                  "
        f"{negative_matrix.size:,}"
    )

    print(
        f"Total conceptual examples:          "
        f"{total_examples:,}"
    )

    print(
        f"Epoch batches:                      "
        f"{epoch_batches:,}"
    )

    print(
        f"Final batch size:                   "
        f"{final_batch_size:,}"
    )

    print()
    print("Positive-order fingerprint:")
    print(positive_logical_sha)

    print()
    print("Negative-matrix fingerprint:")
    print(negative_logical_sha)

    print()
    print("Epoch-order fingerprint:")
    print(epoch_order_logical_sha)

    print()
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Hashes:  {HASH_REGISTRY_PATH}")

    print()
    print("Model instantiated:                 NO")
    print("Forward executed:                   NO")
    print("BCE computed:                       NO")
    print("Optimizer created:                  NO")
    print("Backward executed:                  NO")
    print("Optimizer step:                     NO")
    print("Training performed:                 NO")
    print("Performance inspected:              NO")
    print("T60 scored:                         NO")
    print("April analyzed:                     NO")
    print("Training allowed now:               NO")

    print("=" * 120)


if __name__ == "__main__":
    main()
