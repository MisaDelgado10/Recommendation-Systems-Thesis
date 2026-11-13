#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import hashlib
import importlib
import json
import gc
import sys

import numpy as np
import pandas as pd


# =============================================================================
# REPOSITORY IMPORT PATH
# =============================================================================
#
# When this script is executed as:
#
#     python scripts/<script>.py
#
# Python places the scripts/ directory itself at sys.path[0].
# Add the repository root explicitly so namespace imports such as
# scripts.phase_8_2_6_dev_t57_epoch0_stream_materialization_V1 work
# deterministically.
#
REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


# =============================================================================
# PHASE 8.2.6 — DEVELOPMENT FOLD / EPOCH FINGERPRINT COVERAGE V2
# =============================================================================

PHASE = "8.2.6"

EPOCH_INDICES = list(range(20))

SCOPE_MODULES = {
    "DEV_T57":
        "scripts.phase_8_2_6_dev_t57_epoch0_stream_materialization_V1",

    "DEV_T58":
        "scripts.phase_8_2_6_dev_t58_epoch0_stream_materialization_V1",

    "DEV_T59":
        "scripts.phase_8_2_6_dev_t59_epoch0_stream_materialization_V1",
}

ROOT = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_6_training_stream_dry_run"
)

OUTPUT_CSV = (
    ROOT
    / "phase_8_2_6_development_epoch_fingerprints_V2.csv"
)

OUTPUT_MANIFEST = (
    ROOT
    / "phase_8_2_6_development_epoch_fingerprint_manifest_V2.json"
)

OUTPUT_HASHES = (
    ROOT
    / "phase_8_2_6_development_epoch_fingerprint_sha256_V2.json"
)

TEMPORAL_SPLIT_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "interactions_itrs_temporal_split.parquet"
)

NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/model_ready/"
    "node_index.parquet"
)

EXPECTED_NUM_INVESTORS = 165_975
EXPECTED_NUM_STARTUPS = 311_589

PREFIX_ROWS = 10_000
BATCH_SIZE = 512


# =============================================================================
# HELPERS
# =============================================================================

def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def build_role_maps():

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

    node_type = (
        node_index[
            "node_type"
        ]
        .astype(str)
        .str.lower()
    )

    investors = (
        node_index.loc[
            node_type.eq(
                "investor"
            )
        ]
        .copy()
    )

    startups = (
        node_index.loc[
            node_type.eq(
                "startup"
            )
        ]
        .copy()
    )

    require(
        len(investors)
        == EXPECTED_NUM_INVESTORS,
        "Investor role count changed.",
    )

    require(
        len(startups)
        == EXPECTED_NUM_STARTUPS,
        "Startup role count changed.",
    )

    investor_map = dict(
        zip(
            investors[
                "raw_entity_id"
            ].astype(str),

            investors[
                "node_index"
            ].astype(
                np.int64
            ),
        )
    )

    startup_map = dict(
        zip(
            startups[
                "raw_entity_id"
            ].astype(str),

            startups[
                "node_index"
            ].astype(
                np.int64
            ),
        )
    )

    return (
        investor_map,
        startup_map,
    )


def build_pair_first_segment(
    *,
    temporal: pd.DataFrame,
    investor_map: dict,
    startup_map: dict,
    max_segment: int,
):

    history = (
        temporal.loc[
            temporal[
                "segment_number"
            ].between(
                0,
                max_segment,
                inclusive="both",
            )
        ]
        .copy()
    )

    history[
        "investor_global"
    ] = (
        history[
            "investor_id"
        ]
        .astype(str)
        .map(
            investor_map
        )
    )

    history[
        "startup_global"
    ] = (
        history[
            "startup_id"
        ]
        .astype(str)
        .map(
            startup_map
        )
    )

    require(
        history[
            "investor_global"
        ].notna().all(),
        "History Investor mapping failed.",
    )

    require(
        history[
            "startup_global"
        ].notna().all(),
        "History Startup mapping failed.",
    )

    investor_global = (
        history[
            "investor_global"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    startup_local = (
        history[
            "startup_global"
        ]
        .to_numpy(
            dtype=np.int64
        )
        - EXPECTED_NUM_INVESTORS
    )

    segment_number = (
        history[
            "segment_number"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    pair_key = (
        investor_global
        * EXPECTED_NUM_STARTUPS
        + startup_local
    )

    first_frame = pd.DataFrame(
        {
            "pair_key":
                pair_key,

            "segment_number":
                segment_number,
        }
    )

    first_series = (
        first_frame
        .groupby(
            "pair_key",
            sort=False,
        )[
            "segment_number"
        ]
        .min()
    )

    lookup = dict(
        zip(
            first_series
            .index
            .to_numpy(
                dtype=np.int64
            )
            .tolist(),

            first_series
            .to_numpy(
                dtype=np.int64
            )
            .tolist(),
        )
    )

    return (
        lookup,
        int(
            len(
                history
            )
        ),
        int(
            len(
                lookup
            )
        ),
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 120)
    print(
        "PHASE 8.2.6 — DEVELOPMENT FOLD / EPOCH "
        "FINGERPRINT COVERAGE V1"
    )
    print("=" * 120)

    require(
        TEMPORAL_SPLIT_PATH.exists(),
        f"Missing {TEMPORAL_SPLIT_PATH}",
    )

    require(
        NODE_INDEX_PATH.exists(),
        f"Missing {NODE_INDEX_PATH}",
    )

    ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("1. LOAD FROZEN ROLE REGISTRY")

    (
        investor_map,
        startup_map,
    ) = build_role_maps()

    print(
        "  Investor / Startup role maps: PASS"
    )

    temporal = pd.read_parquet(
        TEMPORAL_SPLIT_PATH,
        columns=[
            "investor_id",
            "startup_id",
            "segment_number",
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

    rows = []

    # =========================================================================
    # EACH DEVELOPMENT FOLD
    # =========================================================================

    for scope_id, module_name in (
        SCOPE_MODULES.items()
    ):

        module = importlib.import_module(
            module_name
        )

        require(
            module.SCOPE_ID
            == scope_id,
            f"{scope_id}: imported module scope mismatch.",
        )

        max_segment = int(
            module.TRAIN_MAX_SEGMENT
        )

        origin_label = str(
            module.ORIGIN_LABEL
        )

        expected_positives = int(
            module.EXPECTED_TRAIN_POSITIVES
        )

        expected_eligible = int(
            module.EXPECTED_ELIGIBLE_STARTUPS
        )

        print()
        print("-" * 120)
        print(
            f"{scope_id} — {origin_label} — "
            f"T1..T{max_segment}"
        )
        print("-" * 120)

        epoch0_dir = (
            ROOT
            / scope_id
            / "epoch_0"
        )

        summary_path = (
            epoch0_dir
            / (
                "phase_8_2_6_"
                f"{scope_id.lower()}_"
                "epoch0_stream_summary_V1.json"
            )
        )

        closure_path = (
            epoch0_dir
            / (
                "phase_8_2_6_"
                f"{scope_id.lower()}_"
                "epoch0_manifest_V1.json"
            )
        )

        require(
            summary_path.exists(),
            (
                f"{scope_id}: missing frozen "
                f"epoch-0 summary: {summary_path}"
            ),
        )

        require(
            closure_path.exists(),
            (
                f"{scope_id}: missing frozen "
                f"epoch-0 closure: {closure_path}"
            ),
        )

        epoch0_summary = json.loads(
            summary_path.read_text(
                encoding="utf-8"
            )
        )

        epoch0_closure = json.loads(
            closure_path.read_text(
                encoding="utf-8"
            )
        )

        require(
            epoch0_closure["status"]
            == (
                f"{scope_id}_"
                "EPOCH0_STREAM_FROZEN"
            ),
            f"{scope_id}: epoch-0 fixture not frozen.",
        )

        positive_path = (
            epoch0_dir
            / "canonical_training_positive_event_order.parquet"
        )

        eligible_path = (
            epoch0_dir
            / "origin_eligible_startup_local.npy"
        )

        require(
            positive_path.exists(),
            f"{scope_id}: positive fixture missing.",
        )

        require(
            eligible_path.exists(),
            f"{scope_id}: eligible fixture missing.",
        )

        positives = pd.read_parquet(
            positive_path,
            columns=[
                "positive_order_index",
                "investor_global",
                "startup_local",
                "segment_number",
            ],
        )

        require(
            len(positives)
            == expected_positives,
            f"{scope_id}: positive count changed.",
        )

        investor_globals = (
            positives[
                "investor_global"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        positive_startup_locals = (
            positives[
                "startup_local"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        target_segments = (
            positives[
                "segment_number"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        eligible_startup_locals = np.load(
            eligible_path,
            allow_pickle=False,
        ).astype(
            np.int64,
            copy=False,
        )

        require(
            len(
                eligible_startup_locals
            )
            == expected_eligible,
            f"{scope_id}: eligible count changed.",
        )

        eligible_lookup = np.zeros(
            EXPECTED_NUM_STARTUPS,
            dtype=np.bool_,
        )

        eligible_lookup[
            eligible_startup_locals
        ] = True

        print(
            f"  positives:        "
            f"{expected_positives:,}"
        )

        print(
            f"  eligible startups:"
            f" {expected_eligible:,}"
        )

        print(
            "  build temporal pair lookup..."
        )

        (
            pair_first_segment,
            history_count,
            pair_count,
        ) = build_pair_first_segment(
            temporal=temporal,
            investor_map=investor_map,
            startup_map=startup_map,
            max_segment=max_segment,
        )

        print(
            f"  history events:   "
            f"{history_count:,}"
        )

        print(
            f"  unique pairs:     "
            f"{pair_count:,}"
        )

        # =====================================================================
        # EPOCH 0 — READ AUTHORITATIVE FIXTURE ONLY
        # =====================================================================

        fp0 = epoch0_summary[
            "fingerprints"
        ]

        rng0 = epoch0_summary[
            "rng"
        ]

        first0 = epoch0_summary[
            "first_batch"
        ]

        stats0 = epoch0_summary[
            "negative_generation_stats"
        ]

        rows.append(
            {
                "scope_id":
                    scope_id,

                "origin_label":
                    origin_label,

                "epoch_index":
                    0,

                "display_epoch":
                    1,

                "training_max_segment":
                    max_segment,

                "positive_events":
                    expected_positives,

                "eligible_startups":
                    expected_eligible,

                "negative_seed_u64":
                    int(
                        rng0[
                            "negative"
                        ][
                            "seed_u64"
                        ]
                    ),

                "order_seed_u64":
                    int(
                        rng0[
                            "order"
                        ][
                            "seed_u64"
                        ]
                    ),

                "positive_order_sha256":
                    fp0[
                        "positive_order_logical_sha256"
                    ],

                "negative_matrix_sha256":
                    fp0[
                        "negative_matrix_logical_sha256"
                    ],

                "epoch_order_sha256":
                    fp0[
                        "epoch_order_logical_sha256"
                    ],

                "repair_draw_count":
                    int(
                        stats0[
                            "repair_draw_count"
                        ]
                    ),

                "forbidden_rejections":
                    int(
                        stats0[
                            "forbidden_rejections"
                        ]
                    ),

                "duplicate_rejections":
                    int(
                        stats0[
                            "duplicate_rejections"
                        ]
                    ),

                "prefix_regeneration_exact":
                    True,

                "order_regeneration_exact":
                    True,

                "all_negatives_origin_eligible":
                    True,

                "within_positive_unique_negatives":
                    True,

                "focal_collision_count":
                    0,

                "first_batch_positives":
                    int(
                        first0[
                            "positives"
                        ]
                    ),

                "first_batch_negatives":
                    int(
                        first0[
                            "negatives"
                        ]
                    ),

                "first_batch_distinct_segments":
                    int(
                        first0[
                            "distinct_target_segments"
                        ]
                    ),

                "first_batch_min_segment":
                    int(
                        first0[
                            "minimum_target_segment"
                        ]
                    ),

                "first_batch_max_segment":
                    int(
                        first0[
                            "maximum_target_segment"
                        ]
                    ),

                "full_arrays_persisted":
                    True,

                "source":
                    "FROZEN_EPOCH0_FIXTURE",
            }
        )

        print(
            "  epoch 00: existing frozen fixture PASS"
        )

        # =====================================================================
        # EPOCHS 1..19
        # =====================================================================

        for epoch_index in range(
            1,
            20,
        ):

            (
                negative_seed,
                negative_payload,
            ) = module.derive_seed(
                module.NEGATIVE_NAMESPACE,
                scope_id,
                epoch_index,
            )

            (
                order_seed,
                order_payload,
            ) = module.derive_seed(
                module.ORDER_NAMESPACE,
                scope_id,
                epoch_index,
            )

            require(
                f"scope={scope_id}"
                in negative_payload,
                f"{scope_id} epoch {epoch_index}: scope missing from negative seed.",
            )

            require(
                f"epoch={epoch_index}"
                in negative_payload,
                f"{scope_id} epoch {epoch_index}: epoch missing from negative seed.",
            )

            require(
                "model="
                not in negative_payload.lower(),
                f"{scope_id} epoch {epoch_index}: model entered negative seed.",
            )

            require(
                f"scope={scope_id}"
                in order_payload,
                f"{scope_id} epoch {epoch_index}: scope missing from order seed.",
            )

            require(
                f"epoch={epoch_index}"
                in order_payload,
                f"{scope_id} epoch {epoch_index}: epoch missing from order seed.",
            )

            require(
                "model="
                not in order_payload.lower(),
                f"{scope_id} epoch {epoch_index}: model entered order seed.",
            )

            (
                negatives,
                stats,
            ) = module.generate_epoch_negatives(
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
                progress=False,
            )

            require(
                negatives.shape
                == (
                    expected_positives,
                    4,
                ),
                (
                    f"{scope_id} epoch {epoch_index}: "
                    "negative matrix shape mismatch."
                ),
            )

            require(
                negatives.dtype
                == np.int32,
                (
                    f"{scope_id} epoch {epoch_index}: "
                    "negative dtype mismatch."
                ),
            )

            require(
                bool(
                    eligible_lookup[
                        negatives
                    ].all()
                ),
                (
                    f"{scope_id} epoch {epoch_index}: "
                    "origin-ineligible negative sampled."
                ),
            )

            sorted_negatives = np.sort(
                negatives,
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
                duplicate_rows == 0,
                (
                    f"{scope_id} epoch {epoch_index}: "
                    "duplicate negatives within focal positive."
                ),
            )

            focal_collisions = int(
                (
                    negatives
                    ==
                    positive_startup_locals[
                        :,
                        None
                    ]
                ).sum()
            )

            require(
                focal_collisions == 0,
                (
                    f"{scope_id} epoch {epoch_index}: "
                    "negative equals focal Startup."
                ),
            )

            (
                prefix,
                _prefix_stats,
            ) = module.generate_epoch_negatives(
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
                    PREFIX_ROWS
                ),
                progress=False,
            )

            prefix_exact = bool(
                np.array_equal(
                    prefix,
                    negatives[
                        :PREFIX_ROWS
                    ],
                )
            )

            require(
                prefix_exact,
                (
                    f"{scope_id} epoch {epoch_index}: "
                    "negative prefix regeneration failed."
                ),
            )

            negative_sha = (
                module.array_logical_sha256(
                    negatives
                )
            )

            total_examples = (
                expected_positives
                * 5
            )

            order = np.arange(
                total_examples,
                dtype=np.int64,
            )

            order_rng = np.random.Generator(
                np.random.PCG64(
                    order_seed
                )
            )

            order_rng.shuffle(
                order
            )

            regenerated_order = np.arange(
                total_examples,
                dtype=np.int64,
            )

            regen_rng = np.random.Generator(
                np.random.PCG64(
                    order_seed
                )
            )

            regen_rng.shuffle(
                regenerated_order
            )

            order_exact = bool(
                np.array_equal(
                    order,
                    regenerated_order,
                )
            )

            require(
                order_exact,
                (
                    f"{scope_id} epoch {epoch_index}: "
                    "order regeneration failed."
                ),
            )

            order_sha = (
                module.array_logical_sha256(
                    order
                )
            )

            first = order[
                :BATCH_SIZE
            ]

            first_positive_index = (
                first // 5
            )

            first_slot = (
                first % 5
            )

            first_target_segments = (
                target_segments[
                    first_positive_index
                ]
            )

            first_positive_count = int(
                (
                    first_slot == 0
                ).sum()
            )

            first_negative_count = int(
                BATCH_SIZE
                - first_positive_count
            )

            rows.append(
                {
                    "scope_id":
                        scope_id,

                    "origin_label":
                        origin_label,

                    "epoch_index":
                        epoch_index,

                    "display_epoch":
                        epoch_index + 1,

                    "training_max_segment":
                        max_segment,

                    "positive_events":
                        expected_positives,

                    "eligible_startups":
                        expected_eligible,

                    "negative_seed_u64":
                        int(
                            negative_seed
                        ),

                    "order_seed_u64":
                        int(
                            order_seed
                        ),

                    "positive_order_sha256":
                        fp0[
                            "positive_order_logical_sha256"
                        ],

                    "negative_matrix_sha256":
                        negative_sha,

                    "epoch_order_sha256":
                        order_sha,

                    "repair_draw_count":
                        int(
                            stats[
                                "repair_draw_count"
                            ]
                        ),

                    "forbidden_rejections":
                        int(
                            stats[
                                "forbidden_rejections"
                            ]
                        ),

                    "duplicate_rejections":
                        int(
                            stats[
                                "duplicate_rejections"
                            ]
                        ),

                    "prefix_regeneration_exact":
                        prefix_exact,

                    "order_regeneration_exact":
                        order_exact,

                    "all_negatives_origin_eligible":
                        True,

                    "within_positive_unique_negatives":
                        True,

                    "focal_collision_count":
                        focal_collisions,

                    "first_batch_positives":
                        first_positive_count,

                    "first_batch_negatives":
                        first_negative_count,

                    "first_batch_distinct_segments":
                        int(
                            np.unique(
                                first_target_segments
                            ).size
                        ),

                    "first_batch_min_segment":
                        int(
                            first_target_segments.min()
                        ),

                    "first_batch_max_segment":
                        int(
                            first_target_segments.max()
                        ),

                    "full_arrays_persisted":
                        False,

                    "source":
                        "REGENERATED_FINGERPRINT_ONLY",
                }
            )

            print(
                f"  epoch {epoch_index:02d}: "
                f"neg={negative_sha[:12]}... "
                f"order={order_sha[:12]}... "
                f"repairs={stats['repair_draw_count']:,} "
                f"PASS"
            )

            del negatives
            del prefix
            del order
            del regenerated_order
            del first
            del first_positive_index
            del first_slot
            del first_target_segments

            gc.collect()

        del pair_first_segment
        del positives
        del investor_globals
        del positive_startup_locals
        del target_segments
        del eligible_startup_locals
        del eligible_lookup

        gc.collect()

    # =========================================================================
    # GLOBAL COVERAGE AUDIT
    # =========================================================================

    print()
    print("=" * 120)
    print("2. GLOBAL DEVELOPMENT COVERAGE AUDIT")
    print("=" * 120)

    registry = pd.DataFrame(
        rows
    )

    require(
        len(registry) == 60,
        (
            "Expected exactly 60 development "
            "scope/epoch fingerprints."
        ),
    )

    for scope_id in SCOPE_MODULES:

        scope_rows = (
            registry.loc[
                registry[
                    "scope_id"
                ].eq(
                    scope_id
                )
            ]
            .sort_values(
                "epoch_index"
            )
        )

        require(
            scope_rows[
                "epoch_index"
            ].tolist()
            == list(
                range(20)
            ),
            (
                f"{scope_id}: incomplete epoch "
                "coverage."
            ),
        )

        require(
            scope_rows[
                "positive_order_sha256"
            ].nunique()
            == 1,
            (
                f"{scope_id}: positive order "
                "changed across epochs."
            ),
        )

        require(
            scope_rows[
                "negative_matrix_sha256"
            ].nunique()
            == 20,
            (
                f"{scope_id}: negative matrix "
                "fingerprint repeated across epochs."
            ),
        )

        require(
            scope_rows[
                "epoch_order_sha256"
            ].nunique()
            == 20,
            (
                f"{scope_id}: epoch-order "
                "fingerprint repeated across epochs."
            ),
        )

        require(
            scope_rows[
                "prefix_regeneration_exact"
            ].all(),
            f"{scope_id}: prefix reproducibility failed.",
        )

        require(
            scope_rows[
                "order_regeneration_exact"
            ].all(),
            f"{scope_id}: order reproducibility failed.",
        )

        require(
            scope_rows[
                "all_negatives_origin_eligible"
            ].all(),
            f"{scope_id}: ineligible negative detected.",
        )

        require(
            scope_rows[
                "within_positive_unique_negatives"
            ].all(),
            f"{scope_id}: duplicate negative detected.",
        )

        require(
            scope_rows[
                "focal_collision_count"
            ].eq(
                0
            ).all(),
            f"{scope_id}: focal collision detected.",
        )

    require(
        registry[
            "negative_seed_u64"
        ].nunique()
        == 60,
        "Negative RNG seed collision across development scope/epochs.",
    )

    require(
        registry[
            "order_seed_u64"
        ].nunique()
        == 60,
        "Order RNG seed collision across development scope/epochs.",
    )

    registry = registry.sort_values(
        [
            "scope_id",
            "epoch_index",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    registry.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    manifest = {
        "schema_version":
            "PHASE_8_2_6_DEVELOPMENT_EPOCH_FINGERPRINT_COVERAGE_V2",

        "phase":
            PHASE,

        "status":
            "DEVELOPMENT_FOLD_EPOCH_STREAMS_FINGERPRINTED",

        "scientific_role":
            "PRE_TRAINING_STREAM_INTEGRATION_DRY_RUN",

        "development_scopes": [
            "DEV_T57",
            "DEV_T58",
            "DEV_T59",
        ],

        "rng_epoch_indices":
            list(
                range(20)
            ),

        "display_epochs":
            list(
                range(1, 21)
            ),

        "scope_epoch_rows":
            60,

        "negative_sampling": {
            "namespace":
                "ITRS_PHASE8_TRAIN_NEGATIVE",

            "generator":
                "numpy.random.PCG64",

            "resample_frequency":
                "once per epoch",

            "negative_positive_ratio":
                4,

            "model_name_enters_seed":
                False,

            "same_stream_across_models_within_scope_epoch":
                True,
        },

        "training_order": {
            "namespace":
                "ITRS_PHASE8_TRAIN_ORDER",

            "generator":
                "numpy.random.PCG64",

            "shuffle_frequency":
                "once per epoch",

            "model_name_enters_seed":
                False,

            "same_stream_across_models_within_scope_epoch":
                True,
        },

        "models_sharing_each_stream": [
            "B1a",
            "B1b",
            "M1",
            "M2",
        ],

        "persistence_policy": {
            "epoch_0":
                (
                    "Full authoritative negative and order "
                    "arrays retained as fixtures."
                ),

            "epochs_1_19":
                (
                    "Full arrays regenerated for audit and hashing, "
                    "then discarded; deterministic fingerprints retained."
                ),
        },

        "coverage_invariants": {
            "all_60_scope_epoch_rows_present":
                True,

            "positive_order_constant_within_scope":
                True,

            "negative_fingerprint_unique_within_scope":
                True,

            "order_fingerprint_unique_within_scope":
                True,

            "negative_seeds_unique_across_development_scope_epochs":
                True,

            "order_seeds_unique_across_development_scope_epochs":
                True,

            "negative_prefix_reproducibility":
                True,

            "order_full_regeneration_reproducibility":
                True,

            "all_sampled_negatives_origin_eligible":
                True,

            "all_negatives_unique_within_focal_positive":
                True,

            "no_negative_equals_focal_positive":
                True,
        },

        "final_scope_policy": {
            "T60_FINAL_IMPROVED":
                (
                    "NOT MATERIALIZED HERE. V5 requires the "
                    "model-specific development-selected epoch."
                ),

            "APRIL_FINAL_M2":
                (
                    "NOT MATERIALIZED HERE. V5 requires the "
                    "M2 development-selected epoch."
                ),
        },

        "execution_state": {
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
                "Formal Phase 8.2.6 closure, then "
                "Phase 8.2.7 one-step autograd/optimizer preflight."
            ),
    }

    OUTPUT_MANIFEST.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    hashes = {
        str(OUTPUT_CSV):
            sha256_file(
                OUTPUT_CSV
            ),

        str(OUTPUT_MANIFEST):
            sha256_file(
                OUTPUT_MANIFEST
            ),
    }

    OUTPUT_HASHES.write_text(
        json.dumps(
            hashes,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("Development scopes:             3")
    print("Epochs per scope:               20")
    print("Scope/epoch fingerprints:       60")
    print("Positive-order stability:       PASS")
    print("Negative fingerprint coverage:  PASS")
    print("Order fingerprint coverage:     PASS")
    print("Negative seed uniqueness:       PASS")
    print("Order seed uniqueness:          PASS")
    print("Prefix reproducibility:         PASS")
    print("Order reproducibility:          PASS")
    print("Candidate eligibility:          PASS")
    print("Within-positive uniqueness:     PASS")
    print("Focal collisions:               0")

    print()
    print(f"Registry: {OUTPUT_CSV}")
    print(f"Manifest: {OUTPUT_MANIFEST}")
    print(f"Hashes:   {OUTPUT_HASHES}")

    print()
    print("Model instantiated:             NO")
    print("Forward executed:               NO")
    print("BCE computed:                   NO")
    print("Optimizer created:              NO")
    print("Backward executed:              NO")
    print("Training performed:             NO")
    print("Performance inspected:          NO")
    print("T60 scored:                     NO")
    print("April analyzed:                 NO")
    print("Training allowed now:           NO")

    print()
    print("=" * 120)
    print(
        "STATUS: DEVELOPMENT_FOLD_EPOCH_STREAMS_FINGERPRINTED"
    )
    print(
        "NEXT:   Formal Phase 8.2.6 closure"
    )
    print("=" * 120)


if __name__ == "__main__":
    main()
