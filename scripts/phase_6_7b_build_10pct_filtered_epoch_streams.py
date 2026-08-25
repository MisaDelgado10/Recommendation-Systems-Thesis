#!/usr/bin/env python3

from __future__ import annotations

import gc
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

NUM_EPOCHS = 20

FULL_POSITIVE_COUNT = 1_073_249
REDUCED_POSITIVE_COUNT = 107_324

SLOTS_PER_POSITIVE = 5
NEGATIVES_PER_POSITIVE = 4

REDUCED_EXAMPLES_PER_EPOCH = (
    REDUCED_POSITIVE_COUNT
    * SLOTS_PER_POSITIVE
)

EXPECTED_SUBSET_SHA256 = (
    "bc0aef0b9f840a58c84e32c1a50fc14a"
    "d69a76e688cbe4697dafb7e91d25bddf"
)

FROZEN_PHASE5_COMMIT = (
    "6c94a4e787d2bc7a27e9c1ebced3ddf41132d915"
)

ROUNDTRIP_PATH = Path(
    "scripts/"
    "phase_5_3_2b_checkpoint_resume_roundtrip_proof.py"
)

GENERATOR_PATH = Path(
    "scripts/"
    "phase_5_3_5_generalized_20_epoch_"
    "training_stream_generator_proof_V6.py"
)

PHASE6_7A_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_7a_reduced_training_10pct_contract.json"
)

SELECTION_MANIFEST_PATH = Path(
    "data/experimental/phase_6/reduced_training/"
    "10pct/positive_selection_manifest_10pct.parquet"
)

OUT_DIR = Path(
    "data/experimental/phase_6/"
    "reduced_training/10pct/epoch_streams"
)

REGISTRY_PATH = (
    OUT_DIR
    / "reduced_epoch_stream_registry.csv"
)

SEED_REGISTRY_PATH = (
    OUT_DIR
    / "reduced_epoch_seed_registry.csv"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_7b_10pct_filtered_epoch_stream_contract.json"
)


# =============================================================================
# Helpers
# =============================================================================

def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def banner(text):
    print()
    print("=" * 112)
    print(text)
    print("=" * 112)


def load_module(path, name):
    require(
        path.exists(),
        f"Missing source: {path}",
    )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    require(
        spec is not None
        and spec.loader is not None,
        f"Cannot import {path}",
    )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def reduced_negative_semantic_audit(
    *,
    matrix,
    positive_investor,
    positive_segment,
    generator,
    pair_keys_sorted,
    first_segment_sorted,
):
    require(
        matrix.shape
        == (
            REDUCED_POSITIVE_COUNT,
            NEGATIVES_PER_POSITIVE,
        ),
        "Reduced negative matrix shape drift.",
    )

    require(
        matrix.dtype == np.dtype(np.int32),
        "Reduced negative dtype is not int32.",
    )

    require(
        bool(
            (matrix >= 0).all()
        )
        and bool(
            (
                matrix
                < generator.NUM_STARTUPS
            ).all()
        ),
        "Reduced negatives outside Startup universe.",
    )

    sorted_rows = np.sort(
        matrix,
        axis=1,
    )

    duplicate_rows = int(
        np.count_nonzero(
            np.any(
                sorted_rows[:, 1:]
                == sorted_rows[:, :-1],
                axis=1,
            )
        )
    )

    investors = np.broadcast_to(
        positive_investor[:, None],
        matrix.shape,
    )

    first_segment = (
        generator.lookup_first_segment(
            pair_keys_sorted=pair_keys_sorted,
            first_segment_sorted=first_segment_sorted,
            investor_global=investors,
            startup_local=matrix,
        )
    )

    targets = (
        positive_segment[:, None]
    )

    forbidden = (
        first_segment <= targets
    )

    forbidden_count = int(
        np.count_nonzero(
            forbidden
        )
    )

    future_positive = (
        (first_segment > targets)
        & (first_segment <= 59)
    )

    never_positive = (
        first_segment == 127
    )

    require(
        duplicate_rows == 0,
        "Duplicate negative within a positive.",
    )

    require(
        forbidden_count == 0,
        "Historically forbidden negative found.",
    )

    require(
        int(
            np.count_nonzero(
                future_positive
            )
        )
        + int(
            np.count_nonzero(
                never_positive
            )
        )
        == (
            REDUCED_POSITIVE_COUNT
            * NEGATIVES_PER_POSITIVE
        ),
        "Negative semantic classification incomplete.",
    )

    return {
        "duplicate_rows": duplicate_rows,
        "forbidden_slots": forbidden_count,
        "future_positive_slots": int(
            np.count_nonzero(
                future_positive
            )
        ),
        "never_positive_slots": int(
            np.count_nonzero(
                never_positive
            )
        ),
    }


def filter_full_order(
    *,
    full_order,
    full_to_reduced,
):
    full_order = np.asarray(
        full_order,
        dtype=np.int64,
    )

    full_positive_index = (
        full_order
        // SLOTS_PER_POSITIVE
    )

    slot = (
        full_order
        % SLOTS_PER_POSITIVE
    )

    reduced_positive_index = (
        full_to_reduced[
            full_positive_index
        ]
    )

    keep = (
        reduced_positive_index >= 0
    )

    reduced_order = (
        reduced_positive_index[
            keep
        ]
        * SLOTS_PER_POSITIVE
        + slot[
            keep
        ]
    ).astype(
        np.int64,
        copy=False,
    )

    require(
        len(reduced_order)
        == REDUCED_EXAMPLES_PER_EPOCH,
        "Filtered epoch-order length drift.",
    )

    require(
        int(reduced_order.min()) == 0,
        "Reduced order minimum drift.",
    )

    require(
        int(reduced_order.max())
        == REDUCED_EXAMPLES_PER_EPOCH - 1,
        "Reduced order maximum drift.",
    )

    require(
        len(
            np.unique(
                reduced_order
            )
        )
        == REDUCED_EXAMPLES_PER_EPOCH,
        "Reduced order is not a permutation.",
    )

    return reduced_order


# =============================================================================
# Main
# =============================================================================

def main():

    banner(
        "PHASE 6.7b — BUILD EXACT FILTERED "
        "10% TRAINING STREAMS FOR EPOCHS 0..19"
    )

    print("Model instantiated:            NO")
    print("Optimizer instantiated:        NO")
    print("Training performed:            NO")
    print("Validation accessed:           NO")
    print("Test accessed:                 NO")
    print()
    print(
        "Stream policy:"
    )
    print(
        "  generate original full Phase-5 epoch stream"
    )
    print(
        "  retain selected 10% positive groups only"
    )
    print(
        "  preserve their original per-epoch negatives"
    )
    print(
        "  preserve their relative position in full epoch shuffle"
    )

    # =========================================================================
    # Source gates
    # =========================================================================

    banner(
        "FROZEN SOURCE / SUBSET CONTRACT GATE"
    )

    for path in (
        ROUNDTRIP_PATH,
        GENERATOR_PATH,
    ):
        require(
            path.exists(),
            f"Missing frozen source: {path}",
        )

        result = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                FROZEN_PHASE5_COMMIT,
                "--",
                str(path),
            ],
            check=False,
        )

        require(
            result.returncode == 0,
            (
                "Frozen Phase-5 source changed "
                f"since {FROZEN_PHASE5_COMMIT}: {path}"
            ),
        )

        print(
            f"UNCHANGED  {path}"
        )

    require(
        PHASE6_7A_CONTRACT_PATH.exists(),
        "Missing Phase-6.7a contract.",
    )

    require(
        SELECTION_MANIFEST_PATH.exists(),
        "Missing Phase-6.7a selection manifest.",
    )

    subset_contract = json.loads(
        PHASE6_7A_CONTRACT_PATH.read_text(
            encoding="utf-8"
        )
    )

    require(
        subset_contract[
            "status"
        ]
        == "FROZEN_FOR_10PCT_PILOT",
        "Phase-6.7a subset is not frozen.",
    )

    require(
        subset_contract[
            "selection"
        ][
            "subset_logical_sha256"
        ]
        == EXPECTED_SUBSET_SHA256,
        "Phase-6.7a subset SHA drift.",
    )

    # =========================================================================
    # Load Phase-5 runtimes
    # =========================================================================

    roundtrip = load_module(
        ROUNDTRIP_PATH,
        "_phase6_7b_roundtrip",
    )

    generator = load_module(
        GENERATOR_PATH,
        "_phase6_7b_generator",
    )

    preflight = (
        roundtrip.load_preflight_runtime()
    )

    stream0 = (
        roundtrip.load_epoch0_stream(
            preflight
        )
    )

    full_positive_order = (
        stream0[
            "positive_order"
        ]
    )

    require(
        len(full_positive_order)
        == FULL_POSITIVE_COUNT,
        "Full positive count drift.",
    )

    selection = pd.read_parquet(
        SELECTION_MANIFEST_PATH
    )

    require(
        len(selection)
        == REDUCED_POSITIVE_COUNT,
        "Reduced selection count drift.",
    )

    selected_full_indices = (
        selection[
            "canonical_positive_order_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    require(
        bool(
            (
                selected_full_indices[1:]
                > selected_full_indices[:-1]
            ).all()
        ),
        (
            "Selected full positive indices "
            "must remain in canonical order."
        ),
    )

    reduced_positive = (
        full_positive_order.iloc[
            selected_full_indices
        ][
            [
                "interaction_id",
                "investor_global",
                "startup_local",
                "segment_number",
            ]
        ]
        .reset_index(
            drop=True
        )
    )

    reduced_positive_sha = (
        roundtrip
        .dataframe_logical_sha256(
            reduced_positive,
            columns=[
                "interaction_id",
                "investor_global",
                "startup_local",
                "segment_number",
            ],
        )
    )

    require(
        reduced_positive_sha
        == EXPECTED_SUBSET_SHA256,
        (
            "Filtering canonical positive order "
            "does not reproduce frozen Phase-6.7a subset."
        ),
    )

    print(
        "Phase-6.7a subset reconstruction: EXACT"
    )
    print(
        f"Reduced positives:              "
        f"{len(reduced_positive):,}"
    )
    print(
        f"Reduced examples / epoch:       "
        f"{REDUCED_EXAMPLES_PER_EPOCH:,}"
    )

    # =========================================================================
    # Full -> reduced positive index mapping
    # =========================================================================

    full_to_reduced = np.full(
        FULL_POSITIVE_COUNT,
        -1,
        dtype=np.int64,
    )

    full_to_reduced[
        selected_full_indices
    ] = np.arange(
        REDUCED_POSITIVE_COUNT,
        dtype=np.int64,
    )

    require(
        int(
            np.count_nonzero(
                full_to_reduced >= 0
            )
        )
        == REDUCED_POSITIVE_COUNT,
        "Full->reduced map coverage drift.",
    )

    # =========================================================================
    # Authoritative Phase-2 negative-eligibility index
    # =========================================================================

    banner(
        "BUILD AUTHORITATIVE PRE-T60 NEGATIVE-ELIGIBILITY INDEX"
    )

    (
        pair_keys_sorted,
        first_segment_sorted,
        pair_metadata,
    ) = (
        generator
        .build_first_positive_segment_index_from_phase2(
            temporal_path=(
                generator.TEMPORAL_INTERACTIONS_PATH
            ),
            node_index_path=(
                generator.NODE_INDEX_PATH
            ),
        )
    )

    full_positive_investor = (
        full_positive_order[
            "investor_global"
        ].to_numpy(
            dtype=np.int64
        )
    )

    full_positive_segment = (
        full_positive_order[
            "segment_number"
        ].to_numpy(
            dtype=np.int16
        )
    )

    reduced_positive_investor = (
        reduced_positive[
            "investor_global"
        ].to_numpy(
            dtype=np.int64
        )
    )

    reduced_positive_segment = (
        reduced_positive[
            "segment_number"
        ].to_numpy(
            dtype=np.int16
        )
    )

    reduced_positive_startup = (
        reduced_positive[
            "startup_local"
        ].to_numpy(
            dtype=np.int64
        )
    )

    focal_first_segment = (
        generator.lookup_first_segment(
            pair_keys_sorted=pair_keys_sorted,
            first_segment_sorted=first_segment_sorted,
            investor_global=(
                reduced_positive_investor
            ),
            startup_local=(
                reduced_positive_startup
            ),
        )
    )

    require(
        bool(
            (
                focal_first_segment
                <= reduced_positive_segment
            ).all()
        ),
        (
            "At least one selected positive is "
            "missing from pre-T60 pair history."
        ),
    )

    print(
        "Selected focal positives found by h: PASS"
    )

    # =========================================================================
    # Build all 20 exact filtered epoch streams
    # =========================================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    registry_rows = []
    seed_rows = []

    selected_traversal = (
        "positive_major_immediate_rejection"
    )

    selected_duplicate_scope = (
        "prior_accepted_within_positive"
    )

    start_all = time.perf_counter()

    for epoch in range(
        NUM_EPOCHS
    ):

        banner(
            f"EPOCH {epoch:02d} — FULL STREAM -> 10% FILTER"
        )

        epoch_start = time.perf_counter()

        negative_seed = (
            generator.derive_epoch_seed(
                generator.NEGATIVE_NAMESPACE,
                epoch,
            )
        )

        order_seed = (
            generator.derive_epoch_seed(
                generator.ORDER_NAMESPACE,
                epoch,
            )
        )

        seed_rows.append(
            {
                "epoch_index": epoch,
                "negative_seed": (
                    negative_seed
                ),
                "order_seed": (
                    order_seed
                ),
            }
        )

        # ---------------------------------------------------------------------
        # Epoch 0 already exists as a frozen exact Phase-5 artifact.
        # Epochs 1..19 are regenerated using the frozen generalized generator.
        # ---------------------------------------------------------------------

        if epoch == 0:

            full_negative = np.asarray(
                stream0[
                    "negative_matrix"
                ]
            )

            full_order = np.asarray(
                stream0[
                    "epoch_order"
                ]
            )

            full_negative_source = (
                "FROZEN_PHASE5_EPOCH0_ARTIFACT"
            )

            full_order_source = (
                "FROZEN_PHASE5_EPOCH0_ARTIFACT"
            )

        else:

            (
                full_negative,
                full_negative_metadata,
            ) = (
                generator
                .generate_epoch_negative_matrix(
                    epoch_index=epoch,
                    positive_investor=(
                        full_positive_investor
                    ),
                    positive_segment=(
                        full_positive_segment
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
                    progress=False,
                )
            )

            (
                full_order,
                regenerated_order_seed,
            ) = (
                generator
                .generate_epoch_order(
                    epoch
                )
            )

            require(
                int(
                    full_negative_metadata[
                        "negative_seed"
                    ]
                )
                == negative_seed,
                "Negative seed drift.",
            )

            require(
                int(
                    regenerated_order_seed
                )
                == order_seed,
                "Order seed drift.",
            )

            full_negative_source = (
                "FROZEN_PHASE5_GENERALIZED_GENERATOR"
            )

            full_order_source = (
                "FROZEN_PHASE5_GENERALIZED_GENERATOR"
            )

        # ---------------------------------------------------------------------
        # Filter negative rows by selected canonical positive indices.
        # ---------------------------------------------------------------------

        reduced_negative = np.ascontiguousarray(
            full_negative[
                selected_full_indices
            ],
            dtype=np.int32,
        )

        require(
            reduced_negative.shape
            == (
                REDUCED_POSITIVE_COUNT,
                NEGATIVES_PER_POSITIVE,
            ),
            "Filtered negative shape drift.",
        )

        # ---------------------------------------------------------------------
        # Filter full epoch permutation while preserving relative ordering.
        # Then remap full serialized indices into compact 10% serialization.
        # ---------------------------------------------------------------------

        reduced_order = filter_full_order(
            full_order=full_order,
            full_to_reduced=full_to_reduced,
        )

        semantic = (
            reduced_negative_semantic_audit(
                matrix=reduced_negative,
                positive_investor=(
                    reduced_positive_investor
                ),
                positive_segment=(
                    reduced_positive_segment
                ),
                generator=generator,
                pair_keys_sorted=(
                    pair_keys_sorted
                ),
                first_segment_sorted=(
                    first_segment_sorted
                ),
            )
        )

        negative_sha = (
            preflight
            .array_logical_sha256(
                reduced_negative
            )
        )

        order_sha = (
            preflight
            .array_logical_sha256(
                reduced_order
            )
        )

        negative_path = (
            OUT_DIR
            / (
                f"epoch_{epoch:02d}_"
                "negative_startup_local.npy"
            )
        )

        order_path = (
            OUT_DIR
            / (
                f"epoch_{epoch:02d}_"
                "example_order.npy"
            )
        )

        np.save(
            negative_path,
            reduced_negative,
            allow_pickle=False,
        )

        np.save(
            order_path,
            reduced_order,
            allow_pickle=False,
        )

        # Reload from disk immediately.
        negative_reload = np.load(
            negative_path,
            mmap_mode="r",
        )

        order_reload = np.load(
            order_path,
            mmap_mode="r",
        )

        require(
            preflight.array_logical_sha256(
                np.asarray(
                    negative_reload
                )
            )
            == negative_sha,
            "Saved negative matrix reload drift.",
        )

        require(
            preflight.array_logical_sha256(
                np.asarray(
                    order_reload
                )
            )
            == order_sha,
            "Saved epoch order reload drift.",
        )

        elapsed = (
            time.perf_counter()
            - epoch_start
        )

        registry_rows.append(
            {
                "epoch_index": epoch,
                "positive_events": (
                    REDUCED_POSITIVE_COUNT
                ),
                "negative_events": (
                    REDUCED_POSITIVE_COUNT
                    * NEGATIVES_PER_POSITIVE
                ),
                "serialized_examples": (
                    REDUCED_EXAMPLES_PER_EPOCH
                ),
                "negative_seed": (
                    negative_seed
                ),
                "order_seed": (
                    order_seed
                ),
                "negative_sha256": (
                    negative_sha
                ),
                "order_sha256": (
                    order_sha
                ),
                "duplicate_negative_rows": (
                    semantic[
                        "duplicate_rows"
                    ]
                ),
                "forbidden_negative_slots": (
                    semantic[
                        "forbidden_slots"
                    ]
                ),
                "future_positive_negative_slots": (
                    semantic[
                        "future_positive_slots"
                    ]
                ),
                "never_positive_negative_slots": (
                    semantic[
                        "never_positive_slots"
                    ]
                ),
                "full_negative_source": (
                    full_negative_source
                ),
                "full_order_source": (
                    full_order_source
                ),
                "relative_full_order_preserved": (
                    True
                ),
                "elapsed_seconds": (
                    elapsed
                ),
            }
        )

        print(
            f"Reduced negative SHA:  {negative_sha}"
        )
        print(
            f"Reduced order SHA:     {order_sha}"
        )
        print(
            "Forbidden slots:       0"
        )
        print(
            "Duplicate rows:        0"
        )
        print(
            f"Elapsed:               {elapsed:.2f} s"
        )

        if epoch != 0:
            del full_negative
            del full_order

        del reduced_negative
        del reduced_order
        gc.collect()

    # =========================================================================
    # Persist registry / contract
    # =========================================================================

    registry = pd.DataFrame(
        registry_rows
    )

    seeds = pd.DataFrame(
        seed_rows
    )

    require(
        len(registry)
        == NUM_EPOCHS,
        "Epoch registry does not contain 20 rows.",
    )

    require(
        bool(
            (
                registry[
                    "forbidden_negative_slots"
                ]
                == 0
            ).all()
        ),
        "At least one epoch has forbidden negatives.",
    )

    require(
        bool(
            (
                registry[
                    "duplicate_negative_rows"
                ]
                == 0
            ).all()
        ),
        "At least one epoch has duplicate negatives.",
    )

    require(
        registry[
            "negative_sha256"
        ].nunique()
        == NUM_EPOCHS,
        "Reduced negative matrices are not distinct across epochs.",
    )

    require(
        registry[
            "order_sha256"
        ].nunique()
        == NUM_EPOCHS,
        "Reduced epoch orders are not distinct across epochs.",
    )

    registry.to_csv(
        REGISTRY_PATH,
        index=False,
    )

    seeds.to_csv(
        SEED_REGISTRY_PATH,
        index=False,
    )

    total_seconds = (
        time.perf_counter()
        - start_all
    )

    contract = {
        "phase": "6.6b",
        "status": "FROZEN_FOR_10PCT_PILOT",
        "experiment": (
            "10PCT_FILTERED_FULL_PHASE5_TRAINING_STREAM"
        ),

        "positive_subset": {
            "events": (
                REDUCED_POSITIVE_COUNT
            ),
            "logical_sha256": (
                EXPECTED_SUBSET_SHA256
            ),
        },

        "stream_construction": {
            "epochs": (
                NUM_EPOCHS
            ),
            "slots_per_positive": (
                SLOTS_PER_POSITIVE
            ),
            "negatives_per_positive": (
                NEGATIVES_PER_POSITIVE
            ),
            "examples_per_epoch": (
                REDUCED_EXAMPLES_PER_EPOCH
            ),

            "negative_policy": (
                "generate original frozen Phase-5 full "
                "epoch negative stream, then retain rows "
                "for selected Phase-6.7a positives"
            ),

            "order_policy": (
                "generate original frozen Phase-5 full "
                "epoch permutation, filter to selected "
                "positive groups while preserving relative "
                "order, then remap to compact 10% serialization"
            ),

            "retained_negative_draws_identical_to_full_run": (
                True
            ),

            "retained_relative_full_epoch_order_identical": (
                True
            ),

            "negative_regeneration_independent_each_epoch": (
                True
            ),
        },

        "semantics": {
            "negative_eligibility": (
                "candidate forbidden iff first positive "
                "segment <= target h"
            ),
            "without_replacement_within_positive": (
                True
            ),
            "future_positive_with_first_segment_gt_h_eligible": (
                True
            ),
            "T60_used": (
                False
            ),
        },

        "registry": str(
            REGISTRY_PATH
        ),

        "seed_registry": str(
            SEED_REGISTRY_PATH
        ),

        "boundary": {
            "model_instantiated": False,
            "optimizer_instantiated": False,
            "training_performed": False,
            "validation_accessed": False,
            "test_accessed": False,
        },
    }

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.write_text(
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # =========================================================================
    # Final
    # =========================================================================

    banner(
        "PHASE 6.7b FINAL STATUS"
    )

    print(
        f"Epoch streams created:         "
        f"{len(registry)} / 20"
    )
    print(
        f"Positive groups / epoch:       "
        f"{REDUCED_POSITIVE_COUNT:,}"
    )
    print(
        f"Serialized examples / epoch:   "
        f"{REDUCED_EXAMPLES_PER_EPOCH:,}"
    )
    print(
        "Forbidden negative slots:      0 / ALL EPOCHS"
    )
    print(
        "Duplicate negative rows:       0 / ALL EPOCHS"
    )
    print(
        "Distinct negative streams:     20 / 20"
    )
    print(
        "Distinct epoch orders:         20 / 20"
    )
    print(
        "Retained negatives match full Phase-5 stream: YES"
    )
    print(
        "Relative full-epoch order preserved:          YES"
    )
    print(
        "Validation accessed:                         NO"
    )
    print(
        "Test accessed:                               NO"
    )
    print(
        f"Total generation time:        "
        f"{total_seconds / 60:.2f} min"
    )
    print()
    print(
        f"WROTE  {REGISTRY_PATH}"
    )
    print(
        f"WROTE  {SEED_REGISTRY_PATH}"
    )
    print(
        f"WROTE  {CONTRACT_PATH}"
    )
    print()
    print(
        "PHASE 6.7b: PASS / "
        "20 FILTERED 10% EPOCH STREAMS FROZEN"
    )


if __name__ == "__main__":
    main()
