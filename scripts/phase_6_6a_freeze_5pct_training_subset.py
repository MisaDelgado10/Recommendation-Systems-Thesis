#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# Phase 6.6a — 5% reduced-supervision experimental protocol
# =============================================================================

EXPERIMENT_SCHEMA = "ITRS_PHASE6_REDUCED_TRAINING_V1"

FRACTION_NUMERATOR = 5
FRACTION_DENOMINATOR = 100

BASE_SEED = 42
SELECTION_NAMESPACE = "ITRS_PHASE6_REDUCED_POSITIVE"

FULL_POSITIVE_COUNT = 1_073_249

NEGATIVES_PER_POSITIVE = 4
SLOTS_PER_POSITIVE = 5
BATCH_SIZE = 512
NUM_EPOCHS = 20

EXPECTED_FULL_POSITIVE_SHA256 = (
    "73b074a80675793b811fbdc8a0609883"
    "c857fb2a687a2e01c31865ade5b509d1"
)

FROZEN_PHASE5_COMMIT = (
    "6c94a4e787d2bc7a27e9c1ebced3ddf41132d915"
)

ROUNDTRIP_PATH = Path(
    "scripts/"
    "phase_5_3_2b_checkpoint_resume_roundtrip_proof.py"
)

OUT_DIR = Path(
    "data/experimental/phase_6/"
    "reduced_training/5pct"
)

POSITIVE_PATH = (
    OUT_DIR
    / "positive_order_5pct.parquet"
)

SELECTION_MANIFEST_PATH = (
    OUT_DIR
    / "positive_selection_manifest_5pct.parquet"
)

STRATUM_AUDIT_PATH = (
    OUT_DIR
    / "stratum_allocation_audit.csv"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_6a_reduced_training_5pct_contract.json"
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

    module = importlib.util.module_from_spec(spec)

    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def build_subset(
    positive_order: pd.DataFrame,
):
    work = positive_order[
        [
            "interaction_id",
            "investor_global",
            "startup_local",
            "segment_number",
        ]
    ].copy()

    work.insert(
        0,
        "canonical_positive_order_index",
        np.arange(
            len(work),
            dtype=np.int64,
        ),
    )

    # =========================================================================
    # Investor historical training activity.
    #
    # IMPORTANT:
    # Computed ONLY from the canonical T1..T59 training-positive stream.
    # No T60 validation/test information is used.
    # =========================================================================

    degree = (
        work.groupby(
            "investor_global",
            sort=False,
        )
        .size()
        .rename(
            "investor_training_positive_count"
        )
        .reset_index()
    )

    degree = degree.sort_values(
        [
            "investor_training_positive_count",
            "investor_global",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    investor_count = len(degree)

    degree[
        "activity_quartile"
    ] = (
        (
            np.arange(
                investor_count,
                dtype=np.int64,
            )
            * 4
        )
        // investor_count
    ).astype(
        np.int8
    )

    require(
        set(
            degree[
                "activity_quartile"
            ].unique()
        )
        == {0, 1, 2, 3},
        "Investor activity quartiles are incomplete.",
    )

    work = work.merge(
        degree,
        on="investor_global",
        how="left",
        validate="many_to_one",
    )

    require(
        work[
            "activity_quartile"
        ].notna().all(),
        "Missing investor activity quartile.",
    )

    # =========================================================================
    # Exact floor(5%) positive-event target.
    # =========================================================================

    target_total = (
        len(work)
        * FRACTION_NUMERATOR
        // FRACTION_DENOMINATOR
    )

    # =========================================================================
    # Strata:
    #
    #       temporal segment × investor training-activity quartile
    #
    # Give every non-empty stratum one event, then allocate remaining slots
    # proportionally with deterministic Hamilton allocation.
    # =========================================================================

    strata = (
        work.groupby(
            [
                "segment_number",
                "activity_quartile",
            ],
            sort=True,
        )
        .size()
        .rename(
            "full_count"
        )
        .reset_index()
    )

    nonempty_strata = len(strata)

    require(
        target_total >= nonempty_strata,
        (
            "5% target is too small to preserve "
            "one event per non-empty stratum."
        ),
    )

    strata[
        "selected_count"
    ] = 1

    remaining_target = (
        target_total
        - nonempty_strata
    )

    residual_capacity = (
        strata[
            "full_count"
        ].to_numpy(
            dtype=np.int64
        )
        - 1
    )

    total_residual_capacity = int(
        residual_capacity.sum()
    )

    numerators = (
        residual_capacity
        * remaining_target
    )

    additional = (
        numerators
        // total_residual_capacity
    )

    remainders = (
        numerators
        % total_residual_capacity
    )

    strata[
        "selected_count"
    ] += additional

    undistributed = (
        target_total
        - int(
            strata[
                "selected_count"
            ].sum()
        )
    )

    require(
        undistributed >= 0,
        "Hamilton allocation over-allocated.",
    )

    if undistributed > 0:

        remainder_order = (
            strata.assign(
                __remainder=remainders
            )
            .sort_values(
                [
                    "__remainder",
                    "segment_number",
                    "activity_quartile",
                ],
                ascending=[
                    False,
                    True,
                    True,
                ],
                kind="mergesort",
            )
            .index
            .to_numpy()
        )

        chosen = (
            remainder_order[
                :undistributed
            ]
        )

        strata.loc[
            chosen,
            "selected_count",
        ] += 1

    require(
        int(
            strata[
                "selected_count"
            ].sum()
        )
        == target_total,
        (
            "Stratified allocation does not "
            "equal exact target."
        ),
    )

    require(
        bool(
            (
                strata[
                    "selected_count"
                ]
                <= strata[
                    "full_count"
                ]
            ).all()
        ),
        (
            "A stratum was allocated more "
            "events than it contains."
        ),
    )

    # =========================================================================
    # Deterministic selection within each stratum.
    # =========================================================================

    work[
        "selection_sha256"
    ] = [
        hashlib.sha256(
            (
                f"{SELECTION_NAMESPACE}|"
                f"{BASE_SEED}|"
                f"{interaction_id}|"
                f"{int(investor_global)}|"
                f"{int(startup_local)}|"
                f"{int(segment_number)}"
            ).encode(
                "utf-8"
            )
        ).hexdigest()
        for (
            interaction_id,
            investor_global,
            startup_local,
            segment_number,
        ) in zip(
            work[
                "interaction_id"
            ].astype(str),
            work[
                "investor_global"
            ].to_numpy(),
            work[
                "startup_local"
            ].to_numpy(),
            work[
                "segment_number"
            ].to_numpy(),
        )
    ]

    allocation = strata[
        [
            "segment_number",
            "activity_quartile",
            "selected_count",
        ]
    ]

    work = work.merge(
        allocation,
        on=[
            "segment_number",
            "activity_quartile",
        ],
        how="left",
        validate="many_to_one",
    )

    ranked = work.sort_values(
        [
            "segment_number",
            "activity_quartile",
            "selection_sha256",
            "canonical_positive_order_index",
        ],
        kind="mergesort",
    ).copy()

    ranked[
        "within_stratum_selection_rank"
    ] = (
        ranked.groupby(
            [
                "segment_number",
                "activity_quartile",
            ],
            sort=False,
        )
        .cumcount()
    )

    selected = ranked.loc[
        ranked[
            "within_stratum_selection_rank"
        ]
        < ranked[
            "selected_count"
        ]
    ].copy()

    # Restore the selected events to their original canonical positive order.
    selected = selected.sort_values(
        [
            "canonical_positive_order_index",
        ],
        kind="mergesort",
    ).reset_index(
        drop=True
    )

    require(
        len(selected)
        == target_total,
        "Selected positive-event count drift.",
    )

    return (
        selected,
        strata,
        degree,
    )


# =============================================================================
# Main
# =============================================================================

def main():

    banner(
        "PHASE 6.6a — FREEZE 5% REDUCED-SUPERVISION "
        "TRAINING EXPERIMENT"
    )

    print("Neural model instantiated:       NO")
    print("Negative examples generated:     NO")
    print("Validation accessed:             NO")
    print("Test accessed:                   NO")
    print("Graph reduced:                   NO")
    print("T0..T59 context reduced:         NO")

    # =========================================================================
    # Phase-5 source integrity
    # =========================================================================

    banner(
        "FROZEN PHASE-5 SOURCE GATE"
    )

    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            FROZEN_PHASE5_COMMIT,
            "--",
            str(ROUNDTRIP_PATH),
        ],
        check=False,
    )

    require(
        result.returncode == 0,
        (
            "Frozen Phase-5 runtime changed since "
            f"{FROZEN_PHASE5_COMMIT}: {ROUNDTRIP_PATH}"
        ),
    )

    print(
        f"UNCHANGED  {ROUNDTRIP_PATH}"
    )

    current_commit = (
        subprocess.check_output(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            text=True,
        )
        .strip()
    )

    # =========================================================================
    # Load canonical positive-event stream
    # =========================================================================

    roundtrip = load_module(
        ROUNDTRIP_PATH,
        "_phase6_6a_roundtrip",
    )

    preflight = (
        roundtrip.load_preflight_runtime()
    )

    positive_order = pd.read_parquet(
        preflight.POSITIVE_ORDER_PATH
    )

    require(
        len(positive_order)
        == FULL_POSITIVE_COUNT,
        "Canonical positive-event count drift.",
    )

    full_positive_sha = (
        preflight
        .positive_stream_logical_sha256(
            positive_order
        )
    )

    require(
        full_positive_sha
        == EXPECTED_FULL_POSITIVE_SHA256,
        "Canonical positive-event SHA drift.",
    )

    required_columns = {
        "interaction_id",
        "investor_global",
        "startup_local",
        "segment_number",
    }

    require(
        required_columns.issubset(
            positive_order.columns
        ),
        (
            "Canonical positive stream "
            "missing required columns."
        ),
    )

    require(
        bool(
            positive_order[
                "segment_number"
            ]
            .between(
                1,
                59,
            )
            .all()
        ),
        (
            "Training source contains target "
            "outside T1..T59."
        ),
    )

    print()
    print(
        f"Canonical positives:            "
        f"{len(positive_order):,}"
    )
    print(
        "Canonical positive SHA256:"
    )
    print(
        full_positive_sha
    )

    # =========================================================================
    # Build subset twice for determinism proof
    # =========================================================================

    banner(
        "DETERMINISTIC TEMPORAL × INVESTOR-ACTIVITY "
        "STRATIFIED SELECTION"
    )

    (
        selected_a,
        strata,
        degree,
    ) = build_subset(
        positive_order
    )

    (
        selected_b,
        _,
        _,
    ) = build_subset(
        positive_order
    )

    index_a = (
        selected_a[
            "canonical_positive_order_index"
        ].to_numpy(
            dtype=np.int64
        )
    )

    index_b = (
        selected_b[
            "canonical_positive_order_index"
        ].to_numpy(
            dtype=np.int64
        )
    )

    require(
        np.array_equal(
            index_a,
            index_b,
        ),
        (
            "Independent subset "
            "regeneration differs."
        ),
    )

    target_positive_count = len(
        selected_a
    )

    reduced_examples = (
        target_positive_count
        * SLOTS_PER_POSITIVE
    )

    batches_per_epoch = int(
        math.ceil(
            reduced_examples
            / BATCH_SIZE
        )
    )

    final_batch_size = (
        reduced_examples
        - (
            batches_per_epoch - 1
        )
        * BATCH_SIZE
    )

    optimizer_steps = (
        batches_per_epoch
        * NUM_EPOCHS
    )

    core = selected_a[
        [
            "interaction_id",
            "investor_global",
            "startup_local",
            "segment_number",
        ]
    ].copy()

    subset_sha = (
        roundtrip
        .dataframe_logical_sha256(
            core,
            columns=[
                "interaction_id",
                "investor_global",
                "startup_local",
                "segment_number",
            ],
        )
    )

    # =========================================================================
    # Integrity / coverage checks
    # =========================================================================

    source_segments = set(
        positive_order[
            "segment_number"
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    selected_segments = set(
        core[
            "segment_number"
        ]
        .astype(int)
        .unique()
        .tolist()
    )

    require(
        selected_segments == source_segments,
        (
            "Reduced sample lost at least "
            "one training temporal segment."
        ),
    )

    require(
        set(
            selected_a[
                "activity_quartile"
            ]
            .astype(int)
            .unique()
            .tolist()
        )
        == {0, 1, 2, 3},
        (
            "Reduced sample lost an "
            "investor activity quartile."
        ),
    )

    require(
        selected_a[
            "canonical_positive_order_index"
        ].is_unique,
        (
            "Selected canonical positive "
            "indices are duplicated."
        ),
    )

    require(
        bool(
            core[
                "segment_number"
            ]
            .between(
                1,
                59,
            )
            .all()
        ),
        "Reduced sample contains T60.",
    )

    stratum_selected = (
        selected_a.groupby(
            [
                "segment_number",
                "activity_quartile",
            ],
            sort=True,
        )
        .size()
        .rename(
            "observed_selected_count"
        )
        .reset_index()
    )

    stratum_audit = strata.merge(
        stratum_selected,
        on=[
            "segment_number",
            "activity_quartile",
        ],
        how="left",
        validate="one_to_one",
    )

    stratum_audit[
        "observed_selected_count"
    ] = (
        stratum_audit[
            "observed_selected_count"
        ]
        .fillna(0)
        .astype(int)
    )

    require(
        bool(
            (
                stratum_audit[
                    "selected_count"
                ]
                == stratum_audit[
                    "observed_selected_count"
                ]
            ).all()
        ),
        (
            "Observed stratum counts "
            "differ from allocation."
        ),
    )

    stratum_audit[
        "full_share"
    ] = (
        stratum_audit[
            "full_count"
        ]
        / FULL_POSITIVE_COUNT
    )

    stratum_audit[
        "selected_share"
    ] = (
        stratum_audit[
            "observed_selected_count"
        ]
        / target_positive_count
    )

    # =========================================================================
    # Write experimental artifacts
    # =========================================================================

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    core.to_parquet(
        POSITIVE_PATH,
        index=False,
    )

    selected_a.to_parquet(
        SELECTION_MANIFEST_PATH,
        index=False,
    )

    stratum_audit.to_csv(
        STRATUM_AUDIT_PATH,
        index=False,
    )

    contract = {
        "phase": "6.6a",
        "schema_version": EXPERIMENT_SCHEMA,
        "status": "FROZEN_FOR_5PCT_PILOT",

        "experiment_type": (
            "REDUCED_SUPERVISED_TRAINING_EVENT_BUDGET"
        ),

        "not_claimed_as": (
            "FULL_DATASET_ITRS_REPRODUCTION"
        ),

        "source": {
            "full_positive_events": (
                FULL_POSITIVE_COUNT
            ),
            "full_positive_logical_sha256": (
                full_positive_sha
            ),
            "frozen_phase5_commit": (
                FROZEN_PHASE5_COMMIT
            ),
            "current_repository_commit": (
                current_commit
            ),
            "target_segments": (
                "T1..T59"
            ),
        },

        "selection": {
            "fraction_numerator": (
                FRACTION_NUMERATOR
            ),
            "fraction_denominator": (
                FRACTION_DENOMINATOR
            ),
            "target_positive_events": (
                target_positive_count
            ),
            "selection_namespace": (
                SELECTION_NAMESPACE
            ),
            "selection_seed": (
                BASE_SEED
            ),

            "stratification": [
                "segment_number",
                "investor_training_activity_quartile",
            ],

            "activity_definition": (
                "count of canonical T1..T59 positive "
                "training events per investor"
            ),

            "activity_quartile_rule": (
                "investors sorted by "
                "(training_positive_count, investor_global); "
                "equal-count deterministic quartiles"
            ),

            "stratum_allocation": (
                "minimum one event per non-empty stratum, "
                "then integer Hamilton proportional allocation"
            ),

            "within_stratum_selection": (
                "ascending SHA256 of "
                "namespace|seed|interaction_id|"
                "investor_global|startup_local|segment_number"
            ),

            "final_positive_order": (
                "original canonical positive-event order"
            ),

            "subset_logical_sha256": (
                subset_sha
            ),

            "independent_regeneration_exact": (
                True
            ),
        },

        "training_arithmetic": {
            "positive_events_per_epoch": (
                target_positive_count
            ),
            "negative_events_per_epoch": (
                target_positive_count
                * NEGATIVES_PER_POSITIVE
            ),
            "examples_per_epoch": (
                reduced_examples
            ),
            "batch_size": (
                BATCH_SIZE
            ),
            "batches_per_epoch": (
                batches_per_epoch
            ),
            "final_batch_size": (
                final_batch_size
            ),
            "epochs": (
                NUM_EPOCHS
            ),
            "optimizer_steps": (
                optimizer_steps
            ),
        },

        "inherited_scientific_semantics": {
            "full_T0_T59_graph_retained": True,
            "full_T0_T59_trend_history_retained": True,
            "full_features_retained": True,
            "model_architecture_unchanged": True,

            "negatives_per_positive": 4,
            "negative_eligibility_rule_unchanged": True,
            "independent_negative_regeneration_each_epoch": True,

            "batch_size_unchanged": True,
            "Adam_unchanged": True,
            "epoch_count_unchanged": True,

            "validation_split_unchanged": True,
            "validation_candidate_sets_unchanged": True,
            "ranking_semantics_unchanged": True,

            "test_split_untouched": True,
        },

        "boundary": {
            "model_instantiated": False,
            "training_performed": False,
            "validation_accessed": False,
            "test_accessed": False,
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

    # =========================================================================
    # Final report
    # =========================================================================

    banner(
        "PHASE 6.6a — 5% PILOT SUBSET RESULT"
    )

    print(
        f"Full training positives:       "
        f"{FULL_POSITIVE_COUNT:,}"
    )

    print(
        f"Selected positives:            "
        f"{target_positive_count:,}"
    )

    print(
        f"Actual fraction:               "
        f"{target_positive_count / FULL_POSITIVE_COUNT:.6%}"
    )

    print(
        f"Selected investors:            "
        f"{core['investor_global'].nunique():,}"
    )

    print(
        f"Selected startups:             "
        f"{core['startup_local'].nunique():,}"
    )

    print(
        f"Temporal segments retained:    "
        f"{len(selected_segments)} / "
        f"{len(source_segments)}"
    )

    print(
        "Activity quartiles retained:   4 / 4"
    )

    print()

    print(
        f"Positive examples / epoch:     "
        f"{target_positive_count:,}"
    )

    print(
        f"Negative examples / epoch:     "
        f"{target_positive_count * 4:,}"
    )

    print(
        f"Total examples / epoch:        "
        f"{reduced_examples:,}"
    )

    print(
        f"Batches / epoch:               "
        f"{batches_per_epoch:,}"
    )

    print(
        f"Final batch size:               "
        f"{final_batch_size:,}"
    )

    print(
        f"20-epoch optimizer steps:       "
        f"{optimizer_steps:,}"
    )

    print()
    print(
        "Subset logical SHA256:"
    )
    print(
        subset_sha
    )

    print()
    print(
        "Independent regeneration:      EXACT"
    )
    print(
        "T60 used in selection:          NO"
    )
    print(
        "Graph reduced:                  NO"
    )
    print(
        "Validation accessed:            NO"
    )
    print(
        "Test accessed:                  NO"
    )

    print()
    print(
        f"WROTE  {POSITIVE_PATH}"
    )
    print(
        f"WROTE  {SELECTION_MANIFEST_PATH}"
    )
    print(
        f"WROTE  {STRATUM_AUDIT_PATH}"
    )
    print(
        f"WROTE  {CONTRACT_PATH}"
    )

    print()
    print(
        "PHASE 6.6a: PASS / "
        "5% PILOT SUBSET FROZEN"
    )


if __name__ == "__main__":
    main()
