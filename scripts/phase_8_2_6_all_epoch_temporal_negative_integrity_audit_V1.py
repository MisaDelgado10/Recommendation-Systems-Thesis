#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import gc
import hashlib
import importlib
import json
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 8.2.6 — ALL-EPOCH TEMPORAL NEGATIVE INTEGRITY AUDIT V1
# =============================================================================
#
# Scientific role:
#   Supplemental independent integrity audit over the already-frozen
#   deterministic development training streams.
#
# Scope:
#   DEV_T57 / DEV_T58 / DEV_T59
#   epoch_index 0..19
#   60 total deterministic negative streams
#
# Central invariant:
#
#   For every accepted negative pair (investor, startup) attached to a
#   positive event at target segment h:
#
#       pair absent from pre-origin history
#       OR
#       first_positive_segment(pair) > h
#
#   Therefore:
#
#       first_positive_segment(pair) <= h
#
#   is FORBIDDEN and must occur zero times.
#
# Future-positive negatives are intentionally allowed by the frozen V5
# protocol when first_positive_segment(pair) > h.
#
# IMPORTANT:
#   - Epoch 0: read existing persisted authoritative negative matrix.
#   - Epochs 1..19: deterministically regenerate from frozen seed and discard.
#   - Every matrix fingerprint must match the existing V2 fingerprint registry.
#   - No model instantiation.
#   - No forward.
#   - No BCE.
#   - No optimizer.
#   - No backward.
#   - No training.
#   - No performance inspection.
#   - No T60.
#   - No April.
# =============================================================================


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


ROOT = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_6_training_stream_dry_run"
)


REGISTRY_PATH = (
    ROOT
    / "phase_8_2_6_development_epoch_fingerprints_V2.csv"
)

COVERAGE_MANIFEST = (
    ROOT
    / "phase_8_2_6_development_epoch_fingerprint_manifest_V2.json"
)

COVERAGE_HASHES = (
    ROOT
    / "phase_8_2_6_development_epoch_fingerprint_sha256_V2.json"
)

INTERPRETATION = (
    ROOT
    / "phase_8_2_6_training_stream_interpretation_closure_V1.json"
)

CLOSURE_V2 = (
    ROOT
    / "phase_8_2_6_training_stream_integration_closure_manifest_V2.json"
)


OUTPUT_CSV = (
    ROOT
    / "phase_8_2_6_all_epoch_temporal_negative_integrity_V1.csv"
)

OUTPUT_MANIFEST = (
    ROOT
    / "phase_8_2_6_all_epoch_temporal_negative_integrity_manifest_V1.json"
)

OUTPUT_HASHES = (
    ROOT
    / "phase_8_2_6_all_epoch_temporal_negative_integrity_sha256_V1.json"
)


SCOPE_MODULES = {
    "DEV_T57":
        "scripts.phase_8_2_6_dev_t57_epoch0_stream_materialization_V1",

    "DEV_T58":
        "scripts.phase_8_2_6_dev_t58_epoch0_stream_materialization_V1",

    "DEV_T59":
        "scripts.phase_8_2_6_dev_t59_epoch0_stream_materialization_V1",
}


CHUNK_ROWS = 100_000
EXPECTED_EPOCHS = list(range(20))
NEGATIVES_PER_POSITIVE = 4


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


def load_json(path: Path):

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def temporal_collision_audit(
    *,
    investor_globals: np.ndarray,
    target_segments: np.ndarray,
    negatives: np.ndarray,
    sorted_pair_keys: np.ndarray,
    sorted_first_segments: np.ndarray,
    num_startups: int,
):
    """
    Independent vectorized audit of accepted negative pairs.

    A negative is forbidden iff the Investor-Startup pair has a known
    first positive segment <= the focal target segment h.

    A known pair with first_segment > h is deliberately allowed.
    An unseen pair is deliberately allowed.
    """

    forbidden = 0
    future_positive_allowed = 0
    historically_unobserved = 0
    matched_history_pairs = 0

    total_rows = int(
        len(
            investor_globals
        )
    )

    for start in range(
        0,
        total_rows,
        CHUNK_ROWS,
    ):

        end = min(
            start + CHUNK_ROWS,
            total_rows,
        )

        chunk_keys = (
            investor_globals[
                start:end,
                None
            ]
            * num_startups
            +
            negatives[
                start:end
            ].astype(
                np.int64,
                copy=False,
            )
        )

        flat_keys = (
            chunk_keys
            .reshape(-1)
        )

        positions = np.searchsorted(
            sorted_pair_keys,
            flat_keys,
        )

        in_bounds = (
            positions
            < len(
                sorted_pair_keys
            )
        )

        matched = np.zeros(
            len(
                flat_keys
            ),
            dtype=np.bool_,
        )

        if bool(
            in_bounds.any()
        ):

            valid_positions = (
                positions[
                    in_bounds
                ]
            )

            matched[
                in_bounds
            ] = (
                sorted_pair_keys[
                    valid_positions
                ]
                ==
                flat_keys[
                    in_bounds
                ]
            )

        matched_count = int(
            matched.sum()
        )

        matched_history_pairs += (
            matched_count
        )

        historically_unobserved += (
            len(
                flat_keys
            )
            - matched_count
        )

        if matched_count > 0:

            first_segments = (
                sorted_first_segments[
                    positions[
                        matched
                    ]
                ]
            )

            repeated_h = np.repeat(
                target_segments[
                    start:end
                ],
                NEGATIVES_PER_POSITIVE,
            )

            matched_h = (
                repeated_h[
                    matched
                ]
            )

            forbidden += int(
                (
                    first_segments
                    <= matched_h
                ).sum()
            )

            future_positive_allowed += int(
                (
                    first_segments
                    > matched_h
                ).sum()
            )

        del chunk_keys
        del flat_keys
        del positions
        del in_bounds
        del matched

    return {
        "forbidden_temporal_collisions":
            int(
                forbidden
            ),

        "known_future_positive_negatives_allowed":
            int(
                future_positive_allowed
            ),

        "historically_unobserved_negatives":
            int(
                historically_unobserved
            ),

        "matched_history_pair_negatives":
            int(
                matched_history_pairs
            ),
    }


print("=" * 120)
print("PHASE 8.2.6 — ALL-EPOCH TEMPORAL NEGATIVE INTEGRITY AUDIT V1")
print("=" * 120)


# =============================================================================
# 1. PREREQUISITES
# =============================================================================

print()
print("1. PREREQUISITES")

required_paths = [
    REGISTRY_PATH,
    COVERAGE_MANIFEST,
    COVERAGE_HASHES,
    INTERPRETATION,
    CLOSURE_V2,
]

for path in required_paths:

    require(
        path.exists(),
        f"Missing prerequisite: {path}",
    )


interpretation = load_json(
    INTERPRETATION
)

coverage_manifest = load_json(
    COVERAGE_MANIFEST
)

closure_v2 = load_json(
    CLOSURE_V2
)


require(
    interpretation[
        "status"
    ]
    == "FROZEN_BEFORE_STREAM_GENERATION",
    "8.2.6 interpretation is not frozen.",
)

require(
    coverage_manifest[
        "status"
    ]
    == "DEVELOPMENT_FOLD_EPOCH_STREAMS_FINGERPRINTED",
    "Development coverage is not frozen.",
)

require(
    coverage_manifest[
        "scope_epoch_rows"
    ]
    == 60,
    "Coverage manifest does not contain 60 streams.",
)

require(
    closure_v2[
        "status"
    ]
    == "TRAINING_STREAM_INTEGRATION_DRY_RUN_CLOSED",
    "8.2.6 V2 closure is not closed.",
)

require(
    closure_v2[
        "training_allowed_now"
    ]
    is False,
    "8.2.6 V2 unexpectedly permits training.",
)


coverage_hash_registry = load_json(
    COVERAGE_HASHES
)

for path in [
    REGISTRY_PATH,
    COVERAGE_MANIFEST,
]:

    require(
        str(path)
        in coverage_hash_registry,
        f"Coverage hash registry missing {path}",
    )

    require(
        sha256_file(
            path
        )
        ==
        coverage_hash_registry[
            str(path)
        ],
        f"Coverage artifact hash mismatch: {path}",
    )


print("  Interpretation contract: PASS")
print("  Coverage manifest:       PASS")
print("  Coverage registry hash:  PASS")
print("  8.2.6 V2 closure:        PASS")
print("  Training allowed:        NO")


# =============================================================================
# 2. FROZEN 60-STREAM REGISTRY
# =============================================================================

print()
print("2. LOAD FROZEN 60-STREAM REGISTRY")

registry = pd.read_csv(
    REGISTRY_PATH
)

required_columns = {
    "scope_id",
    "origin_label",
    "epoch_index",
    "display_epoch",
    "training_max_segment",
    "positive_events",
    "eligible_startups",
    "negative_seed_u64",
    "positive_order_sha256",
    "negative_matrix_sha256",
    "all_negatives_origin_eligible",
    "within_positive_unique_negatives",
    "focal_collision_count",
}

missing_columns = (
    required_columns
    - set(
        registry.columns
    )
)

require(
    not missing_columns,
    (
        "Coverage registry missing columns: "
        f"{sorted(missing_columns)}"
    ),
)

require(
    len(
        registry
    )
    == 60,
    "Coverage registry is not 60 rows.",
)

require(
    not registry.duplicated(
        subset=[
            "scope_id",
            "epoch_index",
        ]
    ).any(),
    "Duplicate scope/epoch registry row.",
)

require(
    set(
        registry[
            "scope_id"
        ].astype(str)
    )
    == set(
        SCOPE_MODULES.keys()
    ),
    "Unexpected development scopes.",
)

for scope_id in SCOPE_MODULES:

    scope_epochs = (
        registry.loc[
            registry[
                "scope_id"
            ].astype(str)
            .eq(
                scope_id
            ),
            "epoch_index",
        ]
        .astype(int)
        .sort_values()
        .tolist()
    )

    require(
        scope_epochs
        == EXPECTED_EPOCHS,
        f"{scope_id}: epochs are not exactly 0..19.",
    )


print("  Development scopes: 3 / 3")
print("  Epochs per scope:   20 / 20")
print("  Total streams:      60 / 60")


# =============================================================================
# 3. ALL-SCOPE / ALL-EPOCH AUDIT
# =============================================================================

print()
print("=" * 120)
print("3. INDEPENDENT TEMPORAL AUDIT — 60 STREAMS")
print("=" * 120)


audit_rows = []


for (
    scope_id,
    module_name,
) in SCOPE_MODULES.items():

    print()
    print("-" * 120)
    print(scope_id)
    print("-" * 120)

    module = importlib.import_module(
        module_name
    )

    require(
        module.SCOPE_ID
        == scope_id,
        f"{scope_id}: imported module scope mismatch.",
    )

    num_investors = int(
        module.EXPECTED_NUM_INVESTORS
    )

    num_startups = int(
        module.EXPECTED_NUM_STARTUPS
    )

    expected_positives = int(
        module.EXPECTED_TRAIN_POSITIVES
    )

    expected_eligible = int(
        module.EXPECTED_ELIGIBLE_STARTUPS
    )

    max_segment = int(
        module.TRAIN_MAX_SEGMENT
    )

    # -------------------------------------------------------------------------
    # Authoritative positive fixture.
    # -------------------------------------------------------------------------

    epoch0_dir = (
        ROOT
        / scope_id
        / "epoch_0"
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

    positives = (
        positives
        .sort_values(
            "positive_order_index",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    require(
        len(
            positives
        )
        == expected_positives,
        f"{scope_id}: positive count changed.",
    )

    require(
        positives[
            "positive_order_index"
        ].to_numpy(
            dtype=np.int64
        ).tolist()
        ==
        list(
            range(
                expected_positives
            )
        ),
        f"{scope_id}: positive order index changed.",
    )

    investor_globals = (
        positives[
            "investor_global"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    focal_startup_locals = (
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

    require(
        bool(
            (
                target_segments
                >= module.TRAIN_MIN_SEGMENT
            ).all()
        )
        and bool(
            (
                target_segments
                <= max_segment
            ).all()
        ),
        f"{scope_id}: target segment range changed.",
    )

    # -------------------------------------------------------------------------
    # Origin-eligible universe.
    # -------------------------------------------------------------------------

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
        f"{scope_id}: eligible Startup count changed.",
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
        f"{scope_id}: eligible array no longer strictly ascending.",
    )

    eligible_lookup = np.zeros(
        num_startups,
        dtype=np.bool_,
    )

    eligible_lookup[
        eligible_startup_locals
    ] = True

    # -------------------------------------------------------------------------
    # Build role maps independently from frozen Phase-3 role registry.
    # -------------------------------------------------------------------------

    node_index = pd.read_parquet(
        module.NODE_INDEX_PATH,
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
        == num_investors,
        f"{scope_id}: Investor role count drift.",
    )

    require(
        len(
            startup_nodes
        )
        == num_startups,
        f"{scope_id}: Startup role count drift.",
    )

    investor_map = dict(
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

    startup_map = dict(
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

    # -------------------------------------------------------------------------
    # Independently construct first-positive-segment lookup using ONLY
    # T0..T(max_segment). No later labels are consumed.
    # -------------------------------------------------------------------------

    temporal = pd.read_parquet(
        module.TEMPORAL_SPLIT_PATH,
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
        f"{scope_id}: Investor history mapping failed.",
    )

    require(
        history[
            "startup_global"
        ].notna().all(),
        f"{scope_id}: Startup history mapping failed.",
    )

    history_investor = (
        history[
            "investor_global"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    history_startup_local = (
        history[
            "startup_global"
        ]
        .to_numpy(
            dtype=np.int64
        )
        - num_investors
    )

    history_segment = (
        history[
            "segment_number"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    history_pair_keys = (
        history_investor
        * num_startups
        +
        history_startup_local
    )

    first_frame = pd.DataFrame(
        {
            "pair_key":
                history_pair_keys,

            "first_segment":
                history_segment,
        }
    )

    first_series = (
        first_frame
        .groupby(
            "pair_key",
            sort=False,
        )[
            "first_segment"
        ]
        .min()
    )

    # Generator lookup. This reproduces the frozen generator input semantics.
    pair_first_segment = dict(
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

    # Independent sorted arrays used ONLY by the post-generation audit.
    audit_pair_keys = (
        first_series
        .index
        .to_numpy(
            dtype=np.int64
        )
    )

    audit_first_segments = (
        first_series
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

    require(
        bool(
            (
                np.diff(
                    audit_pair_keys
                )
                > 0
            ).all()
        ),
        f"{scope_id}: audit pair-key registry not unique/sorted.",
    )

    print(
        f"  positives:             "
        f"{expected_positives:,}"
    )

    print(
        f"  eligible Startups:     "
        f"{expected_eligible:,}"
    )

    print(
        f"  lookup history:        "
        f"T0..T{max_segment}"
    )

    print(
        f"  unique observed pairs: "
        f"{len(audit_pair_keys):,}"
    )

    # -------------------------------------------------------------------------
    # Frozen scope registry.
    # -------------------------------------------------------------------------

    scope_registry = (
        registry.loc[
            registry[
                "scope_id"
            ].astype(str)
            .eq(
                scope_id
            )
        ]
        .sort_values(
            "epoch_index",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    require(
        len(
            scope_registry
        )
        == 20,
        f"{scope_id}: registry coverage not 20 epochs.",
    )

    # =========================================================================
    # EACH EPOCH
    # =========================================================================

    for epoch_index in EXPECTED_EPOCHS:

        registry_row = (
            scope_registry.loc[
                scope_registry[
                    "epoch_index"
                ].astype(int)
                .eq(
                    epoch_index
                )
            ]
        )

        require(
            len(
                registry_row
            )
            == 1,
            (
                f"{scope_id} epoch {epoch_index}: "
                "registry row missing/duplicated."
            ),
        )

        registry_row = (
            registry_row.iloc[0]
        )

        # ---------------------------------------------------------------------
        # Re-derive frozen seed.
        # ---------------------------------------------------------------------

        (
            negative_seed,
            negative_payload,
        ) = module.derive_seed(
            module.NEGATIVE_NAMESPACE,
            scope_id,
            epoch_index,
        )

        require(
            int(
                negative_seed
            )
            ==
            int(
                registry_row[
                    "negative_seed_u64"
                ]
            ),
            (
                f"{scope_id} epoch {epoch_index}: "
                "negative seed differs from frozen registry."
            ),
        )

        require(
            f"scope={scope_id}"
            in negative_payload,
            (
                f"{scope_id} epoch {epoch_index}: "
                "scope missing from seed payload."
            ),
        )

        require(
            f"epoch={epoch_index}"
            in negative_payload,
            (
                f"{scope_id} epoch {epoch_index}: "
                "epoch missing from seed payload."
            ),
        )

        require(
            "model="
            not in negative_payload.lower(),
            (
                f"{scope_id} epoch {epoch_index}: "
                "model entered seed payload."
            ),
        )

        # ---------------------------------------------------------------------
        # Epoch 0: authoritative persisted fixture.
        # Epochs 1..19: deterministic regeneration.
        # ---------------------------------------------------------------------

        if epoch_index == 0:

            negative_matrix_path = Path(
                module.NEGATIVE_MATRIX_PATH
            )

            require(
                negative_matrix_path.exists(),
                (
                    f"{scope_id}: persisted epoch-0 "
                    "negative matrix missing."
                ),
            )

            negatives = np.load(
                negative_matrix_path,
                allow_pickle=False,
            )

            source = (
                "AUTHORITATIVE_PERSISTED_EPOCH0"
            )

        else:

            (
                negatives,
                _generation_stats,
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
                    num_startups
                ),
                seed=(
                    negative_seed
                ),
                max_rows=None,
                progress=False,
            )

            source = (
                "DETERMINISTIC_REGENERATION_FOR_AUDIT"
            )

        require(
            negatives.shape
            == (
                expected_positives,
                NEGATIVES_PER_POSITIVE,
            ),
            (
                f"{scope_id} epoch {epoch_index}: "
                "negative shape mismatch."
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

        # ---------------------------------------------------------------------
        # Fingerprint identity with already-frozen V2 coverage registry.
        # ---------------------------------------------------------------------

        observed_negative_sha = (
            module.array_logical_sha256(
                negatives
            )
        )

        expected_negative_sha = str(
            registry_row[
                "negative_matrix_sha256"
            ]
        )

        require(
            observed_negative_sha
            == expected_negative_sha,
            (
                f"{scope_id} epoch {epoch_index}: "
                "negative fingerprint mismatch.\n"
                f"expected={expected_negative_sha}\n"
                f"observed={observed_negative_sha}"
            ),
        )

        # ---------------------------------------------------------------------
        # Recheck previously-covered integrity invariants.
        # ---------------------------------------------------------------------

        all_origin_eligible = bool(
            eligible_lookup[
                negatives
            ].all()
        )

        require(
            all_origin_eligible,
            (
                f"{scope_id} epoch {epoch_index}: "
                "origin-ineligible negative."
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
            duplicate_rows
            == 0,
            (
                f"{scope_id} epoch {epoch_index}: "
                "duplicate negative within focal positive."
            ),
        )

        focal_collision_count = int(
            (
                negatives
                ==
                focal_startup_locals[
                    :,
                    None
                ]
            ).sum()
        )

        require(
            focal_collision_count
            == 0,
            (
                f"{scope_id} epoch {epoch_index}: "
                "negative equals focal Startup."
            ),
        )

        # ---------------------------------------------------------------------
        # NEW INDEPENDENT TEMPORAL ACCEPTED-PAIR AUDIT.
        # ---------------------------------------------------------------------

        temporal_result = (
            temporal_collision_audit(
                investor_globals=(
                    investor_globals
                ),
                target_segments=(
                    target_segments
                ),
                negatives=(
                    negatives
                ),
                sorted_pair_keys=(
                    audit_pair_keys
                ),
                sorted_first_segments=(
                    audit_first_segments
                ),
                num_startups=(
                    num_startups
                ),
            )
        )

        forbidden_temporal_collisions = int(
            temporal_result[
                "forbidden_temporal_collisions"
            ]
        )

        require(
            forbidden_temporal_collisions
            == 0,
            (
                f"{scope_id} epoch {epoch_index}: "
                "TEMPORALLY FORBIDDEN ACCEPTED NEGATIVE FOUND."
            ),
        )

        total_negatives = int(
            negatives.size
        )

        require(
            (
                temporal_result[
                    "known_future_positive_negatives_allowed"
                ]
                +
                temporal_result[
                    "historically_unobserved_negatives"
                ]
            )
            == total_negatives,
            (
                f"{scope_id} epoch {epoch_index}: "
                "temporal audit accounting mismatch."
            ),
        )

        audit_rows.append(
            {
                "scope_id":
                    scope_id,

                "origin_label":
                    str(
                        module.ORIGIN_LABEL
                    ),

                "epoch_index":
                    int(
                        epoch_index
                    ),

                "display_epoch":
                    int(
                        epoch_index
                        + 1
                    ),

                "training_max_segment":
                    int(
                        max_segment
                    ),

                "positive_events":
                    int(
                        expected_positives
                    ),

                "accepted_negatives":
                    total_negatives,

                "negative_seed_u64":
                    int(
                        negative_seed
                    ),

                "negative_matrix_sha256":
                    observed_negative_sha,

                "fingerprint_matches_frozen_registry":
                    True,

                "all_negatives_origin_eligible":
                    True,

                "duplicate_negative_rows":
                    0,

                "focal_collision_count":
                    0,

                "forbidden_temporal_collisions":
                    0,

                "known_future_positive_negatives_allowed":
                    int(
                        temporal_result[
                            "known_future_positive_negatives_allowed"
                        ]
                    ),

                "historically_unobserved_negatives":
                    int(
                        temporal_result[
                            "historically_unobserved_negatives"
                        ]
                    ),

                "matched_history_pair_negatives":
                    int(
                        temporal_result[
                            "matched_history_pair_negatives"
                        ]
                    ),

                "temporal_lookup_max_segment":
                    int(
                        max_segment
                    ),

                "later_than_origin_labels_used":
                    False,

                "matrix_source":
                    source,
            }
        )

        print(
            f"  epoch {epoch_index:02d}: "
            f"neg={observed_negative_sha[:12]}... "
            f"forbidden=0 "
            f"future_allowed="
            f"{temporal_result['known_future_positive_negatives_allowed']:,} "
            "PASS"
        )

        del negatives

        if epoch_index > 0:
            del _generation_stats

        gc.collect()

    # -------------------------------------------------------------------------
    # Scope cleanup.
    # -------------------------------------------------------------------------

    del positives
    del investor_globals
    del focal_startup_locals
    del target_segments
    del eligible_startup_locals
    del eligible_lookup
    del node_index
    del investor_nodes
    del startup_nodes
    del investor_map
    del startup_map
    del temporal
    del history
    del history_investor
    del history_startup_local
    del history_segment
    del history_pair_keys
    del first_frame
    del first_series
    del pair_first_segment
    del audit_pair_keys
    del audit_first_segments
    del audit_sort

    gc.collect()


# =============================================================================
# 4. GLOBAL AUDIT
# =============================================================================

print()
print("=" * 120)
print("4. GLOBAL TEMPORAL-INTEGRITY AUDIT")
print("=" * 120)


audit = pd.DataFrame(
    audit_rows
)

audit = (
    audit
    .sort_values(
        [
            "scope_id",
            "epoch_index",
        ],
        kind="mergesort",
    )
    .reset_index(
        drop=True
    )
)


require(
    len(
        audit
    )
    == 60,
    "Expected exactly 60 audited streams.",
)

require(
    int(
        audit[
            "forbidden_temporal_collisions"
        ].sum()
    )
    == 0,
    "At least one temporally forbidden accepted negative exists.",
)

require(
    audit[
        "fingerprint_matches_frozen_registry"
    ].all(),
    "At least one regenerated matrix differs from frozen fingerprint.",
)

require(
    audit[
        "all_negatives_origin_eligible"
    ].all(),
    "At least one origin-ineligible negative exists.",
)

require(
    int(
        audit[
            "duplicate_negative_rows"
        ].sum()
    )
    == 0,
    "Duplicate negatives found.",
)

require(
    int(
        audit[
            "focal_collision_count"
        ].sum()
    )
    == 0,
    "Focal collisions found.",
)

require(
    not audit[
        "later_than_origin_labels_used"
    ].any(),
    "Later-than-origin labels entered temporal audit.",
)


for scope_id in SCOPE_MODULES:

    scope = (
        audit.loc[
            audit[
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
        scope[
            "epoch_index"
        ].tolist()
        == EXPECTED_EPOCHS,
        f"{scope_id}: incomplete temporal audit.",
    )

    require(
        int(
            scope[
                "forbidden_temporal_collisions"
            ].sum()
        )
        == 0,
        f"{scope_id}: temporal collisions detected.",
    )


total_accepted_negatives = int(
    audit[
        "accepted_negatives"
    ].sum()
)

total_future_positive_allowed = int(
    audit[
        "known_future_positive_negatives_allowed"
    ].sum()
)

total_unobserved = int(
    audit[
        "historically_unobserved_negatives"
    ].sum()
)


require(
    (
        total_future_positive_allowed
        + total_unobserved
    )
    == total_accepted_negatives,
    "Global temporal accounting mismatch.",
)


print("  Development scopes audited:          3 / 3")
print("  Epochs per scope audited:            20 / 20")
print("  Total deterministic streams:         60 / 60")
print(
    f"  Accepted negatives audited:          "
    f"{total_accepted_negatives:,}"
)
print("  Fingerprints match frozen registry:  60 / 60")
print("  Origin eligibility:                  PASS")
print("  Within-positive uniqueness:          PASS")
print("  Focal collisions:                    0")
print("  Forbidden temporal collisions:       0")
print(
    f"  Allowed future-positive negatives:   "
    f"{total_future_positive_allowed:,}"
)
print(
    f"  Historically unobserved negatives:   "
    f"{total_unobserved:,}"
)
print("  Labels beyond each fold origin used: NO")


# =============================================================================
# 5. PERSIST SUPPLEMENTAL AUDIT
# =============================================================================

print()
print("5. PERSIST SUPPLEMENTAL AUDIT")

audit.to_csv(
    OUTPUT_CSV,
    index=False,
)


manifest = {
    "schema_version":
        "PHASE_8_2_6_ALL_EPOCH_TEMPORAL_NEGATIVE_INTEGRITY_AUDIT_V1",

    "phase":
        "8.2.6",

    "status":
        "ALL_DEVELOPMENT_STREAM_TEMPORAL_NEGATIVE_INTEGRITY_AUDITED",

    "scientific_role":
        "SUPPLEMENTAL_PRETRAINING_TEMPORAL_NEGATIVE_ACCEPTANCE_AUDIT",

    "reason_for_supplement": (
        "The existing V2 development fingerprint coverage independently "
        "verified deterministic fingerprints, origin eligibility, "
        "within-positive uniqueness and focal collisions for all 60 "
        "development streams, while the explicit vectorized "
        "first_segment <= target_segment accepted-pair audit had only "
        "been persisted for the three epoch-0 fixtures. This supplement "
        "extends that hard temporal audit to all 60 streams."
    ),

    "authoritative_existing_artifacts_preserved": {
        "interpretation":
            str(
                INTERPRETATION
            ),

        "coverage_registry":
            str(
                REGISTRY_PATH
            ),

        "coverage_manifest":
            str(
                COVERAGE_MANIFEST
            ),

        "integration_closure_V2":
            str(
                CLOSURE_V2
            ),

        "existing_artifacts_modified":
            False,
    },

    "audit_scope": {
        "development_scopes": [
            "DEV_T57",
            "DEV_T58",
            "DEV_T59",
        ],

        "epoch_indices":
            list(
                range(20)
            ),

        "display_epochs":
            list(
                range(1, 21)
            ),

        "scope_epoch_streams":
            60,

        "negative_positive_ratio":
            4,

        "epoch_0_source":
            "AUTHORITATIVE_PERSISTED_FIXTURE",

        "epochs_1_19_source":
            "DETERMINISTIC_REGENERATION_USING_FROZEN_SEED",

        "regenerated_arrays_persisted":
            False,
    },

    "temporal_acceptance_rule": {
        "pair_key":
            (
                "investor_global * NUM_STARTUPS "
                "+ startup_local"
            ),

        "lookup":
            (
                "minimum observed positive segment for "
                "Investor-Startup pair using T0..fold_train_max only"
            ),

        "forbidden":
            "first_segment <= focal_target_segment",

        "allowed_future_positive":
            "first_segment > focal_target_segment",

        "allowed_unobserved":
            "pair absent from fold history lookup",

        "labels_after_fold_train_max_used":
            False,
    },

    "results": {
        "streams_audited":
            60,

        "accepted_negatives_audited":
            total_accepted_negatives,

        "fingerprints_matching_frozen_registry":
            60,

        "forbidden_temporal_collisions":
            0,

        "origin_ineligible_negatives":
            0,

        "duplicate_negative_rows":
            0,

        "focal_collisions":
            0,

        "known_future_positive_negatives_allowed":
            total_future_positive_allowed,

        "historically_unobserved_negatives":
            total_unobserved,

        "temporal_accounting_complete":
            True,
    },

    "execution_state": {
        "negative_streams_regenerated_for_audit":
            True,

        "regenerated_streams_retained":
            False,

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

    "next_step": (
        "Require this supplemental temporal-integrity PASS in the "
        "Phase 8.2.8 V3 final integration gate, alongside correction "
        "of the 8.2.1D authoritative_execution_contract schema check."
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


hash_targets = [
    INTERPRETATION,
    REGISTRY_PATH,
    COVERAGE_MANIFEST,
    COVERAGE_HASHES,
    CLOSURE_V2,
    Path(
        "scripts/"
        "phase_8_2_6_development_epoch_fingerprint_coverage_V2.py"
    ),
    Path(
        "scripts/"
        "phase_8_2_6_dev_t57_epoch0_stream_materialization_V1.py"
    ),
    Path(
        "scripts/"
        "phase_8_2_6_dev_t58_epoch0_stream_materialization_V1.py"
    ),
    Path(
        "scripts/"
        "phase_8_2_6_dev_t59_epoch0_stream_materialization_V1.py"
    ),
    OUTPUT_CSV,
    OUTPUT_MANIFEST,
]


for path in hash_targets:

    require(
        path.exists(),
        f"Hash target missing: {path}",
    )


hash_registry = {
    str(path):
        sha256_file(
            path
        )
    for path
    in hash_targets
}


OUTPUT_HASHES.write_text(
    json.dumps(
        hash_registry,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)


print(
    f"  Audit CSV:      {OUTPUT_CSV}"
)

print(
    f"  Audit manifest: {OUTPUT_MANIFEST}"
)

print(
    f"  SHA256 registry:{OUTPUT_HASHES}"
)


# =============================================================================
# FINAL STATUS
# =============================================================================

print()
print("=" * 120)
print("PHASE 8.2.6 — TEMPORAL NEGATIVE INTEGRITY SUPPLEMENT COMPLETE")
print("=" * 120)

print("Development scopes:                  3 / 3")
print("Epochs per scope:                    20 / 20")
print("Streams audited:                     60 / 60")
print("Frozen fingerprint matches:          60 / 60")
print(
    f"Accepted negatives audited:          "
    f"{total_accepted_negatives:,}"
)
print("Forbidden temporal collisions:       0")
print("Origin-ineligible negatives:         0")
print("Duplicate negatives within focal:    0")
print("Focal-positive collisions:           0")
print("Later-than-origin labels consumed:   NO")
print()
print("Model instantiated:                  NO")
print("Forward executed:                    NO")
print("Optimizer created:                   NO")
print("Backward executed:                   NO")
print("Training performed:                  NO")
print("Performance inspected:               NO")
print("T60 scored:                          NO")
print("April analyzed:                      NO")
print("Training allowed now:                NO")
print()
print(
    "STATUS: "
    "ALL_DEVELOPMENT_STREAM_TEMPORAL_NEGATIVE_INTEGRITY_AUDITED"
)
print(
    "NEXT:   Phase 8.2.8 V3 final integration gate"
)
print("=" * 120)
