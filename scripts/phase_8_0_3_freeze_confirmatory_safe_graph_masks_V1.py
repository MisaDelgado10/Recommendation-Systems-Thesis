from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 8.0.3 — FREEZE CONFIRMATORY-SAFE GRAPH MASKS V1
# =============================================================================
#
# Scientific role:
#
#   OUTCOME-BLIND STRUCTURAL LEAKAGE-CONTROL ARTIFACT FREEZE
#
# This script:
#
#   - does NOT modify the frozen Phase-3 graph;
#   - does NOT modify edge_index.npy;
#   - does NOT modify edge_type.npy;
#   - does NOT modify Phase-3 graph_variant_masks.npz;
#   - does NOT load a model;
#   - does NOT run inference;
#   - does NOT score any evaluation case;
#   - does NOT generate negatives;
#   - does NOT freeze the April evaluation cases themselves.
#
# It creates Phase-8-only boolean graph masks that remove every frozen
# Phase-3 structural edge whose unordered underlying entity pair matches
# an April-2026 candidate-confirmatory investor-startup pair.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

CANONICAL_PATH = Path(
    "data/processed/interactions.parquet"
)

PHASE3_EDGES_PATH = Path(
    "data/experimental/phase_3/graph/edges.parquet"
)

PHASE3_MODEL_DIR = Path(
    "data/experimental/phase_3/model_ready"
)

EDGE_INDEX_PATH = (
    PHASE3_MODEL_DIR
    / "edge_index.npy"
)

EDGE_TYPE_PATH = (
    PHASE3_MODEL_DIR
    / "edge_type.npy"
)

EDGE_MANIFEST_PATH = (
    PHASE3_MODEL_DIR
    / "edge_manifest.parquet"
)

ORIGINAL_VARIANT_MASKS_PATH = (
    PHASE3_MODEL_DIR
    / "graph_variant_masks.npz"
)


OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_3_confirmatory_safe_graph_masks"
)

MASK_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_3_confirmatory_safe_graph_masks_V1.npz"
)

REMOVED_EDGES_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_3_removed_structural_edges_V1.csv"
)

MASK_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_3_graph_mask_summary_V1.csv"
)

RELATION_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_3_removed_edges_by_relation_V1.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_3_graph_mask_freeze_summary_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_3_graph_mask_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_3_derived_artifact_sha256_V1.json"
)


# =============================================================================
# FROZEN / AUDITED CONTRACT
# =============================================================================

CANDIDATE_HOLDOUT_START = pd.Timestamp(
    "2026-04-01"
)

CANDIDATE_HOLDOUT_END = pd.Timestamp(
    "2026-04-30"
)

EXPECTED_CANONICAL_SHA256 = (
    "03d02c48dc8bc67981fa27b66714641f"
    "6fdf1d8dd907348862517740ba77db8b"
)

EXPECTED_PHASE3_EDGES_SHA256 = (
    "895871308de416b12d001550838cdb27"
    "63c87cd897774c6a55539a0ebb9a1ee2"
)

EXPECTED_PHASE3_EDGES = 158_818

EXPECTED_ORIGINAL_CORE = 158_818
EXPECTED_ORIGINAL_FOUNDER = 94_818
EXPECTED_ORIGINAL_ACQUISITION = 64_000

EXPECTED_REMOVED_TOTAL = 40
EXPECTED_REMOVED_UNIQUE_PAIRS = 11

EXPECTED_REMOVED_BY_RELATION = {
    "shared_founder": 4,
    "acquired": 18,
    "acquired_by": 18,
}

EXPECTED_SAFE_CORE = 158_778
EXPECTED_SAFE_FOUNDER = 94_814
EXPECTED_SAFE_ACQUISITION = 63_964


# =============================================================================
# UTILITIES
# =============================================================================

def separator(char="=", width=110):
    print(char * width)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            default=str,
        )


def ordered_underlying_pair(a, b):

    a = str(a)
    b = str(b)

    if a <= b:
        return (a, b)

    return (b, a)


def pair_key(pair):
    return f"{pair[0]}||{pair[1]}"


# =============================================================================
# MAIN
# =============================================================================

def main():

    separator()

    print(
        "PHASE 8.0.3 — "
        "FREEZE CONFIRMATORY-SAFE GRAPH MASKS V1"
    )

    separator()

    print()
    print(
        "Scientific role:             "
        "OUTCOME-BLIND LEAKAGE-CONTROL FREEZE"
    )

    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Inference executed:          NO")
    print("Ranking metrics scored:      NO")
    print("Negatives generated:         NO")
    print("Phase-3 graph modified:      NO")
    print("Phase-3 arrays modified:     NO")
    print("Phase-3 masks modified:      NO")
    print("Phase-8 overlay created:     YES")

    # =========================================================================
    # 1. Source existence and fingerprints
    # =========================================================================

    separator("-")
    print("8.0.3.1 — SOURCE EXISTENCE AND FINGERPRINTS")
    separator("-")

    required_paths = [
        CANONICAL_PATH,
        PHASE3_EDGES_PATH,
        EDGE_INDEX_PATH,
        EDGE_TYPE_PATH,
        EDGE_MANIFEST_PATH,
        ORIGINAL_VARIANT_MASKS_PATH,
    ]

    source_hashes = {}

    for path in required_paths:

        if not path.exists():
            raise FileNotFoundError(
                f"Required source not found: {path}"
            )

        observed_hash = sha256_file(
            path
        )

        source_hashes[
            str(path)
        ] = observed_hash

        print(f"FOUND  {path}")
        print(f"SHA256 {observed_hash}")

    if (
        source_hashes[
            str(CANONICAL_PATH)
        ]
        != EXPECTED_CANONICAL_SHA256
    ):
        raise ValueError(
            "Canonical interactions SHA256 drift detected."
        )

    if (
        source_hashes[
            str(PHASE3_EDGES_PATH)
        ]
        != EXPECTED_PHASE3_EDGES_SHA256
    ):
        raise ValueError(
            "Frozen Phase-3 edges SHA256 drift detected."
        )

    print()
    print("Frozen source fingerprints: PASS")

    # =========================================================================
    # 2. Reconstruct candidate April underlying-pair set
    # =========================================================================

    separator("-")
    print(
        "8.0.3.2 — RECONSTRUCT APRIL "
        "CONFIRMATORY PAIR SET"
    )
    separator("-")

    interactions = pd.read_parquet(
        CANONICAL_PATH,
        columns=[
            "interaction_id",
            "announced_on",
            "investor_id",
            "startup_id",
        ],
    )

    interactions[
        "announced_on"
    ] = pd.to_datetime(
        interactions[
            "announced_on"
        ],
        errors="raise",
    )

    april = interactions[
        (
            interactions[
                "announced_on"
            ]
            >= CANDIDATE_HOLDOUT_START
        )
        &
        (
            interactions[
                "announced_on"
            ]
            <= CANDIDATE_HOLDOUT_END
        )
    ].copy()

    april[
        "underlying_pair"
    ] = [
        ordered_underlying_pair(
            investor_id,
            startup_id,
        )
        for investor_id, startup_id
        in zip(
            april["investor_id"],
            april["startup_id"],
        )
    ]

    april_pairs = set(
        april[
            "underlying_pair"
        ]
    )

    print(
        f"April events:                     "
        f"{len(april):,}"
    )

    print(
        f"Unique unordered April pairs:     "
        f"{len(april_pairs):,}"
    )

    if len(april) != 6_622:
        raise ValueError(
            "April interaction count drift detected."
        )

    if len(april_pairs) != 6_582:
        raise ValueError(
            "April pair count drift detected."
        )

    # =========================================================================
    # 3. Reconstruct structural overlap independently
    # =========================================================================

    separator("-")
    print(
        "8.0.3.3 — RECONSTRUCT DIRECT "
        "STRUCTURAL OVERLAP"
    )
    separator("-")

    edges = pd.read_parquet(
        PHASE3_EDGES_PATH
    )

    if len(edges) != EXPECTED_PHASE3_EDGES:
        raise ValueError(
            "Frozen Phase-3 edge count drift detected."
        )

    if edges["edge_id"].duplicated().any():
        raise ValueError(
            "Duplicate frozen Phase-3 edge_id detected."
        )

    edges[
        "underlying_pair"
    ] = [
        ordered_underlying_pair(
            src,
            dst,
        )
        for src, dst
        in zip(
            edges[
                "underlying_src_id"
            ],
            edges[
                "underlying_dst_id"
            ],
        )
    ]

    overlap_edges = edges[
        edges[
            "underlying_pair"
        ].isin(
            april_pairs
        )
    ].copy()

    overlap_pairs = set(
        overlap_edges[
            "underlying_pair"
        ]
    )

    print(
        f"Overlap edge records:              "
        f"{len(overlap_edges):,}"
    )

    print(
        f"Overlap unique underlying pairs:   "
        f"{len(overlap_pairs):,}"
    )

    if (
        len(overlap_edges)
        != EXPECTED_REMOVED_TOTAL
    ):
        raise ValueError(
            "Direct-overlap edge count differs "
            "from Phase-8.0.2."
        )

    if (
        len(overlap_pairs)
        != EXPECTED_REMOVED_UNIQUE_PAIRS
    ):
        raise ValueError(
            "Direct-overlap pair count differs "
            "from Phase-8.0.2."
        )

    observed_relation_counts = (
        overlap_edges[
            "relation"
        ]
        .value_counts()
        .to_dict()
    )

    for (
        relation,
        expected_count,
    ) in (
        EXPECTED_REMOVED_BY_RELATION.items()
    ):

        observed_count = int(
            observed_relation_counts.get(
                relation,
                0,
            )
        )

        if (
            observed_count
            != expected_count
        ):
            raise ValueError(
                f"Unexpected removed-edge count "
                f"for {relation}: "
                f"{observed_count} "
                f"!= {expected_count}"
            )

    print("Phase-8.0.2 overlap anchors: PASS")

    # =========================================================================
    # 4. Load model-ready edge ordering
    # =========================================================================

    separator("-")
    print(
        "8.0.3.4 — MODEL-READY EDGE "
        "POSITION INTEGRITY"
    )
    separator("-")

    edge_manifest = pd.read_parquet(
        EDGE_MANIFEST_PATH
    )

    edge_index = np.load(
        EDGE_INDEX_PATH,
        mmap_mode="r",
    )

    edge_type = np.load(
        EDGE_TYPE_PATH,
        mmap_mode="r",
    )

    if len(edge_manifest) != EXPECTED_PHASE3_EDGES:
        raise ValueError(
            "edge_manifest row count mismatch."
        )

    if edge_index.shape != (
        2,
        EXPECTED_PHASE3_EDGES,
    ):
        raise ValueError(
            f"Unexpected edge_index shape: "
            f"{edge_index.shape}"
        )

    if edge_type.shape != (
        EXPECTED_PHASE3_EDGES,
    ):
        raise ValueError(
            f"Unexpected edge_type shape: "
            f"{edge_type.shape}"
        )

    if (
        edge_manifest[
            "edge_position"
        ].nunique()
        != EXPECTED_PHASE3_EDGES
    ):
        raise ValueError(
            "edge_position is not unique."
        )

    expected_positions = np.arange(
        EXPECTED_PHASE3_EDGES,
        dtype=np.int64,
    )

    observed_positions = (
        edge_manifest[
            "edge_position"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    if not np.array_equal(
        expected_positions,
        observed_positions,
    ):
        raise ValueError(
            "edge_manifest positions are not "
            "exactly contiguous 0..158817."
        )

    if edge_manifest[
        "edge_id"
    ].duplicated().any():
        raise ValueError(
            "Duplicate edge IDs in edge_manifest."
        )

    frozen_edge_ids = set(
        edges["edge_id"]
    )

    manifest_edge_ids = set(
        edge_manifest[
            "edge_id"
        ]
    )

    if (
        frozen_edge_ids
        != manifest_edge_ids
    ):
        raise ValueError(
            "edge_manifest does not contain exactly "
            "the frozen Phase-3 edge IDs."
        )

    print(
        "edge_index / edge_type / "
        "edge_manifest alignment: PASS"
    )

    # =========================================================================
    # 5. Map 40 overlap edges to exact tensor positions
    # =========================================================================

    separator("-")
    print(
        "8.0.3.5 — MAP LEAKAGE EDGES "
        "TO MODEL POSITIONS"
    )
    separator("-")

    overlap_edge_ids = set(
        overlap_edges[
            "edge_id"
        ]
    )

    removed_manifest = (
        edge_manifest[
            edge_manifest[
                "edge_id"
            ].isin(
                overlap_edge_ids
            )
        ]
        .copy()
    )

    if (
        len(removed_manifest)
        != EXPECTED_REMOVED_TOTAL
    ):
        raise ValueError(
            "Not all structural overlap edges "
            "mapped to model positions."
        )

    if (
        removed_manifest[
            "edge_id"
        ].nunique()
        != EXPECTED_REMOVED_TOTAL
    ):
        raise ValueError(
            "Overlap-to-manifest mapping "
            "is not one-to-one."
        )

    overlap_metadata = (
        overlap_edges[
            [
                "edge_id",
                "underlying_src_id",
                "underlying_dst_id",
                "relation",
                "temporal_provenance",
                "provenance_source",
            ]
        ]
        .copy()
    )

    removed_manifest = (
        removed_manifest.merge(
            overlap_metadata,
            on=[
                "edge_id",
                "relation",
                "temporal_provenance",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    removed_manifest[
        "underlying_pair_key"
    ] = [
        pair_key(
            ordered_underlying_pair(
                src,
                dst,
            )
        )
        for src, dst
        in zip(
            removed_manifest[
                "underlying_src_id"
            ],
            removed_manifest[
                "underlying_dst_id"
            ],
        )
    ]

    removed_positions = (
        removed_manifest[
            "edge_position"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    print(
        f"Mapped exclusion positions: "
        f"{len(removed_positions):,}"
    )

    print(
        f"Minimum position:           "
        f"{removed_positions.min():,}"
    )

    print(
        f"Maximum position:           "
        f"{removed_positions.max():,}"
    )

    # =========================================================================
    # 6. Load original Phase-3 graph-variant masks
    # =========================================================================

    separator("-")
    print(
        "8.0.3.6 — ORIGINAL PHASE-3 "
        "VARIANT MASK INTEGRITY"
    )
    separator("-")

    original_masks = np.load(
        ORIGINAL_VARIANT_MASKS_PATH
    )

    expected_mask_names = {
        "core",
        "founder_only_ablation",
        "acquisition_only_ablation",
    }

    observed_mask_names = set(
        original_masks.files
    )

    if (
        observed_mask_names
        != expected_mask_names
    ):
        raise ValueError(
            "Unexpected Phase-3 graph mask keys: "
            f"{sorted(observed_mask_names)}"
        )

    core_original = (
        original_masks[
            "core"
        ].astype(bool)
    )

    founder_original = (
        original_masks[
            "founder_only_ablation"
        ].astype(bool)
    )

    acquisition_original = (
        original_masks[
            "acquisition_only_ablation"
        ].astype(bool)
    )

    for name, mask in [
        (
            "core",
            core_original,
        ),
        (
            "founder",
            founder_original,
        ),
        (
            "acquisition",
            acquisition_original,
        ),
    ]:

        if mask.shape != (
            EXPECTED_PHASE3_EDGES,
        ):
            raise ValueError(
                f"{name} mask shape mismatch."
            )

    original_counts = {
        "core":
            int(
                core_original.sum()
            ),

        "founder_only_ablation":
            int(
                founder_original.sum()
            ),

        "acquisition_only_ablation":
            int(
                acquisition_original.sum()
            ),
    }

    print(
        f"Core edges:              "
        f"{original_counts['core']:,}"
    )

    print(
        f"Founder-only edges:      "
        f"{original_counts['founder_only_ablation']:,}"
    )

    print(
        f"Acquisition-only edges:  "
        f"{original_counts['acquisition_only_ablation']:,}"
    )

    if (
        original_counts["core"]
        != EXPECTED_ORIGINAL_CORE
    ):
        raise ValueError(
            "Original core-mask count drift."
        )

    if (
        original_counts[
            "founder_only_ablation"
        ]
        != EXPECTED_ORIGINAL_FOUNDER
    ):
        raise ValueError(
            "Original founder-mask count drift."
        )

    if (
        original_counts[
            "acquisition_only_ablation"
        ]
        != EXPECTED_ORIGINAL_ACQUISITION
    ):
        raise ValueError(
            "Original acquisition-mask count drift."
        )

    print("Original Phase-3 masks: PASS")

    # =========================================================================
    # 7. Build direct-pair exclusion mask
    # =========================================================================

    separator("-")
    print(
        "8.0.3.7 — BUILD PHASE-8 "
        "DIRECT-PAIR EXCLUSION MASK"
    )
    separator("-")

    direct_pair_exclusion = np.zeros(
        EXPECTED_PHASE3_EDGES,
        dtype=bool,
    )

    direct_pair_exclusion[
        removed_positions
    ] = True

    if (
        int(
            direct_pair_exclusion.sum()
        )
        != EXPECTED_REMOVED_TOTAL
    ):
        raise ValueError(
            "Direct-pair exclusion mask does "
            "not contain exactly 40 positions."
        )

    # Every removed edge must exist in the original core graph.
    if not np.all(
        core_original[
            direct_pair_exclusion
        ]
    ):
        raise ValueError(
            "At least one excluded edge is not "
            "present in the original core mask."
        )

    # =========================================================================
    # 8. Construct Phase-8 confirmatory-safe variants
    # =========================================================================

    separator("-")
    print(
        "8.0.3.8 — CONSTRUCT CONFIRMATORY-SAFE "
        "GRAPH VARIANTS"
    )
    separator("-")

    keep_mask = (
        ~direct_pair_exclusion
    )

    core_safe = (
        core_original
        & keep_mask
    )

    founder_safe = (
        founder_original
        & keep_mask
    )

    acquisition_safe = (
        acquisition_original
        & keep_mask
    )

    safe_counts = {
        "core_confirmatory_safe":
            int(
                core_safe.sum()
            ),

        "founder_only_confirmatory_safe":
            int(
                founder_safe.sum()
            ),

        "acquisition_only_confirmatory_safe":
            int(
                acquisition_safe.sum()
            ),
    }

    print(
        f"Confirmatory-safe core:        "
        f"{safe_counts['core_confirmatory_safe']:,}"
    )

    print(
        f"Confirmatory-safe founder:     "
        f"{safe_counts['founder_only_confirmatory_safe']:,}"
    )

    print(
        f"Confirmatory-safe acquisition: "
        f"{safe_counts['acquisition_only_confirmatory_safe']:,}"
    )

    if (
        safe_counts[
            "core_confirmatory_safe"
        ]
        != EXPECTED_SAFE_CORE
    ):
        raise ValueError(
            "Unexpected confirmatory-safe core count."
        )

    if (
        safe_counts[
            "founder_only_confirmatory_safe"
        ]
        != EXPECTED_SAFE_FOUNDER
    ):
        raise ValueError(
            "Unexpected confirmatory-safe founder count."
        )

    if (
        safe_counts[
            "acquisition_only_confirmatory_safe"
        ]
        != EXPECTED_SAFE_ACQUISITION
    ):
        raise ValueError(
            "Unexpected confirmatory-safe "
            "acquisition count."
        )

    # Founder and acquisition masks remain disjoint.
    if np.any(
        founder_safe
        & acquisition_safe
    ):
        raise ValueError(
            "Founder/acquisition safe masks overlap."
        )

    # Their union must still reconstruct safe core.
    if not np.array_equal(
        founder_safe
        | acquisition_safe,
        core_safe,
    ):
        raise ValueError(
            "Safe founder + acquisition masks do "
            "not reconstruct safe core."
        )

    # Exactly 40 core records must have been removed.
    removed_from_core = int(
        (
            core_original
            & ~core_safe
        ).sum()
    )

    if (
        removed_from_core
        != EXPECTED_REMOVED_TOTAL
    ):
        raise ValueError(
            "Safe core did not remove exactly "
            "the 40 audited overlap records."
        )

    print("Confirmatory-safe mask construction: PASS")

    # =========================================================================
    # 9. Verify no April direct structural target pair survives
    # =========================================================================

    separator("-")
    print(
        "8.0.3.9 — ZERO-OVERLAP POST-MASK PROOF"
    )
    separator("-")

    # edge_manifest is indexed exactly by edge_position.
    surviving_positions = np.flatnonzero(
        core_safe
    )

    surviving_edge_ids = set(
        edge_manifest.iloc[
            surviving_positions
        ][
            "edge_id"
        ]
    )

    surviving_overlap_edge_ids = (
        surviving_edge_ids
        & overlap_edge_ids
    )

    surviving_overlap_count = len(
        surviving_overlap_edge_ids
    )

    print(
        "April target-pair structural edges "
        f"surviving safe core: "
        f"{surviving_overlap_count:,}"
    )

    if surviving_overlap_count != 0:
        raise ValueError(
            "Structural target-pair leakage remains "
            "after Phase-8 masking."
        )

    print(
        "Confirmatory-safe core direct-pair "
        "leakage: 0 — PASS"
    )

    # =========================================================================
    # 10. Build summaries
    # =========================================================================

    relation_summary = (
        removed_manifest.groupby(
            [
                "relation",
                "temporal_provenance",
            ],
            dropna=False,
        )
        .agg(
            removed_edge_records=(
                "edge_id",
                "size",
            ),
            unique_underlying_pairs=(
                "underlying_pair_key",
                "nunique",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "removed_edge_records",
                "relation",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    mask_summary = pd.DataFrame(
        [
            {
                "variant":
                    "core",
                "original_edges":
                    int(
                        core_original.sum()
                    ),
                "removed_edges":
                    int(
                        (
                            core_original
                            & direct_pair_exclusion
                        ).sum()
                    ),
                "confirmatory_safe_edges":
                    int(
                        core_safe.sum()
                    ),
            },
            {
                "variant":
                    "founder_only_ablation",
                "original_edges":
                    int(
                        founder_original.sum()
                    ),
                "removed_edges":
                    int(
                        (
                            founder_original
                            & direct_pair_exclusion
                        ).sum()
                    ),
                "confirmatory_safe_edges":
                    int(
                        founder_safe.sum()
                    ),
            },
            {
                "variant":
                    "acquisition_only_ablation",
                "original_edges":
                    int(
                        acquisition_original.sum()
                    ),
                "removed_edges":
                    int(
                        (
                            acquisition_original
                            & direct_pair_exclusion
                        ).sum()
                    ),
                "confirmatory_safe_edges":
                    int(
                        acquisition_safe.sum()
                    ),
            },
        ]
    )

    # =========================================================================
    # 11. Persist Phase-8 overlay
    # =========================================================================

    separator("-")
    print(
        "8.0.3.10 — PERSIST PHASE-8 "
        "LEAKAGE-CONTROL ARTIFACTS"
    )
    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        MASK_OUTPUT,

        direct_pair_exclusion=(
            direct_pair_exclusion
        ),

        core_confirmatory_safe=(
            core_safe
        ),

        founder_only_confirmatory_safe=(
            founder_safe
        ),

        acquisition_only_confirmatory_safe=(
            acquisition_safe
        ),
    )

    removed_manifest.to_csv(
        REMOVED_EDGES_OUTPUT,
        index=False,
    )

    mask_summary.to_csv(
        MASK_SUMMARY_OUTPUT,
        index=False,
    )

    relation_summary.to_csv(
        RELATION_SUMMARY_OUTPUT,
        index=False,
    )

    summary = {
        "schema_version":
            "PHASE_8_0_3_CONFIRMATORY_SAFE_GRAPH_MASKS_V1",

        "scientific_role":
            "OUTCOME_BLIND_STRUCTURAL_LEAKAGE_CONTROL_FREEZE",

        "candidate_confirmatory_period": {
            "start":
                str(
                    CANDIDATE_HOLDOUT_START.date()
                ),
            "end":
                str(
                    CANDIDATE_HOLDOUT_END.date()
                ),
            "status":
                "CANDIDATE_NOT_YET_FINAL_EVALUATION_FREEZE",
        },

        "phase3_graph_modified":
            False,

        "phase3_model_arrays_modified":
            False,

        "phase3_variant_masks_modified":
            False,

        "phase8_overlay_created":
            True,

        "original_edge_count":
            EXPECTED_PHASE3_EDGES,

        "direct_structural_edges_removed":
            int(
                direct_pair_exclusion.sum()
            ),

        "direct_structural_pairs_removed":
            int(
                len(
                    overlap_pairs
                )
            ),

        "safe_edge_counts":
            safe_counts,

        "post_mask_direct_pair_overlap":
            surviving_overlap_count,

        "mask_policy": (
            "Remove every frozen Phase-3 structural "
            "edge whose unordered underlying entity "
            "pair matches any April-2026 candidate "
            "confirmatory investor-startup pair."
        ),

        "temporal_observability_boundary": (
            "This mask prevents direct target-pair "
            "structural overlap. It does not convert "
            "current_snapshot_unversioned founder "
            "relations into historically timestamped "
            "information."
        ),

        "freeze_status":
            "PHASE8_CONFIRMATORY_SAFE_GRAPH_MASK_FROZEN",
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    manifest = {
        "phase":
            "8.0.3",

        "authoritative_inputs": {
            str(path):
                source_hashes[
                    str(path)
                ]
            for path in required_paths
        },

        "immutable_phase3_artifacts": [
            str(PHASE3_EDGES_PATH),
            str(EDGE_INDEX_PATH),
            str(EDGE_TYPE_PATH),
            str(EDGE_MANIFEST_PATH),
            str(ORIGINAL_VARIANT_MASKS_PATH),
        ],

        "derived_phase8_overlay": [
            str(MASK_OUTPUT),
            str(REMOVED_EDGES_OUTPUT),
            str(MASK_SUMMARY_OUTPUT),
            str(RELATION_SUMMARY_OUTPUT),
            str(SUMMARY_OUTPUT),
        ],

        "expected_removed_edge_records":
            EXPECTED_REMOVED_TOTAL,

        "expected_removed_pairs":
            EXPECTED_REMOVED_UNIQUE_PAIRS,

        "expected_safe_counts": {
            "core":
                EXPECTED_SAFE_CORE,
            "founder_only":
                EXPECTED_SAFE_FOUNDER,
            "acquisition_only":
                EXPECTED_SAFE_ACQUISITION,
        },

        "prohibited_operations": [
            "modify_phase3_graph",
            "modify_edge_index",
            "modify_edge_type",
            "modify_phase3_variant_masks",
            "model_loading",
            "inference",
            "ranking_scoring",
            "negative_generation",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    derived_hashes = {
        str(MASK_OUTPUT):
            sha256_file(
                MASK_OUTPUT
            ),

        str(REMOVED_EDGES_OUTPUT):
            sha256_file(
                REMOVED_EDGES_OUTPUT
            ),

        str(MASK_SUMMARY_OUTPUT):
            sha256_file(
                MASK_SUMMARY_OUTPUT
            ),

        str(RELATION_SUMMARY_OUTPUT):
            sha256_file(
                RELATION_SUMMARY_OUTPUT
            ),

        str(SUMMARY_OUTPUT):
            sha256_file(
                SUMMARY_OUTPUT
            ),

        str(MANIFEST_OUTPUT):
            sha256_file(
                MANIFEST_OUTPUT
            ),
    }

    write_json(
        HASH_OUTPUT,
        derived_hashes,
    )

    print("Artifacts persisted: PASS")

    # =========================================================================
    # 12. Final terminal summary
    # =========================================================================

    separator()
    print(
        "PHASE 8.0.3 GRAPH MASK FREEZE COMPLETE"
    )
    separator()

    print()
    print(
        f"Frozen Phase-3 graph edges:       "
        f"{EXPECTED_PHASE3_EDGES:,}"
    )

    print(
        f"Direct overlap edges excluded:    "
        f"{int(direct_pair_exclusion.sum()):,}"
    )

    print(
        f"Direct overlap pairs excluded:    "
        f"{len(overlap_pairs):,}"
    )

    print()
    print(
        f"Confirmatory-safe core edges:     "
        f"{int(core_safe.sum()):,}"
    )

    print(
        f"Safe founder-only edges:          "
        f"{int(founder_safe.sum()):,}"
    )

    print(
        f"Safe acquisition-only edges:      "
        f"{int(acquisition_safe.sum()):,}"
    )

    print()
    print(
        "Direct target-pair leakage after mask: "
        f"{surviving_overlap_count}"
    )

    print()
    print(
        "PHASE-3 GRAPH MODIFIED:               NO"
    )

    print(
        "PHASE-3 MODEL ARRAYS MODIFIED:        NO"
    )

    print(
        "PHASE-8 OVERLAY MASK FROZEN:          YES"
    )

    print(
        "CONFIRMATORY HOLDOUT FULLY FROZEN:    NO"
    )

    print()
    print(
        "The next Phase-8.0 decision is the "
        "formal temporal/evaluation contract "
        "for the April confirmatory holdout."
    )

    print()
    print(
        f"Outputs written to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()