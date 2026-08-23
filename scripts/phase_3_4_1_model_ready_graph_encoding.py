from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 3.4.1 — DETERMINISTIC MODEL-READY GRAPH ENCODING
# =============================================================================
#
# PURPOSE
# -------
#
# Convert the frozen scientific structural graph:
#
#     nodes.parquet
#     edges.parquet
#
# into deterministic integer-indexed artifacts suitable for R-GCN
# implementation.
#
# THIS SCRIPT DOES NOT:
#
#   - alter graph semantics,
#   - add or remove nodes,
#   - add or remove edges,
#   - filter isolates,
#   - add investment interactions,
#   - add self-loop relations,
#   - change relation directionality.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================


GRAPH_DIR = Path(
    "data/experimental/phase_3/graph"
)

AUDIT_DIR = Path(
    "data/experimental/phase_3/audits"
)

MODEL_DIR = Path(
    "data/experimental/phase_3/model_ready"
)


NODES_PATH = (
    GRAPH_DIR
    / "nodes.parquet"
)

EDGES_PATH = (
    GRAPH_DIR
    / "edges.parquet"
)

EXPECTED_TYPED_RELATIONS_PATH = (
    AUDIT_DIR
    / "phase_3_3_typed_relation_projection.csv"
)


# =============================================================================
# OUTPUTS
# =============================================================================


NODE_INDEX_OUTPUT = (
    MODEL_DIR
    / "node_index.parquet"
)

RELATION_INDEX_OUTPUT = (
    MODEL_DIR
    / "relation_index.csv"
)

EDGE_INDEX_OUTPUT = (
    MODEL_DIR
    / "edge_index.npy"
)

EDGE_TYPE_OUTPUT = (
    MODEL_DIR
    / "edge_type.npy"
)

EDGE_MANIFEST_OUTPUT = (
    MODEL_DIR
    / "edge_manifest.parquet"
)

VARIANT_MASK_OUTPUT = (
    MODEL_DIR
    / "graph_variant_masks.npz"
)

METADATA_OUTPUT = (
    MODEL_DIR
    / "model_ready_graph_metadata.json"
)


ENCODING_AUDIT_OUTPUT = (
    AUDIT_DIR
    / "model_ready_graph_encoding_summary.csv"
)

RELATION_ENCODING_AUDIT_OUTPUT = (
    AUDIT_DIR
    / "model_ready_relation_encoding_audit.csv"
)

VARIANT_AUDIT_OUTPUT = (
    AUDIT_DIR
    / "model_ready_graph_variant_mask_summary.csv"
)

ROUNDTRIP_AUDIT_OUTPUT = (
    AUDIT_DIR
    / "model_ready_graph_roundtrip_integrity.csv"
)


# =============================================================================
# FROZEN EXPECTATIONS
# =============================================================================


EXPECTED_NODES = 477_564
EXPECTED_EDGES = 158_818
EXPECTED_TYPED_RELATIONS = 12

EXPECTED_FOUNDER_EDGES = 94_818
EXPECTED_ACQUISITION_EDGES = 64_000


# =============================================================================
# UTILITIES
# =============================================================================


def separator(
    char="=",
    width=120,
):
    print(
        char * width
    )


def file_sha256(
    path,
):

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        for block in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):

            digest.update(
                block
            )

    return digest.hexdigest()


# =============================================================================
# MAIN
# =============================================================================


def main():

    separator()

    print(
        "PHASE 3.4.1 — "
        "DETERMINISTIC MODEL-READY GRAPH ENCODING"
    )

    separator()

    # =========================================================================
    # 1. Load frozen scientific graph
    # =========================================================================

    nodes = pd.read_parquet(
        NODES_PATH
    )

    edges = pd.read_parquet(
        EDGES_PATH
    )

    if len(
        nodes
    ) != EXPECTED_NODES:

        raise ValueError(
            f"Expected {EXPECTED_NODES:,} nodes, "
            f"found {len(nodes):,}."
        )

    if len(
        edges
    ) != EXPECTED_EDGES:

        raise ValueError(
            f"Expected {EXPECTED_EDGES:,} edges, "
            f"found {len(edges):,}."
        )

    # =========================================================================
    # 2. Deterministic node ordering
    # =========================================================================

    node_index = (
        nodes.sort_values(
            [
                "node_type",
                "raw_entity_id",
            ]
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    node_index[
        "node_index"
    ] = np.arange(
        len(
            node_index
        ),
        dtype=np.int64,
    )

    # Put model index first.
    node_index = node_index[
        [
            "node_index",
            "node_id",
            "node_type",
            "raw_entity_id",
            "display_name",
            "source_registry",
            "underlying_entity_has_dual_role",
        ]
    ]

    if (
        node_index[
            "node_index"
        ]
        .iloc[0]
        != 0
    ):

        raise ValueError(
            "Node index does not start at 0."
        )

    if (
        node_index[
            "node_index"
        ]
        .iloc[-1]
        !=
        EXPECTED_NODES - 1
    ):

        raise ValueError(
            "Node index does not end at expected value."
        )

    if (
        node_index[
            "node_index"
        ]
        .nunique()
        !=
        EXPECTED_NODES
    ):

        raise ValueError(
            "Node indices are not unique."
        )

    if (
        node_index[
            "node_id"
        ]
        .nunique()
        !=
        EXPECTED_NODES
    ):

        raise ValueError(
            "Node IDs are not unique."
        )

    node_to_index = (
        node_index.set_index(
            "node_id"
        )[
            "node_index"
        ]
        .to_dict()
    )

    index_to_node = (
        node_index.set_index(
            "node_index"
        )[
            "node_id"
        ]
        .to_dict()
    )

    # =========================================================================
    # 3. Deterministic typed relation vocabulary
    # =========================================================================

    observed_typed_relations = sorted(
        edges[
            "typed_relation_key"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    if (
        len(
            observed_typed_relations
        )
        !=
        EXPECTED_TYPED_RELATIONS
    ):

        raise ValueError(
            f"Expected {EXPECTED_TYPED_RELATIONS} typed relations, "
            f"found {len(observed_typed_relations)}."
        )

    relation_index = pd.DataFrame(
        {
            "relation_id":
                np.arange(
                    EXPECTED_TYPED_RELATIONS,
                    dtype=np.int64,
                ),

            "typed_relation_key":
                observed_typed_relations,
        }
    )

    relation_index[
        [
            "src_type",
            "relation",
            "dst_type",
        ]
    ] = (
        relation_index[
            "typed_relation_key"
        ]
        .str.split(
            "|",
            expand=True,
        )
    )

    relation_index = relation_index[
        [
            "relation_id",
            "src_type",
            "relation",
            "dst_type",
            "typed_relation_key",
        ]
    ]

    relation_to_index = (
        relation_index.set_index(
            "typed_relation_key"
        )[
            "relation_id"
        ]
        .to_dict()
    )

    # =========================================================================
    # 4. Cross-check against frozen Phase-3.3 relation vocabulary
    # =========================================================================

    expected_relations = pd.read_csv(
        EXPECTED_TYPED_RELATIONS_PATH
    )

    expected_keys = set(
        expected_relations[
            "typed_relation_key"
        ]
    )

    observed_keys = set(
        relation_index[
            "typed_relation_key"
        ]
    )

    if (
        expected_keys
        !=
        observed_keys
    ):

        raise ValueError(
            "Model-ready typed relation vocabulary differs "
            "from frozen Phase-3.3 specification."
        )

    # =========================================================================
    # 5. Deterministic edge ordering
    #
    # edges.parquet was already deterministically sorted in Phase 3.3.3,
    # but we sort again explicitly so this encoding is independently
    # reproducible.
    # =========================================================================

    model_edges = (
        edges.sort_values(
            [
                "typed_relation_key",
                "src_node_id",
                "dst_node_id",
            ]
        )
        .reset_index(
            drop=True
        )
        .copy()
    )

    model_edges[
        "edge_position"
    ] = np.arange(
        len(
            model_edges
        ),
        dtype=np.int64,
    )

    # =========================================================================
    # 6. Encode graph-node endpoints
    # =========================================================================

    model_edges[
        "src_index"
    ] = (
        model_edges[
            "src_node_id"
        ]
        .map(
            node_to_index
        )
    )

    model_edges[
        "dst_index"
    ] = (
        model_edges[
            "dst_node_id"
        ]
        .map(
            node_to_index
        )
    )

    if (
        model_edges[
            "src_index"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "At least one source node failed model-index encoding."
        )

    if (
        model_edges[
            "dst_index"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "At least one destination node failed model-index encoding."
        )

    model_edges[
        "src_index"
    ] = (
        model_edges[
            "src_index"
        ]
        .astype(
            np.int64
        )
    )

    model_edges[
        "dst_index"
    ] = (
        model_edges[
            "dst_index"
        ]
        .astype(
            np.int64
        )
    )

    # =========================================================================
    # 7. Encode typed relation IDs
    # =========================================================================

    model_edges[
        "relation_id"
    ] = (
        model_edges[
            "typed_relation_key"
        ]
        .map(
            relation_to_index
        )
    )

    if (
        model_edges[
            "relation_id"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "At least one edge failed relation-ID encoding."
        )

    model_edges[
        "relation_id"
    ] = (
        model_edges[
            "relation_id"
        ]
        .astype(
            np.int64
        )
    )

    # =========================================================================
    # 8. Build model-ready NumPy tensors
    # =========================================================================

    edge_index = np.vstack(
        [
            model_edges[
                "src_index"
            ]
            .to_numpy(
                dtype=np.int64
            ),

            model_edges[
                "dst_index"
            ]
            .to_numpy(
                dtype=np.int64
            ),
        ]
    )

    edge_type = (
        model_edges[
            "relation_id"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    if edge_index.shape != (
        2,
        EXPECTED_EDGES,
    ):

        raise ValueError(
            "Unexpected edge_index shape: "
            f"{edge_index.shape}"
        )

    if edge_type.shape != (
        EXPECTED_EDGES,
    ):

        raise ValueError(
            "Unexpected edge_type shape: "
            f"{edge_type.shape}"
        )

    # =========================================================================
    # 9. Build graph-variant masks
    # =========================================================================

    core_mask = np.ones(
        EXPECTED_EDGES,
        dtype=bool,
    )

    founder_mask = (
        model_edges[
            "relation"
        ]
        .eq(
            "shared_founder"
        )
        .to_numpy(
            dtype=bool
        )
    )

    acquisition_mask = (
        model_edges[
            "relation"
        ]
        .isin(
            [
                "acquired",
                "acquired_by",
            ]
        )
        .to_numpy(
            dtype=bool
        )
    )

    if int(
        founder_mask.sum()
    ) != EXPECTED_FOUNDER_EDGES:

        raise ValueError(
            "Founder-only variant edge count changed."
        )

    if int(
        acquisition_mask.sum()
    ) != EXPECTED_ACQUISITION_EDGES:

        raise ValueError(
            "Acquisition-only variant edge count changed."
        )

    if int(
        core_mask.sum()
    ) != EXPECTED_EDGES:

        raise ValueError(
            "Core graph variant edge count changed."
        )

    if np.any(
        founder_mask
        &
        acquisition_mask
    ):

        raise ValueError(
            "Founder and acquisition masks unexpectedly overlap."
        )

    if not np.array_equal(
        founder_mask
        |
        acquisition_mask,
        core_mask,
    ):

        raise ValueError(
            "Founder + acquisition masks do not reconstruct "
            "the complete core graph."
        )

    # =========================================================================
    # 10. Round-trip node identity audit
    #
    # Decode the integer source/destination indices and verify that they
    # exactly reconstruct the original graph node IDs.
    # =========================================================================

    decoded_src = pd.Series(
        model_edges[
            "src_index"
        ]
        .map(
            index_to_node
        ),
        index=model_edges.index,
    )

    decoded_dst = pd.Series(
        model_edges[
            "dst_index"
        ]
        .map(
            index_to_node
        ),
        index=model_edges.index,
    )

    src_roundtrip_mismatches = int(
        (
            decoded_src
            !=
            model_edges[
                "src_node_id"
            ]
        )
        .sum()
    )

    dst_roundtrip_mismatches = int(
        (
            decoded_dst
            !=
            model_edges[
                "dst_node_id"
            ]
        )
        .sum()
    )

    # =========================================================================
    # 11. Round-trip relation audit
    # =========================================================================

    id_to_relation = (
        relation_index.set_index(
            "relation_id"
        )[
            "typed_relation_key"
        ]
        .to_dict()
    )

    decoded_relation = (
        model_edges[
            "relation_id"
        ]
        .map(
            id_to_relation
        )
    )

    relation_roundtrip_mismatches = int(
        (
            decoded_relation
            !=
            model_edges[
                "typed_relation_key"
            ]
        )
        .sum()
    )

    if (
        src_roundtrip_mismatches
        != 0
        or
        dst_roundtrip_mismatches
        != 0
        or
        relation_roundtrip_mismatches
        != 0
    ):

        raise ValueError(
            "Model-ready graph round-trip integrity failed."
        )

    # =========================================================================
    # 12. Relation encoding audit
    # =========================================================================

    relation_observed_counts = (
        model_edges.groupby(
            [
                "relation_id",
                "typed_relation_key",
            ],
            observed=True,
        )
        .size()
        .reset_index(
            name="edge_records"
        )
    )

    relation_encoding_audit = (
        relation_index.merge(
            relation_observed_counts,
            on=[
                "relation_id",
                "typed_relation_key",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    relation_encoding_audit[
        "edge_records"
    ] = (
        relation_encoding_audit[
            "edge_records"
        ]
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    # =========================================================================
    # 13. Variant audit
    # =========================================================================

    variant_summary = pd.DataFrame(
        [
            {
                "variant":
                    "core",

                "edge_records":
                    int(
                        core_mask.sum()
                    ),

                "relations":
                    "shared_founder | acquired | acquired_by",
            },

            {
                "variant":
                    "founder_only_ablation",

                "edge_records":
                    int(
                        founder_mask.sum()
                    ),

                "relations":
                    "shared_founder",
            },

            {
                "variant":
                    "acquisition_only_ablation",

                "edge_records":
                    int(
                        acquisition_mask.sum()
                    ),

                "relations":
                    "acquired | acquired_by",
            },
        ]
    )

    # =========================================================================
    # 14. Round-trip audit summary
    # =========================================================================

    roundtrip_audit = pd.DataFrame(
        [
            {
                "metric":
                    "source_node_roundtrip_mismatches",

                "value":
                    src_roundtrip_mismatches,

                "status":
                    (
                        "PASS"
                        if src_roundtrip_mismatches
                        == 0
                        else "FAIL"
                    ),
            },

            {
                "metric":
                    "destination_node_roundtrip_mismatches",

                "value":
                    dst_roundtrip_mismatches,

                "status":
                    (
                        "PASS"
                        if dst_roundtrip_mismatches
                        == 0
                        else "FAIL"
                    ),
            },

            {
                "metric":
                    "relation_roundtrip_mismatches",

                "value":
                    relation_roundtrip_mismatches,

                "status":
                    (
                        "PASS"
                        if relation_roundtrip_mismatches
                        == 0
                        else "FAIL"
                    ),
            },
        ]
    )

    # =========================================================================
    # 15. Encoding summary
    # =========================================================================

    investor_indices = (
        node_index[
            node_index[
                "node_type"
            ]
            ==
            "investor"
        ][
            "node_index"
        ]
    )

    startup_indices = (
        node_index[
            node_index[
                "node_type"
            ]
            ==
            "startup"
        ][
            "node_index"
        ]
    )

    encoding_summary = pd.DataFrame(
        [
            {
                "metric":
                    "total_nodes",

                "value":
                    len(
                        node_index
                    ),
            },

            {
                "metric":
                    "minimum_node_index",

                "value":
                    int(
                        node_index[
                            "node_index"
                        ]
                        .min()
                    ),
            },

            {
                "metric":
                    "maximum_node_index",

                "value":
                    int(
                        node_index[
                            "node_index"
                        ]
                        .max()
                    ),
            },

            {
                "metric":
                    "investor_node_index_min",

                "value":
                    int(
                        investor_indices.min()
                    ),
            },

            {
                "metric":
                    "investor_node_index_max",

                "value":
                    int(
                        investor_indices.max()
                    ),
            },

            {
                "metric":
                    "startup_node_index_min",

                "value":
                    int(
                        startup_indices.min()
                    ),
            },

            {
                "metric":
                    "startup_node_index_max",

                "value":
                    int(
                        startup_indices.max()
                    ),
            },

            {
                "metric":
                    "typed_relation_ids",

                "value":
                    len(
                        relation_index
                    ),
            },

            {
                "metric":
                    "minimum_relation_id",

                "value":
                    int(
                        relation_index[
                            "relation_id"
                        ]
                        .min()
                    ),
            },

            {
                "metric":
                    "maximum_relation_id",

                "value":
                    int(
                        relation_index[
                            "relation_id"
                        ]
                        .max()
                    ),
            },

            {
                "metric":
                    "edge_records",

                "value":
                    len(
                        model_edges
                    ),
            },

            {
                "metric":
                    "edge_index_rows",

                "value":
                    int(
                        edge_index.shape[0]
                    ),
            },

            {
                "metric":
                    "edge_index_columns",

                "value":
                    int(
                        edge_index.shape[1]
                    ),
            },

            {
                "metric":
                    "edge_type_length",

                "value":
                    int(
                        edge_type.shape[0]
                    ),
            },

            {
                "metric":
                    "core_variant_edges",

                "value":
                    int(
                        core_mask.sum()
                    ),
            },

            {
                "metric":
                    "founder_only_variant_edges",

                "value":
                    int(
                        founder_mask.sum()
                    ),
            },

            {
                "metric":
                    "acquisition_only_variant_edges",

                "value":
                    int(
                        acquisition_mask.sum()
                    ),
            },
        ]
    )

    # =========================================================================
    # 16. Model edge manifest
    #
    # Retain a human-auditable mapping between tensor position and original
    # graph edge.
    # =========================================================================

    edge_manifest = model_edges[
        [
            "edge_position",
            "edge_id",
            "src_index",
            "dst_index",
            "relation_id",
            "src_node_id",
            "dst_node_id",
            "typed_relation_key",
            "relation",
            "support_count",
            "temporal_provenance",
        ]
    ].copy()

    # =========================================================================
    # 17. Save
    # =========================================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    node_index.to_parquet(
        NODE_INDEX_OUTPUT,
        index=False,
    )

    relation_index.to_csv(
        RELATION_INDEX_OUTPUT,
        index=False,
    )

    np.save(
        EDGE_INDEX_OUTPUT,
        edge_index,
    )

    np.save(
        EDGE_TYPE_OUTPUT,
        edge_type,
    )

    edge_manifest.to_parquet(
        EDGE_MANIFEST_OUTPUT,
        index=False,
    )

    np.savez_compressed(
        VARIANT_MASK_OUTPUT,

        core=core_mask,

        founder_only_ablation=founder_mask,

        acquisition_only_ablation=acquisition_mask,
    )

    encoding_summary.to_csv(
        ENCODING_AUDIT_OUTPUT,
        index=False,
    )

    relation_encoding_audit.to_csv(
        RELATION_ENCODING_AUDIT_OUTPUT,
        index=False,
    )

    variant_summary.to_csv(
        VARIANT_AUDIT_OUTPUT,
        index=False,
    )

    roundtrip_audit.to_csv(
        ROUNDTRIP_AUDIT_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 18. Metadata / checksums
    # =========================================================================

    metadata = {
        "phase":
            "3.4.1",

        "graph":
            "itrs_crunchbase_structural_graph",

        "node_count":
            EXPECTED_NODES,

        "edge_count":
            EXPECTED_EDGES,

        "typed_relation_count":
            EXPECTED_TYPED_RELATIONS,

        "node_index_range": [
            0,
            EXPECTED_NODES - 1,
        ],

        "relation_id_range": [
            0,
            EXPECTED_TYPED_RELATIONS - 1,
        ],

        "edge_index_semantics": {
            "shape":
                [
                    2,
                    EXPECTED_EDGES,
                ],

            "row_0":
                "source_node_index",

            "row_1":
                "destination_node_index",
        },

        "variant_edge_counts": {
            "core":
                int(
                    core_mask.sum()
                ),

            "founder_only_ablation":
                int(
                    founder_mask.sum()
                ),

            "acquisition_only_ablation":
                int(
                    acquisition_mask.sum()
                ),
        },

        "node_ordering":
            "sort by node_type, then raw_entity_id",

        "relation_ordering":
            (
                "lexicographic sort of typed_relation_key"
            ),

        "edge_ordering":
            (
                "sort by typed_relation_key, "
                "src_node_id, dst_node_id"
            ),

        "isolated_nodes_retained":
            True,

        "investment_events_in_structural_graph":
            False,

        "checksums": {
            "nodes_parquet_sha256":
                file_sha256(
                    NODES_PATH
                ),

            "edges_parquet_sha256":
                file_sha256(
                    EDGES_PATH
                ),

            "node_index_parquet_sha256":
                file_sha256(
                    NODE_INDEX_OUTPUT
                ),

            "relation_index_csv_sha256":
                file_sha256(
                    RELATION_INDEX_OUTPUT
                ),

            "edge_index_npy_sha256":
                file_sha256(
                    EDGE_INDEX_OUTPUT
                ),

            "edge_type_npy_sha256":
                file_sha256(
                    EDGE_TYPE_OUTPUT
                ),

            "edge_manifest_parquet_sha256":
                file_sha256(
                    EDGE_MANIFEST_OUTPUT
                ),

            "variant_masks_npz_sha256":
                file_sha256(
                    VARIANT_MASK_OUTPUT
                ),
        },
    }

    with open(
        METADATA_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # =========================================================================
    # 19. Terminal output
    # =========================================================================

    separator("-")

    print(
        "MODEL-READY GRAPH ENCODING SUMMARY"
    )

    separator("-")

    print(
        encoding_summary.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "DETERMINISTIC RELATION-ID VOCABULARY"
    )

    separator("-")

    print(
        relation_encoding_audit.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "GRAPH VARIANT MASKS"
    )

    separator("-")

    print(
        variant_summary.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "ROUND-TRIP INTEGRITY"
    )

    separator("-")

    print(
        roundtrip_audit.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "MODEL-READY ARRAY SHAPES"
    )

    separator("-")

    print(
        f"edge_index.shape: "
        f"{edge_index.shape}"
    )

    print(
        f"edge_index.dtype: "
        f"{edge_index.dtype}"
    )

    print(
        f"edge_type.shape:  "
        f"{edge_type.shape}"
    )

    print(
        f"edge_type.dtype:  "
        f"{edge_type.dtype}"
    )

    separator("-")

    print(
        "MODEL-READY OUTPUTS"
    )

    separator("-")

    print(
        NODE_INDEX_OUTPUT
    )

    print(
        RELATION_INDEX_OUTPUT
    )

    print(
        EDGE_INDEX_OUTPUT
    )

    print(
        EDGE_TYPE_OUTPUT
    )

    print(
        EDGE_MANIFEST_OUTPUT
    )

    print(
        VARIANT_MASK_OUTPUT
    )

    print(
        METADATA_OUTPUT
    )

    separator()

    print(
        "PHASE 3.4.1 COMPLETE"
    )

    separator()

    print(
        """
IMPORTANT

1. The scientific graph itself has NOT changed.

2. node_index is deterministic and contiguous:

       0 ... 477,563

3. relation_id is deterministic and contiguous:

       0 ... 11

4. edge_index convention is:

       row 0 = source
       row 1 = destination

5. edge_type contains the typed R-GCN relation ID for each edge.

6. Isolated graph nodes remain in node_index.

7. Investment events remain outside the structural graph.

8. graph_variant_masks.npz defines:

       core
       founder_only_ablation
       acquisition_only_ablation

9. edge_manifest.parquet preserves complete tensor-to-source
   auditability.

NEXT:

Phase 3.4.2 — Model-Ready Graph Structural Coverage &
              Relation-Degree Audit
"""
    )


if __name__ == "__main__":
    main()