from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 3.4.3 — FINAL INTEGRITY & PHASE-3 CLOSURE
# =============================================================================
#
# PURPOSE
# -------
#
# Perform the final machine-readable closure audit for Phase 3.
#
# This script verifies consistency across:
#
#   Phase 1 canonical interaction universe
#   Phase 2 frozen temporal holdout
#   Phase 3 scientific node table
#   Phase 3 scientific edge table
#   Phase 3 model-ready node encoding
#   Phase 3 model-ready relation encoding
#   Phase 3 model-ready tensors
#   Phase 3 graph-variant masks
#   Phase 3 structural coverage diagnostics
#
# NO graph data is modified.
# NO node or edge is created or removed.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================


INTERACTIONS_PATH = Path(
    "data/processed/interactions.parquet"
)

T60_HOLDOUT_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "t60_holdout_pair_manifest.parquet"
)


GRAPH_DIR = Path(
    "data/experimental/phase_3/graph"
)

MODEL_DIR = Path(
    "data/experimental/phase_3/model_ready"
)

AUDIT_DIR = Path(
    "data/experimental/phase_3/audits"
)

CONFIG_DIR = Path(
    "configs"
)


NODES_PATH = (
    GRAPH_DIR
    / "nodes.parquet"
)

EDGES_PATH = (
    GRAPH_DIR
    / "edges.parquet"
)

NODE_INDEX_PATH = (
    MODEL_DIR
    / "node_index.parquet"
)

RELATION_INDEX_PATH = (
    MODEL_DIR
    / "relation_index.csv"
)

EDGE_INDEX_PATH = (
    MODEL_DIR
    / "edge_index.npy"
)

EDGE_TYPE_PATH = (
    MODEL_DIR
    / "edge_type.npy"
)

EDGE_MANIFEST_PATH = (
    MODEL_DIR
    / "edge_manifest.parquet"
)

VARIANT_MASKS_PATH = (
    MODEL_DIR
    / "graph_variant_masks.npz"
)

MODEL_METADATA_PATH = (
    MODEL_DIR
    / "model_ready_graph_metadata.json"
)

NODE_COVERAGE_PATH = (
    MODEL_DIR
    / "node_structural_coverage.parquet"
)

GRAPH_SCHEMA_CONFIG_PATH = (
    CONFIG_DIR
    / "phase_3_itrs_graph_schema.json"
)


# =============================================================================
# INPUT AUDITS
# =============================================================================


EDGE_INTEGRITY_PATH = (
    AUDIT_DIR
    / "graph_edge_integrity_summary.csv"
)

NODE_IDENTITY_PATH = (
    AUDIT_DIR
    / "graph_node_identity_summary.csv"
)

VARIANT_COVERAGE_PATH = (
    AUDIT_DIR
    / "graph_variant_node_coverage.csv"
)

SOURCE_OVERLAP_PATH = (
    AUDIT_DIR
    / "structural_source_node_overlap.csv"
)

T60_PAIR_COVERAGE_PATH = (
    AUDIT_DIR
    / "t60_holdout_pair_structural_coverage.csv"
)


# =============================================================================
# OUTPUTS
# =============================================================================


FINAL_INTEGRITY_OUTPUT = (
    AUDIT_DIR
    / "phase_3_final_integrity_summary.csv"
)

FINAL_COUNTS_OUTPUT = (
    AUDIT_DIR
    / "phase_3_final_counts.csv"
)

FINAL_RELATION_OUTPUT = (
    AUDIT_DIR
    / "phase_3_final_relation_vocabulary.csv"
)

FINAL_VARIANT_OUTPUT = (
    AUDIT_DIR
    / "phase_3_final_graph_variants.csv"
)

FINAL_LIMITATIONS_OUTPUT = (
    AUDIT_DIR
    / "phase_3_known_limitations.csv"
)

FINAL_MANIFEST_OUTPUT = (
    MODEL_DIR
    / "phase_3_closure_manifest.json"
)


# =============================================================================
# FROZEN EXPECTATIONS
# =============================================================================


EXPECTED_INTERACTIONS = 1_208_051

EXPECTED_INVESTORS = 165_975
EXPECTED_STARTUPS = 311_589

EXPECTED_ROLE_NODES = 477_564
EXPECTED_UNDERLYING_ENTITIES = 467_931
EXPECTED_DUAL_ROLE_IDS = 9_633

EXPECTED_EDGES = 158_818

EXPECTED_SHARED_FOUNDER_EDGES = 94_818
EXPECTED_ACQUIRED_EDGES = 32_000
EXPECTED_ACQUIRED_BY_EDGES = 32_000

EXPECTED_TYPED_RELATIONS = 12

EXPECTED_CORE_CONNECTED_NODES = 74_757
EXPECTED_CORE_ISOLATED_NODES = 402_807

EXPECTED_T60_PAIRS = 22_327
EXPECTED_T60_UNORDERED_MASK_PAIRS = 22_326


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


def sha256(
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


def add_check(
    rows,
    check,
    observed,
    expected,
):

    rows.append(
        {
            "check":
                check,

            "observed":
                observed,

            "expected":
                expected,

            "status":
                (
                    "PASS"
                    if observed
                    ==
                    expected
                    else "FAIL"
                ),
        }
    )


def metric_value(
    dataframe,
    metric,
):

    row = dataframe[
        dataframe[
            "metric"
        ]
        ==
        metric
    ]

    if len(
        row
    ) != 1:

        raise ValueError(
            f"Expected one row for metric '{metric}', "
            f"found {len(row)}."
        )

    return row.iloc[0][
        "value"
    ]


def ordered_underlying_pair(
    value_a,
    value_b,
):

    return tuple(
        sorted(
            [
                str(
                    value_a
                ),
                str(
                    value_b
                ),
            ]
        )
    )


# =============================================================================
# MAIN
# =============================================================================


def main():

    separator()

    print(
        "PHASE 3.4.3 — "
        "FINAL INTEGRITY & PHASE-3 CLOSURE"
    )

    separator()

    # =========================================================================
    # 1. Load core artifacts
    # =========================================================================

    interactions = pd.read_parquet(
        INTERACTIONS_PATH,
        columns=[
            "investor_id",
            "startup_id",
        ],
    )

    holdout = pd.read_parquet(
        T60_HOLDOUT_PATH
    )

    nodes = pd.read_parquet(
        NODES_PATH
    )

    edges = pd.read_parquet(
        EDGES_PATH
    )

    node_index = pd.read_parquet(
        NODE_INDEX_PATH
    )

    relation_index = pd.read_csv(
        RELATION_INDEX_PATH
    )

    edge_index = np.load(
        EDGE_INDEX_PATH
    )

    edge_type = np.load(
        EDGE_TYPE_PATH
    )

    edge_manifest = pd.read_parquet(
        EDGE_MANIFEST_PATH
    )

    variant_masks = np.load(
        VARIANT_MASKS_PATH
    )

    node_coverage = pd.read_parquet(
        NODE_COVERAGE_PATH
    )

    # =========================================================================
    # 2. Load audit summaries
    # =========================================================================

    node_identity = pd.read_csv(
        NODE_IDENTITY_PATH
    )

    edge_integrity = pd.read_csv(
        EDGE_INTEGRITY_PATH
    )

    variant_coverage = pd.read_csv(
        VARIANT_COVERAGE_PATH
    )

    source_overlap = pd.read_csv(
        SOURCE_OVERLAP_PATH
    )

    t60_pair_coverage = pd.read_csv(
        T60_PAIR_COVERAGE_PATH
    )

    # =========================================================================
    # 3. Canonical universe
    # =========================================================================

    investor_ids = set(
        interactions[
            "investor_id"
        ]
        .astype("string")
        .str.strip()
        .dropna()
        .unique()
    )

    startup_ids = set(
        interactions[
            "startup_id"
        ]
        .astype("string")
        .str.strip()
        .dropna()
        .unique()
    )

    dual_role_ids = (
        investor_ids
        &
        startup_ids
    )

    unique_underlying_ids = (
        investor_ids
        |
        startup_ids
    )

    # =========================================================================
    # 4. T60 unique pair universe
    # =========================================================================

    holdout[
        "investor_id"
    ] = (
        holdout[
            "investor_id"
        ]
        .astype("string")
        .str.strip()
    )

    holdout[
        "startup_id"
    ] = (
        holdout[
            "startup_id"
        ]
        .astype("string")
        .str.strip()
    )

    directed_holdout_pairs = set(
        zip(
            holdout[
                "investor_id"
            ],
            holdout[
                "startup_id"
            ],
        )
    )

    unordered_holdout_pairs = {
        ordered_underlying_pair(
            investor_id,
            startup_id,
        )
        for (
            investor_id,
            startup_id,
        ) in directed_holdout_pairs
    }

    # =========================================================================
    # 5. Final integrity checks
    # =========================================================================

    checks = []

    add_check(
        checks,
        "canonical_interaction_rows",
        len(
            interactions
        ),
        EXPECTED_INTERACTIONS,
    )

    add_check(
        checks,
        "canonical_investors",
        len(
            investor_ids
        ),
        EXPECTED_INVESTORS,
    )

    add_check(
        checks,
        "canonical_startups",
        len(
            startup_ids
        ),
        EXPECTED_STARTUPS,
    )

    add_check(
        checks,
        "dual_role_underlying_ids",
        len(
            dual_role_ids
        ),
        EXPECTED_DUAL_ROLE_IDS,
    )

    add_check(
        checks,
        "unique_underlying_entities",
        len(
            unique_underlying_ids
        ),
        EXPECTED_UNDERLYING_ENTITIES,
    )

    add_check(
        checks,
        "scientific_graph_nodes",
        len(
            nodes
        ),
        EXPECTED_ROLE_NODES,
    )

    add_check(
        checks,
        "scientific_graph_edges",
        len(
            edges
        ),
        EXPECTED_EDGES,
    )

    add_check(
        checks,
        "model_node_index_rows",
        len(
            node_index
        ),
        EXPECTED_ROLE_NODES,
    )

    add_check(
        checks,
        "relation_index_rows",
        len(
            relation_index
        ),
        EXPECTED_TYPED_RELATIONS,
    )

    add_check(
        checks,
        "edge_index_rows",
        int(
            edge_index.shape[
                0
            ]
        ),
        2,
    )

    add_check(
        checks,
        "edge_index_columns",
        int(
            edge_index.shape[
                1
            ]
        ),
        EXPECTED_EDGES,
    )

    add_check(
        checks,
        "edge_type_length",
        len(
            edge_type
        ),
        EXPECTED_EDGES,
    )

    add_check(
        checks,
        "edge_manifest_rows",
        len(
            edge_manifest
        ),
        EXPECTED_EDGES,
    )

    add_check(
        checks,
        "node_structural_coverage_rows",
        len(
            node_coverage
        ),
        EXPECTED_ROLE_NODES,
    )

    add_check(
        checks,
        "T60_unique_directed_pairs",
        len(
            directed_holdout_pairs
        ),
        EXPECTED_T60_PAIRS,
    )

    add_check(
        checks,
        "T60_unordered_underlying_mask_pairs",
        len(
            unordered_holdout_pairs
        ),
        EXPECTED_T60_UNORDERED_MASK_PAIRS,
    )

    # =========================================================================
    # 6. Relation counts
    # =========================================================================

    relation_counts = (
        edges[
            "relation"
        ]
        .value_counts()
        .to_dict()
    )

    add_check(
        checks,
        "shared_founder_edge_records",
        int(
            relation_counts.get(
                "shared_founder",
                0,
            )
        ),
        EXPECTED_SHARED_FOUNDER_EDGES,
    )

    add_check(
        checks,
        "acquired_edge_records",
        int(
            relation_counts.get(
                "acquired",
                0,
            )
        ),
        EXPECTED_ACQUIRED_EDGES,
    )

    add_check(
        checks,
        "acquired_by_edge_records",
        int(
            relation_counts.get(
                "acquired_by",
                0,
            )
        ),
        EXPECTED_ACQUIRED_BY_EDGES,
    )

    add_check(
        checks,
        "typed_relation_channels",
        edges[
            "typed_relation_key"
        ]
        .nunique(),
        EXPECTED_TYPED_RELATIONS,
    )

    # =========================================================================
    # 7. Variant-mask consistency
    # =========================================================================

    core_mask = (
        variant_masks[
            "core"
        ]
        .astype(
            bool
        )
    )

    founder_mask = (
        variant_masks[
            "founder_only_ablation"
        ]
        .astype(
            bool
        )
    )

    acquisition_mask = (
        variant_masks[
            "acquisition_only_ablation"
        ]
        .astype(
            bool
        )
    )

    add_check(
        checks,
        "core_variant_edges",
        int(
            core_mask.sum()
        ),
        EXPECTED_EDGES,
    )

    add_check(
        checks,
        "founder_variant_edges",
        int(
            founder_mask.sum()
        ),
        EXPECTED_SHARED_FOUNDER_EDGES,
    )

    add_check(
        checks,
        "acquisition_variant_edges",
        int(
            acquisition_mask.sum()
        ),
        (
            EXPECTED_ACQUIRED_EDGES
            +
            EXPECTED_ACQUIRED_BY_EDGES
        ),
    )

    add_check(
        checks,
        "variant_mask_overlap",
        int(
            (
                founder_mask
                &
                acquisition_mask
            )
            .sum()
        ),
        0,
    )

    add_check(
        checks,
        "variant_union_reconstructs_core",
        int(
            np.array_equal(
                (
                    founder_mask
                    |
                    acquisition_mask
                ),
                core_mask,
            )
        ),
        1,
    )

    # =========================================================================
    # 8. Structural graph leakage re-check
    # =========================================================================

    structural_holdout_matches = 0

    for (
        underlying_src,
        underlying_dst,
    ) in zip(
        edges[
            "underlying_src_id"
        ],
        edges[
            "underlying_dst_id"
        ],
    ):

        pair = ordered_underlying_pair(
            underlying_src,
            underlying_dst,
        )

        if pair in unordered_holdout_pairs:

            structural_holdout_matches += 1

    add_check(
        checks,
        "structural_T60_holdout_edge_records",
        structural_holdout_matches,
        0,
    )

    # =========================================================================
    # 9. Reuse already-passed graph integrity
    # =========================================================================

    failed_edge_checks = int(
        edge_integrity[
            "status"
        ]
        .ne(
            "PASS"
        )
        .sum()
    )

    add_check(
        checks,
        "previous_edge_integrity_failures",
        failed_edge_checks,
        0,
    )

    # =========================================================================
    # 10. Structural connectivity closure
    # =========================================================================

    core_all = (
        variant_coverage[
            (
                variant_coverage[
                    "variant"
                ]
                ==
                "core"
            )
            &
            (
                variant_coverage[
                    "node_type"
                ]
                ==
                "ALL"
            )
        ]
    )

    if len(
        core_all
    ) != 1:

        raise ValueError(
            "Could not locate unique core/ALL coverage row."
        )

    core_connected_nodes = int(
        core_all.iloc[0][
            "connected_nodes"
        ]
    )

    core_isolated_nodes = int(
        core_all.iloc[0][
            "isolated_nodes"
        ]
    )

    add_check(
        checks,
        "core_connected_nodes",
        core_connected_nodes,
        EXPECTED_CORE_CONNECTED_NODES,
    )

    add_check(
        checks,
        "core_isolated_nodes",
        core_isolated_nodes,
        EXPECTED_CORE_ISOLATED_NODES,
    )

    # =========================================================================
    # 11. T60 pair coverage closure
    # =========================================================================

    core_pair_rows = (
        t60_pair_coverage[
            t60_pair_coverage[
                "variant"
            ]
            ==
            "core"
        ]
    )

    core_pair_total = int(
        core_pair_rows[
            "pair_count"
        ]
        .sum()
    )

    add_check(
        checks,
        "core_T60_pair_coverage_rows_sum",
        core_pair_total,
        EXPECTED_T60_PAIRS,
    )

    final_integrity = pd.DataFrame(
        checks
    )

    if (
        final_integrity[
            "status"
        ]
        .ne(
            "PASS"
        )
        .any()
    ):

        failed = (
            final_integrity[
                final_integrity[
                    "status"
                ]
                !=
                "PASS"
            ]
        )

        print(
            "\nFAILED CHECKS:"
        )

        print(
            failed.to_string(
                index=False
            )
        )

        raise ValueError(
            "Phase-3 final integrity failed."
        )

    # =========================================================================
    # 12. Final counts table
    # =========================================================================

    overlap_all = (
        source_overlap[
            source_overlap[
                "node_type"
            ]
            ==
            "ALL"
        ]
        .set_index(
            "coverage_class"
        )[
            "node_count"
        ]
        .to_dict()
    )

    core_pair_dict = (
        core_pair_rows.set_index(
            "coverage_class"
        )[
            "pair_count"
        ]
        .to_dict()
    )

    final_counts = pd.DataFrame(
        [
            {
                "metric":
                    "canonical_interactions",

                "value":
                    EXPECTED_INTERACTIONS,
            },

            {
                "metric":
                    "investor_role_nodes",

                "value":
                    EXPECTED_INVESTORS,
            },

            {
                "metric":
                    "startup_role_nodes",

                "value":
                    EXPECTED_STARTUPS,
            },

            {
                "metric":
                    "total_role_nodes",

                "value":
                    EXPECTED_ROLE_NODES,
            },

            {
                "metric":
                    "unique_underlying_entities",

                "value":
                    EXPECTED_UNDERLYING_ENTITIES,
            },

            {
                "metric":
                    "dual_role_underlying_entities",

                "value":
                    EXPECTED_DUAL_ROLE_IDS,
            },

            {
                "metric":
                    "structural_edge_records",

                "value":
                    EXPECTED_EDGES,
            },

            {
                "metric":
                    "base_semantic_relations",

                "value":
                    3,
            },

            {
                "metric":
                    "typed_relation_channels",

                "value":
                    EXPECTED_TYPED_RELATIONS,
            },

            {
                "metric":
                    "core_connected_nodes",

                "value":
                    core_connected_nodes,
            },

            {
                "metric":
                    "core_isolated_nodes",

                "value":
                    core_isolated_nodes,
            },

            {
                "metric":
                    "nodes_founder_and_acquisition",

                "value":
                    int(
                        overlap_all[
                            "founder_and_acquisition"
                        ]
                    ),
            },

            {
                "metric":
                    "nodes_founder_only",

                "value":
                    int(
                        overlap_all[
                            "founder_only"
                        ]
                    ),
            },

            {
                "metric":
                    "nodes_acquisition_only",

                "value":
                    int(
                        overlap_all[
                            "acquisition_only"
                        ]
                    ),
            },

            {
                "metric":
                    "nodes_neither_structural_source",

                "value":
                    int(
                        overlap_all[
                            "neither"
                        ]
                    ),
            },

            {
                "metric":
                    "T60_holdout_pairs",

                "value":
                    EXPECTED_T60_PAIRS,
            },

            {
                "metric":
                    "T60_pairs_both_endpoints_connected",

                "value":
                    int(
                        core_pair_dict[
                            "both_endpoints_connected"
                        ]
                    ),
            },

            {
                "metric":
                    "T60_pairs_investor_only_connected",

                "value":
                    int(
                        core_pair_dict[
                            "investor_only_connected"
                        ]
                    ),
            },

            {
                "metric":
                    "T60_pairs_startup_only_connected",

                "value":
                    int(
                        core_pair_dict[
                            "startup_only_connected"
                        ]
                    ),
            },

            {
                "metric":
                    "T60_pairs_neither_endpoint_connected",

                "value":
                    int(
                        core_pair_dict[
                            "neither_endpoint_connected"
                        ]
                    ),
            },
        ]
    )

    # =========================================================================
    # 13. Final relation vocabulary
    # =========================================================================

    final_relation_vocab = relation_index.copy()

    final_relation_vocab[
        "edge_records"
    ] = (
        final_relation_vocab[
            "relation_id"
        ]
        .map(
            pd.Series(
                edge_type
            )
            .value_counts()
            .to_dict()
        )
        .fillna(
            0
        )
        .astype(
            int
        )
    )

    # =========================================================================
    # 14. Final variants
    # =========================================================================

    final_variants = pd.DataFrame(
        [
            {
                "variant":
                    "core",

                "relations":
                    (
                        "shared_founder | "
                        "acquired | "
                        "acquired_by"
                    ),

                "edge_records":
                    int(
                        core_mask.sum()
                    ),

                "primary_or_ablation":
                    "primary",
            },

            {
                "variant":
                    "founder_only_ablation",

                "relations":
                    "shared_founder",

                "edge_records":
                    int(
                        founder_mask.sum()
                    ),

                "primary_or_ablation":
                    "ablation",
            },

            {
                "variant":
                    "acquisition_only_ablation",

                "relations":
                    "acquired | acquired_by",

                "edge_records":
                    int(
                        acquisition_mask.sum()
                    ),

                "primary_or_ablation":
                    "ablation",
            },
        ]
    )

    # =========================================================================
    # 15. Known limitations
    # =========================================================================

    final_limitations = pd.DataFrame(
        [
            {
                "limitation":
                    "Full ITRS relation vocabulary unavailable",

                "detail":
                    (
                        "The ITRS paper reports 103 final "
                        "relationship types but does not publish "
                        "the complete vocabulary or preprocessing "
                        "rules needed for exact reconstruction."
                    ),

                "phase_3_handling":
                    (
                        "Reproduce the relationship-construction "
                        "principle using defensible Crunchbase "
                        "sources; do not force 103 relations."
                    ),
            },

            {
                "limitation":
                    "Most Tianyancha primitive relations unavailable",

                "detail":
                    (
                        "Available Crunchbase exports do not expose "
                        "entity-level shareholder, controlling, "
                        "supplier, client, competitor or historical "
                        "shareholding relationships."
                    ),

                "phase_3_handling":
                    (
                        "Unsupported relations excluded rather "
                        "than approximated."
                    ),
            },

            {
                "limitation":
                    "Founder relationships temporally unversioned",

                "detail":
                    (
                        "SHARED_FOUNDER is reconstructed from "
                        "current-snapshot founder information "
                        "without historical relation timestamps."
                    ),

                "phase_3_handling":
                    (
                        "Record temporal provenance as "
                        "current_snapshot_unversioned and disclose "
                        "possible historical-information leakage "
                        "as a dataset limitation."
                    ),
            },

            {
                "limitation":
                    "Acquisition relation is a Crunchbase adaptation",

                "detail":
                    (
                        "ACQUIRED is not one of the ten explicitly "
                        "published Tianyancha primitive relation "
                        "labels."
                    ),

                "phase_3_handling":
                    (
                        "Treat as a Crunchbase-specific structural "
                        "relation following the ITRS company-relation "
                        "construction principle."
                    ),
            },

            {
                "limitation":
                    "Structural graph sparsity",

                "detail":
                    (
                        f"Only {core_connected_nodes:,} of "
                        f"{EXPECTED_ROLE_NODES:,} role nodes have "
                        "at least one structural neighbor."
                    ),

                "phase_3_handling":
                    (
                        "Retain isolates; do not modify the canonical "
                        "recommendation universe. Preserve coverage "
                        "artifact for later evaluation stratification."
                    ),
            },

            {
                "limitation":
                    "Limited structural information for T60 pairs",

                "detail":
                    (
                        "Most T60 held-out prediction pairs have "
                        "no structural neighbor on one or both "
                        "endpoints."
                    ),

                "phase_3_handling":
                    (
                        "Keep frozen evaluation universe unchanged; "
                        "later report structural-coverage diagnostic "
                        "slices."
                    ),
            },
        ]
    )

    # =========================================================================
    # 16. Save CSV closure artifacts
    # =========================================================================

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_integrity.to_csv(
        FINAL_INTEGRITY_OUTPUT,
        index=False,
    )

    final_counts.to_csv(
        FINAL_COUNTS_OUTPUT,
        index=False,
    )

    final_relation_vocab.to_csv(
        FINAL_RELATION_OUTPUT,
        index=False,
    )

    final_variants.to_csv(
        FINAL_VARIANT_OUTPUT,
        index=False,
    )

    final_limitations.to_csv(
        FINAL_LIMITATIONS_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 17. Closure manifest
    # =========================================================================

    manifest = {
        "phase":
            "3",

        "phase_name":
            "Heterogeneous Graph Reconstruction",

        "status":
            "COMPLETE",

        "scientific_graph": {
            "node_table":
                str(
                    NODES_PATH
                ),

            "edge_table":
                str(
                    EDGES_PATH
                ),

            "node_count":
                EXPECTED_ROLE_NODES,

            "edge_count":
                EXPECTED_EDGES,

            "base_relations": [
                "shared_founder",
                "acquired",
                "acquired_by",
            ],

            "typed_relation_channels":
                EXPECTED_TYPED_RELATIONS,
        },

        "model_ready_graph": {
            "node_index":
                str(
                    NODE_INDEX_PATH
                ),

            "relation_index":
                str(
                    RELATION_INDEX_PATH
                ),

            "edge_index":
                str(
                    EDGE_INDEX_PATH
                ),

            "edge_type":
                str(
                    EDGE_TYPE_PATH
                ),

            "edge_manifest":
                str(
                    EDGE_MANIFEST_PATH
                ),

            "variant_masks":
                str(
                    VARIANT_MASKS_PATH
                ),

            "node_structural_coverage":
                str(
                    NODE_COVERAGE_PATH
                ),
        },

        "identity_policy": {
            "graph_node_id":
                (
                    "role-namespaced: investor::<uuid> "
                    "or startup::<uuid>"
                ),

            "underlying_identity":
                "Crunchbase UUID",

            "dual_role_entities":
                EXPECTED_DUAL_ROLE_IDS,
        },

        "temporal_policy": {
            "graph_snapshot":
                "static",

            "acquisition_cutoff":
                "announced_on < 2026-01-01",

            "shared_founder_temporal_provenance":
                "current_snapshot_unversioned",

            "T60_holdout_pair_structural_mask":
                True,

            "T60_structural_leakage_edge_records":
                0,
        },

        "investment_events": {
            "included_as_structural_edges":
                False,

            "authoritative_temporal_layer":
                (
                    "Phase 2 frozen temporal interaction artifacts"
                ),
        },

        "graph_variants": {
            row.variant: {
                "relations":
                    row.relations,

                "edge_records":
                    int(
                        row.edge_records
                    ),

                "type":
                    row.primary_or_ablation,
            }
            for row in (
                final_variants.itertuples(
                    index=False
                )
            )
        },

        "coverage": {
            "connected_nodes":
                core_connected_nodes,

            "isolated_nodes":
                core_isolated_nodes,

            "connected_pct":
                (
                    core_connected_nodes
                    /
                    EXPECTED_ROLE_NODES
                    *
                    100
                ),

            "T60_pairs_both_endpoints_connected":
                int(
                    core_pair_dict[
                        "both_endpoints_connected"
                    ]
                ),

            "T60_pairs_neither_endpoint_connected":
                int(
                    core_pair_dict[
                        "neither_endpoint_connected"
                    ]
                ),
        },

        "checksums": {
            "nodes_parquet_sha256":
                sha256(
                    NODES_PATH
                ),

            "edges_parquet_sha256":
                sha256(
                    EDGES_PATH
                ),

            "node_index_parquet_sha256":
                sha256(
                    NODE_INDEX_PATH
                ),

            "relation_index_csv_sha256":
                sha256(
                    RELATION_INDEX_PATH
                ),

            "edge_index_npy_sha256":
                sha256(
                    EDGE_INDEX_PATH
                ),

            "edge_type_npy_sha256":
                sha256(
                    EDGE_TYPE_PATH
                ),

            "edge_manifest_parquet_sha256":
                sha256(
                    EDGE_MANIFEST_PATH
                ),

            "variant_masks_npz_sha256":
                sha256(
                    VARIANT_MASKS_PATH
                ),

            "node_structural_coverage_sha256":
                sha256(
                    NODE_COVERAGE_PATH
                ),

            "graph_schema_config_sha256":
                sha256(
                    GRAPH_SCHEMA_CONFIG_PATH
                ),

            "model_ready_metadata_sha256":
                sha256(
                    MODEL_METADATA_PATH
                ),
        },

        "next_phase":
            "Phase 4 — Model Reconstruction",
    }

    with open(
        FINAL_MANIFEST_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    # =========================================================================
    # 18. Terminal output
    # =========================================================================

    separator("-")

    print(
        "PHASE-3 FINAL INTEGRITY"
    )

    separator("-")

    print(
        final_integrity.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "PHASE-3 FINAL COUNTS"
    )

    separator("-")

    print(
        final_counts.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "FINAL TYPED RELATION VOCABULARY"
    )

    separator("-")

    print(
        final_relation_vocab.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "FINAL GRAPH VARIANTS"
    )

    separator("-")

    print(
        final_variants.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "KNOWN LIMITATIONS"
    )

    separator("-")

    print(
        final_limitations.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "PHASE-3 CLOSURE MANIFEST"
    )

    separator("-")

    print(
        FINAL_MANIFEST_OUTPUT
    )

    separator()

    print(
        "PHASE 3 COMPLETE"
    )

    separator()

    print(
        """
Phase 3 has now frozen:

    canonical graph identity
    graph node table
    graph edge table
    semantic relations
    typed relation IDs
    model-ready node indices
    model-ready edge tensors
    graph variant masks
    structural coverage diagnostics
    leakage policy
    temporal provenance
    known limitations

NO graph semantics should be changed in Phase 4.

NEXT:

    Phase 4 — Model Reconstruction
"""
    )


if __name__ == "__main__":
    main()