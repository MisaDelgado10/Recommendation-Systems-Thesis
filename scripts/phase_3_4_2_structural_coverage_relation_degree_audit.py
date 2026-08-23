from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PHASE 3.4.2 — MODEL-READY STRUCTURAL COVERAGE &
#               RELATION-DEGREE AUDIT
# =============================================================================
#
# PURPOSE
# -------
#
# Audit how structural information is distributed across the frozen
# model-ready graph.
#
# Specifically:
#
#   - compare founder-only vs acquisition-only node coverage;
#   - measure whether the two structural sources are complementary;
#   - measure graph coverage by Investor / Startup node type;
#   - audit structural coverage of T60 held-out endpoints and pairs;
#   - measure degree statistics for each graph variant;
#   - measure node coverage for each of the 12 typed relation channels;
#   - verify model-ready masks still reconstruct the frozen core graph.
#
# THIS SCRIPT DOES NOT:
#
#   - add or remove nodes;
#   - add or remove edges;
#   - filter isolates;
#   - change graph variants;
#   - change relation IDs;
#   - add investment events to the structural graph.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================


MODEL_DIR = Path(
    "data/experimental/phase_3/model_ready"
)

AUDIT_DIR = Path(
    "data/experimental/phase_3/audits"
)

FIGURE_DIR = Path(
    "data/experimental/phase_3/figures"
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

VARIANT_MASK_PATH = (
    MODEL_DIR
    / "graph_variant_masks.npz"
)


T60_HOLDOUT_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "t60_holdout_pair_manifest.parquet"
)


# =============================================================================
# OUTPUTS
# =============================================================================


INPUT_INTEGRITY_OUTPUT = (
    AUDIT_DIR
    / "model_ready_structural_coverage_input_integrity.csv"
)

VARIANT_COVERAGE_OUTPUT = (
    AUDIT_DIR
    / "graph_variant_node_coverage.csv"
)

SOURCE_OVERLAP_OUTPUT = (
    AUDIT_DIR
    / "structural_source_node_overlap.csv"
)

VARIANT_DEGREE_OUTPUT = (
    AUDIT_DIR
    / "graph_variant_degree_summary.csv"
)

TYPED_RELATION_COVERAGE_OUTPUT = (
    AUDIT_DIR
    / "typed_relation_node_coverage.csv"
)

T60_ENDPOINT_OUTPUT = (
    AUDIT_DIR
    / "t60_holdout_endpoint_structural_coverage.csv"
)

T60_PAIR_OUTPUT = (
    AUDIT_DIR
    / "t60_holdout_pair_structural_coverage.csv"
)

NODE_COVERAGE_OUTPUT = (
    MODEL_DIR
    / "node_structural_coverage.parquet"
)


VARIANT_COVERAGE_FIGURE = (
    FIGURE_DIR
    / "graph_variant_node_coverage.png"
)

SOURCE_OVERLAP_FIGURE = (
    FIGURE_DIR
    / "structural_source_node_overlap.png"
)

T60_PAIR_COVERAGE_FIGURE = (
    FIGURE_DIR
    / "t60_holdout_pair_structural_coverage.png"
)


# =============================================================================
# FROZEN EXPECTATIONS
# =============================================================================


EXPECTED_NODES = 477_564
EXPECTED_EDGES = 158_818
EXPECTED_TYPED_RELATIONS = 12

EXPECTED_CORE_EDGES = 158_818
EXPECTED_FOUNDER_EDGES = 94_818
EXPECTED_ACQUISITION_EDGES = 64_000

EXPECTED_T60_DIRECTED_PAIRS = 22_327


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


def pct(
    numerator,
    denominator,
):

    if denominator == 0:

        return np.nan

    return (
        numerator
        /
        denominator
        *
        100
    )


def summarize_degree(
    values,
):

    values = np.asarray(
        values,
        dtype=np.int64,
    )

    positive = (
        values[
            values > 0
        ]
    )

    return {
        "mean_all_nodes":
            float(
                values.mean()
            ),

        "median_all_nodes":
            float(
                np.median(
                    values
                )
            ),

        "p95_all_nodes":
            float(
                np.quantile(
                    values,
                    0.95,
                )
            ),

        "p99_all_nodes":
            float(
                np.quantile(
                    values,
                    0.99,
                )
            ),

        "max":
            int(
                values.max()
                if len(values)
                > 0
                else 0
            ),

        "mean_connected_nodes":
            (
                float(
                    positive.mean()
                )
                if len(
                    positive
                )
                > 0
                else 0.0
            ),

        "median_connected_nodes":
            (
                float(
                    np.median(
                        positive
                    )
                )
                if len(
                    positive
                )
                > 0
                else 0.0
            ),
    }


def compute_variant_degrees(
    edge_index,
    mask,
    node_count,
):

    selected = (
        edge_index[
            :,
            mask,
        ]
    )

    out_degree = np.bincount(
        selected[
            0
        ],
        minlength=node_count,
    ).astype(
        np.int64
    )

    in_degree = np.bincount(
        selected[
            1
        ],
        minlength=node_count,
    ).astype(
        np.int64
    )

    total_degree = (
        out_degree
        +
        in_degree
    )

    return (
        out_degree,
        in_degree,
        total_degree,
    )


def coverage_category(
    investor_connected,
    startup_connected,
):

    if (
        investor_connected
        and
        startup_connected
    ):

        return (
            "both_endpoints_connected"
        )

    if investor_connected:

        return (
            "investor_only_connected"
        )

    if startup_connected:

        return (
            "startup_only_connected"
        )

    return (
        "neither_endpoint_connected"
    )


# =============================================================================
# MAIN
# =============================================================================


def main():

    separator()

    print(
        "PHASE 3.4.2 — "
        "MODEL-READY STRUCTURAL COVERAGE & "
        "RELATION-DEGREE AUDIT"
    )

    separator()

    # =========================================================================
    # 1. Load frozen model-ready artifacts
    # =========================================================================

    nodes = pd.read_parquet(
        NODE_INDEX_PATH
    )

    relation_index = pd.read_csv(
        RELATION_INDEX_PATH
    )

    edge_manifest = pd.read_parquet(
        EDGE_MANIFEST_PATH
    )

    edge_index = np.load(
        EDGE_INDEX_PATH
    )

    edge_type = np.load(
        EDGE_TYPE_PATH
    )

    masks = np.load(
        VARIANT_MASK_PATH
    )

    # =========================================================================
    # 2. Basic input integrity
    # =========================================================================

    core_mask = (
        masks[
            "core"
        ]
        .astype(
            bool
        )
    )

    founder_mask = (
        masks[
            "founder_only_ablation"
        ]
        .astype(
            bool
        )
    )

    acquisition_mask = (
        masks[
            "acquisition_only_ablation"
        ]
        .astype(
            bool
        )
    )

    input_checks = []

    def add_check(
        check,
        observed,
        expected,
    ):

        input_checks.append(
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

    add_check(
        "node_rows",
        len(
            nodes
        ),
        EXPECTED_NODES,
    )

    add_check(
        "edge_manifest_rows",
        len(
            edge_manifest
        ),
        EXPECTED_EDGES,
    )

    add_check(
        "edge_index_rows",
        int(
            edge_index.shape[
                0
            ]
        ),
        2,
    )

    add_check(
        "edge_index_columns",
        int(
            edge_index.shape[
                1
            ]
        ),
        EXPECTED_EDGES,
    )

    add_check(
        "edge_type_length",
        int(
            len(
                edge_type
            )
        ),
        EXPECTED_EDGES,
    )

    add_check(
        "typed_relation_count",
        len(
            relation_index
        ),
        EXPECTED_TYPED_RELATIONS,
    )

    add_check(
        "core_mask_length",
        len(
            core_mask
        ),
        EXPECTED_EDGES,
    )

    add_check(
        "founder_mask_length",
        len(
            founder_mask
        ),
        EXPECTED_EDGES,
    )

    add_check(
        "acquisition_mask_length",
        len(
            acquisition_mask
        ),
        EXPECTED_EDGES,
    )

    add_check(
        "core_mask_edges",
        int(
            core_mask.sum()
        ),
        EXPECTED_CORE_EDGES,
    )

    add_check(
        "founder_mask_edges",
        int(
            founder_mask.sum()
        ),
        EXPECTED_FOUNDER_EDGES,
    )

    add_check(
        "acquisition_mask_edges",
        int(
            acquisition_mask.sum()
        ),
        EXPECTED_ACQUISITION_EDGES,
    )

    add_check(
        "founder_acquisition_mask_overlap",
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
        "founder_acquisition_union_equals_core",
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

    input_integrity = pd.DataFrame(
        input_checks
    )

    if (
        input_integrity[
            "status"
        ]
        .ne(
            "PASS"
        )
        .any()
    ):

        raise ValueError(
            "Model-ready input integrity failed."
        )

    # =========================================================================
    # 3. Verify edge manifest is position-aligned
    # =========================================================================

    if not np.array_equal(
        edge_manifest[
            "edge_position"
        ]
        .to_numpy(
            dtype=np.int64
        ),
        np.arange(
            EXPECTED_EDGES,
            dtype=np.int64,
        ),
    ):

        raise ValueError(
            "edge_manifest edge_position is not contiguous "
            "and aligned with tensor positions."
        )

    if not np.array_equal(
        edge_manifest[
            "src_index"
        ]
        .to_numpy(
            dtype=np.int64
        ),
        edge_index[
            0
        ],
    ):

        raise ValueError(
            "edge_manifest source indices do not "
            "match edge_index row 0."
        )

    if not np.array_equal(
        edge_manifest[
            "dst_index"
        ]
        .to_numpy(
            dtype=np.int64
        ),
        edge_index[
            1
        ],
    ):

        raise ValueError(
            "edge_manifest destination indices do not "
            "match edge_index row 1."
        )

    if not np.array_equal(
        edge_manifest[
            "relation_id"
        ]
        .to_numpy(
            dtype=np.int64
        ),
        edge_type,
    ):

        raise ValueError(
            "edge_manifest relation IDs do not match edge_type."
        )

    # =========================================================================
    # 4. Compute variant degree vectors
    # =========================================================================

    variants = {
        "core":
            core_mask,

        "founder_only_ablation":
            founder_mask,

        "acquisition_only_ablation":
            acquisition_mask,
    }

    degree_vectors = {}

    for (
        variant,
        mask,
    ) in variants.items():

        (
            out_degree,
            in_degree,
            total_degree,
        ) = compute_variant_degrees(
            edge_index=edge_index,
            mask=mask,
            node_count=EXPECTED_NODES,
        )

        degree_vectors[
            variant
        ] = {
            "out":
                out_degree,

            "in":
                in_degree,

            "total":
                total_degree,

            "connected":
                (
                    total_degree
                    > 0
                ),
        }

    # =========================================================================
    # 5. Variant node coverage by graph role
    # =========================================================================

    node_types = (
        nodes[
            "node_type"
        ]
        .astype(str)
        .to_numpy()
    )

    variant_coverage_rows = []

    for variant in variants:

        connected = (
            degree_vectors[
                variant
            ][
                "connected"
            ]
        )

        for node_type in [
            "investor",
            "startup",
            "ALL",
        ]:

            if (
                node_type
                ==
                "ALL"
            ):

                type_mask = np.ones(
                    EXPECTED_NODES,
                    dtype=bool,
                )

            else:

                type_mask = (
                    node_types
                    ==
                    node_type
                )

            node_count = int(
                type_mask.sum()
            )

            connected_count = int(
                (
                    connected
                    &
                    type_mask
                )
                .sum()
            )

            isolated_count = (
                node_count
                -
                connected_count
            )

            variant_coverage_rows.append(
                {
                    "variant":
                        variant,

                    "node_type":
                        node_type,

                    "node_count":
                        node_count,

                    "connected_nodes":
                        connected_count,

                    "isolated_nodes":
                        isolated_count,

                    "connected_pct":
                        pct(
                            connected_count,
                            node_count,
                        ),

                    "isolated_pct":
                        pct(
                            isolated_count,
                            node_count,
                        ),
                }
            )

    variant_coverage = pd.DataFrame(
        variant_coverage_rows
    )

    # =========================================================================
    # 6. Founder / acquisition source overlap
    # =========================================================================

    founder_connected = (
        degree_vectors[
            "founder_only_ablation"
        ][
            "connected"
        ]
    )

    acquisition_connected = (
        degree_vectors[
            "acquisition_only_ablation"
        ][
            "connected"
        ]
    )

    both_sources = (
        founder_connected
        &
        acquisition_connected
    )

    founder_only_nodes = (
        founder_connected
        &
        ~acquisition_connected
    )

    acquisition_only_nodes = (
        acquisition_connected
        &
        ~founder_connected
    )

    neither_source = (
        ~founder_connected
        &
        ~acquisition_connected
    )

    overlap_categories = {
        "founder_and_acquisition":
            both_sources,

        "founder_only":
            founder_only_nodes,

        "acquisition_only":
            acquisition_only_nodes,

        "neither":
            neither_source,
    }

    overlap_rows = []

    for node_type in [
        "investor",
        "startup",
        "ALL",
    ]:

        if (
            node_type
            ==
            "ALL"
        ):

            type_mask = np.ones(
                EXPECTED_NODES,
                dtype=bool,
            )

        else:

            type_mask = (
                node_types
                ==
                node_type
            )

        type_count = int(
            type_mask.sum()
        )

        for (
            coverage_class,
            class_mask,
        ) in overlap_categories.items():

            count = int(
                (
                    class_mask
                    &
                    type_mask
                )
                .sum()
            )

            overlap_rows.append(
                {
                    "node_type":
                        node_type,

                    "coverage_class":
                        coverage_class,

                    "node_count":
                        count,

                    "share_of_node_type_pct":
                        pct(
                            count,
                            type_count,
                        ),
                }
            )

    source_overlap = pd.DataFrame(
        overlap_rows
    )

    # =========================================================================
    # 7. Variant degree summaries
    # =========================================================================

    variant_degree_rows = []

    for variant in variants:

        for node_type in [
            "investor",
            "startup",
            "ALL",
        ]:

            if (
                node_type
                ==
                "ALL"
            ):

                type_mask = np.ones(
                    EXPECTED_NODES,
                    dtype=bool,
                )

            else:

                type_mask = (
                    node_types
                    ==
                    node_type
                )

            for degree_measure in [
                "out",
                "in",
                "total",
            ]:

                values = (
                    degree_vectors[
                        variant
                    ][
                        degree_measure
                    ][
                        type_mask
                    ]
                )

                stats = summarize_degree(
                    values
                )

                variant_degree_rows.append(
                    {
                        "variant":
                            variant,

                        "node_type":
                            node_type,

                        "degree_measure":
                            degree_measure,

                        **stats,
                    }
                )

    variant_degree_summary = pd.DataFrame(
        variant_degree_rows
    )

    # =========================================================================
    # 8. Typed relation channel node coverage
    # =========================================================================

    typed_relation_rows = []

    for relation_row in (
        relation_index.itertuples(
            index=False
        )
    ):

        relation_id = int(
            relation_row.relation_id
        )

        relation_mask = (
            edge_type
            ==
            relation_id
        )

        relation_edges = (
            edge_index[
                :,
                relation_mask,
            ]
        )

        source_nodes = np.unique(
            relation_edges[
                0
            ]
        )

        destination_nodes = np.unique(
            relation_edges[
                1
            ]
        )

        incident_nodes = np.unique(
            np.concatenate(
                [
                    source_nodes,
                    destination_nodes,
                ]
            )
        )

        source_degree = np.bincount(
            relation_edges[
                0
            ],
            minlength=EXPECTED_NODES,
        )

        destination_degree = np.bincount(
            relation_edges[
                1
            ],
            minlength=EXPECTED_NODES,
        )

        incident_degree = (
            source_degree
            +
            destination_degree
        )

        positive_incident_degree = (
            incident_degree[
                incident_degree > 0
            ]
        )

        typed_relation_rows.append(
            {
                "relation_id":
                    relation_id,

                "src_type":
                    relation_row.src_type,

                "relation":
                    relation_row.relation,

                "dst_type":
                    relation_row.dst_type,

                "typed_relation_key":
                    relation_row.typed_relation_key,

                "edge_records":
                    int(
                        relation_mask.sum()
                    ),

                "unique_source_nodes":
                    len(
                        source_nodes
                    ),

                "unique_destination_nodes":
                    len(
                        destination_nodes
                    ),

                "unique_incident_nodes":
                    len(
                        incident_nodes
                    ),

                "mean_incident_degree_connected_nodes":
                    (
                        float(
                            positive_incident_degree.mean()
                        )
                        if len(
                            positive_incident_degree
                        )
                        > 0
                        else 0.0
                    ),

                "max_incident_degree":
                    int(
                        incident_degree.max()
                    ),
            }
        )

    typed_relation_coverage = pd.DataFrame(
        typed_relation_rows
    )

    # =========================================================================
    # 9. Build node-level coverage artifact
    # =========================================================================

    node_coverage = nodes[
        [
            "node_index",
            "node_id",
            "node_type",
            "raw_entity_id",
        ]
    ].copy()

    for variant in variants:

        node_coverage[
            f"{variant}_out_degree"
        ] = (
            degree_vectors[
                variant
            ][
                "out"
            ]
        )

        node_coverage[
            f"{variant}_in_degree"
        ] = (
            degree_vectors[
                variant
            ][
                "in"
            ]
        )

        node_coverage[
            f"{variant}_total_degree"
        ] = (
            degree_vectors[
                variant
            ][
                "total"
            ]
        )

        node_coverage[
            f"{variant}_connected"
        ] = (
            degree_vectors[
                variant
            ][
                "connected"
            ]
        )

    node_coverage[
        "structural_source_class"
    ] = np.select(
        [
            both_sources,
            founder_only_nodes,
            acquisition_only_nodes,
        ],
        [
            "founder_and_acquisition",
            "founder_only",
            "acquisition_only",
        ],
        default="neither",
    )

    # =========================================================================
    # 10. T60 held-out endpoint universe
    # =========================================================================

    holdout = pd.read_parquet(
        T60_HOLDOUT_PATH
    )

    required_holdout_columns = {
        "investor_id",
        "startup_id",
    }

    if not (
        required_holdout_columns
        .issubset(
            holdout.columns
        )
    ):

        raise ValueError(
            "T60 holdout manifest missing "
            "investor_id/startup_id."
        )

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

    holdout_pairs = (
        holdout[
            [
                "investor_id",
                "startup_id",
            ]
        ]
        .drop_duplicates()
        .reset_index(
            drop=True
        )
    )

    if (
        len(
            holdout_pairs
        )
        !=
        EXPECTED_T60_DIRECTED_PAIRS
    ):

        raise ValueError(
            "Unexpected T60 unique directed-pair count: "
            f"{len(holdout_pairs):,}"
        )

    node_id_to_index = (
        nodes.set_index(
            "node_id"
        )[
            "node_index"
        ]
        .to_dict()
    )

    holdout_pairs[
        "investor_node_id"
    ] = (
        "investor::"
        +
        holdout_pairs[
            "investor_id"
        ]
    )

    holdout_pairs[
        "startup_node_id"
    ] = (
        "startup::"
        +
        holdout_pairs[
            "startup_id"
        ]
    )

    holdout_pairs[
        "investor_node_index"
    ] = (
        holdout_pairs[
            "investor_node_id"
        ]
        .map(
            node_id_to_index
        )
    )

    holdout_pairs[
        "startup_node_index"
    ] = (
        holdout_pairs[
            "startup_node_id"
        ]
        .map(
            node_id_to_index
        )
    )

    if (
        holdout_pairs[
            "investor_node_index"
        ]
        .isna()
        .any()
        or
        holdout_pairs[
            "startup_node_index"
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "At least one T60 holdout endpoint "
            "is missing from the frozen node universe."
        )

    holdout_pairs[
        "investor_node_index"
    ] = (
        holdout_pairs[
            "investor_node_index"
        ]
        .astype(
            np.int64
        )
    )

    holdout_pairs[
        "startup_node_index"
    ] = (
        holdout_pairs[
            "startup_node_index"
        ]
        .astype(
            np.int64
        )
    )

    # =========================================================================
    # 11. T60 endpoint coverage
    # =========================================================================

    unique_holdout_investor_indices = np.unique(
        holdout_pairs[
            "investor_node_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    unique_holdout_startup_indices = np.unique(
        holdout_pairs[
            "startup_node_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    endpoint_rows = []

    for variant in variants:

        connected = (
            degree_vectors[
                variant
            ][
                "connected"
            ]
        )

        for (
            endpoint_type,
            endpoint_indices,
        ) in [
            (
                "investor",
                unique_holdout_investor_indices,
            ),
            (
                "startup",
                unique_holdout_startup_indices,
            ),
        ]:

            connected_count = int(
                connected[
                    endpoint_indices
                ]
                .sum()
            )

            total_count = len(
                endpoint_indices
            )

            endpoint_rows.append(
                {
                    "variant":
                        variant,

                    "endpoint_type":
                        endpoint_type,

                    "unique_T60_endpoints":
                        total_count,

                    "connected_endpoints":
                        connected_count,

                    "isolated_endpoints":
                        (
                            total_count
                            -
                            connected_count
                        ),

                    "connected_pct":
                        pct(
                            connected_count,
                            total_count,
                        ),
                }
            )

    t60_endpoint_coverage = pd.DataFrame(
        endpoint_rows
    )

    # =========================================================================
    # 12. T60 pair-level coverage
    # =========================================================================

    pair_coverage_rows = []

    for variant in variants:

        connected = (
            degree_vectors[
                variant
            ][
                "connected"
            ]
        )

        investor_connected = connected[
            holdout_pairs[
                "investor_node_index"
            ]
            .to_numpy(
                dtype=np.int64
            )
        ]

        startup_connected = connected[
            holdout_pairs[
                "startup_node_index"
            ]
            .to_numpy(
                dtype=np.int64
            )
        ]

        categories = [
            coverage_category(
                bool(
                    investor_value
                ),
                bool(
                    startup_value
                ),
            )
            for (
                investor_value,
                startup_value,
            ) in zip(
                investor_connected,
                startup_connected,
            )
        ]

        category_counts = (
            pd.Series(
                categories
            )
            .value_counts()
        )

        for category in [
            "both_endpoints_connected",
            "investor_only_connected",
            "startup_only_connected",
            "neither_endpoint_connected",
        ]:

            count = int(
                category_counts.get(
                    category,
                    0,
                )
            )

            pair_coverage_rows.append(
                {
                    "variant":
                        variant,

                    "coverage_class":
                        category,

                    "pair_count":
                        count,

                    "share_pct":
                        pct(
                            count,
                            len(
                                holdout_pairs
                            ),
                        ),
                }
            )

    t60_pair_coverage = pd.DataFrame(
        pair_coverage_rows
    )

    # =========================================================================
    # 13. Save outputs
    # =========================================================================

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_integrity.to_csv(
        INPUT_INTEGRITY_OUTPUT,
        index=False,
    )

    variant_coverage.to_csv(
        VARIANT_COVERAGE_OUTPUT,
        index=False,
    )

    source_overlap.to_csv(
        SOURCE_OVERLAP_OUTPUT,
        index=False,
    )

    variant_degree_summary.to_csv(
        VARIANT_DEGREE_OUTPUT,
        index=False,
    )

    typed_relation_coverage.to_csv(
        TYPED_RELATION_COVERAGE_OUTPUT,
        index=False,
    )

    t60_endpoint_coverage.to_csv(
        T60_ENDPOINT_OUTPUT,
        index=False,
    )

    t60_pair_coverage.to_csv(
        T60_PAIR_OUTPUT,
        index=False,
    )

    node_coverage.to_parquet(
        NODE_COVERAGE_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 14. Figures
    # =========================================================================

    coverage_plot = (
        variant_coverage[
            variant_coverage[
                "node_type"
            ]
            !=
            "ALL"
        ]
        .copy()
    )

    pivot = (
        coverage_plot.pivot(
            index="node_type",
            columns="variant",
            values="connected_pct",
        )
    )

    ax = pivot.plot(
        kind="bar",
        figsize=(9, 5),
    )

    ax.set_ylabel(
        "Nodes with structural neighbor (%)"
    )

    ax.set_title(
        "Structural Coverage by Graph Variant"
    )

    ax.tick_params(
        axis="x",
        rotation=0,
    )

    ax.figure.tight_layout()

    ax.figure.savefig(
        VARIANT_COVERAGE_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        ax.figure
    )

    # -------------------------------------------------------------------------

    overlap_all = (
        source_overlap[
            source_overlap[
                "node_type"
            ]
            ==
            "ALL"
        ]
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        overlap_all[
            "coverage_class"
        ],
        overlap_all[
            "node_count"
        ],
    )

    ax.set_ylabel(
        "Graph nodes"
    )

    ax.set_title(
        "Founder vs Acquisition Structural Coverage"
    )

    ax.tick_params(
        axis="x",
        rotation=15,
    )

    fig.tight_layout()

    fig.savefig(
        SOURCE_OVERLAP_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # -------------------------------------------------------------------------

    pair_plot = (
        t60_pair_coverage.pivot(
            index="coverage_class",
            columns="variant",
            values="share_pct",
        )
    )

    ax = pair_plot.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_ylabel(
        "T60 held-out pairs (%)"
    )

    ax.set_title(
        "Structural Information Available to T60 Prediction Pairs"
    )

    ax.tick_params(
        axis="x",
        rotation=15,
    )

    ax.figure.tight_layout()

    ax.figure.savefig(
        T60_PAIR_COVERAGE_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        ax.figure
    )

    # =========================================================================
    # 15. Terminal report
    # =========================================================================

    separator("-")

    print(
        "MODEL-READY INPUT INTEGRITY"
    )

    separator("-")

    print(
        input_integrity.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "GRAPH VARIANT NODE COVERAGE"
    )

    separator("-")

    print(
        variant_coverage.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "STRUCTURAL SOURCE OVERLAP"
    )

    separator("-")

    print(
        source_overlap.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "GRAPH VARIANT DEGREE SUMMARY"
    )

    separator("-")

    print(
        variant_degree_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "TYPED RELATION NODE COVERAGE"
    )

    separator("-")

    print(
        typed_relation_coverage.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "T60 HOLDOUT ENDPOINT COVERAGE"
    )

    separator("-")

    print(
        t60_endpoint_coverage.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "T60 HOLDOUT PAIR COVERAGE"
    )

    separator("-")

    print(
        t60_pair_coverage.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "NODE-LEVEL STRUCTURAL COVERAGE ARTIFACT"
    )

    separator("-")

    print(
        f"{NODE_COVERAGE_OUTPUT}"
    )

    print(
        f"Rows:    {len(node_coverage):,}"
    )

    print(
        f"Columns: {len(node_coverage.columns):,}"
    )

    separator()

    print(
        "PHASE 3.4.2 COMPLETE"
    )

    separator()

    print(
        f"""
Outputs written to:

{INPUT_INTEGRITY_OUTPUT}
{VARIANT_COVERAGE_OUTPUT}
{SOURCE_OVERLAP_OUTPUT}
{VARIANT_DEGREE_OUTPUT}
{TYPED_RELATION_COVERAGE_OUTPUT}
{T60_ENDPOINT_OUTPUT}
{T60_PAIR_OUTPUT}

Model-ready diagnostic artifact:

{NODE_COVERAGE_OUTPUT}

Figures written to:

{VARIANT_COVERAGE_FIGURE}
{SOURCE_OVERLAP_FIGURE}
{T60_PAIR_COVERAGE_FIGURE}


IMPORTANT

1. No graph semantics were changed.

2. No node or edge was removed.

3. Structural isolates remain valid model nodes.

4. The audit distinguishes:
       founder-only coverage
       acquisition-only coverage
       coverage from both
       coverage from neither

5. T60 pair coverage is diagnostic only.
   It does NOT change the frozen Phase-2 test/validation universe.

6. Investment interactions remain separate from structural graph edges.

NEXT:

Phase 3.4.3 — Phase-3 Final Integrity & Closure

That subphase will freeze the complete Phase-3 deliverable set and
prepare the final Phase-3 documentation/reproduction record.
"""
    )


if __name__ == "__main__":
    main()