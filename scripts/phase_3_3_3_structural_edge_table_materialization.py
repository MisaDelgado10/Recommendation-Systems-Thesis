from pathlib import Path
from collections import Counter
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PHASE 3.3.3 — STRUCTURAL EDGE-TABLE MATERIALIZATION &
#               RELATION INTEGRITY AUDIT
# =============================================================================
#
# PURPOSE
# -------
#
# Materialize the frozen Phase-3.3 structural graph relations:
#
#   1. SHARED_FOUNDER
#        - symmetric
#        - stored in both directions with the same relation label
#
#   2. ACQUIRED
#        - directed
#
#   3. ACQUIRED_BY
#        - explicit inverse of ACQUIRED
#
# This script verifies:
#
#   - exact frozen edge counts;
#   - exact 12 typed relation channels;
#   - endpoint existence in nodes.parquet;
#   - source/destination role integrity;
#   - underlying UUID integrity;
#   - SHARED_FOUNDER symmetry;
#   - ACQUIRED / ACQUIRED_BY inverse consistency;
#   - no duplicate edge triples;
#   - no graph-node self loops;
#   - no same-underlying-entity relations;
#   - zero T60 held-out underlying-pair leakage;
#   - graph-node structural connectivity.
#
# Historical INVESTMENT_EVENT records remain outside this graph.
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

FIGURE_DIR = Path(
    "data/experimental/phase_3/figures"
)

CONFIG_PATH = Path(
    "configs/phase_3_itrs_graph_schema.json"
)


NODES_PATH = (
    GRAPH_DIR
    / "nodes.parquet"
)

SHARED_FOUNDER_PATH = (
    AUDIT_DIR
    / "shared_founder_strict_role_pair_candidates.parquet"
)

ACQUISITION_PATH = (
    AUDIT_DIR
    / "acquisition_strict_underlying_relation_candidates.parquet"
)

EXPECTED_TYPED_RELATION_PATH = (
    AUDIT_DIR
    / "phase_3_3_typed_relation_projection.csv"
)

HOLDOUT_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "t60_holdout_pair_manifest.parquet"
)


# =============================================================================
# OUTPUTS
# =============================================================================


EDGE_OUTPUT = (
    GRAPH_DIR
    / "edges.parquet"
)

INTEGRITY_SUMMARY_OUTPUT = (
    AUDIT_DIR
    / "graph_edge_integrity_summary.csv"
)

RELATION_SUMMARY_OUTPUT = (
    AUDIT_DIR
    / "graph_edge_relation_summary.csv"
)

TYPED_RELATION_SUMMARY_OUTPUT = (
    AUDIT_DIR
    / "graph_edge_typed_relation_summary.csv"
)

ENDPOINT_INTEGRITY_OUTPUT = (
    AUDIT_DIR
    / "graph_edge_endpoint_integrity.csv"
)

RECIPROCITY_OUTPUT = (
    AUDIT_DIR
    / "graph_edge_reciprocity_integrity.csv"
)

HOLDOUT_LEAKAGE_OUTPUT = (
    AUDIT_DIR
    / "graph_edge_holdout_leakage_audit.csv"
)

CONNECTIVITY_OUTPUT = (
    AUDIT_DIR
    / "graph_node_structural_connectivity_summary.csv"
)

DEGREE_OUTPUT = (
    AUDIT_DIR
    / "graph_node_degree_summary.csv"
)


RELATION_FIGURE = (
    FIGURE_DIR
    / "graph_edge_relation_counts.png"
)

TYPED_RELATION_FIGURE = (
    FIGURE_DIR
    / "graph_edge_typed_relation_counts.png"
)

CONNECTIVITY_FIGURE = (
    FIGURE_DIR
    / "graph_node_structural_connectivity.png"
)

DEGREE_FIGURE = (
    FIGURE_DIR
    / "graph_node_degree_distribution.png"
)


# =============================================================================
# FROZEN EXPECTATIONS
# =============================================================================


EXPECTED_NODE_COUNT = 477_564

EXPECTED_SHARED_DIRECTED_EDGES = 94_818

EXPECTED_ACQUIRED_EDGES = 32_000

EXPECTED_ACQUIRED_BY_EDGES = 32_000

EXPECTED_TOTAL_EDGES = 158_818

EXPECTED_TYPED_RELATIONS = 12


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


def clean_string(
    series,
):
    return (
        series
        .astype("string")
        .str.strip()
        .replace(
            "",
            pd.NA,
        )
    )


def role_memberships(
    label,
):

    if label == "investor":

        return [
            "investor"
        ]

    if label == "startup":

        return [
            "startup"
        ]

    if label == "investor+startup":

        return [
            "investor",
            "startup",
        ]

    raise ValueError(
        f"Unknown role membership: {label}"
    )


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


def make_edge_id(
    relation,
    src_node_id,
    dst_node_id,
):

    payload = (
        f"{relation}|"
        f"{src_node_id}|"
        f"{dst_node_id}"
    )

    digest = hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()

    return (
        f"edge::{digest}"
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


# =============================================================================
# MAIN
# =============================================================================


def main():

    separator()

    print(
        "PHASE 3.3.3 — "
        "STRUCTURAL EDGE-TABLE MATERIALIZATION & "
        "RELATION INTEGRITY AUDIT"
    )

    separator()

    # =========================================================================
    # 1. Load frozen graph specification
    # =========================================================================

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(
            f
        )

    config_expected_edges = int(
        config[
            "expected_counts"
        ][
            "core_directed_edges"
        ]
    )

    config_expected_channels = int(
        config[
            "expected_counts"
        ][
            "typed_relation_channels"
        ]
    )

    if (
        config_expected_edges
        !=
        EXPECTED_TOTAL_EDGES
    ):

        raise ValueError(
            "Frozen config edge count differs from "
            "Phase-3.3.3 expectation."
        )

    if (
        config_expected_channels
        !=
        EXPECTED_TYPED_RELATIONS
    ):

        raise ValueError(
            "Frozen config typed-relation count differs "
            "from Phase-3.3.3 expectation."
        )

    # =========================================================================
    # 2. Load frozen node table
    # =========================================================================

    nodes = pd.read_parquet(
        NODES_PATH
    )

    required_node_columns = {
        "node_id",
        "node_type",
        "raw_entity_id",
        "display_name",
        "source_registry",
        "underlying_entity_has_dual_role",
    }

    missing_node_columns = (
        required_node_columns
        -
        set(
            nodes.columns
        )
    )

    if missing_node_columns:

        raise ValueError(
            "nodes.parquet missing frozen columns: "
            f"{sorted(missing_node_columns)}"
        )

    if len(
        nodes
    ) != EXPECTED_NODE_COUNT:

        raise ValueError(
            f"Expected {EXPECTED_NODE_COUNT:,} nodes, "
            f"found {len(nodes):,}."
        )

    if (
        nodes[
            "node_id"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "nodes.parquet contains duplicate node_id values."
        )

    node_type_map = (
        nodes.set_index(
            "node_id"
        )[
            "node_type"
        ]
        .to_dict()
    )

    node_underlying_map = (
        nodes.set_index(
            "node_id"
        )[
            "raw_entity_id"
        ]
        .astype(str)
        .to_dict()
    )

    node_id_set = set(
        nodes[
            "node_id"
        ]
    )

    print(
        f"\nFrozen graph nodes: "
        f"{len(nodes):,}"
    )

    # =========================================================================
    # 3. Materialize SHARED_FOUNDER directed edges
    # =========================================================================

    shared = pd.read_parquet(
        SHARED_FOUNDER_PATH
    )

    required_shared_columns = {
        "role_node_a",
        "role_node_b",
        "underlying_entity_a",
        "underlying_entity_b",
        "role_a",
        "role_b",
        "shared_founder_count",
    }

    missing_shared_columns = (
        required_shared_columns
        -
        set(
            shared.columns
        )
    )

    if missing_shared_columns:

        raise ValueError(
            "SHARED_FOUNDER candidate artifact "
            "missing columns: "
            f"{sorted(missing_shared_columns)}"
        )

    shared_forward = pd.DataFrame(
        {
            "src_node_id":
                shared[
                    "role_node_a"
                ],

            "dst_node_id":
                shared[
                    "role_node_b"
                ],

            "src_type":
                shared[
                    "role_a"
                ],

            "dst_type":
                shared[
                    "role_b"
                ],

            "relation":
                "shared_founder",

            "underlying_src_id":
                shared[
                    "underlying_entity_a"
                ],

            "underlying_dst_id":
                shared[
                    "underlying_entity_b"
                ],

            "provenance_source":
                (
                    "companies.founder_links->people.link"
                ),

            "temporal_provenance":
                (
                    "current_snapshot_unversioned"
                ),

            "first_observed_on":
                pd.NaT,

            "last_observed_on":
                pd.NaT,

            "support_count":
                shared[
                    "shared_founder_count"
                ]
                .astype(
                    "int64"
                ),

            "is_inverse_edge":
                False,
        }
    )

    shared_reverse = pd.DataFrame(
        {
            "src_node_id":
                shared[
                    "role_node_b"
                ],

            "dst_node_id":
                shared[
                    "role_node_a"
                ],

            "src_type":
                shared[
                    "role_b"
                ],

            "dst_type":
                shared[
                    "role_a"
                ],

            "relation":
                "shared_founder",

            "underlying_src_id":
                shared[
                    "underlying_entity_b"
                ],

            "underlying_dst_id":
                shared[
                    "underlying_entity_a"
                ],

            "provenance_source":
                (
                    "companies.founder_links->people.link"
                ),

            "temporal_provenance":
                (
                    "current_snapshot_unversioned"
                ),

            "first_observed_on":
                pd.NaT,

            "last_observed_on":
                pd.NaT,

            "support_count":
                shared[
                    "shared_founder_count"
                ]
                .astype(
                    "int64"
                ),

            "is_inverse_edge":
                False,
        }
    )

    shared_edges = pd.concat(
        [
            shared_forward,
            shared_reverse,
        ],
        ignore_index=True,
    )

    if (
        len(
            shared_edges
        )
        !=
        EXPECTED_SHARED_DIRECTED_EDGES
    ):

        raise ValueError(
            "Unexpected SHARED_FOUNDER directed edge count: "
            f"{len(shared_edges):,}"
        )

    # =========================================================================
    # 4. Materialize ACQUIRED / ACQUIRED_BY
    # =========================================================================

    acquisitions = pd.read_parquet(
        ACQUISITION_PATH
    )

    required_acquisition_columns = {
        "acquirer_resolved_id",
        "acquiree_resolved_id",
        "acquisition_event_count",
        "first_acquisition_date",
        "last_acquisition_date",
        "acquirer_role_membership",
        "acquiree_role_membership",
    }

    missing_acquisition_columns = (
        required_acquisition_columns
        -
        set(
            acquisitions.columns
        )
    )

    if missing_acquisition_columns:

        raise ValueError(
            "Acquisition candidate artifact missing columns: "
            f"{sorted(missing_acquisition_columns)}"
        )

    acquisitions[
        "first_acquisition_date"
    ] = pd.to_datetime(
        acquisitions[
            "first_acquisition_date"
        ],
        errors="coerce",
    )

    acquisitions[
        "last_acquisition_date"
    ] = pd.to_datetime(
        acquisitions[
            "last_acquisition_date"
        ],
        errors="coerce",
    )

    acquisition_forward_rows = []

    acquisition_inverse_rows = []

    for row in acquisitions.itertuples(
        index=False
    ):

        acquirer_id = str(
            row.acquirer_resolved_id
        )

        acquiree_id = str(
            row.acquiree_resolved_id
        )

        source_roles = role_memberships(
            row.acquirer_role_membership
        )

        target_roles = role_memberships(
            row.acquiree_role_membership
        )

        support_count = int(
            row.acquisition_event_count
        )

        first_date = (
            row.first_acquisition_date
        )

        last_date = (
            row.last_acquisition_date
        )

        for source_role in source_roles:

            for target_role in target_roles:

                source_node = (
                    f"{source_role}::{acquirer_id}"
                )

                target_node = (
                    f"{target_role}::{acquiree_id}"
                )

                acquisition_forward_rows.append(
                    {
                        "src_node_id":
                            source_node,

                        "dst_node_id":
                            target_node,

                        "src_type":
                            source_role,

                        "dst_type":
                            target_role,

                        "relation":
                            "acquired",

                        "underlying_src_id":
                            acquirer_id,

                        "underlying_dst_id":
                            acquiree_id,

                        "provenance_source":
                            "acquisitions",

                        "temporal_provenance":
                            "timestamped_pre_T60",

                        "first_observed_on":
                            first_date,

                        "last_observed_on":
                            last_date,

                        "support_count":
                            support_count,

                        "is_inverse_edge":
                            False,
                    }
                )

                acquisition_inverse_rows.append(
                    {
                        "src_node_id":
                            target_node,

                        "dst_node_id":
                            source_node,

                        "src_type":
                            target_role,

                        "dst_type":
                            source_role,

                        "relation":
                            "acquired_by",

                        "underlying_src_id":
                            acquiree_id,

                        "underlying_dst_id":
                            acquirer_id,

                        "provenance_source":
                            "acquisitions",

                        "temporal_provenance":
                            "timestamped_pre_T60",

                        "first_observed_on":
                            first_date,

                        "last_observed_on":
                            last_date,

                        "support_count":
                            support_count,

                        "is_inverse_edge":
                            True,
                    }
                )

    acquired_edges = pd.DataFrame(
        acquisition_forward_rows
    )

    acquired_by_edges = pd.DataFrame(
        acquisition_inverse_rows
    )

    if (
        len(
            acquired_edges
        )
        !=
        EXPECTED_ACQUIRED_EDGES
    ):

        raise ValueError(
            "Unexpected ACQUIRED edge count: "
            f"{len(acquired_edges):,}"
        )

    if (
        len(
            acquired_by_edges
        )
        !=
        EXPECTED_ACQUIRED_BY_EDGES
    ):

        raise ValueError(
            "Unexpected ACQUIRED_BY edge count: "
            f"{len(acquired_by_edges):,}"
        )

    # =========================================================================
    # 5. Combine structural graph
    # =========================================================================

    edges = pd.concat(
        [
            shared_edges,
            acquired_edges,
            acquired_by_edges,
        ],
        ignore_index=True,
    )

    if len(
        edges
    ) != EXPECTED_TOTAL_EDGES:

        raise ValueError(
            f"Expected {EXPECTED_TOTAL_EDGES:,} structural edges, "
            f"found {len(edges):,}."
        )

    # =========================================================================
    # 6. Frozen typed relation key
    # =========================================================================

    edges[
        "typed_relation_key"
    ] = (
        edges[
            "src_type"
        ]
        +
        "|"
        +
        edges[
            "relation"
        ]
        +
        "|"
        +
        edges[
            "dst_type"
        ]
    )

    # =========================================================================
    # 7. Deterministic edge IDs
    # =========================================================================

    edges[
        "edge_id"
    ] = [
        make_edge_id(
            relation,
            src_node,
            dst_node,
        )
        for relation, src_node, dst_node
        in zip(
            edges[
                "relation"
            ],
            edges[
                "src_node_id"
            ],
            edges[
                "dst_node_id"
            ],
        )
    ]

    # =========================================================================
    # 8. Frozen edge column order
    # =========================================================================

    edge_columns = [
        "edge_id",
        "src_node_id",
        "dst_node_id",
        "src_type",
        "dst_type",
        "relation",
        "typed_relation_key",
        "underlying_src_id",
        "underlying_dst_id",
        "provenance_source",
        "temporal_provenance",
        "first_observed_on",
        "last_observed_on",
        "support_count",
        "is_inverse_edge",
    ]

    edges = edges[
        edge_columns
    ].copy()

    # =========================================================================
    # 9. Basic edge integrity
    # =========================================================================

    duplicate_edge_ids = int(
        edges[
            "edge_id"
        ]
        .duplicated()
        .sum()
    )

    duplicate_edge_triples = int(
        edges[
            [
                "src_node_id",
                "relation",
                "dst_node_id",
            ]
        ]
        .duplicated()
        .sum()
    )

    graph_self_loops = int(
        edges[
            "src_node_id"
        ]
        .eq(
            edges[
                "dst_node_id"
            ]
        )
        .sum()
    )

    same_underlying_relations = int(
        edges[
            "underlying_src_id"
        ]
        .astype(str)
        .eq(
            edges[
                "underlying_dst_id"
            ]
            .astype(str)
        )
        .sum()
    )

    if duplicate_edge_ids != 0:

        raise ValueError(
            f"Duplicate edge IDs: "
            f"{duplicate_edge_ids:,}"
        )

    if duplicate_edge_triples != 0:

        raise ValueError(
            "Duplicate (src, relation, dst) edge triples: "
            f"{duplicate_edge_triples:,}"
        )

    if graph_self_loops != 0:

        raise ValueError(
            f"Graph node self loops found: "
            f"{graph_self_loops:,}"
        )

    if same_underlying_relations != 0:

        raise ValueError(
            "Same-underlying-entity structural relations found: "
            f"{same_underlying_relations:,}"
        )

    # =========================================================================
    # 10. Endpoint existence and identity integrity
    # =========================================================================

    missing_src_nodes = int(
        (
            ~edges[
                "src_node_id"
            ]
            .isin(
                node_id_set
            )
        )
        .sum()
    )

    missing_dst_nodes = int(
        (
            ~edges[
                "dst_node_id"
            ]
            .isin(
                node_id_set
            )
        )
        .sum()
    )

    src_type_mismatches = 0
    dst_type_mismatches = 0

    src_underlying_mismatches = 0
    dst_underlying_mismatches = 0

    for row in edges.itertuples(
        index=False
    ):

        if row.src_node_id in node_type_map:

            if (
                node_type_map[
                    row.src_node_id
                ]
                !=
                row.src_type
            ):

                src_type_mismatches += 1

            if (
                str(
                    node_underlying_map[
                        row.src_node_id
                    ]
                )
                !=
                str(
                    row.underlying_src_id
                )
            ):

                src_underlying_mismatches += 1

        if row.dst_node_id in node_type_map:

            if (
                node_type_map[
                    row.dst_node_id
                ]
                !=
                row.dst_type
            ):

                dst_type_mismatches += 1

            if (
                str(
                    node_underlying_map[
                        row.dst_node_id
                    ]
                )
                !=
                str(
                    row.underlying_dst_id
                )
            ):

                dst_underlying_mismatches += 1

    endpoint_integrity = pd.DataFrame(
        [
            {
                "metric":
                    "missing_source_nodes",

                "value":
                    missing_src_nodes,
            },

            {
                "metric":
                    "missing_destination_nodes",

                "value":
                    missing_dst_nodes,
            },

            {
                "metric":
                    "source_type_mismatches",

                "value":
                    src_type_mismatches,
            },

            {
                "metric":
                    "destination_type_mismatches",

                "value":
                    dst_type_mismatches,
            },

            {
                "metric":
                    "source_underlying_id_mismatches",

                "value":
                    src_underlying_mismatches,
            },

            {
                "metric":
                    "destination_underlying_id_mismatches",

                "value":
                    dst_underlying_mismatches,
            },
        ]
    )

    if (
        endpoint_integrity[
            "value"
        ]
        .sum()
        != 0
    ):

        raise ValueError(
            "Graph endpoint identity integrity failed."
        )

    # =========================================================================
    # 11. Relation and typed-relation count verification
    # =========================================================================

    relation_summary = (
        edges.groupby(
            "relation",
            observed=True,
        )
        .agg(
            directed_edge_records=(
                "edge_id",
                "size",
            ),

            unique_source_nodes=(
                "src_node_id",
                "nunique",
            ),

            unique_destination_nodes=(
                "dst_node_id",
                "nunique",
            ),

            unique_underlying_source_ids=(
                "underlying_src_id",
                "nunique",
            ),

            unique_underlying_destination_ids=(
                "underlying_dst_id",
                "nunique",
            ),
        )
        .reset_index()
    )

    expected_relation_counts = {
        "shared_founder":
            EXPECTED_SHARED_DIRECTED_EDGES,

        "acquired":
            EXPECTED_ACQUIRED_EDGES,

        "acquired_by":
            EXPECTED_ACQUIRED_BY_EDGES,
    }

    observed_relation_counts = (
        relation_summary.set_index(
            "relation"
        )[
            "directed_edge_records"
        ]
        .to_dict()
    )

    if (
        observed_relation_counts
        !=
        expected_relation_counts
    ):

        raise ValueError(
            "Base relation counts do not match frozen schema."
        )

    typed_relation_summary = (
        edges[
            "typed_relation_key"
        ]
        .value_counts()
        .rename_axis(
            "typed_relation_key"
        )
        .reset_index(
            name="directed_edge_records"
        )
    )

    typed_relation_summary[
        [
            "src_type",
            "relation",
            "dst_type",
        ]
    ] = (
        typed_relation_summary[
            "typed_relation_key"
        ]
        .str.split(
            "|",
            expand=True,
        )
    )

    typed_relation_summary = (
        typed_relation_summary[
            [
                "src_type",
                "relation",
                "dst_type",
                "typed_relation_key",
                "directed_edge_records",
            ]
        ]
        .sort_values(
            [
                "src_type",
                "relation",
                "dst_type",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if (
        len(
            typed_relation_summary
        )
        !=
        EXPECTED_TYPED_RELATIONS
    ):

        raise ValueError(
            "Unexpected number of typed relation channels: "
            f"{len(typed_relation_summary):,}"
        )

    expected_typed = pd.read_csv(
        EXPECTED_TYPED_RELATION_PATH
    )

    expected_typed = (
        expected_typed[
            [
                "src_type",
                "relation",
                "dst_type",
                "typed_relation_key",
                "directed_edge_records",
            ]
        ]
        .sort_values(
            [
                "src_type",
                "relation",
                "dst_type",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    comparison = (
        expected_typed.merge(
            typed_relation_summary,
            on=[
                "src_type",
                "relation",
                "dst_type",
                "typed_relation_key",
            ],
            how="outer",
            suffixes=(
                "_expected",
                "_observed",
            ),
            indicator=True,
        )
    )

    comparison[
        "count_matches"
    ] = (
        comparison[
            "directed_edge_records_expected"
        ]
        .eq(
            comparison[
                "directed_edge_records_observed"
            ]
        )
    )

    typed_projection_pass = bool(
        (
            comparison[
                "_merge"
            ]
            == "both"
        )
        .all()
        and
        comparison[
            "count_matches"
        ]
        .all()
    )

    if not typed_projection_pass:

        raise ValueError(
            "Typed relation projection does not match "
            "the frozen Phase-3.3.1 specification."
        )

    # =========================================================================
    # 12. SHARED_FOUNDER reciprocity
    # =========================================================================

    shared_materialized = (
        edges[
            edges[
                "relation"
            ]
            ==
            "shared_founder"
        ]
        .copy()
    )

    shared_signature_set = {
        (
            row.src_node_id,
            row.dst_node_id,
            int(
                row.support_count
            ),
        )
        for row in shared_materialized.itertuples(
            index=False
        )
    }

    shared_missing_reverse = 0

    for (
        src_node,
        dst_node,
        support_count,
    ) in shared_signature_set:

        expected_reverse = (
            dst_node,
            src_node,
            support_count,
        )

        if (
            expected_reverse
            not in shared_signature_set
        ):

            shared_missing_reverse += 1

    # =========================================================================
    # 13. ACQUIRED <-> ACQUIRED_BY inverse consistency
    # =========================================================================

    acquired_materialized = (
        edges[
            edges[
                "relation"
            ]
            ==
            "acquired"
        ]
        .copy()
    )

    acquired_by_materialized = (
        edges[
            edges[
                "relation"
            ]
            ==
            "acquired_by"
        ]
        .copy()
    )

    def date_key(
        value,
    ):

        if pd.isna(
            value
        ):

            return None

        return pd.Timestamp(
            value
        ).normalize()

    acquired_by_signature_set = {
        (
            row.src_node_id,
            row.dst_node_id,
            int(
                row.support_count
            ),
            date_key(
                row.first_observed_on
            ),
            date_key(
                row.last_observed_on
            ),
        )
        for row in acquired_by_materialized.itertuples(
            index=False
        )
    }

    acquisition_missing_inverse = 0

    for row in acquired_materialized.itertuples(
        index=False
    ):

        expected_inverse = (
            row.dst_node_id,
            row.src_node_id,
            int(
                row.support_count
            ),
            date_key(
                row.first_observed_on
            ),
            date_key(
                row.last_observed_on
            ),
        )

        if (
            expected_inverse
            not in acquired_by_signature_set
        ):

            acquisition_missing_inverse += 1

    reciprocity_summary = pd.DataFrame(
        [
            {
                "integrity_rule":
                    "SHARED_FOUNDER reverse edge exists "
                    "with identical support_count",

                "tested_forward_records":
                    len(
                        shared_materialized
                    ),

                "missing_or_mismatched_counterparts":
                    shared_missing_reverse,

                "status":
                    (
                        "PASS"
                        if shared_missing_reverse
                        == 0
                        else "FAIL"
                    ),
            },

            {
                "integrity_rule":
                    "ACQUIRED has exact ACQUIRED_BY inverse",

                "tested_forward_records":
                    len(
                        acquired_materialized
                    ),

                "missing_or_mismatched_counterparts":
                    acquisition_missing_inverse,

                "status":
                    (
                        "PASS"
                        if acquisition_missing_inverse
                        == 0
                        else "FAIL"
                    ),
            },
        ]
    )

    if (
        shared_missing_reverse
        != 0
        or acquisition_missing_inverse
        != 0
    ):

        raise ValueError(
            "Structural reciprocity/inverse integrity failed."
        )

    # =========================================================================
    # 14. T60 structural leakage re-audit
    # =========================================================================

    holdout = pd.read_parquet(
        HOLDOUT_PATH,
        columns=[
            "investor_id",
            "startup_id",
        ],
    )

    holdout[
        "investor_id"
    ] = clean_string(
        holdout[
            "investor_id"
        ]
    )

    holdout[
        "startup_id"
    ] = clean_string(
        holdout[
            "startup_id"
        ]
    )

    holdout_underlying_pairs = {
        ordered_underlying_pair(
            investor_id,
            startup_id,
        )
        for investor_id, startup_id
        in zip(
            holdout[
                "investor_id"
            ],
            holdout[
                "startup_id"
            ],
        )
    }

    edge_underlying_pairs = [
        ordered_underlying_pair(
            source_id,
            destination_id,
        )
        for source_id, destination_id
        in zip(
            edges[
                "underlying_src_id"
            ],
            edges[
                "underlying_dst_id"
            ],
        )
    ]

    leakage_flags = [
        pair
        in holdout_underlying_pairs
        for pair in edge_underlying_pairs
    ]

    leakage_edge_records = int(
        sum(
            leakage_flags
        )
    )

    leaking_unique_underlying_pairs = len(
        {
            pair
            for pair, is_leak
            in zip(
                edge_underlying_pairs,
                leakage_flags,
            )
            if is_leak
        }
    )

    leakage_by_relation = (
        edges.assign(
            is_holdout_pair=(
                leakage_flags
            )
        )
        .groupby(
            "relation",
            observed=True,
        )
        .agg(
            edge_records=(
                "edge_id",
                "size",
            ),

            leaking_edge_records=(
                "is_holdout_pair",
                "sum",
            ),
        )
        .reset_index()
    )

    holdout_leakage_summary = pd.DataFrame(
        [
            {
                "metric":
                    "T60_unique_unordered_underlying_holdout_pairs",

                "value":
                    len(
                        holdout_underlying_pairs
                    ),
            },

            {
                "metric":
                    "structural_edge_records_matching_holdout_pair",

                "value":
                    leakage_edge_records,
            },

            {
                "metric":
                    "unique_leaking_underlying_pairs",

                "value":
                    leaking_unique_underlying_pairs,
            },
        ]
    )

    if leakage_edge_records != 0:

        raise ValueError(
            "T60 held-out structural leakage detected."
        )

    # =========================================================================
    # 15. Node structural connectivity
    # =========================================================================

    out_degree = (
        edges[
            "src_node_id"
        ]
        .value_counts()
    )

    in_degree = (
        edges[
            "dst_node_id"
        ]
        .value_counts()
    )

    degree_table = (
        nodes[
            [
                "node_id",
                "node_type",
            ]
        ]
        .copy()
    )

    degree_table[
        "out_degree"
    ] = (
        degree_table[
            "node_id"
        ]
        .map(
            out_degree
        )
        .fillna(
            0
        )
        .astype(
            "int64"
        )
    )

    degree_table[
        "in_degree"
    ] = (
        degree_table[
            "node_id"
        ]
        .map(
            in_degree
        )
        .fillna(
            0
        )
        .astype(
            "int64"
        )
    )

    degree_table[
        "total_directed_degree"
    ] = (
        degree_table[
            "out_degree"
        ]
        +
        degree_table[
            "in_degree"
        ]
    )

    degree_table[
        "has_structural_neighbor"
    ] = (
        degree_table[
            "total_directed_degree"
        ]
        > 0
    )

    connectivity_rows = []

    for node_type, group in (
        degree_table.groupby(
            "node_type",
            observed=True,
        )
    ):

        node_count = len(
            group
        )

        connected = int(
            group[
                "has_structural_neighbor"
            ]
            .sum()
        )

        isolated = (
            node_count
            -
            connected
        )

        connectivity_rows.append(
            {
                "node_type":
                    node_type,

                "node_count":
                    node_count,

                "nodes_with_structural_neighbor":
                    connected,

                "isolated_nodes":
                    isolated,

                "connected_pct":
                    pct(
                        connected,
                        node_count,
                    ),

                "isolated_pct":
                    pct(
                        isolated,
                        node_count,
                    ),
            }
        )

    total_connected = int(
        degree_table[
            "has_structural_neighbor"
        ]
        .sum()
    )

    connectivity_rows.append(
        {
            "node_type":
                "ALL",

            "node_count":
                len(
                    degree_table
                ),

            "nodes_with_structural_neighbor":
                total_connected,

            "isolated_nodes":
                (
                    len(
                        degree_table
                    )
                    -
                    total_connected
                ),

            "connected_pct":
                pct(
                    total_connected,
                    len(
                        degree_table
                    ),
                ),

            "isolated_pct":
                pct(
                    (
                        len(
                            degree_table
                        )
                        -
                        total_connected
                    ),
                    len(
                        degree_table
                    ),
                ),
        }
    )

    connectivity_summary = pd.DataFrame(
        connectivity_rows
    )

    # =========================================================================
    # 16. Degree summary
    # =========================================================================

    degree_summary_rows = []

    for node_type, group in (
        degree_table.groupby(
            "node_type",
            observed=True,
        )
    ):

        for degree_name in [
            "out_degree",
            "in_degree",
            "total_directed_degree",
        ]:

            values = (
                group[
                    degree_name
                ]
                .astype(
                    float
                )
            )

            degree_summary_rows.append(
                {
                    "node_type":
                        node_type,

                    "degree_measure":
                        degree_name,

                    "mean":
                        values.mean(),

                    "median":
                        values.median(),

                    "p95":
                        values.quantile(
                            0.95
                        ),

                    "p99":
                        values.quantile(
                            0.99
                        ),

                    "max":
                        values.max(),
                }
            )

    degree_summary = pd.DataFrame(
        degree_summary_rows
    )

    # =========================================================================
    # 17. Final integrity summary
    # =========================================================================

    integrity_summary = pd.DataFrame(
        [
            {
                "metric":
                    "total_structural_edge_records",

                "value":
                    len(
                        edges
                    ),

                "status":
                    "PASS",
            },

            {
                "metric":
                    "base_semantic_relations",

                "value":
                    edges[
                        "relation"
                    ]
                    .nunique(),

                "status":
                    (
                        "PASS"
                        if edges[
                            "relation"
                        ]
                        .nunique()
                        == 3
                        else "FAIL"
                    ),
            },

            {
                "metric":
                    "typed_relation_channels",

                "value":
                    edges[
                        "typed_relation_key"
                    ]
                    .nunique(),

                "status":
                    (
                        "PASS"
                        if edges[
                            "typed_relation_key"
                        ]
                        .nunique()
                        ==
                        EXPECTED_TYPED_RELATIONS
                        else "FAIL"
                    ),
            },

            {
                "metric":
                    "duplicate_edge_ids",

                "value":
                    duplicate_edge_ids,

                "status":
                    (
                        "PASS"
                        if duplicate_edge_ids
                        == 0
                        else "FAIL"
                    ),
            },

            {
                "metric":
                    "duplicate_src_relation_dst_triples",

                "value":
                    duplicate_edge_triples,

                "status":
                    (
                        "PASS"
                        if duplicate_edge_triples
                        == 0
                        else "FAIL"
                    ),
            },

            {
                "metric":
                    "graph_node_self_loops",

                "value":
                    graph_self_loops,

                "status":
                    (
                        "PASS"
                        if graph_self_loops
                        == 0
                        else "FAIL"
                    ),
            },

            {
                "metric":
                    "same_underlying_entity_relations",

                "value":
                    same_underlying_relations,

                "status":
                    (
                        "PASS"
                        if same_underlying_relations
                        == 0
                        else "FAIL"
                    ),
            },

            {
                "metric":
                    "missing_edge_endpoints",

                "value":
                    (
                        missing_src_nodes
                        +
                        missing_dst_nodes
                    ),

                "status":
                    (
                        "PASS"
                        if (
                            missing_src_nodes
                            +
                            missing_dst_nodes
                        )
                        == 0
                        else "FAIL"
                    ),
            },

            {
                "metric":
                    "endpoint_identity_mismatches",

                "value":
                    (
                        src_type_mismatches
                        +
                        dst_type_mismatches
                        +
                        src_underlying_mismatches
                        +
                        dst_underlying_mismatches
                    ),

                "status":
                    (
                        "PASS"
                        if (
                            src_type_mismatches
                            +
                            dst_type_mismatches
                            +
                            src_underlying_mismatches
                            +
                            dst_underlying_mismatches
                        )
                        == 0
                        else "FAIL"
                    ),
            },

            {
                "metric":
                    "shared_founder_missing_reverse_records",

                "value":
                    shared_missing_reverse,

                "status":
                    (
                        "PASS"
                        if shared_missing_reverse
                        == 0
                        else "FAIL"
                    ),
            },

            {
                "metric":
                    "acquired_missing_inverse_records",

                "value":
                    acquisition_missing_inverse,

                "status":
                    (
                        "PASS"
                        if acquisition_missing_inverse
                        == 0
                        else "FAIL"
                    ),
            },

            {
                "metric":
                    "T60_holdout_leaking_edge_records",

                "value":
                    leakage_edge_records,

                "status":
                    (
                        "PASS"
                        if leakage_edge_records
                        == 0
                        else "FAIL"
                    ),
            },
        ]
    )

    if (
        integrity_summary[
            "status"
        ]
        .ne(
            "PASS"
        )
        .any()
    ):

        raise ValueError(
            "One or more final structural graph "
            "integrity checks failed."
        )

    # =========================================================================
    # 18. Sort materialized edges deterministically
    # =========================================================================

    edges = (
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
    )

    # =========================================================================
    # 19. Save materialized graph and audits
    # =========================================================================

    GRAPH_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    edges.to_parquet(
        EDGE_OUTPUT,
        index=False,
    )

    integrity_summary.to_csv(
        INTEGRITY_SUMMARY_OUTPUT,
        index=False,
    )

    relation_summary.to_csv(
        RELATION_SUMMARY_OUTPUT,
        index=False,
    )

    typed_relation_summary.to_csv(
        TYPED_RELATION_SUMMARY_OUTPUT,
        index=False,
    )

    endpoint_integrity.to_csv(
        ENDPOINT_INTEGRITY_OUTPUT,
        index=False,
    )

    reciprocity_summary.to_csv(
        RECIPROCITY_OUTPUT,
        index=False,
    )

    holdout_leakage_summary.to_csv(
        HOLDOUT_LEAKAGE_OUTPUT,
        index=False,
    )

    connectivity_summary.to_csv(
        CONNECTIVITY_OUTPUT,
        index=False,
    )

    degree_summary.to_csv(
        DEGREE_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 20. Figures
    # =========================================================================

    relation_plot = (
        relation_summary
        .sort_values(
            "directed_edge_records",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        relation_plot[
            "relation"
        ],
        relation_plot[
            "directed_edge_records"
        ],
    )

    ax.set_xlabel(
        "Directed edge records"
    )

    ax.set_title(
        "Materialized Structural Relations"
    )

    fig.tight_layout()

    fig.savefig(
        RELATION_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # -------------------------------------------------------------------------

    typed_plot = (
        typed_relation_summary
        .sort_values(
            "directed_edge_records",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.barh(
        typed_plot[
            "typed_relation_key"
        ],
        typed_plot[
            "directed_edge_records"
        ],
    )

    ax.set_xlabel(
        "Directed edge records"
    )

    ax.set_title(
        "Materialized Typed R-GCN Relation Channels"
    )

    fig.tight_layout()

    fig.savefig(
        TYPED_RELATION_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # -------------------------------------------------------------------------

    connectivity_plot = (
        connectivity_summary[
            connectivity_summary[
                "node_type"
            ]
            != "ALL"
        ]
        .copy()
    )

    x = np.arange(
        len(
            connectivity_plot
        )
    )

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        x - width / 2,
        connectivity_plot[
            "nodes_with_structural_neighbor"
        ],
        width,
        label="Connected",
    )

    ax.bar(
        x + width / 2,
        connectivity_plot[
            "isolated_nodes"
        ],
        width,
        label="Isolated",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        connectivity_plot[
            "node_type"
        ]
    )

    ax.set_ylabel(
        "Graph nodes"
    )

    ax.set_title(
        "Structural Graph Connectivity by Node Type"
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        CONNECTIVITY_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # -------------------------------------------------------------------------

    positive_degree = (
        degree_table[
            degree_table[
                "total_directed_degree"
            ]
            > 0
        ][
            "total_directed_degree"
        ]
    )

    if len(
        positive_degree
    ) > 0:

        degree_counts = (
            positive_degree
            .value_counts()
            .sort_index()
        )

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.scatter(
            degree_counts.index,
            degree_counts.values,
        )

        if (
            degree_counts.index.max()
            > 20
        ):

            ax.set_xscale(
                "log"
            )

        if (
            degree_counts.max()
            /
            max(
                1,
                degree_counts.min(),
            )
            > 100
        ):

            ax.set_yscale(
                "log"
            )

        ax.set_xlabel(
            "Total directed degree"
        )

        ax.set_ylabel(
            "Nodes"
        )

        ax.set_title(
            "Structural Degree Distribution "
            "(Non-Isolated Nodes)"
        )

        fig.tight_layout()

        fig.savefig(
            DEGREE_FIGURE,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    # =========================================================================
    # 21. Terminal report
    # =========================================================================

    separator("-")

    print(
        "STRUCTURAL EDGE INTEGRITY SUMMARY"
    )

    separator("-")

    print(
        integrity_summary.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "BASE RELATION SUMMARY"
    )

    separator("-")

    print(
        relation_summary.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "TYPED RELATION SUMMARY"
    )

    separator("-")

    print(
        typed_relation_summary.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "ENDPOINT IDENTITY INTEGRITY"
    )

    separator("-")

    print(
        endpoint_integrity.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "SYMMETRY / INVERSE INTEGRITY"
    )

    separator("-")

    print(
        reciprocity_summary.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "T60 STRUCTURAL LEAKAGE AUDIT"
    )

    separator("-")

    print(
        holdout_leakage_summary.to_string(
            index=False
        )
    )

    print(
        "\nBy relation:"
    )

    print(
        leakage_by_relation.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "STRUCTURAL NODE CONNECTIVITY"
    )

    separator("-")

    print(
        connectivity_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "DEGREE SUMMARY"
    )

    separator("-")

    print(
        degree_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "EDGE TABLE MATERIALIZATION"
    )

    separator("-")

    print(
        f"Edge table: "
        f"{EDGE_OUTPUT}"
    )

    print(
        f"Rows:       "
        f"{len(edges):,}"
    )

    print(
        f"Columns:    "
        f"{len(edges.columns):,}"
    )

    separator()

    print(
        "PHASE 3.3.3 COMPLETE"
    )

    separator()

    print(
        f"""
Materialized structural edge table:

{EDGE_OUTPUT}

Audits written to:

{INTEGRITY_SUMMARY_OUTPUT}
{RELATION_SUMMARY_OUTPUT}
{TYPED_RELATION_SUMMARY_OUTPUT}
{ENDPOINT_INTEGRITY_OUTPUT}
{RECIPROCITY_OUTPUT}
{HOLDOUT_LEAKAGE_OUTPUT}
{CONNECTIVITY_OUTPUT}
{DEGREE_OUTPUT}

Figures written to:

{RELATION_FIGURE}
{TYPED_RELATION_FIGURE}
{CONNECTIVITY_FIGURE}
{DEGREE_FIGURE}


FROZEN STRUCTURAL GRAPH EXPECTATION

Nodes:
    477,564

Directed structural edge records:
    158,818

Relations:
    shared_founder     94,818
    acquired           32,000
    acquired_by        32,000

Typed relation channels:
    12

Historical investment edges:
    0

T60 structural leakage:
    MUST BE 0


IMPORTANT

1. nodes.parquet and edges.parquet together define the first complete
   materialized structural graph.

2. Investment interactions remain entirely separate in the frozen
   Phase-2 temporal interaction artifact.

3. Isolated nodes are retained. Structural degree is diagnostic and
   does not alter the canonical recommendation universe.

4. No integer node indices or R-GCN relation IDs have yet been assigned.

5. Founder edges retain temporal provenance:

       current_snapshot_unversioned

6. Acquisition edges retain temporal provenance:

       timestamped_pre_T60


IF ALL CHECKS PASS:

    Phase 3.3 — COMPLETE

NEXT:

    Phase 3.4 — Model-Ready Graph Encoding & Final Graph Integrity
"""
    )


if __name__ == "__main__":
    main()