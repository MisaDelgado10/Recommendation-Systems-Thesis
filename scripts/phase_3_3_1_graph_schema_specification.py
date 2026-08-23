from pathlib import Path
from collections import Counter
import json

import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PHASE 3.3.1 — GRAPH-SCHEMA SPECIFICATION & COUNT VERIFICATION
# =============================================================================
#
# PURPOSE
# -------
#
# Freeze the graph representation BEFORE graph materialization.
#
# This script:
#
#   - verifies canonical node counts;
#   - verifies audited structural relation candidate counts;
#   - projects semantic relations into typed source/relation/target channels;
#   - freezes inverse-relation behavior;
#   - specifies node and edge table schemas;
#   - defines core and ablation graph variants;
#   - writes a versionable JSON graph specification.
#
# NO graph node table is materialized.
# NO graph edge table is materialized.
# =============================================================================


INTERACTIONS_PATH = Path(
    "data/processed/interactions.parquet"
)

SHARED_FOUNDER_PATH = Path(
    "data/experimental/phase_3/audits/"
    "shared_founder_strict_role_pair_candidates.parquet"
)

ACQUISITION_PATH = Path(
    "data/experimental/phase_3/audits/"
    "acquisition_strict_underlying_relation_candidates.parquet"
)

CLOSURE_PATH = Path(
    "data/experimental/phase_3/audits/"
    "phase_3_2_closure_summary.csv"
)


AUDIT_DIR = Path(
    "data/experimental/phase_3/audits"
)

FIGURE_DIR = Path(
    "data/experimental/phase_3/figures"
)

CONFIG_DIR = Path(
    "configs"
)


SELECTION_SUMMARY_OUTPUT = (
    AUDIT_DIR
    / "phase_3_3_graph_schema_selection_summary.csv"
)

TYPED_RELATION_OUTPUT = (
    AUDIT_DIR
    / "phase_3_3_typed_relation_projection.csv"
)

VARIANT_OUTPUT = (
    AUDIT_DIR
    / "phase_3_3_graph_variant_specification.csv"
)

NODE_SCHEMA_OUTPUT = (
    AUDIT_DIR
    / "phase_3_3_node_table_schema.csv"
)

EDGE_SCHEMA_OUTPUT = (
    AUDIT_DIR
    / "phase_3_3_edge_table_schema.csv"
)

CONFIG_OUTPUT = (
    CONFIG_DIR
    / "phase_3_itrs_graph_schema.json"
)

RELATION_FIGURE = (
    FIGURE_DIR
    / "phase_3_3_selected_relation_edge_counts.png"
)


EXPECTED_INVESTORS = 165_975
EXPECTED_STARTUPS = 311_589
EXPECTED_SHARED_PAIRS = 47_409
EXPECTED_ACQUISITION_FORWARD_ROLE_EDGES = 32_000


def separator(char="=", width=120):
    print(char * width)


def metric_value(dataframe, metric):

    row = dataframe[
        dataframe["metric"]
        == metric
    ]

    if len(row) != 1:

        raise ValueError(
            f"Expected exactly one row for metric "
            f"'{metric}', found {len(row)}."
        )

    return int(
        row.iloc[0]["value"]
    )


def role_memberships(label):

    if label == "investor":
        return ["investor"]

    if label == "startup":
        return ["startup"]

    if label == "investor+startup":
        return [
            "investor",
            "startup",
        ]

    raise ValueError(
        f"Unknown canonical role membership: {label}"
    )


def add_count(
    counter,
    src_type,
    relation,
    dst_type,
    count=1,
):

    counter[
        (
            src_type,
            relation,
            dst_type,
        )
    ] += count


def main():

    separator()

    print(
        "PHASE 3.3.1 — "
        "GRAPH-SCHEMA SPECIFICATION & COUNT VERIFICATION"
    )

    separator()

    # =========================================================================
    # 1. Canonical node universe
    # =========================================================================

    interactions = pd.read_parquet(
        INTERACTIONS_PATH,
        columns=[
            "investor_id",
            "startup_id",
        ],
    )

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

    if len(investor_ids) != EXPECTED_INVESTORS:

        raise ValueError(
            f"Investor count changed: "
            f"{len(investor_ids):,}"
        )

    if len(startup_ids) != EXPECTED_STARTUPS:

        raise ValueError(
            f"Startup count changed: "
            f"{len(startup_ids):,}"
        )

    graph_node_count = (
        len(investor_ids)
        +
        len(startup_ids)
    )

    # =========================================================================
    # 2. Load audited relation candidates
    # =========================================================================

    shared = pd.read_parquet(
        SHARED_FOUNDER_PATH
    )

    acquisitions = pd.read_parquet(
        ACQUISITION_PATH
    )

    closure = pd.read_csv(
        CLOSURE_PATH
    )

    if len(shared) != EXPECTED_SHARED_PAIRS:

        raise ValueError(
            f"Expected {EXPECTED_SHARED_PAIRS:,} "
            f"SHARED_FOUNDER pairs, "
            f"found {len(shared):,}."
        )

    if metric_value(
        closure,
        "shared_founder_unique_symmetric_candidates",
    ) != len(shared):

        raise ValueError(
            "Phase-3.2 closure count does not match "
            "SHARED_FOUNDER candidate artifact."
        )

    expected_acquisition_role_edges = (
        metric_value(
            closure,
            "acquisition_potential_role_specific_directed_relations",
        )
    )

    if (
        expected_acquisition_role_edges
        !=
        EXPECTED_ACQUISITION_FORWARD_ROLE_EDGES
    ):

        raise ValueError(
            "Unexpected acquisition role-projection count."
        )

    # =========================================================================
    # 3. Typed relation projection — SHARED_FOUNDER
    #
    # Symmetric pair -> two directed message-passing records.
    # Same semantic relation label in both directions.
    # =========================================================================

    typed_counts = Counter()

    for row in shared.itertuples(
        index=False
    ):

        src_role = row.role_a
        dst_role = row.role_b

        add_count(
            typed_counts,
            src_role,
            "shared_founder",
            dst_role,
        )

        add_count(
            typed_counts,
            dst_role,
            "shared_founder",
            src_role,
        )

    shared_directed_edges = (
        len(shared)
        * 2
    )

    # =========================================================================
    # 4. Typed relation projection — ACQUIRED / ACQUIRED_BY
    # =========================================================================

    acquisition_forward_count = 0
    acquisition_inverse_count = 0

    required_acquisition_columns = {
        "acquirer_resolved_id",
        "acquiree_resolved_id",
        "acquirer_role_membership",
        "acquiree_role_membership",
    }

    missing = (
        required_acquisition_columns
        -
        set(acquisitions.columns)
    )

    if missing:

        raise ValueError(
            "Acquisition candidate artifact missing columns: "
            f"{sorted(missing)}"
        )

    for row in acquisitions.itertuples(
        index=False
    ):

        source_roles = role_memberships(
            row.acquirer_role_membership
        )

        target_roles = role_memberships(
            row.acquiree_role_membership
        )

        for source_role in source_roles:

            for target_role in target_roles:

                # -------------------------------------------------------------
                # Forward semantic relation
                # -------------------------------------------------------------

                add_count(
                    typed_counts,
                    source_role,
                    "acquired",
                    target_role,
                )

                acquisition_forward_count += 1

                # -------------------------------------------------------------
                # Explicit semantic inverse
                # -------------------------------------------------------------

                add_count(
                    typed_counts,
                    target_role,
                    "acquired_by",
                    source_role,
                )

                acquisition_inverse_count += 1

    if (
        acquisition_forward_count
        !=
        expected_acquisition_role_edges
    ):

        raise ValueError(
            "Acquisition forward role projection changed. "
            f"Expected {expected_acquisition_role_edges:,}, "
            f"found {acquisition_forward_count:,}."
        )

    if (
        acquisition_inverse_count
        !=
        acquisition_forward_count
    ):

        raise ValueError(
            "ACQUIRED_BY inverse projection does not "
            "match ACQUIRED forward projection."
        )

    acquisition_total_edges = (
        acquisition_forward_count
        +
        acquisition_inverse_count
    )

    total_directed_edges = (
        shared_directed_edges
        +
        acquisition_total_edges
    )

    # =========================================================================
    # 5. Typed relation table
    # =========================================================================

    typed_rows = []

    for (
        src_type,
        relation,
        dst_type,
    ), count in sorted(
        typed_counts.items()
    ):

        typed_rows.append(
            {
                "src_type":
                    src_type,

                "relation":
                    relation,

                "dst_type":
                    dst_type,

                "typed_relation_key":
                    (
                        f"{src_type}"
                        f"|{relation}"
                        f"|{dst_type}"
                    ),

                "directed_edge_records":
                    count,
            }
        )

    typed_relations = pd.DataFrame(
        typed_rows
    )

    typed_relation_count = len(
        typed_relations
    )

    # =========================================================================
    # 6. Base graph variants
    # =========================================================================

    variants = pd.DataFrame(
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

                "directed_edge_records":
                    total_directed_edges,

                "purpose":
                    (
                        "Primary ITRS-Crunchbase "
                        "structural graph."
                    ),
            },

            {
                "variant":
                    "founder_only_ablation",

                "relations":
                    "shared_founder",

                "directed_edge_records":
                    shared_directed_edges,

                "purpose":
                    (
                        "Measure contribution of "
                        "Human-intermediary relation."
                    ),
            },

            {
                "variant":
                    "acquisition_only_ablation",

                "relations":
                    "acquired | acquired_by",

                "directed_edge_records":
                    acquisition_total_edges,

                "purpose":
                    (
                        "Measure contribution of "
                        "Crunchbase acquisition relation."
                    ),
            },
        ]
    )

    # =========================================================================
    # 7. Node-table specification
    # =========================================================================

    node_schema = pd.DataFrame(
        [
            {
                "column":
                    "node_id",

                "dtype":
                    "string",

                "required":
                    True,

                "description":
                    (
                        "Role-namespaced graph identity: "
                        "investor::<uuid> or startup::<uuid>."
                    ),
            },

            {
                "column":
                    "node_type",

                "dtype":
                    "string",

                "required":
                    True,

                "description":
                    "investor or startup.",
            },

            {
                "column":
                    "raw_entity_id",

                "dtype":
                    "string",

                "required":
                    True,

                "description":
                    "Underlying Crunchbase UUID.",
            },

            {
                "column":
                    "display_name",

                "dtype":
                    "string",

                "required":
                    True,

                "description":
                    "Human-readable canonical entity name.",
            },

            {
                "column":
                    "source_registry",

                "dtype":
                    "string",

                "required":
                    True,

                "description":
                    (
                        "Authoritative source registry "
                        "used for role identity."
                    ),
            },

            {
                "column":
                    "underlying_entity_has_dual_role",

                "dtype":
                    "bool",

                "required":
                    True,

                "description":
                    (
                        "Whether raw_entity_id occurs in both "
                        "canonical Investor and Startup roles."
                    ),
            },
        ]
    )

    # =========================================================================
    # 8. Edge-table specification
    # =========================================================================

    edge_schema = pd.DataFrame(
        [
            {
                "column":
                    "edge_id",

                "dtype":
                    "string",

                "required":
                    True,

                "description":
                    "Deterministic unique edge-record ID.",
            },

            {
                "column":
                    "src_node_id",

                "dtype":
                    "string",

                "required":
                    True,

                "description":
                    "Role-namespaced source node.",
            },

            {
                "column":
                    "dst_node_id",

                "dtype":
                    "string",

                "required":
                    True,

                "description":
                    "Role-namespaced destination node.",
            },

            {
                "column":
                    "src_type",

                "dtype":
                    "string",

                "required":
                    True,

                "description":
                    "investor or startup.",
            },

            {
                "column":
                    "dst_type",

                "dtype":
                    "string",

                "required":
                    True,

                "description":
                    "investor or startup.",
            },

            {
                "column":
                    "relation",

                "dtype":
                    "string",

                "required":
                    True,

                "description":
                    (
                        "Base semantic relation: "
                        "shared_founder, acquired, acquired_by."
                    ),
            },

            {
                "column":
                    "typed_relation_key",

                "dtype":
                    "string",

                "required":
                    True,

                "description":
                    "src_type|relation|dst_type.",
            },

            {
                "column":
                    "underlying_src_id",

                "dtype":
                    "string",

                "required":
                    True,

                "description":
                    "Underlying Crunchbase UUID of source.",
            },

            {
                "column":
                    "underlying_dst_id",

                "dtype":
                    "string",

                "required":
                    True,

                "description":
                    "Underlying Crunchbase UUID of destination.",
            },

            {
                "column":
                    "provenance_source",

                "dtype":
                    "string",

                "required":
                    True,

                "description":
                    (
                        "Source used to generate relation, "
                        "e.g. founder bridge or acquisitions."
                    ),
            },

            {
                "column":
                    "temporal_provenance",

                "dtype":
                    "string",

                "required":
                    True,

                "description":
                    (
                        "timestamped_pre_T60 or "
                        "current_snapshot_unversioned."
                    ),
            },

            {
                "column":
                    "first_observed_on",

                "dtype":
                    "date/null",

                "required":
                    False,

                "description":
                    (
                        "First observed relationship date "
                        "where available."
                    ),
            },

            {
                "column":
                    "last_observed_on",

                "dtype":
                    "date/null",

                "required":
                    False,

                "description":
                    (
                        "Last observed relationship date "
                        "where available."
                    ),
            },

            {
                "column":
                    "support_count",

                "dtype":
                    "integer",

                "required":
                    True,

                "description":
                    (
                        "Shared-founder multiplicity or "
                        "acquisition-event multiplicity."
                    ),
            },

            {
                "column":
                    "is_inverse_edge",

                "dtype":
                    "bool",

                "required":
                    True,

                "description":
                    (
                        "True only for explicit ACQUIRED_BY "
                        "inverse records."
                    ),
            },
        ]
    )

    # =========================================================================
    # 9. Selection summary
    # =========================================================================

    selection_summary = pd.DataFrame(
        [
            {
                "metric":
                    "graph_node_types",

                "value":
                    2,
            },

            {
                "metric":
                    "canonical_investor_role_nodes",

                "value":
                    len(investor_ids),
            },

            {
                "metric":
                    "canonical_startup_role_nodes",

                "value":
                    len(startup_ids),
            },

            {
                "metric":
                    "total_role_nodes",

                "value":
                    graph_node_count,
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
                    typed_relation_count,
            },

            {
                "metric":
                    "shared_founder_directed_edges",

                "value":
                    shared_directed_edges,
            },

            {
                "metric":
                    "acquired_forward_edges",

                "value":
                    acquisition_forward_count,
            },

            {
                "metric":
                    "acquired_by_inverse_edges",

                "value":
                    acquisition_inverse_count,
            },

            {
                "metric":
                    "core_total_directed_structural_edges",

                "value":
                    total_directed_edges,
            },

            {
                "metric":
                    "explicit_self_loop_relation",

                "value":
                    0,
            },

            {
                "metric":
                    "investment_event_structural_relation",

                "value":
                    0,
            },
        ]
    )

    # =========================================================================
    # 10. Frozen config
    # =========================================================================

    config = {
        "phase":
            "3.3.1",

        "graph_name":
            "itrs_crunchbase_structural_graph",

        "node_types": [
            "investor",
            "startup",
        ],

        "node_id_format": {
            "investor":
                "investor::<crunchbase_uuid>",

            "startup":
                "startup::<crunchbase_uuid>",
        },

        "base_relations": {
            "shared_founder": {
                "semantics":
                    "symmetric",

                "storage":
                    "materialize_both_directions_same_relation",

                "temporal_provenance":
                    "current_snapshot_unversioned",
            },

            "acquired": {
                "semantics":
                    "directed",

                "storage":
                    "forward_relation",

                "temporal_policy":
                    "announced_on < 2026-01-01",
            },

            "acquired_by": {
                "semantics":
                    "inverse_of_acquired",

                "storage":
                    "explicit_inverse_relation",
            },
        },

        "typed_relation_key":
            "src_type|relation|dst_type",

        "holdout_mask_policy":
            (
                "Remove any direct structural relation "
                "whose unordered underlying entity pair "
                "matches a T60 held-out Investor-Startup pair."
            ),

        "investment_events_as_structural_edges":
            False,

        "explicit_self_loop_relation":
            False,

        "node_filtering":
            "none",

        "graph_snapshot":
            "static",

        "variants": {
            row.variant: {
                "relations":
                    row.relations,

                "expected_directed_edges":
                    int(
                        row.directed_edge_records
                    ),
            }
            for row in variants.itertuples(
                index=False
            )
        },

        "expected_counts": {
            "investor_nodes":
                len(investor_ids),

            "startup_nodes":
                len(startup_ids),

            "total_role_nodes":
                graph_node_count,

            "shared_founder_directed_edges":
                shared_directed_edges,

            "acquired_forward_edges":
                acquisition_forward_count,

            "acquired_by_inverse_edges":
                acquisition_inverse_count,

            "core_directed_edges":
                total_directed_edges,

            "typed_relation_channels":
                typed_relation_count,
        },
    }

    # =========================================================================
    # 11. Save
    # =========================================================================

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONFIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    selection_summary.to_csv(
        SELECTION_SUMMARY_OUTPUT,
        index=False,
    )

    typed_relations.to_csv(
        TYPED_RELATION_OUTPUT,
        index=False,
    )

    variants.to_csv(
        VARIANT_OUTPUT,
        index=False,
    )

    node_schema.to_csv(
        NODE_SCHEMA_OUTPUT,
        index=False,
    )

    edge_schema.to_csv(
        EDGE_SCHEMA_OUTPUT,
        index=False,
    )

    with open(
        CONFIG_OUTPUT,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    # =========================================================================
    # 12. Figure
    # =========================================================================

    relation_plot = pd.DataFrame(
        [
            {
                "relation":
                    "SHARED_FOUNDER",

                "directed_edge_records":
                    shared_directed_edges,
            },

            {
                "relation":
                    "ACQUIRED",

                "directed_edge_records":
                    acquisition_forward_count,
            },

            {
                "relation":
                    "ACQUIRED_BY",

                "directed_edge_records":
                    acquisition_inverse_count,
            },
        ]
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        relation_plot[
            "relation"
        ],
        relation_plot[
            "directed_edge_records"
        ],
    )

    ax.set_ylabel(
        "Expected directed edge records"
    )

    ax.set_title(
        "Selected ITRS-Crunchbase Structural Relations"
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

    # =========================================================================
    # 13. Terminal output
    # =========================================================================

    separator("-")

    print(
        "GRAPH-SCHEMA SELECTION SUMMARY"
    )

    separator("-")

    print(
        selection_summary.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "TYPED RELATION PROJECTION"
    )

    separator("-")

    print(
        typed_relations.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "GRAPH VARIANTS"
    )

    separator("-")

    print(
        variants.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "BASE RELATION VOCABULARY"
    )

    separator("-")

    print(
        "shared_founder  — symmetric, both directions"
    )

    print(
        "acquired        — directed forward relation"
    )

    print(
        "acquired_by     — explicit inverse relation"
    )

    separator("-")

    print(
        "GRAPH REPRESENTATION POLICY"
    )

    separator("-")

    print(
        "Node IDs:           investor::<uuid>, startup::<uuid>"
    )

    print(
        "Node filtering:     NONE"
    )

    print(
        "Graph snapshot:     STATIC"
    )

    print(
        "Investment edges:   EXCLUDED from structural graph"
    )

    print(
        "Self-loop relation: EXCLUDED from edge table"
    )

    print(
        "T60 pair masking:   GLOBAL underlying-pair policy"
    )

    separator()

    print(
        "PHASE 3.3.1 COMPLETE"
    )

    separator()

    print(
        f"""
Outputs written to:

{SELECTION_SUMMARY_OUTPUT}
{TYPED_RELATION_OUTPUT}
{VARIANT_OUTPUT}
{NODE_SCHEMA_OUTPUT}
{EDGE_SCHEMA_OUTPUT}

Versionable config written to:

{CONFIG_OUTPUT}

Figure written to:

{RELATION_FIGURE}


NO FINAL GRAPH HAS BEEN MATERIALIZED.

NEXT:

Phase 3.3.2 — Graph Node-Table Materialization & Identity Integrity Audit
"""
    )


if __name__ == "__main__":
    main()