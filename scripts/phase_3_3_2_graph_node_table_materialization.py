from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PHASE 3.3.2 — GRAPH NODE-TABLE MATERIALIZATION &
#               IDENTITY INTEGRITY AUDIT
# =============================================================================
#
# PURPOSE
# -------
#
# Materialize the frozen Investor / Startup role-node universe.
#
# Identity authorities:
#
#   Investor role -> Crunchbase investor registry
#   Startup role  -> Crunchbase companies registry
#
# Graph identity:
#
#   investor::<uuid>
#   startup::<uuid>
#
# IMPORTANT
# ---------
#
# Names are DISPLAY ATTRIBUTES ONLY.
#
# Graph identity is determined exclusively by:
#
#   role + Crunchbase UUID
#
# This script creates NO structural edges.
# =============================================================================


RAW_DIR = Path(
    "data/raw"
)

INTERACTIONS_PATH = Path(
    "data/processed/interactions.parquet"
)

CONFIG_PATH = Path(
    "configs/phase_3_itrs_graph_schema.json"
)


GRAPH_DIR = Path(
    "data/experimental/phase_3/graph"
)

AUDIT_DIR = Path(
    "data/experimental/phase_3/audits"
)

FIGURE_DIR = Path(
    "data/experimental/phase_3/figures"
)


NODE_OUTPUT = (
    GRAPH_DIR
    / "nodes.parquet"
)

IDENTITY_SUMMARY_OUTPUT = (
    AUDIT_DIR
    / "graph_node_identity_summary.csv"
)

NODE_TYPE_OUTPUT = (
    AUDIT_DIR
    / "graph_node_type_summary.csv"
)

REGISTRY_COVERAGE_OUTPUT = (
    AUDIT_DIR
    / "graph_node_registry_coverage.csv"
)

DUAL_ROLE_OUTPUT = (
    AUDIT_DIR
    / "graph_node_dual_role_identity_audit.csv"
)

COMPANY_NAME_CONFLICT_OUTPUT = (
    AUDIT_DIR
    / "graph_node_company_name_conflicts.csv"
)


NODE_TYPE_FIGURE = (
    FIGURE_DIR
    / "graph_node_type_counts.png"
)

IDENTITY_STRUCTURE_FIGURE = (
    FIGURE_DIR
    / "graph_node_identity_structure.png"
)


EXPECTED_INVESTORS = 165_975
EXPECTED_STARTUPS = 311_589
EXPECTED_ROLE_NODES = 477_564

EXPECTED_DUAL_ROLE_UNDERLYING_IDS = 9_633

EXPECTED_UNIQUE_UNDERLYING_IDS = (
    EXPECTED_INVESTORS
    +
    EXPECTED_STARTUPS
    -
    EXPECTED_DUAL_ROLE_UNDERLYING_IDS
)


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


def name_key(
    value,
):

    if pd.isna(
        value
    ):
        return None

    return (
        str(
            value
        )
        .strip()
        .casefold()
    )


def main():

    separator()

    print(
        "PHASE 3.3.2 — "
        "GRAPH NODE-TABLE MATERIALIZATION & "
        "IDENTITY INTEGRITY AUDIT"
    )

    separator()

    # =========================================================================
    # 1. Load canonical interaction universe
    # =========================================================================

    interactions = pd.read_parquet(
        INTERACTIONS_PATH,
        columns=[
            "investor_id",
            "investor_name",
            "startup_id",
            "startup_name",
        ],
    )

    for column in [
        "investor_id",
        "investor_name",
        "startup_id",
        "startup_name",
    ]:

        interactions[
            column
        ] = clean_string(
            interactions[
                column
            ]
        )

    investor_ids = set(
        interactions[
            "investor_id"
        ]
        .dropna()
        .unique()
    )

    startup_ids = set(
        interactions[
            "startup_id"
        ]
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
    # 2. Frozen-count verification
    # =========================================================================

    if (
        len(
            investor_ids
        )
        !=
        EXPECTED_INVESTORS
    ):

        raise ValueError(
            "Canonical investor count changed: "
            f"{len(investor_ids):,}"
        )

    if (
        len(
            startup_ids
        )
        !=
        EXPECTED_STARTUPS
    ):

        raise ValueError(
            "Canonical startup count changed: "
            f"{len(startup_ids):,}"
        )

    if (
        len(
            dual_role_ids
        )
        !=
        EXPECTED_DUAL_ROLE_UNDERLYING_IDS
    ):

        raise ValueError(
            "Dual-role underlying-ID count changed: "
            f"{len(dual_role_ids):,}"
        )

    if (
        len(
            unique_underlying_ids
        )
        !=
        EXPECTED_UNIQUE_UNDERLYING_IDS
    ):

        raise ValueError(
            "Underlying unique-ID universe changed: "
            f"{len(unique_underlying_ids):,}"
        )

    print(
        f"\nCanonical investors:            "
        f"{len(investor_ids):,}"
    )

    print(
        f"Canonical startups:             "
        f"{len(startup_ids):,}"
    )

    print(
        f"Dual-role underlying IDs:       "
        f"{len(dual_role_ids):,}"
    )

    print(
        f"Unique underlying entities:     "
        f"{len(unique_underlying_ids):,}"
    )

    # =========================================================================
    # 3. Phase-1 display-name reference
    #
    # Names are not identity keys.
    # We retain them only to help audit registry display-name consistency.
    # =========================================================================

    phase1_investor_names = (
        interactions[
            [
                "investor_id",
                "investor_name",
            ]
        ]
        .dropna()
        .drop_duplicates()
    )

    phase1_startup_names = (
        interactions[
            [
                "startup_id",
                "startup_name",
            ]
        ]
        .dropna()
        .drop_duplicates()
    )

    # =========================================================================
    # 4. Investor authoritative registry
    # =========================================================================

    investor_files = sorted(
        RAW_DIR.glob(
            "CRUNCHBASE_investor*.csv"
        )
    )

    if (
        len(
            investor_files
        )
        != 1
    ):

        raise ValueError(
            "Expected exactly one investor registry file."
        )

    investor_registry = pd.read_csv(
        investor_files[0],
        usecols=[
            "id",
            "name",
        ],
        dtype="string",
        low_memory=False,
    )

    investor_registry[
        "id"
    ] = clean_string(
        investor_registry[
            "id"
        ]
    )

    investor_registry[
        "name"
    ] = clean_string(
        investor_registry[
            "name"
        ]
    )

    if (
        investor_registry[
            "id"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Investor registry contains duplicate IDs."
        )

    canonical_investor_registry = (
        investor_registry[
            investor_registry[
                "id"
            ]
            .isin(
                investor_ids
            )
        ]
        .copy()
    )

    investor_registry_ids = set(
        canonical_investor_registry[
            "id"
        ]
    )

    missing_investor_registry = (
        investor_ids
        -
        investor_registry_ids
    )

    if (
        len(
            missing_investor_registry
        )
        > 0
    ):

        raise ValueError(
            f"{len(missing_investor_registry):,} "
            "canonical investors are missing from "
            "the Investor registry."
        )

    # =========================================================================
    # 5. Startup authoritative Company registry
    #
    # Scan only id + name and retain canonical Startup IDs.
    # =========================================================================

    company_files = sorted(
        RAW_DIR.glob(
            "companies*.csv"
        )
    )

    company_frames = []

    for file_index, path in enumerate(
        company_files,
        start=1,
    ):

        print(
            f"Scanning companies "
            f"[{file_index}/{len(company_files)}] "
            f"{path.name}"
        )

        chunk = pd.read_csv(
            path,
            usecols=[
                "id",
                "name",
            ],
            dtype="string",
            low_memory=False,
        )

        chunk[
            "id"
        ] = clean_string(
            chunk[
                "id"
            ]
        )

        chunk[
            "name"
        ] = clean_string(
            chunk[
                "name"
            ]
        )

        subset = (
            chunk[
                chunk[
                    "id"
                ]
                .isin(
                    startup_ids
                )
            ]
            .copy()
        )

        if len(
            subset
        ) == 0:
            continue

        subset[
            "source_file"
        ] = (
            path.name
        )

        company_frames.append(
            subset
        )

    startup_registry_rows = pd.concat(
        company_frames,
        ignore_index=True,
    )

    startup_registry_ids = set(
        startup_registry_rows[
            "id"
        ]
        .dropna()
        .unique()
    )

    missing_startup_registry = (
        startup_ids
        -
        startup_registry_ids
    )

    if (
        len(
            missing_startup_registry
        )
        > 0
    ):

        raise ValueError(
            f"{len(missing_startup_registry):,} "
            "canonical startups are missing from "
            "Companies."
        )

    # =========================================================================
    # 6. Audit company-name consistency across overlapping chunks
    # =========================================================================

    company_name_stats = (
        startup_registry_rows.groupby(
            "id",
            observed=True,
        )
        .agg(
            raw_registry_rows=(
                "id",
                "size",
            ),

            distinct_nonmissing_names=(
                "name",
                lambda values:
                    len(
                        {
                            name_key(
                                value
                            )
                            for value in (
                                values.dropna()
                            )
                            if name_key(
                                value
                            )
                            is not None
                        }
                    ),
            ),
        )
        .reset_index()
    )

    name_conflict_ids = set(
        company_name_stats.loc[
            company_name_stats[
                "distinct_nonmissing_names"
            ]
            > 1,
            "id",
        ]
    )

    company_name_conflicts = (
        startup_registry_rows[
            startup_registry_rows[
                "id"
            ]
            .isin(
                name_conflict_ids
            )
        ]
        .sort_values(
            [
                "id",
                "source_file",
            ]
        )
        .copy()
    )

    # =========================================================================
    # 7. Resolve one DISPLAY name per Startup UUID
    #
    # Identity is already known from UUID.
    #
    # Rule:
    #
    # 1. If Companies has one unique nonmissing name -> use it.
    #
    # 2. If Companies contains multiple display names, use a Phase-1
    #    startup_name only when exactly one Phase-1 name matches one of the
    #    Company-registry names case-insensitively.
    #
    # 3. If a display-name conflict cannot be resolved this way, STOP rather
    #    than choose an arbitrary label.
    # =========================================================================

    phase1_startup_name_map = (
        phase1_startup_names.groupby(
            "startup_id",
            observed=True,
        )[
            "startup_name"
        ]
        .apply(
            lambda values:
                list(
                    dict.fromkeys(
                        values.dropna()
                        .astype(str)
                        .tolist()
                    )
                )
        )
        .to_dict()
    )

    startup_display_rows = []

    unresolved_display_conflicts = []

    for entity_id, group in (
        startup_registry_rows.groupby(
            "id",
            observed=True,
        )
    ):

        registry_names = []

        seen_name_keys = set()

        for value in (
            group[
                "name"
            ]
            .dropna()
        ):

            key = name_key(
                value
            )

            if (
                key is None
                or key in seen_name_keys
            ):
                continue

            seen_name_keys.add(
                key
            )

            registry_names.append(
                str(
                    value
                ).strip()
            )

        if len(
            registry_names
        ) == 1:

            display_name = (
                registry_names[0]
            )

            resolution_method = (
                "unique_company_registry_name"
            )

        elif len(
            registry_names
        ) == 0:

            phase1_names = (
                phase1_startup_name_map.get(
                    entity_id,
                    [],
                )
            )

            unique_phase1 = list(
                dict.fromkeys(
                    [
                        str(
                            value
                        ).strip()
                        for value in phase1_names
                        if str(
                            value
                        ).strip()
                    ]
                )
            )

            if (
                len(
                    unique_phase1
                )
                == 1
            ):

                display_name = (
                    unique_phase1[0]
                )

                resolution_method = (
                    "phase1_name_registry_missing"
                )

            else:

                unresolved_display_conflicts.append(
                    entity_id
                )

                continue

        else:

            registry_by_key = {
                name_key(
                    value
                ):
                    value
                for value in (
                    registry_names
                )
            }

            phase1_names = (
                phase1_startup_name_map.get(
                    entity_id,
                    [],
                )
            )

            matching_keys = {
                name_key(
                    value
                )
                for value in phase1_names
                if (
                    name_key(
                        value
                    )
                    in registry_by_key
                )
            }

            if (
                len(
                    matching_keys
                )
                == 1
            ):

                matched_key = next(
                    iter(
                        matching_keys
                    )
                )

                display_name = (
                    registry_by_key[
                        matched_key
                    ]
                )

                resolution_method = (
                    "company_conflict_resolved_by_phase1_name"
                )

            else:

                unresolved_display_conflicts.append(
                    entity_id
                )

                continue

        startup_display_rows.append(
            {
                "id":
                    entity_id,

                "name":
                    display_name,

                "display_name_resolution":
                    resolution_method,
            }
        )

    if (
        len(
            unresolved_display_conflicts
        )
        > 0
    ):

        unresolved_path = (
            AUDIT_DIR
            / "graph_node_unresolved_display_name_conflicts.csv"
        )

        AUDIT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            company_name_conflicts[
                company_name_conflicts[
                    "id"
                ]
                .isin(
                    unresolved_display_conflicts
                )
            ]
            .to_csv(
                unresolved_path,
                index=False,
            )
        )

        raise ValueError(
            f"{len(unresolved_display_conflicts):,} "
            "Startup display-name conflicts could not "
            "be resolved conservatively. "
            f"See {unresolved_path}"
        )

    startup_registry = pd.DataFrame(
        startup_display_rows
    )

    if (
        startup_registry[
            "id"
        ]
        .nunique()
        !=
        EXPECTED_STARTUPS
    ):

        raise ValueError(
            "Startup registry consolidation did not "
            "produce exactly one row per canonical Startup."
        )

    # =========================================================================
    # 8. Build role-node tables
    # =========================================================================

    investor_nodes = (
        canonical_investor_registry[
            [
                "id",
                "name",
            ]
        ]
        .rename(
            columns={
                "id":
                    "raw_entity_id",

                "name":
                    "display_name",
            }
        )
        .copy()
    )

    investor_nodes[
        "node_type"
    ] = (
        "investor"
    )

    investor_nodes[
        "node_id"
    ] = (
        "investor::"
        +
        investor_nodes[
            "raw_entity_id"
        ]
    )

    investor_nodes[
        "source_registry"
    ] = (
        "investor"
    )

    investor_nodes[
        "underlying_entity_has_dual_role"
    ] = (
        investor_nodes[
            "raw_entity_id"
        ]
        .isin(
            dual_role_ids
        )
    )

    # -------------------------------------------------------------------------

    startup_nodes = (
        startup_registry[
            [
                "id",
                "name",
            ]
        ]
        .rename(
            columns={
                "id":
                    "raw_entity_id",

                "name":
                    "display_name",
            }
        )
        .copy()
    )

    startup_nodes[
        "node_type"
    ] = (
        "startup"
    )

    startup_nodes[
        "node_id"
    ] = (
        "startup::"
        +
        startup_nodes[
            "raw_entity_id"
        ]
    )

    startup_nodes[
        "source_registry"
    ] = (
        "companies"
    )

    startup_nodes[
        "underlying_entity_has_dual_role"
    ] = (
        startup_nodes[
            "raw_entity_id"
        ]
        .isin(
            dual_role_ids
        )
    )

    # =========================================================================
    # 9. Final frozen column order
    # =========================================================================

    node_columns = [
        "node_id",
        "node_type",
        "raw_entity_id",
        "display_name",
        "source_registry",
        "underlying_entity_has_dual_role",
    ]

    nodes = pd.concat(
        [
            investor_nodes[
                node_columns
            ],

            startup_nodes[
                node_columns
            ],
        ],
        ignore_index=True,
    )

    nodes = (
        nodes.sort_values(
            [
                "node_type",
                "raw_entity_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # =========================================================================
    # 10. Node-table integrity checks
    # =========================================================================

    if len(
        nodes
    ) != EXPECTED_ROLE_NODES:

        raise ValueError(
            f"Expected {EXPECTED_ROLE_NODES:,} nodes, "
            f"found {len(nodes):,}."
        )

    duplicate_node_ids = int(
        nodes[
            "node_id"
        ]
        .duplicated()
        .sum()
    )

    if (
        duplicate_node_ids
        != 0
    ):

        raise ValueError(
            f"Duplicate graph node IDs: "
            f"{duplicate_node_ids:,}"
        )

    missing_raw_ids = int(
        nodes[
            "raw_entity_id"
        ]
        .isna()
        .sum()
    )

    missing_names = int(
        nodes[
            "display_name"
        ]
        .isna()
        .sum()
    )

    if (
        missing_raw_ids
        != 0
    ):

        raise ValueError(
            f"Missing raw_entity_id values: "
            f"{missing_raw_ids:,}"
        )

    if (
        missing_names
        != 0
    ):

        raise ValueError(
            f"Missing display_name values: "
            f"{missing_names:,}"
        )

    node_underlying_unique = (
        nodes[
            "raw_entity_id"
        ]
        .nunique()
    )

    if (
        node_underlying_unique
        !=
        EXPECTED_UNIQUE_UNDERLYING_IDS
    ):

        raise ValueError(
            "Unexpected underlying entity count in node table: "
            f"{node_underlying_unique:,}"
        )

    dual_flagged_role_nodes = int(
        nodes[
            "underlying_entity_has_dual_role"
        ]
        .sum()
    )

    expected_dual_flagged_nodes = (
        EXPECTED_DUAL_ROLE_UNDERLYING_IDS
        * 2
    )

    if (
        dual_flagged_role_nodes
        !=
        expected_dual_flagged_nodes
    ):

        raise ValueError(
            "Dual-role graph-node flag count is wrong. "
            f"Expected {expected_dual_flagged_nodes:,}, "
            f"found {dual_flagged_role_nodes:,}."
        )

    # =========================================================================
    # 11. Dual-role display-name agreement
    # =========================================================================

    investor_dual = (
        investor_nodes[
            investor_nodes[
                "raw_entity_id"
            ]
            .isin(
                dual_role_ids
            )
        ][
            [
                "raw_entity_id",
                "display_name",
            ]
        ]
        .rename(
            columns={
                "display_name":
                    "investor_display_name"
            }
        )
    )

    startup_dual = (
        startup_nodes[
            startup_nodes[
                "raw_entity_id"
            ]
            .isin(
                dual_role_ids
            )
        ][
            [
                "raw_entity_id",
                "display_name",
            ]
        ]
        .rename(
            columns={
                "display_name":
                    "startup_display_name"
            }
        )
    )

    dual_role_audit = (
        investor_dual.merge(
            startup_dual,
            on="raw_entity_id",
            how="outer",
            validate="one_to_one",
        )
    )

    dual_role_audit[
        "names_exact_match_casefold"
    ] = [
        (
            name_key(
                investor_name
            )
            ==
            name_key(
                startup_name
            )
        )
        for investor_name, startup_name in zip(
            dual_role_audit[
                "investor_display_name"
            ],
            dual_role_audit[
                "startup_display_name"
            ],
        )
    ]

    dual_role_name_matches = int(
        dual_role_audit[
            "names_exact_match_casefold"
        ]
        .sum()
    )

    dual_role_name_disagreements = (
        len(
            dual_role_audit
        )
        -
        dual_role_name_matches
    )

    # =========================================================================
    # 12. Summaries
    # =========================================================================

    node_type_summary = (
        nodes.groupby(
            "node_type",
            observed=True,
        )
        .agg(
            node_count=(
                "node_id",
                "size",
            ),

            unique_underlying_ids=(
                "raw_entity_id",
                "nunique",
            ),

            dual_role_flagged_nodes=(
                "underlying_entity_has_dual_role",
                "sum",
            ),
        )
        .reset_index()
    )

    registry_coverage = pd.DataFrame(
        [
            {
                "role":
                    "investor",

                "canonical_role_ids":
                    EXPECTED_INVESTORS,

                "registry_matched_ids":
                    len(
                        investor_registry_ids
                    ),

                "coverage_pct":
                    (
                        len(
                            investor_registry_ids
                        )
                        /
                        EXPECTED_INVESTORS
                        * 100
                    ),

                "registry":
                    "investor",
            },

            {
                "role":
                    "startup",

                "canonical_role_ids":
                    EXPECTED_STARTUPS,

                "registry_matched_ids":
                    len(
                        startup_registry_ids
                    ),

                "coverage_pct":
                    (
                        len(
                            startup_registry_ids
                        )
                        /
                        EXPECTED_STARTUPS
                        * 100
                    ),

                "registry":
                    "companies",
            },
        ]
    )

    identity_summary = pd.DataFrame(
        [
            {
                "metric":
                    "total_role_nodes",

                "value":
                    len(
                        nodes
                    ),
            },

            {
                "metric":
                    "investor_nodes",

                "value":
                    int(
                        (
                            nodes[
                                "node_type"
                            ]
                            == "investor"
                        )
                        .sum()
                    ),
            },

            {
                "metric":
                    "startup_nodes",

                "value":
                    int(
                        (
                            nodes[
                                "node_type"
                            ]
                            == "startup"
                        )
                        .sum()
                    ),
            },

            {
                "metric":
                    "unique_underlying_entity_ids",

                "value":
                    node_underlying_unique,
            },

            {
                "metric":
                    "dual_role_underlying_entity_ids",

                "value":
                    len(
                        dual_role_ids
                    ),
            },

            {
                "metric":
                    "dual_role_graph_nodes",

                "value":
                    dual_flagged_role_nodes,
            },

            {
                "metric":
                    "duplicate_node_ids",

                "value":
                    duplicate_node_ids,
            },

            {
                "metric":
                    "missing_raw_entity_ids",

                "value":
                    missing_raw_ids,
            },

            {
                "metric":
                    "missing_display_names",

                "value":
                    missing_names,
            },

            {
                "metric":
                    "startup_company_name_conflict_ids",

                "value":
                    len(
                        name_conflict_ids
                    ),
            },

            {
                "metric":
                    "dual_role_display_name_matches",

                "value":
                    dual_role_name_matches,
            },

            {
                "metric":
                    "dual_role_display_name_disagreements",

                "value":
                    dual_role_name_disagreements,
            },
        ]
    )

    # =========================================================================
    # 13. Save materialized node table and audits
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

    nodes.to_parquet(
        NODE_OUTPUT,
        index=False,
    )

    identity_summary.to_csv(
        IDENTITY_SUMMARY_OUTPUT,
        index=False,
    )

    node_type_summary.to_csv(
        NODE_TYPE_OUTPUT,
        index=False,
    )

    registry_coverage.to_csv(
        REGISTRY_COVERAGE_OUTPUT,
        index=False,
    )

    dual_role_audit.to_csv(
        DUAL_ROLE_OUTPUT,
        index=False,
    )

    company_name_conflicts.to_csv(
        COMPANY_NAME_CONFLICT_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 14. Figures
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        node_type_summary[
            "node_type"
        ],
        node_type_summary[
            "node_count"
        ],
    )

    ax.set_ylabel(
        "Graph nodes"
    )

    ax.set_title(
        "ITRS-Crunchbase Graph Node Types"
    )

    fig.tight_layout()

    fig.savefig(
        NODE_TYPE_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # -------------------------------------------------------------------------

    identity_plot = pd.DataFrame(
        [
            {
                "identity_class":
                    "Unique underlying entities",

                "count":
                    node_underlying_unique,
            },

            {
                "identity_class":
                    "Role nodes",

                "count":
                    len(
                        nodes
                    ),
            },

            {
                "identity_class":
                    "Dual-role underlying entities",

                "count":
                    len(
                        dual_role_ids
                    ),
            },
        ]
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        identity_plot[
            "identity_class"
        ],
        identity_plot[
            "count"
        ],
    )

    ax.set_ylabel(
        "Count"
    )

    ax.set_title(
        "Underlying Identity vs Semantic Graph Roles"
    )

    ax.tick_params(
        axis="x",
        rotation=15,
    )

    fig.tight_layout()

    fig.savefig(
        IDENTITY_STRUCTURE_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =========================================================================
    # 15. Terminal output
    # =========================================================================

    separator("-")

    print(
        "GRAPH NODE IDENTITY SUMMARY"
    )

    separator("-")

    print(
        identity_summary.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "NODE TYPE SUMMARY"
    )

    separator("-")

    print(
        node_type_summary.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "AUTHORITATIVE REGISTRY COVERAGE"
    )

    separator("-")

    print(
        registry_coverage.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "DUAL-ROLE IDENTITY AUDIT"
    )

    separator("-")

    print(
        f"Dual-role underlying IDs:       "
        f"{len(dual_role_audit):,}"
    )

    print(
        f"Display-name matches:           "
        f"{dual_role_name_matches:,}"
    )

    print(
        f"Display-name disagreements:     "
        f"{dual_role_name_disagreements:,}"
    )

    if (
        dual_role_name_disagreements
        > 0
    ):

        print(
            "\nSample disagreements:"
        )

        print(
            dual_role_audit[
                ~dual_role_audit[
                    "names_exact_match_casefold"
                ]
            ]
            .head(
                20
            )
            .to_string(
                index=False
            )
        )

    separator("-")

    print(
        "NODE TABLE MATERIALIZATION"
    )

    separator("-")

    print(
        f"Node table: "
        f"{NODE_OUTPUT}"
    )

    print(
        f"Rows:       "
        f"{len(nodes):,}"
    )

    print(
        f"Columns:    "
        f"{len(nodes.columns):,}"
    )

    separator()

    print(
        "PHASE 3.3.2 COMPLETE"
    )

    separator()

    print(
        f"""
Materialized node table:

{NODE_OUTPUT}

Audits written to:

{IDENTITY_SUMMARY_OUTPUT}
{NODE_TYPE_OUTPUT}
{REGISTRY_COVERAGE_OUTPUT}
{DUAL_ROLE_OUTPUT}
{COMPANY_NAME_CONFLICT_OUTPUT}

Figures written to:

{NODE_TYPE_FIGURE}
{IDENTITY_STRUCTURE_FIGURE}


IMPORTANT

1. Graph identity is role + Crunchbase UUID.

2. Display names are attributes only and are never used as graph keys.

3. Canonical Investor and Startup universes remain unchanged.

4. Dual-role underlying entities are represented by TWO graph nodes.

5. No node filtering was applied.

6. NO STRUCTURAL EDGE HAS YET BEEN MATERIALIZED.

NEXT:

Phase 3.3.3 — Structural Edge-Table Materialization & Relation Integrity Audit
"""
    )


if __name__ == "__main__":
    main()