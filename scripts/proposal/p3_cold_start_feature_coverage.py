from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse


# =============================================================================
# P3 — COLD-START FEATURE COVERAGE AUDIT
# =============================================================================
#
# Purpose:
#   Determine whether startups with little or zero pre-T60 investment history
#   still possess non-investment side information that could support an
#   inductive startup representation.
#
# IMPORTANT:
#   - NO training
#   - NO inference
#   - NO checkpoint access
#   - NO T60-as-history
#   - uses ONLY already-frozen Phase-3 / Phase-4 model inputs
#
# This is an AVAILABILITY audit, not a new temporal-validity guarantee.
# =============================================================================


ROOT = Path(__file__).resolve().parents[2]

NUM_INVESTORS = 165_975
NUM_STARTUPS = 311_589
TOTAL_ROLE_NODES = 477_564
EXPECTED_TEST_EVENTS = 20_264


# =============================================================================
# Inputs
# =============================================================================

PROPOSAL_ROOT = (
    ROOT
    / "data"
    / "experimental"
    / "proposal_evidence"
)

P1_EVENTS = (
    PROPOSAL_ROOT
    / "06_test_case_history_diagnostics.csv"
)

NODE_INDEX = (
    ROOT
    / "data"
    / "experimental"
    / "phase_3"
    / "model_ready"
    / "node_index.parquet"
)

EDGE_INDEX = (
    ROOT
    / "data"
    / "experimental"
    / "phase_3"
    / "model_ready"
    / "edge_index.npy"
)

DOC2VEC = (
    ROOT
    / "data"
    / "experimental"
    / "phase_4"
    / "doc2vec"
    / "vectors"
    / "doc2vec_vectors_all.npy"
)

DESCRIPTION_LABELS = (
    ROOT
    / "data"
    / "experimental"
    / "phase_4"
    / "description_labels"
    / "description_label_multihot.npz"
)


# =============================================================================
# Outputs
# =============================================================================

OUT_EVENT_FEATURES = (
    PROPOSAL_ROOT
    / "15_cold_start_feature_coverage_events.csv"
)

OUT_UNIQUE_STARTUPS = (
    PROPOSAL_ROOT
    / "16_cold_start_feature_coverage_unique_startups.csv"
)

OUT_BY_HISTORY = (
    PROPOSAL_ROOT
    / "17_feature_coverage_by_startup_history.csv"
)

OUT_BY_GROUP = (
    PROPOSAL_ROOT
    / "18_feature_coverage_by_diagnostic_group.csv"
)

OUT_SUMMARY = (
    PROPOSAL_ROOT
    / "p3_feature_coverage_summary.json"
)


# =============================================================================
# Helpers
# =============================================================================

def banner(text: str) -> None:
    print()
    print("=" * 112)
    print(text)
    print("=" * 112)


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def history_bin(count: int) -> str:
    count = int(count)

    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 9:
        return "5-9"

    return "10+"


def bool_share(frame: pd.DataFrame, column: str) -> float:
    if len(frame) == 0:
        return np.nan

    return float(
        frame[column].astype(bool).mean()
    )


def coverage_row(
    frame: pd.DataFrame,
    label: str,
    level: str,
) -> dict:

    return {
        "level": level,
        "group": label,
        "n": int(len(frame)),

        "doc2vec_coverage":
            bool_share(
                frame,
                "has_doc2vec",
            ),

        "category_coverage":
            bool_share(
                frame,
                "has_category",
            ),

        "incoming_structural_coverage":
            bool_share(
                frame,
                "has_incoming_structural_relation",
            ),

        "any_structural_relation_coverage":
            bool_share(
                frame,
                "has_any_structural_relation",
            ),

        "description_and_category_coverage":
            bool_share(
                frame,
                "has_description_and_category",
            ),

        "content_side_info_coverage":
            bool_share(
                frame,
                "has_content_side_info",
            ),

        "model_usable_side_info_coverage":
            bool_share(
                frame,
                "has_model_usable_side_info",
            ),

        "no_model_usable_side_info_share":
            bool_share(
                frame,
                "no_model_usable_side_info",
            ),

        "all_three_model_channels_coverage":
            bool_share(
                frame,
                "has_all_three_model_channels",
            ),

        "mean_side_info_channel_count":
            (
                float(
                    frame[
                        "model_side_info_channel_count"
                    ].mean()
                )
                if len(frame)
                else np.nan
            ),

        "mean_incoming_structural_degree":
            (
                float(
                    frame[
                        "incoming_structural_degree"
                    ].mean()
                )
                if len(frame)
                else np.nan
            ),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "P3 — COLD-START FEATURE COVERAGE AUDIT"
    )

    for path in [
        P1_EVENTS,
        NODE_INDEX,
        EDGE_INDEX,
        DOC2VEC,
        DESCRIPTION_LABELS,
    ]:
        require(
            path.exists(),
            f"Missing required input: {path}",
        )

        print(
            "FOUND ",
            path.relative_to(ROOT),
        )

    # =========================================================================
    # 1. Load final test events
    # =========================================================================

    events = pd.read_csv(
        P1_EVENTS
    )

    require(
        len(events)
        == EXPECTED_TEST_EVENTS,
        (
            f"Expected {EXPECTED_TEST_EVENTS:,} events; "
            f"found {len(events):,}."
        ),
    )

    require(
        events["interaction_id"].is_unique,
        "interaction_id not unique.",
    )

    events[
        "startup_history_count"
    ] = pd.to_numeric(
        events[
            "startup_history_count"
        ],
        errors="raise",
    ).astype(np.int64)

    # =========================================================================
    # 2. Frozen role-node mapping
    # =========================================================================

    banner(
        "LOAD FROZEN ROLE-NODE REGISTRY"
    )

    node_index = pd.read_parquet(
        NODE_INDEX,
        columns=[
            "node_index",
            "raw_entity_id",
            "node_type",
        ],
    )

    require(
        len(node_index)
        == TOTAL_ROLE_NODES,
        "Phase-3 role-node count drift.",
    )

    node_index[
        "node_index"
    ] = pd.to_numeric(
        node_index["node_index"],
        errors="raise",
    ).astype(np.int64)

    startup_nodes = (
        node_index.loc[
            (
                node_index["node_index"]
                >= NUM_INVESTORS
            )
            & (
                node_index["node_index"]
                < TOTAL_ROLE_NODES
            ),
            [
                "node_index",
                "raw_entity_id",
            ],
        ]
        .copy()
    )

    require(
        len(startup_nodes)
        == NUM_STARTUPS,
        "Startup numeric role slice drift.",
    )

    require(
        startup_nodes[
            "raw_entity_id"
        ].astype(str).is_unique,
        "Startup raw IDs not unique.",
    )

    startup_map = pd.Series(
        startup_nodes[
            "node_index"
        ].to_numpy(
            dtype=np.int64
        ),
        index=startup_nodes[
            "raw_entity_id"
        ].astype(str),
    ).to_dict()

    startup_global = (
        events[
            "startup_id"
        ]
        .astype(str)
        .map(startup_map)
    )

    require(
        startup_global.notna().all(),
        "Some test startups could not be mapped.",
    )

    events[
        "startup_global_node_index"
    ] = startup_global.astype(
        np.int64
    )

    events[
        "startup_local_index"
    ] = (
        events[
            "startup_global_node_index"
        ]
        - NUM_INVESTORS
    )

    # =========================================================================
    # 3. Doc2Vec coverage
    # =========================================================================

    banner(
        "AUDIT DOC2VEC DESCRIPTION COVERAGE"
    )

    doc2vec = np.load(
        DOC2VEC,
        mmap_mode="r",
    )

    require(
        doc2vec.shape
        == (
            TOTAL_ROLE_NODES,
            32,
        ),
        f"Unexpected Doc2Vec shape: {doc2vec.shape}",
    )

    has_doc2vec_global = np.any(
        np.asarray(doc2vec) != 0,
        axis=1,
    )

    global_idx = events[
        "startup_global_node_index"
    ].to_numpy(
        dtype=np.int64
    )

    events[
        "has_doc2vec"
    ] = has_doc2vec_global[
        global_idx
    ]

    print(
        "Global all-zero Doc2Vec rows:",
        f"{int((~has_doc2vec_global).sum()):,}",
    )

    # =========================================================================
    # 4. Category-label coverage
    # =========================================================================

    banner(
        "AUDIT CATEGORY FEATURE COVERAGE"
    )

    labels = sparse.load_npz(
        DESCRIPTION_LABELS
    ).tocsr()

    require(
        labels.shape
        == (
            TOTAL_ROLE_NODES,
            802,
        ),
        (
            "Unexpected description-label shape: "
            f"{labels.shape}"
        ),
    )

    label_count_global = np.diff(
        labels.indptr
    ).astype(
        np.int64
    )

    has_category_global = (
        label_count_global > 0
    )

    events[
        "category_label_count"
    ] = label_count_global[
        global_idx
    ]

    events[
        "has_category"
    ] = has_category_global[
        global_idx
    ]

    print(
        "Global zero-category rows:",
        f"{int((~has_category_global).sum()):,}",
    )

    # =========================================================================
    # 5. Heterogeneous structural coverage
    # =========================================================================

    banner(
        "AUDIT HETEROGENEOUS STRUCTURAL COVERAGE"
    )

    edge_index = np.load(
        EDGE_INDEX,
        mmap_mode="r",
    )

    require(
        edge_index.shape
        == (
            2,
            158_818,
        ),
        (
            "Unexpected structural edge shape: "
            f"{edge_index.shape}"
        ),
    )

    src = np.asarray(
        edge_index[0],
        dtype=np.int64,
    )

    dst = np.asarray(
        edge_index[1],
        dtype=np.int64,
    )

    require(
        (
            (src >= 0)
            & (src < TOTAL_ROLE_NODES)
            & (dst >= 0)
            & (dst < TOTAL_ROLE_NODES)
        ).all(),
        "Structural edge endpoint outside node range.",
    )

    incoming_degree = np.bincount(
        dst,
        minlength=TOTAL_ROLE_NODES,
    ).astype(np.int64)

    outgoing_degree = np.bincount(
        src,
        minlength=TOTAL_ROLE_NODES,
    ).astype(np.int64)

    incident_degree = (
        incoming_degree
        + outgoing_degree
    )

    events[
        "incoming_structural_degree"
    ] = incoming_degree[
        global_idx
    ]

    events[
        "outgoing_structural_degree"
    ] = outgoing_degree[
        global_idx
    ]

    events[
        "incident_structural_degree"
    ] = incident_degree[
        global_idx
    ]

    # This is the stricter signal corresponding to actual incoming
    # message availability in the frozen directed R-GCN.
    events[
        "has_incoming_structural_relation"
    ] = (
        events[
            "incoming_structural_degree"
        ] > 0
    )

    # Broader diagnostic: any incident heterogeneous relation.
    events[
        "has_any_structural_relation"
    ] = (
        events[
            "incident_structural_degree"
        ] > 0
    )

    # =========================================================================
    # 6. Combined inductive-information availability
    # =========================================================================

    events[
        "has_description_and_category"
    ] = (
        events["has_doc2vec"]
        & events["has_category"]
    )

    events[
        "has_content_side_info"
    ] = (
        events["has_doc2vec"]
        | events["has_category"]
    )

    # Conservative "usable by the current model components" definition:
    # description OR category OR incoming heterogeneous structural message.
    events[
        "has_model_usable_side_info"
    ] = (
        events["has_doc2vec"]
        | events["has_category"]
        | events[
            "has_incoming_structural_relation"
        ]
    )

    events[
        "no_model_usable_side_info"
    ] = ~events[
        "has_model_usable_side_info"
    ]

    events[
        "has_all_three_model_channels"
    ] = (
        events["has_doc2vec"]
        & events["has_category"]
        & events[
            "has_incoming_structural_relation"
        ]
    )

    events[
        "model_side_info_channel_count"
    ] = (
        events[
            [
                "has_doc2vec",
                "has_category",
                "has_incoming_structural_relation",
            ]
        ]
        .astype(np.int8)
        .sum(axis=1)
    )

    events[
        "startup_history_bin"
    ] = events[
        "startup_history_count"
    ].map(
        history_bin
    )

    # =========================================================================
    # 7. Feature profile label
    # =========================================================================

    def feature_profile(row) -> str:

        channels = []

        if row["has_doc2vec"]:
            channels.append("description")

        if row["has_category"]:
            channels.append("category")

        if row[
            "has_incoming_structural_relation"
        ]:
            channels.append("structure")

        if not channels:
            return "none"

        return "+".join(channels)

    events[
        "inductive_feature_profile"
    ] = events.apply(
        feature_profile,
        axis=1,
    )

    # =========================================================================
    # 8. Save event-level feature coverage
    # =========================================================================

    events.to_csv(
        OUT_EVENT_FEATURES,
        index=False,
    )

    # =========================================================================
    # 9. Unique-startup view
    # =========================================================================

    banner(
        "BUILD UNIQUE-STARTUP COVERAGE VIEW"
    )

    consistency = (
        events
        .groupby(
            "startup_id",
            observed=True,
        )[
            "startup_history_count"
        ]
        .nunique()
    )

    require(
        int(
            (consistency > 1).sum()
        ) == 0,
        (
            "Same startup has inconsistent "
            "pre-T60 history counts."
        ),
    )

    feature_columns = [
        "startup_id",
        "startup_name",
        "startup_global_node_index",
        "startup_local_index",
        "startup_history_count",
        "startup_history_bin",
        "has_doc2vec",
        "category_label_count",
        "has_category",
        "incoming_structural_degree",
        "outgoing_structural_degree",
        "incident_structural_degree",
        "has_incoming_structural_relation",
        "has_any_structural_relation",
        "has_description_and_category",
        "has_content_side_info",
        "has_model_usable_side_info",
        "no_model_usable_side_info",
        "has_all_three_model_channels",
        "model_side_info_channel_count",
        "inductive_feature_profile",
    ]

    unique_startups = (
        events[
            feature_columns
        ]
        .drop_duplicates(
            subset=[
                "startup_id"
            ]
        )
        .copy()
    )

    event_counts = (
        events
        .groupby(
            "startup_id",
            observed=True,
        )
        .size()
        .rename(
            "test_event_count"
        )
    )

    discovery_counts = (
        events.loc[
            events[
                "new_to_investor"
            ].astype(bool)
        ]
        .groupby(
            "startup_id",
            observed=True,
        )
        .size()
        .rename(
            "new_to_investor_test_event_count"
        )
    )

    unique_startups = (
        unique_startups
        .merge(
            event_counts,
            left_on="startup_id",
            right_index=True,
            how="left",
            validate="one_to_one",
        )
        .merge(
            discovery_counts,
            left_on="startup_id",
            right_index=True,
            how="left",
            validate="one_to_one",
        )
    )

    unique_startups[
        "new_to_investor_test_event_count"
    ] = (
        unique_startups[
            "new_to_investor_test_event_count"
        ]
        .fillna(0)
        .astype(np.int64)
    )

    unique_startups.to_csv(
        OUT_UNIQUE_STARTUPS,
        index=False,
    )

    print(
        "Unique test startups:",
        f"{len(unique_startups):,}",
    )

    print(
        "Unique zero-history test startups:",
        f"{int((unique_startups['startup_history_count'] == 0).sum()):,}",
    )

    # =========================================================================
    # 10. Coverage by history amount
    # =========================================================================

    history_order = [
        "0",
        "1",
        "2-4",
        "5-9",
        "10+",
    ]

    history_rows = []

    for level_name, frame in [
        (
            "event",
            events,
        ),
        (
            "unique_startup",
            unique_startups,
        ),
    ]:
        for history_group in history_order:

            subset = frame.loc[
                frame[
                    "startup_history_bin"
                ].eq(
                    history_group
                )
            ]

            history_rows.append(
                coverage_row(
                    subset,
                    history_group,
                    level_name,
                )
            )

    history_df = pd.DataFrame(
        history_rows
    )

    history_df.to_csv(
        OUT_BY_HISTORY,
        index=False,
    )

    # =========================================================================
    # 11. Coverage by thesis diagnostic group
    # =========================================================================

    group_order = [
        "repeat_pair",
        "novel_warm_warm",
        "novel_cold_startup",
        "novel_cold_investor",
        "novel_both_cold",
    ]

    group_rows = []

    for group in group_order:

        subset = events.loc[
            events[
                "proposal_diagnostic_group"
            ].eq(group)
        ]

        group_rows.append(
            coverage_row(
                subset,
                group,
                "event",
            )
        )

    group_df = pd.DataFrame(
        group_rows
    )

    group_df.to_csv(
        OUT_BY_GROUP,
        index=False,
    )

    # =========================================================================
    # 12. Core cold-start summary
    # =========================================================================

    cold_events = events.loc[
        events[
            "startup_history_count"
        ].eq(0)
    ].copy()

    cold_unique = (
        unique_startups.loc[
            unique_startups[
                "startup_history_count"
            ].eq(0)
        ]
        .copy()
    )

    require(
        len(cold_events) == 8_104,
        (
            "Expected 8,104 zero-history "
            f"test events, found {len(cold_events):,}."
        ),
    )

    summary = {
        "schema_version":
            "PROPOSAL_EVIDENCE_P3_V1",

        "status":
            "P3_COMPLETE",

        "scientific_role":
            (
                "Availability audit of already-frozen "
                "Phase-4 description/category inputs and "
                "Phase-3 heterogeneous structural relations."
            ),

        "temporal_guard":
            (
                "Startup investment history is measured "
                "strictly from T0-T59. This audit does not "
                "create a new temporal-validity guarantee "
                "for static metadata."
            ),

        "test_events":
            int(len(events)),

        "unique_test_startups":
            int(len(unique_startups)),

        "cold_start_events":
            int(len(cold_events)),

        "cold_start_unique_startups":
            int(len(cold_unique)),

        "cold_start_event_feature_coverage": {
            "doc2vec":
                bool_share(
                    cold_events,
                    "has_doc2vec",
                ),

            "category":
                bool_share(
                    cold_events,
                    "has_category",
                ),

            "incoming_structure":
                bool_share(
                    cold_events,
                    "has_incoming_structural_relation",
                ),

            "any_structural_relation":
                bool_share(
                    cold_events,
                    "has_any_structural_relation",
                ),

            "description_and_category":
                bool_share(
                    cold_events,
                    "has_description_and_category",
                ),

            "content_side_info":
                bool_share(
                    cold_events,
                    "has_content_side_info",
                ),

            "any_model_usable_side_info":
                bool_share(
                    cold_events,
                    "has_model_usable_side_info",
                ),

            "no_model_usable_side_info":
                bool_share(
                    cold_events,
                    "no_model_usable_side_info",
                ),

            "all_three_model_channels":
                bool_share(
                    cold_events,
                    "has_all_three_model_channels",
                ),
        },

        "cold_start_unique_startup_feature_coverage": {
            "doc2vec":
                bool_share(
                    cold_unique,
                    "has_doc2vec",
                ),

            "category":
                bool_share(
                    cold_unique,
                    "has_category",
                ),

            "incoming_structure":
                bool_share(
                    cold_unique,
                    "has_incoming_structural_relation",
                ),

            "any_structural_relation":
                bool_share(
                    cold_unique,
                    "has_any_structural_relation",
                ),

            "description_and_category":
                bool_share(
                    cold_unique,
                    "has_description_and_category",
                ),

            "content_side_info":
                bool_share(
                    cold_unique,
                    "has_content_side_info",
                ),

            "any_model_usable_side_info":
                bool_share(
                    cold_unique,
                    "has_model_usable_side_info",
                ),

            "no_model_usable_side_info":
                bool_share(
                    cold_unique,
                    "no_model_usable_side_info",
                ),

            "all_three_model_channels":
                bool_share(
                    cold_unique,
                    "has_all_three_model_channels",
                ),
        },

        "outputs": {
            "event_features":
                str(
                    OUT_EVENT_FEATURES.relative_to(
                        ROOT
                    )
                ),

            "unique_startups":
                str(
                    OUT_UNIQUE_STARTUPS.relative_to(
                        ROOT
                    )
                ),

            "coverage_by_history":
                str(
                    OUT_BY_HISTORY.relative_to(
                        ROOT
                    )
                ),

            "coverage_by_diagnostic_group":
                str(
                    OUT_BY_GROUP.relative_to(
                        ROOT
                    )
                ),
        },
    }

    with OUT_SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    # =========================================================================
    # 13. Console reporting
    # =========================================================================

    banner(
        "FEATURE COVERAGE BY STARTUP PRE-T60 HISTORY"
    )

    display = (
        history_df.loc[
            history_df[
                "level"
            ].eq(
                "unique_startup"
            )
        ]
        .copy()
    )

    percentage_cols = [
        "doc2vec_coverage",
        "category_coverage",
        "incoming_structural_coverage",
        "any_structural_relation_coverage",
        "description_and_category_coverage",
        "content_side_info_coverage",
        "model_usable_side_info_coverage",
        "no_model_usable_side_info_share",
        "all_three_model_channels_coverage",
    ]

    for col in percentage_cols:
        display[col] = (
            display[col]
            * 100
        )

    print(
        display.to_string(
            index=False,
            formatters={
                col:
                    (
                        lambda x:
                            f"{x:.2f}%"
                    )
                for col in percentage_cols
            },
        )
    )

    banner(
        "COLD-START STARTUP FEATURE PROFILES"
    )

    profile_counts = (
        cold_unique[
            "inductive_feature_profile"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "feature_profile"
        )
        .reset_index(
            name="unique_startups"
        )
    )

    profile_counts[
        "share"
    ] = (
        profile_counts[
            "unique_startups"
        ]
        / len(cold_unique)
    )

    print(
        profile_counts.to_string(
            index=False,
            formatters={
                "share":
                    lambda x:
                        f"{x:.2%}",
            },
        )
    )

    banner(
        "P3 OUTPUTS"
    )

    for path in [
        OUT_EVENT_FEATURES,
        OUT_UNIQUE_STARTUPS,
        OUT_BY_HISTORY,
        OUT_BY_GROUP,
        OUT_SUMMARY,
    ]:
        print(
            "WROTE ",
            path.relative_to(ROOT),
        )

    banner(
        "P3 COMPLETE — FEATURE AVAILABILITY AUDITED / "
        "NO TRAINING / NO INFERENCE"
    )


if __name__ == "__main__":
    main()
