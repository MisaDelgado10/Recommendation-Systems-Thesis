#!/usr/bin/env python3

from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 8.2.2 — STRICT INDUCTIVE HETEROGENEOUS GRAPH MATERIALIZATION V1
# =============================================================================

PHASE8_ROOT = Path("data/experimental/phase_8")

METHOD_ROOT = (
    PHASE8_ROOT
    / "phase_8_1_inductive_startup_representation"
)

FRAMEWORK_ROOT = (
    PHASE8_ROOT
    / "phase_8_2_framework_development"
)

FEATURE_ROOT = (
    FRAMEWORK_ROOT
    / "phase_8_2_1_feature_materialization"
)

OUT_DIR = (
    FRAMEWORK_ROOT
    / "phase_8_2_2_strict_inductive_heterogeneous_graph"
)


# =============================================================================
# AUTHORITATIVE CONTRACTS
# =============================================================================

GRAPH_SCHEMA = (
    METHOD_ROOT
    / "phase_8_1_9_heterogeneous_graph_schema"
    / "phase_8_1_9_graph_schema_V2.json"
)

CONSISTENCY_CONTRACT = (
    METHOD_ROOT
    / "phase_8_1_10a_inductive_consistency_closure"
    / "phase_8_1_10a_consistency_amendment_V2.json"
)

ARCHITECTURE_CONTRACT = (
    METHOD_ROOT
    / "phase_8_1_11_exact_neural_architecture"
    / "phase_8_1_11_architecture_contract_V1.json"
)

COLD_MASK_CONTRACT = (
    METHOD_ROOT
    / "phase_8_1_12a_outcome_independent_cold_start_masking"
    / "phase_8_1_12a_outcome_independent_masking_contract_V1.json"
)

EXECUTION_CONTRACT = (
    METHOD_ROOT
    / "phase_8_1_12_execution_contract"
    / "phase_8_1_12_execution_contract_V5.json"
)


# =============================================================================
# AUTHORITATIVE MATERIALIZED INPUTS
# =============================================================================

PHASE3_NODE_INDEX = Path(
    "data/experimental/phase_3/"
    "model_ready/"
    "node_index.parquet"
)

ACQUISITION_RESOLVED = Path(
    "data/experimental/phase_3/"
    "audits/"
    "acquisition_resolved_relation_candidates.parquet"
)

STARTUP_SOURCE = (
    FEATURE_ROOT
    / "phase_8_2_1a_canonical_startup_source_V1.parquet"
)

PERSON_SOURCE = (
    FEATURE_ROOT
    / "phase_8_2_1b_person_node_source_V1.parquet"
)

FOUNDER_EDGES = (
    FEATURE_ROOT
    / "phase_8_2_1b_startup_founder_edges_V1.parquet"
)

ORIGIN_MASKS = (
    FEATURE_ROOT
    / "phase_8_2_1c_origin_startup_masks_V2.parquet"
)

CONFIRMATORY_MASK_DIR = (
    PHASE8_ROOT
    / "phase_8_0_evaluation_reset"
    / "phase_8_0_3_confirmatory_safe_graph_masks"
)

REMOVED_STRUCTURAL_EDGES = (
    CONFIRMATORY_MASK_DIR
    / "phase_8_0_3_removed_structural_edges_V1.csv"
)

MASK_FREEZE_SUMMARY = (
    CONFIRMATORY_MASK_DIR
    / "phase_8_0_3_graph_mask_freeze_summary_V1.json"
)


# =============================================================================
# FROZEN CONSTANTS
# =============================================================================

ORIGIN_ORDER = [
    "T57",
    "T58",
    "T59",
    "T60",
    "APRIL_CONFIRMATORY",
]

ACCEPTED_ENDPOINT_STATUSES = {
    "name_website_agree",
    "ambiguous_name_resolved_by_website",
    "unique_name_only",
}

EXPECTED_ROLE_NODE_COUNT = 477_564
EXPECTED_INVESTOR_COUNT = 165_975
EXPECTED_STARTUP_COUNT = 311_589
EXPECTED_PERSON_COUNT = 308_422
EXPECTED_FOUNDER_EDGE_COUNT = 341_444

EXPECTED_FINAL_NODE_COUNT = (
    EXPECTED_ROLE_NODE_COUNT
    + EXPECTED_PERSON_COUNT
)

RELATION_VOCAB = {
    "startup|founded_by|person": 0,
    "person|founded|startup": 1,
    "investor|acquired|investor": 2,
    "investor|acquired|startup": 3,
    "startup|acquired|investor": 4,
    "startup|acquired|startup": 5,
    "investor|acquired_by|investor": 6,
    "investor|acquired_by|startup": 7,
    "startup|acquired_by|investor": 8,
    "startup|acquired_by|startup": 9,
}


# =============================================================================
# UTILITIES
# =============================================================================

def separator(char="=", width=120):
    print(char * width)


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            default=str,
        )


def sha256_file(path, chunk_size=1024 * 1024):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def clean_string(series):

    return (
        series
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )


def unordered_pair_key(a, b):

    x, y = sorted(
        [
            str(a),
            str(b),
        ]
    )

    return f"{x}||{y}"


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
        payload.encode("utf-8")
    ).hexdigest()

    return f"edge::{digest}"


def role_memberships(
    entity_id,
    investor_ids,
    startup_ids,
):

    roles = []

    if entity_id in investor_ids:
        roles.append("investor")

    if entity_id in startup_ids:
        roles.append("startup")

    return roles


def typed_relation_key(
    src_type,
    relation,
    dst_type,
):

    return (
        f"{src_type}|"
        f"{relation}|"
        f"{dst_type}"
    )


# =============================================================================
# FOUNDER EDGE MATERIALIZATION
# =============================================================================

def build_founder_edges(
    eligible_founders,
    person_index_map,
):

    direct = eligible_founders.copy()

    direct["src_index"] = (
        direct["startup_node_index"]
        .astype(np.int64)
    )

    direct["dst_index"] = (
        direct["person_id"]
        .map(person_index_map)
        .astype(np.int64)
    )

    direct["src_node_id"] = (
        direct["startup_node_id"]
    )

    direct["dst_node_id"] = (
        "person::"
        + direct["person_id"].astype(str)
    )

    direct["src_type"] = "startup"
    direct["dst_type"] = "person"

    direct["relation"] = "founded_by"

    direct["typed_relation_key"] = (
        "startup|founded_by|person"
    )

    direct["relation_id"] = 0

    direct["underlying_src_id"] = (
        direct["startup_id"]
    )

    direct["underlying_dst_id"] = (
        direct["person_id"]
    )

    direct["first_observed_on"] = (
        direct["focal_founded_on"]
    )

    direct["last_observed_on"] = (
        direct["focal_founded_on"]
    )

    direct["support_count"] = 1

    direct["provenance_source"] = (
        "companies.founder_links->people.link"
    )

    direct["temporal_provenance"] = (
        "startup_founded_on_strictly_pre_origin"
    )

    inverse = direct.copy()

    inverse[
        [
            "src_index",
            "dst_index",
        ]
    ] = inverse[
        [
            "dst_index",
            "src_index",
        ]
    ].to_numpy()

    inverse[
        [
            "src_node_id",
            "dst_node_id",
        ]
    ] = inverse[
        [
            "dst_node_id",
            "src_node_id",
        ]
    ].to_numpy()

    inverse[
        [
            "src_type",
            "dst_type",
        ]
    ] = inverse[
        [
            "dst_type",
            "src_type",
        ]
    ].to_numpy()

    inverse[
        [
            "underlying_src_id",
            "underlying_dst_id",
        ]
    ] = inverse[
        [
            "underlying_dst_id",
            "underlying_src_id",
        ]
    ].to_numpy()

    inverse["relation"] = "founded"

    inverse["typed_relation_key"] = (
        "person|founded|startup"
    )

    inverse["relation_id"] = 1

    columns = [
        "src_index",
        "dst_index",
        "relation_id",
        "src_node_id",
        "dst_node_id",
        "src_type",
        "dst_type",
        "relation",
        "typed_relation_key",
        "underlying_src_id",
        "underlying_dst_id",
        "first_observed_on",
        "last_observed_on",
        "support_count",
        "provenance_source",
        "temporal_provenance",
    ]

    return pd.concat(
        [
            direct[columns],
            inverse[columns],
        ],
        ignore_index=True,
    )


# =============================================================================
# ACQUISITION EDGE MATERIALIZATION
# =============================================================================

def build_acquisition_edges(
    origin_acquisitions,
    role_node_lookup,
    investor_ids,
    startup_ids,
):

    grouped = (
        origin_acquisitions
        .groupby(
            [
                "acquirer_resolved_id",
                "acquiree_resolved_id",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            first_observed_on=(
                "announced_on",
                "min",
            ),
            last_observed_on=(
                "announced_on",
                "max",
            ),
            support_count=(
                "announced_on",
                "size",
            ),
        )
    )

    rows = []

    for row in grouped.itertuples(
        index=False
    ):

        source_id = str(
            row.acquirer_resolved_id
        )

        target_id = str(
            row.acquiree_resolved_id
        )

        source_roles = role_memberships(
            source_id,
            investor_ids,
            startup_ids,
        )

        target_roles = role_memberships(
            target_id,
            investor_ids,
            startup_ids,
        )

        require(
            len(source_roles) > 0,
            (
                "Acquirer lost canonical "
                "role membership."
            ),
        )

        require(
            len(target_roles) > 0,
            (
                "Acquiree lost canonical "
                "role membership."
            ),
        )

        for source_role in source_roles:

            for target_role in target_roles:

                source_key = (
                    source_role,
                    source_id,
                )

                target_key = (
                    target_role,
                    target_id,
                )

                require(
                    source_key
                    in role_node_lookup,
                    (
                        "Missing canonical "
                        "source role node."
                    ),
                )

                require(
                    target_key
                    in role_node_lookup,
                    (
                        "Missing canonical "
                        "target role node."
                    ),
                )

                (
                    source_index,
                    source_node_id,
                ) = role_node_lookup[
                    source_key
                ]

                (
                    target_index,
                    target_node_id,
                ) = role_node_lookup[
                    target_key
                ]

                direct_key = (
                    typed_relation_key(
                        source_role,
                        "acquired",
                        target_role,
                    )
                )

                inverse_key = (
                    typed_relation_key(
                        target_role,
                        "acquired_by",
                        source_role,
                    )
                )

                require(
                    direct_key
                    in RELATION_VOCAB,
                    (
                        "Unknown direct typed "
                        f"relation: {direct_key}"
                    ),
                )

                require(
                    inverse_key
                    in RELATION_VOCAB,
                    (
                        "Unknown inverse typed "
                        f"relation: {inverse_key}"
                    ),
                )

                rows.append(
                    {
                        "src_index":
                            int(source_index),

                        "dst_index":
                            int(target_index),

                        "relation_id":
                            int(
                                RELATION_VOCAB[
                                    direct_key
                                ]
                            ),

                        "src_node_id":
                            source_node_id,

                        "dst_node_id":
                            target_node_id,

                        "src_type":
                            source_role,

                        "dst_type":
                            target_role,

                        "relation":
                            "acquired",

                        "typed_relation_key":
                            direct_key,

                        "underlying_src_id":
                            source_id,

                        "underlying_dst_id":
                            target_id,

                        "first_observed_on":
                            row.first_observed_on,

                        "last_observed_on":
                            row.last_observed_on,

                        "support_count":
                            int(
                                row.support_count
                            ),

                        "provenance_source":
                            "acquisitions",

                        "temporal_provenance":
                            "timestamped_strictly_pre_origin",
                    }
                )

                rows.append(
                    {
                        "src_index":
                            int(target_index),

                        "dst_index":
                            int(source_index),

                        "relation_id":
                            int(
                                RELATION_VOCAB[
                                    inverse_key
                                ]
                            ),

                        "src_node_id":
                            target_node_id,

                        "dst_node_id":
                            source_node_id,

                        "src_type":
                            target_role,

                        "dst_type":
                            source_role,

                        "relation":
                            "acquired_by",

                        "typed_relation_key":
                            inverse_key,

                        "underlying_src_id":
                            target_id,

                        "underlying_dst_id":
                            source_id,

                        "first_observed_on":
                            row.first_observed_on,

                        "last_observed_on":
                            row.last_observed_on,

                        "support_count":
                            int(
                                row.support_count
                            ),

                        "provenance_source":
                            "acquisitions",

                        "temporal_provenance":
                            "timestamped_strictly_pre_origin",
                    }
                )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    separator()

    print(
        "PHASE 8.2.2 — "
        "STRICT INDUCTIVE HETEROGENEOUS "
        "GRAPH MATERIALIZATION V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "GRAPH MATERIALIZATION + LEAKAGE AUDIT"
    )

    print(
        "Feature discovery reopened:  NO"
    )

    print(
        "Evaluation candidate matrix: NO"
    )

    print(
        "Model instantiated:          NO"
    )

    print(
        "Optimizer created:           NO"
    )

    print(
        "Training performed:          NO"
    )

    print(
        "Performance inspected:       NO"
    )

    print(
        "T60 scored:                  NO"
    )

    print(
        "April analyzed:              NO"
    )


    # =========================================================================
    # 8.2.2.1 — CONTRACT GATES
    # =========================================================================

    separator("-")

    print(
        "8.2.2.1 — "
        "PREREQUISITE / CONTRACT GATES"
    )

    separator("-")

    required = [
        GRAPH_SCHEMA,
        CONSISTENCY_CONTRACT,
        ARCHITECTURE_CONTRACT,
        COLD_MASK_CONTRACT,
        EXECUTION_CONTRACT,
        PHASE3_NODE_INDEX,
        ACQUISITION_RESOLVED,
        STARTUP_SOURCE,
        PERSON_SOURCE,
        FOUNDER_EDGES,
        ORIGIN_MASKS,
        REMOVED_STRUCTURAL_EDGES,
        MASK_FREEZE_SUMMARY,
    ]

    for path in required:

        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

        print(
            f"FOUND  {path}"
        )

    graph_contract = read_json(
        GRAPH_SCHEMA
    )

    consistency_contract = read_json(
        CONSISTENCY_CONTRACT
    )

    architecture_contract = read_json(
        ARCHITECTURE_CONTRACT
    )

    cold_contract = read_json(
        COLD_MASK_CONTRACT
    )

    execution_contract = read_json(
        EXECUTION_CONTRACT
    )

    mask_freeze = read_json(
        MASK_FREEZE_SUMMARY
    )

    require(
        graph_contract.get("status")
        ==
        "FROZEN",
        (
            "8.1.9 V2 graph schema "
            "is not FROZEN."
        ),
    )

    require(
        consistency_contract.get(
            "status"
        )
        ==
        "FROZEN",
        (
            "8.1.10A V2 consistency "
            "contract is not FROZEN."
        ),
    )

    require(
        architecture_contract.get(
            "status"
        )
        ==
        "FROZEN",
        (
            "8.1.11 architecture "
            "contract is not FROZEN."
        ),
    )

    require(
        cold_contract.get("status")
        ==
        "FROZEN",
        (
            "8.1.12A cold-mask "
            "contract is not FROZEN."
        ),
    )

    cold_execution = (
        execution_contract[
            "cold_start_execution"
        ]
    )

    require(
        "no evaluation holdout mask"
        in
        cold_execution[
            "acquisition_existence_policy"
        ].lower(),
        (
            "V5 acquisition existence "
            "policy drift."
        ),
    )

    require(
        cold_execution[
            "inference_incoming_messages"
        ]
        is True,
        (
            "V5 incoming-message "
            "policy drift."
        ),
    )

    require(
        cold_execution[
            "inference_outgoing_messages"
        ]
        is False,
        (
            "V5 outgoing-message "
            "policy drift."
        ),
    )

    require(
        graph_contract[
            "structural_safety_rules"
        ][
            "target_pair_direct_edge_masking"
        ]
        ==
        "REUSE_FROZEN_PHASE8_POLICY",
        (
            "Frozen target-pair "
            "policy is not active."
        ),
    )

    require(
        int(
            mask_freeze[
                "direct_structural_pairs_removed"
            ]
        )
        ==
        11,
        (
            "Expected 11 frozen "
            "target-pair keys."
        ),
    )

    contract_relation_vocab = {
        item[
            "typed_relation_key"
        ]:
            int(
                item[
                    "relation_id"
                ]
            )

        for item
        in architecture_contract[
            "typed_relation_vocabulary"
        ]
    }

    require(
        contract_relation_vocab
        ==
        RELATION_VOCAB,
        (
            "Typed relation vocabulary "
            "differs from 8.1.11."
        ),
    )

    print(
        "Graph schema V2:             FROZEN"
    )

    print(
        "Cold contract 8.1.12A:       FROZEN"
    )

    print(
        "Execution contract V5:       ACTIVE"
    )

    print(
        "Typed relation vocabulary:   EXACT 10/10"
    )

    print(
        "Target-pair policy:          REUSE 8.0.3"
    )

    print(
        "Training allowed now:        NO"
    )


    # =========================================================================
    # 8.2.2.2 — GLOBAL NODE INDEX
    # =========================================================================

    separator("-")

    print(
        "8.2.2.2 — "
        "STABLE GLOBAL NODE INDEX"
    )

    separator("-")

    role_nodes = pd.read_parquet(
        PHASE3_NODE_INDEX
    )

    require(
        len(role_nodes)
        ==
        EXPECTED_ROLE_NODE_COUNT,
        (
            "Phase-3 role-node "
            "count drift."
        ),
    )

    require(
        role_nodes[
            "node_index"
        ].min()
        ==
        0,
        (
            "Phase-3 node index "
            "does not begin at zero."
        ),
    )

    require(
        role_nodes[
            "node_index"
        ].max()
        ==
        EXPECTED_ROLE_NODE_COUNT - 1,
        (
            "Phase-3 node-index "
            "boundary drift."
        ),
    )

    require(
        role_nodes[
            "node_index"
        ].is_unique,
        (
            "Phase-3 node indices "
            "are not unique."
        ),
    )

    require(
        int(
            role_nodes[
                "node_type"
            ].eq(
                "investor"
            ).sum()
        )
        ==
        EXPECTED_INVESTOR_COUNT,
        "Investor node-count drift.",
    )

    require(
        int(
            role_nodes[
                "node_type"
            ].eq(
                "startup"
            ).sum()
        )
        ==
        EXPECTED_STARTUP_COUNT,
        "Startup node-count drift.",
    )

    persons = pd.read_parquet(
        PERSON_SOURCE
    )

    require(
        len(persons)
        ==
        EXPECTED_PERSON_COUNT,
        "Person node-count drift.",
    )

    require(
        persons[
            "person_id"
        ].nunique()
        ==
        EXPECTED_PERSON_COUNT,
        "Person IDs are not unique.",
    )

    require(
        persons[
            "person_source_index"
        ].min()
        ==
        0,
        (
            "Person source index "
            "does not begin at zero."
        ),
    )

    require(
        persons[
            "person_source_index"
        ].max()
        ==
        EXPECTED_PERSON_COUNT - 1,
        (
            "Person source-index "
            "boundary drift."
        ),
    )

    require(
        (
            ~persons[
                "identity_trainable"
            ]
        ).all(),
        (
            "At least one Person identity "
            "is marked trainable."
        ),
    )

    require(
        persons[
            "initial_feature_policy"
        ].eq(
            "ZERO_40"
        ).all(),
        (
            "Person initial-feature "
            "policy drift."
        ),
    )

    person_nodes = pd.DataFrame(
        {
            "node_index":
                persons[
                    "person_source_index"
                ].astype(
                    np.int64
                )
                +
                EXPECTED_ROLE_NODE_COUNT,

            "node_id":
                "person::"
                +
                persons[
                    "person_id"
                ].astype(str),

            "node_type":
                "person",

            "raw_entity_id":
                persons[
                    "person_id"
                ].astype(str),

            "display_name":
                persons[
                    "person_url"
                ].astype(
                    "string"
                ),

            "source_registry":
                (
                    "phase_8_2_1b_"
                    "person_node_source_V1"
                ),

            "underlying_entity_has_dual_role":
                False,
        }
    )

    node_index = pd.concat(
        [
            role_nodes[
                [
                    "node_index",
                    "node_id",
                    "node_type",
                    "raw_entity_id",
                    "display_name",
                    "source_registry",
                    "underlying_entity_has_dual_role",
                ]
            ],

            person_nodes,
        ],

        ignore_index=True,
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

    require(
        len(node_index)
        ==
        EXPECTED_FINAL_NODE_COUNT,
        "Final node-count drift.",
    )

    require(
        node_index[
            "node_index"
        ].is_unique,
        (
            "Final node indices "
            "are not unique."
        ),
    )

    require(
        node_index[
            "node_id"
        ].is_unique,
        (
            "Final node IDs "
            "are not unique."
        ),
    )

    require(
        node_index[
            "node_index"
        ].min()
        ==
        0
        and
        node_index[
            "node_index"
        ].max()
        ==
        EXPECTED_FINAL_NODE_COUNT - 1,
        (
            "Final node index "
            "is not dense."
        ),
    )

    global_node_path = (
        OUT_DIR
        / "phase_8_2_2_global_node_index_V1.parquet"
    )

    node_index.to_parquet(
        global_node_path,
        index=False,
    )

    print(
        f"Investor nodes:              "
        f"{EXPECTED_INVESTOR_COUNT:,}"
    )

    print(
        f"Startup nodes:               "
        f"{EXPECTED_STARTUP_COUNT:,}"
    )

    print(
        f"Person nodes:                "
        f"{EXPECTED_PERSON_COUNT:,}"
    )

    print(
        f"Total nodes:                 "
        f"{EXPECTED_FINAL_NODE_COUNT:,}"
    )

    print(
        f"Person index range:          "
        f"{EXPECTED_ROLE_NODE_COUNT:,}"
        f".."
        f"{EXPECTED_FINAL_NODE_COUNT - 1:,}"
    )


    # =========================================================================
    # 8.2.2.3 — ROLE LOOKUPS + TARGET-PAIR OVERLAY
    # =========================================================================

    separator("-")

    print(
        "8.2.2.3 — "
        "ROLE LOOKUPS + FROZEN TARGET-PAIR OVERLAY"
    )

    separator("-")

    role_nodes_work = (
        role_nodes.copy()
    )

    role_nodes_work[
        "raw_entity_id"
    ] = clean_string(
        role_nodes_work[
            "raw_entity_id"
        ]
    )

    investor_ids = set(
        role_nodes_work.loc[
            role_nodes_work[
                "node_type"
            ].eq(
                "investor"
            ),
            "raw_entity_id",
        ]
        .dropna()
        .astype(str)
    )

    startup_ids = set(
        role_nodes_work.loc[
            role_nodes_work[
                "node_type"
            ].eq(
                "startup"
            ),
            "raw_entity_id",
        ]
        .dropna()
        .astype(str)
    )

    canonical_role_ids = (
        investor_ids
        |
        startup_ids
    )

    role_node_lookup = {}

    for row in role_nodes_work.itertuples(
        index=False
    ):

        key = (
            str(
                row.node_type
            ),
            str(
                row.raw_entity_id
            ),
        )

        require(
            key
            not in role_node_lookup,
            (
                "Duplicate role-node "
                f"key: {key}"
            ),
        )

        role_node_lookup[
            key
        ] = (
            int(
                row.node_index
            ),
            str(
                row.node_id
            ),
        )

    person_index_map = dict(
        zip(
            persons[
                "person_id"
            ].astype(str),

            person_nodes[
                "node_index"
            ].astype(
                np.int64
            ),
        )
    )

    removed_edges = pd.read_csv(
        REMOVED_STRUCTURAL_EDGES
    )

    require(
        len(removed_edges)
        ==
        40,
        (
            "Expected 40 frozen "
            "8.0.3 removed edge records."
        ),
    )

    target_pair_keys = set(
        removed_edges[
            "underlying_pair_key"
        ]
        .dropna()
        .astype(str)
    )

    require(
        len(target_pair_keys)
        ==
        11,
        (
            "Expected 11 frozen "
            "direct target pairs."
        ),
    )

    acquisition_legacy_pairs = set(
        removed_edges.loc[
            removed_edges[
                "provenance_source"
            ].eq(
                "acquisitions"
            ),
            "underlying_pair_key",
        ]
        .dropna()
        .astype(str)
    )

    require(
        len(
            acquisition_legacy_pairs
        )
        ==
        9,
        (
            "Expected 9 acquisition-bearing "
            "pairs in the frozen 8.0.3 audit."
        ),
    )

    print(
        f"Frozen direct-pair keys:     "
        f"{len(target_pair_keys):,}"
    )

    print(
        f"Legacy acquisition pairs:    "
        f"{len(acquisition_legacy_pairs):,}"
    )

    print(
        "Overlay unit:                "
        "UNORDERED UNDERLYING PAIR"
    )

    print(
        "Overlay applied to graph:    "
        "ALL 11 FROZEN KEYS"
    )


    # =========================================================================
    # 8.2.2.4 — STARTUP / FOUNDER INDEX INTEGRITY
    # =========================================================================

    separator("-")

    print(
        "8.2.2.4 — "
        "STARTUP / FOUNDER INDEX INTEGRITY"
    )

    separator("-")

    startup_source = pd.read_parquet(
        STARTUP_SOURCE,
        columns=[
            "startup_node_index",
            "startup_node_id",
            "startup_id",
            "founded_on",
        ],
    )

    startup_source[
        "startup_id"
    ] = clean_string(
        startup_source[
            "startup_id"
        ]
    )

    startup_role_index = (
        role_nodes_work.loc[
            role_nodes_work[
                "node_type"
            ].eq(
                "startup"
            ),
            [
                "node_index",
                "node_id",
                "raw_entity_id",
            ],
        ]
        .rename(
            columns={
                "node_index":
                    "expected_node_index",

                "node_id":
                    "expected_node_id",

                "raw_entity_id":
                    "startup_id",
            }
        )
    )

    startup_alignment = (
        startup_source
        .merge(
            startup_role_index,
            on="startup_id",
            how="left",
            validate="one_to_one",
        )
    )

    require(
        startup_alignment[
            "expected_node_index"
        ].notna().all(),
        (
            "Canonical Startup source "
            "contains unknown role nodes."
        ),
    )

    require(
        (
            startup_alignment[
                "startup_node_index"
            ].astype(
                np.int64
            )
            ==
            startup_alignment[
                "expected_node_index"
            ].astype(
                np.int64
            )
        ).all(),
        (
            "Startup node-index alignment "
            "differs from frozen Phase-3 index."
        ),
    )

    require(
        (
            startup_alignment[
                "startup_node_id"
            ].astype(str)
            ==
            startup_alignment[
                "expected_node_id"
            ].astype(str)
        ).all(),
        (
            "Startup node-ID alignment "
            "differs from frozen Phase-3 index."
        ),
    )

    founders = pd.read_parquet(
        FOUNDER_EDGES
    )

    require(
        len(founders)
        ==
        EXPECTED_FOUNDER_EDGE_COUNT,
        (
            "Startup-founder "
            "edge-count drift."
        ),
    )

    founders[
        "startup_id"
    ] = clean_string(
        founders[
            "startup_id"
        ]
    )

    founders[
        "person_id"
    ] = clean_string(
        founders[
            "person_id"
        ]
    )

    founders[
        "focal_founded_on"
    ] = pd.to_datetime(
        founders[
            "focal_founded_on"
        ],
        errors="coerce",
    )

    require(
        founders[
            [
                "startup_id",
                "person_id",
            ]
        ]
        .drop_duplicates()
        .shape[0]
        ==
        len(founders),
        (
            "Duplicate Startup-Person "
            "founder edges detected."
        ),
    )

    require(
        founders[
            "person_id"
        ].isin(
            set(
                person_index_map
            )
        ).all(),
        (
            "Founder edge references "
            "unknown Person."
        ),
    )

    require(
        founders[
            "startup_id"
        ].isin(
            startup_ids
        ).all(),
        (
            "Founder edge references "
            "unknown Startup."
        ),
    )

    founder_expected_index = (
        founders[
            "startup_id"
        ].map(
            dict(
                zip(
                    startup_role_index[
                        "startup_id"
                    ].astype(str),

                    startup_role_index[
                        "expected_node_index"
                    ].astype(
                        np.int64
                    ),
                )
            )
        )
    )

    require(
        (
            founders[
                "startup_node_index"
            ].astype(
                np.int64
            )
            ==
            founder_expected_index.astype(
                np.int64
            )
        ).all(),
        (
            "Founder Startup indices "
            "do not match frozen role index."
        ),
    )

    print(
        "Canonical Startup index:     PASS"
    )

    print(
        f"Startup-founder pairs:       "
        f"{len(founders):,}"
    )

    print(
        f"Unique founder Persons:      "
        f"{founders['person_id'].nunique():,}"
    )


    # =========================================================================
    # 8.2.2.5 — ACQUISITION SUBSTRATE
    # =========================================================================

    separator("-")

    print(
        "8.2.2.5 — "
        "AUTHORITATIVE ACQUISITION SUBSTRATE"
    )

    separator("-")

    acquisitions = pd.read_parquet(
        ACQUISITION_RESOLVED,
        columns=[
            "announced_on",
            "acquirer_resolved_id",
            "acquiree_resolved_id",
            "acquirer_resolution_status",
            "acquiree_resolution_status",
        ],
    )

    acquisitions[
        "announced_on"
    ] = pd.to_datetime(
        acquisitions[
            "announced_on"
        ],
        errors="coerce",
    )

    acquisitions[
        "acquirer_resolved_id"
    ] = clean_string(
        acquisitions[
            "acquirer_resolved_id"
        ]
    )

    acquisitions[
        "acquiree_resolved_id"
    ] = clean_string(
        acquisitions[
            "acquiree_resolved_id"
        ]
    )

    accepted = acquisitions.loc[
        acquisitions[
            "announced_on"
        ].notna()
        &
        acquisitions[
            "acquirer_resolution_status"
        ].isin(
            ACCEPTED_ENDPOINT_STATUSES
        )
        &
        acquisitions[
            "acquiree_resolution_status"
        ].isin(
            ACCEPTED_ENDPOINT_STATUSES
        )
        &
        acquisitions[
            "acquirer_resolved_id"
        ].isin(
            canonical_role_ids
        )
        &
        acquisitions[
            "acquiree_resolved_id"
        ].isin(
            canonical_role_ids
        )
        &
        (
            acquisitions[
                "acquirer_resolved_id"
            ]
            !=
            acquisitions[
                "acquiree_resolved_id"
            ]
        )
    ].copy()

    accepted[
        "underlying_pair_key"
    ] = [
        unordered_pair_key(
            a,
            b,
        )

        for a, b
        in zip(
            accepted[
                "acquirer_resolved_id"
            ],

            accepted[
                "acquiree_resolved_id"
            ],
        )
    ]

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # Reconstruct existence evidence BEFORE applying the 8.0.3 target-pair
    # graph overlay.
    #
    # V5 explicitly freezes acquisition existence as:
    #
    #   conservative endpoint resolution
    #   + canonical endpoints
    #   + no underlying self relation
    #   + NO evaluation holdout mask
    #
    # The 8.0.3 overlay is therefore a graph-leakage control only.
    # -------------------------------------------------------------------------

    acquirer_startup_evidence = (
        accepted.loc[
            accepted[
                "acquirer_resolved_id"
            ].isin(
                startup_ids
            ),
            [
                "acquirer_resolved_id",
                "announced_on",
            ],
        ]
        .rename(
            columns={
                "acquirer_resolved_id":
                    "startup_id",
            }
        )
    )

    acquiree_startup_evidence = (
        accepted.loc[
            accepted[
                "acquiree_resolved_id"
            ].isin(
                startup_ids
            ),
            [
                "acquiree_resolved_id",
                "announced_on",
            ],
        ]
        .rename(
            columns={
                "acquiree_resolved_id":
                    "startup_id",
            }
        )
    )

    reconstructed_first_acquisition = (
        pd.concat(
            [
                acquirer_startup_evidence,
                acquiree_startup_evidence,
            ],
            ignore_index=True,
        )
        .groupby(
            "startup_id",
            observed=True,
        )[
            "announced_on"
        ]
        .min()
        .rename(
            "reconstructed_first_acquisition_on"
        )
    )

    origin_masks = pd.read_parquet(
        ORIGIN_MASKS
    )

    origin_masks[
        "startup_id"
    ] = clean_string(
        origin_masks[
            "startup_id"
        ]
    )

    origin_masks[
        "prediction_origin"
    ] = pd.to_datetime(
        origin_masks[
            "prediction_origin"
        ],
        errors="raise",
    )

    origin_masks[
        "first_acquisition_on"
    ] = pd.to_datetime(
        origin_masks[
            "first_acquisition_on"
        ],
        errors="coerce",
    )

    authoritative_first_acquisition = (
        origin_masks[
            [
                "startup_id",
                "first_acquisition_on",
            ]
        ]
        .drop_duplicates()
        .set_index(
            "startup_id"
        )[
            "first_acquisition_on"
        ]
    )

    comparison_ids = sorted(
        set(
            authoritative_first_acquisition
            .index
            .astype(str)
        )
        |
        set(
            reconstructed_first_acquisition
            .index
            .astype(str)
        )
    )

    lhs = (
        authoritative_first_acquisition
        .reindex(
            comparison_ids
        )
    )

    rhs = (
        reconstructed_first_acquisition
        .reindex(
            comparison_ids
        )
    )

    lhs_values = lhs.to_numpy(
        dtype="datetime64[ns]"
    )

    rhs_values = rhs.to_numpy(
        dtype="datetime64[ns]"
    )

    first_acquisition_agreement = (
        (
            pd.isna(
                lhs_values
            )
            &
            pd.isna(
                rhs_values
            )
        )
        |
        (
            lhs_values
            ==
            rhs_values
        )
    )

    require(
        bool(
            np.all(
                first_acquisition_agreement
            )
        ),
        (
            "Reconstructed acquisition "
            "existence differs from "
            "authoritative 8.2.1C V2."
        ),
    )

    print(
        f"Resolved acquisition rows:   "
        f"{len(acquisitions):,}"
    )

    print(
        f"Strict accepted rows:        "
        f"{len(accepted):,}"
    )

    print(
        "8.2.1C existence agreement:  PASS"
    )

    print(
        "Existence holdout mask:      NO"
    )

    print(
        "Graph target-pair overlay:   YES"
    )


    # =========================================================================
    # 8.2.2.6 — ORIGIN-SPECIFIC GRAPH MATERIALIZATION
    # =========================================================================

    separator("-")

    print(
        "8.2.2.6 — "
        "ORIGIN-SPECIFIC GRAPH MATERIALIZATION"
    )

    separator("-")

    summary_rows = []

    output_paths = [
        global_node_path
    ]

    for origin_label in ORIGIN_ORDER:

        separator(".")

        print(
            origin_label
        )

        separator(".")

        origin_mask = (
            origin_masks.loc[
                origin_masks[
                    "origin_label"
                ].eq(
                    origin_label
                )
            ]
            .copy()
        )

        require(
            len(origin_mask)
            ==
            EXPECTED_STARTUP_COUNT,
            (
                f"{origin_label}: "
                "origin mask does not contain "
                "all Startup-role nodes."
            ),
        )

        require(
            origin_mask[
                "startup_id"
            ].nunique()
            ==
            EXPECTED_STARTUP_COUNT,
            (
                f"{origin_label}: "
                "duplicate/missing Startup masks."
            ),
        )

        require(
            origin_mask[
                "prediction_origin"
            ].nunique()
            ==
            1,
            (
                f"{origin_label}: "
                "multiple prediction origins."
            ),
        )

        prediction_origin = pd.Timestamp(
            origin_mask[
                "prediction_origin"
            ].iloc[0]
        )

        eligible_startups = set(
            origin_mask.loc[
                origin_mask[
                    "origin_eligible"
                ],
                "startup_id",
            ]
            .astype(str)
        )

        cold_startups = set(
            origin_mask.loc[
                origin_mask[
                    "investment_cold"
                ],
                "startup_id",
            ]
            .astype(str)
        )

        warm_startups = set(
            origin_mask.loc[
                origin_mask[
                    "investment_warm"
                ],
                "startup_id",
            ]
            .astype(str)
        )

        require(
            cold_startups.isdisjoint(
                warm_startups
            ),
            (
                f"{origin_label}: "
                "cold/warm overlap."
            ),
        )

        require(
            cold_startups.issubset(
                eligible_startups
            ),
            (
                f"{origin_label}: "
                "cold Startup outside "
                "origin-eligible universe."
            ),
        )


        # ---------------------------------------------------------------------
        # Founder graph:
        # startup.founded_on < prediction_origin
        # ---------------------------------------------------------------------

        founder_origin = founders.loc[
            founders[
                "focal_founded_on"
            ].notna()
            &
            (
                founders[
                    "focal_founded_on"
                ]
                <
                prediction_origin
            )
        ].copy()

        require(
            founder_origin[
                "startup_id"
            ].isin(
                eligible_startups
            ).all(),
            (
                f"{origin_label}: "
                "temporally eligible founder "
                "edge points to a non-eligible "
                "Startup."
            ),
        )

        founder_edge_table = (
            build_founder_edges(
                founder_origin,
                person_index_map,
            )
        )


        # ---------------------------------------------------------------------
        # Acquisition graph:
        #
        # 1. accepted endpoint evidence
        # 2. both canonical endpoints
        # 3. no underlying self relation
        # 4. announced_on < prediction_origin
        # 5. apply frozen 8.0.3 underlying-pair overlay
        # 6. role projection
        # 7. explicit inverse
        # ---------------------------------------------------------------------

        acquisition_pre_overlay = (
            accepted.loc[
                accepted[
                    "announced_on"
                ]
                <
                prediction_origin
            ]
            .copy()
        )

        masked_acquisition_event_rows = int(
            acquisition_pre_overlay[
                "underlying_pair_key"
            ].isin(
                target_pair_keys
            ).sum()
        )

        acquisition_origin = (
            acquisition_pre_overlay.loc[
                ~acquisition_pre_overlay[
                    "underlying_pair_key"
                ].isin(
                    target_pair_keys
                )
            ]
            .copy()
        )

        acquisition_edge_table = (
            build_acquisition_edges(
                acquisition_origin,
                role_node_lookup,
                investor_ids,
                startup_ids,
            )
        )


        # ---------------------------------------------------------------------
        # Base origin-safe graph
        # ---------------------------------------------------------------------

        edges = pd.concat(
            [
                founder_edge_table,
                acquisition_edge_table,
            ],
            ignore_index=True,
        )

        edges[
            "edge_id"
        ] = [
            make_edge_id(
                relation,
                src_node_id,
                dst_node_id,
            )

            for (
                relation,
                src_node_id,
                dst_node_id,
            )
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

        require(
            edges[
                "edge_id"
            ].is_unique,
            (
                f"{origin_label}: "
                "duplicate structural edge IDs."
            ),
        )

        require(
            edges[
                "typed_relation_key"
            ].map(
                RELATION_VOCAB
            ).notna().all(),
            (
                f"{origin_label}: "
                "unknown typed relation."
            ),
        )

        require(
            (
                edges[
                    "typed_relation_key"
                ]
                .map(
                    RELATION_VOCAB
                )
                .astype(
                    np.int64
                )
                ==
                edges[
                    "relation_id"
                ].astype(
                    np.int64
                )
            ).all(),
            (
                f"{origin_label}: "
                "relation-ID mismatch."
            ),
        )

        require(
            edges[
                "src_index"
            ].between(
                0,
                EXPECTED_FINAL_NODE_COUNT - 1,
            ).all(),
            (
                f"{origin_label}: "
                "source index outside "
                "global node range."
            ),
        )

        require(
            edges[
                "dst_index"
            ].between(
                0,
                EXPECTED_FINAL_NODE_COUNT - 1,
            ).all(),
            (
                f"{origin_label}: "
                "destination index outside "
                "global node range."
            ),
        )


        # ---------------------------------------------------------------------
        # Verify frozen direct target-pair exclusion.
        # ---------------------------------------------------------------------

        acquisition_edges_only = (
            edges.loc[
                edges[
                    "provenance_source"
                ].eq(
                    "acquisitions"
                )
            ]
        )

        acquisition_graph_pair_keys = {
            unordered_pair_key(
                source_id,
                target_id,
            )

            for (
                source_id,
                target_id,
            )
            in zip(
                acquisition_edges_only[
                    "underlying_src_id"
                ],

                acquisition_edges_only[
                    "underlying_dst_id"
                ],
            )
        }

        target_pair_overlap = (
            acquisition_graph_pair_keys
            &
            target_pair_keys
        )

        require(
            len(
                target_pair_overlap
            )
            ==
            0,
            (
                f"{origin_label}: "
                "frozen direct target-pair "
                "overlap remains."
            ),
        )


        # ---------------------------------------------------------------------
        # Verify every Startup endpoint is origin-eligible.
        # ---------------------------------------------------------------------

        startup_source_ids = set(
            edges.loc[
                edges[
                    "src_type"
                ].eq(
                    "startup"
                ),
                "underlying_src_id",
            ]
            .astype(str)
        )

        startup_destination_ids = set(
            edges.loc[
                edges[
                    "dst_type"
                ].eq(
                    "startup"
                ),
                "underlying_dst_id",
            ]
            .astype(str)
        )

        require(
            startup_source_ids.issubset(
                eligible_startups
            ),
            (
                f"{origin_label}: "
                "source Startup outside "
                "origin-eligible universe."
            ),
        )

        require(
            startup_destination_ids.issubset(
                eligible_startups
            ),
            (
                f"{origin_label}: "
                "destination Startup outside "
                "origin-eligible universe."
            ),
        )


        # ---------------------------------------------------------------------
        # Deterministic edge ordering.
        # ---------------------------------------------------------------------

        edges = (
            edges
            .sort_values(
                [
                    "relation_id",
                    "src_index",
                    "dst_index",
                    "edge_id",
                ],
                kind="mergesort",
            )
            .reset_index(
                drop=True
            )
        )


        # ---------------------------------------------------------------------
        # COLD_t ROLE-NODE MASKING
        #
        # The cold-start population is defined over Startup-role nodes.
        #
        # If one UUID also has an Investor role, the Investor-role node is
        # NOT masked merely because the corresponding Startup-role node is cold.
        # ---------------------------------------------------------------------

        source_is_cold_startup = (
            edges[
                "src_type"
            ].eq(
                "startup"
            )
            &
            edges[
                "underlying_src_id"
            ].isin(
                cold_startups
            )
        )

        destination_is_cold_startup = (
            edges[
                "dst_type"
            ].eq(
                "startup"
            )
            &
            edges[
                "underlying_dst_id"
            ].isin(
                cold_startups
            )
        )


        # FIT:
        # remove ALL founder/acquisition edges incident to COLD_t Startup.
        fit_edge_mask = ~(
            source_is_cold_startup
            |
            destination_is_cold_startup
        )


        # INFERENCE:
        # COLD_t is sink-only.
        #
        # incoming -> allowed
        # outgoing -> suppressed
        inference_edge_mask = (
            ~source_is_cold_startup
        )


        require(
            not bool(
                (
                    fit_edge_mask
                    &
                    (
                        source_is_cold_startup
                        |
                        destination_is_cold_startup
                    )
                ).any()
            ),
            (
                f"{origin_label}: "
                "cold incident edge survives "
                "fit masking."
            ),
        )

        require(
            not bool(
                (
                    inference_edge_mask
                    &
                    source_is_cold_startup
                ).any()
            ),
            (
                f"{origin_label}: "
                "outgoing COLD_t message "
                "survives inference masking."
            ),
        )

        require(
            bool(
                inference_edge_mask.loc[
                    destination_is_cold_startup
                ].all()
            ),
            (
                f"{origin_label}: "
                "incoming COLD_t edge was "
                "not restorable."
            ),
        )


        # ---------------------------------------------------------------------
        # Tensor export.
        # ---------------------------------------------------------------------

        edge_index = np.vstack(
            [
                edges[
                    "src_index"
                ].to_numpy(
                    dtype=np.int64
                ),

                edges[
                    "dst_index"
                ].to_numpy(
                    dtype=np.int64
                ),
            ]
        )

        edge_type = (
            edges[
                "relation_id"
            ].to_numpy(
                dtype=np.int64
            )
        )

        fit_edge_mask_array = (
            fit_edge_mask
            .to_numpy(
                dtype=np.bool_
            )
        )

        inference_edge_mask_array = (
            inference_edge_mask
            .to_numpy(
                dtype=np.bool_
            )
        )

        prefix = (
            "phase_8_2_2_"
            +
            origin_label.lower()
            +
            "_V1"
        )

        edge_manifest_path = (
            OUT_DIR
            / f"{prefix}_edge_manifest.parquet"
        )

        edge_index_path = (
            OUT_DIR
            / f"{prefix}_edge_index.npy"
        )

        edge_type_path = (
            OUT_DIR
            / f"{prefix}_edge_type.npy"
        )

        fit_mask_path = (
            OUT_DIR
            / f"{prefix}_fit_edge_mask.npy"
        )

        inference_mask_path = (
            OUT_DIR
            / f"{prefix}_inference_edge_mask.npy"
        )

        node_state_path = (
            OUT_DIR
            / f"{prefix}_node_state.parquet"
        )

        relation_summary_path = (
            OUT_DIR
            / f"{prefix}_relation_summary.csv"
        )

        edges.to_parquet(
            edge_manifest_path,
            index=False,
        )

        np.save(
            edge_index_path,
            edge_index,
            allow_pickle=False,
        )

        np.save(
            edge_type_path,
            edge_type,
            allow_pickle=False,
        )

        np.save(
            fit_mask_path,
            fit_edge_mask_array,
            allow_pickle=False,
        )

        np.save(
            inference_mask_path,
            inference_edge_mask_array,
            allow_pickle=False,
        )


        # ---------------------------------------------------------------------
        # Origin-specific node state.
        #
        # Node indices are NEVER reassigned.
        # Person nodes are retained even if isolated.
        # ---------------------------------------------------------------------

        node_state = (
            node_index[
                [
                    "node_index",
                    "node_id",
                    "node_type",
                    "raw_entity_id",
                ]
            ]
            .copy()
        )

        startup_state = (
            origin_mask[
                [
                    "startup_id",
                    "origin_eligible",
                    "investment_cold",
                    "investment_warm",
                    "fit_startup_features_visible",
                    "fit_startup_structural_edges_visible",
                    "inference_startup_features_restorable",
                    "cold_inference_sink_only",
                ]
            ]
            .copy()
        )

        node_state = (
            node_state
            .merge(
                startup_state,
                left_on="raw_entity_id",
                right_on="startup_id",
                how="left",
                validate="many_to_one",
            )
        )

        non_startup_rows = (
            ~node_state[
                "node_type"
            ].eq(
                "startup"
            )
        )

        startup_state_columns = [
            "origin_eligible",
            "investment_cold",
            "investment_warm",
            "fit_startup_features_visible",
            "fit_startup_structural_edges_visible",
            "inference_startup_features_restorable",
            "cold_inference_sink_only",
        ]

        for column in startup_state_columns:

            node_state[
                column
            ] = node_state[
                column
            ].astype(
                "boolean"
            )

            node_state.loc[
                non_startup_rows,
                column,
            ] = pd.NA

        node_state.drop(
            columns=[
                "startup_id"
            ],
            inplace=True,
        )

        node_state[
            "origin_label"
        ] = origin_label

        node_state[
            "prediction_origin"
        ] = prediction_origin

        node_state[
            "node_row_retained"
        ] = True

        require(
            len(node_state)
            ==
            EXPECTED_FINAL_NODE_COUNT,
            (
                f"{origin_label}: "
                "node-state row-count drift."
            ),
        )

        require(
            node_state[
                "node_index"
            ].is_unique,
            (
                f"{origin_label}: "
                "node-state indices duplicated."
            ),
        )

        node_state.to_parquet(
            node_state_path,
            index=False,
        )


        # ---------------------------------------------------------------------
        # Relation summary.
        # ---------------------------------------------------------------------

        relation_summary = (
            edges
            .groupby(
                [
                    "relation_id",
                    "typed_relation_key",
                    "relation",
                ],
                observed=True,
            )
            .agg(
                base_edges=(
                    "edge_id",
                    "size",
                )
            )
            .reset_index()
        )

        fit_counts = (
            edges.loc[
                fit_edge_mask_array
            ]
            .groupby(
                "relation_id",
                observed=True,
            )
            .size()
            .rename(
                "fit_edges"
            )
        )

        inference_counts = (
            edges.loc[
                inference_edge_mask_array
            ]
            .groupby(
                "relation_id",
                observed=True,
            )
            .size()
            .rename(
                "inference_edges"
            )
        )

        relation_summary = (
            relation_summary
            .merge(
                fit_counts,
                on="relation_id",
                how="left",
            )
            .merge(
                inference_counts,
                on="relation_id",
                how="left",
            )
            .fillna(
                {
                    "fit_edges": 0,
                    "inference_edges": 0,
                }
            )
        )

        relation_summary[
            [
                "fit_edges",
                "inference_edges",
            ]
        ] = relation_summary[
            [
                "fit_edges",
                "inference_edges",
            ]
        ].astype(
            np.int64
        )

        relation_summary.to_csv(
            relation_summary_path,
            index=False,
        )


        output_paths.extend(
            [
                edge_manifest_path,
                edge_index_path,
                edge_type_path,
                fit_mask_path,
                inference_mask_path,
                node_state_path,
                relation_summary_path,
            ]
        )


        # ---------------------------------------------------------------------
        # Origin summary.
        # ---------------------------------------------------------------------

        inference_restored_incoming = int(
            (
                inference_edge_mask
                &
                destination_is_cold_startup
            ).sum()
        )

        summary_rows.append(
            {
                "origin_label":
                    origin_label,

                "prediction_origin":
                    prediction_origin,

                "eligible_startups":
                    len(
                        eligible_startups
                    ),

                "cold_startups":
                    len(
                        cold_startups
                    ),

                "warm_startups":
                    len(
                        warm_startups
                    ),

                "founder_direct_pairs":
                    len(
                        founder_origin
                    ),

                "founder_directed_edges":
                    len(
                        founder_edge_table
                    ),

                "accepted_acquisition_event_rows_pre_origin":
                    len(
                        acquisition_pre_overlay
                    ),

                "target_pair_masked_acquisition_event_rows":
                    masked_acquisition_event_rows,

                "acquisition_event_rows_after_overlay":
                    len(
                        acquisition_origin
                    ),

                "acquisition_directed_role_edges":
                    len(
                        acquisition_edge_table
                    ),

                "base_graph_edges":
                    len(
                        edges
                    ),

                "fit_graph_edges":
                    int(
                        fit_edge_mask.sum()
                    ),

                "inference_graph_edges":
                    int(
                        inference_edge_mask.sum()
                    ),

                "fit_removed_cold_incident_edges":
                    int(
                        (
                            ~fit_edge_mask
                        ).sum()
                    ),

                "inference_suppressed_cold_outgoing_edges":
                    int(
                        (
                            ~inference_edge_mask
                        ).sum()
                    ),

                "inference_restored_incoming_to_cold":
                    inference_restored_incoming,

                "target_pair_overlap_after_overlay":
                    len(
                        target_pair_overlap
                    ),
            }
        )

        print(
            f"Prediction origin:           "
            f"{prediction_origin}"
        )

        print(
            f"Eligible / cold / warm:      "
            f"{len(eligible_startups):,} / "
            f"{len(cold_startups):,} / "
            f"{len(warm_startups):,}"
        )

        print(
            f"Founder directed edges:      "
            f"{len(founder_edge_table):,}"
        )

        print(
            f"Acquisition directed edges:  "
            f"{len(acquisition_edge_table):,}"
        )

        print(
            f"Base graph edges:            "
            f"{len(edges):,}"
        )

        print(
            f"Fit graph edges:             "
            f"{int(fit_edge_mask.sum()):,}"
        )

        print(
            f"Inference graph edges:       "
            f"{int(inference_edge_mask.sum()):,}"
        )

        print(
            f"Target-pair overlap:         "
            f"{len(target_pair_overlap)}"
        )

        print(
            "COLD fit incident edges:     MASKED"
        )

        print(
            "COLD inference outgoing:     SUPPRESSED"
        )


    # =========================================================================
    # 8.2.2.7 — CROSS-ORIGIN AUDIT
    # =========================================================================

    separator("-")

    print(
        "8.2.2.7 — "
        "CROSS-ORIGIN AUDIT"
    )

    separator("-")

    summary = pd.DataFrame(
        summary_rows
    )

    require(
        summary[
            "founder_directed_edges"
        ].is_monotonic_increasing,
        (
            "Founder edge availability "
            "is not origin-monotone."
        ),
    )

    require(
        summary[
            "accepted_acquisition_event_rows_pre_origin"
        ].is_monotonic_increasing,
        (
            "Acquisition event availability "
            "is not origin-monotone."
        ),
    )

    require(
        summary[
            "acquisition_directed_role_edges"
        ].is_monotonic_increasing,
        (
            "Acquisition role-edge availability "
            "is not origin-monotone."
        ),
    )

    require(
        (
            summary[
                "target_pair_overlap_after_overlay"
            ]
            ==
            0
        ).all(),
        (
            "Frozen direct target-pair "
            "overlap remains."
        ),
    )

    summary_path = (
        OUT_DIR
        / "phase_8_2_2_origin_graph_summary_V1.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    output_paths.append(
        summary_path
    )

    print(
        summary.to_string(
            index=False
        )
    )


    # =========================================================================
    # 8.2.2.8 — MANIFEST + HASH REGISTRY
    # =========================================================================

    separator("-")

    print(
        "8.2.2.8 — "
        "MANIFEST + HASH REGISTRY"
    )

    separator("-")

    authoritative_inputs = {
        str(path):
            sha256_file(
                path
            )

        for path in [
            GRAPH_SCHEMA,
            CONSISTENCY_CONTRACT,
            ARCHITECTURE_CONTRACT,
            COLD_MASK_CONTRACT,
            EXECUTION_CONTRACT,
            PHASE3_NODE_INDEX,
            ACQUISITION_RESOLVED,
            STARTUP_SOURCE,
            PERSON_SOURCE,
            FOUNDER_EDGES,
            ORIGIN_MASKS,
            REMOVED_STRUCTURAL_EDGES,
            MASK_FREEZE_SUMMARY,
        ]
    }

    manifest = {
        "phase":
            "8.2.2",

        "version":
            "V1",

        "status":
            "MATERIALIZED_AND_AUDITED",

        "scientific_role":
            (
                "STRICT_INDUCTIVE_"
                "HETEROGENEOUS_GRAPH_MATERIALIZATION"
            ),

        "authoritative_inputs":
            authoritative_inputs,

        "node_policy": {
            "role_node_indices_preserved":
                True,

            "person_nodes_appended_after_role_nodes":
                True,

            "investor_nodes":
                EXPECTED_INVESTOR_COUNT,

            "startup_nodes":
                EXPECTED_STARTUP_COUNT,

            "person_nodes":
                EXPECTED_PERSON_COUNT,

            "total_nodes":
                EXPECTED_FINAL_NODE_COUNT,

            "startup_identity_embedding":
                False,

            "person_identity_embedding":
                False,

            "person_initial_feature":
                "ZERO_40",
        },

        "founder_policy": {
            "source":
                str(
                    FOUNDER_EDGES
                ),

            "temporal_gate":
                (
                    "startup.founded_on "
                    "< prediction_origin"
                ),

            "relations": [
                "startup|founded_by|person",
                "person|founded|startup",
            ],
        },

        "acquisition_policy": {
            "source":
                str(
                    ACQUISITION_RESOLVED
                ),

            "accepted_endpoint_statuses":
                sorted(
                    ACCEPTED_ENDPOINT_STATUSES
                ),

            "both_endpoints_in_canonical_role_universe":
                True,

            "underlying_self_relations_removed":
                True,

            "temporal_gate":
                (
                    "announced_on "
                    "< prediction_origin"
                ),

            "existence_evidence_uses_target_pair_mask":
                False,

            "graph_uses_frozen_target_pair_overlay":
                True,

            "frozen_target_pair_count":
                len(
                    target_pair_keys
                ),

            "target_pair_overlay_source":
                str(
                    REMOVED_STRUCTURAL_EDGES
                ),
        },

        "cold_start_graph_policy": {
            "population":
                (
                    "ALL origin-eligible "
                    "investment-cold Startup-role nodes"
                ),

            "fit":
                (
                    "mask every founded_by/founded/"
                    "acquired/acquired_by edge incident "
                    "to COLD_t Startup role nodes"
                ),

            "inference":
                (
                    "restore incoming origin-eligible "
                    "messages only; suppress every edge "
                    "sourced from COLD_t Startup role nodes"
                ),

            "node_rows_retained":
                True,

            "parameters_updated_after_restore":
                False,

            "evaluation_membership_used_to_define_cold":
                False,
        },

        "relation_vocabulary":
            RELATION_VOCAB,

        "origins":
            summary.to_dict(
                orient="records"
            ),

        "model_instantiated":
            False,

        "optimizer_created":
            False,

        "training_performed":
            False,

        "performance_inspected":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,

        "training_allowed_after_phase":
            False,

        "next_step":
            (
                "Phase 8.2.3 — "
                "Phase-8 candidate model implementation"
            ),
    }

    manifest_path = (
        OUT_DIR
        / "phase_8_2_2_manifest_V1.json"
    )

    write_json(
        manifest_path,
        manifest,
    )

    output_paths.append(
        manifest_path
    )

    artifact_hashes = {
        str(path):
            sha256_file(
                path
            )

        for path in output_paths
    }

    hash_path = (
        OUT_DIR
        / "phase_8_2_2_artifact_sha256_V1.json"
    )

    write_json(
        hash_path,
        artifact_hashes,
    )

    print(
        f"Manifest:                    "
        f"{manifest_path}"
    )

    print(
        f"SHA256 registry:             "
        f"{hash_path}"
    )


    # =========================================================================
    # FINAL STATUS
    # =========================================================================

    separator()

    print(
        "PHASE 8.2.2 V1 — "
        "MATERIALIZED_AND_AUDITED"
    )

    separator()

    print()

    print(
        "Stable role-node indices preserved:       YES"
    )

    print(
        "Person nodes appended deterministically:  YES"
    )

    print(
        "Typed relation vocabulary 10/10 exact:    YES"
    )

    print(
        "Founder gate strictly pre-origin:         YES"
    )

    print(
        "Acquisition gate strictly pre-origin:     YES"
    )

    print(
        "Acquisition existence holdout mask:       NO"
    )

    print(
        "Frozen direct target-pair overlay:        YES"
    )

    print(
        "ALL COLD_t incident edges hidden at fit:  YES"
    )

    print(
        "COLD_t incoming edges restorable:         YES"
    )

    print(
        "COLD_t outgoing messages suppressed:      YES"
    )

    print(
        "Evaluation candidate matrices read:       NO"
    )

    print(
        "Model instantiated:                       NO"
    )

    print(
        "Training performed:                       NO"
    )

    print(
        "Performance inspected:                    NO"
    )

    print(
        "T60 scored:                               NO"
    )

    print(
        "April analyzed:                           NO"
    )

    print(
        "Training allowed now:                     NO"
    )

    print()

    print(
        "NEXT: Phase 8.2.3 — implement "
        "B1a/B1b/M1/M2 modules against "
        "the frozen Phase-8 graph artifacts."
    )


if __name__ == "__main__":

    try:

        main()

    except Exception as exc:

        separator()

        print(
            "PHASE 8.2.2 V1 — "
            "FAILED / NON-AUTHORITATIVE"
        )

        separator()

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise
