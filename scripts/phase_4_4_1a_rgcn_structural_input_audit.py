from pathlib import Path
import importlib.util
import importlib.metadata
import json
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 4.4.1a — R-GCN STRUCTURAL MESSAGE-PASSING INPUT AUDIT
#
# PURPOSE
# -------
# Audit the exact frozen Phase-3 graph artifacts that will feed the ITRS
# preference-propagation / R-GCN module.
#
# THIS SCRIPT DOES NOT:
#   - alter graph semantics,
#   - add self-loops,
#   - add investment-event edges,
#   - recreate Tianyancha relations,
#   - initialize neural parameters,
#   - instantiate an R-GCN,
#   - train anything.
#
# IMPORTANT PHASE-3 ARTIFACT DISTINCTION
# --------------------------------------
# relation_index.csv:
#   semantic dictionary mapping relation_id -> typed relation meaning
#
# edge_type.npy:
#   authoritative per-edge relation IDs and therefore authoritative source
#   for actual relation edge counts
#
# edge_index.npy:
#   authoritative source/destination node indices for every structural edge
#
# Phase 2, Phase 3 and all completed Phase-4 contracts remain frozen.
# =============================================================================


# =============================================================================
# INPUTS
# =============================================================================

GRAPH_DIR = Path(
    "data/experimental/phase_3/"
    "model_ready"
)

NODE_INDEX_PATH = (
    GRAPH_DIR
    / "node_index.parquet"
)

RELATION_INDEX_PATH = (
    GRAPH_DIR
    / "relation_index.csv"
)

EDGE_INDEX_PATH = (
    GRAPH_DIR
    / "edge_index.npy"
)

EDGE_TYPE_PATH = (
    GRAPH_DIR
    / "edge_type.npy"
)

EDGE_MANIFEST_PATH = (
    GRAPH_DIR
    / "edge_manifest.parquet"
)

GRAPH_MASKS_PATH = (
    GRAPH_DIR
    / "graph_variant_masks.npz"
)

GRAPH_METADATA_PATH = (
    GRAPH_DIR
    / "model_ready_graph_metadata.json"
)

NODE_COVERAGE_PATH = (
    GRAPH_DIR
    / "node_structural_coverage.parquet"
)


# =============================================================================
# OUTPUTS
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_4/"
    "rgcn_input_audit"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FROZEN PHASE-3 EXPECTATIONS
# =============================================================================

EXPECTED_INVESTORS = 165_975
EXPECTED_STARTUPS = 311_589

EXPECTED_NODES = 477_564

EXPECTED_EDGES = 158_818

EXPECTED_RELATIONS = 12

EXPECTED_CONNECTED_NODES = 74_757
EXPECTED_ISOLATES = 402_807


EXPECTED_CORE_EDGES = 158_818
EXPECTED_FOUNDER_EDGES = 94_818
EXPECTED_ACQUISITION_EDGES = 64_000


INVESTOR_INDEX_START = 0
INVESTOR_INDEX_END = 165_974

STARTUP_INDEX_START = 165_975
STARTUP_INDEX_END = 477_563


# =============================================================================
# EXACT FROZEN TYPED RELATION VOCABULARY
#
# relation_id:
#   (
#       source_type,
#       semantic_relation,
#       target_type,
#       expected_edge_count
#   )
#
# IMPORTANT:
# expected_edge_count is a frozen Phase-3 expectation.
# It is NOT read from relation_index.csv.
#
# Actual edge counts will be reconstructed independently from edge_type.npy.
# =============================================================================

EXPECTED_RELATION_VOCABULARY = {

    0: (
        "investor",
        "acquired_by",
        "investor",
        1_565,
    ),

    1: (
        "investor",
        "acquired_by",
        "startup",
        1_271,
    ),

    2: (
        "investor",
        "acquired",
        "investor",
        1_565,
    ),

    3: (
        "investor",
        "acquired",
        "startup",
        11_427,
    ),

    4: (
        "investor",
        "shared_founder",
        "investor",
        2_460,
    ),

    5: (
        "investor",
        "shared_founder",
        "startup",
        7_768,
    ),

    6: (
        "startup",
        "acquired_by",
        "investor",
        11_427,
    ),

    7: (
        "startup",
        "acquired_by",
        "startup",
        17_737,
    ),

    8: (
        "startup",
        "acquired",
        "investor",
        1_271,
    ),

    9: (
        "startup",
        "acquired",
        "startup",
        17_737,
    ),

    10: (
        "startup",
        "shared_founder",
        "investor",
        7_768,
    ),

    11: (
        "startup",
        "shared_founder",
        "startup",
        76_822,
    ),
}


# =============================================================================
# HELPERS
# =============================================================================

def banner(title):

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def require(
    condition,
    message,
):

    if not condition:

        raise AssertionError(
            message
        )


def node_type_from_index(
    indices,
):

    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    result = np.empty(
        len(indices),
        dtype=object,
    )

    investor_mask = (
        (indices >= INVESTOR_INDEX_START)
        &
        (indices <= INVESTOR_INDEX_END)
    )

    startup_mask = (
        (indices >= STARTUP_INDEX_START)
        &
        (indices <= STARTUP_INDEX_END)
    )

    valid = (
        investor_mask
        | startup_mask
    )

    require(
        bool(
            np.all(valid)
        ),
        "Node index outside frozen role ranges.",
    )

    result[
        investor_mask
    ] = "investor"

    result[
        startup_mask
    ] = "startup"

    return result


def resolve_relation_column(
    dataframe,
    candidates,
    semantic_name,
):

    normalized = {
        str(column)
        .strip()
        .casefold():
            column

        for column
        in dataframe.columns
    }


    matches = []


    for candidate in candidates:

        candidate_norm = (
            candidate
            .strip()
            .casefold()
        )

        if (
            candidate_norm
            in normalized
        ):

            matches.append(
                normalized[
                    candidate_norm
                ]
            )


    matches = list(
        dict.fromkeys(
            matches
        )
    )


    if len(matches) != 1:

        print()
        print(
            f"Could not uniquely resolve "
            f"{semantic_name}."
        )

        print(
            f"Candidates: {candidates}"
        )

        print(
            f"Actual columns: "
            f"{dataframe.columns.tolist()}"
        )

        raise AssertionError(
            f"Ambiguous relation-index schema "
            f"for {semantic_name}."
        )


    return matches[0]


# =============================================================================
# START
# =============================================================================

banner(
    "PHASE 4.4.1a — "
    "R-GCN STRUCTURAL MESSAGE-PASSING INPUT AUDIT"
)


# =============================================================================
# 1. ENVIRONMENT
# =============================================================================

banner(
    "ENVIRONMENT"
)


print(
    f"Python: "
    f"{sys.version.splitlines()[0]}"
)


pyg_spec = importlib.util.find_spec(
    "torch_geometric"
)


pyg_available = (
    pyg_spec is not None
)


pyg_version = None


if pyg_available:

    try:

        pyg_version = (
            importlib.metadata.version(
                "torch-geometric"
            )
        )

    except importlib.metadata.PackageNotFoundError:

        # Fallback only if distribution metadata cannot be resolved.
        # Do not fail the structural audit because PyG is optional here.
        pyg_version = "installed_version_unresolved"


print(
    f"PyTorch Geometric installed: "
    f"{pyg_available}"
)

print(
    f"PyTorch Geometric version:   "
    f"{pyg_version}"
)


print()
print(
    "PyG is NOT required for this audit."
)


# =============================================================================
# 2. INPUT FILE EXISTENCE
# =============================================================================

banner(
    "FROZEN PHASE-3 ARTIFACT EXISTENCE"
)


input_paths = [
    NODE_INDEX_PATH,
    RELATION_INDEX_PATH,
    EDGE_INDEX_PATH,
    EDGE_TYPE_PATH,
    EDGE_MANIFEST_PATH,
    GRAPH_MASKS_PATH,
    GRAPH_METADATA_PATH,
    NODE_COVERAGE_PATH,
]


for path in input_paths:

    exists = path.exists()

    print(
        f"{str(path):<100} "
        f"{'FOUND' if exists else 'MISSING'}"
    )

    require(
        exists,
        f"Missing frozen Phase-3 artifact: {path}",
    )


# =============================================================================
# 3. LOAD NODE INDEX
# =============================================================================

banner(
    "NODE INDEX INTEGRITY"
)


nodes = pd.read_parquet(
    NODE_INDEX_PATH
)


print(
    f"Node rows: "
    f"{len(nodes):,}"
)

print(
    f"Columns:   "
    f"{nodes.columns.tolist()}"
)


require(
    len(nodes)
    == EXPECTED_NODES,
    "Frozen node population changed.",
)


required_node_columns = [
    "node_index",
    "node_id",
    "node_type",
    "raw_entity_id",
]


missing_node_columns = [
    column

    for column
    in required_node_columns

    if column not in nodes.columns
]


require(
    len(
        missing_node_columns
    ) == 0,
    (
        "Missing required node fields: "
        f"{missing_node_columns}"
    ),
)


node_indices = (
    nodes[
        "node_index"
    ]
    .to_numpy(
        dtype=np.int64
    )
)


require(
    np.array_equal(
        node_indices,
        np.arange(
            EXPECTED_NODES,
            dtype=np.int64,
        ),
    ),
    "Node indices are no longer contiguous.",
)


node_type_counts = (
    nodes[
        "node_type"
    ]
    .astype(str)
    .value_counts()
)


print()
print(
    node_type_counts.to_string()
)


require(
    int(
        node_type_counts.get(
            "investor",
            0,
        )
    )
    == EXPECTED_INVESTORS,
    "Investor node count changed.",
)


require(
    int(
        node_type_counts.get(
            "startup",
            0,
        )
    )
    == EXPECTED_STARTUPS,
    "Startup node count changed.",
)


range_types = node_type_from_index(
    node_indices
)


node_type_range_mismatches = int(
    np.sum(
        range_types
        != nodes[
            "node_type"
        ]
        .astype(str)
        .to_numpy()
    )
)


print()
print(
    f"node_index / node_type "
    f"range mismatches: "
    f"{node_type_range_mismatches:,}"
)


require(
    node_type_range_mismatches == 0,
    "Frozen role node ranges changed.",
)


print()
print(
    "Node index integrity: PASS"
)


# =============================================================================
# 4. LOAD RELATION INDEX
#
# IMPORTANT:
#
# Frozen actual schema:
#
#   relation_id
#   src_type
#   relation
#   dst_type
#   typed_relation_key
#
# This file is a semantic dictionary only.
# It does NOT contain edge counts.
# =============================================================================

banner(
    "TYPED RELATION VOCABULARY"
)


relation_index = pd.read_csv(
    RELATION_INDEX_PATH
)


print(
    f"Relation rows: "
    f"{len(relation_index)}"
)

print(
    f"Columns:       "
    f"{relation_index.columns.tolist()}"
)


require(
    len(
        relation_index
    )
    == EXPECTED_RELATIONS,
    "Typed relation vocabulary changed.",
)


relation_id_col = resolve_relation_column(
    relation_index,
    [
        "relation_id",
        "id",
    ],
    "relation ID",
)


source_col = resolve_relation_column(
    relation_index,
    [
        "src_type",
        "source_type",
        "source",
    ],
    "source type",
)


relation_col = resolve_relation_column(
    relation_index,
    [
        "relation",
        "semantic_relation",
    ],
    "semantic relation",
)


target_col = resolve_relation_column(
    relation_index,
    [
        "dst_type",
        "target_type",
        "target",
    ],
    "target type",
)


key_col = resolve_relation_column(
    relation_index,
    [
        "typed_relation_key",
        "relation_key",
        "typed_key",
    ],
    "typed relation key",
)


print()
print(
    "Resolved relation-index schema:"
)

print(
    f"  ID:        {relation_id_col}"
)

print(
    f"  Source:    {source_col}"
)

print(
    f"  Relation:  {relation_col}"
)

print(
    f"  Target:    {target_col}"
)

print(
    f"  Key:       {key_col}"
)


print()
print(
    "Relation edge-count column:"
)

print(
    "  NOT PRESENT — correct"
)

print(
    "  Actual counts will be reconstructed "
    "from edge_type.npy"
)


# =============================================================================
# 5. EXACT FROZEN RELATION SEMANTIC ROUNDTRIP
#
# relation_index.csv is a semantic ID dictionary only.
#
# Counts are NOT expected in this CSV.
# =============================================================================

banner(
    "EXACT FROZEN RELATION ROUNDTRIP"
)


relation_index[
    relation_id_col
] = (
    relation_index[
        relation_id_col
    ]
    .astype(int)
)


relation_index = (
    relation_index
    .sort_values(
        relation_id_col
    )
    .reset_index(
        drop=True
    )
)


require(
    np.array_equal(
        relation_index[
            relation_id_col
        ]
        .to_numpy(),
        np.arange(
            EXPECTED_RELATIONS
        ),
    ),
    "Relation IDs are not exactly 0..11.",
)


relation_contract_records = []


for relation_id in range(
    EXPECTED_RELATIONS
):

    row = relation_index.loc[
        relation_index[
            relation_id_col
        ]
        .eq(
            relation_id
        )
    ].iloc[0]


    (
        expected_source,
        expected_relation,
        expected_target,
        expected_edges,
    ) = EXPECTED_RELATION_VOCABULARY[
        relation_id
    ]


    actual_source = str(
        row[
            source_col
        ]
    ).strip()


    actual_relation = str(
        row[
            relation_col
        ]
    ).strip()


    actual_target = str(
        row[
            target_col
        ]
    ).strip()


    actual_key = str(
        row[
            key_col
        ]
    ).strip()


    expected_key = (
        f"{expected_source}|"
        f"{expected_relation}|"
        f"{expected_target}"
    )


    semantic_exact = (
        actual_source
        == expected_source

        and actual_relation
        == expected_relation

        and actual_target
        == expected_target

        and actual_key
        == expected_key
    )


    print(
        f"{relation_id:>2}  "
        f"{actual_key:<45} "
        f"{'PASS' if semantic_exact else 'FAIL'}"
    )


    require(
        semantic_exact,
        (
            "Frozen relation vocabulary mismatch "
            f"at relation ID {relation_id}."
        ),
    )


    relation_contract_records.append(
        {
            "relation_id":
                relation_id,

            "source_type":
                actual_source,

            "relation":
                actual_relation,

            "target_type":
                actual_target,

            "typed_relation_key":
                actual_key,

            # Frozen Phase-3 expectation.
            # Not read from relation_index.csv.
            "expected_edge_count":
                expected_edges,
        }
    )


print()
print(
    "Relation semantic vocabulary: PASS"
)

print(
    "Relation edge counts:         "
    "DEFERRED TO edge_type.npy AUDIT"
)


# =============================================================================
# 6. LOAD MODEL-READY EDGE ARRAYS
# =============================================================================

banner(
    "MODEL-READY EDGE ARRAY INTEGRITY"
)


edge_index = np.load(
    EDGE_INDEX_PATH,
    mmap_mode="r",
)


edge_type = np.load(
    EDGE_TYPE_PATH,
    mmap_mode="r",
)


print(
    f"edge_index shape: "
    f"{edge_index.shape}"
)

print(
    f"edge_index dtype: "
    f"{edge_index.dtype}"
)

print(
    f"edge_type shape:  "
    f"{edge_type.shape}"
)

print(
    f"edge_type dtype:  "
    f"{edge_type.dtype}"
)


require(
    edge_index.shape
    == (
        2,
        EXPECTED_EDGES,
    ),
    "edge_index shape changed.",
)


require(
    edge_type.shape
    == (
        EXPECTED_EDGES,
    ),
    "edge_type shape changed.",
)


require(
    np.issubdtype(
        edge_index.dtype,
        np.integer,
    ),
    "edge_index must be integer.",
)


require(
    np.issubdtype(
        edge_type.dtype,
        np.integer,
    ),
    "edge_type must be integer.",
)


sources = np.asarray(
    edge_index[
        0
    ],
    dtype=np.int64,
)


destinations = np.asarray(
    edge_index[
        1
    ],
    dtype=np.int64,
)


types = np.asarray(
    edge_type,
    dtype=np.int64,
)


require(
    bool(
        np.all(
            (
                sources >= 0
            )
            &
            (
                sources
                < EXPECTED_NODES
            )
        )
    ),
    "Source index outside frozen node universe.",
)


require(
    bool(
        np.all(
            (
                destinations >= 0
            )
            &
            (
                destinations
                < EXPECTED_NODES
            )
        )
    ),
    "Destination index outside frozen node universe.",
)


require(
    bool(
        np.all(
            (
                types >= 0
            )
            &
            (
                types
                < EXPECTED_RELATIONS
            )
        )
    ),
    "Relation ID outside 0..11.",
)


print()
print(
    "Edge array integrity: PASS"
)


# =============================================================================
# 7. STRUCTURAL EDGE UNIQUENESS
# =============================================================================

banner(
    "STRUCTURAL EDGE UNIQUENESS"
)


self_loop_count = int(
    np.sum(
        sources
        == destinations
    )
)


edge_triplets = pd.DataFrame(
    {
        "src":
            sources,

        "relation_id":
            types,

        "dst":
            destinations,
    }
)


duplicate_edge_triplets = int(
    edge_triplets
    .duplicated()
    .sum()
)


print(
    f"Explicit graph self-loops:       "
    f"{self_loop_count:,}"
)

print(
    f"Duplicate src-relation-dst rows: "
    f"{duplicate_edge_triplets:,}"
)


require(
    self_loop_count == 0,
    (
        "Frozen Phase-3 graph unexpectedly "
        "contains explicit self-loops."
    ),
)


require(
    duplicate_edge_triplets == 0,
    (
        "Frozen Phase-3 graph unexpectedly "
        "contains duplicate typed edges."
    ),
)


print()
print(
    "Important:"
)

print(
    "  R-GCN self transformation will be "
    "handled neurally."
)

print(
    "  No structural self-loop edges "
    "will be added."
)


# =============================================================================
# 8. EDGE COUNTS BY TYPED RELATION
#
# edge_type.npy is the authoritative model-ready source for actual typed
# relation counts.
# =============================================================================

banner(
    "EDGE COUNTS BY TYPED RELATION"
)


actual_relation_counts = np.bincount(
    types,
    minlength=EXPECTED_RELATIONS,
)


require(
    len(
        actual_relation_counts
    )
    == EXPECTED_RELATIONS,
    "Unexpected relation-count vector length.",
)


require(
    int(
        actual_relation_counts.sum()
    )
    == EXPECTED_EDGES,
    (
        "Per-relation edge counts do not "
        "sum to frozen graph edge total."
    ),
)


for relation_id in range(
    EXPECTED_RELATIONS
):

    (
        expected_source,
        expected_relation,
        expected_target,
        expected_count,
    ) = EXPECTED_RELATION_VOCABULARY[
        relation_id
    ]


    actual_count = int(
        actual_relation_counts[
            relation_id
        ]
    )


    typed_key = (
        f"{expected_source}|"
        f"{expected_relation}|"
        f"{expected_target}"
    )


    exact = (
        actual_count
        == expected_count
    )


    print(
        f"{relation_id:>2}  "
        f"{typed_key:<45} "
        f"{actual_count:>8,}  "
        f"{'PASS' if exact else 'FAIL'}"
    )


    require(
        exact,
        (
            "edge_type.npy count mismatch "
            f"for relation ID {relation_id}. "
            f"Expected {expected_count:,}, "
            f"found {actual_count:,}."
        ),
    )


print()
print(
    f"Total typed edges reconstructed: "
    f"{actual_relation_counts.sum():,}"
)

print(
    "Typed relation edge counts: PASS"
)


# Add edge_type-derived counts to semantic relation audit records.

for record in relation_contract_records:

    relation_id = int(
        record[
            "relation_id"
        ]
    )

    actual_count = int(
        actual_relation_counts[
            relation_id
        ]
    )

    record[
        "actual_edge_count_from_edge_type"
    ] = actual_count

    record[
        "edge_count_match"
    ] = (
        actual_count
        == record[
            "expected_edge_count"
        ]
    )


# =============================================================================
# 9. SOURCE / TARGET ROLE COMPATIBILITY
#
# Frozen Phase-3 typed relation key:
#
#     source_type | relation | target_type
#
# Frozen edge_index convention:
#
#     edge_index[0] = source / message sender
#     edge_index[1] = destination / message receiver
# =============================================================================

banner(
    "EDGE DIRECTION / ROLE COMPATIBILITY"
)


source_types = node_type_from_index(
    sources
)


destination_types = node_type_from_index(
    destinations
)


role_mismatch_count = 0


for relation_id in range(
    EXPECTED_RELATIONS
):

    (
        expected_source,
        _,
        expected_target,
        _,
    ) = EXPECTED_RELATION_VOCABULARY[
        relation_id
    ]


    mask = (
        types
        == relation_id
    )


    source_mismatches = int(
        np.sum(
            source_types[
                mask
            ]
            != expected_source
        )
    )


    destination_mismatches = int(
        np.sum(
            destination_types[
                mask
            ]
            != expected_target
        )
    )


    relation_mismatches = (
        source_mismatches
        + destination_mismatches
    )


    role_mismatch_count += (
        relation_mismatches
    )


    print(
        f"Relation {relation_id:>2}: "
        f"source mismatches="
        f"{source_mismatches:,}, "
        f"destination mismatches="
        f"{destination_mismatches:,}"
    )


require(
    role_mismatch_count == 0,
    (
        "Typed relation IDs disagree with "
        "source/destination node roles."
    ),
)


print()
print(
    "Typed relation directionality: PASS"
)


# =============================================================================
# 10. RELATION-SPECIFIC INCOMING DEGREE
#
# ITRS Eq. (9) normalizes each relation-specific neighborhood by:
#
#     1 / |N_e^r|
#
# Therefore we audit destination-side degree separately for each typed
# relation channel.
# =============================================================================

banner(
    "RELATION-SPECIFIC INCOMING DEGREE"
)


relation_degree_records = []


for relation_id in range(
    EXPECTED_RELATIONS
):

    (
        expected_source,
        expected_relation,
        expected_target,
        expected_count,
    ) = EXPECTED_RELATION_VOCABULARY[
        relation_id
    ]


    mask = (
        types
        == relation_id
    )


    relation_destinations = (
        destinations[
            mask
        ]
    )


    unique_destinations, counts = np.unique(
        relation_destinations,
        return_counts=True,
    )


    require(
        int(
            counts.sum()
        )
        == expected_count,
        (
            "Relation degree counts do not "
            "reconstruct relation edges."
        ),
    )


    degree_summary = {

        "relation_id":
            relation_id,

        "typed_relation_key":
            (
                f"{expected_source}|"
                f"{expected_relation}|"
                f"{expected_target}"
            ),

        "edge_count":
            expected_count,

        "destination_nodes_with_relation":
            int(
                len(
                    unique_destinations
                )
            ),

        "mean_in_degree_active":
            float(
                counts.mean()
            ),

        "median_in_degree_active":
            float(
                np.median(
                    counts
                )
            ),

        "p90_in_degree_active":
            float(
                np.quantile(
                    counts,
                    0.90,
                )
            ),

        "p95_in_degree_active":
            float(
                np.quantile(
                    counts,
                    0.95,
                )
            ),

        "p99_in_degree_active":
            float(
                np.quantile(
                    counts,
                    0.99,
                )
            ),

        "max_in_degree":
            int(
                counts.max()
            ),
    }


    relation_degree_records.append(
        degree_summary
    )


    print()
    print(
        f"Relation {relation_id}: "
        f"{degree_summary['typed_relation_key']}"
    )

    print(
        f"  edges:                  "
        f"{expected_count:,}"
    )

    print(
        f"  destination nodes:      "
        f"{len(unique_destinations):,}"
    )

    print(
        f"  mean active in-degree:  "
        f"{counts.mean():.3f}"
    )

    print(
        f"  median:                 "
        f"{np.median(counts):.3f}"
    )

    print(
        f"  p90:                    "
        f"{np.quantile(counts, 0.90):.3f}"
    )

    print(
        f"  p95:                    "
        f"{np.quantile(counts, 0.95):.3f}"
    )

    print(
        f"  p99:                    "
        f"{np.quantile(counts, 0.99):.3f}"
    )

    print(
        f"  max:                    "
        f"{counts.max():,}"
    )


# =============================================================================
# 11. GLOBAL STRUCTURAL COVERAGE FROM EDGE ARRAYS
# =============================================================================

banner(
    "GLOBAL STRUCTURAL COVERAGE"
)


connected_nodes = np.union1d(
    np.unique(
        sources
    ),
    np.unique(
        destinations
    ),
)


connected_count = int(
    len(
        connected_nodes
    )
)


isolate_count = (
    EXPECTED_NODES
    - connected_count
)


print(
    f"Connected role nodes: "
    f"{connected_count:,}"
)

print(
    f"Structural isolates:  "
    f"{isolate_count:,}"
)


require(
    connected_count
    == EXPECTED_CONNECTED_NODES,
    "Connected-node total changed.",
)


require(
    isolate_count
    == EXPECTED_ISOLATES,
    "Structural-isolate total changed.",
)


# =============================================================================
# 12. DIRECTIONAL MESSAGE-PASSING COVERAGE
# =============================================================================

banner(
    "MESSAGE-PASSING DIRECTIONAL COVERAGE"
)


nodes_with_incoming = np.unique(
    destinations
)


nodes_with_outgoing = np.unique(
    sources
)


incoming_count = int(
    len(
        nodes_with_incoming
    )
)


outgoing_count = int(
    len(
        nodes_with_outgoing
    )
)


incoming_only = np.setdiff1d(
    nodes_with_incoming,
    nodes_with_outgoing,
)


outgoing_only = np.setdiff1d(
    nodes_with_outgoing,
    nodes_with_incoming,
)


both_directional = np.intersect1d(
    nodes_with_incoming,
    nodes_with_outgoing,
)


print(
    f"Nodes receiving messages: "
    f"{incoming_count:,}"
)

print(
    f"Nodes sending messages:   "
    f"{outgoing_count:,}"
)

print(
    f"Incoming only:            "
    f"{len(incoming_only):,}"
)

print(
    f"Outgoing only:            "
    f"{len(outgoing_only):,}"
)

print(
    f"Both:                     "
    f"{len(both_directional):,}"
)


coverage_union = np.union1d(
    nodes_with_incoming,
    nodes_with_outgoing,
)


require(
    len(
        coverage_union
    )
    == EXPECTED_CONNECTED_NODES,
    (
        "Directional coverage does not "
        "reconstruct connected-node total."
    ),
)


# =============================================================================
# 13. GRAPH VARIANT MASK AUDIT
# =============================================================================

banner(
    "GRAPH VARIANT MASKS"
)


variant_masks = np.load(
    GRAPH_MASKS_PATH
)


mask_names = list(
    variant_masks.files
)


print(
    f"Stored masks: "
    f"{mask_names}"
)


mask_records = []


expected_mask_counts = {

    "core":
        EXPECTED_CORE_EDGES,

    "founder_only_ablation":
        EXPECTED_FOUNDER_EDGES,

    "acquisition_only_ablation":
        EXPECTED_ACQUISITION_EDGES,
}


for mask_name in mask_names:

    mask = variant_masks[
        mask_name
    ]


    require(
        mask.shape
        == (
            EXPECTED_EDGES,
        ),
        (
            f"Graph mask {mask_name} "
            "has unexpected shape."
        ),
    )


    if mask.dtype == bool:

        selected_count = int(
            mask.sum()
        )

    else:

        unique_values = set(
            np.unique(
                mask
            ).tolist()
        )


        require(
            unique_values.issubset(
                {
                    0,
                    1,
                    False,
                    True,
                }
            ),
            (
                f"Graph mask {mask_name} "
                "is not binary."
            ),
        )


        selected_count = int(
            np.sum(
                mask
            )
        )


    print(
        f"{mask_name:<35} "
        f"{selected_count:>8,}"
    )


    mask_records.append(
        {
            "variant":
                mask_name,

            "selected_edges":
                selected_count,
        }
    )


    if (
        mask_name
        in expected_mask_counts
    ):

        require(
            selected_count
            == expected_mask_counts[
                mask_name
            ],
            (
                f"Frozen mask count changed "
                f"for {mask_name}."
            ),
        )


missing_expected_masks = (
    set(
        expected_mask_counts
    )
    - set(
        mask_names
    )
)


require(
    len(
        missing_expected_masks
    ) == 0,
    (
        "Missing expected graph variants: "
        f"{sorted(missing_expected_masks)}"
    ),
)


# =============================================================================
# 14. SEMANTIC RELATION FAMILY TOTALS
# =============================================================================

banner(
    "SEMANTIC RELATION FAMILY TOTALS"
)


semantic_family_counts = {}


for relation_id, (
    _source,
    semantic_relation,
    _target,
    expected_edge_count,
) in EXPECTED_RELATION_VOCABULARY.items():

    semantic_family_counts[
        semantic_relation
    ] = (
        semantic_family_counts.get(
            semantic_relation,
            0,
        )
        + expected_edge_count
    )


for relation_name, count in sorted(
    semantic_family_counts.items()
):

    print(
        f"{relation_name:<20} "
        f"{count:>8,}"
    )


require(
    semantic_family_counts[
        "shared_founder"
    ]
    == 94_818,
    "shared_founder total changed.",
)


require(
    semantic_family_counts[
        "acquired"
    ]
    == 32_000,
    "acquired total changed.",
)


require(
    semantic_family_counts[
        "acquired_by"
    ]
    == 32_000,
    "acquired_by total changed.",
)


require(
    sum(
        semantic_family_counts.values()
    )
    == EXPECTED_EDGES,
    (
        "Semantic relation-family totals "
        "do not reconstruct core graph."
    ),
)


# =============================================================================
# 15. R-GCN INPUT DIMENSION CONTRACT
#
# Paper:
#
#     n^(0) = [L_o ; L_b]
#
# This concatenation is along the NODE dimension, not feature dimension.
#
# Investor latent embeddings: [165975, 40]
# Startup latent embeddings:  [311589, 40]
#
# Combined node state:
#
#     n^(0): [477564, 40]
#
# Paper implementation details:
#   latent dimension      = 40
#   structural dimension  = 40
#   R-GCN layers          = 2
#   basis count           = 5
#
# Therefore the current architectural implication is:
#
#     40 -> 40 -> 40
#
# Exact neural implementation is intentionally deferred to Phase 4.4.1b.
# =============================================================================

banner(
    "R-GCN DIMENSION IMPLICATION"
)


LATENT_DIM = 40
STRUCTURAL_DIM = 40

RGCN_LAYERS = 2
RGCN_BASES = 5


print(
    f"Initial Investor latent dim: "
    f"{LATENT_DIM}"
)

print(
    f"Initial Startup latent dim:  "
    f"{LATENT_DIM}"
)

print(
    f"Investor latent matrix:      "
    f"[{EXPECTED_INVESTORS:,}, {LATENT_DIM}]"
)

print(
    f"Startup latent matrix:       "
    f"[{EXPECTED_STARTUPS:,}, {LATENT_DIM}]"
)

print(
    f"Combined node matrix:        "
    f"[{EXPECTED_NODES:,}, {LATENT_DIM}]"
)

print(
    f"R-GCN layers:                "
    f"{RGCN_LAYERS}"
)

print(
    f"R-GCN bases:                 "
    f"{RGCN_BASES}"
)

print(
    f"Structural output dim:       "
    f"{STRUCTURAL_DIM}"
)


require(
    LATENT_DIM
    == STRUCTURAL_DIM,
    (
        "Current paper-grounded reconstruction "
        "expects 40 -> 40 R-GCN channels."
    ),
)


# =============================================================================
# 16. ITEMS DELIBERATELY NOT YET FROZEN
# =============================================================================

banner(
    "NEURAL DETAILS NOT YET FROZEN"
)


not_yet_frozen = [

    (
        "PyG RGCNConv versus explicit "
        "custom implementation"
    ),

    (
        "exact basis-decomposition "
        "tensor parameterization"
    ),

    (
        "exact self/root transform "
        "implementation"
    ),

    (
        "whether a separate additive "
        "layer bias is used"
    ),

    (
        "activation sigma in Eq. (9)"
    ),

    (
        "exact layer-to-layer "
        "activation placement"
    ),

    (
        "exact basis coefficient "
        "initialization"
    ),

    (
        "exact global Kaiming variant"
    ),
]


for item in not_yet_frozen:

    print(
        f"  - {item}"
    )


print()
print(
    "No neural implementation choice "
    "is made by Phase 4.4.1a."
)


# =============================================================================
# 17. SAVE RELATION DEGREE AUDIT
# =============================================================================

banner(
    "SAVING AUDIT OUTPUTS"
)


relation_degree_df = pd.DataFrame(
    relation_degree_records
)


relation_degree_path = (
    OUT_DIR
    / "rgcn_relation_incoming_degree_audit.csv"
)


relation_degree_df.to_csv(
    relation_degree_path,
    index=False,
)


# =============================================================================
# 18. SAVE TYPED RELATION CONTRACT AUDIT
#
# This table now explicitly distinguishes:
#
# expected_edge_count
# actual_edge_count_from_edge_type
# edge_count_match
# =============================================================================

relation_contract_df = pd.DataFrame(
    relation_contract_records
)


relation_contract_path = (
    OUT_DIR
    / "rgcn_typed_relation_contract_audit.csv"
)


relation_contract_df.to_csv(
    relation_contract_path,
    index=False,
)


# =============================================================================
# 19. SAVE GRAPH VARIANT MASK AUDIT
# =============================================================================

mask_df = pd.DataFrame(
    mask_records
)


mask_path = (
    OUT_DIR
    / "rgcn_graph_variant_mask_audit.csv"
)


mask_df.to_csv(
    mask_path,
    index=False,
)


# =============================================================================
# 20. METADATA
# =============================================================================

metadata = {

    "phase":
        "4.4.1a",

    "status":
        "COMPLETE_AUDIT_ONLY",

    "component":
        (
            "ITRS preference-propagation "
            "structural input audit"
        ),

    "environment": {

        "python":
            sys.version.splitlines()[0],

        "torch_geometric_available":
            pyg_available,

        "torch_geometric_version":
            pyg_version,
    },

    "frozen_graph": {

        "node_count":
            EXPECTED_NODES,

        "investor_nodes":
            EXPECTED_INVESTORS,

        "startup_nodes":
            EXPECTED_STARTUPS,

        "edge_count":
            EXPECTED_EDGES,

        "typed_relation_count":
            EXPECTED_RELATIONS,

        "relation_semantics_source":
            "relation_index.csv",

        "relation_count_source":
            "edge_type.npy",

        "edge_endpoint_source":
            "edge_index.npy",

        "connected_nodes":
            connected_count,

        "isolates":
            isolate_count,
    },

    "relation_index_schema": {

        "relation_id":
            relation_id_col,

        "source_type":
            source_col,

        "semantic_relation":
            relation_col,

        "target_type":
            target_col,

        "typed_relation_key":
            key_col,

        "edge_count_column_present":
            False,

        "edge_count_policy":
            (
                "Actual counts reconstructed "
                "from edge_type.npy."
            ),
    },

    "edge_integrity": {

        "explicit_self_loops":
            self_loop_count,

        "duplicate_typed_edges":
            duplicate_edge_triplets,

        "source_target_role_mismatches":
            role_mismatch_count,
    },

    "direction_semantics": {

        "edge_index_row_0":
            "message source",

        "edge_index_row_1":
            "message destination",

        "relation_key":
            (
                "source_type|relation|"
                "target_type"
            ),

        "normalization_audit_side":
            (
                "destination incoming degree "
                "per typed relation"
            ),
    },

    "directional_coverage": {

        "nodes_with_incoming_edges":
            incoming_count,

        "nodes_with_outgoing_edges":
            outgoing_count,

        "incoming_only":
            int(
                len(
                    incoming_only
                )
            ),

        "outgoing_only":
            int(
                len(
                    outgoing_only
                )
            ),

        "both":
            int(
                len(
                    both_directional
                )
            ),
    },

    "paper_grounded_dimensions": {

        "initial_latent_dim":
            LATENT_DIM,

        "combined_node_matrix":
            [
                EXPECTED_NODES,
                LATENT_DIM,
            ],

        "rgcn_layers":
            RGCN_LAYERS,

        "rgcn_bases":
            RGCN_BASES,

        "structural_output_dim":
            STRUCTURAL_DIM,
    },

    "self_loop_policy": {

        "explicit_graph_self_loop_relation":
            False,

        "structural_self_loop_edges_added":
            False,

        "paper_contains_separate_self_transform":
            True,

        "exact_neural_self_transform":
            "NOT_YET_FROZEN",
    },

    "semantic_relation_family_counts":
        {
            key:
                int(value)

            for key, value
            in semantic_family_counts.items()
        },

    "graph_variants": {

        record[
            "variant"
        ]:
            record[
                "selected_edges"
            ]

        for record
        in mask_records
    },

    "not_yet_frozen":
        not_yet_frozen,

    "training_performed":
        False,

    "graph_modified":
        False,

    "investment_event_edges_added":
        False,

    "explicit_self_loop_edges_added":
        False,

    "upstream_reopened": {

        "phase_2":
            False,

        "phase_3":
            False,

        "phase_4_2":
            False,

        "phase_4_3":
            False,
    },
}


metadata_path = (
    OUT_DIR
    / "rgcn_structural_input_audit_metadata.json"
)


with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

banner(
    "PHASE 4.4.1a FINAL SUMMARY"
)


print(
    f"Role nodes:                     "
    f"{EXPECTED_NODES:,}"
)

print(
    f"Directed structural edges:      "
    f"{EXPECTED_EDGES:,}"
)

print(
    f"Typed relation channels:        "
    f"{EXPECTED_RELATIONS}"
)


print()
print(
    f"Connected nodes:                "
    f"{connected_count:,}"
)

print(
    f"Structural isolates:            "
    f"{isolate_count:,}"
)


print()
print(
    f"Explicit self-loop edges:       "
    f"{self_loop_count:,}"
)

print(
    f"Duplicate typed edges:          "
    f"{duplicate_edge_triplets:,}"
)

print(
    f"Relation-role mismatches:       "
    f"{role_mismatch_count:,}"
)


print()
print(
    "Relation semantics source:      "
    "relation_index.csv"
)

print(
    "Relation edge-count source:     "
    "edge_type.npy"
)

print(
    "Edge endpoint source:           "
    "edge_index.npy"
)


print()
print(
    "Typed relation vocabulary:      PASS"
)

print(
    "Relation edge counts:           PASS"
)

print(
    "Source/destination roles:       PASS"
)

print(
    "Relation incoming-degree audit: PASS"
)

print(
    "Graph variant masks:            PASS"
)

print(
    "Structural coverage:            PASS"
)


print()
print(
    "R-GCN initial node input:"
)

print(
    f"  [{EXPECTED_NODES:,}, 40]"
)

print(
    "  latent embeddings only"
)


print()
print(
    "Paper-fixed R-GCN depth:         2"
)

print(
    "Paper-fixed basis count:         5"
)

print(
    "Paper-fixed structural dim:      40"
)


print()
print(
    "Explicit graph self-loops added: NO"
)

print(
    "Investment-event edges added:    NO"
)

print(
    "Phase-3 graph modified:          NO"
)

print(
    "Training performed:              NO"
)


print()
print(
    f"PyTorch Geometric available:     "
    f"{pyg_available}"
)

print(
    f"PyTorch Geometric version:       "
    f"{pyg_version}"
)


print()
print(
    "Neural R-GCN implementation:     "
    "NOT YET FROZEN"
)


print()
print("Outputs:")

for path in [
    relation_degree_path,
    relation_contract_path,
    mask_path,
    metadata_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.4.1a STATUS: COMPLETE — "
    "STRUCTURAL R-GCN INPUTS AUDITED ONLY"
)