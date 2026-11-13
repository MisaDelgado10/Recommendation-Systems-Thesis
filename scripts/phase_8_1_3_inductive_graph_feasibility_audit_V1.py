from pathlib import Path
import hashlib
import json
from collections import defaultdict

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 8.1.3 — INDUCTIVE GRAPH FEASIBILITY AUDIT V1
# =============================================================================
#
# Scientific role:
#
#   DEVELOPMENT-ONLY PRE-ARCHITECTURE GRAPH FEASIBILITY AUDIT
#
# PURPOSE
# -------
#
# Determine whether the frozen Phase-8 graph can support an ID-free
# inductive startup structural encoder.
#
# DEVELOPMENT EVIDENCE ONLY:
#
#   T57
#   T58
#   T59
#
# April is NOT analyzed for graph-connectivity architecture evidence.
#
# Questions:
#
#   1. Do nodes used by each graph track have raw content features?
#   2. How many primary cold-start cases are structurally connected?
#   3. When connected, do they have at least one content-encodable neighbor?
#   4. How sparse are the usable neighborhoods?
#
# NO:
#
#   - architecture selection
#   - model construction
#   - training
#   - inference
#   - ranking metrics
#   - model selection
#   - April graph-connectivity analysis
#   - April performance inspection
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

PHASE80_CLOSURE = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_13_phase_closure/"
    "phase_8_0_13_closure_manifest_V1.json"
)


PHASE812_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_2_input_contract"
)

INPUT_CONTRACT_PATH = (
    PHASE812_DIR
    / "phase_8_1_2_inductive_startup_input_contract_V1.json"
)

INPUT_MANIFEST_PATH = (
    PHASE812_DIR
    / "phase_8_1_2_manifest_V1.json"
)


DESCRIPTION_MANIFEST_PATH = Path(
    "data/experimental/phase_4/"
    "description_contract/"
    "description_input_manifest.parquet"
)

NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/"
    "model_ready/"
    "node_index.parquet"
)

EDGE_INDEX_PATH = Path(
    "data/experimental/phase_3/"
    "model_ready/"
    "edge_index.npy"
)


PHASE806_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_6_dual_track_graph_policy"
)

GRAPH_MASKS_PATH = (
    PHASE806_DIR
    / "phase_8_0_6_dual_track_graph_masks_V1.npz"
)

GRAPH_POLICY_PATH = (
    PHASE806_DIR
    / "phase_8_0_6_graph_information_policy_V1.json"
)


PHASE809_TARGETS_PATH = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_9_origin_eligibility_candidate_universe/"
    "phase_8_0_9_origin_eligible_targets_V5.parquet"
)


# =============================================================================
# OUTPUTS
# =============================================================================

OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_3_graph_feasibility"
)

ACTIVE_NODE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_3_active_graph_node_content_coverage_V1.csv"
)

PRIMARY_CONNECTIVITY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_3_primary_connectivity_summary_V1.csv"
)

NEIGHBOR_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_3_primary_neighbor_content_summary_V1.csv"
)

DEGREE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_3_primary_degree_summary_V1.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_3_graph_feasibility_summary_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_3_analysis_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_3_derived_artifact_sha256_V1.json"
)


# =============================================================================
# FROZEN DEVELOPMENT CONTRACT
# =============================================================================

DEVELOPMENT_ORIGINS = [
    "T57",
    "T58",
    "T59",
]

APRIL_ORIGIN = (
    "APRIL_CONFIRMATORY"
)

TRACKS = {
    "strict_temporal":
        "strict_temporal",

    "founder_proxy":
        "founder_proxy",
}

EXPECTED_PRIMARY = {
    "T57": 6_850,
    "T58": 6_872,
    "T59": 6_764,
}


# =============================================================================
# HELPERS
# =============================================================================

def separator(
    char="=",
    width=118,
):
    print(
        char * width
    )


def require(
    condition,
    message,
):
    if not condition:
        raise ValueError(
            message
        )


def sha256_file(
    path,
    chunk_size=1024 * 1024,
):
    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def read_json(
    path,
):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def write_json(
    path,
    obj,
):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            default=str,
        )


def pct(
    numerator,
    denominator,
):
    if denominator == 0:
        return float("nan")

    return (
        100.0
        * float(
            numerator
        )
        / float(
            denominator
        )
    )


def build_neighbor_sets(
    src,
    dst,
):
    """
    Build undirected unique-neighbor sets for structural feasibility.

    The frozen graph stores directed typed edges. For the question
    "does this node have usable structural context?" we care about
    incident structural neighbors irrespective of stored direction.
    """

    mapping = defaultdict(
        set
    )

    for s, d in zip(
        src,
        dst,
    ):
        s = int(
            s
        )

        d = int(
            d
        )

        if s == d:
            continue

        mapping[
            s
        ].add(
            d
        )

        mapping[
            d
        ].add(
            s
        )

    return mapping


# =============================================================================
# MAIN
# =============================================================================

def main():

    separator()

    print(
        "PHASE 8.1.3 — "
        "INDUCTIVE GRAPH FEASIBILITY AUDIT V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "DEVELOPMENT-ONLY PRE-ARCHITECTURE AUDIT"
    )

    print(
        "Development origins:         "
        "T57 / T58 / T59"
    )

    print(
        "April connectivity analyzed: NO"
    )

    print(
        "Architecture selected:       NO"
    )

    print(
        "Model constructed:           NO"
    )

    print(
        "Training performed:          NO"
    )

    print(
        "Inference executed:          NO"
    )

    print(
        "Ranking metrics scored:      NO"
    )

    print(
        "April performance inspected: NO"
    )

    # =========================================================================
    # 1. Frozen-contract gates
    # =========================================================================

    separator("-")

    print(
        "8.1.3.1 — FROZEN CONTRACT GATES"
    )

    separator("-")

    required_paths = [
        PHASE80_CLOSURE,
        INPUT_CONTRACT_PATH,
        INPUT_MANIFEST_PATH,
        DESCRIPTION_MANIFEST_PATH,
        NODE_INDEX_PATH,
        EDGE_INDEX_PATH,
        GRAPH_MASKS_PATH,
        GRAPH_POLICY_PATH,
        PHASE809_TARGETS_PATH,
    ]

    source_hashes = {}

    for path in required_paths:

        require(
            path.exists(),
            f"Missing prerequisite artifact: {path}",
        )

        observed_hash = sha256_file(
            path
        )

        source_hashes[
            str(
                path
            )
        ] = observed_hash

        print(
            f"FOUND  {path}"
        )

        print(
            f"SHA256 {observed_hash}"
        )

    closure = read_json(
        PHASE80_CLOSURE
    )

    input_contract = read_json(
        INPUT_CONTRACT_PATH
    )

    input_manifest = read_json(
        INPUT_MANIFEST_PATH
    )

    graph_policy = read_json(
        GRAPH_POLICY_PATH
    )

    require(
        closure.get(
            "phase_8_0_status"
        )
        ==
        "CLOSED",
        "Phase 8.0 is not closed.",
    )

    require(
        closure.get(
            "April_performance_inspected"
        )
        is False,
        "April performance is no longer untouched.",
    )

    require(
        input_contract.get(
            "status"
        )
        ==
        "FROZEN",
        "Phase-8.1.2 input contract not frozen.",
    )

    require(
        input_manifest.get(
            "status"
        )
        ==
        "FROZEN",
        "Phase-8.1.2 manifest not frozen.",
    )

    require(
        input_contract[
            "prohibited_startup_identity_inputs"
        ]
        ==
        [
            "startup_id_latent_embedding",
            "phase4_doc2vec_document_vector",
        ],
        "Inductive identity prohibition drift.",
    )

    require(
        graph_policy[
            "strict_temporal_track"
        ][
            "status"
        ]
        ==
        "FROZEN",
        "Strict graph policy not frozen.",
    )

    require(
        graph_policy[
            "founder_proxy_track"
        ][
            "status"
        ]
        ==
        "FROZEN",
        "Founder graph policy not frozen.",
    )

    print()

    print(
        "Phase-8.0 closure:       PASS"
    )

    print(
        "Phase-8.1.2 input freeze: PASS"
    )

    # =========================================================================
    # 2. Build graph-node content registry
    # =========================================================================

    separator("-")

    print(
        "8.1.3.2 — GRAPH-NODE "
        "CONTENT REGISTRY"
    )

    separator("-")

    nodes = pd.read_parquet(
        NODE_INDEX_PATH,
        columns=[
            "node_index",
            "node_type",
            "raw_entity_id",
        ],
    )

    nodes[
        "raw_entity_id"
    ] = (
        nodes[
            "raw_entity_id"
        ].astype(str)
    )

    description = pd.read_parquet(
        DESCRIPTION_MANIFEST_PATH,
        columns=[
            "node_type",
            "raw_entity_id",
            "has_text",
            "has_labels",
        ],
    )

    description[
        "raw_entity_id"
    ] = (
        description[
            "raw_entity_id"
        ].astype(str)
    )

    registry = nodes.merge(
        description,
        on=[
            "node_type",
            "raw_entity_id",
        ],
        how="left",
        validate="one_to_one",
    )

    require(
        len(
            registry
        )
        ==
        len(
            nodes
        ),
        "Graph/content registry population drift.",
    )

    require(
        not registry[
            "has_text"
        ].isna().any(),
        (
            "At least one graph node is absent "
            "from the frozen content manifest."
        ),
    )

    require(
        not registry[
            "node_index"
        ].duplicated().any(),
        "Duplicate node_index detected.",
    )

    max_node_index = int(
        registry[
            "node_index"
        ].max()
    )

    has_text = np.zeros(
        max_node_index + 1,
        dtype=bool,
    )

    has_labels = np.zeros(
        max_node_index + 1,
        dtype=bool,
    )

    node_types = np.empty(
        max_node_index + 1,
        dtype=object,
    )

    node_indices = registry[
        "node_index"
    ].to_numpy(
        dtype=np.int64
    )

    has_text[
        node_indices
    ] = registry[
        "has_text"
    ].to_numpy(
        dtype=bool
    )

    has_labels[
        node_indices
    ] = registry[
        "has_labels"
    ].to_numpy(
        dtype=bool
    )

    node_types[
        node_indices
    ] = registry[
        "node_type"
    ].astype(str).to_numpy()

    print(
        f"Graph nodes with content registry: "
        f"{len(registry):,} PASS"
    )

    # Startup-ID -> node index.
    startup_nodes = registry[
        registry[
            "node_type"
        ]
        ==
        "startup"
    ]

    startup_id_to_index = (
        startup_nodes
        .set_index(
            "raw_entity_id"
        )[
            "node_index"
        ]
        .astype(
            np.int64
        )
        .to_dict()
    )

    # =========================================================================
    # 3. Load frozen structural arrays
    # =========================================================================

    separator("-")

    print(
        "8.1.3.3 — FROZEN STRUCTURAL ARRAYS"
    )

    separator("-")

    edge_index = np.load(
        EDGE_INDEX_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    require(
        edge_index.ndim
        ==
        2,
        "edge_index must be rank-2.",
    )

    if (
        edge_index.shape[
            0
        ]
        ==
        2
    ):

        src_all = np.asarray(
            edge_index[
                0
            ],
            dtype=np.int64,
        )

        dst_all = np.asarray(
            edge_index[
                1
            ],
            dtype=np.int64,
        )

    elif (
        edge_index.shape[
            1
        ]
        ==
        2
    ):

        src_all = np.asarray(
            edge_index[
                :,
                0
            ],
            dtype=np.int64,
        )

        dst_all = np.asarray(
            edge_index[
                :,
                1
            ],
            dtype=np.int64,
        )

    else:

        raise ValueError(
            f"Unexpected edge_index shape: "
            f"{edge_index.shape}"
        )

    masks = np.load(
        GRAPH_MASKS_PATH,
        allow_pickle=False,
    )

    require(
        len(
            src_all
        )
        ==
        len(
            dst_all
        ),
        "Structural edge-array length mismatch.",
    )

    print(
        f"Structural directed edges: "
        f"{len(src_all):,}"
    )

    print(
        f"Available frozen masks: "
        f"{len(masks.files):,}"
    )

    # Confirm April masks exist but DO NOT analyze them.
    for track_suffix in (
        TRACKS.values()
    ):

        april_key = (
            f"{APRIL_ORIGIN}_"
            f"{track_suffix}"
        )

        require(
            april_key
            in
            masks.files,
            (
                "Missing frozen April mask: "
                f"{april_key}"
            ),
        )

    print(
        "April masks existence: PASS"
    )

    print(
        "April masks analyzed:  NO"
    )

    # =========================================================================
    # 4. Load DEVELOPMENT primary cases only
    # =========================================================================

    separator("-")

    print(
        "8.1.3.4 — DEVELOPMENT PRIMARY "
        "COLD-START CASES"
    )

    separator("-")

    targets = pd.read_parquet(
        PHASE809_TARGETS_PATH,
        columns=[
            "origin",
            "startup_id_str",
            "is_primary_stratum",
        ],
    )

    targets[
        "startup_id_str"
    ] = (
        targets[
            "startup_id_str"
        ].astype(str)
    )

    development_targets = targets[
        targets[
            "origin"
        ].isin(
            DEVELOPMENT_ORIGINS
        )
    ].copy()

    primary_targets = development_targets[
        development_targets[
            "is_primary_stratum"
        ]
    ].copy()

    for origin, expected in (
        EXPECTED_PRIMARY.items()
    ):

        n = int(
            (
                primary_targets[
                    "origin"
                ]
                ==
                origin
            ).sum()
        )

        require(
            n
            ==
            expected,
            (
                f"{origin}: primary-case "
                "count drift."
            ),
        )

        print(
            f"{origin}: {n:,} primary cases PASS"
        )

    # =========================================================================
    # 5. Development graph feasibility
    # =========================================================================

    separator("-")

    print(
        "8.1.3.5 — DEVELOPMENT GRAPH "
        "CONTENT / CONNECTIVITY AUDIT"
    )

    separator("-")

    active_node_rows = []
    connectivity_rows = []
    neighbor_rows = []
    degree_rows = []

    for origin in (
        DEVELOPMENT_ORIGINS
    ):

        origin_primary = (
            primary_targets[
                primary_targets[
                    "origin"
                ]
                ==
                origin
            ]
            .copy()
            .reset_index(
                drop=True
            )
        )

        startup_indices = []

        for startup_id in (
            origin_primary[
                "startup_id_str"
            ]
        ):

            require(
                startup_id
                in
                startup_id_to_index,
                (
                    f"{origin}: primary Startup "
                    f"{startup_id} missing from "
                    "node index."
                ),
            )

            startup_indices.append(
                int(
                    startup_id_to_index[
                        startup_id
                    ]
                )
            )

        startup_indices = np.asarray(
            startup_indices,
            dtype=np.int64,
        )

        for track_name, track_suffix in (
            TRACKS.items()
        ):

            mask_key = (
                f"{origin}_"
                f"{track_suffix}"
            )

            require(
                mask_key
                in
                masks.files,
                (
                    f"Missing graph mask: "
                    f"{mask_key}"
                ),
            )

            mask = np.asarray(
                masks[
                    mask_key
                ],
                dtype=bool,
            )

            require(
                len(
                    mask
                )
                ==
                len(
                    src_all
                ),
                (
                    f"{mask_key}: mask length "
                    "does not match edge array."
                ),
            )

            src = src_all[
                mask
            ]

            dst = dst_all[
                mask
            ]

            # -----------------------------------------------------------------
            # Active graph-node content coverage
            # -----------------------------------------------------------------

            active_nodes = np.unique(
                np.concatenate(
                    [
                        src,
                        dst,
                    ]
                )
            )

            for role in [
                "investor",
                "startup",
            ]:

                role_active = active_nodes[
                    np.asarray(
                        [
                            node_types[
                                idx
                            ]
                            ==
                            role
                            for idx
                            in active_nodes
                        ],
                        dtype=bool,
                    )
                ]

                if len(
                    role_active
                ) == 0:

                    text_n = 0
                    labels_n = 0

                else:

                    text_n = int(
                        has_text[
                            role_active
                        ].sum()
                    )

                    labels_n = int(
                        has_labels[
                            role_active
                        ].sum()
                    )

                active_node_rows.append(
                    {
                        "origin":
                            origin,

                        "track":
                            track_name,

                        "node_type":
                            role,

                        "active_nodes":
                            len(
                                role_active
                            ),

                        "with_text":
                            text_n,

                        "text_coverage_pct":
                            pct(
                                text_n,
                                len(
                                    role_active
                                ),
                            ),

                        "with_labels":
                            labels_n,

                        "label_coverage_pct":
                            pct(
                                labels_n,
                                len(
                                    role_active
                                ),
                            ),
                    }
                )

            # -----------------------------------------------------------------
            # Build unique undirected neighborhoods.
            # -----------------------------------------------------------------

            neighbors = build_neighbor_sets(
                src,
                dst,
            )

            connected_count = 0

            usable_neighbor_count = 0

            all_neighbors_have_text_count = 0

            degree_values = []

            text_neighbor_fraction_values = []

            label_neighbor_fraction_values = []

            for node_index in (
                startup_indices
            ):

                node_neighbors = (
                    neighbors.get(
                        int(
                            node_index
                        ),
                        set(),
                    )
                )

                degree = len(
                    node_neighbors
                )

                degree_values.append(
                    degree
                )

                if degree == 0:
                    continue

                connected_count += 1

                neighbor_array = np.asarray(
                    sorted(
                        node_neighbors
                    ),
                    dtype=np.int64,
                )

                neighbor_text = (
                    has_text[
                        neighbor_array
                    ]
                )

                neighbor_labels = (
                    has_labels[
                        neighbor_array
                    ]
                )

                text_fraction = float(
                    neighbor_text.mean()
                )

                label_fraction = float(
                    neighbor_labels.mean()
                )

                text_neighbor_fraction_values.append(
                    text_fraction
                )

                label_neighbor_fraction_values.append(
                    label_fraction
                )

                if bool(
                    neighbor_text.any()
                ):

                    usable_neighbor_count += 1

                if bool(
                    neighbor_text.all()
                ):

                    all_neighbors_have_text_count += 1

            degree_array = np.asarray(
                degree_values,
                dtype=np.int64,
            )

            connected_degree = degree_array[
                degree_array
                >
                0
            ]

            connectivity_rows.append(
                {
                    "origin":
                        origin,

                    "track":
                        track_name,

                    "primary_cases":
                        len(
                            startup_indices
                        ),

                    "connected_cases":
                        connected_count,

                    "connected_pct":
                        pct(
                            connected_count,
                            len(
                                startup_indices
                            ),
                        ),

                    "isolated_cases":
                        (
                            len(
                                startup_indices
                            )
                            -
                            connected_count
                        ),

                    "isolated_pct":
                        pct(
                            (
                                len(
                                    startup_indices
                                )
                                -
                                connected_count
                            ),
                            len(
                                startup_indices
                            ),
                        ),
                }
            )

            neighbor_rows.append(
                {
                    "origin":
                        origin,

                    "track":
                        track_name,

                    "connected_primary_cases":
                        connected_count,

                    "connected_with_at_least_one_text_neighbor":
                        usable_neighbor_count,

                    "usable_neighbor_pct_of_connected":
                        pct(
                            usable_neighbor_count,
                            connected_count,
                        ),

                    "connected_with_all_neighbors_text":
                        all_neighbors_have_text_count,

                    "all_neighbors_text_pct_of_connected":
                        pct(
                            all_neighbors_have_text_count,
                            connected_count,
                        ),

                    "mean_neighbor_text_fraction":
                        (
                            float(
                                np.mean(
                                    text_neighbor_fraction_values
                                )
                            )
                            if
                            text_neighbor_fraction_values
                            else
                            float("nan")
                        ),

                    "mean_neighbor_label_fraction":
                        (
                            float(
                                np.mean(
                                    label_neighbor_fraction_values
                                )
                            )
                            if
                            label_neighbor_fraction_values
                            else
                            float("nan")
                        ),
                }
            )

            degree_rows.append(
                {
                    "origin":
                        origin,

                    "track":
                        track_name,

                    "primary_cases":
                        len(
                            degree_array
                        ),

                    "degree_zero_cases":
                        int(
                            (
                                degree_array
                                ==
                                0
                            ).sum()
                        ),

                    "degree_mean_all":
                        float(
                            degree_array.mean()
                        ),

                    "degree_median_all":
                        float(
                            np.median(
                                degree_array
                            )
                        ),

                    "connected_degree_mean":
                        (
                            float(
                                connected_degree.mean()
                            )
                            if
                            len(
                                connected_degree
                            )
                            >
                            0
                            else
                            float("nan")
                        ),

                    "connected_degree_median":
                        (
                            float(
                                np.median(
                                    connected_degree
                                )
                            )
                            if
                            len(
                                connected_degree
                            )
                            >
                            0
                            else
                            float("nan")
                        ),

                    "connected_degree_p95":
                        (
                            float(
                                np.percentile(
                                    connected_degree,
                                    95,
                                )
                            )
                            if
                            len(
                                connected_degree
                            )
                            >
                            0
                            else
                            float("nan")
                        ),

                    "connected_degree_max":
                        (
                            int(
                                connected_degree.max()
                            )
                            if
                            len(
                                connected_degree
                            )
                            >
                            0
                            else
                            0
                        ),
                }
            )

            print()

            print(
                f"{origin} / {track_name}"
            )

            print(
                f"  masked directed edges: "
                f"{int(mask.sum()):,}"
            )

            print(
                f"  primary cases:         "
                f"{len(startup_indices):,}"
            )

            print(
                f"  connected:             "
                f"{connected_count:,} "
                f"({pct(connected_count, len(startup_indices)):.4f}%)"
            )

            print(
                f"  isolated:              "
                f"{len(startup_indices) - connected_count:,}"
            )

            print(
                f"  connected with >=1 "
                f"text neighbor:      "
                f"{usable_neighbor_count:,} "
                f"({pct(usable_neighbor_count, connected_count):.4f}%)"
            )

            if len(
                connected_degree
            ) > 0:

                print(
                    f"  connected median degree:"
                    f" {np.median(connected_degree):.2f}"
                )

                print(
                    f"  connected p95 degree:   "
                    f"{np.percentile(connected_degree, 95):.2f}"
                )

    active_node_df = pd.DataFrame(
        active_node_rows
    )

    connectivity_df = pd.DataFrame(
        connectivity_rows
    )

    neighbor_df = pd.DataFrame(
        neighbor_rows
    )

    degree_df = pd.DataFrame(
        degree_rows
    )

    # =========================================================================
    # 6. Feasibility conclusions
    # =========================================================================

    separator("-")

    print(
        "8.1.3.6 — FEASIBILITY CONCLUSIONS"
    )

    separator("-")

    # A graph branch is technically content-usable if every connected
    # primary case has at least one text-encodable structural neighbor.
    all_connected_content_usable = bool(
        (
            neighbor_df[
                "connected_primary_cases"
            ]
            ==
            neighbor_df[
                "connected_with_at_least_one_text_neighbor"
            ]
        ).all()
    )

    require(
        all_connected_content_usable,
        (
            "At least one development graph track "
            "contains connected primary cases without "
            "a text-encodable neighbor."
        ),
    )

    strict_connectivity = (
        connectivity_df[
            connectivity_df[
                "track"
            ]
            ==
            "strict_temporal"
        ][
            "connected_pct"
        ]
        .mean()
    )

    founder_connectivity = (
        connectivity_df[
            connectivity_df[
                "track"
            ]
            ==
            "founder_proxy"
        ][
            "connected_pct"
        ]
        .mean()
    )

    print(
        "Every connected primary case has "
        ">=1 text-encodable neighbor: PASS"
    )

    print(
        f"Mean strict-track primary connectivity: "
        f"{strict_connectivity:.4f}%"
    )

    print(
        f"Mean founder-track primary connectivity: "
        f"{founder_connectivity:.4f}%"
    )

    print()

    print(
        "Architectural implication:"
    )

    print(
        "  Content branch must remain independently "
        "usable for isolated startups."
    )

    print(
        "  Graph branch should act as additive "
        "relational evidence, not as the only "
        "startup representation."
    )

    # =========================================================================
    # 7. Persist
    # =========================================================================

    separator("-")

    print(
        "8.1.3.7 — PERSIST AUDIT ARTIFACTS"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    active_node_df.to_csv(
        ACTIVE_NODE_OUTPUT,
        index=False,
    )

    connectivity_df.to_csv(
        PRIMARY_CONNECTIVITY_OUTPUT,
        index=False,
    )

    neighbor_df.to_csv(
        NEIGHBOR_OUTPUT,
        index=False,
    )

    degree_df.to_csv(
        DEGREE_OUTPUT,
        index=False,
    )

    summary = {
        "schema_version":
            "PHASE_8_1_3_GRAPH_FEASIBILITY_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            (
                "DEVELOPMENT_ONLY_PRE_ARCHITECTURE_"
                "GRAPH_FEASIBILITY_AUDIT"
            ),

        "development_origins":
            DEVELOPMENT_ORIGINS,

        "April_graph_connectivity_analyzed":
            False,

        "all_connected_primary_cases_have_text_neighbor":
            all_connected_content_usable,

        "mean_primary_connectivity_pct": {
            "strict_temporal":
                float(
                    strict_connectivity
                ),

            "founder_proxy":
                float(
                    founder_connectivity
                ),
        },

        "architecture_implications": [
            (
                "content encoder must produce a valid "
                "startup representation without graph "
                "neighbors"
            ),
            (
                "graph encoder should provide additive "
                "relational information"
            ),
            (
                "identity-indexed graph initialization "
                "remains prohibited"
            ),
            (
                "relation ablations must use frozen "
                "Phase-8.0.6 masks"
            ),
        ],

        "architecture_selected":
            False,

        "model_constructed":
            False,

        "training_performed":
            False,

        "ranking_metrics_scored":
            False,

        "April_performance_inspected":
            False,

        "next_step":
            (
                "Freeze a small architecture-candidate "
                "set for content-only and inductive "
                "heterogeneous graph encoders."
            ),
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    manifest = {
        "phase":
            "8.1.3",

        "status":
            "COMPLETE",

        "authoritative_inputs":
            source_hashes,

        "outputs": [
            str(
                ACTIVE_NODE_OUTPUT
            ),
            str(
                PRIMARY_CONNECTIVITY_OUTPUT
            ),
            str(
                NEIGHBOR_OUTPUT
            ),
            str(
                DEGREE_OUTPUT
            ),
            str(
                SUMMARY_OUTPUT
            ),
        ],

        "prohibited_operations": [
            "architecture_selection",
            "model_construction",
            "training",
            "inference",
            "ranking_scoring",
            "model_selection",
            "April_graph_connectivity_analysis",
            "April_performance_inspection",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    hashes = {
        str(
            ACTIVE_NODE_OUTPUT
        ):
            sha256_file(
                ACTIVE_NODE_OUTPUT
            ),

        str(
            PRIMARY_CONNECTIVITY_OUTPUT
        ):
            sha256_file(
                PRIMARY_CONNECTIVITY_OUTPUT
            ),

        str(
            NEIGHBOR_OUTPUT
        ):
            sha256_file(
                NEIGHBOR_OUTPUT
            ),

        str(
            DEGREE_OUTPUT
        ):
            sha256_file(
                DEGREE_OUTPUT
            ),

        str(
            SUMMARY_OUTPUT
        ):
            sha256_file(
                SUMMARY_OUTPUT
            ),

        str(
            MANIFEST_OUTPUT
        ):
            sha256_file(
                MANIFEST_OUTPUT
            ),
    }

    write_json(
        HASH_OUTPUT,
        hashes,
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    separator()

    print(
        "PHASE 8.1.3 INDUCTIVE GRAPH "
        "FEASIBILITY AUDIT COMPLETE"
    )

    separator()

    print()

    print(
        "DEVELOPMENT FOLDS ANALYZED:       "
        "T57 / T58 / T59"
    )

    print(
        "APRIL CONNECTIVITY ANALYZED:      NO"
    )

    print(
        "CONNECTED CASE CONTENT USABILITY: PASS"
    )

    print(
        "CONTENT-ONLY FALLBACK REQUIRED:   YES"
    )

    print(
        "GRAPH BRANCH ROLE:                "
        "ADDITIVE RELATIONAL EVIDENCE"
    )

    print()

    print(
        "ARCHITECTURE SELECTED:            NO"
    )

    print(
        "MODEL CONSTRUCTED:                NO"
    )

    print(
        "TRAINING PERFORMED:               NO"
    )

    print(
        "APRIL PERFORMANCE INSPECTED:      NO"
    )

    print()

    print(
        "Next:"
    )

    print(
        "Phase 8.1.4 — freeze the "
        "small architecture-candidate set."
    )

    print()

    print(
        f"Outputs written to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()