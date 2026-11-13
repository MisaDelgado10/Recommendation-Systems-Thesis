from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 8.0.6 — FREEZE DUAL-TRACK GRAPH INFORMATION POLICY V1
# =============================================================================
#
# Scientific role:
#
#   OUTCOME-BLIND GRAPH INFORMATION-POLICY FREEZE
#
# Two graph tracks are frozen:
#
# A) STRICT_TEMPORAL
#
#    - timestamped acquisition edges only;
#    - observation date <= prediction origin;
#    - direct target-pair structural edges removed;
#    - shared-founder edges excluded because historical relation
#      observation time is unavailable.
#
# B) FOUNDER_PROXY
#
#    - all STRICT_TEMPORAL acquisition edges;
#    - shared-founder edges may additionally be used only when:
#
#          both endpoint founded_on values are known
#          AND both endpoint founding dates <= prediction origin
#          AND the edge is not a direct target-pair edge
#
#    - these founder edges are NOT called historically observed;
#    - they are called entity-existence-compatible founder edges.
#
# This script:
#
#   - does NOT modify Phase 3;
#   - does NOT load a model;
#   - does NOT score any ranking metric;
#   - does NOT generate negatives;
#   - does NOT perform model selection;
#   - does NOT inspect April confirmatory outcomes.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

RAW_DIR = Path(
    "data/raw"
)

TEMPORAL_PATH = Path(
    "data/experimental/phase_2/"
    "temporal/interactions_itrs_temporal.parquet"
)

CANONICAL_PATH = Path(
    "data/processed/interactions.parquet"
)

PHASE3_GRAPH_DIR = Path(
    "data/experimental/phase_3/graph"
)

PHASE3_MODEL_DIR = Path(
    "data/experimental/phase_3/model_ready"
)

EDGES_PATH = (
    PHASE3_GRAPH_DIR
    / "edges.parquet"
)

EDGE_MANIFEST_PATH = (
    PHASE3_MODEL_DIR
    / "edge_manifest.parquet"
)

EDGE_INDEX_PATH = (
    PHASE3_MODEL_DIR
    / "edge_index.npy"
)

EDGE_TYPE_PATH = (
    PHASE3_MODEL_DIR
    / "edge_type.npy"
)

ORIGINAL_MASK_PATH = (
    PHASE3_MODEL_DIR
    / "graph_variant_masks.npz"
)


OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_6_dual_track_graph_policy"
)

MASK_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_6_dual_track_graph_masks_V1.npz"
)

ORIGIN_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_6_origin_graph_summary_V1.csv"
)

FOUNDER_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_6_founder_proxy_summary_V1.csv"
)

RELATION_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_6_relation_summary_V1.csv"
)

POLICY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_6_graph_information_policy_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_6_graph_mask_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_6_derived_artifact_sha256_V1.json"
)


# =============================================================================
# FROZEN EXPECTATIONS
# =============================================================================

EXPECTED_PHASE3_SHA256 = (
    "895871308de416b12d001550838cdb276"
    "3c87cd897774c6a55539a0ebb9a1ee2"
)

EXPECTED_EDGES = 158_818

EXPECTED_FOUNDER_EDGES = 94_818
EXPECTED_ACQUIRED_EDGES = 32_000
EXPECTED_ACQUIRED_BY_EDGES = 32_000

EXPECTED_STRICT_COUNTS = {
    "T57": 60_144,
    "T58": 61_358,
    "T59": 62_554,
    "APRIL_CONFIRMATORY": 63_964,
}


# =============================================================================
# ORIGINS
# =============================================================================

ORIGINS = [
    {
        "label": "T57",
        "history_end": "2025-03-31",
        "target_source": "temporal",
        "segment_number": 57,
    },
    {
        "label": "T58",
        "history_end": "2025-06-30",
        "target_source": "temporal",
        "segment_number": 58,
    },
    {
        "label": "T59",
        "history_end": "2025-09-30",
        "target_source": "temporal",
        "segment_number": 59,
    },
    {
        "label": "APRIL_CONFIRMATORY",
        "history_end": "2026-03-31",
        "target_source": "canonical_april",
        "segment_number": None,
    },
]


# =============================================================================
# UTILITIES
# =============================================================================

def separator(
    char="=",
    width=118,
):
    print(
        char * width
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


def ordered_pair(
    a,
    b,
):

    a = str(a)
    b = str(b)

    if a <= b:
        return (a, b)

    return (b, a)


def clean_string(
    series,
):

    result = (
        series
        .astype("string")
        .str.strip()
    )

    return result.mask(
        result.eq("")
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    separator()

    print(
        "PHASE 8.0.6 — "
        "FREEZE DUAL-TRACK GRAPH "
        "INFORMATION POLICY V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "OUTCOME-BLIND GRAPH POLICY FREEZE"
    )

    print(
        "Model loaded:                NO"
    )

    print(
        "Checkpoint loaded:           NO"
    )

    print(
        "Inference executed:          NO"
    )

    print(
        "Ranking metrics scored:      NO"
    )

    print(
        "Negatives generated:         NO"
    )

    print(
        "Model selection performed:   NO"
    )

    print(
        "Phase-3 graph modified:      NO"
    )

    print(
        "April outcomes inspected:    NO"
    )

    # =========================================================================
    # 1. Source integrity
    # =========================================================================

    separator("-")

    print(
        "8.0.6.1 — SOURCE EXISTENCE "
        "AND INTEGRITY"
    )

    separator("-")

    required_paths = [
        TEMPORAL_PATH,
        CANONICAL_PATH,
        EDGES_PATH,
        EDGE_MANIFEST_PATH,
        EDGE_INDEX_PATH,
        EDGE_TYPE_PATH,
        ORIGINAL_MASK_PATH,
    ]

    source_hashes = {}

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        observed_hash = sha256_file(
            path
        )

        source_hashes[
            str(path)
        ] = observed_hash

        print(
            f"FOUND  {path}"
        )

        print(
            f"SHA256 {observed_hash}"
        )

    if (
        source_hashes[
            str(EDGES_PATH)
        ]
        != EXPECTED_PHASE3_SHA256
    ):

        raise ValueError(
            "Frozen Phase-3 graph SHA drift."
        )

    # =========================================================================
    # 2. Load graph + model ordering
    # =========================================================================

    separator("-")

    print(
        "8.0.6.2 — FROZEN GRAPH / "
        "MODEL POSITION INTEGRITY"
    )

    separator("-")

    edges = pd.read_parquet(
        EDGES_PATH
    )

    edge_manifest = pd.read_parquet(
        EDGE_MANIFEST_PATH
    )

    edge_index = np.load(
        EDGE_INDEX_PATH,
        mmap_mode="r",
    )

    edge_type = np.load(
        EDGE_TYPE_PATH,
        mmap_mode="r",
    )

    if len(
        edges
    ) != EXPECTED_EDGES:

        raise ValueError(
            "Phase-3 edge count drift."
        )

    if len(
        edge_manifest
    ) != EXPECTED_EDGES:

        raise ValueError(
            "Edge-manifest count drift."
        )

    if edge_index.shape != (
        2,
        EXPECTED_EDGES,
    ):

        raise ValueError(
            "edge_index shape drift."
        )

    if edge_type.shape != (
        EXPECTED_EDGES,
    ):

        raise ValueError(
            "edge_type shape drift."
        )

    if not np.array_equal(
        edge_manifest[
            "edge_position"
        ].to_numpy(
            dtype=np.int64
        ),
        np.arange(
            EXPECTED_EDGES,
            dtype=np.int64,
        ),
    ):

        raise ValueError(
            "edge_position ordering drift."
        )

    if (
        set(
            edge_manifest[
                "edge_id"
            ]
        )
        !=
        set(
            edges[
                "edge_id"
            ]
        )
    ):

        raise ValueError(
            "edge_manifest / edges mismatch."
        )

    position_lookup = (
        edge_manifest
        .set_index(
            "edge_id"
        )[
            "edge_position"
        ]
        .to_dict()
    )

    edges[
        "edge_position"
    ] = (
        edges[
            "edge_id"
        ]
        .map(
            position_lookup
        )
    )

    if edges[
        "edge_position"
    ].isna().any():

        raise ValueError(
            "At least one edge failed "
            "position resolution."
        )

    edges[
        "edge_position"
    ] = (
        edges[
            "edge_position"
        ]
        .astype(
            np.int64
        )
    )

    edges[
        "underlying_pair"
    ] = [
        ordered_pair(
            src,
            dst,
        )
        for src, dst
        in zip(
            edges[
                "underlying_src_id"
            ],
            edges[
                "underlying_dst_id"
            ],
        )
    ]

    edges[
        "first_observed_on_parsed"
    ] = pd.to_datetime(
        edges[
            "first_observed_on"
        ],
        errors="coerce",
    )

    relation_counts = (
        edges[
            "relation"
        ]
        .value_counts()
        .to_dict()
    )

    if (
        int(
            relation_counts.get(
                "shared_founder",
                0,
            )
        )
        != EXPECTED_FOUNDER_EDGES
    ):

        raise ValueError(
            "Founder edge count drift."
        )

    if (
        int(
            relation_counts.get(
                "acquired",
                0,
            )
        )
        != EXPECTED_ACQUIRED_EDGES
    ):

        raise ValueError(
            "Acquired edge count drift."
        )

    if (
        int(
            relation_counts.get(
                "acquired_by",
                0,
            )
        )
        != EXPECTED_ACQUIRED_BY_EDGES
    ):

        raise ValueError(
            "Acquired-by edge count drift."
        )

    print(
        "Frozen graph/model alignment: PASS"
    )

    # =========================================================================
    # 3. Reconstruct founder endpoint founding dates
    # =========================================================================

    separator("-")

    print(
        "8.0.6.3 — FOUNDER ENDPOINT "
        "FOUNDING-DATE RECONSTRUCTION"
    )

    separator("-")

    founder_edges = edges[
        edges[
            "relation"
        ]
        ==
        "shared_founder"
    ].copy()

    founder_entity_ids = set(
        founder_edges[
            "underlying_src_id"
        ].astype(str)
    )

    founder_entity_ids.update(
        founder_edges[
            "underlying_dst_id"
        ].astype(str)
    )

    company_files = sorted(
        RAW_DIR.glob(
            "companies*.csv"
        )
    )

    if not company_files:

        raise FileNotFoundError(
            "No companies*.csv files found."
        )

    company_frames = []

    for path in company_files:

        header = pd.read_csv(
            path,
            nrows=0,
        )

        if (
            "id"
            not in header.columns
            or
            "founded_on"
            not in header.columns
        ):

            raise ValueError(
                f"Missing required columns in {path}"
            )

        for chunk in pd.read_csv(
            path,
            usecols=[
                "id",
                "founded_on",
            ],
            dtype="string",
            chunksize=200_000,
            low_memory=False,
        ):

            chunk[
                "id"
            ] = clean_string(
                chunk[
                    "id"
                ]
            )

            subset = chunk[
                chunk[
                    "id"
                ].isin(
                    founder_entity_ids
                )
            ].copy()

            if len(
                subset
            ) == 0:

                continue

            subset[
                "source_file"
            ] = path.name

            company_frames.append(
                subset
            )

    if not company_frames:

        raise ValueError(
            "No founder endpoint Company "
            "records resolved."
        )

    companies = pd.concat(
        company_frames,
        ignore_index=True,
    )

    companies = (
        companies
        .sort_values(
            [
                "id",
                "source_file",
            ]
        )
        .drop_duplicates(
            subset=[
                "id"
            ],
            keep="last",
        )
        .copy()
    )

    companies[
        "founded_on_parsed"
    ] = pd.to_datetime(
        companies[
            "founded_on"
        ],
        errors="coerce",
    )

    founder_dates = (
        companies[
            [
                "id",
                "founded_on_parsed",
            ]
        ]
        .copy()
    )

    src_dates = (
        founder_dates.rename(
            columns={
                "id":
                    "underlying_src_id",

                "founded_on_parsed":
                    "src_founded_on",
            }
        )
    )

    dst_dates = (
        founder_dates.rename(
            columns={
                "id":
                    "underlying_dst_id",

                "founded_on_parsed":
                    "dst_founded_on",
            }
        )
    )

    edges = (
        edges.merge(
            src_dates,
            on="underlying_src_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            dst_dates,
            on="underlying_dst_id",
            how="left",
            validate="many_to_one",
        )
    )

    if len(
        edges
    ) != EXPECTED_EDGES:

        raise ValueError(
            "Founding-date merge changed "
            "graph edge count."
        )

    founder_after_merge = edges[
        edges[
            "relation"
        ]
        ==
        "shared_founder"
    ]

    both_founder_dates_known = (
        founder_after_merge[
            "src_founded_on"
        ].notna()
        &
        founder_after_merge[
            "dst_founded_on"
        ].notna()
    )

    print(
        f"Shared-founder edges:               "
        f"{len(founder_after_merge):,}"
    )

    print(
        f"Both endpoint founded_on known:     "
        f"{int(both_founder_dates_known.sum()):,}"
    )

    # =========================================================================
    # 4. Load target interaction universes
    # =========================================================================

    separator("-")

    print(
        "8.0.6.4 — TARGET-PAIR "
        "UNIVERSE RECONSTRUCTION"
    )

    separator("-")

    temporal = pd.read_parquet(
        TEMPORAL_PATH,
        columns=[
            "interaction_id",
            "announced_on",
            "investor_id",
            "startup_id",
            "segment_number",
        ],
    )

    temporal[
        "announced_on"
    ] = pd.to_datetime(
        temporal[
            "announced_on"
        ],
        errors="raise",
    )

    canonical = pd.read_parquet(
        CANONICAL_PATH,
        columns=[
            "interaction_id",
            "announced_on",
            "investor_id",
            "startup_id",
        ],
    )

    canonical[
        "announced_on"
    ] = pd.to_datetime(
        canonical[
            "announced_on"
        ],
        errors="raise",
    )

    # =========================================================================
    # 5. Construct masks for each prediction origin
    # =========================================================================

    mask_outputs = {}

    origin_rows = []
    founder_rows = []
    relation_rows = []

    for spec in ORIGINS:

        label = spec[
            "label"
        ]

        origin = pd.Timestamp(
            spec[
                "history_end"
            ]
        )

        separator("-")

        print(
            f"8.0.6 — {label} "
            f"GRAPH POLICY"
        )

        separator("-")

        if (
            spec[
                "target_source"
            ]
            ==
            "temporal"
        ):

            targets = temporal[
                temporal[
                    "segment_number"
                ]
                ==
                spec[
                    "segment_number"
                ]
            ].copy()

        else:

            targets = canonical[
                (
                    canonical[
                        "announced_on"
                    ]
                    >=
                    pd.Timestamp(
                        "2026-04-01"
                    )
                )
                &
                (
                    canonical[
                        "announced_on"
                    ]
                    <=
                    pd.Timestamp(
                        "2026-04-30"
                    )
                )
            ].copy()

        targets[
            "underlying_pair"
        ] = [
            ordered_pair(
                investor,
                startup,
            )
            for investor, startup
            in zip(
                targets[
                    "investor_id"
                ],
                targets[
                    "startup_id"
                ],
            )
        ]

        target_pairs = set(
            targets[
                "underlying_pair"
            ]
        )

        # ---------------------------------------------------------------------
        # Direct target-pair exclusion
        # ---------------------------------------------------------------------

        direct_pair_exclusion = (
            edges[
                "underlying_pair"
            ]
            .isin(
                target_pairs
            )
            .to_numpy(
                dtype=bool
            )
        )

        # ---------------------------------------------------------------------
        # Timestamped acquisition eligibility
        # ---------------------------------------------------------------------

        is_acquisition = (
            edges[
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

        observed_by_origin = (
            edges[
                "first_observed_on_parsed"
            ]
            .notna()
            &
            (
                edges[
                    "first_observed_on_parsed"
                ]
                <= origin
            )
        ).to_numpy(
            dtype=bool
        )

        acquisition_safe = (
            is_acquisition
            &
            observed_by_origin
            &
            ~direct_pair_exclusion
        )

        # ---------------------------------------------------------------------
        # STRICT TEMPORAL TRACK
        #
        # No founder edges are admitted.
        # ---------------------------------------------------------------------

        strict_temporal = (
            acquisition_safe.copy()
        )

        # ---------------------------------------------------------------------
        # FOUNDER-PROXY TRACK
        #
        # Necessary conditions:
        #
        # - shared_founder relation;
        # - both endpoint founding dates known;
        # - both endpoint founding dates <= origin;
        # - not a direct target-pair relation.
        #
        # Passing these conditions does NOT prove historical observation.
        # ---------------------------------------------------------------------

        is_founder = (
            edges[
                "relation"
            ]
            .eq(
                "shared_founder"
            )
            .to_numpy(
                dtype=bool
            )
        )

        founder_dates_known = (
            edges[
                "src_founded_on"
            ].notna()
            &
            edges[
                "dst_founded_on"
            ].notna()
        ).to_numpy(
            dtype=bool
        )

        founder_entities_exist = (
            (
                edges[
                    "src_founded_on"
                ]
                <= origin
            )
            &
            (
                edges[
                    "dst_founded_on"
                ]
                <= origin
            )
        ).fillna(
            False
        ).to_numpy(
            dtype=bool
        )

        founder_proxy_eligible = (
            is_founder
            &
            founder_dates_known
            &
            founder_entities_exist
            &
            ~direct_pair_exclusion
        )

        founder_proxy_track = (
            acquisition_safe
            |
            founder_proxy_eligible
        )

        # ---------------------------------------------------------------------
        # Assertions
        # ---------------------------------------------------------------------

        strict_count = int(
            strict_temporal.sum()
        )

        expected_strict = (
            EXPECTED_STRICT_COUNTS[
                label
            ]
        )

        if (
            strict_count
            != expected_strict
        ):

            raise ValueError(
                f"{label} strict count "
                f"{strict_count:,} "
                f"!= expected "
                f"{expected_strict:,}"
            )

        if np.any(
            strict_temporal
            &
            is_founder
        ):

            raise ValueError(
                f"{label}: founder edge "
                "entered strict track."
            )

        if np.any(
            strict_temporal
            &
            direct_pair_exclusion
        ):

            raise ValueError(
                f"{label}: direct target-pair "
                "edge entered strict graph."
            )

        if np.any(
            founder_proxy_track
            &
            direct_pair_exclusion
        ):

            raise ValueError(
                f"{label}: direct target-pair "
                "edge entered founder-proxy graph."
            )

        if np.any(
            founder_proxy_eligible
            &
            ~founder_dates_known
        ):

            raise ValueError(
                f"{label}: founder edge with "
                "unknown founded_on entered "
                "proxy track."
            )

        # ---------------------------------------------------------------------
        # Counts
        # ---------------------------------------------------------------------

        founder_total = int(
            is_founder.sum()
        )

        founder_proxy_count = int(
            founder_proxy_eligible.sum()
        )

        founder_unknown_date = int(
            (
                is_founder
                &
                ~founder_dates_known
            ).sum()
        )

        founder_future_entity = int(
            (
                is_founder
                &
                founder_dates_known
                &
                ~founder_entities_exist
            ).sum()
        )

        founder_direct_overlap = int(
            (
                is_founder
                &
                direct_pair_exclusion
            ).sum()
        )

        combined_count = int(
            founder_proxy_track.sum()
        )

        mask_outputs[
            f"{label}_strict_temporal"
        ] = strict_temporal

        mask_outputs[
            f"{label}_founder_proxy"
        ] = founder_proxy_track

        mask_outputs[
            f"{label}_founder_proxy_component"
        ] = founder_proxy_eligible

        # ---------------------------------------------------------------------
        # Summaries
        # ---------------------------------------------------------------------

        origin_rows.append(
            {
                "origin":
                    label,

                "history_end":
                    origin.date(),

                "target_events":
                    len(
                        targets
                    ),

                "target_unique_unordered_pairs":
                    len(
                        target_pairs
                    ),

                "strict_temporal_edges":
                    strict_count,

                "founder_proxy_edges":
                    combined_count,

                "founder_proxy_component_edges":
                    founder_proxy_count,

                "direct_pair_excluded_edges":
                    int(
                        direct_pair_exclusion.sum()
                    ),
            }
        )

        founder_rows.append(
            {
                "origin":
                    label,

                "founder_edges_total":
                    founder_total,

                "founder_edges_unknown_founded_on":
                    founder_unknown_date,

                "founder_edges_entity_not_yet_founded":
                    founder_future_entity,

                "founder_direct_target_pair_edges":
                    founder_direct_overlap,

                "founder_proxy_component_edges":
                    founder_proxy_count,

                "historically_observed_founder_edges_proven":
                    0,
            }
        )

        for relation in [
            "shared_founder",
            "acquired",
            "acquired_by",
        ]:

            relation_mask = (
                edges[
                    "relation"
                ]
                .eq(
                    relation
                )
                .to_numpy(
                    dtype=bool
                )
            )

            relation_rows.append(
                {
                    "origin":
                        label,

                    "relation":
                        relation,

                    "original_edges":
                        int(
                            relation_mask.sum()
                        ),

                    "strict_temporal_edges":
                        int(
                            (
                                strict_temporal
                                &
                                relation_mask
                            ).sum()
                        ),

                    "founder_proxy_track_edges":
                        int(
                            (
                                founder_proxy_track
                                &
                                relation_mask
                            ).sum()
                        ),
                }
            )

        print(
            f"Prediction origin:              "
            f"{origin.date()}"
        )

        print(
            f"Target events:                  "
            f"{len(targets):,}"
        )

        print(
            f"Direct-pair excluded edges:     "
            f"{int(direct_pair_exclusion.sum()):,}"
        )

        print()
        print(
            f"STRICT temporal edges:          "
            f"{strict_count:,}"
        )

        print()
        print(
            f"Founder edges total:            "
            f"{founder_total:,}"
        )

        print(
            f"Founder unknown founded_on:     "
            f"{founder_unknown_date:,}"
        )

        print(
            f"Founder entity future:          "
            f"{founder_future_entity:,}"
        )

        print(
            f"Founder direct-pair overlap:    "
            f"{founder_direct_overlap:,}"
        )

        print(
            f"Founder proxy component:        "
            f"{founder_proxy_count:,}"
        )

        print()
        print(
            f"FOUNDER-PROXY total edges:      "
            f"{combined_count:,}"
        )

    # =========================================================================
    # 6. Persist masks
    # =========================================================================

    separator("-")

    print(
        "8.0.6.6 — PERSIST DUAL-TRACK MASKS"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        MASK_OUTPUT,
        **mask_outputs,
    )

    origin_summary = pd.DataFrame(
        origin_rows
    )

    founder_summary = pd.DataFrame(
        founder_rows
    )

    relation_summary = pd.DataFrame(
        relation_rows
    )

    origin_summary.to_csv(
        ORIGIN_SUMMARY_OUTPUT,
        index=False,
    )

    founder_summary.to_csv(
        FOUNDER_SUMMARY_OUTPUT,
        index=False,
    )

    relation_summary.to_csv(
        RELATION_SUMMARY_OUTPUT,
        index=False,
    )

    policy = {
        "schema_version":
            "PHASE_8_0_6_DUAL_TRACK_GRAPH_POLICY_V1",

        "strict_temporal_track": {
            "status":
                "FROZEN",

            "allowed_relations": [
                "acquired",
                "acquired_by",
            ],

            "requirements": [
                "edge has timestamp",
                "edge first_observed_on <= prediction origin",
                "edge underlying pair is not a target pair",
            ],

            "shared_founder_allowed":
                False,

            "interpretation":
                (
                    "Historically auditable structural "
                    "graph under available timestamps."
                ),
        },

        "founder_proxy_track": {
            "status":
                "FROZEN",

            "base":
                "strict_temporal_track",

            "additional_relation":
                "shared_founder",

            "requirements": [
                "both endpoint founded_on values known",
                "both endpoint founded_on <= prediction origin",
                "edge underlying pair is not a target pair",
            ],

            "historically_observed_claim_allowed":
                False,

            "approved_term":
                (
                    "entity-existence-compatible "
                    "founder side information"
                ),

            "interpretation":
                (
                    "Current-snapshot founder relation "
                    "restricted using entity-existence "
                    "necessary conditions. "
                    "Historical relation observability "
                    "is not established."
                ),
        },

        "prohibited_claims": [
            (
                "Founder relation was known at the "
                "historical prediction origin."
            ),
            (
                "founded_on is the timestamp of the "
                "founder relation."
            ),
            (
                "Founder-proxy graph is fully "
                "temporally observed."
            ),
        ],

        "phase3_graph_modified":
            False,

        "model_selection_performed":
            False,

        "april_confirmatory_outcomes_inspected":
            False,
    }

    write_json(
        POLICY_OUTPUT,
        policy,
    )

    manifest = {
        "phase":
            "8.0.6",

        "authoritative_inputs":
            source_hashes,

        "origins":
            ORIGINS,

        "mask_names":
            sorted(
                mask_outputs.keys()
            ),

        "outputs": [
            str(
                MASK_OUTPUT
            ),
            str(
                ORIGIN_SUMMARY_OUTPUT
            ),
            str(
                FOUNDER_SUMMARY_OUTPUT
            ),
            str(
                RELATION_SUMMARY_OUTPUT
            ),
            str(
                POLICY_OUTPUT
            ),
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    derived_hashes = {
        str(MASK_OUTPUT):
            sha256_file(
                MASK_OUTPUT
            ),

        str(ORIGIN_SUMMARY_OUTPUT):
            sha256_file(
                ORIGIN_SUMMARY_OUTPUT
            ),

        str(FOUNDER_SUMMARY_OUTPUT):
            sha256_file(
                FOUNDER_SUMMARY_OUTPUT
            ),

        str(RELATION_SUMMARY_OUTPUT):
            sha256_file(
                RELATION_SUMMARY_OUTPUT
            ),

        str(POLICY_OUTPUT):
            sha256_file(
                POLICY_OUTPUT
            ),

        str(MANIFEST_OUTPUT):
            sha256_file(
                MANIFEST_OUTPUT
            ),
    }

    write_json(
        HASH_OUTPUT,
        derived_hashes,
    )

    # =========================================================================
    # 7. Final summary
    # =========================================================================

    separator()

    print(
        "PHASE 8.0.6 DUAL-TRACK "
        "GRAPH POLICY FREEZE COMPLETE"
    )

    separator()

    print()

    print(
        origin_summary.to_string(
            index=False
        )
    )

    print()

    print(
        founder_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "STRICT TEMPORAL GRAPH POLICY:       FROZEN"
    )

    print(
        "FOUNDER-PROXY GRAPH POLICY:         FROZEN"
    )

    print(
        "PHASE-3 GRAPH MODIFIED:             NO"
    )

    print(
        "MODEL SELECTION PERFORMED:          NO"
    )

    print(
        "APRIL CONFIRMATORY OUTCOMES TOUCHED:NO"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "Founder-proxy edges are NOT "
        "historically observed edges."
    )

    print(
        "They are current-snapshot founder "
        "relations whose endpoint entities are "
        "known to exist by the prediction origin."
    )

    print()

    print(
        "Next:"
    )

    print(
        "Freeze the Phase-8 development / "
        "validation / confirmatory evaluation "
        "contract."
    )

    print()

    print(
        f"Outputs written to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()