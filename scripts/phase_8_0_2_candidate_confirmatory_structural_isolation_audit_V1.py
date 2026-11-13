from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.0.2 — CANDIDATE CONFIRMATORY STRUCTURAL ISOLATION AUDIT V1
# =============================================================================
#
# Scientific role:
#
#   OUTCOME-BLIND STRUCTURAL-LEAKAGE AUDIT ONLY
#
# This script:
#   - does NOT load a model;
#   - does NOT load a checkpoint;
#   - does NOT run inference;
#   - does NOT score any ranking metric;
#   - does NOT generate negatives;
#   - does NOT modify the frozen Phase-3 graph;
#   - does NOT create a graph mask;
#   - does NOT freeze the confirmatory holdout.
#
# Objective:
#
#   Audit whether the candidate April-2026 confirmatory investor-startup
#   pairs are directly represented by any structural edge in the frozen
#   Phase-3 heterogeneous graph.
#
#   This mirrors the Phase-3 T60 underlying-pair leakage policy without
#   reopening or modifying the frozen Phase-3 graph.
#
# =============================================================================


CANONICAL_PATH = Path(
    "data/processed/interactions.parquet"
)

PHASE3_EDGES_PATH = Path(
    "data/experimental/phase_3/graph/edges.parquet"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_2_candidate_confirmatory_structural_isolation"
)

OVERLAP_PAIR_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_2_direct_structural_overlap_pairs_V1.csv"
)

RELATION_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_2_direct_structural_overlap_by_relation_V1.csv"
)

PROVENANCE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_2_graph_temporal_provenance_V1.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_2_structural_isolation_summary_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_2_analysis_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_2_derived_artifact_sha256_V1.json"
)


# -----------------------------------------------------------------------------
# Frozen / audited boundaries
# -----------------------------------------------------------------------------

HISTORY_END = pd.Timestamp(
    "2026-03-31"
)

CANDIDATE_HOLDOUT_START = pd.Timestamp(
    "2026-04-01"
)

CANDIDATE_HOLDOUT_END = pd.Timestamp(
    "2026-04-30"
)


# Phase-8.0.1 observed anchors
EXPECTED_APRIL_EVENTS = 6_622
EXPECTED_APRIL_UNIQUE_PAIRS = 6_582
EXPECTED_APRIL_PRIMARY_EVENTS = 1_931

# Frozen Phase-3 graph anchor
EXPECTED_PHASE3_EDGE_ROWS = 158_818


# -----------------------------------------------------------------------------
# Canonical fields
# -----------------------------------------------------------------------------

ID_COL = "interaction_id"
DATE_COL = "announced_on"
INVESTOR_COL = "investor_id"
STARTUP_COL = "startup_id"

CANONICAL_COLUMNS = [
    ID_COL,
    DATE_COL,
    INVESTOR_COL,
    STARTUP_COL,
]


# -----------------------------------------------------------------------------
# Frozen Phase-3 edge fields
# -----------------------------------------------------------------------------

EDGE_COLUMNS = [
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


def separator(char="=", width=110):
    print(char * width)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            default=str,
        )


def ordered_underlying_pair(a, b):
    """
    Match the conservative Phase-3 leakage policy:

    structural relations are compared against held-out investor-startup
    relationships using the unordered pair of underlying entity IDs.
    """

    a = str(a)
    b = str(b)

    if a <= b:
        return (a, b)

    return (b, a)


def pair_key(pair):
    return f"{pair[0]}||{pair[1]}"


def main():

    separator()
    print(
        "PHASE 8.0.2 — "
        "CANDIDATE CONFIRMATORY STRUCTURAL ISOLATION AUDIT V1"
    )
    separator()

    print()
    print(
        "Scientific role:             "
        "OUTCOME-BLIND STRUCTURAL-LEAKAGE AUDIT ONLY"
    )
    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Inference executed:          NO")
    print("Ranking metrics scored:      NO")
    print("Negatives generated:         NO")
    print("Phase-3 graph modified:      NO")
    print("Graph mask created:          NO")
    print("Holdout frozen:              NO")

    # =========================================================================
    # 1. Source existence
    # =========================================================================

    separator("-")
    print("8.0.2.1 — SOURCE EXISTENCE AND FINGERPRINTS")
    separator("-")

    for path in [
        CANONICAL_PATH,
        PHASE3_EDGES_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required source not found: {path}"
            )

        print(f"FOUND  {path}")
        print(
            f"SHA256 {sha256_file(path)}"
        )

    # =========================================================================
    # 2. Load canonical interaction history
    # =========================================================================

    separator("-")
    print(
        "8.0.2.2 — CANDIDATE HOLDOUT RECONSTRUCTION"
    )
    separator("-")

    interactions = pd.read_parquet(
        CANONICAL_PATH,
        columns=CANONICAL_COLUMNS,
    )

    interactions[DATE_COL] = pd.to_datetime(
        interactions[DATE_COL],
        errors="raise",
    )

    history = interactions[
        interactions[DATE_COL]
        <= HISTORY_END
    ].copy()

    april = interactions[
        (
            interactions[DATE_COL]
            >= CANDIDATE_HOLDOUT_START
        )
        & (
            interactions[DATE_COL]
            <= CANDIDATE_HOLDOUT_END
        )
    ].copy()

    print(
        f"History boundary:      <= "
        f"{HISTORY_END.date()}"
    )

    print(
        f"Candidate holdout:     "
        f"{CANDIDATE_HOLDOUT_START.date()} "
        f"-> "
        f"{CANDIDATE_HOLDOUT_END.date()}"
    )

    print(
        f"April events:          "
        f"{len(april):,}"
    )

    if len(april) != EXPECTED_APRIL_EVENTS:
        raise ValueError(
            "April event count differs from "
            "Phase-8.0.1."
        )

    # =========================================================================
    # 3. Reconstruct April novelty / cold-start semantics
    # =========================================================================

    separator("-")
    print(
        "8.0.2.3 — ENDPOINT-RELATIVE "
        "APRIL STRATUM RECONSTRUCTION"
    )
    separator("-")

    historical_investors = set(
        history[INVESTOR_COL].astype(str)
    )

    historical_startups = set(
        history[STARTUP_COL].astype(str)
    )

    historical_pairs = {
        (
            str(investor_id),
            str(startup_id),
        )
        for investor_id, startup_id
        in zip(
            history[INVESTOR_COL],
            history[STARTUP_COL],
        )
    }

    april[
        "new_to_investor_pair_at_origin"
    ] = [
        (
            str(investor_id),
            str(startup_id),
        )
        not in historical_pairs
        for investor_id, startup_id
        in zip(
            april[INVESTOR_COL],
            april[STARTUP_COL],
        )
    ]

    april[
        "cold_start_investor_at_origin"
    ] = ~april[
        INVESTOR_COL
    ].astype(str).isin(
        historical_investors
    )

    april[
        "cold_start_startup_at_origin"
    ] = ~april[
        STARTUP_COL
    ].astype(str).isin(
        historical_startups
    )

    primary_mask = (
        april[
            "new_to_investor_pair_at_origin"
        ]
        & ~april[
            "cold_start_investor_at_origin"
        ]
        & april[
            "cold_start_startup_at_origin"
        ]
    )

    primary_events = int(
        primary_mask.sum()
    )

    april_unique_pairs = int(
        april[
            [
                INVESTOR_COL,
                STARTUP_COL,
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"April unique directed pairs:  "
        f"{april_unique_pairs:,}"
    )

    print(
        f"Primary-stratum events:       "
        f"{primary_events:,}"
    )

    if (
        april_unique_pairs
        != EXPECTED_APRIL_UNIQUE_PAIRS
    ):
        raise ValueError(
            "April unique-pair count differs "
            "from Phase-8.0.1."
        )

    if (
        primary_events
        != EXPECTED_APRIL_PRIMARY_EVENTS
    ):
        raise ValueError(
            "April primary-stratum count differs "
            "from Phase-8.0.1."
        )

    print("Phase-8.0.1 anchors: PASS")

    # =========================================================================
    # 4. Build conservative unordered April holdout pair set
    # =========================================================================

    separator("-")
    print(
        "8.0.2.4 — APRIL UNDERLYING-PAIR HOLDOUT SET"
    )
    separator("-")

    april[
        "underlying_pair"
    ] = [
        ordered_underlying_pair(
            investor_id,
            startup_id,
        )
        for investor_id, startup_id
        in zip(
            april[INVESTOR_COL],
            april[STARTUP_COL],
        )
    ]

    april_underlying_pairs = set(
        april["underlying_pair"]
    )

    print(
        f"Unique directed investor-startup pairs: "
        f"{april_unique_pairs:,}"
    )

    print(
        f"Unique unordered underlying pairs:       "
        f"{len(april_underlying_pairs):,}"
    )

    # =========================================================================
    # 5. Load frozen Phase-3 structural graph
    # =========================================================================

    separator("-")
    print(
        "8.0.2.5 — FROZEN PHASE-3 GRAPH INTEGRITY"
    )
    separator("-")

    edges = pd.read_parquet(
        PHASE3_EDGES_PATH,
        columns=EDGE_COLUMNS,
    )

    print(
        f"Structural edge records: "
        f"{len(edges):,}"
    )

    if (
        len(edges)
        != EXPECTED_PHASE3_EDGE_ROWS
    ):
        raise ValueError(
            "Frozen Phase-3 structural edge count "
            "does not match closure record."
        )

    if edges["edge_id"].duplicated().any():
        raise ValueError(
            "Duplicate Phase-3 edge_id detected."
        )

    edges[
        "underlying_pair"
    ] = [
        ordered_underlying_pair(
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

    print("Frozen Phase-3 graph integrity: PASS")

    # =========================================================================
    # 6. Direct structural overlap with April targets
    # =========================================================================

    separator("-")
    print(
        "8.0.2.6 — DIRECT STRUCTURAL TARGET-PAIR OVERLAP"
    )
    separator("-")

    overlap_edges = edges[
        edges[
            "underlying_pair"
        ].isin(
            april_underlying_pairs
        )
    ].copy()

    overlap_pairs = set(
        overlap_edges[
            "underlying_pair"
        ]
    )

    print(
        f"Overlapping structural edge records: "
        f"{len(overlap_edges):,}"
    )

    print(
        f"Unique overlapping underlying pairs: "
        f"{len(overlap_pairs):,}"
    )

    april[
        "direct_structural_pair_overlap"
    ] = april[
        "underlying_pair"
    ].isin(
        overlap_pairs
    )

    overlapping_april_events = int(
        april[
            "direct_structural_pair_overlap"
        ].sum()
    )

    overlapping_primary_events = int(
        (
            primary_mask
            & april[
                "direct_structural_pair_overlap"
            ]
        ).sum()
    )

    print(
        f"April events on overlapping pairs:    "
        f"{overlapping_april_events:,}"
    )

    print(
        f"Primary-stratum events affected:      "
        f"{overlapping_primary_events:,}"
    )

    # =========================================================================
    # 7. Breakdown by structural relation
    # =========================================================================

    separator("-")
    print(
        "8.0.2.7 — OVERLAP BY RELATION / PROVENANCE"
    )
    separator("-")

    if len(overlap_edges) > 0:

        overlap_edges[
            "pair_key"
        ] = overlap_edges[
            "underlying_pair"
        ].map(
            pair_key
        )

        relation_summary = (
            overlap_edges.groupby(
                [
                    "relation",
                    "temporal_provenance",
                ],
                dropna=False,
            )
            .agg(
                edge_records=(
                    "edge_id",
                    "size",
                ),
                unique_underlying_pairs=(
                    "pair_key",
                    "nunique",
                ),
            )
            .reset_index()
            .sort_values(
                [
                    "unique_underlying_pairs",
                    "edge_records",
                ],
                ascending=False,
            )
        )

    else:

        relation_summary = pd.DataFrame(
            columns=[
                "relation",
                "temporal_provenance",
                "edge_records",
                "unique_underlying_pairs",
            ]
        )

    print(
        relation_summary.to_string(
            index=False
        )
    )

    # =========================================================================
    # 8. Global temporal provenance audit
    # =========================================================================

    separator("-")
    print(
        "8.0.2.8 — GRAPH TEMPORAL PROVENANCE AUDIT"
    )
    separator("-")

    provenance_summary = (
        edges.groupby(
            [
                "relation",
                "temporal_provenance",
            ],
            dropna=False,
        )
        .agg(
            edge_records=(
                "edge_id",
                "size",
            ),
            first_observed_non_null=(
                "first_observed_on",
                lambda s: int(
                    s.notna().sum()
                ),
            ),
            last_observed_non_null=(
                "last_observed_on",
                lambda s: int(
                    s.notna().sum()
                ),
            ),
        )
        .reset_index()
    )

    print(
        provenance_summary.to_string(
            index=False
        )
    )

    first_observed = pd.to_datetime(
        edges[
            "first_observed_on"
        ],
        errors="coerce",
    )

    dated_edges_after_origin = int(
        (
            first_observed
            > HISTORY_END
        ).sum()
    )

    print()
    print(
        "Timestamped edges first observed "
        f"after 2026-03-31: "
        f"{dated_edges_after_origin:,}"
    )

    unversioned_edges = int(
        (
            edges[
                "temporal_provenance"
            ]
            == "current_snapshot_unversioned"
        ).sum()
    )

    overlapping_unversioned_edges = int(
        (
            overlap_edges[
                "temporal_provenance"
            ]
            == "current_snapshot_unversioned"
        ).sum()
    )

    print(
        f"Current-snapshot unversioned edges: "
        f"{unversioned_edges:,}"
    )

    print(
        "Overlapping current-snapshot "
        f"unversioned edges: "
        f"{overlapping_unversioned_edges:,}"
    )

    # =========================================================================
    # 9. Build pair-level overlap artifact
    # =========================================================================

    pair_rows = []

    if len(overlap_pairs) > 0:

        overlap_relation_sets = (
            overlap_edges.assign(
                pair_key=overlap_edges[
                    "underlying_pair"
                ].map(
                    pair_key
                )
            )
            .groupby(
                "pair_key"
            )
            .agg(
                relations=(
                    "relation",
                    lambda s: "|".join(
                        sorted(
                            set(
                                s.astype(str)
                            )
                        )
                    ),
                ),
                temporal_provenance=(
                    "temporal_provenance",
                    lambda s: "|".join(
                        sorted(
                            set(
                                s.astype(str)
                            )
                        )
                    ),
                ),
                structural_edge_records=(
                    "edge_id",
                    "size",
                ),
            )
            .reset_index()
        )

        april_pair_events = (
            april.assign(
                pair_key=april[
                    "underlying_pair"
                ].map(
                    pair_key
                )
            )
            .groupby(
                "pair_key"
            )
            .agg(
                april_events=(
                    ID_COL,
                    "size",
                ),
                primary_stratum_events=(
                    "new_to_investor_pair_at_origin",
                    lambda s: 0,
                ),
            )
            .reset_index()
        )

        # Replace placeholder primary count with exact event-level aggregation.
        primary_by_pair = (
            april.loc[
                primary_mask
            ]
            .assign(
                pair_key=lambda d:
                    d[
                        "underlying_pair"
                    ].map(
                        pair_key
                    )
            )
            .groupby(
                "pair_key"
            )
            .size()
            .rename(
                "primary_stratum_events"
            )
            .reset_index()
        )

        april_pair_events = (
            april_pair_events
            .drop(
                columns=[
                    "primary_stratum_events"
                ]
            )
            .merge(
                primary_by_pair,
                on="pair_key",
                how="left",
            )
        )

        april_pair_events[
            "primary_stratum_events"
        ] = (
            april_pair_events[
                "primary_stratum_events"
            ]
            .fillna(0)
            .astype(int)
        )

        pair_table = (
            overlap_relation_sets.merge(
                april_pair_events,
                on="pair_key",
                how="left",
                validate="one_to_one",
            )
            .sort_values(
                [
                    "primary_stratum_events",
                    "april_events",
                    "pair_key",
                ],
                ascending=[
                    False,
                    False,
                    True,
                ],
            )
        )

    else:

        pair_table = pd.DataFrame(
            columns=[
                "pair_key",
                "relations",
                "temporal_provenance",
                "structural_edge_records",
                "april_events",
                "primary_stratum_events",
            ]
        )

    # =========================================================================
    # 10. Persist audit outputs
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pair_table.to_csv(
        OVERLAP_PAIR_OUTPUT,
        index=False,
    )

    relation_summary.to_csv(
        RELATION_OUTPUT,
        index=False,
    )

    provenance_summary.to_csv(
        PROVENANCE_OUTPUT,
        index=False,
    )

    requires_phase8_mask = (
        len(overlap_pairs) > 0
    )

    summary = {
        "schema_version":
            "PHASE_8_0_2_STRUCTURAL_ISOLATION_V1",

        "scientific_role":
            "OUTCOME_BLIND_STRUCTURAL_LEAKAGE_AUDIT",

        "model_loaded": False,
        "checkpoint_loaded": False,
        "inference_executed": False,
        "ranking_metrics_scored": False,
        "negatives_generated": False,
        "phase3_graph_modified": False,
        "graph_mask_created": False,
        "holdout_frozen": False,

        "history_end":
            str(
                HISTORY_END.date()
            ),

        "candidate_holdout_start":
            str(
                CANDIDATE_HOLDOUT_START.date()
            ),

        "candidate_holdout_end":
            str(
                CANDIDATE_HOLDOUT_END.date()
            ),

        "candidate_holdout_events":
            int(len(april)),

        "candidate_holdout_unique_directed_pairs":
            april_unique_pairs,

        "candidate_holdout_unique_unordered_pairs":
            int(
                len(
                    april_underlying_pairs
                )
            ),

        "primary_stratum_events":
            primary_events,

        "phase3_edge_records":
            int(len(edges)),

        "direct_structural_overlap_edge_records":
            int(
                len(
                    overlap_edges
                )
            ),

        "direct_structural_overlap_unique_pairs":
            int(
                len(
                    overlap_pairs
                )
            ),

        "april_events_on_direct_overlap_pairs":
            overlapping_april_events,

        "primary_events_on_direct_overlap_pairs":
            overlapping_primary_events,

        "timestamped_edges_first_observed_after_origin":
            dated_edges_after_origin,

        "current_snapshot_unversioned_edges":
            unversioned_edges,

        "overlapping_current_snapshot_unversioned_edges":
            overlapping_unversioned_edges,

        "phase8_confirmatory_graph_mask_required":
            requires_phase8_mask,

        "interpretation_boundary": (
            "Direct structural target-pair overlap "
            "is a leakage-control issue. "
            "Current-snapshot unversioned founder "
            "relations are a separate temporal-"
            "observability limitation and are not "
            "automatically equivalent to target-label "
            "leakage."
        ),

        "decision_status":
            (
                "MASK_REQUIRED_BEFORE_HOLDOUT_FREEZE"
                if requires_phase8_mask
                else
                "NO_DIRECT_STRUCTURAL_PAIR_OVERLAP"
            ),
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    manifest = {
        "phase":
            "8.0.2",

        "authoritative_inputs": [
            str(CANONICAL_PATH),
            str(PHASE3_EDGES_PATH),
        ],

        "candidate_holdout": {
            "start":
                str(
                    CANDIDATE_HOLDOUT_START.date()
                ),
            "end":
                str(
                    CANDIDATE_HOLDOUT_END.date()
                ),
            "status":
                "CANDIDATE_NOT_FROZEN",
        },

        "outputs": [
            str(OVERLAP_PAIR_OUTPUT),
            str(RELATION_OUTPUT),
            str(PROVENANCE_OUTPUT),
            str(SUMMARY_OUTPUT),
        ],

        "prohibited_operations": [
            "model_loading",
            "checkpoint_loading",
            "inference",
            "ranking_scoring",
            "negative_generation",
            "phase3_graph_modification",
            "holdout_freeze",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    derived_hashes = {
        str(OVERLAP_PAIR_OUTPUT):
            sha256_file(
                OVERLAP_PAIR_OUTPUT
            ),

        str(RELATION_OUTPUT):
            sha256_file(
                RELATION_OUTPUT
            ),

        str(PROVENANCE_OUTPUT):
            sha256_file(
                PROVENANCE_OUTPUT
            ),

        str(SUMMARY_OUTPUT):
            sha256_file(
                SUMMARY_OUTPUT
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
    # 11. Final audit summary
    # =========================================================================

    separator()
    print(
        "PHASE 8.0.2 AUDIT COMPLETE"
    )
    separator()

    print()
    print(
        f"Candidate holdout:               "
        f"{CANDIDATE_HOLDOUT_START.date()} "
        f"-> "
        f"{CANDIDATE_HOLDOUT_END.date()}"
    )

    print(
        f"April events:                    "
        f"{len(april):,}"
    )

    print(
        f"Primary-stratum events:          "
        f"{primary_events:,}"
    )

    print(
        f"Phase-3 structural edges:        "
        f"{len(edges):,}"
    )

    print(
        f"Direct overlap edge records:     "
        f"{len(overlap_edges):,}"
    )

    print(
        f"Direct overlap unique pairs:     "
        f"{len(overlap_pairs):,}"
    )

    print(
        f"April events affected:           "
        f"{overlapping_april_events:,}"
    )

    print(
        f"Primary events affected:         "
        f"{overlapping_primary_events:,}"
    )

    print(
        f"Dated edges after origin:        "
        f"{dated_edges_after_origin:,}"
    )

    print(
        f"Unversioned founder-side edges:  "
        f"{unversioned_edges:,}"
    )

    print()

    if requires_phase8_mask:

        print(
            "PHASE-8 CONFIRMATORY GRAPH MASK "
            "REQUIRED: YES"
        )

        print(
            "Do NOT freeze or evaluate the "
            "confirmatory holdout yet."
        )

    else:

        print(
            "PHASE-8 CONFIRMATORY GRAPH MASK "
            "REQUIRED: NO"
        )

        print(
            "No direct Phase-3 structural edge "
            "matches an April holdout underlying pair."
        )

    print()
    print(
        "CONFIRMATORY HOLDOUT FROZEN: NO"
    )

    print()
    print(
        f"Outputs written to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()