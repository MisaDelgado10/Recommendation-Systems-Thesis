from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# P2 — STRICT NEW-TO-INVESTOR DISCOVERY CERTIFICATION
# =============================================================================
#
# Purpose
# -------
# Demonstrate that the already-frozen Phase-6 candidate sets constitute a
# strict sampled new-to-investor discovery evaluation for those test positives
# that are themselves new-to-investor.
#
# IMPORTANT
# ---------
# - NO training
# - NO inference
# - NO rescoring
# - NO checkpoint access required
# - NO candidate regeneration
# - NO T60 event is treated as history
#
# History = T0-T59 only.
#
# Scientific change relative to generic Phase-6 reporting:
#   restrict reported positives to new-to-investor cases.
#
# Candidate sets themselves are inherited unchanged from frozen Phase 6.
# =============================================================================


ROOT = Path(__file__).resolve().parents[2]

NUM_INVESTORS = 165_975
NUM_STARTUPS = 311_589
EXPECTED_TEST_CASES = 20_264
EXPECTED_DISCOVERY_CASES = 16_446
EXPECTED_CANDIDATES = 100


# =============================================================================
# Input paths
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

FINAL_BINDING = (
    ROOT
    / "data"
    / "experimental"
    / "phase_6"
    / "final_test"
    / "final_t60_test_case_binding.parquet"
)

FINAL_CANDIDATES = (
    ROOT
    / "data"
    / "experimental"
    / "phase_6"
    / "final_test"
    / "final_t60_test_candidate_startup_local.npy"
)

TEMPORAL_SPLIT = (
    ROOT
    / "data"
    / "experimental"
    / "phase_2"
    / "model_ready"
    / "interactions_itrs_temporal_split.parquet"
)

NODE_INDEX = (
    ROOT
    / "data"
    / "experimental"
    / "phase_3"
    / "model_ready"
    / "node_index.parquet"
)


# =============================================================================
# Output paths
# =============================================================================

OUT_CASE_AUDIT = (
    PROPOSAL_ROOT
    / "11_discovery_candidate_case_audit.csv"
)

OUT_CANDIDATE_STATS = (
    PROPOSAL_ROOT
    / "12_discovery_candidate_statistics.csv"
)

OUT_DISCOVERY_METRICS = (
    PROPOSAL_ROOT
    / "13_new_to_investor_discovery_metrics.csv"
)

OUT_INVARIANTS = (
    PROPOSAL_ROOT
    / "14_p2_discovery_invariants.csv"
)

OUT_SUMMARY = (
    PROPOSAL_ROOT
    / "p2_discovery_diagnostic_summary.json"
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


def normalize_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    normalized = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "false": False,
        "0": False,
        "no": False,
    }

    result = normalized.map(mapping)

    require(
        result.notna().all(),
        f"Could not normalize Boolean column {series.name}.",
    )

    return result.astype(bool)


def membership_in_sorted(
    sorted_values: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """
    Vectorized exact membership lookup.
    """

    query = np.asarray(
        query,
        dtype=np.int64,
    )

    positions = np.searchsorted(
        sorted_values,
        query,
        side="left",
    )

    in_range = positions < len(sorted_values)

    clipped = np.minimum(
        positions,
        len(sorted_values) - 1,
    )

    return (
        in_range
        & (
            sorted_values[clipped]
            == query
        )
    )


def metric_row(
    frame: pd.DataFrame,
    label: str,
) -> dict:
    n = len(frame)

    return {
        "evaluation_group": label,
        "n": int(n),
        "HR@10": (
            float(frame["HR@10"].mean())
            if n
            else np.nan
        ),
        "NDCG@10": (
            float(frame["NDCG@10"].mean())
            if n
            else np.nan
        ),
        "mean_positive_rank": (
            float(frame["positive_rank"].mean())
            if n
            else np.nan
        ),
        "median_positive_rank": (
            float(frame["positive_rank"].median())
            if n
            else np.nan
        ),
    }


def pool_summary(
    frame: pd.DataFrame,
    label: str,
) -> dict:

    values = frame[
        "discovery_candidate_universe_size"
    ].to_numpy(
        dtype=np.float64
    )

    negative_values = frame[
        "discovery_negative_pool_size"
    ].to_numpy(
        dtype=np.float64
    )

    return {
        "group": label,
        "n": int(len(frame)),

        "candidate_pool_min":
            int(np.min(values)),

        "candidate_pool_median":
            float(np.median(values)),

        "candidate_pool_mean":
            float(np.mean(values)),

        "candidate_pool_p75":
            float(np.percentile(values, 75)),

        "candidate_pool_p90":
            float(np.percentile(values, 90)),

        "candidate_pool_p95":
            float(np.percentile(values, 95)),

        "candidate_pool_max":
            int(np.max(values)),

        "negative_pool_min":
            int(np.min(negative_values)),

        "mean_sampled_cold_candidates":
            float(
                frame[
                    "sampled_cold_candidate_count"
                ].mean()
            ),

        "mean_sampled_warm_candidates":
            float(
                frame[
                    "sampled_warm_candidate_count"
                ].mean()
            ),

        "mean_sampled_cold_share":
            float(
                frame[
                    "sampled_cold_candidate_share"
                ].mean()
            ),
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:

    banner(
        "P2 — STRICT NEW-TO-INVESTOR DISCOVERY CERTIFICATION"
    )

    # =========================================================================
    # 1. Input existence
    # =========================================================================

    for path in [
        P1_EVENTS,
        FINAL_BINDING,
        FINAL_CANDIDATES,
        TEMPORAL_SPLIT,
        NODE_INDEX,
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
    # 2. Load P1 diagnostic events
    # =========================================================================

    events = pd.read_csv(
        P1_EVENTS
    )

    require(
        len(events) == EXPECTED_TEST_CASES,
        (
            f"Expected {EXPECTED_TEST_CASES:,} P1 events; "
            f"found {len(events):,}."
        ),
    )

    require(
        events["interaction_id"].is_unique,
        "P1 interaction IDs are not unique.",
    )

    events["new_to_investor"] = (
        normalize_bool(
            events["new_to_investor"]
        )
    )

    discovery = (
        events.loc[
            events["new_to_investor"]
        ]
        .copy()
    )

    require(
        len(discovery)
        == EXPECTED_DISCOVERY_CASES,
        (
            "Expected exactly "
            f"{EXPECTED_DISCOVERY_CASES:,} "
            "new-to-investor test cases; "
            f"found {len(discovery):,}."
        ),
    )

    print()
    print(
        f"All Phase-6 test cases:       "
        f"{len(events):,}"
    )
    print(
        f"Strict discovery positives:   "
        f"{len(discovery):,}"
    )
    print(
        f"Discovery share of test:      "
        f"{len(discovery) / len(events):.2%}"
    )

    # =========================================================================
    # 3. Load frozen Phase-6 binding and candidate matrix
    # =========================================================================

    binding = pd.read_parquet(
        FINAL_BINDING
    )

    candidate_matrix = np.load(
        FINAL_CANDIDATES,
        mmap_mode="r",
    )

    require(
        len(binding) == EXPECTED_TEST_CASES,
        "Frozen binding row-count drift.",
    )

    require(
        candidate_matrix.shape
        == (
            EXPECTED_TEST_CASES,
            EXPECTED_CANDIDATES,
        ),
        (
            "Frozen candidate matrix shape drift: "
            f"{candidate_matrix.shape}"
        ),
    )

    require(
        binding["interaction_id"].is_unique,
        "Frozen binding interaction IDs not unique.",
    )

    require(
        binding[
            "test_case_position"
        ].is_unique,
        "test_case_position not unique.",
    )

    # Join P1 event labels to authoritative Phase-6 positions.

    discovery = discovery.merge(
        binding[
            [
                "interaction_id",
                "test_case_position",
                "matrix_row_index",
                "investor_global",
                "positive_startup_local",
            ]
        ],
        on="interaction_id",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_binding"),
    )

    require(
        len(discovery)
        == EXPECTED_DISCOVERY_CASES,
        "P1 / Phase-6 binding join changed discovery count.",
    )

    discovery = (
        discovery
        .sort_values(
            "test_case_position"
        )
        .reset_index(drop=True)
    )

    test_positions = discovery[
        "test_case_position"
    ].to_numpy(
        dtype=np.int64
    )

    discovery_candidates = np.asarray(
        candidate_matrix[
            test_positions
        ],
        dtype=np.int64,
    )

    require(
        discovery_candidates.shape
        == (
            EXPECTED_DISCOVERY_CASES,
            EXPECTED_CANDIDATES,
        ),
        "Discovery candidate subset shape drift.",
    )

    # Positive must remain candidate position 0.

    positive_local = discovery[
        "positive_startup_local"
    ].to_numpy(
        dtype=np.int64
    )

    require(
        np.array_equal(
            discovery_candidates[:, 0],
            positive_local,
        ),
        (
            "Frozen positive is not candidate position 0 "
            "for all discovery cases."
        ),
    )

    # Candidate uniqueness check.

    sorted_candidates = np.sort(
        discovery_candidates,
        axis=1,
    )

    duplicate_mask = (
        sorted_candidates[:, 1:]
        == sorted_candidates[:, :-1]
    )

    require(
        not bool(
            np.any(
                duplicate_mask
            )
        ),
        "At least one discovery case has duplicate candidates.",
    )

    # =========================================================================
    # 4. Build authoritative Phase-3 role mapping
    # =========================================================================

    banner(
        "RECONSTRUCT PRE-T60 INVESTOR–STARTUP PAIR INDEX"
    )

    nodes = pd.read_parquet(
        NODE_INDEX,
        columns=[
            "node_index",
            "node_type",
            "raw_entity_id",
        ],
    )

    require(
        len(nodes)
        == NUM_INVESTORS + NUM_STARTUPS,
        "Phase-3 role-node count drift.",
    )

    nodes["node_index"] = pd.to_numeric(
        nodes["node_index"],
        errors="raise",
    ).astype(np.int64)

    investor_nodes = (
        nodes.loc[
            (
                nodes["node_index"] >= 0
            )
            & (
                nodes["node_index"]
                < NUM_INVESTORS
            )
        ]
        .copy()
    )

    startup_nodes = (
        nodes.loc[
            (
                nodes["node_index"]
                >= NUM_INVESTORS
            )
            & (
                nodes["node_index"]
                < (
                    NUM_INVESTORS
                    + NUM_STARTUPS
                )
            )
        ]
        .copy()
    )

    require(
        len(investor_nodes)
        == NUM_INVESTORS,
        "Investor numeric slice count drift.",
    )

    require(
        len(startup_nodes)
        == NUM_STARTUPS,
        "Startup numeric slice count drift.",
    )

    investor_map = pd.Series(
        investor_nodes[
            "node_index"
        ].to_numpy(
            dtype=np.int64
        ),
        index=investor_nodes[
            "raw_entity_id"
        ].astype(str),
    ).to_dict()

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

    # =========================================================================
    # 5. Build strict T0-T59 history
    # =========================================================================

    temporal = pd.read_parquet(
        TEMPORAL_SPLIT,
        columns=[
            "investor_id",
            "startup_id",
            "segment_number",
        ],
    )

    history = (
        temporal.loc[
            temporal[
                "segment_number"
            ] < 60
        ]
        .copy()
    )

    require(
        len(history) == 1_173_422,
        (
            "Expected 1,173,422 T0-T59 history rows; "
            f"found {len(history):,}."
        ),
    )

    investor_global = (
        history[
            "investor_id"
        ]
        .astype(str)
        .map(investor_map)
    )

    startup_global = (
        history[
            "startup_id"
        ]
        .astype(str)
        .map(startup_map)
    )

    require(
        investor_global.notna().all(),
        "Historical investor role mapping incomplete.",
    )

    require(
        startup_global.notna().all(),
        "Historical startup role mapping incomplete.",
    )

    investor_global_np = (
        investor_global.to_numpy(
            dtype=np.int64
        )
    )

    startup_local_np = (
        startup_global.to_numpy(
            dtype=np.int64
        )
        - NUM_INVESTORS
    )

    require(
        (
            (startup_local_np >= 0)
            & (
                startup_local_np
                < NUM_STARTUPS
            )
        ).all(),
        "Historical startup-local index outside range.",
    )

    pair_keys = (
        investor_global_np
        * NUM_STARTUPS
        + startup_local_np
    )

    unique_pair_keys = np.unique(
        pair_keys.astype(
            np.int64,
            copy=False,
        )
    )

    require(
        len(unique_pair_keys)
        == 963_374,
        (
            "Expected 963,374 unique pre-T60 pairs; "
            f"found {len(unique_pair_keys):,}."
        ),
    )

    print(
        f"T0-T59 historical events:   "
        f"{len(history):,}"
    )

    print(
        f"Unique historical pairs:    "
        f"{len(unique_pair_keys):,}"
    )

    # =========================================================================
    # 6. Verify EVERY sampled discovery candidate is new-to-investor
    # =========================================================================

    banner(
        "STRICT DISCOVERY CANDIDATE INTEGRITY"
    )

    focal_investor = discovery[
        "investor_global"
    ].to_numpy(
        dtype=np.int64
    )

    candidate_pair_keys = (
        focal_investor[:, None]
        * NUM_STARTUPS
        + discovery_candidates
    )

    historical_collision = membership_in_sorted(
        unique_pair_keys,
        candidate_pair_keys.reshape(-1),
    ).reshape(
        discovery_candidates.shape
    )

    collision_per_case = (
        historical_collision
        .sum(axis=1)
        .astype(np.int64)
    )

    total_collisions = int(
        historical_collision.sum()
    )

    require(
        total_collisions == 0,
        (
            "Strict discovery certification FAILED: "
            f"{total_collisions:,} sampled candidate slots "
            "have a pre-T60 focal-investor pair."
        ),
    )

    require(
        not bool(
            historical_collision[:, 0].any()
        ),
        (
            "At least one focal positive is not actually "
            "new-to-investor."
        ),
    )

    print(
        "Discovery positives with prior pair:       0  PASS"
    )
    print(
        "Sampled candidate prior-pair collisions:  0  PASS"
    )
    print(
        "100 unique candidates per case:              PASS"
    )

    # =========================================================================
    # 7. Candidate-universe capacity per investor
    # =========================================================================
    #
    # For strict discovery:
    #
    # valid startups =
    #     all 311,589 startup role nodes
    #     minus startups funded by focal investor before T60.
    #
    # Focal positive is one member of that universe.
    #
    # Available negative pool =
    #     discovery universe size - 1 focal positive.
    # =========================================================================

    historical_pair_investor = (
        unique_pair_keys
        // NUM_STARTUPS
    )

    portfolio_size_by_investor = np.bincount(
        historical_pair_investor,
        minlength=NUM_INVESTORS,
    ).astype(np.int64)

    portfolio_size = (
        portfolio_size_by_investor[
            focal_investor
        ]
    )

    discovery_universe_size = (
        NUM_STARTUPS
        - portfolio_size
    )

    discovery_negative_pool_size = (
        discovery_universe_size
        - 1
    )

    require(
        (
            discovery_negative_pool_size
            >= 99
        ).all(),
        (
            "At least one strict-discovery case "
            "cannot support 99 negatives."
        ),
    )

    # =========================================================================
    # 8. Startup-history coverage inside sampled candidate sets
    # =========================================================================

    startup_history_count = np.bincount(
        startup_local_np,
        minlength=NUM_STARTUPS,
    ).astype(np.int64)

    sampled_history = startup_history_count[
        discovery_candidates
    ]

    sampled_cold = (
        sampled_history == 0
    )

    sampled_cold_count = (
        sampled_cold.sum(
            axis=1
        )
    )

    sampled_warm_count = (
        EXPECTED_CANDIDATES
        - sampled_cold_count
    )

    # =========================================================================
    # 9. Save per-case certification
    # =========================================================================

    case_audit = pd.DataFrame(
        {
            "interaction_id":
                discovery["interaction_id"].astype(str),

            "test_case_position":
                test_positions,

            "investor_id":
                discovery["investor_id"].astype(str),

            "startup_id":
                discovery["startup_id"].astype(str),

            "proposal_diagnostic_group":
                discovery[
                    "proposal_diagnostic_group"
                ].astype(str),

            "investor_global":
                focal_investor,

            "positive_startup_local":
                positive_local,

            "positive_startup_history_count":
                discovery[
                    "startup_history_count"
                ].to_numpy(
                    dtype=np.int64
                ),

            "prior_portfolio_size":
                portfolio_size,

            "discovery_candidate_universe_size":
                discovery_universe_size,

            "discovery_negative_pool_size":
                discovery_negative_pool_size,

            "sampled_candidate_count":
                EXPECTED_CANDIDATES,

            "sampled_prior_pair_collision_count":
                collision_per_case,

            "sampled_cold_candidate_count":
                sampled_cold_count,

            "sampled_warm_candidate_count":
                sampled_warm_count,

            "sampled_cold_candidate_share":
                (
                    sampled_cold_count
                    / EXPECTED_CANDIDATES
                ),

            "positive_rank":
                discovery[
                    "positive_rank"
                ].to_numpy(
                    dtype=np.int64
                ),

            "HR@10":
                discovery[
                    "HR@10"
                ].to_numpy(
                    dtype=np.float64
                ),

            "NDCG@10":
                discovery[
                    "NDCG@10"
                ].to_numpy(
                    dtype=np.float64
                ),
        }
    )

    case_audit.to_csv(
        OUT_CASE_AUDIT,
        index=False,
    )

    # =========================================================================
    # 10. Candidate statistics
    # =========================================================================

    warm_positive = (
        case_audit[
            "positive_startup_history_count"
        ] > 0
    )

    cold_positive = (
        case_audit[
            "positive_startup_history_count"
        ] == 0
    )

    candidate_stats = pd.DataFrame(
        [
            pool_summary(
                case_audit,
                "all_new_to_investor",
            ),
            pool_summary(
                case_audit.loc[
                    warm_positive
                ],
                "new_to_investor_warm_startup",
            ),
            pool_summary(
                case_audit.loc[
                    cold_positive
                ],
                "new_to_investor_cold_startup",
            ),
        ]
    )

    candidate_stats.to_csv(
        OUT_CANDIDATE_STATS,
        index=False,
    )

    # =========================================================================
    # 11. Strict discovery metrics
    # =========================================================================

    metric_rows = [
        metric_row(
            events,
            "original_all_test_reference",
        ),
        metric_row(
            discovery,
            "strict_new_to_investor_all",
        ),
        metric_row(
            discovery.loc[
                discovery[
                    "startup_history_count"
                ] > 0
            ],
            "strict_new_to_investor_warm_startup",
        ),
        metric_row(
            discovery.loc[
                discovery[
                    "startup_history_count"
                ] == 0
            ],
            "strict_new_to_investor_cold_startup",
        ),
        metric_row(
            discovery.loc[
                discovery[
                    "proposal_diagnostic_group"
                ].eq(
                    "novel_warm_warm"
                )
            ],
            "strict_novel_warm_investor_warm_startup",
        ),
        metric_row(
            discovery.loc[
                discovery[
                    "proposal_diagnostic_group"
                ].eq(
                    "novel_cold_startup"
                )
            ],
            "strict_warm_investor_cold_startup",
        ),
        metric_row(
            discovery.loc[
                discovery[
                    "proposal_diagnostic_group"
                ].eq(
                    "novel_cold_investor"
                )
            ],
            "strict_cold_investor_warm_startup",
        ),
        metric_row(
            discovery.loc[
                discovery[
                    "proposal_diagnostic_group"
                ].eq(
                    "novel_both_cold"
                )
            ],
            "strict_both_cold",
        ),
    ]

    metrics_df = pd.DataFrame(
        metric_rows
    )

    metrics_df.to_csv(
        OUT_DISCOVERY_METRICS,
        index=False,
    )

    # =========================================================================
    # 12. Integrity checks
    # =========================================================================

    p1_expected_hr = 0.304633
    p1_expected_ndcg = 0.158578

    strict_hr = float(
        discovery[
            "HR@10"
        ].mean()
    )

    strict_ndcg = float(
        discovery[
            "NDCG@10"
        ].mean()
    )

    # We use a looser check only against the rounded console values.
    require(
        abs(
            strict_hr
            - p1_expected_hr
        ) < 1e-6,
        (
            "Strict discovery HR@10 does not "
            "match P1 new-to-investor result."
        ),
    )

    require(
        abs(
            strict_ndcg
            - p1_expected_ndcg
        ) < 1e-6,
        (
            "Strict discovery NDCG@10 does not "
            "match P1 new-to-investor result."
        ),
    )

    invariant_rows = [
        {
            "check":
                "phase6_test_cases",
            "actual":
                len(events),
            "expected":
                EXPECTED_TEST_CASES,
            "status":
                "PASS",
        },
        {
            "check":
                "new_to_investor_positive_cases",
            "actual":
                len(discovery),
            "expected":
                EXPECTED_DISCOVERY_CASES,
            "status":
                "PASS",
        },
        {
            "check":
                "candidates_per_case",
            "actual":
                discovery_candidates.shape[1],
            "expected":
                EXPECTED_CANDIDATES,
            "status":
                "PASS",
        },
        {
            "check":
                "positive_at_candidate_position_zero",
            "actual":
                True,
            "expected":
                True,
            "status":
                "PASS",
        },
        {
            "check":
                "sampled_prior_pair_collisions",
            "actual":
                total_collisions,
            "expected":
                0,
            "status":
                "PASS",
        },
        {
            "check":
                "all_candidate_sets_unique",
            "actual":
                True,
            "expected":
                True,
            "status":
                "PASS",
        },
        {
            "check":
                "all_discovery_pools_support_99_negatives",
            "actual":
                bool(
                    (
                        discovery_negative_pool_size
                        >= 99
                    ).all()
                ),
            "expected":
                True,
            "status":
                "PASS",
        },
        {
            "check":
                "no_new_neural_scoring",
            "actual":
                True,
            "expected":
                True,
            "status":
                "PASS",
        },
    ]

    invariant_df = pd.DataFrame(
        invariant_rows
    )

    invariant_df.to_csv(
        OUT_INVARIANTS,
        index=False,
    )

    # =========================================================================
    # 13. Summary
    # =========================================================================

    summary = {
        "schema_version":
            "PROPOSAL_EVIDENCE_P2_V1",

        "status":
            "P2_COMPLETE",

        "scientific_role":
            (
                "Post-hoc certification that frozen Phase-6 "
                "sampled candidate sets are strict "
                "new-to-investor discovery candidate sets "
                "for new-to-investor focal positives."
            ),

        "candidate_policy":
            {
                "startup_universe":
                    "all frozen startup-role nodes",

                "startup_universe_size":
                    NUM_STARTUPS,

                "history_cutoff":
                    "T0-T59 only",

                "prior_portfolio_excluded":
                    True,

                "focal_positive_is_new_to_investor":
                    True,

                "other_T60_labels_used_for_exclusion":
                    False,

                "candidate_sets_regenerated":
                    False,

                "neural_rescoring_performed":
                    False,
            },

        "cases":
            {
                "original_test":
                    int(len(events)),

                "strict_discovery":
                    int(len(discovery)),

                "strict_discovery_share":
                    float(
                        len(discovery)
                        / len(events)
                    ),
            },

        "strict_discovery_metrics":
            {
                "HR@10":
                    strict_hr,

                "NDCG@10":
                    strict_ndcg,

                "mean_positive_rank":
                    float(
                        discovery[
                            "positive_rank"
                        ].mean()
                    ),

                "median_positive_rank":
                    float(
                        discovery[
                            "positive_rank"
                        ].median()
                    ),
            },

        "integrity":
            {
                "historical_pair_collisions":
                    total_collisions,

                "all_candidate_sets_unique":
                    True,

                "all_pools_support_99_negatives":
                    True,
            },

        "outputs":
            {
                "case_audit":
                    str(
                        OUT_CASE_AUDIT.relative_to(
                            ROOT
                        )
                    ),

                "candidate_statistics":
                    str(
                        OUT_CANDIDATE_STATS.relative_to(
                            ROOT
                        )
                    ),

                "discovery_metrics":
                    str(
                        OUT_DISCOVERY_METRICS.relative_to(
                            ROOT
                        )
                    ),

                "invariants":
                    str(
                        OUT_INVARIANTS.relative_to(
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
    # 14. Console output
    # =========================================================================

    banner(
        "STRICT DISCOVERY METRICS"
    )

    print(
        metrics_df.to_string(
            index=False,
            formatters={
                "HR@10":
                    lambda x:
                        f"{x:.6f}",

                "NDCG@10":
                    lambda x:
                        f"{x:.6f}",

                "mean_positive_rank":
                    lambda x:
                        f"{x:.2f}",

                "median_positive_rank":
                    lambda x:
                        f"{x:.1f}",
            },
        )
    )

    banner(
        "DISCOVERY CANDIDATE-UNIVERSE STATISTICS"
    )

    print(
        candidate_stats.to_string(
            index=False
        )
    )

    banner(
        "P2 OUTPUTS"
    )

    for path in [
        OUT_CASE_AUDIT,
        OUT_CANDIDATE_STATS,
        OUT_DISCOVERY_METRICS,
        OUT_INVARIANTS,
        OUT_SUMMARY,
    ]:
        print(
            "WROTE ",
            path.relative_to(ROOT),
        )

    banner(
        "P2 COMPLETE — STRICT DISCOVERY CERTIFIED / "
        "NO TRAINING / NO INFERENCE / NO RESCORING"
    )


if __name__ == "__main__":
    main()
