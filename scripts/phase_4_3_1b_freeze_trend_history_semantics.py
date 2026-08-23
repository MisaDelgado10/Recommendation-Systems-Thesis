from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 4.3.1b — FREEZE TREND-HISTORY SEMANTICS
# =============================================================================

TEMPORAL_SPLIT_PATH = Path(
    "data/experimental/phase_2/"
    "model_ready/"
    "interactions_itrs_temporal_split.parquet"
)

NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/"
    "model_ready/"
    "node_index.parquet"
)

AUDIT_METADATA_PATH = Path(
    "data/experimental/phase_4/"
    "audits/"
    "trend_temporal_input_audit_metadata.json"
)


OUT_DIR = Path(
    "data/experimental/phase_4/"
    "trend_contract"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Frozen expectations from Phase 4.3.1a
# =============================================================================

EXPECTED_HISTORICAL_EVENTS = 1_173_422

EXPECTED_PAIR_PERIOD_MEMBERSHIPS = 1_145_364

EXPECTED_REPEATED_PAIR_PERIODS = 20_473

EXPECTED_EXTRA_EVENTS = 28_058

EXPECTED_ACTIVE_INVESTORS = 161_065

EXPECTED_NEVER_HISTORICAL_INVESTORS = 4_910

EXPECTED_ACTIVE_INVESTOR_PERIODS = 554_171

EXPECTED_T60_INVESTORS = 11_884

EXPECTED_T60_INVESTORS_WITHOUT_HISTORY = 3_060

EXPECTED_T60_EVENTS_WITHOUT_HISTORY = 3_541


# =============================================================================
# Frozen Phase-3 node populations
# =============================================================================

EXPECTED_INVESTOR_NODES = 165_975
EXPECTED_STARTUP_NODES = 311_589

EXPECTED_TOTAL_NODES = 477_564


# =============================================================================
# Frozen temporal semantics
# =============================================================================

HISTORY_FIRST_PERIOD = 0
HISTORY_LAST_PERIOD = 59

T60_PERIOD = 60

STARTUP_LATENT_DIM = 40
STARTUP_DESCRIPTION_DIM = 40

PERIOD_VECTOR_DIM = (
    STARTUP_LATENT_DIM
    + STARTUP_DESCRIPTION_DIM
)

TREND_OUTPUT_DIM = 40


assert PERIOD_VECTOR_DIM == 80


# =============================================================================
# Helpers
# =============================================================================

def banner(title):

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def sha256_file(path):

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        for chunk in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


# =============================================================================
# Start
# =============================================================================

banner(
    "PHASE 4.3.1b — "
    "FREEZE TREND-HISTORY SEMANTICS"
)


# =============================================================================
# 1. Verify upstream audit
# =============================================================================

banner(
    "UPSTREAM AUDIT CONTRACT"
)


with open(
    AUDIT_METADATA_PATH,
    "r",
    encoding="utf-8",
) as f:

    audit_metadata = json.load(f)


if (
    audit_metadata.get(
        "status"
    )
    != "COMPLETE_AUDIT_ONLY"
):

    raise AssertionError(
        "Phase 4.3.1a audit is not "
        "in the expected completed state."
    )


audit_pair_period = (
    audit_metadata[
        "pair_period_multiplicity"
    ]
)


audit_investor_history = (
    audit_metadata[
        "investor_history"
    ]
)


audit_t60 = (
    audit_metadata[
        "t60_history_coverage"
    ]
)


assert (
    audit_pair_period[
        "unique_investor_startup_period_tuples"
    ]
    == EXPECTED_PAIR_PERIOD_MEMBERSHIPS
)

assert (
    audit_pair_period[
        "repeated_pair_period_tuples"
    ]
    == EXPECTED_REPEATED_PAIR_PERIODS
)

assert (
    audit_pair_period[
        "extra_event_rows_beyond_set_membership"
    ]
    == EXPECTED_EXTRA_EVENTS
)

assert (
    audit_investor_history[
        "historically_active_investors"
    ]
    == EXPECTED_ACTIVE_INVESTORS
)

assert (
    audit_investor_history[
        "never_historical_investors"
    ]
    == EXPECTED_NEVER_HISTORICAL_INVESTORS
)

assert (
    audit_t60[
        "t60_investors_without_history"
    ]
    == EXPECTED_T60_INVESTORS_WITHOUT_HISTORY
)


print(
    "Phase 4.3.1a audit loaded: PASS"
)


# =============================================================================
# 2. Load frozen Phase-2 history
# =============================================================================

banner(
    "LOADING FROZEN T0-T59 HISTORY"
)


temporal = pd.read_parquet(
    TEMPORAL_SPLIT_PATH,
    columns=[
        "interaction_id",
        "funding_round_id",
        "investor_id",
        "startup_id",
        "announced_on",
        "segment_number",
        "segment_label",
        "temporal_role",
        "experiment_split",
    ],
)


history = (
    temporal.loc[
        temporal[
            "segment_number"
        ].between(
            HISTORY_FIRST_PERIOD,
            HISTORY_LAST_PERIOD,
        )
    ]
    .copy()
)


if (
    len(history)
    != EXPECTED_HISTORICAL_EVENTS
):

    raise AssertionError(
        "Historical population differs "
        "from frozen Phase-2 history."
    )


if not (
    history[
        "experiment_split"
    ]
    .astype(str)
    .eq("train")
    .all()
):

    raise AssertionError(
        "Non-training row found in "
        "trend history."
    )


if (
    history[
        "segment_number"
    ]
    .eq(T60_PERIOD)
    .any()
):

    raise AssertionError(
        "T60 leakage into trend history."
    )


print(
    f"Historical event rows: "
    f"{len(history):,}"
)

print(
    "T60 rows consumed:      0"
)


# =============================================================================
# 3. Materialize paper-style SET membership
#
# One row:
#
#   Investor × Startup × Period
#
# Repeated funding events inside the same temporal segment do not produce
# repeated attention items.
# =============================================================================

banner(
    "MATERIALIZING SET-VALUED INVESTOR-PERIOD HISTORY"
)


membership = (
    history
    .groupby(
        [
            "investor_id",
            "startup_id",
            "segment_number",
        ],
        sort=False,
    )
    .agg(
        source_event_count=(
            "interaction_id",
            "size",
        ),

        source_funding_round_count=(
            "funding_round_id",
            "nunique",
        ),

        first_source_announced_on=(
            "announced_on",
            "min",
        ),

        last_source_announced_on=(
            "announced_on",
            "max",
        ),
    )
    .reset_index()
)


if (
    len(membership)
    != EXPECTED_PAIR_PERIOD_MEMBERSHIPS
):

    raise AssertionError(
        "Pair-period membership population "
        "differs from Phase 4.3.1a."
    )


repeated_memberships = int(
    (
        membership[
            "source_event_count"
        ]
        > 1
    ).sum()
)


extra_events = int(
    (
        membership[
            "source_event_count"
        ]
        - 1
    )
    .clip(
        lower=0
    )
    .sum()
)


if (
    repeated_memberships
    != EXPECTED_REPEATED_PAIR_PERIODS
):

    raise AssertionError(
        "Repeated pair-period count changed."
    )


if (
    extra_events
    != EXPECTED_EXTRA_EVENTS
):

    raise AssertionError(
        "Extra-event count changed."
    )


if (
    int(
        membership[
            "source_event_count"
        ].sum()
    )
    != EXPECTED_HISTORICAL_EVENTS
):

    raise AssertionError(
        "Membership source-event counts "
        "do not reconstruct history."
    )


print(
    f"Unique set memberships:  "
    f"{len(membership):,}"
)

print(
    f"Repeated source tuples:  "
    f"{repeated_memberships:,}"
)

print(
    f"Collapsed extra events:  "
    f"{extra_events:,}"
)


# =============================================================================
# 4. Load frozen Phase-3 node index
# =============================================================================

banner(
    "LOADING PHASE-3 ROLE NODE INDEX"
)


nodes = pd.read_parquet(
    NODE_INDEX_PATH,
    columns=[
        "node_index",
        "node_id",
        "node_type",
        "raw_entity_id",
    ],
)


if len(nodes) != EXPECTED_TOTAL_NODES:

    raise AssertionError(
        "Phase-3 node population changed."
    )


investor_nodes = (
    nodes.loc[
        nodes[
            "node_type"
        ]
        .eq("investor"),
        [
            "node_index",
            "node_id",
            "raw_entity_id",
        ],
    ]
    .copy()
)


startup_nodes = (
    nodes.loc[
        nodes[
            "node_type"
        ]
        .eq("startup"),
        [
            "node_index",
            "node_id",
            "raw_entity_id",
        ],
    ]
    .copy()
)


if (
    len(investor_nodes)
    != EXPECTED_INVESTOR_NODES
):

    raise AssertionError(
        "Investor node population changed."
    )


if (
    len(startup_nodes)
    != EXPECTED_STARTUP_NODES
):

    raise AssertionError(
        "Startup node population changed."
    )


if (
    investor_nodes[
        "raw_entity_id"
    ]
    .duplicated()
    .any()
):

    raise AssertionError(
        "Duplicate raw Investor ID "
        "inside Investor node role."
    )


if (
    startup_nodes[
        "raw_entity_id"
    ]
    .duplicated()
    .any()
):

    raise AssertionError(
        "Duplicate raw Startup ID "
        "inside Startup node role."
    )


expected_investor_indices = np.arange(
    0,
    EXPECTED_INVESTOR_NODES,
    dtype=np.int64,
)


actual_investor_indices = (
    investor_nodes[
        "node_index"
    ]
    .sort_values()
    .to_numpy(
        dtype=np.int64
    )
)


if not np.array_equal(
    expected_investor_indices,
    actual_investor_indices,
):

    raise AssertionError(
        "Frozen Investor node-index range changed."
    )


expected_startup_indices = np.arange(
    EXPECTED_INVESTOR_NODES,
    EXPECTED_TOTAL_NODES,
    dtype=np.int64,
)


actual_startup_indices = (
    startup_nodes[
        "node_index"
    ]
    .sort_values()
    .to_numpy(
        dtype=np.int64
    )
)


if not np.array_equal(
    expected_startup_indices,
    actual_startup_indices,
):

    raise AssertionError(
        "Frozen Startup node-index range changed."
    )


print(
    f"Investor nodes: "
    f"{len(investor_nodes):,}"
)

print(
    f"Startup nodes:  "
    f"{len(startup_nodes):,}"
)

print(
    "Role node-index ranges: PASS"
)


# =============================================================================
# 5. Map historical memberships to deterministic graph-node indices
# =============================================================================

banner(
    "MAPPING HISTORY TO PHASE-3 NODE INDICES"
)


investor_lookup = (
    investor_nodes
    .rename(
        columns={
            "raw_entity_id":
                "investor_id",

            "node_index":
                "investor_node_index",

            "node_id":
                "investor_node_id",
        }
    )
)


startup_lookup = (
    startup_nodes
    .rename(
        columns={
            "raw_entity_id":
                "startup_id",

            "node_index":
                "startup_node_index",

            "node_id":
                "startup_node_id",
        }
    )
)


membership[
    "investor_id"
] = (
    membership[
        "investor_id"
    ].astype(str)
)


membership[
    "startup_id"
] = (
    membership[
        "startup_id"
    ].astype(str)
)


investor_lookup[
    "investor_id"
] = (
    investor_lookup[
        "investor_id"
    ].astype(str)
)


startup_lookup[
    "startup_id"
] = (
    startup_lookup[
        "startup_id"
    ].astype(str)
)


membership = (
    membership
    .merge(
        investor_lookup,
        on="investor_id",
        how="left",
        validate="many_to_one",
    )
)


membership = (
    membership
    .merge(
        startup_lookup,
        on="startup_id",
        how="left",
        validate="many_to_one",
    )
)


unmapped_investors = int(
    membership[
        "investor_node_index"
    ]
    .isna()
    .sum()
)


unmapped_startups = int(
    membership[
        "startup_node_index"
    ]
    .isna()
    .sum()
)


print(
    f"Unmapped Investor memberships: "
    f"{unmapped_investors:,}"
)

print(
    f"Unmapped Startup memberships:  "
    f"{unmapped_startups:,}"
)


if (
    unmapped_investors
    or unmapped_startups
):

    raise AssertionError(
        "Historical trend membership "
        "does not map completely to "
        "Phase-3 role nodes."
    )


membership[
    "investor_node_index"
] = (
    membership[
        "investor_node_index"
    ]
    .astype(
        np.int64
    )
)


membership[
    "startup_node_index"
] = (
    membership[
        "startup_node_index"
    ]
    .astype(
        np.int64
    )
)


# =============================================================================
# 6. Deterministic semantic ordering
#
# No within-period chronology is introduced.
#
# Startup node index is used ONLY to provide deterministic attention-item
# ordering.
# =============================================================================

banner(
    "DETERMINISTIC SET ORDERING"
)


membership = (
    membership
    .sort_values(
        [
            "investor_node_index",
            "segment_number",
            "startup_node_index",
        ],
        kind="mergesort",
    )
    .reset_index(
        drop=True
    )
)


membership[
    "trend_membership_row"
] = np.arange(
    len(membership),
    dtype=np.int64,
)


duplicate_semantic_memberships = int(
    membership[
        [
            "investor_node_index",
            "segment_number",
            "startup_node_index",
        ]
    ]
    .duplicated()
    .sum()
)


print(
    f"Duplicate semantic memberships "
    f"after set materialization: "
    f"{duplicate_semantic_memberships:,}"
)


if duplicate_semantic_memberships:

    raise AssertionError(
        "Set-valued history still contains "
        "duplicate membership."
    )


print(
    "Ordering key:"
)

print(
    "  investor_node_index ASC"
)

print(
    "  segment_number ASC"
)

print(
    "  startup_node_index ASC"
)

print()
print(
    "Within-period chronology introduced: NO"
)


# =============================================================================
# 7. Materialize sparse ACTIVE Investor-period manifest
#
# Empty periods are intentionally NOT stored as rows.
#
# Their absence is frozen to mean:
#
#     x_i,h = zero_80
#
# during the later trend forward pass.
# =============================================================================

banner(
    "MATERIALIZING ACTIVE INVESTOR-PERIOD SLOTS"
)


active_slots = (
    membership
    .groupby(
        [
            "investor_node_index",
            "investor_node_id",
            "investor_id",
            "segment_number",
        ],
        sort=False,
    )
    .agg(
        unique_startups=(
            "startup_node_index",
            "size",
        ),

        source_events=(
            "source_event_count",
            "sum",
        ),
    )
    .reset_index()
)


if (
    len(active_slots)
    != EXPECTED_ACTIVE_INVESTOR_PERIODS
):

    raise AssertionError(
        "Active Investor-period count "
        "changed from Phase 4.3.1a."
    )


if (
    active_slots[
        "unique_startups"
    ]
    .lt(1)
    .any()
):

    raise AssertionError(
        "Stored active period has "
        "zero startups."
    )


print(
    f"Stored active Investor-periods: "
    f"{len(active_slots):,}"
)

print(
    "Stored empty Investor-periods:  0"
)

print(
    "Meaning of absent period row:  "
    "zero_80 period representation"
)


# =============================================================================
# 8. Investor-level historical summary including zero-history Investors
# =============================================================================

banner(
    "INVESTOR HISTORY SUMMARY"
)


investor_summary = (
    investor_nodes
    .rename(
        columns={
            "node_index":
                "investor_node_index",

            "node_id":
                "investor_node_id",

            "raw_entity_id":
                "investor_id",
        }
    )
    .copy()
)


active_summary = (
    active_slots
    .groupby(
        [
            "investor_node_index",
        ],
        sort=False,
    )
    .agg(
        active_period_count=(
            "segment_number",
            "nunique",
        ),

        first_active_period=(
            "segment_number",
            "min",
        ),

        last_active_period=(
            "segment_number",
            "max",
        ),

        historical_unique_memberships=(
            "unique_startups",
            "sum",
        ),

        historical_source_events=(
            "source_events",
            "sum",
        ),
    )
    .reset_index()
)


investor_summary = (
    investor_summary
    .merge(
        active_summary,
        on="investor_node_index",
        how="left",
        validate="one_to_one",
    )
)


investor_summary[
    "active_period_count"
] = (
    investor_summary[
        "active_period_count"
    ]
    .fillna(0)
    .astype(
        np.int16
    )
)


investor_summary[
    "historical_unique_memberships"
] = (
    investor_summary[
        "historical_unique_memberships"
    ]
    .fillna(0)
    .astype(
        np.int64
    )
)


investor_summary[
    "historical_source_events"
] = (
    investor_summary[
        "historical_source_events"
    ]
    .fillna(0)
    .astype(
        np.int64
    )
)


investor_summary[
    "has_historical_investment"
] = (
    investor_summary[
        "active_period_count"
    ]
    > 0
)


investor_summary[
    "empty_period_count_t0_t59"
] = (
    60
    - investor_summary[
        "active_period_count"
    ]
)


active_investor_count = int(
    investor_summary[
        "has_historical_investment"
    ].sum()
)


zero_history_investor_count = int(
    (
        ~investor_summary[
            "has_historical_investment"
        ]
    ).sum()
)


if (
    active_investor_count
    != EXPECTED_ACTIVE_INVESTORS
):

    raise AssertionError(
        "Historically active Investor "
        "population changed."
    )


if (
    zero_history_investor_count
    != EXPECTED_NEVER_HISTORICAL_INVESTORS
):

    raise AssertionError(
        "Zero-history Investor population "
        "changed."
    )


print(
    f"Historically active Investors: "
    f"{active_investor_count:,}"
)

print(
    f"Zero-history Investors:        "
    f"{zero_history_investor_count:,}"
)


# =============================================================================
# 9. Recheck T60 cold-Investor behavior
# =============================================================================

banner(
    "T60 COLD-INVESTOR CONTRACT"
)


t60 = (
    temporal.loc[
        temporal[
            "segment_number"
        ]
        .eq(T60_PERIOD)
    ]
    .copy()
)


historical_investor_ids = set(
    investor_summary.loc[
        investor_summary[
            "has_historical_investment"
        ],
        "investor_id",
    ]
    .astype(str)
)


t60[
    "investor_has_history"
] = (
    t60[
        "investor_id"
    ]
    .astype(str)
    .isin(
        historical_investor_ids
    )
)


t60_unique_investors = int(
    t60[
        "investor_id"
    ]
    .nunique()
)


t60_no_history_investors = int(
    t60.loc[
        ~t60[
            "investor_has_history"
        ],
        "investor_id",
    ]
    .nunique()
)


t60_no_history_events = int(
    (
        ~t60[
            "investor_has_history"
        ]
    ).sum()
)


assert (
    t60_unique_investors
    == EXPECTED_T60_INVESTORS
)


assert (
    t60_no_history_investors
    == EXPECTED_T60_INVESTORS_WITHOUT_HISTORY
)


assert (
    t60_no_history_events
    == EXPECTED_T60_EVENTS_WITHOUT_HISTORY
)


print(
    f"T60 unique Investors:        "
    f"{t60_unique_investors:,}"
)

print(
    f"T60 no-history Investors:    "
    f"{t60_no_history_investors:,}"
)

print(
    f"T60 no-history events:       "
    f"{t60_no_history_events:,}"
)

print()
print(
    "Frozen cold-Investor input behavior:"
)

print(
    "  no special history embedding"
)

print(
    "  absent historical periods -> zero_80"
)

print(
    "  T60 no-history Investor -> "
    "60 × zero_80 input sequence"
)


# =============================================================================
# 10. Save model-ready temporal membership artifacts
# =============================================================================

banner(
    "SAVING FROZEN TREND-HISTORY ARTIFACTS"
)


membership_path = (
    OUT_DIR
    / "trend_history_membership.parquet"
)


active_slots_path = (
    OUT_DIR
    / "trend_active_investor_periods.parquet"
)


investor_summary_path = (
    OUT_DIR
    / "trend_investor_history_summary.parquet"
)


membership[
    [
        "trend_membership_row",

        "investor_node_index",
        "investor_node_id",
        "investor_id",

        "segment_number",

        "startup_node_index",
        "startup_node_id",
        "startup_id",

        "source_event_count",
        "source_funding_round_count",

        "first_source_announced_on",
        "last_source_announced_on",
    ]
].to_parquet(
    membership_path,
    index=False,
)


active_slots.to_parquet(
    active_slots_path,
    index=False,
)


investor_summary.to_parquet(
    investor_summary_path,
    index=False,
)


# =============================================================================
# 11. Persistence reload integrity
# =============================================================================

banner(
    "PERSISTENCE RELOAD INTEGRITY"
)


membership_reload = pd.read_parquet(
    membership_path
)


active_slots_reload = pd.read_parquet(
    active_slots_path
)


investor_summary_reload = pd.read_parquet(
    investor_summary_path
)


assert (
    len(membership_reload)
    == EXPECTED_PAIR_PERIOD_MEMBERSHIPS
)


assert (
    len(active_slots_reload)
    == EXPECTED_ACTIVE_INVESTOR_PERIODS
)


assert (
    len(investor_summary_reload)
    == EXPECTED_INVESTOR_NODES
)


if not np.array_equal(
    membership_reload[
        "trend_membership_row"
    ]
    .to_numpy(
        dtype=np.int64
    ),

    np.arange(
        EXPECTED_PAIR_PERIOD_MEMBERSHIPS,
        dtype=np.int64,
    ),
):

    raise AssertionError(
        "Membership deterministic row "
        "index changed after persistence."
    )


print(
    "Membership reload:       PASS"
)

print(
    "Active-slot reload:      PASS"
)

print(
    "Investor-summary reload: PASS"
)


# =============================================================================
# 12. Freeze semantic contract
# =============================================================================

banner(
    "FREEZING TREND-HISTORY SEMANTIC CONTRACT"
)


contract = {
    "phase":
        "4.3.1b",

    "status":
        "FROZEN",

    "component":
        "ITRS trend-history temporal semantics",

    "paper_specified": {

        "history_rule":
            (
                "For a target event in segment T_h, "
                "only segments T0 through T_{h-1} "
                "are used for trend extraction."
            ),

        "within_segment_sequence":
            False,

        "period_item_collection":
            "set of invested startups",

        "period_representation":
            (
                "attention-weighted aggregation of "
                "startup latent + description features"
            ),

        "period_sequence":
            (
                "(x_i,0, x_i,1, ..., x_i,h-1) "
                "is provided to the GRU"
            ),
    },

    "paper_grounded_reproduction_interpretation": {

        "duplicate_pair_period_policy":
            (
                "Collapse repeated Investor-Startup "
                "events inside the same temporal "
                "segment to one startup membership."
            ),

        "same_segment_target_exclusion":
            (
                "No event in target segment T_h "
                "is allowed to influence the trend "
                "representation used for another "
                "target event in T_h."
            ),

        "t60_history":
            "T0 through T59 only",
    },

    "paper_unspecified_reproduction_choices": {

        "empty_period_input": {
            "policy":
                "explicit zero vector",

            "dimension":
                PERIOD_VECTOR_DIM,

            "value":
                "zero_80",

            "slot_retained":
                True,

            "reason":
                (
                    "Preserve elapsed segmented time; "
                    "omitting empty periods would "
                    "compress temporal gaps that the "
                    "trend-aware model is intended "
                    "to represent."
                ),
        },

        "no_history_investor": {
            "special_embedding":
                False,

            "input_policy":
                (
                    "all historical period inputs "
                    "are zero_80"
                ),

            "downstream_special_case":
                False,

            "note":
                (
                    "The downstream GRU processes "
                    "the same input format as for "
                    "historically active Investors."
                ),
        },

        "t0_target_event": {
            "prior_periods":
                0,

            "trend_feature_policy":
                "zero_40",

            "reason":
                (
                    "No pre-T0 temporal segment "
                    "exists in the frozen experiment."
                ),
        },

        "within_period_deterministic_order":
            "startup_node_index ascending",

        "within_period_order_semantics":
            "none",
    },

    "dimensions": {
        "startup_latent":
            STARTUP_LATENT_DIM,

        "startup_description":
            STARTUP_DESCRIPTION_DIM,

        "startup_attention_item":
            PERIOD_VECTOR_DIM,

        "period_attention_output":
            PERIOD_VECTOR_DIM,

        "trend_output":
            TREND_OUTPUT_DIM,
    },

    "target_segment_history_rule": {

        "T0":
            "no prior sequence; trend zero_40",

        "T1":
            "T0",

        "T2":
            "T0-T1",

        "general_Th":
            "T0 through T_{h-1}",

        "T60":
            "T0-T59",
    },

    "storage_contract": {

        "history_membership":
            (
                "Sparse table containing only active "
                "Investor-Startup-Period memberships."
            ),

        "active_period_table":
            (
                "Sparse table containing only active "
                "Investor-period slots."
            ),

        "empty_period_materialization":
            (
                "Empty period rows are not persisted; "
                "absence is semantically equivalent "
                "to zero_80 at model forward time."
            ),

        "membership_sort_order": [
            "investor_node_index",
            "segment_number",
            "startup_node_index",
        ],
    },

    "frozen_counts": {

        "historical_source_events":
            EXPECTED_HISTORICAL_EVENTS,

        "unique_pair_period_memberships":
            EXPECTED_PAIR_PERIOD_MEMBERSHIPS,

        "collapsed_extra_events":
            EXPECTED_EXTRA_EVENTS,

        "active_investor_periods":
            EXPECTED_ACTIVE_INVESTOR_PERIODS,

        "historically_active_investors":
            EXPECTED_ACTIVE_INVESTORS,

        "zero_history_investors":
            EXPECTED_NEVER_HISTORICAL_INVESTORS,

        "t60_no_history_investors":
            EXPECTED_T60_INVESTORS_WITHOUT_HISTORY,
    },

    "data_leakage": {

        "t60_in_history":
            False,

        "validation_in_history":
            False,

        "test_in_history":
            False,

        "same_target_segment_in_history":
            False,
    },

    "not_yet_frozen": [

        "bilinear attention parameterization",

        "attention matrix initialization",

        "latent embedding initialization",

        "GRU exact PyTorch architecture",

        "GRU initial hidden state",

        "GRU bias configuration",

        "trend output projection W_o",

        "trend output sigmoid implementation",
    ],

    "upstream_reopened": {

        "phase_2":
            False,

        "phase_3":
            False,

        "phase_4_2":
            False,
    },
}


contract_path = (
    OUT_DIR
    / "trend_history_semantics_contract.json"
)


with open(
    contract_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        contract,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 13. Audit summary
# =============================================================================

audit_df = pd.DataFrame(
    [
        {
            "metric":
                "historical_source_events",

            "value":
                EXPECTED_HISTORICAL_EVENTS,
        },

        {
            "metric":
                "unique_pair_period_memberships",

            "value":
                EXPECTED_PAIR_PERIOD_MEMBERSHIPS,
        },

        {
            "metric":
                "collapsed_extra_events",

            "value":
                EXPECTED_EXTRA_EVENTS,
        },

        {
            "metric":
                "active_investor_periods",

            "value":
                EXPECTED_ACTIVE_INVESTOR_PERIODS,
        },

        {
            "metric":
                "historically_active_investors",

            "value":
                EXPECTED_ACTIVE_INVESTORS,
        },

        {
            "metric":
                "zero_history_investors",

            "value":
                EXPECTED_NEVER_HISTORICAL_INVESTORS,
        },

        {
            "metric":
                "t60_no_history_investors",

            "value":
                EXPECTED_T60_INVESTORS_WITHOUT_HISTORY,
        },

        {
            "metric":
                "period_vector_dim",

            "value":
                PERIOD_VECTOR_DIM,
        },

        {
            "metric":
                "t0_target_trend_dim",

            "value":
                TREND_OUTPUT_DIM,
        },
    ]
)


audit_path = (
    OUT_DIR
    / "trend_history_semantics_audit.csv"
)


audit_df.to_csv(
    audit_path,
    index=False,
)


# =============================================================================
# 14. Artifact hashes
# =============================================================================

hash_records = []


for artifact_name, path in [
    (
        "history_membership",
        membership_path,
    ),

    (
        "active_investor_periods",
        active_slots_path,
    ),

    (
        "investor_history_summary",
        investor_summary_path,
    ),

    (
        "semantic_contract",
        contract_path,
    ),

    (
        "semantic_audit",
        audit_path,
    ),
]:

    digest = sha256_file(
        path
    )

    hash_records.append(
        {
            "artifact":
                artifact_name,

            "path":
                str(path),

            "sha256":
                digest,

            "bytes":
                path.stat().st_size,
        }
    )


hash_df = pd.DataFrame(
    hash_records
)


hash_path = (
    OUT_DIR
    / "trend_history_artifact_hashes.csv"
)


hash_df.to_csv(
    hash_path,
    index=False,
)


# =============================================================================
# Final summary
# =============================================================================

banner(
    "PHASE 4.3.1b FINAL SUMMARY"
)


print(
    f"Historical source events:        "
    f"{EXPECTED_HISTORICAL_EVENTS:,}"
)

print(
    f"Unique set memberships:          "
    f"{EXPECTED_PAIR_PERIOD_MEMBERSHIPS:,}"
)

print(
    f"Collapsed repeated event rows:   "
    f"{EXPECTED_EXTRA_EVENTS:,}"
)

print(
    f"Active Investor-period slots:    "
    f"{EXPECTED_ACTIVE_INVESTOR_PERIODS:,}"
)


print()
print(
    "Same pair repeated in period:"
)

print(
    "  one attention item"
)


print()
print(
    "Empty historical period:"
)

print(
    "  retained as temporal slot"
)

print(
    "  x_i,h = zero_80"
)


print()
print(
    "No-history Investor:"
)

print(
    "  no special embedding"
)

print(
    "  historical inputs all zero_80"
)


print()
print(
    "T0 target event:"
)

print(
    "  no preceding sequence"
)

print(
    "  trend feature = zero_40"
)


print()
print(
    "T60 target:"
)

print(
    "  history = T0-T59"
)

print(
    "  T60 information consumed = NO"
)


print()
print(
    "Within-period chronology:       NONE"
)

print(
    "Within-period deterministic "
    "ordering: startup_node_index ASC"
)


print()
print(
    "History set semantics:          FROZEN"
)

print(
    "Empty-period representation:    FROZEN"
)

print(
    "No-history input behavior:      FROZEN"
)

print(
    "T0 target behavior:             FROZEN"
)

print(
    "Temporal leakage policy:        FROZEN"
)


print()
print("Outputs:")

for path in [
    membership_path,
    active_slots_path,
    investor_summary_path,
    contract_path,
    audit_path,
    hash_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.3.1b STATUS: COMPLETE — "
    "TREND-HISTORY SEMANTICS FROZEN"
)