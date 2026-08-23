from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# =============================================================================
# PHASE 4.3.1a — TREND-INPUT TEMPORAL SCHEMA AND HISTORY AUDIT
#
# PURPOSE
# -------
# Audit the frozen Phase-2 temporal interaction layer before reconstructing
# the ITRS trend-extraction module.
#
# IMPORTANT
# ---------
# This script is AUDIT ONLY.
#
# It does NOT:
#   - modify Phase-2 temporal assignments,
#   - redefine T0..T60,
#   - create trend tensors,
#   - collapse repeated investor-startup events,
#   - define empty-period behavior,
#   - define cold-investor trend behavior,
#   - train attention or GRU components.
#
# Phase-2 decisions remain frozen.
# =============================================================================


# =============================================================================
# INPUTS
# =============================================================================

SPLIT_PATH = Path(
    "data/experimental/phase_2/"
    "model_ready/"
    "interactions_itrs_temporal_split.parquet"
)

HOLDOUT_PATH = Path(
    "data/experimental/phase_2/"
    "model_ready/"
    "t60_holdout_pair_manifest.parquet"
)

NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/"
    "model_ready/"
    "node_index.parquet"
)


# =============================================================================
# OUTPUTS
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_4/"
    "audits"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FROZEN PHASE-2 EXPECTATIONS
# =============================================================================

EXPECTED_EXPERIMENT_ROWS = 1_195_937

EXPECTED_T0 = 100_173
EXPECTED_T1_T59 = 1_073_249

EXPECTED_HISTORICAL = 1_173_422
EXPECTED_T60 = 22_515

EXPECTED_VALIDATION = 2_251
EXPECTED_TEST = 20_264


# =============================================================================
# FROZEN ENTITY POPULATION
# =============================================================================

EXPECTED_INVESTORS = 165_975
EXPECTED_STARTUPS = 311_589


# =============================================================================
# FROZEN PHASE-2 MODEL-READY COLUMN NAMES
#
# These are NOT discovered in Phase 4.
#
# They were frozen and documented during Phase 2.
# =============================================================================

PERIOD_COLUMN = "segment_number"
PERIOD_LABEL_COLUMN = "segment_label"
TEMPORAL_ROLE_COLUMN = "temporal_role"
SPLIT_COLUMN = "experiment_split"


# =============================================================================
# EXPECTED TEMPORAL-ROLE VALUES
# =============================================================================

ROLE_T0 = "compressed_prehistory"
ROLE_HISTORY = "detailed_train_history"
ROLE_T60 = "t60_evaluation_pool"


# =============================================================================
# EXPECTED SPLIT VALUES
# =============================================================================

SPLIT_TRAIN = "train"
SPLIT_VALIDATION = "validation"
SPLIT_TEST = "test"


# =============================================================================
# HELPERS
# =============================================================================

def banner(title):

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def print_schema(path):

    parquet_file = pq.ParquetFile(
        path
    )

    schema = parquet_file.schema_arrow

    for i, field in enumerate(
        schema
    ):

        print(
            f"{i:>3}. "
            f"{field.name:<45} "
            f"{field.type}"
        )

    return [
        field.name
        for field in schema
    ]


# =============================================================================
# START
# =============================================================================

banner(
    "PHASE 4.3.1a — "
    "TREND-INPUT TEMPORAL SCHEMA AND HISTORY AUDIT"
)


# =============================================================================
# 1. FILE EXISTENCE
# =============================================================================

banner(
    "INPUT FILE EXISTENCE"
)


for path in [
    SPLIT_PATH,
    HOLDOUT_PATH,
    NODE_INDEX_PATH,
]:

    exists = path.exists()

    print(
        f"{str(path):<100} "
        f"{'FOUND' if exists else 'MISSING'}"
    )

    if not exists:

        raise FileNotFoundError(
            path
        )


# =============================================================================
# 2. INSPECT FROZEN PHASE-2 SCHEMA
# =============================================================================

banner(
    "PHASE-2 TEMPORAL SPLIT SCHEMA"
)


columns = print_schema(
    SPLIT_PATH
)


# =============================================================================
# 3. VERIFY REQUIRED CANONICAL + TEMPORAL FIELDS
# =============================================================================

required_columns = [
    "interaction_id",
    "funding_round_id",
    "investor_id",
    "startup_id",
    "announced_on",

    PERIOD_COLUMN,
    PERIOD_LABEL_COLUMN,
    TEMPORAL_ROLE_COLUMN,
    SPLIT_COLUMN,
]


missing_required = [
    column
    for column in required_columns
    if column not in columns
]


if missing_required:

    raise AssertionError(
        "Frozen Phase-2 model-ready schema changed. "
        f"Missing fields: {missing_required}"
    )


print()
print(
    "Frozen Phase-2 model-ready schema fields: PASS"
)

print()
print(
    f"Temporal-period column: "
    f"{PERIOD_COLUMN}"
)

print(
    f"Temporal-label column:  "
    f"{PERIOD_LABEL_COLUMN}"
)

print(
    f"Temporal-role column:   "
    f"{TEMPORAL_ROLE_COLUMN}"
)

print(
    f"Experiment-split column:"
    f" {SPLIT_COLUMN}"
)


# =============================================================================
# 4. LOAD TEMPORAL EXPERIMENT
# =============================================================================

banner(
    "LOADING FROZEN TEMPORAL EXPERIMENT"
)


usecols = [
    "interaction_id",
    "funding_round_id",
    "investor_id",
    "startup_id",
    "announced_on",

    PERIOD_COLUMN,
    PERIOD_LABEL_COLUMN,
    TEMPORAL_ROLE_COLUMN,
    SPLIT_COLUMN,
]


df = pd.read_parquet(
    SPLIT_PATH,
    columns=usecols,
)


print(
    f"Experiment rows: "
    f"{len(df):,}"
)


if (
    len(df)
    != EXPECTED_EXPERIMENT_ROWS
):

    raise AssertionError(
        "Phase-2 temporal experiment "
        "population changed."
    )


# =============================================================================
# 5. BASIC ROW INTEGRITY
# =============================================================================

banner(
    "ROW-LEVEL INTEGRITY"
)


duplicate_interaction_ids = int(
    df[
        "interaction_id"
    ]
    .duplicated()
    .sum()
)


null_interaction_ids = int(
    df[
        "interaction_id"
    ]
    .isna()
    .sum()
)


null_investor_ids = int(
    df[
        "investor_id"
    ]
    .isna()
    .sum()
)


null_startup_ids = int(
    df[
        "startup_id"
    ]
    .isna()
    .sum()
)


null_dates = int(
    df[
        "announced_on"
    ]
    .isna()
    .sum()
)


print(
    f"Duplicate interaction IDs: "
    f"{duplicate_interaction_ids:,}"
)

print(
    f"Null interaction IDs:      "
    f"{null_interaction_ids:,}"
)

print(
    f"Null investor IDs:         "
    f"{null_investor_ids:,}"
)

print(
    f"Null startup IDs:          "
    f"{null_startup_ids:,}"
)

print(
    f"Null announced_on:         "
    f"{null_dates:,}"
)


if any(
    [
        duplicate_interaction_ids,
        null_interaction_ids,
        null_investor_ids,
        null_startup_ids,
        null_dates,
    ]
):

    raise AssertionError(
        "Unexpected row-level integrity "
        "failure in frozen Phase-2 table."
    )


# =============================================================================
# 6. AUDIT RAW PERIOD VALUES
# =============================================================================

banner(
    "TEMPORAL PERIOD VALUES"
)


period_counts = (
    df[
        PERIOD_COLUMN
    ]
    .value_counts(
        dropna=False
    )
    .sort_index()
)


print(
    period_counts.to_string()
)


print()
print(
    f"Period dtype: "
    f"{df[PERIOD_COLUMN].dtype}"
)


# =============================================================================
# 7. NORMALIZE SEGMENT NUMBER
#
# The frozen model-ready field should already be integer-valued 0..60.
# We still normalize defensively for auditing without changing the source.
# =============================================================================

def normalize_period(value):

    if pd.isna(value):

        raise ValueError(
            "Null segment_number."
        )


    if isinstance(
        value,
        (int, np.integer),
    ):

        period = int(
            value
        )

    elif isinstance(
        value,
        (float, np.floating),
    ):

        if not float(
            value
        ).is_integer():

            raise ValueError(
                f"Non-integer segment_number: "
                f"{repr(value)}"
            )

        period = int(
            value
        )

    else:

        text = str(
            value
        ).strip()

        if text.isdigit():

            period = int(
                text
            )

        else:

            raise ValueError(
                f"Unsupported segment_number: "
                f"{repr(value)}"
            )


    if not (
        0 <= period <= 60
    ):

        raise ValueError(
            f"segment_number outside "
            f"0..60: {period}"
        )


    return period


df[
    "period_int"
] = (
    df[
        PERIOD_COLUMN
    ]
    .map(
        normalize_period
    )
    .astype(
        np.int16
    )
)


normalized_counts = (
    df[
        "period_int"
    ]
    .value_counts()
    .sort_index()
)


expected_periods = set(
    range(61)
)


observed_periods = set(
    normalized_counts
    .index
    .astype(int)
)


if (
    observed_periods
    != expected_periods
):

    missing_periods = sorted(
        expected_periods
        - observed_periods
    )

    extra_periods = sorted(
        observed_periods
        - expected_periods
    )

    raise AssertionError(
        "Expected exact segment_number "
        "coverage 0..60.\n"
        f"Missing: {missing_periods}\n"
        f"Extra:   {extra_periods}"
    )


print()
print(
    "Normalized temporal periods: "
    "0..60 PASS"
)


# =============================================================================
# 8. SEGMENT NUMBER ↔ SEGMENT LABEL INTEGRITY
# =============================================================================

banner(
    "SEGMENT NUMBER / LABEL INTEGRITY"
)


expected_segment_labels = (
    "T"
    + df[
        "period_int"
    ]
    .astype(str)
)


actual_segment_labels = (
    df[
        PERIOD_LABEL_COLUMN
    ]
    .astype(str)
)


segment_label_mismatches = int(
    (
        actual_segment_labels
        != expected_segment_labels
    ).sum()
)


print(
    f"segment_number / segment_label "
    f"mismatches: "
    f"{segment_label_mismatches:,}"
)


if segment_label_mismatches:

    mismatch_sample = (
        df.loc[
            actual_segment_labels
            != expected_segment_labels,
            [
                "interaction_id",
                PERIOD_COLUMN,
                PERIOD_LABEL_COLUMN,
            ],
        ]
        .head(20)
    )

    print()
    print(
        mismatch_sample.to_string(
            index=False
        )
    )

    raise AssertionError(
        "Frozen segment_number and "
        "segment_label disagree."
    )


print(
    "segment_number ↔ segment_label: PASS"
)


# =============================================================================
# 9. SEGMENT NUMBER ↔ TEMPORAL ROLE INTEGRITY
# =============================================================================

banner(
    "SEGMENT / TEMPORAL-ROLE INTEGRITY"
)


expected_temporal_role = np.where(
    df[
        "period_int"
    ].eq(0),

    ROLE_T0,

    np.where(
        df[
            "period_int"
        ].between(
            1,
            59,
        ),

        ROLE_HISTORY,

        ROLE_T60,
    ),
)


actual_temporal_role = (
    df[
        TEMPORAL_ROLE_COLUMN
    ]
    .astype(str)
    .to_numpy()
)


temporal_role_mismatches = int(
    np.sum(
        actual_temporal_role
        != expected_temporal_role
    )
)


print(
    f"segment / temporal_role mismatches: "
    f"{temporal_role_mismatches:,}"
)


if temporal_role_mismatches:

    mismatch_mask = (
        actual_temporal_role
        != expected_temporal_role
    )

    sample = df.loc[
        mismatch_mask,
        [
            "interaction_id",
            PERIOD_COLUMN,
            PERIOD_LABEL_COLUMN,
            TEMPORAL_ROLE_COLUMN,
        ],
    ].head(20)

    print()
    print(
        sample.to_string(
            index=False
        )
    )

    raise AssertionError(
        "Frozen temporal_role does not "
        "match segment_number."
    )


role_counts = (
    df[
        TEMPORAL_ROLE_COLUMN
    ]
    .value_counts()
)


print()
print(
    role_counts.to_string()
)


print()
print(
    "segment_number ↔ temporal_role: PASS"
)


# =============================================================================
# 10. EXPERIMENT SPLIT AUDIT
# =============================================================================

banner(
    "EXPERIMENT SPLIT VALUES"
)


split_counts = (
    df[
        SPLIT_COLUMN
    ]
    .astype(str)
    .value_counts(
        dropna=False
    )
)


print(
    split_counts.to_string()
)


expected_split_counts = {
    SPLIT_TRAIN:
        EXPECTED_HISTORICAL,

    SPLIT_VALIDATION:
        EXPECTED_VALIDATION,

    SPLIT_TEST:
        EXPECTED_TEST,
}


actual_split_counts = {
    str(key):
        int(value)

    for key, value
    in split_counts.items()
}


if (
    actual_split_counts
    != expected_split_counts
):

    raise AssertionError(
        "Frozen experiment_split counts changed.\n"
        f"Expected: {expected_split_counts}\n"
        f"Actual:   {actual_split_counts}"
    )


print()
print(
    "Frozen experiment_split counts: PASS"
)


# =============================================================================
# 11. SEGMENT ↔ SPLIT CONSISTENCY
#
# T0-T59 must all be train.
# T60 must contain only validation/test.
# =============================================================================

banner(
    "SEGMENT / EXPERIMENT-SPLIT CONSISTENCY"
)


history_not_train = int(
    (
        df[
            "period_int"
        ]
        .le(59)
        &
        ~df[
            SPLIT_COLUMN
        ]
        .astype(str)
        .eq(
            SPLIT_TRAIN
        )
    ).sum()
)


t60_marked_train = int(
    (
        df[
            "period_int"
        ]
        .eq(60)
        &
        df[
            SPLIT_COLUMN
        ]
        .astype(str)
        .eq(
            SPLIT_TRAIN
        )
    ).sum()
)


t60_invalid_split = int(
    (
        df[
            "period_int"
        ]
        .eq(60)
        &
        ~df[
            SPLIT_COLUMN
        ]
        .astype(str)
        .isin(
            [
                SPLIT_VALIDATION,
                SPLIT_TEST,
            ]
        )
    ).sum()
)


print(
    f"T0-T59 rows not marked train: "
    f"{history_not_train:,}"
)

print(
    f"T60 rows marked train:         "
    f"{t60_marked_train:,}"
)

print(
    f"T60 rows with invalid split:   "
    f"{t60_invalid_split:,}"
)


if any(
    [
        history_not_train,
        t60_marked_train,
        t60_invalid_split,
    ]
):

    raise AssertionError(
        "Frozen segment/split relationship "
        "is inconsistent."
    )


print()
print(
    "segment_number ↔ experiment_split: PASS"
)


# =============================================================================
# 12. FROZEN PHASE-2 COUNT INTEGRITY
# =============================================================================

banner(
    "FROZEN PHASE-2 COUNT INTEGRITY"
)


t0_count = int(
    (
        df[
            "period_int"
        ]
        == 0
    ).sum()
)


t1_t59_count = int(
    (
        df[
            "period_int"
        ]
        .between(
            1,
            59,
        )
    ).sum()
)


historical_count = int(
    (
        df[
            "period_int"
        ]
        <= 59
    ).sum()
)


t60_count = int(
    (
        df[
            "period_int"
        ]
        == 60
    ).sum()
)


validation_count = int(
    df[
        SPLIT_COLUMN
    ]
    .astype(str)
    .eq(
        SPLIT_VALIDATION
    )
    .sum()
)


test_count = int(
    df[
        SPLIT_COLUMN
    ]
    .astype(str)
    .eq(
        SPLIT_TEST
    )
    .sum()
)


print(
    f"T0:             "
    f"{t0_count:,}"
)

print(
    f"T1-T59:         "
    f"{t1_t59_count:,}"
)

print(
    f"T0-T59 history: "
    f"{historical_count:,}"
)

print(
    f"T60:            "
    f"{t60_count:,}"
)

print(
    f"Validation:     "
    f"{validation_count:,}"
)

print(
    f"Test:           "
    f"{test_count:,}"
)


assert (
    t0_count
    == EXPECTED_T0
)

assert (
    t1_t59_count
    == EXPECTED_T1_T59
)

assert (
    historical_count
    == EXPECTED_HISTORICAL
)

assert (
    t60_count
    == EXPECTED_T60
)

assert (
    validation_count
    == EXPECTED_VALIDATION
)

assert (
    test_count
    == EXPECTED_TEST
)


# =============================================================================
# 13. HISTORICAL TREND INPUT ISOLATION
#
# For T60 prediction, only T0-T59 may be used as trend history.
# =============================================================================

banner(
    "HISTORICAL TREND INPUT ISOLATION"
)


history = (
    df.loc[
        df[
            "period_int"
        ]
        <= 59
    ]
    .copy()
)


if (
    len(history)
    != EXPECTED_HISTORICAL
):

    raise AssertionError(
        "Historical trend population changed."
    )


if (
    history[
        "period_int"
    ].max()
    != 59
):

    raise AssertionError(
        "Historical max segment "
        "must be T59."
    )


t60_rows_in_history = int(
    history[
        "period_int"
    ]
    .eq(60)
    .sum()
)


if t60_rows_in_history:

    raise AssertionError(
        "T60 leakage into trend history."
    )


history_validation_rows = int(
    history[
        SPLIT_COLUMN
    ]
    .astype(str)
    .eq(
        SPLIT_VALIDATION
    )
    .sum()
)


history_test_rows = int(
    history[
        SPLIT_COLUMN
    ]
    .astype(str)
    .eq(
        SPLIT_TEST
    )
    .sum()
)


print(
    f"Historical events isolated: "
    f"{len(history):,}"
)

print(
    f"T60 events in history:      "
    f"{t60_rows_in_history:,}"
)

print(
    f"Validation events in history:"
    f" {history_validation_rows:,}"
)

print(
    f"Test events in history:      "
    f"{history_test_rows:,}"
)


if any(
    [
        history_validation_rows,
        history_test_rows,
    ]
):

    raise AssertionError(
        "Held-out split rows leaked "
        "into historical trend input."
    )


print()
print(
    "Historical T0-T59 isolation: PASS"
)


# =============================================================================
# 14. WITHIN-PERIOD INVESTOR-STARTUP MULTIPLICITY
#
# The ITRS paper describes v_i,h as a set of startups.
#
# We do NOT collapse them yet.
# We only quantify how much duplication exists in Crunchbase.
# =============================================================================

banner(
    "WITHIN-PERIOD INVESTOR-STARTUP MULTIPLICITY"
)


pair_period = (
    history
    .groupby(
        [
            "investor_id",
            "startup_id",
            "period_int",
        ],
        sort=False,
    )
    .agg(
        event_count=(
            "interaction_id",
            "size",
        ),

        funding_round_count=(
            "funding_round_id",
            "nunique",
        ),

        first_announced_on=(
            "announced_on",
            "min",
        ),

        last_announced_on=(
            "announced_on",
            "max",
        ),
    )
    .reset_index()
)


unique_pair_periods = int(
    len(
        pair_period
    )
)


repeated_pair_periods = int(
    (
        pair_period[
            "event_count"
        ]
        > 1
    ).sum()
)


extra_event_rows = int(
    (
        pair_period[
            "event_count"
        ]
        - 1
    )
    .clip(
        lower=0
    )
    .sum()
)


max_events_same_pair_period = int(
    pair_period[
        "event_count"
    ]
    .max()
)


repeated_pair_period_share = (
    repeated_pair_periods
    / unique_pair_periods
    if unique_pair_periods
    else 0.0
)


extra_event_share = (
    extra_event_rows
    / len(history)
    if len(history)
    else 0.0
)


print(
    f"Historical event rows:                 "
    f"{len(history):,}"
)

print(
    f"Unique Investor-Startup-Period tuples:  "
    f"{unique_pair_periods:,}"
)

print(
    f"Tuples with >1 event:                  "
    f"{repeated_pair_periods:,}"
)

print(
    f"Repeated tuple share:                  "
    f"{repeated_pair_period_share:.4%}"
)

print(
    f"Extra rows beyond set membership:      "
    f"{extra_event_rows:,}"
)

print(
    f"Extra-event share of historical rows:  "
    f"{extra_event_share:.4%}"
)

print(
    f"Maximum events in one tuple:           "
    f"{max_events_same_pair_period:,}"
)


# =============================================================================
# 15. EVENT-COUNT MULTIPLICITY DISTRIBUTION
# =============================================================================

banner(
    "PAIR-PERIOD EVENT MULTIPLICITY DISTRIBUTION"
)


multiplicity_distribution = (
    pair_period[
        "event_count"
    ]
    .value_counts()
    .sort_index()
    .rename_axis(
        "events_per_pair_period"
    )
    .reset_index(
        name="pair_period_tuples"
    )
)


multiplicity_distribution[
    "share"
] = (
    multiplicity_distribution[
        "pair_period_tuples"
    ]
    / unique_pair_periods
)


print(
    multiplicity_distribution
    .head(30)
    .to_string(
        index=False
    )
)


# =============================================================================
# 16. INVESTOR-PERIOD ACTIVITY
#
# Because pair_period now has one row per Investor-Startup-Period,
# startup count inside each Investor-period is naturally deduplicated
# for audit purposes.
# =============================================================================

banner(
    "INVESTOR-PERIOD ACTIVITY"
)


investor_period = (
    pair_period
    .groupby(
        [
            "investor_id",
            "period_int",
        ],
        sort=False,
    )
    .agg(
        unique_startups=(
            "startup_id",
            "nunique",
        ),

        unique_pair_memberships=(
            "startup_id",
            "size",
        ),

        source_events=(
            "event_count",
            "sum",
        ),
    )
    .reset_index()
)


if not np.array_equal(
    investor_period[
        "unique_startups"
    ].to_numpy(),

    investor_period[
        "unique_pair_memberships"
    ].to_numpy(),
):

    raise AssertionError(
        "pair_period unexpectedly contains "
        "duplicate startup membership."
    )


active_investors = int(
    history[
        "investor_id"
    ]
    .nunique()
)


never_historical_investors = (
    EXPECTED_INVESTORS
    - active_investors
)


if (
    never_historical_investors
    < 0
):

    raise AssertionError(
        "Historical investor count exceeds "
        "frozen Investor population."
    )


print(
    f"Investors with >=1 T0-T59 event: "
    f"{active_investors:,}"
)

print(
    f"Investors with zero T0-T59 events:"
    f" {never_historical_investors:,}"
)


print()
print(
    "Unique startups per active "
    "Investor-period:"
)


print(
    investor_period[
        "unique_startups"
    ]
    .describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
    .to_string()
)


# =============================================================================
# 17. FIRST HISTORICAL ACTIVE PERIOD PER INVESTOR
# =============================================================================

banner(
    "FIRST ACTIVE PERIOD"
)


first_active = (
    history
    .groupby(
        "investor_id"
    )[
        "period_int"
    ]
    .min()
    .rename(
        "first_active_period"
    )
    .reset_index()
)


first_active_counts = (
    first_active[
        "first_active_period"
    ]
    .value_counts()
    .sort_index()
)


print(
    first_active_counts.to_string()
)


t0_first_active_investors = int(
    (
        first_active[
            "first_active_period"
        ]
        == 0
    ).sum()
)


print()
print(
    f"Investors first active in T0: "
    f"{t0_first_active_investors:,}"
)


# =============================================================================
# 18. ACTIVE-PERIOD / EMPTY-PERIOD BURDEN
#
# T60 prediction conceptually has up to 60 historical slots:
# T0, T1, ..., T59.
#
# We measure active slots but DO NOT yet define how empty slots enter the GRU.
# =============================================================================

banner(
    "T60 HISTORICAL PERIOD SPARSITY"
)


active_period_counts = (
    investor_period
    .groupby(
        "investor_id"
    )[
        "period_int"
    ]
    .nunique()
    .rename(
        "active_period_count_t0_t59"
    )
    .reset_index()
)


active_period_counts[
    "empty_period_count_t0_t59"
] = (
    60
    - active_period_counts[
        "active_period_count_t0_t59"
    ]
)


if (
    active_period_counts[
        "active_period_count_t0_t59"
    ]
    .gt(60)
    .any()
):

    raise AssertionError(
        "Investor has >60 active "
        "historical periods."
    )


print(
    active_period_counts[
        [
            "active_period_count_t0_t59",
            "empty_period_count_t0_t59",
        ]
    ]
    .describe(
        percentiles=[
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
    .to_string()
)


# =============================================================================
# 19. GLOBAL ACTIVE-PERIOD DENSITY
# =============================================================================

possible_active_slots = (
    active_investors
    * 60
)


observed_active_slots = int(
    len(
        investor_period
    )
)


global_empty_slots = (
    possible_active_slots
    - observed_active_slots
)


global_active_density = (
    observed_active_slots
    / possible_active_slots
    if possible_active_slots
    else 0.0
)


print()
print(
    f"Possible historical slots among "
    f"historically active Investors: "
    f"{possible_active_slots:,}"
)

print(
    f"Observed active Investor-periods: "
    f"{observed_active_slots:,}"
)

print(
    f"Empty Investor-period slots:       "
    f"{global_empty_slots:,}"
)

print(
    f"Historical slot activity density:  "
    f"{global_active_density:.4%}"
)


# =============================================================================
# 20. T60 HISTORY COVERAGE
# =============================================================================

banner(
    "T60 INVESTOR HISTORY COVERAGE"
)


t60 = (
    df.loc[
        df[
            "period_int"
        ]
        .eq(60)
    ]
    .copy()
)


historical_investor_ids = set(
    history[
        "investor_id"
    ]
    .astype(str)
)


t60[
    "investor_has_prior_history"
] = (
    t60[
        "investor_id"
    ]
    .astype(str)
    .isin(
        historical_investor_ids
    )
)


t60_investors = int(
    t60[
        "investor_id"
    ]
    .nunique()
)


t60_investors_with_history = int(
    t60.loc[
        t60[
            "investor_has_prior_history"
        ],
        "investor_id",
    ]
    .nunique()
)


t60_investors_without_history = (
    t60_investors
    - t60_investors_with_history
)


t60_events_with_history = int(
    t60[
        "investor_has_prior_history"
    ]
    .sum()
)


t60_events_without_history = int(
    (
        ~t60[
            "investor_has_prior_history"
        ]
    ).sum()
)


t60_cold_investor_share = (
    t60_investors_without_history
    / t60_investors
    if t60_investors
    else 0.0
)


t60_cold_event_share = (
    t60_events_without_history
    / len(t60)
    if len(t60)
    else 0.0
)


print(
    f"Unique T60 Investors:              "
    f"{t60_investors:,}"
)

print(
    f"T60 Investors with prior history: "
    f"{t60_investors_with_history:,}"
)

print(
    f"T60 Investors with no history:    "
    f"{t60_investors_without_history:,}"
)

print(
    f"Cold-Investor share:              "
    f"{t60_cold_investor_share:.4%}"
)

print()

print(
    f"T60 events with prior history:    "
    f"{t60_events_with_history:,}"
)

print(
    f"T60 events with no history:       "
    f"{t60_events_without_history:,}"
)

print(
    f"Cold-Investor event share:        "
    f"{t60_cold_event_share:.4%}"
)


# =============================================================================
# 21. T60 PRIOR ACTIVE-PERIOD DEPTH
# =============================================================================

banner(
    "T60 INVESTOR PRIOR ACTIVE-PERIOD DEPTH"
)


t60_unique_investors = (
    t60[
        [
            "investor_id",
            "investor_has_prior_history",
        ]
    ]
    .drop_duplicates(
        subset=[
            "investor_id"
        ]
    )
)


t60_history_depth = (
    t60_unique_investors
    .merge(
        active_period_counts,
        on="investor_id",
        how="left",
        validate="one_to_one",
    )
)


t60_history_depth[
    "active_period_count_t0_t59"
] = (
    t60_history_depth[
        "active_period_count_t0_t59"
    ]
    .fillna(0)
    .astype(
        np.int16
    )
)


t60_history_depth[
    "empty_period_count_t0_t59"
] = (
    60
    - t60_history_depth[
        "active_period_count_t0_t59"
    ]
)


print(
    t60_history_depth[
        "active_period_count_t0_t59"
    ]
    .describe(
        percentiles=[
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
    .to_string()
)


# =============================================================================
# 22. PHASE-3 NODE INDEX SCHEMA
#
# No mapping decisions are made here.
# This merely records the deterministic model node-index schema available
# for the next trend materialization stage.
# =============================================================================

banner(
    "PHASE-3 NODE INDEX SCHEMA"
)


node_index_columns = print_schema(
    NODE_INDEX_PATH
)


print()
print(
    "No Phase-3 node mapping "
    "decision is made in 4.3.1a."
)


# =============================================================================
# 23. SAVE AUDIT OUTPUTS
# =============================================================================

banner(
    "SAVING AUDIT OUTPUTS"
)


period_counts_path = (
    OUT_DIR
    / "trend_period_counts.csv"
)


multiplicity_distribution_path = (
    OUT_DIR
    / "trend_pair_period_multiplicity_distribution.csv"
)


pair_period_path = (
    OUT_DIR
    / "trend_investor_startup_period_multiplicity.csv"
)


investor_period_path = (
    OUT_DIR
    / "trend_investor_period_activity.csv"
)


first_active_path = (
    OUT_DIR
    / "trend_investor_first_active_period.csv"
)


active_period_path = (
    OUT_DIR
    / "trend_investor_active_period_counts.csv"
)


t60_history_path = (
    OUT_DIR
    / "trend_t60_history_coverage.csv"
)


t60_depth_path = (
    OUT_DIR
    / "trend_t60_investor_history_depth.csv"
)


metadata_path = (
    OUT_DIR
    / "trend_temporal_input_audit_metadata.json"
)


# -----------------------------------------------------------------------------
# Period counts
# -----------------------------------------------------------------------------

(
    normalized_counts
    .rename(
        "events"
    )
    .rename_axis(
        "period_int"
    )
    .reset_index()
    .to_csv(
        period_counts_path,
        index=False,
    )
)


# -----------------------------------------------------------------------------
# Multiplicity distribution
# -----------------------------------------------------------------------------

multiplicity_distribution.to_csv(
    multiplicity_distribution_path,
    index=False,
)


# -----------------------------------------------------------------------------
# Detailed pair-period multiplicity
# -----------------------------------------------------------------------------

pair_period.to_csv(
    pair_period_path,
    index=False,
)


# -----------------------------------------------------------------------------
# Investor-period activity
# -----------------------------------------------------------------------------

investor_period.to_csv(
    investor_period_path,
    index=False,
)


# -----------------------------------------------------------------------------
# First active period
# -----------------------------------------------------------------------------

first_active.to_csv(
    first_active_path,
    index=False,
)


# -----------------------------------------------------------------------------
# Active-period counts
# -----------------------------------------------------------------------------

active_period_counts.to_csv(
    active_period_path,
    index=False,
)


# -----------------------------------------------------------------------------
# T60 event-level history coverage
# -----------------------------------------------------------------------------

t60[
    [
        "interaction_id",
        "investor_id",
        "startup_id",
        SPLIT_COLUMN,
        "investor_has_prior_history",
    ]
].to_csv(
    t60_history_path,
    index=False,
)


# -----------------------------------------------------------------------------
# T60 investor-level history depth
# -----------------------------------------------------------------------------

t60_history_depth.to_csv(
    t60_depth_path,
    index=False,
)


# =============================================================================
# 24. METADATA
# =============================================================================

metadata = {
    "phase":
        "4.3.1a",

    "status":
        "COMPLETE_AUDIT_ONLY",

    "purpose":
        (
            "Audit frozen Phase-2 temporal inputs "
            "required for ITRS trend extraction."
        ),

    "frozen_phase_2_schema": {
        "period_column":
            PERIOD_COLUMN,

        "period_label_column":
            PERIOD_LABEL_COLUMN,

        "temporal_role_column":
            TEMPORAL_ROLE_COLUMN,

        "split_column":
            SPLIT_COLUMN,
    },

    "phase_2_counts": {
        "experiment_rows":
            int(
                len(df)
            ),

        "t0":
            t0_count,

        "t1_t59":
            t1_t59_count,

        "historical_t0_t59":
            historical_count,

        "t60":
            t60_count,

        "validation":
            validation_count,

        "test":
            test_count,
    },

    "schema_integrity": {
        "segment_number_segment_label_mismatches":
            segment_label_mismatches,

        "segment_temporal_role_mismatches":
            temporal_role_mismatches,

        "history_not_train":
            history_not_train,

        "t60_marked_train":
            t60_marked_train,

        "t60_invalid_split":
            t60_invalid_split,
    },

    "trend_history_isolation": {
        "candidate_history_periods":
            "T0-T59",

        "t60_rows_consumed_as_history":
            t60_rows_in_history,

        "validation_rows_consumed_as_history":
            history_validation_rows,

        "test_rows_consumed_as_history":
            history_test_rows,
    },

    "pair_period_multiplicity": {
        "historical_event_rows":
            int(
                len(history)
            ),

        "unique_investor_startup_period_tuples":
            unique_pair_periods,

        "repeated_pair_period_tuples":
            repeated_pair_periods,

        "repeated_pair_period_share":
            repeated_pair_period_share,

        "extra_event_rows_beyond_set_membership":
            extra_event_rows,

        "extra_event_share":
            extra_event_share,

        "maximum_events_same_pair_period":
            max_events_same_pair_period,
    },

    "investor_history": {
        "historically_active_investors":
            active_investors,

        "never_historical_investors":
            never_historical_investors,

        "observed_investor_period_slots":
            observed_active_slots,

        "possible_slots_among_active_investors":
            possible_active_slots,

        "global_active_slot_density":
            global_active_density,
    },

    "t60_history_coverage": {
        "unique_t60_investors":
            t60_investors,

        "t60_investors_with_history":
            t60_investors_with_history,

        "t60_investors_without_history":
            t60_investors_without_history,

        "t60_cold_investor_share":
            t60_cold_investor_share,

        "t60_events_without_history":
            t60_events_without_history,

        "t60_cold_event_share":
            t60_cold_event_share,
    },

    "phase_2_reopened":
        False,

    "phase_3_reopened":
        False,

    "model_training_performed":
        False,

    "not_yet_frozen": [
        (
            "whether repeated Investor-Startup "
            "events inside one segment are collapsed "
            "to one set membership"
        ),

        (
            "representation of an Investor-period "
            "with zero invested startups"
        ),

        (
            "trend representation for Investors "
            "with no T0-T59 investment history"
        ),

        (
            "trend behavior for events occurring "
            "in T0 where there is no earlier segment"
        ),

        (
            "whether empty temporal periods remain "
            "explicit GRU steps or are omitted"
        ),

        (
            "exact trend-history tensor / ragged "
            "representation"
        ),
    ],
}


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
# 25. FINAL SUMMARY
# =============================================================================

banner(
    "PHASE 4.3.1a FINAL SUMMARY"
)


print(
    f"Temporal-period column:        "
    f"{PERIOD_COLUMN}"
)

print(
    f"Temporal-label column:         "
    f"{PERIOD_LABEL_COLUMN}"
)

print(
    f"Temporal-role column:          "
    f"{TEMPORAL_ROLE_COLUMN}"
)

print(
    f"Dataset-split column:          "
    f"{SPLIT_COLUMN}"
)


print()
print(
    f"T0-T59 historical events:      "
    f"{historical_count:,}"
)

print(
    f"T60 events excluded:            "
    f"{t60_count:,}"
)

print(
    f"T60 rows leaked into history:   "
    f"{t60_rows_in_history:,}"
)


print()
print(
    f"Unique pair-period memberships: "
    f"{unique_pair_periods:,}"
)

print(
    f"Repeated pair-period tuples:    "
    f"{repeated_pair_periods:,}"
)

print(
    f"Extra historical event rows:    "
    f"{extra_event_rows:,}"
)

print(
    f"Max events same pair/period:     "
    f"{max_events_same_pair_period:,}"
)


print()
print(
    f"Historical active Investors:     "
    f"{active_investors:,}"
)

print(
    f"Never-historical Investors:      "
    f"{never_historical_investors:,}"
)

print(
    f"Observed active period slots:    "
    f"{observed_active_slots:,}"
)

print(
    f"Active-slot density:             "
    f"{global_active_density:.4%}"
)


print()
print(
    f"Unique T60 Investors:            "
    f"{t60_investors:,}"
)

print(
    f"T60 Investors with no history:   "
    f"{t60_investors_without_history:,}"
)

print(
    f"T60 cold-Investor share:         "
    f"{t60_cold_investor_share:.4%}"
)

print(
    f"T60 events from no-history inv.: "
    f"{t60_events_without_history:,}"
)


print()
print(
    "segment_number ↔ label:          PASS"
)

print(
    "segment_number ↔ temporal_role:  PASS"
)

print(
    "segment_number ↔ split:          PASS"
)

print(
    "Frozen Phase-2 counts:           PASS"
)

print(
    "T60 trend-history exclusion:     PASS"
)


print()
print(
    "Historical set semantics:        "
    "NOT YET FROZEN"
)

print(
    "Empty-period representation:     "
    "NOT YET FROZEN"
)

print(
    "No-history trend representation: "
    "NOT YET FROZEN"
)

print(
    "Empty-period GRU handling:        "
    "NOT YET FROZEN"
)


print()
print("Outputs:")

for path in [
    period_counts_path,
    multiplicity_distribution_path,
    pair_period_path,
    investor_period_path,
    first_active_path,
    active_period_path,
    t60_history_path,
    t60_depth_path,
    metadata_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.3.1a STATUS: COMPLETE — "
    "TEMPORAL TREND INPUT AUDITED ONLY"
)