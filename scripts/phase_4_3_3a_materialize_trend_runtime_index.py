from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 4.3.3a — MATERIALIZE TREND RUNTIME HISTORY INDEX
#
# PURPOSE
# -------
# Convert the frozen sparse Investor-Startup-Period history into an immutable
# CSR-style runtime index suitable for neural forward passes.
#
# IMPORTANT
# ---------
# This stores ONLY frozen membership identities.
#
# It does NOT store:
#   - latent embeddings,
#   - description outputs,
#   - attention outputs,
#   - GRU outputs,
#   - learned trend vectors.
#
# Therefore the artifacts remain valid while model parameters change.
# =============================================================================


# =============================================================================
# INPUTS
# =============================================================================

MEMBERSHIP_PATH = Path(
    "data/experimental/phase_4/"
    "trend_contract/"
    "trend_history_membership.parquet"
)

HISTORY_CONTRACT_PATH = Path(
    "data/experimental/phase_4/"
    "trend_contract/"
    "trend_history_semantics_contract.json"
)

NEURAL_CONTRACT_PATH = Path(
    "data/experimental/phase_4/"
    "trend_neural_contract/"
    "trend_neural_contract.json"
)


# =============================================================================
# OUTPUTS
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_4/"
    "trend_runtime"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FROZEN DIMENSIONS / COUNTS
# =============================================================================

N_INVESTORS = 165_975
N_PERIODS = 60

N_SLOTS = (
    N_INVESTORS
    * N_PERIODS
)

EXPECTED_MEMBERSHIPS = 1_145_364
EXPECTED_ACTIVE_SLOTS = 554_171

EXPECTED_FIRST_STARTUP_INDEX = 165_975
EXPECTED_LAST_STARTUP_INDEX = 477_563


# =============================================================================
# HELPERS
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
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


# =============================================================================
# START
# =============================================================================

banner(
    "PHASE 4.3.3a — "
    "MATERIALIZE TREND RUNTIME HISTORY INDEX"
)


# =============================================================================
# 1. CONTRACT INTEGRITY
# =============================================================================

banner(
    "UPSTREAM CONTRACT INTEGRITY"
)


with open(
    HISTORY_CONTRACT_PATH,
    "r",
    encoding="utf-8",
) as f:

    history_contract = json.load(f)


with open(
    NEURAL_CONTRACT_PATH,
    "r",
    encoding="utf-8",
) as f:

    neural_contract = json.load(f)


if (
    history_contract.get("status")
    != "FROZEN"
):

    raise AssertionError(
        "Trend-history contract is not frozen."
    )


if (
    neural_contract.get("status")
    != "FROZEN"
):

    raise AssertionError(
        "Trend neural contract is not frozen."
    )


assert (
    history_contract[
        "frozen_counts"
    ][
        "unique_pair_period_memberships"
    ]
    == EXPECTED_MEMBERSHIPS
)


print(
    "Trend-history contract: PASS"
)

print(
    "Trend-neural contract:  PASS"
)


# =============================================================================
# 2. LOAD FROZEN MEMBERSHIPS
# =============================================================================

banner(
    "LOADING FROZEN HISTORY MEMBERSHIPS"
)


membership = pd.read_parquet(
    MEMBERSHIP_PATH,
    columns=[
        "trend_membership_row",
        "investor_node_index",
        "segment_number",
        "startup_node_index",
    ],
)


if (
    len(membership)
    != EXPECTED_MEMBERSHIPS
):

    raise AssertionError(
        "Frozen membership count changed."
    )


print(
    f"Membership rows: "
    f"{len(membership):,}"
)


# =============================================================================
# 3. VERIFY SOURCE ORDERING
#
# Phase 4.3.1b froze:
#
# investor_node_index ASC
# segment_number ASC
# startup_node_index ASC
# =============================================================================

banner(
    "SOURCE ORDERING INTEGRITY"
)


expected_order = (
    membership
    .sort_values(
        [
            "investor_node_index",
            "segment_number",
            "startup_node_index",
        ],
        kind="mergesort",
    )
    .index
    .to_numpy()
)


source_is_sorted = np.array_equal(
    expected_order,
    np.arange(
        len(membership)
    ),
)


print(
    f"Membership already in frozen order: "
    f"{source_is_sorted}"
)


if not source_is_sorted:

    raise AssertionError(
        "Frozen membership order changed."
    )


# =============================================================================
# 4. INDEX RANGE INTEGRITY
# =============================================================================

banner(
    "INDEX RANGE INTEGRITY"
)


investor_indices = (
    membership[
        "investor_node_index"
    ]
    .to_numpy(
        dtype=np.int64
    )
)


periods = (
    membership[
        "segment_number"
    ]
    .to_numpy(
        dtype=np.int64
    )
)


startup_indices = (
    membership[
        "startup_node_index"
    ]
    .to_numpy(
        dtype=np.int64
    )
)


if not np.all(
    (
        investor_indices >= 0
    )
    &
    (
        investor_indices < N_INVESTORS
    )
):

    raise AssertionError(
        "Investor index outside frozen range."
    )


if not np.all(
    (
        periods >= 0
    )
    &
    (
        periods < N_PERIODS
    )
):

    raise AssertionError(
        "Historical period outside T0-T59."
    )


if not np.all(
    (
        startup_indices
        >= EXPECTED_FIRST_STARTUP_INDEX
    )
    &
    (
        startup_indices
        <= EXPECTED_LAST_STARTUP_INDEX
    )
):

    raise AssertionError(
        "Startup node index outside "
        "frozen Startup role range."
    )


print(
    f"Investor range: "
    f"[{investor_indices.min()}, "
    f"{investor_indices.max()}]"
)

print(
    f"Period range:   "
    f"[{periods.min()}, "
    f"{periods.max()}]"
)

print(
    f"Startup range:  "
    f"[{startup_indices.min()}, "
    f"{startup_indices.max()}]"
)


# =============================================================================
# 5. CREATE FLATTENED INVESTOR-PERIOD SLOT
#
# slot = investor_index * 60 + period
# =============================================================================

banner(
    "FLATTENED INVESTOR-PERIOD SLOT INDEX"
)


slot_indices = (
    investor_indices
    * N_PERIODS
    + periods
)


if not np.all(
    (
        slot_indices >= 0
    )
    &
    (
        slot_indices < N_SLOTS
    )
):

    raise AssertionError(
        "Flattened slot outside valid range."
    )


# Membership source order should imply nondecreasing slots.
slot_monotonic = bool(
    np.all(
        slot_indices[1:]
        >= slot_indices[:-1]
    )
)


print(
    f"Possible Investor-period slots: "
    f"{N_SLOTS:,}"
)

print(
    f"Slot ordering monotonic:         "
    f"{slot_monotonic}"
)


if not slot_monotonic:

    raise AssertionError(
        "Flattened period slots are "
        "not monotonically ordered."
    )


# =============================================================================
# 6. BUILD CSR PERIOD POINTERS
#
# period_ptr has N_SLOTS + 1 entries.
#
# For flattened period slot k:
#
# startup_node_indices[
#     period_ptr[k] : period_ptr[k+1]
# ]
#
# gives the unique startups in that Investor-period.
# =============================================================================

banner(
    "BUILDING CSR PERIOD POINTERS"
)


slot_counts = np.bincount(
    slot_indices,
    minlength=N_SLOTS,
)


if len(slot_counts) != N_SLOTS:

    raise AssertionError(
        "Unexpected slot-count length."
    )


active_slot_count = int(
    np.count_nonzero(
        slot_counts
    )
)


empty_slot_count = (
    N_SLOTS
    - active_slot_count
)


if (
    active_slot_count
    != EXPECTED_ACTIVE_SLOTS
):

    raise AssertionError(
        "Active Investor-period count changed."
    )


if (
    int(
        slot_counts.sum()
    )
    != EXPECTED_MEMBERSHIPS
):

    raise AssertionError(
        "Slot counts do not reconstruct "
        "all memberships."
    )


period_ptr = np.empty(
    N_SLOTS + 1,
    dtype=np.int64,
)


period_ptr[0] = 0


np.cumsum(
    slot_counts,
    dtype=np.int64,
    out=period_ptr[1:],
)


if (
    period_ptr[-1]
    != EXPECTED_MEMBERSHIPS
):

    raise AssertionError(
        "Final CSR pointer does not equal "
        "membership population."
    )


print(
    f"Active slots: "
    f"{active_slot_count:,}"
)

print(
    f"Empty slots:  "
    f"{empty_slot_count:,}"
)

print(
    f"Final pointer:"
    f" {period_ptr[-1]:,}"
)


# =============================================================================
# 7. FLATTENED STARTUP MEMBERSHIP ARRAY
#
# Source membership is already ordered by slot then startup index.
# =============================================================================

banner(
    "FLATTENED STARTUP MEMBERSHIP ARRAY"
)


flat_startups = startup_indices.astype(
    np.int64,
    copy=True,
)


if (
    len(flat_startups)
    != EXPECTED_MEMBERSHIPS
):

    raise AssertionError(
        "Flattened startup membership "
        "population changed."
    )


print(
    f"Flattened Startup entries: "
    f"{len(flat_startups):,}"
)


# =============================================================================
# 8. ROUNDTRIP AUDIT
#
# Reconstruct membership rows through the CSR index and compare exactly
# against the frozen source table.
# =============================================================================

banner(
    "EXACT CSR ROUNDTRIP AUDIT"
)


reconstructed_slots = np.repeat(
    np.arange(
        N_SLOTS,
        dtype=np.int64,
    ),
    slot_counts,
)


if (
    len(reconstructed_slots)
    != EXPECTED_MEMBERSHIPS
):

    raise AssertionError(
        "Reconstructed slot population changed."
    )


slot_exact = np.array_equal(
    reconstructed_slots,
    slot_indices,
)


startup_exact = np.array_equal(
    flat_startups,
    startup_indices,
)


print(
    f"Slot roundtrip exact:    "
    f"{slot_exact}"
)

print(
    f"Startup roundtrip exact: "
    f"{startup_exact}"
)


if not (
    slot_exact
    and startup_exact
):

    raise AssertionError(
        "CSR runtime index does not "
        "exactly reconstruct frozen membership."
    )


# =============================================================================
# 9. WITHIN-SLOT STARTUP ORDER AUDIT
# =============================================================================

banner(
    "WITHIN-SLOT STARTUP ORDER AUDIT"
)


bad_order_slots = 0


active_slots = np.flatnonzero(
    slot_counts
)


for slot in active_slots:

    start = period_ptr[
        slot
    ]

    end = period_ptr[
        slot + 1
    ]

    values = flat_startups[
        start:end
    ]

    if (
        len(values) > 1
        and not np.all(
            values[1:]
            > values[:-1]
        )
    ):

        bad_order_slots += 1


print(
    f"Active slots checked: "
    f"{len(active_slots):,}"
)

print(
    f"Slots with invalid Startup "
    f"ordering: "
    f"{bad_order_slots:,}"
)


if bad_order_slots:

    raise AssertionError(
        "Startup set ordering changed."
    )


# =============================================================================
# 10. SELECTED LOOKUP EXAMPLES
# =============================================================================

banner(
    "RUNTIME LOOKUP EXAMPLES"
)


# Deterministic examples:
# first active slot, first empty slot, largest slot.

first_active_slot = int(
    active_slots[0]
)


first_empty_slot = int(
    np.flatnonzero(
        slot_counts == 0
    )[0]
)


largest_slot = int(
    np.argmax(
        slot_counts
    )
)


def inspect_slot(slot):

    investor = (
        slot // N_PERIODS
    )

    period = (
        slot % N_PERIODS
    )

    start = period_ptr[
        slot
    ]

    end = period_ptr[
        slot + 1
    ]

    startups = flat_startups[
        start:end
    ]

    return {
        "slot":
            int(slot),

        "investor_node_index":
            int(investor),

        "segment_number":
            int(period),

        "startup_count":
            int(
                len(startups)
            ),

        "first_startup":
            (
                int(startups[0])
                if len(startups)
                else None
            ),

        "last_startup":
            (
                int(startups[-1])
                if len(startups)
                else None
            ),
    }


examples = [
    inspect_slot(
        first_active_slot
    ),

    inspect_slot(
        first_empty_slot
    ),

    inspect_slot(
        largest_slot
    ),
]


for example in examples:

    print()

    print(
        f"slot={example['slot']}"
    )

    print(
        f"  Investor: "
        f"{example['investor_node_index']}"
    )

    print(
        f"  Period:   "
        f"T{example['segment_number']}"
    )

    print(
        f"  Startups: "
        f"{example['startup_count']}"
    )

    print(
        f"  First:    "
        f"{example['first_startup']}"
    )

    print(
        f"  Last:     "
        f"{example['last_startup']}"
    )


# =============================================================================
# 11. SAVE ARRAYS
# =============================================================================

banner(
    "SAVING RUNTIME ARRAYS"
)


period_ptr_path = (
    OUT_DIR
    / "trend_period_ptr.npy"
)


flat_startups_path = (
    OUT_DIR
    / "trend_startup_node_indices.npy"
)


slot_counts_path = (
    OUT_DIR
    / "trend_period_startup_counts.npy"
)


np.save(
    period_ptr_path,
    period_ptr,
)


np.save(
    flat_startups_path,
    flat_startups,
)


np.save(
    slot_counts_path,
    slot_counts.astype(
        np.int32
    ),
)


# =============================================================================
# 12. PERSISTENCE RELOAD
# =============================================================================

banner(
    "PERSISTENCE RELOAD AUDIT"
)


period_ptr_reload = np.load(
    period_ptr_path,
    mmap_mode="r",
)


flat_startups_reload = np.load(
    flat_startups_path,
    mmap_mode="r",
)


slot_counts_reload = np.load(
    slot_counts_path,
    mmap_mode="r",
)


ptr_exact = np.array_equal(
    period_ptr,
    period_ptr_reload,
)


startups_exact = np.array_equal(
    flat_startups,
    flat_startups_reload,
)


counts_exact = np.array_equal(
    slot_counts.astype(
        np.int32
    ),
    slot_counts_reload,
)


print(
    f"Period pointers exact: "
    f"{ptr_exact}"
)

print(
    f"Startup array exact:   "
    f"{startups_exact}"
)

print(
    f"Slot counts exact:     "
    f"{counts_exact}"
)


if not (
    ptr_exact
    and startups_exact
    and counts_exact
):

    raise AssertionError(
        "Runtime-array persistence failure."
    )


# =============================================================================
# 13. MEMORY FOOTPRINT
# =============================================================================

banner(
    "RUNTIME INDEX MEMORY FOOTPRINT"
)


ptr_bytes = (
    period_ptr.nbytes
)


startup_bytes = (
    flat_startups.nbytes
)


counts_bytes = (
    slot_counts.astype(
        np.int32
    ).nbytes
)


total_bytes = (
    ptr_bytes
    + startup_bytes
    + counts_bytes
)


print(
    f"period_ptr:        "
    f"{ptr_bytes / 1024**2:.2f} MiB"
)

print(
    f"startup indices:   "
    f"{startup_bytes / 1024**2:.2f} MiB"
)

print(
    f"period counts:     "
    f"{counts_bytes / 1024**2:.2f} MiB"
)

print(
    f"TOTAL:             "
    f"{total_bytes / 1024**2:.2f} MiB"
)


# =============================================================================
# 14. FREEZE INTEGRATION CONTRACT
# =============================================================================

banner(
    "FREEZING TREND RUNTIME CONTRACT"
)


contract = {

    "phase":
        "4.3.3a",

    "status":
        "FROZEN",

    "component":
        "ITRS trend runtime history indexing",

    "semantic_source":
        (
            "Phase 4.3.1b frozen "
            "Investor-Startup-Period set memberships"
        ),

    "indexing": {

        "investor_period_slot_equation":
            "slot = investor_node_index * 60 + segment_number",

        "investor_count":
            N_INVESTORS,

        "historical_period_count":
            N_PERIODS,

        "slot_count":
            N_SLOTS,

        "period_pointer_length":
            N_SLOTS + 1,

        "membership_count":
            EXPECTED_MEMBERSHIPS,

        "active_slots":
            EXPECTED_ACTIVE_SLOTS,

        "startup_order_within_slot":
            "startup_node_index ascending",
    },

    "lookup": {

        "startup_members_for_period":
            (
                "trend_startup_node_indices["
                "trend_period_ptr[slot]:"
                "trend_period_ptr[slot+1]]"
            ),

        "empty_period_test":
            (
                "trend_period_ptr[slot] == "
                "trend_period_ptr[slot+1]"
            ),
    },

    "runtime_model_semantics": {

        "target_T0":
            "trend zero_40; no history lookup",

        "target_Th":
            (
                "retrieve period slots "
                "T0 through T(h-1)"
            ),

        "target_T60":
            "retrieve T0 through T59",

        "same_target_period_consumed":
            False,

        "future_period_consumed":
            False,
    },

    "training_integration": {

        "trend_feature_persisted":
            False,

        "reason":
            (
                "Trend features depend on trainable "
                "latent embeddings, trainable "
                "description projections, attention, "
                "GRU, and output projection."
            ),

        "trend_recomputed_during_training":
            True,

        "deduplicate_within_minibatch_by":
            [
                "investor_node_index",
                "target_segment",
            ],

        "candidate_startup_affects_trend":
            False,

        "negative_candidate_reuses_same_trend":
            True,
    },

    "stored_information": {

        "learned_parameters":
            False,

        "latent_features":
            False,

        "description_features":
            False,

        "attention_outputs":
            False,

        "trend_features":
            False,

        "only_frozen_history_membership":
            True,
    },

    "artifacts": {

        "period_ptr":
            str(
                period_ptr_path
            ),

        "startup_node_indices":
            str(
                flat_startups_path
            ),

        "period_startup_counts":
            str(
                slot_counts_path
            ),
    },

    "upstream_reopened": {

        "phase_2":
            False,

        "phase_3":
            False,

        "phase_4_2":
            False,

        "phase_4_3_1":
            False,

        "phase_4_3_2":
            False,
    },
}


contract_path = (
    OUT_DIR
    / "trend_runtime_contract.json"
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
# 15. ARTIFACT HASHES
# =============================================================================

hash_records = []


for name, path in [
    (
        "period_ptr",
        period_ptr_path,
    ),

    (
        "startup_node_indices",
        flat_startups_path,
    ),

    (
        "period_startup_counts",
        slot_counts_path,
    ),

    (
        "runtime_contract",
        contract_path,
    ),
]:

    hash_records.append(
        {
            "artifact":
                name,

            "path":
                str(path),

            "sha256":
                sha256_file(
                    path
                ),

            "bytes":
                path.stat().st_size,
        }
    )


hash_df = pd.DataFrame(
    hash_records
)


hash_path = (
    OUT_DIR
    / "trend_runtime_artifact_hashes.csv"
)


hash_df.to_csv(
    hash_path,
    index=False,
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

banner(
    "PHASE 4.3.3a FINAL SUMMARY"
)


print(
    f"Investors:                   "
    f"{N_INVESTORS:,}"
)

print(
    f"Historical periods:          "
    f"{N_PERIODS}"
)

print(
    f"Possible period slots:       "
    f"{N_SLOTS:,}"
)

print(
    f"Active period slots:         "
    f"{active_slot_count:,}"
)

print(
    f"Empty period slots:          "
    f"{empty_slot_count:,}"
)

print(
    f"Startup memberships:         "
    f"{len(flat_startups):,}"
)


print()
print(
    "Runtime representation:"
)

print(
    "  CSR-style period pointers"
)

print(
    "  flattened Startup node indices"
)


print()
print(
    "Empty period:"
)

print(
    "  ptr[k] == ptr[k+1]"
)

print(
    "  model generates zero_80"
)


print()
print(
    "Trend features persisted:        NO"
)

print(
    "Trend recomputed while training: YES"
)

print(
    "Candidate Startup changes trend: NO"
)

print(
    "Batch duplicate trend reuse:     YES"
)


print()
print(
    "Exact source roundtrip:          PASS"
)

print(
    "Persistence reload:              PASS"
)


print()
print(
    f"Runtime-index memory:            "
    f"{total_bytes / 1024**2:.2f} MiB"
)


print()
print("Outputs:")

for path in [
    period_ptr_path,
    flat_startups_path,
    slot_counts_path,
    contract_path,
    hash_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.3.3a STATUS: COMPLETE — "
    "TREND RUNTIME HISTORY INDEX FROZEN"
)