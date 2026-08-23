from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# PHASE 4.3.3b — TREND RUNTIME INTEGRATION AUDIT
#
# PURPOSE
# -------
# Verify that the frozen CSR runtime index from Phase 4.3.3a is exactly
# equivalent to the frozen Parquet history representation from Phase 4.3.1b,
# both at:
#
#   1. membership lookup level
#   2. temporal sequence level
#   3. neural trend-output level
#
# NO TRAINING OCCURS.
#
# Audit-only synthetic latent/description vectors and deterministic model
# weights are used exactly as in Phase 4.3.2b.
# =============================================================================


# =============================================================================
# INPUTS
# =============================================================================

MEMBERSHIP_PATH = Path(
    "data/experimental/phase_4/"
    "trend_contract/"
    "trend_history_membership.parquet"
)

PERIOD_PTR_PATH = Path(
    "data/experimental/phase_4/"
    "trend_runtime/"
    "trend_period_ptr.npy"
)

STARTUP_INDEX_PATH = Path(
    "data/experimental/phase_4/"
    "trend_runtime/"
    "trend_startup_node_indices.npy"
)

PERIOD_COUNT_PATH = Path(
    "data/experimental/phase_4/"
    "trend_runtime/"
    "trend_period_startup_counts.npy"
)

RUNTIME_CONTRACT_PATH = Path(
    "data/experimental/phase_4/"
    "trend_runtime/"
    "trend_runtime_contract.json"
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

FORWARD_AUDIT_METADATA_PATH = Path(
    "data/experimental/phase_4/"
    "trend_module/"
    "trend_forward_audit_metadata.json"
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
# FROZEN CONSTANTS
# =============================================================================

N_INVESTORS = 165_975
N_PERIODS = 60

N_SLOTS = (
    N_INVESTORS
    * N_PERIODS
)

EXPECTED_MEMBERSHIPS = 1_145_364
EXPECTED_ACTIVE_SLOTS = 554_171

LATENT_DIM = 40
DESCRIPTION_DIM = 40

QUERY_DIM = 80
ITEM_DIM = 80
PERIOD_VECTOR_DIM = 80

GRU_HIDDEN_DIM = 40
GRU_NUM_LAYERS = 2

TREND_OUTPUT_DIM = 40


# =============================================================================
# HELPERS
# =============================================================================

def banner(title):

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def require(condition, message):

    if not condition:

        raise AssertionError(
            message
        )


# =============================================================================
# AUDIT-ONLY DETERMINISTIC FEATURES
#
# Same functions as Phase 4.3.2b.
# =============================================================================

def deterministic_latent(
    node_indices,
    salt,
):

    node_indices = np.asarray(
        node_indices,
        dtype=np.int64,
    ).reshape(
        -1,
        1,
    )

    dimensions = np.arange(
        LATENT_DIM,
        dtype=np.int64,
    ).reshape(
        1,
        -1,
    )

    values = (
        (
            node_indices * 37
            + dimensions * 101
            + salt
        )
        % 1009
    ).astype(
        np.float32
    )

    values = (
        values / 1009.0
        - 0.5
    )

    return torch.from_numpy(
        values
    )


def deterministic_description(
    node_indices,
    salt,
):

    node_indices = np.asarray(
        node_indices,
        dtype=np.int64,
    ).reshape(
        -1,
        1,
    )

    dimensions = np.arange(
        DESCRIPTION_DIM,
        dtype=np.int64,
    ).reshape(
        1,
        -1,
    )

    values = (
        (
            node_indices * 53
            + dimensions * 67
            + salt
        )
        % 1013
    ).astype(
        np.float32
    )

    values = (
        values / 1013.0
    )

    return torch.from_numpy(
        values
    )


def investor_query(
    investor_node_index,
):

    latent = deterministic_latent(
        [investor_node_index],
        salt=11,
    )

    description = deterministic_description(
        [investor_node_index],
        salt=23,
    )

    result = torch.cat(
        [
            latent,
            description,
        ],
        dim=1,
    )

    require(
        tuple(result.shape)
        == (1, 80),

        "Investor query shape mismatch.",
    )

    return result.squeeze(0)


def startup_items(
    startup_node_indices,
):

    startup_node_indices = np.asarray(
        startup_node_indices,
        dtype=np.int64,
    )

    if len(
        startup_node_indices
    ) == 0:

        return torch.empty(
            (
                0,
                80,
            ),
            dtype=torch.float32,
        )


    latent = deterministic_latent(
        startup_node_indices,
        salt=31,
    )

    description = deterministic_description(
        startup_node_indices,
        salt=47,
    )


    result = torch.cat(
        [
            latent,
            description,
        ],
        dim=1,
    )


    require(
        result.shape[1] == 80,

        "Startup item dimension mismatch.",
    )


    return result


# =============================================================================
# TREND EXTRACTOR
#
# Same frozen architecture as Phase 4.3.2a / 4.3.2b.
# =============================================================================

class TrendExtractor(nn.Module):

    def __init__(self):

        super().__init__()


        self.attention_weight = nn.Parameter(
            torch.empty(
                QUERY_DIM,
                ITEM_DIM,
            )
        )


        self.gru = nn.GRU(
            input_size=PERIOD_VECTOR_DIM,
            hidden_size=GRU_HIDDEN_DIM,
            num_layers=GRU_NUM_LAYERS,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )


        self.output_projection = nn.Linear(
            GRU_HIDDEN_DIM,
            TREND_OUTPUT_DIM,
            bias=False,
        )


        self.output_activation = nn.Sigmoid()


    def attend_period(
        self,
        query,
        items,
    ):

        if items.shape[0] == 0:

            return torch.zeros(
                PERIOD_VECTOR_DIM,
                dtype=torch.float32,
            )


        scores = (
            query
            @ self.attention_weight
            @ items.T
        )


        alpha = F.softmax(
            scores,
            dim=0,
        )


        return (
            alpha
            @ items
        )


    def encode_sequence(
        self,
        sequence,
        target_segment,
    ):

        if target_segment == 0:

            require(
                tuple(sequence.shape)
                == (0, 80),

                "T0 sequence must be empty.",
            )

            return torch.zeros(
                40,
                dtype=torch.float32,
            )


        require(
            tuple(sequence.shape)
            == (
                target_segment,
                80,
            ),

            "Historical sequence length mismatch.",
        )


        batch = sequence.unsqueeze(
            0
        )


        h0 = torch.zeros(
            (
                2,
                1,
                40,
            ),
            dtype=torch.float32,
        )


        output, hidden = self.gru(
            batch,
            h0,
        )


        require(
            torch.equal(
                output[
                    0,
                    -1,
                    :
                ],

                hidden[
                    -1,
                    0,
                    :
                ],
            ),

            "Final GRU output semantics changed.",
        )


        return self.output_activation(
            self.output_projection(
                output[
                    0,
                    -1,
                    :
                ]
            )
        )


# =============================================================================
# START
# =============================================================================

banner(
    "PHASE 4.3.3b — "
    "TREND RUNTIME INTEGRATION AUDIT"
)


print("\nENVIRONMENT")
print("-" * 120)

print(
    f"Python:  "
    f"{sys.version.splitlines()[0]}"
)

print(
    f"PyTorch: "
    f"{torch.__version__}"
)

print(
    "Device:  CPU"
)


torch.set_grad_enabled(
    False
)


# =============================================================================
# 1. CONTRACT INTEGRITY
# =============================================================================

banner(
    "FROZEN CONTRACT INTEGRITY"
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


with open(
    RUNTIME_CONTRACT_PATH,
    "r",
    encoding="utf-8",
) as f:

    runtime_contract = json.load(f)


with open(
    FORWARD_AUDIT_METADATA_PATH,
    "r",
    encoding="utf-8",
) as f:

    forward_metadata = json.load(f)


require(
    history_contract.get("status")
    == "FROZEN",

    "History contract not frozen.",
)


require(
    neural_contract.get("status")
    == "FROZEN",

    "Neural contract not frozen.",
)


require(
    runtime_contract.get("status")
    == "FROZEN",

    "Runtime contract not frozen.",
)


require(
    forward_metadata.get("status")
    == "COMPLETE",

    "Phase 4.3.2b forward audit "
    "not complete.",
)


print(
    "Trend-history contract: PASS"
)

print(
    "Trend-neural contract:  PASS"
)

print(
    "Trend-runtime contract: PASS"
)

print(
    "Previous forward audit: PASS"
)


# =============================================================================
# 2. LOAD PARQUET + RUNTIME ARRAYS
# =============================================================================

banner(
    "LOADING HISTORY REPRESENTATIONS"
)


membership = pd.read_parquet(
    MEMBERSHIP_PATH,
    columns=[
        "investor_node_index",
        "segment_number",
        "startup_node_index",
    ],
)


period_ptr = np.load(
    PERIOD_PTR_PATH,
    mmap_mode="r",
)


flat_startups = np.load(
    STARTUP_INDEX_PATH,
    mmap_mode="r",
)


period_counts = np.load(
    PERIOD_COUNT_PATH,
    mmap_mode="r",
)


require(
    len(membership)
    == EXPECTED_MEMBERSHIPS,

    "Parquet membership count changed.",
)


require(
    len(period_ptr)
    == N_SLOTS + 1,

    "Runtime pointer length changed.",
)


require(
    len(flat_startups)
    == EXPECTED_MEMBERSHIPS,

    "Runtime Startup population changed.",
)


require(
    len(period_counts)
    == N_SLOTS,

    "Runtime period-count length changed.",
)


print(
    f"Parquet memberships: "
    f"{len(membership):,}"
)

print(
    f"CSR pointers:        "
    f"{len(period_ptr):,}"
)

print(
    f"CSR Startup entries:"
    f" {len(flat_startups):,}"
)

print(
    f"CSR slot counts:     "
    f"{len(period_counts):,}"
)


# =============================================================================
# 3. CSR INTERNAL CONSISTENCY AFTER RELOAD
# =============================================================================

banner(
    "CSR INTERNAL CONSISTENCY"
)


pointer_counts = np.diff(
    period_ptr
)


pointer_counts_match = np.array_equal(
    pointer_counts,
    period_counts,
)


active_slots = int(
    np.count_nonzero(
        pointer_counts
    )
)


final_pointer_correct = (
    int(
        period_ptr[-1]
    )
    == EXPECTED_MEMBERSHIPS
)


print(
    f"ptr differences == stored counts: "
    f"{pointer_counts_match}"
)

print(
    f"Active slots:                    "
    f"{active_slots:,}"
)

print(
    f"Final pointer correct:           "
    f"{final_pointer_correct}"
)


require(
    pointer_counts_match,

    "CSR pointer/count arrays disagree.",
)


require(
    active_slots
    == EXPECTED_ACTIVE_SLOTS,

    "Runtime active-slot count changed.",
)


require(
    final_pointer_correct,

    "Final runtime pointer changed.",
)


# =============================================================================
# 4. GLOBAL PARQUET ↔ CSR EQUIVALENCE
#
# Reconstruct flattened slot index from the persisted CSR representation
# and compare exactly with frozen Parquet memberships.
# =============================================================================

banner(
    "GLOBAL PARQUET / CSR EQUIVALENCE"
)


parquet_investors = (
    membership[
        "investor_node_index"
    ]
    .to_numpy(
        dtype=np.int64
    )
)


parquet_periods = (
    membership[
        "segment_number"
    ]
    .to_numpy(
        dtype=np.int64
    )
)


parquet_startups = (
    membership[
        "startup_node_index"
    ]
    .to_numpy(
        dtype=np.int64
    )
)


parquet_slots = (
    parquet_investors
    * 60
    + parquet_periods
)


runtime_slots = np.repeat(
    np.arange(
        N_SLOTS,
        dtype=np.int64,
    ),
    pointer_counts,
)


slot_equivalence = np.array_equal(
    parquet_slots,
    runtime_slots,
)


startup_equivalence = np.array_equal(
    parquet_startups,
    flat_startups,
)


print(
    f"Investor-period slots exact: "
    f"{slot_equivalence}"
)

print(
    f"Startup membership exact:     "
    f"{startup_equivalence}"
)


require(
    slot_equivalence
    and startup_equivalence,

    "Persisted CSR history is not "
    "globally equivalent to Parquet history.",
)


# =============================================================================
# 5. LOOKUP HELPERS
# =============================================================================

def csr_startups(
    investor_node_index,
    segment_number,
):

    require(
        0
        <= investor_node_index
        < N_INVESTORS,

        "Investor index outside range.",
    )


    require(
        0
        <= segment_number
        < N_PERIODS,

        "Historical period outside T0-T59.",
    )


    slot = (
        investor_node_index
        * N_PERIODS
        + segment_number
    )


    start = int(
        period_ptr[
            slot
        ]
    )

    end = int(
        period_ptr[
            slot + 1
        ]
    )


    return np.asarray(
        flat_startups[
            start:end
        ],
        dtype=np.int64,
    )


# Only representative Investors need a Parquet lookup dictionary.

cases = forward_metadata[
    "representative_cases"
]


singleton_investor = int(
    cases[
        "singleton"
    ][
        "investor_node_index"
    ]
)


singleton_period = int(
    cases[
        "singleton"
    ][
        "segment_number"
    ]
)


multi_investor = int(
    cases[
        "multi_startup"
    ][
        "investor_node_index"
    ]
)


multi_period = int(
    cases[
        "multi_startup"
    ][
        "segment_number"
    ]
)


warm_investor = int(
    cases[
        "warm_t60"
    ][
        "investor_node_index"
    ]
)


cold_investor = int(
    cases[
        "cold_t60"
    ][
        "investor_node_index"
    ]
)


selected_investors = {
    singleton_investor,
    multi_investor,
    warm_investor,
    cold_investor,
}


selected_membership = (
    membership.loc[
        membership[
            "investor_node_index"
        ]
        .isin(
            selected_investors
        )
    ]
    .copy()
)


parquet_lookup = {}


for (
    investor,
    period,
), group in selected_membership.groupby(
    [
        "investor_node_index",
        "segment_number",
    ],
    sort=False,
):

    parquet_lookup[
        (
            int(investor),
            int(period),
        )
    ] = (
        group[
            "startup_node_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )


def parquet_startups(
    investor_node_index,
    segment_number,
):

    return parquet_lookup.get(
        (
            int(
                investor_node_index
            ),
            int(
                segment_number
            ),
        ),

        np.empty(
            0,
            dtype=np.int64,
        ),
    )


# =============================================================================
# 6. REPRESENTATIVE LOOKUP EQUIVALENCE
# =============================================================================

banner(
    "REPRESENTATIVE PERIOD LOOKUP EQUIVALENCE"
)


lookup_cases = [
    (
        "singleton",
        singleton_investor,
        singleton_period,
    ),

    (
        "multi_startup",
        multi_investor,
        multi_period,
    ),

    (
        "cold_T0",
        cold_investor,
        0,
    ),
]


lookup_records = []


for (
    name,
    investor,
    period,
) in lookup_cases:

    parquet_values = parquet_startups(
        investor,
        period,
    )

    csr_values = csr_startups(
        investor,
        period,
    )


    exact = np.array_equal(
        parquet_values,
        csr_values,
    )


    print()
    print(
        f"{name}:"
    )

    print(
        f"  Investor:     "
        f"{investor}"
    )

    print(
        f"  Period:       "
        f"T{period}"
    )

    print(
        f"  Parquet size: "
        f"{len(parquet_values)}"
    )

    print(
        f"  CSR size:     "
        f"{len(csr_values)}"
    )

    print(
        f"  Exact:        "
        f"{exact}"
    )


    require(
        exact,

        f"{name} runtime lookup mismatch.",
    )


    lookup_records.append(
        {
            "case":
                name,

            "investor_node_index":
                investor,

            "segment_number":
                period,

            "startup_count":
                len(
                    csr_values
                ),

            "exact":
                True,
        }
    )


# =============================================================================
# 7. MODEL INSTANTIATION
# =============================================================================

banner(
    "TREND MODEL INSTANTIATION"
)


model = TrendExtractor()


parameter_count = sum(
    parameter.numel()
    for parameter
    in model.parameters()
)


require(
    parameter_count == 32_480,

    "Trend parameter count changed.",
)


print(model)

print()
print(
    f"Trainable parameters: "
    f"{parameter_count:,}"
)


# =============================================================================
# 8. SAME AUDIT-ONLY INITIALIZATION AS PHASE 4.3.2b
# =============================================================================

banner(
    "AUDIT-ONLY DETERMINISTIC INITIALIZATION"
)


with torch.no_grad():

    model.attention_weight.copy_(
        torch.eye(
            80,
            dtype=torch.float32,
        )
        * 0.05
    )


    for name, parameter in (
        model.gru.named_parameters()
    ):

        if "weight_ih" in name:

            parameter.fill_(
                0.005
            )

        elif "weight_hh" in name:

            parameter.fill_(
                0.003
            )

        elif "bias" in name:

            parameter.zero_()


    model.output_projection.weight.copy_(
        torch.eye(
            40,
            dtype=torch.float32,
        )
    )


model.eval()


print(
    "Audit initialization installed."
)

print(
    "No model state will be saved."
)


# =============================================================================
# 9. GENERIC TEMPORAL SEQUENCE BUILDER
# =============================================================================

def build_sequence(
    investor_node_index,
    target_segment,
    lookup_function,
):

    require(
        0
        <= target_segment
        <= 60,

        "Target segment outside 0..60.",
    )


    if target_segment == 0:

        return torch.empty(
            (
                0,
                80,
            ),
            dtype=torch.float32,
        )


    query = investor_query(
        investor_node_index
    )


    period_vectors = []


    for period in range(
        target_segment
    ):

        startups = lookup_function(
            investor_node_index,
            period,
        )


        items = startup_items(
            startups
        )


        period_vector = model.attend_period(
            query,
            items,
        )


        period_vectors.append(
            period_vector
        )


    sequence = torch.stack(
        period_vectors,
        dim=0,
    )


    require(
        tuple(sequence.shape)
        == (
            target_segment,
            80,
        ),

        "Temporal sequence shape mismatch.",
    )


    return sequence


# =============================================================================
# 10. FIND A T0-ACTIVE REPRESENTATIVE FOR T1
# =============================================================================

t0_members = membership.loc[
    membership[
        "segment_number"
    ]
    .eq(0)
]


require(
    len(t0_members) > 0,

    "No T0 membership available.",
)


t1_investor = int(
    t0_members.iloc[0][
        "investor_node_index"
    ]
)


# Add its memberships to the Parquet lookup if needed.

if (
    t1_investor
    not in selected_investors
):

    t1_rows = membership.loc[
        membership[
            "investor_node_index"
        ]
        .eq(
            t1_investor
        )
    ]


    for (
        investor,
        period,
    ), group in t1_rows.groupby(
        [
            "investor_node_index",
            "segment_number",
        ],
        sort=False,
    ):

        parquet_lookup[
            (
                int(investor),
                int(period),
            )
        ] = (
            group[
                "startup_node_index"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )


# =============================================================================
# 11. SEQUENCE EQUIVALENCE CASES
# =============================================================================

banner(
    "PARQUET / CSR TEMPORAL SEQUENCE EQUIVALENCE"
)


sequence_cases = [
    (
        "T0",
        singleton_investor,
        0,
    ),

    (
        "T1",
        t1_investor,
        1,
    ),

    (
        "same_target_exclusion",
        multi_investor,
        multi_period,
    ),

    (
        "warm_T60",
        warm_investor,
        60,
    ),

    (
        "cold_T60",
        cold_investor,
        60,
    ),
]


sequence_audit_records = []


for (
    case_name,
    investor,
    target_segment,
) in sequence_cases:

    parquet_sequence = build_sequence(
        investor,
        target_segment,
        parquet_startups,
    )


    csr_sequence = build_sequence(
        investor,
        target_segment,
        csr_startups,
    )


    sequence_exact = torch.equal(
        parquet_sequence,
        csr_sequence,
    )


    parquet_trend = model.encode_sequence(
        parquet_sequence,
        target_segment,
    )


    csr_trend = model.encode_sequence(
        csr_sequence,
        target_segment,
    )


    trend_exact = torch.equal(
        parquet_trend,
        csr_trend,
    )


    print()
    print(
        f"{case_name}:"
    )

    print(
        f"  Investor:          "
        f"{investor}"
    )

    print(
        f"  Target segment:    "
        f"T{target_segment}"
    )

    print(
        f"  Sequence shape:    "
        f"{tuple(csr_sequence.shape)}"
    )

    print(
        f"  Sequence exact:    "
        f"{sequence_exact}"
    )

    print(
        f"  Trend shape:       "
        f"{tuple(csr_trend.shape)}"
    )

    print(
        f"  Trend exact:       "
        f"{trend_exact}"
    )


    require(
        sequence_exact,

        (
            f"{case_name}: Parquet and CSR "
            "temporal sequences differ."
        ),
    )


    require(
        trend_exact,

        (
            f"{case_name}: Parquet and CSR "
            "trend outputs differ."
        ),
    )


    sequence_audit_records.append(
        {
            "case":
                case_name,

            "investor_node_index":
                investor,

            "target_segment":
                target_segment,

            "sequence_rows":
                target_segment,

            "sequence_exact":
                True,

            "trend_exact":
                True,
        }
    )


# =============================================================================
# 12. COLD T60 INPUT SEMANTICS
# =============================================================================

banner(
    "COLD T60 RUNTIME SEMANTICS"
)


cold_runtime_sequence = build_sequence(
    cold_investor,
    60,
    csr_startups,
)


cold_exact_zero = torch.equal(
    cold_runtime_sequence,
    torch.zeros(
        (
            60,
            80,
        ),
        dtype=torch.float32,
    ),
)


print(
    f"Cold T60 Investor:    "
    f"{cold_investor}"
)

print(
    f"Runtime shape:         "
    f"{tuple(cold_runtime_sequence.shape)}"
)

print(
    f"Exact 60 x zero_80:   "
    f"{cold_exact_zero}"
)


require(
    cold_exact_zero,

    "CSR runtime violated frozen "
    "cold-Investor semantics.",
)


# =============================================================================
# 13. SAME-TARGET-PERIOD EXCLUSION THROUGH RUNTIME INDEX
# =============================================================================

banner(
    "RUNTIME TARGET-PERIOD EXCLUSION"
)


same_target_sequence = build_sequence(
    multi_investor,
    multi_period,
    csr_startups,
)


require(
    same_target_sequence.shape[0]
    == multi_period,

    "Target history has wrong length.",
)


# Last retrieved period must be target - 1.
last_retrieved_period = (
    multi_period - 1
    if multi_period > 0
    else None
)


print(
    f"Target segment:          "
    f"T{multi_period}"
)

print(
    f"Runtime sequence length: "
    f"{same_target_sequence.shape[0]}"
)

print(
    f"Last history period:     "
    f"T{last_retrieved_period}"
)

print(
    "Target-period slot queried: NO"
)

print(
    "Future-period slot queried: NO"
)


# =============================================================================
# 14. MINI-BATCH TREND REQUEST DEDUPLICATION
#
# Trend depends on:
#
#   (Investor, target_segment)
#
# NOT candidate Startup.
# =============================================================================

banner(
    "MINIBATCH TREND-REQUEST DEDUPLICATION"
)


batch_requests = [
    (
        warm_investor,
        60,
    ),

    (
        warm_investor,
        60,
    ),

    (
        cold_investor,
        60,
    ),

    (
        multi_investor,
        multi_period,
    ),

    (
        multi_investor,
        multi_period,
    ),
]


unique_requests = list(
    dict.fromkeys(
        batch_requests
    )
)


print(
    f"Raw pair-level requests:    "
    f"{len(batch_requests)}"
)

print(
    f"Unique trend requests:      "
    f"{len(unique_requests)}"
)

print(
    f"Redundant computations "
    f"avoided: "
    f"{len(batch_requests) - len(unique_requests)}"
)


require(
    len(unique_requests) == 3,

    "Unexpected deduplication behavior.",
)


# Compute only once per unique request.

trend_cache = {}


for key in unique_requests:

    investor, target_segment = key

    sequence = build_sequence(
        investor,
        target_segment,
        csr_startups,
    )

    trend_cache[
        key
    ] = model.encode_sequence(
        sequence,
        target_segment,
    )


expanded_trends = torch.stack(
    [
        trend_cache[
            key
        ]

        for key in batch_requests
    ],
    dim=0,
)


require(
    tuple(
        expanded_trends.shape
    )
    == (
        len(batch_requests),
        40,
    ),

    "Expanded batch trend shape mismatch.",
)


duplicate_warm_exact = torch.equal(
    expanded_trends[0],
    expanded_trends[1],
)


duplicate_multi_exact = torch.equal(
    expanded_trends[3],
    expanded_trends[4],
)


print(
    f"Duplicate warm request exact:  "
    f"{duplicate_warm_exact}"
)

print(
    f"Duplicate multi request exact: "
    f"{duplicate_multi_exact}"
)


require(
    duplicate_warm_exact
    and duplicate_multi_exact,

    "Batch trend reuse changed results.",
)


# =============================================================================
# 15. SAVE AUDIT TABLES
# =============================================================================

banner(
    "SAVING INTEGRATION AUDIT"
)


lookup_audit_df = pd.DataFrame(
    lookup_records
)


lookup_audit_path = (
    OUT_DIR
    / "trend_runtime_lookup_equivalence_audit.csv"
)


lookup_audit_df.to_csv(
    lookup_audit_path,
    index=False,
)


sequence_audit_df = pd.DataFrame(
    sequence_audit_records
)


sequence_audit_path = (
    OUT_DIR
    / "trend_runtime_sequence_equivalence_audit.csv"
)


sequence_audit_df.to_csv(
    sequence_audit_path,
    index=False,
)


# =============================================================================
# 16. METADATA
# =============================================================================

metadata = {

    "phase":
        "4.3.3b",

    "status":
        "COMPLETE",

    "component":
        "Trend runtime integration audit",

    "global_equivalence": {

        "parquet_csr_slot_exact":
            bool(
                slot_equivalence
            ),

        "parquet_csr_startup_exact":
            bool(
                startup_equivalence
            ),

        "membership_count":
            EXPECTED_MEMBERSHIPS,
    },

    "forward_equivalence": {

        "T0":
            True,

        "T1":
            True,

        "same_target_exclusion_case":
            True,

        "warm_T60":
            True,

        "cold_T60":
            True,
    },

    "cold_start": {

        "cold_T60_input_exact_zero":
            bool(
                cold_exact_zero
            ),

        "special_embedding":
            False,
    },

    "runtime_behavior": {

        "target_period_consumed":
            False,

        "future_period_consumed":
            False,

        "trend_feature_persisted":
            False,

        "trend_computed_from_current_parameters":
            True,
    },

    "batch_deduplication": {

        "key":
            [
                "investor_node_index",
                "target_segment",
            ],

        "candidate_startup_in_key":
            False,

        "audit_raw_requests":
            len(
                batch_requests
            ),

        "audit_unique_requests":
            len(
                unique_requests
            ),

        "result_equivalent":
            True,
    },

    "training_performed":
        False,

    "model_state_saved":
        False,

    "synthetic_features_saved":
        False,

    "final_kaiming_variant_frozen":
        False,

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

        "phase_4_3_3a":
            False,
    },
}


metadata_path = (
    OUT_DIR
    / "trend_runtime_integration_audit_metadata.json"
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
    "PHASE 4.3.3b FINAL SUMMARY"
)


print(
    "Global Parquet/CSR slots:        PASS"
)

print(
    "Global Startup memberships:      PASS"
)

print(
    "Persisted CSR integrity:         PASS"
)


print()
print(
    "Singleton period lookup:         PASS"
)

print(
    "Multi-startup period lookup:     PASS"
)

print(
    "Empty-period lookup:             PASS"
)


print()
print(
    "T0 sequence equivalence:         PASS"
)

print(
    "T1 sequence equivalence:         PASS"
)

print(
    "Same-target exclusion:           PASS"
)

print(
    "Warm T60 sequence equivalence:   PASS"
)

print(
    "Cold T60 sequence equivalence:   PASS"
)


print()
print(
    "Parquet/CSR trend outputs:        PASS"
)

print(
    "Cold T60 = exact 60 x zero_80:   PASS"
)

print(
    "Target/future period leakage:    NONE"
)


print()
print(
    "Batch deduplication key:"
)

print(
    "  (investor_node_index, target_segment)"
)

print(
    "Candidate Startup in trend key:  NO"
)

print(
    "Duplicate trend reuse:           PASS"
)


print()
print(
    "Training performed:              NO"
)

print(
    "Model state persisted:           NO"
)

print(
    "Learned trend features persisted:NO"
)

print(
    "Final Kaiming variant frozen:    NO"
)


print()
print("Outputs:")

for path in [
    lookup_audit_path,
    sequence_audit_path,
    metadata_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.3.3b STATUS: COMPLETE — "
    "TREND RUNTIME INTEGRATION VERIFIED"
)