from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# PHASE 4.3.2b — TREND ATTENTION + GRU FORWARD AUDIT
#
# PURPOSE
# -------
# Verify operational compatibility between:
#
#   1. frozen Phase-4.3.1 trend-history semantics
#   2. frozen Phase-4.3.2a attention/GRU neural contract
#
# IMPORTANT
# ---------
# NO training occurs.
#
# Latent embeddings and learned description features do not yet exist.
# Deterministic synthetic 40-D vectors are therefore used ONLY for the audit.
#
# No model state is persisted.
# No synthetic feature is persisted.
# No upstream contract is modified.
# =============================================================================


# =============================================================================
# INPUTS
# =============================================================================

MEMBERSHIP_PATH = Path(
    "data/experimental/phase_4/"
    "trend_contract/"
    "trend_history_membership.parquet"
)

ACTIVE_PERIODS_PATH = Path(
    "data/experimental/phase_4/"
    "trend_contract/"
    "trend_active_investor_periods.parquet"
)

INVESTOR_SUMMARY_PATH = Path(
    "data/experimental/phase_4/"
    "trend_contract/"
    "trend_investor_history_summary.parquet"
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

TEMPORAL_SPLIT_PATH = Path(
    "data/experimental/phase_2/"
    "model_ready/"
    "interactions_itrs_temporal_split.parquet"
)


# =============================================================================
# OUTPUTS
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_4/"
    "trend_module"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FROZEN DIMENSIONS
# =============================================================================

LATENT_DIM = 40
DESCRIPTION_DIM = 40

QUERY_DIM = 80
ITEM_DIM = 80

PERIOD_VECTOR_DIM = 80

GRU_HIDDEN_DIM = 40
GRU_NUM_LAYERS = 2

TREND_OUTPUT_DIM = 40


# =============================================================================
# FROZEN EXPECTATIONS
# =============================================================================

EXPECTED_MEMBERSHIPS = 1_145_364
EXPECTED_ACTIVE_PERIODS = 554_171

EXPECTED_INVESTORS = 165_975

EXPECTED_T60_INVESTORS = 11_884
EXPECTED_T60_COLD_INVESTORS = 3_060

EXPECTED_T60_COLD_EVENTS = 3_541


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
# DETERMINISTIC AUDIT-ONLY FEATURES
#
# These are NOT model features.
#
# They merely provide stable numerical 40-D vectors so the frozen neural
# architecture can be executed before learned embeddings/features exist.
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

    # Nonnegative, consistent with the ReLU description contract.
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
        == (1, QUERY_DIM),

        "Investor query dimension mismatch.",
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
                ITEM_DIM,
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
        result.shape[1]
        == ITEM_DIM,

        "Startup attention-item "
        "dimension mismatch.",
    )


    return result


# =============================================================================
# TREND MODULE
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


        # Audit instrumentation only.
        self.gru_forward_calls = 0


    # =========================================================================
    # Period-level attention
    # =========================================================================

    def attend_period(
        self,
        query,
        items,
    ):

        require(
            tuple(
                query.shape
            )
            == (QUERY_DIM,),

            "Unexpected attention query shape.",
        )


        require(
            items.ndim == 2
            and items.shape[1] == ITEM_DIM,

            "Unexpected attention item shape.",
        )


        # -------------------------------------------------------------
        # Empty period semantics frozen in Phase 4.3.1b.
        # -------------------------------------------------------------

        if items.shape[0] == 0:

            zero_period = torch.zeros(
                PERIOD_VECTOR_DIM,
                dtype=query.dtype,
                device=query.device,
            )

            return {
                "scores":
                    torch.empty(
                        0,
                        dtype=query.dtype,
                        device=query.device,
                    ),

                "alpha":
                    torch.empty(
                        0,
                        dtype=query.dtype,
                        device=query.device,
                    ),

                "period_vector":
                    zero_period,

                "attention_called":
                    False,
            }


        # -------------------------------------------------------------
        # Paper equation:
        #
        # score = u @ W @ V.T
        # -------------------------------------------------------------

        scores = (
            query
            @ self.attention_weight
            @ items.T
        )


        alpha = F.softmax(
            scores,
            dim=0,
        )


        period_vector = (
            alpha
            @ items
        )


        return {
            "scores":
                scores,

            "alpha":
                alpha,

            "period_vector":
                period_vector,

            "attention_called":
                True,
        }


    # =========================================================================
    # GRU sequence encoder
    # =========================================================================

    def encode_period_sequence(
        self,
        period_sequence,
        target_segment,
    ):

        require(
            0 <= target_segment <= 60,

            "target_segment must be 0..60.",
        )


        # -------------------------------------------------------------
        # Frozen T0 policy:
        #
        # no preceding sequence
        # trend = exact zero_40
        # -------------------------------------------------------------

        if target_segment == 0:

            require(
                period_sequence.shape
                == (0, PERIOD_VECTOR_DIM),

                "T0 must receive an empty "
                "historical sequence.",
            )

            return torch.zeros(
                TREND_OUTPUT_DIM,
                dtype=torch.float32,
            )


        require(
            tuple(
                period_sequence.shape
            )
            == (
                target_segment,
                PERIOD_VECTOR_DIM,
            ),

            (
                "Historical sequence length must "
                "equal target segment number."
            ),
        )


        batch_sequence = (
            period_sequence
            .unsqueeze(0)
        )


        h0 = torch.zeros(
            (
                GRU_NUM_LAYERS,
                1,
                GRU_HIDDEN_DIM,
            ),
            dtype=period_sequence.dtype,
            device=period_sequence.device,
        )


        self.gru_forward_calls += 1


        output, h_n = self.gru(
            batch_sequence,
            h0,
        )


        require(
            tuple(output.shape)
            == (
                1,
                target_segment,
                GRU_HIDDEN_DIM,
            ),

            "GRU output shape mismatch.",
        )


        require(
            tuple(h_n.shape)
            == (
                GRU_NUM_LAYERS,
                1,
                GRU_HIDDEN_DIM,
            ),

            "GRU hidden-state shape mismatch.",
        )


        # PyTorch top-layer final hidden state should correspond
        # to final output time step.

        final_output = output[
            0,
            -1,
            :
        ]


        final_hidden = h_n[
            -1,
            0,
            :
        ]


        require(
            torch.allclose(
                final_output,
                final_hidden,
                atol=0.0,
                rtol=0.0,
            ),

            (
                "Final top-layer GRU output "
                "does not equal h_n[-1]."
            ),
        )


        trend = self.output_activation(
            self.output_projection(
                final_output
            )
        )


        require(
            tuple(trend.shape)
            == (TREND_OUTPUT_DIM,),

            "Trend output shape mismatch.",
        )


        return trend


# =============================================================================
# 1. ENVIRONMENT
# =============================================================================

banner(
    "PHASE 4.3.2b — "
    "TREND ATTENTION + GRU FORWARD AUDIT"
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
# 2. LOAD AND VERIFY CONTRACTS
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


require(
    history_contract.get(
        "status"
    )
    == "FROZEN",

    "History contract is not frozen.",
)


require(
    neural_contract.get(
        "status"
    )
    == "FROZEN",

    "Trend neural contract is not frozen.",
)


require(
    history_contract[
        "dimensions"
    ][
        "period_attention_output"
    ]
    == PERIOD_VECTOR_DIM,

    "Frozen period-vector dim changed.",
)


require(
    neural_contract[
        "paper_specified"
    ][
        "gru_hidden_dim"
    ]
    == GRU_HIDDEN_DIM,

    "Frozen GRU hidden dimension changed.",
)


print(
    "Trend-history contract: PASS"
)

print(
    "Trend-neural contract:  PASS"
)


# =============================================================================
# 3. LOAD FROZEN HISTORY ARTIFACTS
# =============================================================================

banner(
    "LOADING FROZEN HISTORY ARTIFACTS"
)


membership = pd.read_parquet(
    MEMBERSHIP_PATH,
    columns=[
        "trend_membership_row",
        "investor_node_index",
        "investor_node_id",
        "investor_id",
        "segment_number",
        "startup_node_index",
        "startup_node_id",
        "startup_id",
    ],
)


active_periods = pd.read_parquet(
    ACTIVE_PERIODS_PATH
)


investor_summary = pd.read_parquet(
    INVESTOR_SUMMARY_PATH
)


require(
    len(membership)
    == EXPECTED_MEMBERSHIPS,

    "Membership count changed.",
)


require(
    len(active_periods)
    == EXPECTED_ACTIVE_PERIODS,

    "Active-period count changed.",
)


require(
    len(investor_summary)
    == EXPECTED_INVESTORS,

    "Investor-summary count changed.",
)


print(
    f"Membership rows:       "
    f"{len(membership):,}"
)

print(
    f"Active periods:        "
    f"{len(active_periods):,}"
)

print(
    f"Investor summary rows: "
    f"{len(investor_summary):,}"
)


# =============================================================================
# 4. Verify global deterministic membership ordering
# =============================================================================

banner(
    "MEMBERSHIP ORDERING CONTRACT"
)


expected_sorted = (
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


ordering_exact = np.array_equal(
    membership[
        "trend_membership_row"
    ]
    .to_numpy(
        dtype=np.int64
    ),

    expected_sorted[
        "trend_membership_row"
    ]
    .to_numpy(
        dtype=np.int64
    ),
)


print(
    f"Frozen membership ordering exact: "
    f"{ordering_exact}"
)


require(
    ordering_exact,

    "Frozen membership ordering changed.",
)


# =============================================================================
# 5. T60 Investor population
# =============================================================================

banner(
    "T60 INVESTOR SELECTION"
)


t60 = pd.read_parquet(
    TEMPORAL_SPLIT_PATH,
    columns=[
        "investor_id",
        "segment_number",
        "experiment_split",
    ],
)


t60 = (
    t60.loc[
        t60[
            "segment_number"
        ]
        .eq(60)
    ]
    .copy()
)


t60_investors = (
    t60[
        ["investor_id"]
    ]
    .drop_duplicates()
)


t60_investors[
    "investor_id"
] = (
    t60_investors[
        "investor_id"
    ]
    .astype(str)
)


investor_summary[
    "investor_id"
] = (
    investor_summary[
        "investor_id"
    ]
    .astype(str)
)


t60_investor_summary = (
    t60_investors
    .merge(
        investor_summary[
            [
                "investor_id",
                "investor_node_index",
                "active_period_count",
                "has_historical_investment",
            ]
        ],
        on="investor_id",
        how="left",
        validate="one_to_one",
    )
)


require(
    len(t60_investor_summary)
    == EXPECTED_T60_INVESTORS,

    "T60 Investor count changed.",
)


cold_t60 = (
    t60_investor_summary.loc[
        ~t60_investor_summary[
            "has_historical_investment"
        ]
    ]
)


warm_t60 = (
    t60_investor_summary.loc[
        t60_investor_summary[
            "has_historical_investment"
        ]
    ]
)


require(
    len(cold_t60)
    == EXPECTED_T60_COLD_INVESTORS,

    "T60 cold-Investor count changed.",
)


print(
    f"T60 Investors:       "
    f"{len(t60_investor_summary):,}"
)

print(
    f"T60 warm Investors:  "
    f"{len(warm_t60):,}"
)

print(
    f"T60 cold Investors:  "
    f"{len(cold_t60):,}"
)


# =============================================================================
# 6. SELECT REPRESENTATIVE REAL HISTORY CASES
# =============================================================================

banner(
    "REPRESENTATIVE HISTORY CASES"
)


# Singleton active period.
singleton_row = (
    active_periods.loc[
        active_periods[
            "unique_startups"
        ]
        .eq(1)
    ]
    .iloc[0]
)


# Multi-startup period away from T0 so it can also be used for
# same-target-period exclusion checks.
multi_row = (
    active_periods.loc[
        active_periods[
            "unique_startups"
        ]
        .ge(5)
        &
        active_periods[
            "segment_number"
        ]
        .gt(0)
    ]
    .iloc[0]
)


# Investor with active T0 for a T1 target.
t0_active_row = (
    active_periods.loc[
        active_periods[
            "segment_number"
        ]
        .eq(0)
    ]
    .iloc[0]
)


# Warm T60 Investor with the deepest available history.
warm_t60_row = (
    warm_t60
    .sort_values(
        [
            "active_period_count",
            "investor_node_index",
        ],
        ascending=[
            False,
            True,
        ],
    )
    .iloc[0]
)


# Deterministic cold T60 Investor.
cold_t60_row = (
    cold_t60
    .sort_values(
        "investor_node_index"
    )
    .iloc[0]
)


selected_investors = {
    int(
        singleton_row[
            "investor_node_index"
        ]
    ),

    int(
        multi_row[
            "investor_node_index"
        ]
    ),

    int(
        t0_active_row[
            "investor_node_index"
        ]
    ),

    int(
        warm_t60_row[
            "investor_node_index"
        ]
    ),

    int(
        cold_t60_row[
            "investor_node_index"
        ]
    ),
}


print(
    "Singleton period:"
)

print(
    f"  Investor node: "
    f"{int(singleton_row['investor_node_index'])}"
)

print(
    f"  Period:        "
    f"T{int(singleton_row['segment_number'])}"
)


print()
print(
    "Multi-startup period:"
)

print(
    f"  Investor node: "
    f"{int(multi_row['investor_node_index'])}"
)

print(
    f"  Period:        "
    f"T{int(multi_row['segment_number'])}"
)

print(
    f"  Startups:      "
    f"{int(multi_row['unique_startups'])}"
)


print()
print(
    "T1-history Investor:"
)

print(
    f"  Investor node: "
    f"{int(t0_active_row['investor_node_index'])}"
)


print()
print(
    "Warm T60 Investor:"
)

print(
    f"  Investor node: "
    f"{int(warm_t60_row['investor_node_index'])}"
)

print(
    f"  Active periods:"
    f" {int(warm_t60_row['active_period_count'])}"
)


print()
print(
    "Cold T60 Investor:"
)

print(
    f"  Investor node: "
    f"{int(cold_t60_row['investor_node_index'])}"
)


# =============================================================================
# 7. BUILD SMALL LOOKUP ONLY FOR SELECTED INVESTORS
#
# We intentionally avoid constructing a giant Python dictionary for all
# 554,171 Investor-period groups.
# =============================================================================

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


membership_lookup = {}


for (
    investor_node_index,
    segment_number,
), group in selected_membership.groupby(
    [
        "investor_node_index",
        "segment_number",
    ],
    sort=False,
):

    startups = (
        group[
            "startup_node_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )


    require(
        np.array_equal(
            startups,
            np.sort(
                startups
            ),
        ),

        (
            "Selected Investor-period startup "
            "ordering is not ascending."
        ),
    )


    membership_lookup[
        (
            int(
                investor_node_index
            ),
            int(
                segment_number
            ),
        )
    ] = startups


def period_startups(
    investor_node_index,
    segment_number,
):

    return membership_lookup.get(
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
# 8. INSTANTIATE MODULE
# =============================================================================

banner(
    "TREND MODULE INSTANTIATION"
)


model = TrendExtractor()


parameter_count = sum(
    parameter.numel()
    for parameter
    in model.parameters()
)


print(model)

print()
print(
    f"Trainable parameters: "
    f"{parameter_count:,}"
)


require(
    parameter_count
    == 32_480,

    "Trend parameter count changed.",
)


# =============================================================================
# 9. AUDIT-ONLY DETERMINISTIC INITIALIZATION
#
# These values are never persisted.
# They are NOT the final ITRS Kaiming initialization.
# =============================================================================

banner(
    "AUDIT-ONLY DETERMINISTIC INITIALIZATION"
)


with torch.no_grad():

    # Bilinear attention.
    model.attention_weight.copy_(
        torch.eye(
            80,
            dtype=torch.float32,
        )
        * 0.05
    )


    # Small deterministic GRU weights.
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


    # Simple deterministic W_o.
    model.output_projection.weight.copy_(
        torch.eye(
            40,
            dtype=torch.float32,
        )
    )


model.eval()


print(
    "Temporary audit weights installed."
)

print(
    "Final Kaiming initialization unchanged/unfrozen."
)

print(
    "State dict will NOT be saved."
)


# =============================================================================
# 10. SINGLETON ATTENTION AUDIT
# =============================================================================

banner(
    "SINGLETON ACTIVE-PERIOD ATTENTION"
)


singleton_investor = int(
    singleton_row[
        "investor_node_index"
    ]
)

singleton_period = int(
    singleton_row[
        "segment_number"
    ]
)


singleton_startups = period_startups(
    singleton_investor,
    singleton_period,
)


require(
    len(singleton_startups) == 1,

    "Selected singleton period does "
    "not contain exactly one startup.",
)


singleton_query = investor_query(
    singleton_investor
)


singleton_items = startup_items(
    singleton_startups
)


singleton_result = model.attend_period(
    singleton_query,
    singleton_items,
)


print(
    f"Startup count: "
    f"{len(singleton_startups)}"
)

print(
    f"Alpha: "
    f"{singleton_result['alpha'].tolist()}"
)


singleton_alpha_exact = torch.equal(
    singleton_result[
        "alpha"
    ],

    torch.ones(
        1,
        dtype=torch.float32,
    ),
)


singleton_output_exact = torch.equal(
    singleton_result[
        "period_vector"
    ],

    singleton_items[
        0
    ],
)


print(
    f"Singleton alpha exactly 1: "
    f"{singleton_alpha_exact}"
)

print(
    f"Period vector exactly item: "
    f"{singleton_output_exact}"
)


require(
    singleton_alpha_exact,

    "Singleton softmax did not equal 1.",
)


require(
    singleton_output_exact,

    "Singleton attention output "
    "does not equal its item.",
)


# =============================================================================
# 11. MULTI-STARTUP ATTENTION AUDIT
# =============================================================================

banner(
    "MULTI-STARTUP ATTENTION"
)


multi_investor = int(
    multi_row[
        "investor_node_index"
    ]
)

multi_period = int(
    multi_row[
        "segment_number"
    ]
)


multi_startups = period_startups(
    multi_investor,
    multi_period,
)


require(
    len(multi_startups) >= 5,

    "Selected multi-period contains "
    "fewer than five startups.",
)


multi_query = investor_query(
    multi_investor
)


multi_items = startup_items(
    multi_startups
)


multi_result = model.attend_period(
    multi_query,
    multi_items,
)


manual_scores = (
    multi_query
    @ model.attention_weight
    @ multi_items.T
)


manual_alpha = F.softmax(
    manual_scores,
    dim=0,
)


manual_period_vector = (
    manual_alpha
    @ multi_items
)


score_match = torch.allclose(
    multi_result[
        "scores"
    ],
    manual_scores,
    atol=0.0,
    rtol=0.0,
)


alpha_match = torch.allclose(
    multi_result[
        "alpha"
    ],
    manual_alpha,
    atol=0.0,
    rtol=0.0,
)


period_vector_match = torch.allclose(
    multi_result[
        "period_vector"
    ],
    manual_period_vector,
    atol=0.0,
    rtol=0.0,
)


alpha_sum = float(
    multi_result[
        "alpha"
    ].sum()
)


print(
    f"Startup count:          "
    f"{len(multi_startups):,}"
)

print(
    f"Attention score shape:  "
    f"{tuple(multi_result['scores'].shape)}"
)

print(
    f"Alpha shape:            "
    f"{tuple(multi_result['alpha'].shape)}"
)

print(
    f"Alpha sum:              "
    f"{alpha_sum:.9f}"
)

print(
    f"Manual score match:     "
    f"{score_match}"
)

print(
    f"Manual alpha match:     "
    f"{alpha_match}"
)

print(
    f"Manual output match:    "
    f"{period_vector_match}"
)


require(
    abs(
        alpha_sum
        - 1.0
    )
    < 1e-6,

    "Attention weights do not sum to 1.",
)


require(
    score_match
    and alpha_match
    and period_vector_match,

    "Bilinear attention does not match "
    "the frozen paper equation.",
)


require(
    torch.isfinite(
        multi_result[
            "period_vector"
        ]
    ).all(),

    "Multi-startup period vector "
    "contains non-finite values.",
)


# =============================================================================
# 12. EMPTY-PERIOD AUDIT
# =============================================================================

banner(
    "EMPTY-PERIOD BEHAVIOR"
)


# Find an actually absent period for the warm T60 Investor.
warm_investor = int(
    warm_t60_row[
        "investor_node_index"
    ]
)


warm_active_periods = {
    int(period)
    for period
    in selected_membership.loc[
        selected_membership[
            "investor_node_index"
        ]
        .eq(
            warm_investor
        ),
        "segment_number",
    ]
    .unique()
}


missing_periods = [
    period
    for period
    in range(60)
    if period not in warm_active_periods
]


if len(missing_periods) == 0:

    # The deepest-history Investor may theoretically be active in all
    # 60 periods. Use the singleton Investor to locate an empty slot.
    empty_test_investor = singleton_investor

    singleton_active_periods = {
        int(period)
        for period
        in selected_membership.loc[
            selected_membership[
                "investor_node_index"
            ]
            .eq(
                singleton_investor
            ),
            "segment_number",
        ]
        .unique()
    }

    missing_periods = [
        period
        for period
        in range(60)
        if period not in singleton_active_periods
    ]

else:

    empty_test_investor = warm_investor


require(
    len(missing_periods) > 0,

    "Could not locate representative "
    "empty historical period.",
)


empty_period = int(
    missing_periods[0]
)


empty_startups = period_startups(
    empty_test_investor,
    empty_period,
)


require(
    len(empty_startups) == 0,

    "Chosen empty period unexpectedly "
    "contains startups.",
)


empty_query = investor_query(
    empty_test_investor
)


empty_items = startup_items(
    empty_startups
)


empty_result = model.attend_period(
    empty_query,
    empty_items,
)


empty_exact_zero = torch.equal(
    empty_result[
        "period_vector"
    ],

    torch.zeros(
        PERIOD_VECTOR_DIM,
        dtype=torch.float32,
    ),
)


print(
    f"Investor node:      "
    f"{empty_test_investor}"
)

print(
    f"Empty period:       "
    f"T{empty_period}"
)

print(
    f"Attention called:   "
    f"{empty_result['attention_called']}"
)

print(
    f"Output exact zero:  "
    f"{empty_exact_zero}"
)


require(
    not empty_result[
        "attention_called"
    ],

    "Attention was called for an "
    "empty period.",
)


require(
    empty_exact_zero,

    "Empty period did not produce "
    "exact zero_80.",
)


# =============================================================================
# 13. PERIOD-SEQUENCE BUILDER
# =============================================================================

def build_sequence(
    investor_node_index,
    target_segment,
):

    require(
        0 <= target_segment <= 60,

        "target_segment outside 0..60.",
    )


    query = investor_query(
        investor_node_index
    )


    period_vectors = []

    diagnostics = []


    for period in range(
        target_segment
    ):

        startups = period_startups(
            investor_node_index,
            period,
        )


        items = startup_items(
            startups
        )


        result = model.attend_period(
            query,
            items,
        )


        period_vectors.append(
            result[
                "period_vector"
            ]
        )


        diagnostics.append(
            {
                "period":
                    period,

                "startup_count":
                    len(
                        startups
                    ),

                "attention_called":
                    result[
                        "attention_called"
                    ],
            }
        )


    if (
        target_segment == 0
    ):

        sequence = torch.empty(
            (
                0,
                PERIOD_VECTOR_DIM,
            ),
            dtype=torch.float32,
        )

    else:

        sequence = torch.stack(
            period_vectors,
            dim=0,
        )


    require(
        tuple(
            sequence.shape
        )
        == (
            target_segment,
            PERIOD_VECTOR_DIM,
        ),

        "Built temporal sequence has "
        "incorrect shape.",
    )


    return (
        sequence,
        diagnostics,
    )


# =============================================================================
# 14. T0 TARGET AUDIT
# =============================================================================

banner(
    "T0 TARGET BEHAVIOR"
)


t0_sequence, _ = build_sequence(
    singleton_investor,
    target_segment=0,
)


gru_calls_before_t0 = (
    model.gru_forward_calls
)


t0_trend = model.encode_period_sequence(
    t0_sequence,
    target_segment=0,
)


gru_calls_after_t0 = (
    model.gru_forward_calls
)


t0_exact_zero = torch.equal(
    t0_trend,
    torch.zeros(
        TREND_OUTPUT_DIM,
        dtype=torch.float32,
    ),
)


print(
    f"Sequence shape:       "
    f"{tuple(t0_sequence.shape)}"
)

print(
    f"GRU called:           "
    f"{gru_calls_after_t0 != gru_calls_before_t0}"
)

print(
    f"Trend shape:          "
    f"{tuple(t0_trend.shape)}"
)

print(
    f"Trend exact zero_40:  "
    f"{t0_exact_zero}"
)


require(
    gru_calls_after_t0
    == gru_calls_before_t0,

    "T0 incorrectly called the GRU.",
)


require(
    t0_exact_zero,

    "T0 trend is not exact zero_40.",
)


# =============================================================================
# 15. T1 TARGET AUDIT
# =============================================================================

banner(
    "T1 TARGET BEHAVIOR"
)


t1_investor = int(
    t0_active_row[
        "investor_node_index"
    ]
)


t1_sequence, t1_diag = (
    build_sequence(
        t1_investor,
        target_segment=1,
    )
)


require(
    len(t1_diag) == 1,

    "T1 should contain exactly "
    "one historical slot.",
)


require(
    t1_diag[0][
        "period"
    ] == 0,

    "T1 history is not exactly T0.",
)


t1_trend = model.encode_period_sequence(
    t1_sequence,
    target_segment=1,
)


print(
    f"Sequence shape: "
    f"{tuple(t1_sequence.shape)}"
)

print(
    f"Historical periods: "
    f"{[x['period'] for x in t1_diag]}"
)

print(
    f"T0 startup count: "
    f"{t1_diag[0]['startup_count']}"
)

print(
    f"Trend shape: "
    f"{tuple(t1_trend.shape)}"
)

print(
    f"Trend finite: "
    f"{bool(torch.isfinite(t1_trend).all())}"
)


require(
    torch.isfinite(
        t1_trend
    ).all(),

    "T1 trend contains "
    "non-finite values.",
)


# =============================================================================
# 16. SAME-TARGET-SEGMENT EXCLUSION
#
# The selected multi-startup activity happens in multi_period.
#
# For a target IN multi_period, its sequence must end at multi_period - 1.
# None of the current-period startups may be passed to attention for that
# target through the trend path.
# =============================================================================

banner(
    "SAME-TARGET-SEGMENT EXCLUSION"
)


same_period_sequence, same_period_diag = (
    build_sequence(
        multi_investor,
        target_segment=multi_period,
    )
)


history_periods = [
    record[
        "period"
    ]
    for record in same_period_diag
]


current_period_present = (
    multi_period
    in history_periods
)


future_period_present = any(
    period >= multi_period
    for period in history_periods
)


print(
    f"Target segment:          "
    f"T{multi_period}"
)

print(
    f"History length:          "
    f"{len(history_periods)}"
)

print(
    f"Last history segment:    "
    f"T{history_periods[-1] if history_periods else 'NONE'}"
)

print(
    f"Target segment included: "
    f"{current_period_present}"
)

print(
    f"Future segment included: "
    f"{future_period_present}"
)


require(
    not current_period_present,

    "Current target segment leaked "
    "into trend history.",
)


require(
    not future_period_present,

    "Future segment leaked into "
    "trend history.",
)


# =============================================================================
# 17. WARM T60 AUDIT
# =============================================================================

banner(
    "WARM T60 FORWARD AUDIT"
)


warm_sequence, warm_diag = (
    build_sequence(
        warm_investor,
        target_segment=60,
    )
)


warm_active_count = sum(
    record[
        "startup_count"
    ] > 0

    for record in warm_diag
)


warm_empty_count = (
    60
    - warm_active_count
)


expected_warm_active = int(
    warm_t60_row[
        "active_period_count"
    ]
)


require(
    warm_active_count
    == expected_warm_active,

    "Warm T60 active-period count "
    "does not match frozen summary.",
)


require(
    [
        record["period"]
        for record
        in warm_diag
    ]
    == list(
        range(60)
    ),

    "Warm T60 sequence is not "
    "exactly T0..T59.",
)


warm_trend = model.encode_period_sequence(
    warm_sequence,
    target_segment=60,
)


print(
    f"Sequence shape:       "
    f"{tuple(warm_sequence.shape)}"
)

print(
    f"Active periods:       "
    f"{warm_active_count}"
)

print(
    f"Empty periods:        "
    f"{warm_empty_count}"
)

print(
    f"First period:         "
    f"T{warm_diag[0]['period']}"
)

print(
    f"Last period:          "
    f"T{warm_diag[-1]['period']}"
)

print(
    f"Trend shape:          "
    f"{tuple(warm_trend.shape)}"
)

print(
    f"Trend finite:         "
    f"{bool(torch.isfinite(warm_trend).all())}"
)

print(
    f"Trend range:          "
    f"[{float(warm_trend.min()):.6f}, "
    f"{float(warm_trend.max()):.6f}]"
)


require(
    torch.isfinite(
        warm_trend
    ).all(),

    "Warm T60 trend is non-finite.",
)


require(
    torch.all(
        warm_trend >= 0
    )
    and torch.all(
        warm_trend <= 1
    ),

    "Sigmoid trend output is "
    "outside [0,1].",
)


# =============================================================================
# 18. COLD T60 AUDIT
# =============================================================================

banner(
    "COLD T60 FORWARD AUDIT"
)


cold_investor = int(
    cold_t60_row[
        "investor_node_index"
    ]
)


cold_sequence, cold_diag = (
    build_sequence(
        cold_investor,
        target_segment=60,
    )
)


cold_active_periods = sum(
    record[
        "startup_count"
    ] > 0

    for record
    in cold_diag
)


cold_sequence_exact_zero = torch.equal(
    cold_sequence,
    torch.zeros(
        (
            60,
            PERIOD_VECTOR_DIM,
        ),
        dtype=torch.float32,
    ),
)


cold_trend = model.encode_period_sequence(
    cold_sequence,
    target_segment=60,
)


print(
    f"Sequence shape:            "
    f"{tuple(cold_sequence.shape)}"
)

print(
    f"Active periods:            "
    f"{cold_active_periods}"
)

print(
    f"Sequence exact all-zero:   "
    f"{cold_sequence_exact_zero}"
)

print(
    f"Trend shape:               "
    f"{tuple(cold_trend.shape)}"
)

print(
    f"Trend finite:              "
    f"{bool(torch.isfinite(cold_trend).all())}"
)

print(
    f"Trend itself exact zero:   "
    f"{bool(torch.equal(cold_trend, torch.zeros_like(cold_trend)))}"
)


require(
    cold_active_periods == 0,

    "Cold T60 Investor unexpectedly "
    "contains historical activity.",
)


require(
    cold_sequence_exact_zero,

    "Cold T60 Investor does not "
    "receive exact 60 x zero_80.",
)


require(
    torch.isfinite(
        cold_trend
    ).all(),

    "Cold T60 trend is non-finite.",
)


# IMPORTANT:
# We DO NOT require cold_trend itself to be zero.
#
# The GRU and sigmoid are trainable transformations.
# The frozen policy concerns the INPUT sequence, not a forced downstream mask.


# =============================================================================
# 19. BATCH-FIRST / FINAL-HIDDEN AUDIT
# =============================================================================

banner(
    "BATCHED 60-STEP GRU AUDIT"
)


batch_60 = torch.stack(
    [
        warm_sequence,
        cold_sequence,
    ],
    dim=0,
)


h0 = torch.zeros(
    (
        GRU_NUM_LAYERS,
        2,
        GRU_HIDDEN_DIM,
    ),
    dtype=torch.float32,
)


batch_output, batch_hidden = (
    model.gru(
        batch_60,
        h0,
    )
)


print(
    f"Input shape:  "
    f"{tuple(batch_60.shape)}"
)

print(
    f"Output shape: "
    f"{tuple(batch_output.shape)}"
)

print(
    f"Hidden shape: "
    f"{tuple(batch_hidden.shape)}"
)


require(
    tuple(batch_60.shape)
    == (
        2,
        60,
        80,
    ),

    "Batch-first input shape mismatch.",
)


require(
    tuple(batch_output.shape)
    == (
        2,
        60,
        40,
    ),

    "Batch GRU output shape mismatch.",
)


require(
    tuple(batch_hidden.shape)
    == (
        2,
        2,
        40,
    ),

    "Batch GRU hidden shape mismatch.",
)


final_output_hidden_match = torch.allclose(
    batch_output[
        :,
        -1,
        :
    ],

    batch_hidden[
        -1,
        :,
        :
    ],

    atol=0.0,
    rtol=0.0,
)


print(
    f"Final output == "
    f"top-layer h_n: "
    f"{final_output_hidden_match}"
)


require(
    final_output_hidden_match,

    "Final GRU output / h_n "
    "relationship changed.",
)


# =============================================================================
# 20. FINAL OUTPUT PROJECTION AUDIT
# =============================================================================

banner(
    "TREND OUTPUT PROJECTION"
)


batch_trend = (
    model.output_activation(
        model.output_projection(
            batch_output[
                :,
                -1,
                :
            ]
        )
    )
)


print(
    f"Input hidden shape: "
    f"{tuple(batch_output[:, -1, :].shape)}"
)

print(
    f"Trend shape:        "
    f"{tuple(batch_trend.shape)}"
)

print(
    f"Finite:             "
    f"{bool(torch.isfinite(batch_trend).all())}"
)

print(
    f"Within [0,1]:       "
    f"{bool(torch.all((batch_trend >= 0) & (batch_trend <= 1)))}"
)


require(
    tuple(batch_trend.shape)
    == (
        2,
        40,
    ),

    "Batched trend output shape mismatch.",
)


require(
    torch.isfinite(
        batch_trend
    ).all(),

    "Batched trend output contains "
    "non-finite values.",
)


# =============================================================================
# 21. RECORD AUDIT RESULTS
# =============================================================================

audit_records = [
    {
        "audit":
            "singleton_attention",

        "status":
            "PASS",

        "detail":
            (
                "alpha exactly 1; period vector "
                "exactly equals sole startup item"
            ),
    },

    {
        "audit":
            "multi_startup_attention",

        "status":
            "PASS",

        "detail":
            (
                "scores, softmax weights and "
                "weighted aggregation exactly match "
                "bilinear attention implementation"
            ),
    },

    {
        "audit":
            "empty_period",

        "status":
            "PASS",

        "detail":
            (
                "attention bypassed; exact zero_80"
            ),
    },

    {
        "audit":
            "t0_target",

        "status":
            "PASS",

        "detail":
            (
                "GRU bypassed; exact zero_40"
            ),
    },

    {
        "audit":
            "t1_target",

        "status":
            "PASS",

        "detail":
            (
                "exact one-slot T0 sequence"
            ),
    },

    {
        "audit":
            "same_target_segment_exclusion",

        "status":
            "PASS",

        "detail":
            (
                "history includes only periods "
                "strictly before target segment"
            ),
    },

    {
        "audit":
            "warm_t60",

        "status":
            "PASS",

        "detail":
            (
                "exact 60-slot T0-T59 history"
            ),
    },

    {
        "audit":
            "cold_t60",

        "status":
            "PASS",

        "detail":
            (
                "exact 60 x zero_80 input sequence"
            ),
    },

    {
        "audit":
            "batch_first_gru",

        "status":
            "PASS",

        "detail":
            (
                "[2,60,80] -> [2,60,40], "
                "h_n [2,2,40]"
            ),
    },

    {
        "audit":
            "trend_projection",

        "status":
            "PASS",

        "detail":
            (
                "Linear(40,40,bias=False)+Sigmoid "
                "produces finite [B,40]"
            ),
    },
]


audit_df = pd.DataFrame(
    audit_records
)


audit_path = (
    OUT_DIR
    / "trend_forward_audit.csv"
)


audit_df.to_csv(
    audit_path,
    index=False,
)


# =============================================================================
# 22. METADATA
# =============================================================================

metadata = {
    "phase":
        "4.3.2b",

    "status":
        "COMPLETE",

    "component":
        "ITRS trend forward implementation audit",

    "training_performed":
        False,

    "model_state_saved":
        False,

    "audit_features_saved":
        False,

    "audit_initialization": {
        "attention":
            "0.05 * identity",

        "gru_input_weights":
            0.005,

        "gru_hidden_weights":
            0.003,

        "gru_biases":
            0.0,

        "output_projection":
            "identity",

        "purpose":
            (
                "deterministic forward-contract "
                "audit only"
            ),

        "final_kaiming_variant_frozen":
            False,
    },

    "verified_semantics": {

        "singleton_attention":
            True,

        "multi_startup_attention":
            True,

        "empty_period_zero_80":
            True,

        "t0_zero_40":
            True,

        "same_target_segment_excluded":
            True,

        "t1_history_exactly_t0":
            True,

        "t60_history_exactly_t0_t59":
            True,

        "cold_t60_input_all_zero":
            True,

        "batch_first":
            True,

        "final_hidden_equivalence":
            True,
    },

    "representative_cases": {

        "singleton": {
            "investor_node_index":
                singleton_investor,

            "segment_number":
                singleton_period,

            "startup_count":
                1,
        },

        "multi_startup": {
            "investor_node_index":
                multi_investor,

            "segment_number":
                multi_period,

            "startup_count":
                int(
                    len(
                        multi_startups
                    )
                ),
        },

        "warm_t60": {
            "investor_node_index":
                warm_investor,

            "active_periods":
                warm_active_count,

            "empty_periods":
                warm_empty_count,
        },

        "cold_t60": {
            "investor_node_index":
                cold_investor,

            "active_periods":
                cold_active_periods,
        },
    },

    "important_interpretation": {
        "cold_investor_input":
            (
                "The frozen no-history policy forces "
                "the period input sequence to zero; "
                "it does not force the trainable GRU "
                "trend output itself to zero."
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

        "phase_4_3_2a":
            False,
    },
}


metadata_path = (
    OUT_DIR
    / "trend_forward_audit_metadata.json"
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
    "PHASE 4.3.2b FINAL SUMMARY"
)


print(
    "Singleton-period attention:     PASS"
)

print(
    "Multi-startup attention:        PASS"
)

print(
    "Attention softmax normalization:PASS"
)

print(
    "Empty-period zero_80:           PASS"
)

print(
    "T0 trend zero_40:               PASS"
)

print(
    "T1 history = T0 only:           PASS"
)

print(
    "Same-target segment exclusion:  PASS"
)

print(
    "Warm T60 = T0-T59:              PASS"
)

print(
    "Cold T60 = 60 x zero_80:        PASS"
)

print(
    "2-layer GRU forward shapes:     PASS"
)

print(
    "Final hidden-state semantics:   PASS"
)

print(
    "Trend projection + sigmoid:     PASS"
)


print()
print(
    f"Trend neural parameters:         "
    f"{parameter_count:,}"
)


print()
print(
    "Training performed:             NO"
)

print(
    "Audit model saved:              NO"
)

print(
    "Synthetic audit features saved: NO"
)

print(
    "Final Kaiming variant frozen:   NO"
)


print()
print("Outputs:")

for path in [
    audit_path,
    metadata_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.3.2b STATUS: COMPLETE — "
    "TREND FORWARD CONTRACT VERIFIED"
)