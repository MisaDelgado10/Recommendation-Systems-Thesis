#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import gc
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy import sparse

import torch
import torch.nn as nn


# =============================================================================
# PHASE 8.2.7 — DEV_T57 FROZEN FIRST-BATCH
# REAL FORWARD / BCE / BACKWARD / EXACTLY-ONE-ADAM-STEP PREFLIGHT V1
# =============================================================================
#
# Scientific role:
#   implementation preflight ONLY.
#
# Allowed:
#   - instantiate canonical Phase-8 candidate models
#   - frozen DEV_T57 epoch-0 first batch
#   - real forward
#   - BCEWithLogitsLoss
#   - backward
#   - exactly ONE Adam step per family
#
# Forbidden:
#   - full epoch
#   - model selection
#   - ranking metrics
#   - T60
#   - April
#   - checkpoint persistence
#   - reuse of post-step state
#
# Updated model / optimizer objects are destroyed after each family.
# =============================================================================


REPO_ROOT = Path(__file__).resolve().parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(REPO_ROOT),
    )


import scripts.phase_8_2_3_candidate_model_modules_V1 as model_module
import scripts.phase_8_2_4_canonical_initialization_audit_V1 as init_module


# =============================================================================
# CONSTANTS
# =============================================================================

FAMILIES = [
    "B1a",
    "B1b",
    "M1",
    "M2",
]

SCOPE = "DEV_T57"
ORIGIN = "T57"
ORIGIN_SLUG = "t57"

NUM_INVESTORS = 165_975

BATCH_SIZE = 512

EXPECTED_POSITIVES = 98
EXPECTED_NEGATIVES = 414

ADAM_LR = 1e-3


ROOT = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_7_autograd_optimizer_preflight"
)

ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


INTERPRETATION = (
    ROOT
    / "phase_8_2_7_autograd_optimizer_interpretation_closure_V2.json"
)


STREAM_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_6_training_stream_dry_run/"
    "DEV_T57/epoch_0"
)

FIRST_BATCH = (
    STREAM_DIR
    / "epoch_0_first_batch_manifest.parquet"
)


FEATURE_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_1_feature_materialization/"
    "phase_8_2_1d_preprocessing_V2"
)


GRAPH_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_2_strict_inductive_heterogeneous_graph"
)


TREND_MEMBERSHIP = Path(
    "data/experimental/phase_4/"
    "trend_contract/"
    "trend_history_membership.parquet"
)


INVESTOR_DOC2VEC = Path(
    "data/experimental/phase_4/"
    "doc2vec/vectors/"
    "investor_doc2vec_vectors.npy"
)


INVESTOR_LABELS = Path(
    "data/experimental/phase_4/"
    "description_labels/"
    "description_label_multihot.npz"
)


OUTPUT_RESULT = (
    ROOT
    / "phase_8_2_7_dev_t57_frozen_batch_one_step_results_V1.json"
)

OUTPUT_HASHES = (
    ROOT
    / "phase_8_2_7_dev_t57_frozen_batch_one_step_sha256_V1.json"
)


# =============================================================================
# HELPERS
# =============================================================================

def require(condition, message):

    if not bool(condition):
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def tensor_bytes_hash(
    tensor: torch.Tensor,
) -> str:

    array = (
        tensor.detach()
        .cpu()
        .contiguous()
        .numpy()
    )

    return hashlib.sha256(
        array.tobytes()
    ).hexdigest()


def parameter_hashes(
    model: nn.Module,
):

    return {
        name:
            tensor_bytes_hash(
                parameter
            )
        for (
            name,
            parameter,
        )
        in model.named_parameters()
    }


def full_parameter_hash(
    model: nn.Module,
) -> str:

    digest = hashlib.sha256()

    for (
        name,
        parameter,
    ) in model.named_parameters():

        digest.update(
            name.encode(
                "utf-8"
            )
        )

        digest.update(
            str(
                tuple(
                    parameter.shape
                )
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            str(
                parameter.dtype
            ).encode(
                "utf-8"
            )
        )

        digest.update(
            parameter.detach()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        )

    return digest.hexdigest()


def require_same_hashes(
    before,
    after,
    label,
):

    require(
        before == after,
        (
            f"Parameter mutation occurred during "
            f"{label}."
        ),
    )


def scipy_rows_dense_tensor(
    matrix,
    row_indices,
):

    dense = (
        matrix[
            row_indices
        ]
        .toarray()
        .astype(
            np.float32,
            copy=False,
        )
    )

    return torch.from_numpy(
        dense
    )


def make_structured_tensor(
    *,
    row_indices,
    family,
    matrices,
    numeric_base,
    prior_count,
    base_dim,
    extended_dim,
):

    if family == "B1a":
        return None

    dim = (
        base_dim
        if family == "B1b"
        else extended_dim
    )

    out = np.empty(
        (
            len(row_indices),
            dim,
        ),
        dtype=np.float32,
    )

    cursor = 0

    ordered_base = [
        (
            "category_groups",
            matrices[
                "category_groups"
            ],
        ),
        (
            "country",
            matrices[
                "country"
            ],
        ),
        (
            "region_location_groups",
            matrices[
                "region_location_groups"
            ],
        ),
        (
            "company_type",
            matrices[
                "company_type"
            ],
        ),
    ]

    for (
        _name,
        matrix,
    ) in ordered_base:

        width = int(
            matrix.shape[1]
        )

        block = (
            matrix[
                row_indices
            ]
            .toarray()
            .astype(
                np.float32,
                copy=False,
            )
        )

        out[
            :,
            cursor:
            cursor + width
        ] = block

        cursor += width

        del block

    numeric = np.asarray(
        numeric_base[
            row_indices
        ],
        dtype=np.float32,
    )

    require(
        numeric.shape[1] == 2,
        "numeric_base width changed.",
    )

    out[
        :,
        cursor:
        cursor + 2
    ] = numeric

    cursor += 2

    del numeric

    if family in (
        "M1",
        "M2",
    ):

        prior = np.asarray(
            prior_count[
                row_indices
            ],
            dtype=np.float32,
        )

        require(
            prior.shape[1] == 1,
            "prior_count width changed.",
        )

        out[
            :,
            cursor:
            cursor + 1
        ] = prior

        cursor += 1

        del prior

        prior_sector = matrices[
            "prior_sector"
        ]

        width = int(
            prior_sector.shape[1]
        )

        block = (
            prior_sector[
                row_indices
            ]
            .toarray()
            .astype(
                np.float32,
                copy=False,
            )
        )

        out[
            :,
            cursor:
            cursor + width
        ] = block

        cursor += width

        del block

    require(
        cursor == dim,
        (
            f"{family}: structured width "
            f"{cursor} != {dim}"
        ),
    )

    return torch.from_numpy(
        out
    )


def positions_in_sorted(
    sorted_values,
    values,
    *,
    require_present=True,
    label="values",
):

    values = np.asarray(
        values,
        dtype=np.int64,
    )

    pos = np.searchsorted(
        sorted_values,
        values,
    )

    valid = (
        pos
        < len(
            sorted_values
        )
    )

    safe_pos = np.minimum(
        pos,
        max(
            len(
                sorted_values
            )
            - 1,
            0,
        ),
    )

    valid &= (
        sorted_values[
            safe_pos
        ]
        ==
        values
    )

    if require_present:

        require(
            bool(
                valid.all()
            ),
            (
                f"{label}: "
                f"{int((~valid).sum())} "
                "values missing from sorted registry."
            ),
        )

    return (
        pos,
        valid,
    )


# =============================================================================
# 1. PREREQUISITES
# =============================================================================

print("=" * 120)
print(
    "PHASE 8.2.7 — DEV_T57 FROZEN FIRST-BATCH "
    "ONE-STEP AUTOGRAD / ADAM PREFLIGHT V1"
)
print("=" * 120)


required_paths = [
    INTERPRETATION,
    FIRST_BATCH,
    TREND_MEMBERSHIP,
    INVESTOR_DOC2VEC,
    INVESTOR_LABELS,
]

for path in required_paths:

    require(
        path.exists(),
        f"Missing prerequisite: {path}",
    )


interpretation = json.loads(
    INTERPRETATION.read_text(
        encoding="utf-8"
    )
)

require(
    interpretation["status"]
    == "FROZEN_BEFORE_MODEL_EXECUTION",
    "8.2.7 interpretation is not frozen.",
)

require(
    interpretation[
        "training_allowed_now"
    ]
    is False,
    "Unexpected training authorization.",
)

print()
print("1. PREREQUISITE GATES")
print("  Phase 8.2.7 interpretation: PASS")
print("  Comparative training allowed: NO")


# =============================================================================
# 2. FROZEN FIRST BATCH
# =============================================================================

print()
print("2. FROZEN DEV_T57 EPOCH-0 FIRST BATCH")

batch = pd.read_parquet(
    FIRST_BATCH
).sort_values(
    "batch_position",
    kind="mergesort",
).reset_index(
    drop=True
)

require(
    len(batch) == BATCH_SIZE,
    "Frozen first batch no longer has 512 rows.",
)

require(
    batch[
        "batch_position"
    ].tolist()
    == list(
        range(
            BATCH_SIZE
        )
    ),
    "Frozen batch position ordering changed.",
)

positive_count = int(
    (
        batch[
            "label"
        ]
        == 1
    ).sum()
)

negative_count = int(
    (
        batch[
            "label"
        ]
        == 0
    ).sum()
)

require(
    positive_count
    == EXPECTED_POSITIVES,
    "Positive count changed.",
)

require(
    negative_count
    == EXPECTED_NEGATIVES,
    "Negative count changed.",
)

labels = torch.tensor(
    batch[
        "label"
    ].to_numpy(
        dtype=np.float32
    ),
    dtype=torch.float32,
).view(
    -1,
    1,
)

batch_investors = (
    batch[
        "investor_global"
    ]
    .to_numpy(
        dtype=np.int64
    )
)

batch_startups = (
    batch[
        "startup_global"
    ]
    .to_numpy(
        dtype=np.int64
    )
)

batch_segments = (
    batch[
        "segment_number"
    ]
    .to_numpy(
        dtype=np.int64
    )
)

require(
    int(
        batch[
            "trend_history_period_count"
        ].to_numpy(
            dtype=np.int64
        ).min()
    )
    >= 1,
    "Unexpected zero target segment.",
)

print(
    f"  Rows:              {len(batch):,}"
)

print(
    f"  Positives:         {positive_count:,}"
)

print(
    f"  Negatives:         {negative_count:,}"
)

print(
    f"  Investors:         "
    f"{np.unique(batch_investors).size:,}"
)

print(
    f"  Startups:          "
    f"{np.unique(batch_startups).size:,}"
)

print(
    f"  Segments:          "
    f"T{batch_segments.min()}..T{batch_segments.max()}"
)


# =============================================================================
# 3. STARTUP FEATURE REGISTRY + MATRICES
# =============================================================================

print()
print("3. T57 STARTUP FEATURE REGISTRY")

startup_rows_path = (
    FEATURE_DIR
    / "phase_8_2_1d_t57_startup_rows_V2.parquet"
)

startup_rows = pd.read_parquet(
    startup_rows_path,
    columns=[
        "startup_node_index",
        "fit_startup_features_visible",
    ],
)

startup_rows = (
    startup_rows
    .sort_values(
        "startup_node_index",
        kind="mergesort",
    )
    .reset_index(
        drop=True
    )
)

available_startup_globals = (
    startup_rows[
        "startup_node_index"
    ]
    .to_numpy(
        dtype=np.int64
    )
)

fit_visibility_all = (
    startup_rows[
        "fit_startup_features_visible"
    ]
    .to_numpy(
        dtype=bool
    )
)


text_svd32 = np.load(
    FEATURE_DIR
    / "phase_8_2_1d_t57_text_svd32_V2.npy",
    mmap_mode="r",
)

numeric_base = np.load(
    FEATURE_DIR
    / "phase_8_2_1d_t57_numeric_base_V2.npy",
    mmap_mode="r",
)

prior_count = np.load(
    FEATURE_DIR
    / "phase_8_2_1d_t57_prior_count_V2.npy",
    mmap_mode="r",
)


feature_matrices = {
    "categories":
        sparse.load_npz(
            FEATURE_DIR
            / "phase_8_2_1d_t57_categories_V2.npz"
        ).tocsr(),

    "category_groups":
        sparse.load_npz(
            FEATURE_DIR
            / "phase_8_2_1d_t57_category_groups_V2.npz"
        ).tocsr(),

    "country":
        sparse.load_npz(
            FEATURE_DIR
            / "phase_8_2_1d_t57_country_V2.npz"
        ).tocsr(),

    "region_location_groups":
        sparse.load_npz(
            FEATURE_DIR
            / "phase_8_2_1d_t57_region_location_groups_V2.npz"
        ).tocsr(),

    "company_type":
        sparse.load_npz(
            FEATURE_DIR
            / "phase_8_2_1d_t57_company_type_V2.npz"
        ).tocsr(),

    "prior_sector":
        sparse.load_npz(
            FEATURE_DIR
            / "phase_8_2_1d_t57_prior_sector_V2.npz"
        ).tocsr(),
}


# =============================================================================
# 4. INVESTOR INPUTS
# =============================================================================

print()
print("4. INVESTOR DESCRIPTION INPUTS")

investor_doc2vec_all = np.load(
    INVESTOR_DOC2VEC,
    mmap_mode="r",
)

investor_labels_all = (
    sparse.load_npz(
        INVESTOR_LABELS
    ).tocsr()
)

require(
    investor_doc2vec_all.shape
    == (
        NUM_INVESTORS,
        32,
    ),
    "Investor Doc2Vec shape changed.",
)

require(
    investor_labels_all.shape[1]
    == 802,
    "Investor label width changed.",
)

unique_investors = np.unique(
    batch_investors
)

investor_to_unique = {
    int(value):
        idx
    for (
        idx,
        value,
    )
    in enumerate(
        unique_investors
    )
}

batch_investor_unique_positions = np.asarray(
    [
        investor_to_unique[
            int(value)
        ]
        for value
        in batch_investors
    ],
    dtype=np.int64,
)

unique_investor_doc2vec = torch.from_numpy(
    np.asarray(
        investor_doc2vec_all[
            unique_investors
        ],
        dtype=np.float32,
    ).copy()
)

unique_investor_labels = (
    scipy_rows_dense_tensor(
        investor_labels_all,
        unique_investors,
    )
)

unique_investor_indices = torch.tensor(
    unique_investors,
    dtype=torch.long,
)

batch_investor_position_tensor = torch.tensor(
    batch_investor_unique_positions,
    dtype=torch.long,
)

print(
    f"  Unique batch Investors: "
    f"{len(unique_investors):,}"
)


# =============================================================================
# 5. TREND HISTORY CONTRACT FOR THIS BATCH
# =============================================================================

print()
print("5. STRICT PRE-TARGET TREND HISTORY")

membership = pd.read_parquet(
    TREND_MEMBERSHIP,
    columns=[
        "investor_node_index",
        "segment_number",
        "startup_node_index",
    ],
)

max_h_by_investor = (
    batch.groupby(
        "investor_global"
    )[
        "segment_number"
    ]
    .max()
    .astype(int)
    .to_dict()
)

membership = membership.loc[
    membership[
        "investor_node_index"
    ].isin(
        unique_investors
    )
].copy()

membership[
    "max_h"
] = (
    membership[
        "investor_node_index"
    ]
    .map(
        max_h_by_investor
    )
)

membership = membership.loc[
    membership[
        "segment_number"
    ]
    <
    membership[
        "max_h"
    ]
].copy()

history_startups = np.unique(
    membership[
        "startup_node_index"
    ].to_numpy(
        dtype=np.int64
    )
)

(
    _history_positions,
    history_present,
) = positions_in_sorted(
    available_startup_globals,
    history_startups,
    require_present=False,
    label="Trend history Startups",
)

require(
    bool(
        history_present.all()
    ),
    (
        "At least one strict pre-target history Startup "
        "is absent from the T57 origin-eligible feature registry."
    ),
)

history_groups_global = {}

for (
    investor,
    segment,
), frame in membership.groupby(
    [
        "investor_node_index",
        "segment_number",
    ],
    sort=False,
):

    history_groups_global[
        (
            int(investor),
            int(segment),
        )
    ] = (
        frame[
            "startup_node_index"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

print(
    f"  Relevant memberships: "
    f"{len(membership):,}"
)

print(
    f"  Unique history Startups: "
    f"{len(history_startups):,}"
)


# =============================================================================
# 6. EXACT TWO-HOP FIT GRAPH CLOSURE
# =============================================================================

print()
print("6. EXACT TWO-HOP T57 FIT-GRAPH CLOSURE")

edge_index = np.load(
    GRAPH_DIR
    / "phase_8_2_2_t57_V2_edge_index.npy",
    mmap_mode="r",
)

edge_type = np.load(
    GRAPH_DIR
    / "phase_8_2_2_t57_V2_edge_type.npy",
    mmap_mode="r",
)

fit_edge_mask = np.load(
    GRAPH_DIR
    / "phase_8_2_2_t57_V2_fit_edge_mask.npy",
    mmap_mode="r",
)

fit_mask = np.asarray(
    fit_edge_mask,
    dtype=bool,
)

fit_edges = np.asarray(
    edge_index[
        :,
        fit_mask,
    ],
    dtype=np.int64,
)

fit_types = np.asarray(
    edge_type[
        fit_mask
    ],
    dtype=np.int64,
)

batch_target_nodes = np.unique(
    np.concatenate(
        [
            batch_investors,
            batch_startups,
        ]
    )
)

src = fit_edges[0]
dst = fit_edges[1]

incoming_to_targets = np.isin(
    dst,
    batch_target_nodes,
)

level1_sources = np.unique(
    src[
        incoming_to_targets
    ]
)

layer1_destinations_needed = np.unique(
    np.concatenate(
        [
            batch_target_nodes,
            level1_sources,
        ]
    )
)

closure_edge_mask = np.isin(
    dst,
    layer1_destinations_needed,
)

closure_edges_global = fit_edges[
    :,
    closure_edge_mask,
]

closure_types_np = fit_types[
    closure_edge_mask
]

closure_nodes = np.unique(
    np.concatenate(
        [
            batch_target_nodes,
            closure_edges_global.reshape(
                -1
            ),
        ]
    )
)

closure_edges_local = np.searchsorted(
    closure_nodes,
    closure_edges_global,
).astype(
    np.int64
)

require(
    bool(
        np.array_equal(
            closure_nodes[
                closure_edges_local
            ],
            closure_edges_global,
        )
    ),
    "Graph local remapping failed.",
)

closure_edge_tensor = torch.tensor(
    closure_edges_local,
    dtype=torch.long,
)

closure_type_tensor = torch.tensor(
    closure_types_np,
    dtype=torch.long,
)

batch_investor_graph_local = torch.tensor(
    np.searchsorted(
        closure_nodes,
        batch_investors,
    ),
    dtype=torch.long,
)

batch_startup_graph_local = torch.tensor(
    np.searchsorted(
        closure_nodes,
        batch_startups,
    ),
    dtype=torch.long,
)

closure_startup_globals = closure_nodes[
    closure_nodes
    >= NUM_INVESTORS
]

(
    closure_startup_registry_pos,
    closure_startup_present,
) = positions_in_sorted(
    available_startup_globals,
    closure_startup_globals,
    require_present=False,
    label="Closure Startups",
)

eligible_closure_startups = (
    closure_startup_globals[
        closure_startup_present
    ]
)

print(
    f"  Closure nodes:      "
    f"{len(closure_nodes):,}"
)

print(
    f"  Closure edges:      "
    f"{closure_edges_global.shape[1]:,}"
)

print(
    f"  Relations present:  "
    f"{sorted(np.unique(closure_types_np).tolist())}"
)


# =============================================================================
# 7. UNION OF ALL STARTUP FEATURE ROWS REQUIRED BY THE ONE BATCH
# =============================================================================

print()
print("7. ONE-BATCH STARTUP FEATURE UNION")

needed_feature_globals = np.unique(
    np.concatenate(
        [
            batch_startups,
            history_startups,
            eligible_closure_startups,
        ]
    )
)

(
    needed_feature_rows,
    needed_present,
) = positions_in_sorted(
    available_startup_globals,
    needed_feature_globals,
    require_present=True,
    label="Required feature Startups",
)

require(
    bool(
        needed_present.all()
    ),
    "Required Startup feature row missing.",
)

feature_global_to_position = {
    int(value):
        idx
    for (
        idx,
        value,
    )
    in enumerate(
        needed_feature_globals
    )
}

feature_visibility_np = (
    fit_visibility_all[
        needed_feature_rows
    ]
)

feature_visibility = torch.tensor(
    feature_visibility_np,
    dtype=torch.bool,
)


batch_feature_positions = torch.tensor(
    [
        feature_global_to_position[
            int(value)
        ]
        for value
        in batch_startups
    ],
    dtype=torch.long,
)


history_groups_feature_positions = {
    key:
        torch.tensor(
            [
                feature_global_to_position[
                    int(value)
                ]
                for value
                in globals_
            ],
            dtype=torch.long,
        )
    for (
        key,
        globals_,
    )
    in history_groups_global.items()
}


closure_feature_global_positions = torch.tensor(
    [
        feature_global_to_position[
            int(value)
        ]
        for value
        in eligible_closure_startups
    ],
    dtype=torch.long,
)


closure_feature_local_positions = torch.tensor(
    np.searchsorted(
        closure_nodes,
        eligible_closure_startups,
    ),
    dtype=torch.long,
)


print(
    f"  Required unique Startups: "
    f"{len(needed_feature_globals):,}"
)

print(
    f"  Fit-visible:             "
    f"{int(feature_visibility_np.sum()):,}"
)

print(
    f"  Fit-hidden COLD_t:       "
    f"{int((~feature_visibility_np).sum()):,}"
)


# =============================================================================
# 8. RAW STARTUP FEATURE TENSORS SHARED AS INPUT DATA
# =============================================================================

print()
print("8. MATERIALIZE ONE-BATCH RAW INPUTS")

startup_text_tensor = torch.from_numpy(
    np.asarray(
        text_svd32[
            needed_feature_rows
        ],
        dtype=np.float32,
    ).copy()
)

startup_categories_tensor = (
    scipy_rows_dense_tensor(
        feature_matrices[
            "categories"
        ],
        needed_feature_rows,
    )
)

dimensions = (
    init_module
    .load_origin_dimensions(
        ORIGIN_SLUG
    )
)

require(
    dimensions[
        "category_dim"
    ]
    == startup_categories_tensor.shape[1],
    "Category dimension drift.",
)

require(
    dimensions[
        "base_dim"
    ]
    == 1896,
    "DEV_T57 base structured width drift.",
)

require(
    dimensions[
        "extended_dim"
    ]
    == 2744,
    "DEV_T57 extended structured width drift.",
)

print(
    f"  Text tensor:       "
    f"{tuple(startup_text_tensor.shape)}"
)

print(
    f"  Categories tensor: "
    f"{tuple(startup_categories_tensor.shape)}"
)


# =============================================================================
# 9. CANONICAL INITIALIZATION NAMESPACE
# =============================================================================

namespace_rows = (
    init_module
    .load_namespace_table()
)

namespace_lookup = (
    init_module
    .build_namespace_lookup(
        namespace_rows
    )
)


# =============================================================================
# 10. FAMILY EXECUTION
# =============================================================================

print()
print("=" * 120)
print("9. REAL ONE-BATCH FAMILY PREFLIGHTS")
print("=" * 120)


family_results = {}


for family in FAMILIES:

    print()
    print("-" * 120)
    print(f"{family}")
    print("-" * 120)

    # -------------------------------------------------------------------------
    # Fresh canonical instance.
    # -------------------------------------------------------------------------

    model = (
        init_module
        .instantiate_model(
            model_module,
            family,
            dimensions,
        )
    )

    init_module.initialize_model(
        model=model,
        family=family,
        scope=SCOPE,
        namespace_lookup=namespace_lookup,
        reverse_order=False,
    )

    model.train()

    initial_full_hash = (
        full_parameter_hash(
            model
        )
    )

    initial_hashes = (
        parameter_hashes(
            model
        )
    )

    # -------------------------------------------------------------------------
    # Family-specific structured raw input.
    # -------------------------------------------------------------------------

    structured_input = (
        make_structured_tensor(
            row_indices=(
                needed_feature_rows
            ),
            family=family,
            matrices=feature_matrices,
            numeric_base=numeric_base,
            prior_count=prior_count,
            base_dim=dimensions[
                "base_dim"
            ],
            extended_dim=dimensions[
                "extended_dim"
            ],
        )
    )

    # -------------------------------------------------------------------------
    # Encode all Startup representations needed by this one batch.
    # Fit-time visibility applied after biased projections.
    # -------------------------------------------------------------------------

    startup_content_all = (
        model
        .encode_startup_content(
            startup_text_tensor,
            startup_categories_tensor,
            visibility_mask=(
                feature_visibility
            ),
        )
    )

    startup_structured_all = (
        model
        .encode_startup_structured(
            startup_content_40=(
                startup_content_all
            ),
            structured_input=(
                structured_input
            ),
            visibility_mask=(
                feature_visibility
            ),
        )
    )

    require(
        startup_content_all.shape
        == (
            len(
                needed_feature_globals
            ),
            40,
        ),
        f"{family}: content shape mismatch.",
    )

    require(
        startup_structured_all.shape
        == (
            len(
                needed_feature_globals
            ),
            40,
        ),
        f"{family}: structured shape mismatch.",
    )

    # -------------------------------------------------------------------------
    # Investor common representations.
    # -------------------------------------------------------------------------

    investor_latent_unique = (
        model
        .encode_investor_latent(
            unique_investor_indices
        )
    )

    investor_description_unique = (
        model
        .encode_investor_description(
            unique_investor_doc2vec,
            unique_investor_labels,
        )
    )

    investor_latent_batch = (
        investor_latent_unique[
            batch_investor_position_tensor
        ]
    )

    investor_description_batch = (
        investor_description_unique[
            batch_investor_position_tensor
        ]
    )

    startup_content_batch = (
        startup_content_all[
            batch_feature_positions
        ]
    )

    startup_structured_batch = (
        startup_structured_all[
            batch_feature_positions
        ]
    )

    # -------------------------------------------------------------------------
    # Strict trend:
    # compute Investor-period vectors using only segments < focal h.
    # No post-h padding.
    # -------------------------------------------------------------------------

    if (
        model
        .family_spec
        .trend_active
    ):

        trend_query_unique = (
            model
            .compose_trend_query(
                investor_latent_unique,
                investor_description_unique,
            )
        )

        trend_by_inv_h = {}

        for (
            investor_unique_pos,
            investor_global,
        ) in enumerate(
            unique_investors
        ):

            investor_global = int(
                investor_global
            )

            investor_batch_rows = np.flatnonzero(
                batch_investors
                == investor_global
            )

            unique_h = sorted(
                set(
                    int(
                        batch_segments[
                            idx
                        ]
                    )
                    for idx
                    in investor_batch_rows
                )
            )

            max_h = max(
                unique_h
            )

            query = (
                trend_query_unique[
                    investor_unique_pos
                ]
            )

            period_vectors = []

            for period in range(
                max_h
            ):

                key = (
                    investor_global,
                    period,
                )

                item_positions = (
                    history_groups_feature_positions
                    .get(
                        key
                    )
                )

                if (
                    item_positions
                    is None
                    or item_positions.numel()
                    == 0
                ):

                    period_vector = (
                        query.new_zeros(
                            80
                        )
                    )

                else:

                    history_content = (
                        startup_content_all[
                            item_positions
                        ]
                    )

                    history_structured = (
                        startup_structured_all[
                            item_positions
                        ]
                    )

                    history_items = (
                        model
                        .compose_history_items(
                            history_structured,
                            history_content,
                        )
                    )

                    (
                        period_vector,
                        attention,
                    ) = (
                        model
                        .trend_extractor
                        .attend_period(
                            query,
                            history_items,
                        )
                    )

                    require(
                        bool(
                            torch.isfinite(
                                attention
                            ).all()
                        ),
                        (
                            f"{family}: non-finite "
                            "trend attention."
                        ),
                    )

                period_vectors.append(
                    period_vector
                )

            period_stack = torch.stack(
                period_vectors,
                dim=0,
            )

            for h in unique_h:

                sequence = (
                    period_stack[
                        :h
                    ]
                    .unsqueeze(
                        0
                    )
                )

                (
                    trend,
                    _gru_output,
                ) = (
                    model
                    .encode_trend_sequence(
                        sequence
                    )
                )

                require(
                    trend.shape
                    == (
                        1,
                        40,
                    ),
                    (
                        f"{family}: trend "
                        f"shape mismatch for "
                        f"Investor {investor_global}, h={h}."
                    ),
                )

                trend_by_inv_h[
                    (
                        investor_global,
                        h,
                    )
                ] = trend[
                    0
                ]

        investor_trend_raw_batch = torch.stack(
            [
                trend_by_inv_h[
                    (
                        int(
                            batch_investors[i]
                        ),
                        int(
                            batch_segments[i]
                        ),
                    )
                ]
                for i
                in range(
                    BATCH_SIZE
                )
            ],
            dim=0,
        )

        investor_trend_batch = (
            model
            .resolve_trend_slot(
                investor_latent_batch,
                investor_trend_raw_batch,
            )
        )

    else:

        investor_trend_batch = (
            model
            .resolve_trend_slot(
                investor_latent_batch,
                None,
            )
        )

    # -------------------------------------------------------------------------
    # Exact two-layer graph closure for M1/M2.
    # -------------------------------------------------------------------------

    if (
        model
        .family_spec
        .graph_active
    ):

        closure_input = torch.zeros(
            (
                len(
                    closure_nodes
                ),
                40,
            ),
            dtype=torch.float32,
        )

        closure_investor_mask = (
            closure_nodes
            < NUM_INVESTORS
        )

        closure_investor_globals = (
            closure_nodes[
                closure_investor_mask
            ]
        )

        closure_investor_local = torch.tensor(
            np.flatnonzero(
                closure_investor_mask
            ),
            dtype=torch.long,
        )

        closure_investor_global_tensor = torch.tensor(
            closure_investor_globals,
            dtype=torch.long,
        )

        closure_input[
            closure_investor_local
        ] = (
            model
            .investor_embedding(
                closure_investor_global_tensor
            )
        )

        closure_startup_content = (
            startup_content_all[
                closure_feature_global_positions
            ]
        )

        closure_startup_structured = (
            startup_structured_all[
                closure_feature_global_positions
            ]
        )

        closure_startup_visibility = (
            feature_visibility[
                closure_feature_global_positions
            ]
        )

        closure_startup_graph_input = (
            model
            .encode_startup_graph_input(
                closure_startup_content,
                closure_startup_structured,
                visibility_mask=(
                    closure_startup_visibility
                ),
            )
        )

        closure_input[
            closure_feature_local_positions
        ] = (
            closure_startup_graph_input
        )

        graph_output = (
            model
            .propagate_graph(
                closure_input,
                closure_edge_tensor,
                closure_type_tensor,
            )
        )

        h2 = graph_output[
            "h2"
        ]

        investor_structural_batch = (
            model
            .resolve_investor_structural_slot(
                investor_latent_batch,
                h2[
                    batch_investor_graph_local
                ],
            )
        )

        startup_structural_batch = (
            model
            .resolve_startup_structural_slot(
                startup_content_batch,
                h2[
                    batch_startup_graph_local
                ],
            )
        )

    else:

        investor_structural_batch = (
            model
            .resolve_investor_structural_slot(
                investor_latent_batch,
                None,
            )
        )

        startup_structural_batch = (
            model
            .resolve_startup_structural_slot(
                startup_content_batch,
                None,
            )
        )

    # -------------------------------------------------------------------------
    # Exact 7 x 40 pair representation.
    # -------------------------------------------------------------------------

    pair = (
        model
        .assemble_pair(
            investor_trend_40=(
                investor_trend_batch
            ),
            investor_latent_40=(
                investor_latent_batch
            ),
            investor_description_40=(
                investor_description_batch
            ),
            investor_structural_40=(
                investor_structural_batch
            ),
            startup_structured_40=(
                startup_structured_batch
            ),
            startup_content_40=(
                startup_content_batch
            ),
            startup_structural_40=(
                startup_structural_batch
            ),
        )
    )

    require(
        pair.shape
        == (
            BATCH_SIZE,
            280,
        ),
        f"{family}: pair width mismatch.",
    )

    require(
        bool(
            torch.isfinite(
                pair
            ).all()
        ),
        f"{family}: non-finite pair features.",
    )

    # -------------------------------------------------------------------------
    # Real forward.
    # -------------------------------------------------------------------------

    logits = (
        model
        .score_pair(
            pair
        )
    )

    require(
        logits.shape
        == (
            BATCH_SIZE,
            1,
        ),
        f"{family}: logit shape mismatch.",
    )

    require(
        bool(
            torch.isfinite(
                logits
            ).all()
        ),
        f"{family}: non-finite logits.",
    )

    after_forward_hashes = (
        parameter_hashes(
            model
        )
    )

    require_same_hashes(
        initial_hashes,
        after_forward_hashes,
        f"{family} forward",
    )

    # -------------------------------------------------------------------------
    # BCEWithLogitsLoss.
    # Numerical audit only; NOT a selection/performance metric.
    # -------------------------------------------------------------------------

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    loss = criterion(
        logits,
        labels,
    )

    require(
        bool(
            torch.isfinite(
                loss
            )
        ),
        f"{family}: non-finite BCE.",
    )

    # -------------------------------------------------------------------------
    # Adam.
    # -------------------------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ADAM_LR,
        betas=(
            0.9,
            0.999,
        ),
        eps=1e-8,
        weight_decay=0.0,
        amsgrad=False,
        foreach=False,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )

    require(
        len(
            optimizer.state
        )
        == 0,
        (
            f"{family}: Adam state exists "
            "before first step."
        ),
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    # -------------------------------------------------------------------------
    # Exactly one backward.
    # -------------------------------------------------------------------------

    loss.backward()

    after_backward_hashes = (
        parameter_hashes(
            model
        )
    )

    require_same_hashes(
        initial_hashes,
        after_backward_hashes,
        f"{family} backward",
    )

    gradient_rows = []

    component_grad_abs_sum = {}

    missing_gradients = []
    nonfinite_gradients = []

    for (
        name,
        parameter,
    ) in model.named_parameters():

        gradient = parameter.grad

        if gradient is None:

            missing_gradients.append(
                name
            )

            continue

        if not bool(
            torch.isfinite(
                gradient
            ).all()
        ):

            nonfinite_gradients.append(
                name
            )

            continue

        grad_abs_sum = float(
            gradient.detach()
            .abs()
            .sum()
            .item()
        )

        grad_l2 = float(
            gradient.detach()
            .norm()
            .item()
        )

        component = (
            init_module
            .resolve_semantic_component(
                model,
                name,
            )
        )

        component_grad_abs_sum[
            component
        ] = (
            component_grad_abs_sum
            .get(
                component,
                0.0,
            )
            + grad_abs_sum
        )

        gradient_rows.append(
            {
                "parameter":
                    name,

                "component":
                    component,

                "grad_abs_sum":
                    grad_abs_sum,

                "grad_l2":
                    grad_l2,
            }
        )

    require(
        len(
            missing_gradients
        )
        == 0,
        (
            f"{family}: parameters with "
            f"grad=None: {missing_gradients}"
        ),
    )

    require(
        len(
            nonfinite_gradients
        )
        == 0,
        (
            f"{family}: non-finite gradients: "
            f"{nonfinite_gradients}"
        ),
    )

    zero_components = [
        component
        for (
            component,
            value,
        )
        in component_grad_abs_sum.items()
        if not (
            np.isfinite(
                value
            )
            and value > 0.0
        )
    ]

    require(
        len(
            zero_components
        )
        == 0,
        (
            f"{family}: active semantic "
            f"components with zero aggregate "
            f"gradient: {zero_components}"
        ),
    )

    # -------------------------------------------------------------------------
    # Exactly ONE temporary optimizer step.
    # -------------------------------------------------------------------------

    optimizer.step()

    post_step_hashes = (
        parameter_hashes(
            model
        )
    )

    post_step_full_hash = (
        full_parameter_hash(
            model
        )
    )

    changed_parameters = [
        name
        for name
        in initial_hashes
        if (
            initial_hashes[
                name
            ]
            !=
            post_step_hashes[
                name
            ]
        )
    ]

    require(
        len(
            changed_parameters
        )
        > 0,
        (
            f"{family}: Adam step changed "
            "zero parameters."
        ),
    )

    require(
        post_step_full_hash
        != initial_full_hash,
        (
            f"{family}: full model hash "
            "unchanged after Adam step."
        ),
    )

    require(
        len(
            optimizer.state
        )
        > 0,
        (
            f"{family}: Adam state was not "
            "created after optimizer.step()."
        ),
    )

    for (
        parameter,
        state,
    ) in optimizer.state.items():

        require(
            "step" in state,
            f"{family}: Adam state missing step.",
        )

        require(
            "exp_avg" in state,
            f"{family}: Adam state missing exp_avg.",
        )

        require(
            "exp_avg_sq" in state,
            f"{family}: Adam state missing exp_avg_sq.",
        )

        require(
            bool(
                torch.isfinite(
                    state[
                        "exp_avg"
                    ]
                ).all()
            ),
            (
                f"{family}: non-finite Adam exp_avg."
            ),
        )

        require(
            bool(
                torch.isfinite(
                    state[
                        "exp_avg_sq"
                    ]
                ).all()
            ),
            (
                f"{family}: non-finite Adam exp_avg_sq."
            ),
        )

    # -------------------------------------------------------------------------
    # No post-step forward or scoring.
    # -------------------------------------------------------------------------

    family_results[
        family
    ] = {
        "family":
            family,

        "batch": {
            "rows":
                BATCH_SIZE,

            "positives":
                positive_count,

            "negatives":
                negative_count,

            "segment_min":
                int(
                    batch_segments.min()
                ),

            "segment_max":
                int(
                    batch_segments.max()
                ),
        },

        "architecture": {
            "trend_active":
                bool(
                    model
                    .family_spec
                    .trend_active
                ),

            "graph_active":
                bool(
                    model
                    .family_spec
                    .graph_active
                ),

            "structured_mode":
                str(
                    model
                    .family_spec
                    .structured_mode
                ),

            "parameter_tensors":
                len(
                    list(
                        model.parameters()
                    )
                ),

            "parameter_elements":
                int(
                    sum(
                        parameter.numel()
                        for parameter
                        in model.parameters()
                    )
                ),
        },

        "numerical_audit": {
            "loss_bce_with_logits":
                float(
                    loss.detach()
                    .item()
                ),

            "logits_finite":
                True,

            "loss_finite":
                True,

            "gradient_tensors":
                len(
                    gradient_rows
                ),

            "gradient_components":
                {
                    key:
                        float(value)
                    for (
                        key,
                        value,
                    )
                    in sorted(
                        component_grad_abs_sum.items()
                    )
                },

            "all_gradients_present":
                True,

            "all_gradients_finite":
                True,

            "all_active_components_nonzero_gradient":
                True,
        },

        "state_audit": {
            "initial_parameter_sha256":
                initial_full_hash,

            "forward_parameter_neutrality":
                True,

            "backward_parameter_neutrality":
                True,

            "post_step_parameter_sha256":
                post_step_full_hash,

            "post_step_hash_differs":
                True,

            "changed_parameter_tensors":
                len(
                    changed_parameters
                ),

            "optimizer_state_entries_after_step":
                len(
                    optimizer.state
                ),

            "optimizer_step_calls":
                1,
        },

        "post_step_policy": {
            "post_step_forward_executed":
                False,

            "post_step_scoring_executed":
                False,

            "checkpoint_written":
                False,

            "updated_state_reused":
                False,

            "updated_state_retained":
                False,
        },
    }

    print(
        f"  Canonical initialization:        PASS"
    )

    print(
        f"  Pair shape:                      "
        f"{tuple(pair.shape)}"
    )

    print(
        f"  Raw-logit forward:               PASS"
    )

    print(
        f"  BCEWithLogitsLoss finite:        PASS"
    )

    print(
        f"  Forward parameter neutrality:    PASS"
    )

    print(
        f"  Backward parameter neutrality:   PASS"
    )

    print(
        f"  Gradients present/finite:        PASS"
    )

    print(
        f"  Active components grad > 0:      PASS"
    )

    print(
        f"  Adam optimizer.step calls:       1"
    )

    print(
        f"  Changed parameter tensors:       "
        f"{len(changed_parameters):,}"
    )

    print(
        f"  Adam state created:              PASS"
    )

    print(
        f"  Post-step forward/scoring:       NO"
    )

    print(
        f"  Checkpoint written:              NO"
    )

    # -------------------------------------------------------------------------
    # Explicitly destroy updated state.
    # -------------------------------------------------------------------------

    del optimizer
    del criterion
    del loss
    del logits
    del pair
    del investor_trend_batch
    del investor_latent_batch
    del investor_description_batch
    del investor_latent_unique
    del investor_description_unique
    del startup_content_batch
    del startup_structured_batch
    del startup_content_all
    del startup_structured_all

    if structured_input is not None:
        del structured_input

    if "graph_output" in locals():
        del graph_output

    if "h2" in locals():
        del h2

    if "closure_input" in locals():
        del closure_input

    if "closure_startup_graph_input" in locals():
        del closure_startup_graph_input

    if "trend_query_unique" in locals():
        del trend_query_unique

    if "trend_by_inv_h" in locals():
        del trend_by_inv_h

    del model

    gc.collect()

    print(
        "  Updated model + optimizer state: DISCARDED"
    )


# =============================================================================
# 11. GLOBAL AUDIT
# =============================================================================

print()
print("=" * 120)
print("10. GLOBAL PHASE 8.2.7 EXECUTION AUDIT")
print("=" * 120)

require(
    set(
        family_results.keys()
    )
    == set(
        FAMILIES
    ),
    "Not all four families completed.",
)

for family in FAMILIES:

    result = family_results[
        family
    ]

    require(
        result[
            "state_audit"
        ][
            "optimizer_step_calls"
        ]
        == 1,
        (
            f"{family}: optimizer.step count "
            "is not exactly one."
        ),
    )

    require(
        result[
            "post_step_policy"
        ][
            "checkpoint_written"
        ]
        is False,
        (
            f"{family}: checkpoint unexpectedly written."
        ),
    )

    require(
        result[
            "post_step_policy"
        ][
            "post_step_forward_executed"
        ]
        is False,
        (
            f"{family}: post-step forward occurred."
        ),
    )

    require(
        result[
            "post_step_policy"
        ][
            "updated_state_retained"
        ]
        is False,
        (
            f"{family}: updated state retained."
        ),
    )


# =============================================================================
# 12. SAVE AUDIT RESULT — NEVER MODEL STATE
# =============================================================================

result_manifest = {
    "schema_version":
        "PHASE_8_2_7_DEV_T57_FROZEN_BATCH_ONE_STEP_PREFLIGHT_V1",

    "phase":
        "8.2.7",

    "status":
        "ONE_STEP_AUTOGRAD_OPTIMIZER_PREFLIGHT_EXECUTED_AND_AUDITED",

    "scientific_role":
        "TEMPORARY_IMPLEMENTATION_PREFLIGHT_ONLY",

    "scope":
        SCOPE,

    "origin":
        ORIGIN,

    "epoch_index":
        0,

    "batch":
        {
            "source":
                str(
                    FIRST_BATCH
                ),

            "rows":
                BATCH_SIZE,

            "positives":
                EXPECTED_POSITIVES,

            "negatives":
                EXPECTED_NEGATIVES,

            "same_batch_all_families":
                True,

            "negative_regeneration":
                False,

            "order_regeneration":
                False,
        },

    "real_runtime_inputs": {
        "unique_batch_investors":
            int(
                len(
                    unique_investors
                )
            ),

        "unique_batch_startups":
            int(
                np.unique(
                    batch_startups
                ).size
            ),

        "strict_history_memberships":
            int(
                len(
                    membership
                )
            ),

        "strict_history_unique_startups":
            int(
                len(
                    history_startups
                )
            ),

        "graph_closure_nodes":
            int(
                len(
                    closure_nodes
                )
            ),

        "graph_closure_edges":
            int(
                closure_edges_global.shape[
                    1
                ]
            ),

        "graph_relations_present":
            [
                int(value)
                for value
                in sorted(
                    np.unique(
                        closure_types_np
                    ).tolist()
                )
            ],

        "startup_feature_union_rows":
            int(
                len(
                    needed_feature_globals
                )
            ),
    },

    "optimizer": {
        "class":
            "torch.optim.Adam",

        "lr":
            ADAM_LR,

        "betas":
            [
                0.9,
                0.999,
            ],

        "eps":
            1e-8,

        "weight_decay":
            0.0,

        "amsgrad":
            False,

        "foreach":
            False,

        "fused":
            False,
    },

    "families":
        family_results,

    "execution_summary": {
        "models_instantiated":
            True,

        "canonical_initialization_applied":
            True,

        "real_forward_executed":
            True,

        "BCEWithLogitsLoss_computed":
            True,

        "backward_executed":
            True,

        "optimizer_created":
            True,

        "optimizer_step_executed":
            True,

        "optimizer_step_calls_per_family":
            1,

        "full_epoch_training_executed":
            False,

        "ranking_metrics_computed":
            False,

        "model_selection_performed":
            False,

        "post_step_forward_executed":
            False,

        "checkpoint_written":
            False,

        "updated_state_retained":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,
    },

    "training_allowed_now":
        False,

    "next_step":
        "Formal Phase 8.2.7 closure before Phase 8.2.8 integration gate.",
}


OUTPUT_RESULT.write_text(
    json.dumps(
        result_manifest,
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)


hash_targets = [
    INTERPRETATION,
    FIRST_BATCH,
    OUTPUT_RESULT,
]

OUTPUT_HASHES.write_text(
    json.dumps(
        {
            str(path):
                sha256_file(
                    path
                )
            for path
            in hash_targets
        },
        indent=2,
        sort_keys=True,
    ) + "\n",
    encoding="utf-8",
)


print()
print(
    "Families completed:                 4 / 4"
)

print(
    "Real frozen-batch forward:          PASS"
)

print(
    "BCEWithLogitsLoss:                  PASS"
)

print(
    "Backward:                           PASS"
)

print(
    "Gradient integrity:                 PASS"
)

print(
    "Forward/backward state neutrality:  PASS"
)

print(
    "Exactly one Adam step / family:     PASS"
)

print(
    "Post-step parameter change:         PASS"
)

print(
    "Adam state creation:                PASS"
)

print(
    "Post-step forward:                  NO"
)

print(
    "Checkpoint written:                 NO"
)

print(
    "Updated state retained:             NO"
)

print(
    "Performance inspected:              NO"
)

print(
    "T60 scored:                         NO"
)

print(
    "April analyzed:                     NO"
)

print(
    "Training allowed now:               NO"
)

print()
print(
    f"Results: {OUTPUT_RESULT}"
)

print(
    f"Hashes:  {OUTPUT_HASHES}"
)

print()
print("=" * 120)
print(
    "STATUS: ONE_STEP_AUTOGRAD_OPTIMIZER_PREFLIGHT_EXECUTED_AND_AUDITED"
)
print(
    "NEXT:   Formal Phase 8.2.7 closure"
)
print("=" * 120)
