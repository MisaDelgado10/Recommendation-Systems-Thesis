"""
Phase 8.2.3 — Candidate Model Modules V1
========================================

Scientific role
---------------
Implement the frozen Phase-8 candidate-model topology for:

    B1a
    B1b
    M1
    M2

This file defines modules only.

It does NOT:
    - load evaluation candidate matrices,
    - instantiate a candidate model,
    - initialize parameters under the Phase-8 initialization protocol,
    - create an optimizer,
    - train,
    - score T60,
    - analyze April,
    - inspect performance.

Authoritative semantics
-----------------------
Pair representation:

    [
        investor_trend_40,
        investor_latent_40,
        investor_description_40,
        investor_structural_40,
        startup_structured_40,
        startup_content_40,
        startup_structural_40,
    ]

Total:
    7 * 40 = 280 dimensions.

Candidate-family activation:

    B1a
        trend                  ITRS_TREND_40
        investor_structural    ZERO_40
        startup_structured     ZERO_40
        startup_content        INDUCTIVE_CONTENT_40
        startup_structural     ZERO_40
        graph                  inactive

    B1b
        trend                  ITRS_TREND_40
        investor_structural    ZERO_40
        startup_structured     INDUCTIVE_BASE_STRUCTURED_40
        startup_content        INDUCTIVE_CONTENT_40
        startup_structural     ZERO_40
        graph                  inactive

    M1
        trend                  ZERO_40
        investor_structural    HETEROGENEOUS_RGCN_40
        startup_structured     INDUCTIVE_EXTENDED_STRUCTURED_40
        startup_content        INDUCTIVE_CONTENT_40
        startup_structural     HETEROGENEOUS_RGCN_40
        graph                  active

    M2
        trend                  ITRS_TREND_40
        investor_structural    HETEROGENEOUS_RGCN_40
        startup_structured     INDUCTIVE_EXTENDED_STRUCTURED_40
        startup_content        INDUCTIVE_CONTENT_40
        startup_structural     HETEROGENEOUS_RGCN_40
        graph                  active

Cold-start rule
---------------
During parameter fitting, every COLD_t Startup must have its allowed
Startup-side neural feature slots zeroed after trainable projections.
This is necessary because biased shared Linear layers would otherwise
produce nonzero vectors from all-zero raw inputs.

For graph-active models, the resulting Startup graph-input vector is also
zeroed after the 80->40 projection. Structural graph edges incident to
COLD_t are separately masked by the authoritative Phase-8.2.2 graph
artifacts/runtime.

At inference, the already-fitted encoders may process the restored allowed
origin-time Startup features. The graph runtime then uses the frozen
sink-only inference graph:
    incoming to COLD_t from non-cold source: allowed
    outgoing from COLD_t: suppressed

No Startup-ID embedding exists in this module.
No Person-ID embedding exists in this module.
No Phase-4 Startup Doc2Vec vector is used by candidate Startup encoders.

Trend consistency closure
-------------------------
Phase-4 query:
    [Investor latent 40 || Investor description 40]

Phase-4 historical item:
    [Startup latent 40 || Startup description 40]

Phase-8 inductive replacement:
    [Startup structured inductive 40 || Startup content inductive 40]

Therefore:
    B1a trend item = [ZERO_40 || content_40]
    B1b trend item = [base_structured_40 || content_40]
    M1  trend      = ZERO_40
    M2 trend item  = [extended_structured_40 || content_40]

Startup graph output is NOT part of the trend item.
There is NO separate history encoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# FROZEN DIMENSIONS
# =============================================================================

NUM_INVESTORS = 165_975

LATENT_DIM = 40

INVESTOR_DESCRIPTION_TEXT_INPUT_DIM = 32
INVESTOR_DESCRIPTION_TEXT_DIM = 20
INVESTOR_DESCRIPTION_LABEL_DIM = 20
INVESTOR_DESCRIPTION_DIM = 40

STARTUP_TEXT_INPUT_DIM = 32
STARTUP_TEXT_PROJECTED_DIM = 20
STARTUP_CATEGORY_PROJECTED_DIM = 20
STARTUP_CONTENT_DIM = 40

STRUCTURED_HIDDEN_DIM = 64
STARTUP_STRUCTURED_DIM = 40

GRAPH_INPUT_DIM = 40
GRAPH_HIDDEN_DIM = 40
GRAPH_OUTPUT_DIM = 40

NUM_TYPED_RELATIONS = 10
NUM_RGCN_BASES = 5
NUM_RGCN_LAYERS = 2

TREND_QUERY_DIM = 80
TREND_ITEM_DIM = 80
TREND_HIDDEN_DIM = 40
TREND_DIM = 40

INVESTOR_SCORING_DIM = 160
STARTUP_SCORING_DIM = 120
PAIR_DIM = 280

SCORER_HIDDEN_WIDTHS = (
    128,
    64,
    32,
    16,
)

SCORER_OUTPUT_DIM = 1


PAIR_COMPONENT_ORDER = (
    "investor_trend",
    "investor_latent",
    "investor_description",
    "investor_structural",
    "startup_structured",
    "startup_content",
    "startup_structural",
)


TYPED_RELATION_VOCABULARY = (
    "startup|founded_by|person",
    "person|founded|startup",
    "investor|acquired|investor",
    "investor|acquired|startup",
    "startup|acquired|investor",
    "startup|acquired|startup",
    "investor|acquired_by|investor",
    "investor|acquired_by|startup",
    "startup|acquired_by|investor",
    "startup|acquired_by|startup",
)


# =============================================================================
# BASIC VALIDATION HELPERS
# =============================================================================

def require(
    condition: bool,
    message: str,
) -> None:

    if not condition:
        raise ValueError(message)


def require_rank_2(
    x: Tensor,
    name: str,
) -> None:

    require(
        x.ndim == 2,
        f"{name} must be rank-2; got shape {tuple(x.shape)}.",
    )


def require_width(
    x: Tensor,
    width: int,
    name: str,
) -> None:

    require_rank_2(
        x,
        name,
    )

    require(
        x.shape[1] == width,
        (
            f"{name} width mismatch: "
            f"expected {width}, got {x.shape[1]}."
        ),
    )


def require_same_rows(
    *named_tensors: Tuple[str, Tensor],
) -> None:

    require(
        len(named_tensors) > 0,
        "require_same_rows requires at least one tensor.",
    )

    expected = named_tensors[0][1].shape[0]

    for name, tensor in named_tensors:

        require_rank_2(
            tensor,
            name,
        )

        require(
            tensor.shape[0] == expected,
            (
                f"{name} row count mismatch: "
                f"expected {expected}, got {tensor.shape[0]}."
            ),
        )


def zero_40_like(
    reference: Tensor,
) -> Tensor:

    require_rank_2(
        reference,
        "reference",
    )

    return reference.new_zeros(
        (
            reference.shape[0],
            LATENT_DIM,
        )
    )


def normalize_visibility_mask(
    visibility_mask: Tensor,
    row_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """
    Convert a Startup visibility mask to shape [N, 1].

    True / 1:
        Startup feature representation is visible.

    False / 0:
        Startup feature representation must be exactly ZERO after projection.
    """

    require(
        visibility_mask.ndim in (1, 2),
        (
            "visibility_mask must be rank-1 or rank-2; "
            f"got shape {tuple(visibility_mask.shape)}."
        ),
    )

    if visibility_mask.ndim == 2:

        require(
            visibility_mask.shape[1] == 1,
            (
                "Rank-2 visibility_mask must have shape [N,1]; "
                f"got {tuple(visibility_mask.shape)}."
            ),
        )

        mask = visibility_mask

    else:

        mask = visibility_mask.unsqueeze(1)

    require(
        mask.shape[0] == row_count,
        (
            "visibility_mask row count mismatch: "
            f"expected {row_count}, got {mask.shape[0]}."
        ),
    )

    return mask.to(
        device=device,
        dtype=dtype,
    )


def apply_startup_visibility(
    representation: Tensor,
    visibility_mask: Optional[Tensor],
) -> Tensor:
    """
    Enforce outcome-independent COLD_t feature masking AFTER projection.

    Applying the mask after a biased Linear layer is deliberate:
    zero raw input alone is insufficient because the layer bias could create
    a nonzero representation.
    """

    require_rank_2(
        representation,
        "representation",
    )

    if visibility_mask is None:
        return representation

    mask = normalize_visibility_mask(
        visibility_mask=visibility_mask,
        row_count=representation.shape[0],
        device=representation.device,
        dtype=representation.dtype,
    )

    return representation * mask


# =============================================================================
# SPARSE-AWARE LINEAR PROJECTION
# =============================================================================

class SparseLinear(nn.Module):
    """
    Linear projection that accepts either dense or torch sparse rank-2 input.

    Parameter topology mirrors nn.Linear:
        weight: [out_features, in_features]
        bias:   [out_features] when enabled

    Initialization is intentionally NOT customized here.
    Phase 8.2.4 owns canonical Phase-8 parameter initialization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:

        super().__init__()

        require(
            in_features > 0,
            "SparseLinear in_features must be positive.",
        )

        require(
            out_features > 0,
            "SparseLinear out_features must be positive.",
        )

        self.in_features = int(
            in_features
        )

        self.out_features = int(
            out_features
        )

        self.weight = nn.Parameter(
            torch.empty(
                self.out_features,
                self.in_features,
            )
        )

        if bias:

            self.bias = nn.Parameter(
                torch.empty(
                    self.out_features,
                )
            )

        else:

            self.register_parameter(
                "bias",
                None,
            )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:

        require_rank_2(
            x,
            "SparseLinear input",
        )

        require(
            x.shape[1] == self.in_features,
            (
                "SparseLinear input width mismatch: "
                f"expected {self.in_features}, got {x.shape[1]}."
            ),
        )

        if x.layout == torch.strided:

            output = (
                x
                @ self.weight.T
            )

        else:

            output = torch.sparse.mm(
                x,
                self.weight.T,
            )

        if self.bias is not None:

            output = (
                output
                + self.bias
            )

        return output


# =============================================================================
# BASELINE INVESTOR DESCRIPTION ENCODER
# =============================================================================

class BaselineInvestorDescriptionEncoder(nn.Module):
    """
    Frozen-topology Phase-4 Investor description encoder.

    Investor description remains the reproduced baseline representation.

    This module is NOT used for Startup content in Phase 8.
    """

    semantic_component_id = (
        "investor_description_encoder"
    )

    def __init__(
        self,
        label_input_dim: int,
    ) -> None:

        super().__init__()

        require(
            label_input_dim > 0,
            "Investor label_input_dim must be positive.",
        )

        self.label_input_dim = int(
            label_input_dim
        )

        self.text_projection = nn.Linear(
            INVESTOR_DESCRIPTION_TEXT_INPUT_DIM,
            INVESTOR_DESCRIPTION_TEXT_DIM,
            bias=True,
        )

        self.label_projection = nn.Linear(
            self.label_input_dim,
            INVESTOR_DESCRIPTION_LABEL_DIM,
            bias=True,
        )

    def forward(
        self,
        doc2vec_32: Tensor,
        labels: Tensor,
    ) -> Tensor:

        require_width(
            doc2vec_32,
            INVESTOR_DESCRIPTION_TEXT_INPUT_DIM,
            "Investor Doc2Vec",
        )

        require_width(
            labels,
            self.label_input_dim,
            "Investor labels",
        )

        require_same_rows(
            (
                "Investor Doc2Vec",
                doc2vec_32,
            ),
            (
                "Investor labels",
                labels,
            ),
        )

        text_feature = F.relu(
            self.text_projection(
                doc2vec_32
            )
        )

        label_feature = F.relu(
            self.label_projection(
                labels
            )
        )

        output = torch.cat(
            [
                text_feature,
                label_feature,
            ],
            dim=1,
        )

        require_width(
            output,
            INVESTOR_DESCRIPTION_DIM,
            "Investor description",
        )

        return output


# =============================================================================
# INDUCTIVE STARTUP CONTENT ENCODER
# =============================================================================

class StartupContentEncoder(nn.Module):
    """
    Identity-free Startup content encoder.

    text:
        origin-frozen TF-IDF -> TruncatedSVD32 artifact
        -> Linear(32,20,bias=True) -> ReLU

    categories:
        origin-frozen sparse multi-hot
        -> SparseLinear(V_categories,20,bias=True) -> ReLU

    output:
        concat(text20, category20) = 40D
    """

    semantic_component_id = (
        "startup_content_encoder"
    )

    def __init__(
        self,
        category_input_dim: int,
    ) -> None:

        super().__init__()

        require(
            category_input_dim > 0,
            "category_input_dim must be positive.",
        )

        self.category_input_dim = int(
            category_input_dim
        )

        self.text_projection = nn.Linear(
            STARTUP_TEXT_INPUT_DIM,
            STARTUP_TEXT_PROJECTED_DIM,
            bias=True,
        )

        self.category_projection = SparseLinear(
            self.category_input_dim,
            STARTUP_CATEGORY_PROJECTED_DIM,
            bias=True,
        )

    def forward(
        self,
        text_svd32: Tensor,
        categories: Tensor,
        visibility_mask: Optional[Tensor] = None,
    ) -> Tensor:

        require_width(
            text_svd32,
            STARTUP_TEXT_INPUT_DIM,
            "Startup text_svd32",
        )

        require_width(
            categories,
            self.category_input_dim,
            "Startup categories",
        )

        require_same_rows(
            (
                "Startup text_svd32",
                text_svd32,
            ),
            (
                "Startup categories",
                categories,
            ),
        )

        text_feature = F.relu(
            self.text_projection(
                text_svd32
            )
        )

        category_feature = F.relu(
            self.category_projection(
                categories
            )
        )

        content = torch.cat(
            [
                text_feature,
                category_feature,
            ],
            dim=1,
        )

        require_width(
            content,
            STARTUP_CONTENT_DIM,
            "Startup content",
        )

        content = apply_startup_visibility(
            content,
            visibility_mask,
        )

        return content


# =============================================================================
# INDUCTIVE STARTUP STRUCTURED ENCODER
# =============================================================================

class StructuredStartupEncoder(nn.Module):
    """
    Shared Startup structured encoder.

    Architecture:
        Linear(input_dim,64,bias=True)
        -> ReLU
        -> Linear(64,40,bias=True)
        -> ReLU

    The caller determines whether input_dim corresponds to:

        B1b BASE:
            category_groups
            country
            region_location_groups
            company_type
            numeric_base

        M1/M2 EXTENDED:
            all BASE channels
            + prior_count
            + prior_sector

    Input dimensions are origin-specific and therefore constructor-driven.
    """

    def __init__(
        self,
        input_dim: int,
        semantic_component_id: str,
    ) -> None:

        super().__init__()

        require(
            input_dim > 0,
            "Structured input_dim must be positive.",
        )

        require(
            semantic_component_id
            in {
                "startup_structured_base_encoder",
                "startup_structured_extended_encoder",
            },
            (
                "Unexpected structured semantic_component_id: "
                f"{semantic_component_id}"
            ),
        )

        self.input_dim = int(
            input_dim
        )

        self.semantic_component_id = (
            semantic_component_id
        )

        self.hidden = nn.Linear(
            self.input_dim,
            STRUCTURED_HIDDEN_DIM,
            bias=True,
        )

        self.output = nn.Linear(
            STRUCTURED_HIDDEN_DIM,
            STARTUP_STRUCTURED_DIM,
            bias=True,
        )

    def forward(
        self,
        structured_input: Tensor,
        visibility_mask: Optional[Tensor] = None,
    ) -> Tensor:

        require_width(
            structured_input,
            self.input_dim,
            "Startup structured input",
        )

        hidden = F.relu(
            self.hidden(
                structured_input
            )
        )

        output = F.relu(
            self.output(
                hidden
            )
        )

        require_width(
            output,
            STARTUP_STRUCTURED_DIM,
            "Startup structured representation",
        )

        output = apply_startup_visibility(
            output,
            visibility_mask,
        )

        return output


# =============================================================================
# STARTUP GRAPH INITIALIZATION
# =============================================================================

class StartupGraphInputProjector(nn.Module):
    """
    Shared Startup graph-input projection for M1/M2.

        [content_40 || structured_40]
        -> Linear(80,40,bias=True)
        -> ReLU

    Cold visibility masking is applied AFTER this biased projection.
    """

    semantic_component_id = (
        "startup_graph_input_projector"
    )

    def __init__(self) -> None:

        super().__init__()

        self.projection = nn.Linear(
            STARTUP_CONTENT_DIM
            + STARTUP_STRUCTURED_DIM,
            GRAPH_INPUT_DIM,
            bias=True,
        )

    def forward(
        self,
        content_40: Tensor,
        structured_40: Tensor,
        visibility_mask: Optional[Tensor] = None,
    ) -> Tensor:

        require_width(
            content_40,
            STARTUP_CONTENT_DIM,
            "Startup content_40",
        )

        require_width(
            structured_40,
            STARTUP_STRUCTURED_DIM,
            "Startup structured_40",
        )

        require_same_rows(
            (
                "Startup content_40",
                content_40,
            ),
            (
                "Startup structured_40",
                structured_40,
            ),
        )

        concatenated = torch.cat(
            [
                content_40,
                structured_40,
            ],
            dim=1,
        )

        require_width(
            concatenated,
            80,
            "Startup graph projection input",
        )

        output = F.relu(
            self.projection(
                concatenated
            )
        )

        output = apply_startup_visibility(
            output,
            visibility_mask,
        )

        require_width(
            output,
            GRAPH_INPUT_DIM,
            "Startup graph input",
        )

        return output


# =============================================================================
# TREND EXTRACTOR
# =============================================================================

class TrendExtractor(nn.Module):
    """
    Phase-4-compatible trend topology with Phase-8 inductive history items.

    Query:
        [Investor latent 40 || Investor description 40] = 80D

    Item:
        [Startup structured inductive 40 || Startup content inductive 40]
        = 80D

    Attention:
        projected_query = query @ attention_weight
        score_i = projected_query dot item_i
        alpha = softmax(scores)
        period_vector = sum_i alpha_i * item_i

    Sequence:
        two-layer GRU
        80 -> hidden 40
        final time-step output
        -> Linear(40,40,bias=False)
        -> sigmoid
    """

    semantic_component_id = (
        "investor_trend_extractor"
    )

    def __init__(self) -> None:

        super().__init__()

        self.attention_weight = nn.Parameter(
            torch.empty(
                TREND_QUERY_DIM,
                TREND_ITEM_DIM,
            )
        )

        self.gru = nn.GRU(
            input_size=TREND_ITEM_DIM,
            hidden_size=TREND_HIDDEN_DIM,
            num_layers=2,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )

        self.output_projection = nn.Linear(
            TREND_HIDDEN_DIM,
            TREND_DIM,
            bias=False,
        )

    @staticmethod
    def compose_query(
        investor_latent_40: Tensor,
        investor_description_40: Tensor,
    ) -> Tensor:

        require_width(
            investor_latent_40,
            LATENT_DIM,
            "Investor latent",
        )

        require_width(
            investor_description_40,
            INVESTOR_DESCRIPTION_DIM,
            "Investor description",
        )

        require_same_rows(
            (
                "Investor latent",
                investor_latent_40,
            ),
            (
                "Investor description",
                investor_description_40,
            ),
        )

        query = torch.cat(
            [
                investor_latent_40,
                investor_description_40,
            ],
            dim=1,
        )

        require_width(
            query,
            TREND_QUERY_DIM,
            "Trend query",
        )

        return query

    @staticmethod
    def compose_history_items(
        startup_structured_40: Tensor,
        startup_content_40: Tensor,
    ) -> Tensor:

        require_width(
            startup_structured_40,
            STARTUP_STRUCTURED_DIM,
            "Historical Startup structured",
        )

        require_width(
            startup_content_40,
            STARTUP_CONTENT_DIM,
            "Historical Startup content",
        )

        require_same_rows(
            (
                "Historical Startup structured",
                startup_structured_40,
            ),
            (
                "Historical Startup content",
                startup_content_40,
            ),
        )

        items = torch.cat(
            [
                startup_structured_40,
                startup_content_40,
            ],
            dim=1,
        )

        require_width(
            items,
            TREND_ITEM_DIM,
            "Trend history items",
        )

        return items

    def attend_period(
        self,
        query_80: Tensor,
        items_80: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Attend one historical period for one Investor.

        query_80:
            [80]

        items_80:
            [num_items, 80]

        Returns:
            period_vector: [80]
            attention:     [num_items]

        Empty periods are intentionally handled by the runtime/integration
        layer as ZERO_80 before sequence construction.
        """

        require(
            query_80.ndim == 1,
            (
                "Trend period query must be rank-1; "
                f"got {tuple(query_80.shape)}."
            ),
        )

        require(
            query_80.shape[0]
            == TREND_QUERY_DIM,
            (
                "Trend period query width mismatch: "
                f"expected {TREND_QUERY_DIM}, "
                f"got {query_80.shape[0]}."
            ),
        )

        require_width(
            items_80,
            TREND_ITEM_DIM,
            "Trend period items",
        )

        require(
            items_80.shape[0] > 0,
            (
                "attend_period requires at least one item. "
                "Empty periods must be represented by ZERO_80 "
                "by the integration runtime."
            ),
        )

        projected_query = (
            query_80
            @ self.attention_weight
        )

        scores = (
            projected_query
            @ items_80.T
        )

        alpha = F.softmax(
            scores,
            dim=0,
        )

        period_vector = (
            alpha
            @ items_80
        )

        return (
            period_vector,
            alpha,
        )

    def encode_sequence(
        self,
        sequence_80: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        require(
            sequence_80.ndim == 3,
            (
                "Trend sequence must be rank-3 [B,T,80]; "
                f"got shape {tuple(sequence_80.shape)}."
            ),
        )

        require(
            sequence_80.shape[2]
            == TREND_ITEM_DIM,
            (
                "Trend sequence item width mismatch: "
                f"expected {TREND_ITEM_DIM}, "
                f"got {sequence_80.shape[2]}."
            ),
        )

        require(
            sequence_80.shape[1] > 0,
            "Trend sequence must contain at least one period.",
        )

        gru_output, _ = self.gru(
            sequence_80
        )

        last_hidden_output = (
            gru_output[
                :,
                -1,
                :
            ]
        )

        trend = torch.sigmoid(
            self.output_projection(
                last_hidden_output
            )
        )

        require_width(
            trend,
            TREND_DIM,
            "Investor trend",
        )

        return (
            trend,
            gru_output,
        )


# =============================================================================
# BASIS-DECOMPOSED R-GCN
# =============================================================================

class BasisRGCNLayer(nn.Module):
    """
    Exact Phase-4-compatible relation propagation semantics.

    For relation r:

        W_r = sum_b a[r,b] * V_b

        message(u -> v, r)
            = x[u] @ W_r

        normalization
            = incoming degree of v WITHIN relation r

        aggregate[v]
            += mean relation-r incoming message

    Root:
        root[v] = x[v] @ root_weight

    Output:
        aggregate + root

    No layer bias.
    No explicit self-loop edges.
    Root transform is explicit.
    """

    semantic_component_id = (
        "heterogeneous_rgcn"
    )

    def __init__(
        self,
        in_dim: int = GRAPH_INPUT_DIM,
        out_dim: int = GRAPH_OUTPUT_DIM,
        num_relations: int = NUM_TYPED_RELATIONS,
        num_bases: int = NUM_RGCN_BASES,
    ) -> None:

        super().__init__()

        require(
            in_dim > 0,
            "R-GCN in_dim must be positive.",
        )

        require(
            out_dim > 0,
            "R-GCN out_dim must be positive.",
        )

        require(
            num_relations == NUM_TYPED_RELATIONS,
            (
                "Phase-8 primary R-GCN requires exactly "
                f"{NUM_TYPED_RELATIONS} relations."
            ),
        )

        require(
            num_bases == NUM_RGCN_BASES,
            (
                "Phase-8 primary R-GCN requires exactly "
                f"{NUM_RGCN_BASES} bases."
            ),
        )

        self.in_dim = int(
            in_dim
        )

        self.out_dim = int(
            out_dim
        )

        self.num_relations = int(
            num_relations
        )

        self.num_bases = int(
            num_bases
        )

        self.bases = nn.Parameter(
            torch.empty(
                self.num_bases,
                self.in_dim,
                self.out_dim,
            )
        )

        self.coefficients = nn.Parameter(
            torch.empty(
                self.num_relations,
                self.num_bases,
            )
        )

        self.root_weight = nn.Parameter(
            torch.empty(
                self.in_dim,
                self.out_dim,
            )
        )

    def effective_weight(
        self,
        relation_id: int,
    ) -> Tensor:

        require(
            0
            <= relation_id
            < self.num_relations,
            (
                "relation_id out of range: "
                f"{relation_id}"
            ),
        )

        return torch.einsum(
            "b,bio->io",
            self.coefficients[
                relation_id
            ],
            self.bases,
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Tensor:

        require_width(
            x,
            self.in_dim,
            "R-GCN node input",
        )

        require(
            edge_index.ndim == 2
            and edge_index.shape[0] == 2,
            (
                "edge_index must have shape [2,E]; "
                f"got {tuple(edge_index.shape)}."
            ),
        )

        require(
            edge_type.ndim == 1,
            (
                "edge_type must be rank-1; "
                f"got {tuple(edge_type.shape)}."
            ),
        )

        require(
            edge_index.shape[1]
            == edge_type.shape[0],
            "edge_index / edge_type length mismatch.",
        )

        require(
            edge_index.dtype == torch.long,
            (
                "edge_index must use torch.long; "
                f"got {edge_index.dtype}."
            ),
        )

        require(
            edge_type.dtype == torch.long,
            (
                "edge_type must use torch.long; "
                f"got {edge_type.dtype}."
            ),
        )

        num_nodes = x.shape[0]

        aggregated = x.new_zeros(
            (
                num_nodes,
                self.out_dim,
            )
        )

        source_all = (
            edge_index[
                0
            ]
        )

        destination_all = (
            edge_index[
                1
            ]
        )

        for relation_id in range(
            self.num_relations
        ):

            mask = (
                edge_type
                == relation_id
            )

            # A zero-edge typed relation contributes exactly zero.
            # This preserves the mathematical propagation rule while allowing
            # origin/mask-specific graphs to remain valid.
            if not bool(
                mask.any()
            ):
                continue

            relation_sources = (
                source_all[
                    mask
                ]
            )

            relation_destinations = (
                destination_all[
                    mask
                ]
            )

            weight = self.effective_weight(
                relation_id
            )

            messages = (
                x[
                    relation_sources
                ]
                @ weight
            )

            relation_degree = torch.bincount(
                relation_destinations,
                minlength=num_nodes,
            )

            normalization = (
                relation_degree[
                    relation_destinations
                ]
                .to(
                    dtype=messages.dtype
                )
                .unsqueeze(
                    1
                )
            )

            require(
                bool(
                    torch.all(
                        normalization
                        > 0
                    )
                ),
                (
                    "Encountered zero incoming "
                    "relation normalization denominator."
                ),
            )

            messages = (
                messages
                / normalization
            )

            aggregated.index_add_(
                0,
                relation_destinations,
                messages,
            )

        root = (
            x
            @ self.root_weight
        )

        return (
            aggregated
            + root
        )


class HeterogeneousPreferencePropagation(nn.Module):
    """
    Two-layer 40 -> 40 -> 40 basis R-GCN.

    ReLU is applied after BOTH layers.
    """

    semantic_component_id = (
        "heterogeneous_rgcn"
    )

    def __init__(self) -> None:

        super().__init__()

        self.layer_1 = BasisRGCNLayer(
            GRAPH_INPUT_DIM,
            GRAPH_HIDDEN_DIM,
            num_relations=NUM_TYPED_RELATIONS,
            num_bases=NUM_RGCN_BASES,
        )

        self.layer_2 = BasisRGCNLayer(
            GRAPH_HIDDEN_DIM,
            GRAPH_OUTPUT_DIM,
            num_relations=NUM_TYPED_RELATIONS,
            num_bases=NUM_RGCN_BASES,
        )

    def forward(
        self,
        node_input_40: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Dict[str, Tensor]:

        h1_pre = self.layer_1(
            node_input_40,
            edge_index,
            edge_type,
        )

        h1 = F.relu(
            h1_pre
        )

        h2_pre = self.layer_2(
            h1,
            edge_index,
            edge_type,
        )

        h2 = F.relu(
            h2_pre
        )

        return {
            "h1_pre":
                h1_pre,

            "h1":
                h1,

            "h2_pre":
                h2_pre,

            "h2":
                h2,
        }


# =============================================================================
# GLOBAL GRAPH INPUT ASSEMBLY
# =============================================================================

def build_global_graph_input(
    total_node_count: int,
    investor_latent_weight: Tensor,
    startup_node_indices: Tensor,
    startup_graph_input_40: Tensor,
) -> Tensor:
    """
    Build the 40D input matrix for the Phase-8 heterogeneous graph.

    Global role-node indexing is supplied by Phase 8.2.2.

    Investor nodes:
        existing trainable Investor latent L_o.

    Startup nodes:
        shared inductive Startup graph-input projection.

    Person nodes:
        ZERO_40.

    Any Startup-role node without an origin-eligible materialized Startup row:
        ZERO_40.

    This function creates no identity parameter.
    """

    require(
        total_node_count > NUM_INVESTORS,
        (
            "total_node_count must exceed Investor population."
        ),
    )

    require_width(
        investor_latent_weight,
        LATENT_DIM,
        "Investor latent weight",
    )

    require(
        investor_latent_weight.shape[0]
        == NUM_INVESTORS,
        (
            "Investor latent table row count mismatch: "
            f"expected {NUM_INVESTORS}, "
            f"got {investor_latent_weight.shape[0]}."
        ),
    )

    require(
        startup_node_indices.ndim == 1,
        (
            "startup_node_indices must be rank-1; "
            f"got {tuple(startup_node_indices.shape)}."
        ),
    )

    require(
        startup_node_indices.dtype == torch.long,
        (
            "startup_node_indices must use torch.long; "
            f"got {startup_node_indices.dtype}."
        ),
    )

    require_width(
        startup_graph_input_40,
        GRAPH_INPUT_DIM,
        "Startup graph input",
    )

    require(
        startup_node_indices.shape[0]
        == startup_graph_input_40.shape[0],
        (
            "startup_node_indices / startup graph input "
            "row-count mismatch."
        ),
    )

    require(
        startup_graph_input_40.device
        == investor_latent_weight.device,
        (
            "Investor and Startup graph inputs must "
            "be on the same device."
        ),
    )

    require(
        startup_graph_input_40.dtype
        == investor_latent_weight.dtype,
        (
            "Investor and Startup graph inputs must "
            "have the same dtype."
        ),
    )

    if startup_node_indices.numel() > 0:

        require(
            bool(
                torch.all(
                    startup_node_indices
                    >= NUM_INVESTORS
                )
            ),
            (
                "Startup global node index overlaps "
                "Investor role-node range."
            ),
        )

        require(
            bool(
                torch.all(
                    startup_node_indices
                    < total_node_count
                )
            ),
            (
                "Startup global node index exceeds "
                "global graph node universe."
            ),
        )

        require(
            int(
                torch.unique(
                    startup_node_indices
                ).numel()
            )
            == int(
                startup_node_indices.numel()
            ),
            (
                "startup_node_indices contains duplicates."
            ),
        )

    node_input = investor_latent_weight.new_zeros(
        (
            total_node_count,
            GRAPH_INPUT_DIM,
        )
    )

    node_input[
        :NUM_INVESTORS
    ] = investor_latent_weight

    node_input[
        startup_node_indices
    ] = startup_graph_input_40

    return node_input


def apply_edge_mask(
    edge_index: Tensor,
    edge_type: Tensor,
    edge_mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    """
    Select the authoritative fit or inference graph.

    The mask itself must come from Phase 8.2.2.
    """

    require(
        edge_index.ndim == 2
        and edge_index.shape[0] == 2,
        "edge_index must have shape [2,E].",
    )

    require(
        edge_type.ndim == 1,
        "edge_type must be rank-1.",
    )

    require(
        edge_mask.ndim == 1,
        "edge_mask must be rank-1.",
    )

    require(
        edge_index.shape[1]
        == edge_type.shape[0]
        == edge_mask.shape[0],
        "edge_index / edge_type / edge_mask length mismatch.",
    )

    require(
        edge_mask.dtype == torch.bool,
        (
            "edge_mask must use torch.bool; "
            f"got {edge_mask.dtype}."
        ),
    )

    return (
        edge_index[
            :,
            edge_mask
        ],
        edge_type[
            edge_mask
        ],
    )


# =============================================================================
# PAIR REPRESENTATION
# =============================================================================

def concatenate_pair_features(
    investor_trend_40: Tensor,
    investor_latent_40: Tensor,
    investor_description_40: Tensor,
    investor_structural_40: Tensor,
    startup_structured_40: Tensor,
    startup_content_40: Tensor,
    startup_structural_40: Tensor,
) -> Tensor:
    """
    Exact frozen pair order:

        0 investor_trend
        1 investor_latent
        2 investor_description
        3 investor_structural
        4 startup_structured_inductive
        5 startup_content_inductive
        6 startup_structural_inductive
    """

    components = (
        (
            "investor_trend",
            investor_trend_40,
        ),
        (
            "investor_latent",
            investor_latent_40,
        ),
        (
            "investor_description",
            investor_description_40,
        ),
        (
            "investor_structural",
            investor_structural_40,
        ),
        (
            "startup_structured",
            startup_structured_40,
        ),
        (
            "startup_content",
            startup_content_40,
        ),
        (
            "startup_structural",
            startup_structural_40,
        ),
    )

    for name, tensor in components:

        require_width(
            tensor,
            LATENT_DIM,
            name,
        )

    require_same_rows(
        *components
    )

    pair_features = torch.cat(
        [
            tensor
            for _, tensor in components
        ],
        dim=1,
    )

    require_width(
        pair_features,
        PAIR_DIM,
        "Pair representation",
    )

    return pair_features


# =============================================================================
# SCORER
# =============================================================================

class ScoringMLP(nn.Module):
    """
    Frozen scorer topology:

        280 -> 128 -> 64 -> 32 -> 16 -> 1

    ReLU after each hidden layer.
    Hidden bias: true.
    Output bias: true.
    No dropout.
    No BN/LN.
    Output: raw logit.
    """

    semantic_component_id = (
        "pair_scorer"
    )

    def __init__(self) -> None:

        super().__init__()

        self.hidden_1 = nn.Linear(
            PAIR_DIM,
            SCORER_HIDDEN_WIDTHS[0],
            bias=True,
        )

        self.hidden_2 = nn.Linear(
            SCORER_HIDDEN_WIDTHS[0],
            SCORER_HIDDEN_WIDTHS[1],
            bias=True,
        )

        self.hidden_3 = nn.Linear(
            SCORER_HIDDEN_WIDTHS[1],
            SCORER_HIDDEN_WIDTHS[2],
            bias=True,
        )

        self.hidden_4 = nn.Linear(
            SCORER_HIDDEN_WIDTHS[2],
            SCORER_HIDDEN_WIDTHS[3],
            bias=True,
        )

        self.output = nn.Linear(
            SCORER_HIDDEN_WIDTHS[3],
            SCORER_OUTPUT_DIM,
            bias=True,
        )

    def forward_logits(
        self,
        pair_features: Tensor,
    ) -> Tensor:

        require_width(
            pair_features,
            PAIR_DIM,
            "Scoring pair_features",
        )

        h = F.relu(
            self.hidden_1(
                pair_features
            )
        )

        h = F.relu(
            self.hidden_2(
                h
            )
        )

        h = F.relu(
            self.hidden_3(
                h
            )
        )

        h = F.relu(
            self.hidden_4(
                h
            )
        )

        return self.output(
            h
        )

    def forward(
        self,
        pair_features: Tensor,
    ) -> Tensor:

        return self.forward_logits(
            pair_features
        )

    def predict_probability(
        self,
        pair_features: Tensor,
    ) -> Tensor:

        return torch.sigmoid(
            self.forward_logits(
                pair_features
            )
        )


# =============================================================================
# MODEL-FAMILY SEMANTICS
# =============================================================================

@dataclass(frozen=True)
class CandidateFamilySpec:

    model: str

    trend_active: bool

    graph_active: bool

    structured_mode: str

    graph_family: str

    investor_trend_slot: str

    investor_structural_slot: str

    startup_structured_slot: str

    startup_content_slot: str

    startup_structural_slot: str


FAMILY_SPECS: Dict[
    str,
    CandidateFamilySpec,
] = {
    "B1a":
        CandidateFamilySpec(
            model="B1a",
            trend_active=True,
            graph_active=False,
            structured_mode="ZERO",
            graph_family="NONE",
            investor_trend_slot="ITRS_TREND_40",
            investor_structural_slot="ZERO_40",
            startup_structured_slot="ZERO_40",
            startup_content_slot="INDUCTIVE_CONTENT_40",
            startup_structural_slot="ZERO_40",
        ),

    "B1b":
        CandidateFamilySpec(
            model="B1b",
            trend_active=True,
            graph_active=False,
            structured_mode="BASE",
            graph_family="NONE",
            investor_trend_slot="ITRS_TREND_40",
            investor_structural_slot="ZERO_40",
            startup_structured_slot="INDUCTIVE_BASE_STRUCTURED_40",
            startup_content_slot="INDUCTIVE_CONTENT_40",
            startup_structural_slot="ZERO_40",
        ),

    "M1":
        CandidateFamilySpec(
            model="M1",
            trend_active=False,
            graph_active=True,
            structured_mode="EXTENDED",
            graph_family="G_PRIMARY",
            investor_trend_slot="ZERO_40",
            investor_structural_slot="HETEROGENEOUS_RGCN_40",
            startup_structured_slot="INDUCTIVE_EXTENDED_STRUCTURED_40",
            startup_content_slot="INDUCTIVE_CONTENT_40",
            startup_structural_slot="HETEROGENEOUS_RGCN_40",
        ),

    "M2":
        CandidateFamilySpec(
            model="M2",
            trend_active=True,
            graph_active=True,
            structured_mode="EXTENDED",
            graph_family="G_PRIMARY",
            investor_trend_slot="ITRS_TREND_40",
            investor_structural_slot="HETEROGENEOUS_RGCN_40",
            startup_structured_slot="INDUCTIVE_EXTENDED_STRUCTURED_40",
            startup_content_slot="INDUCTIVE_CONTENT_40",
            startup_structural_slot="HETEROGENEOUS_RGCN_40",
        ),
}


# =============================================================================
# COMPLETE CANDIDATE-MODEL MODULE CONTAINER
# =============================================================================

class Phase8CandidateModel(nn.Module):
    """
    Module container for one candidate family at one origin.

    Origin-specific dimensions are supplied to the constructor because the
    training-side vocabularies differ slightly across origins.

    This class intentionally separates:
        feature encoding,
        trend-sequence construction,
        graph input construction,
        graph propagation,
        pair assembly,
        scoring.

    Phase 8.2.5 will audit the integrated forward/insertion behavior.
    """

    def __init__(
        self,
        model_family: str,
        investor_label_input_dim: int,
        startup_category_input_dim: int,
        startup_structured_input_dim: Optional[int],
        num_investors: int = NUM_INVESTORS,
    ) -> None:

        super().__init__()

        require(
            model_family
            in FAMILY_SPECS,
            (
                "model_family must be one of "
                f"{tuple(FAMILY_SPECS.keys())}; "
                f"got {model_family}."
            ),
        )

        require(
            num_investors
            == NUM_INVESTORS,
            (
                "Phase-8 Investor-role universe changed: "
                f"expected {NUM_INVESTORS}, "
                f"got {num_investors}."
            ),
        )

        self.model_family = (
            model_family
        )

        self.family_spec = (
            FAMILY_SPECS[
                model_family
            ]
        )

        # ---------------------------------------------------------------------
        # Investor-side baseline-preserving modules.
        # ---------------------------------------------------------------------

        self.investor_embedding = nn.Embedding(
            num_investors,
            LATENT_DIM,
        )

        self.investor_embedding.semantic_component_id = (
            "investor_latent"
        )

        self.investor_description_encoder = (
            BaselineInvestorDescriptionEncoder(
                label_input_dim=investor_label_input_dim,
            )
        )

        # ---------------------------------------------------------------------
        # Identity-free Startup content.
        # ---------------------------------------------------------------------

        self.startup_content_encoder = (
            StartupContentEncoder(
                category_input_dim=startup_category_input_dim,
            )
        )

        # ---------------------------------------------------------------------
        # Structured branch.
        # ---------------------------------------------------------------------

        if (
            self.family_spec.structured_mode
            == "ZERO"
        ):

            require(
                startup_structured_input_dim
                is None,
                (
                    "B1a structured input must be None "
                    "because its structured slot is ZERO_40."
                ),
            )

            self.startup_structured_encoder = None

        else:

            require(
                startup_structured_input_dim
                is not None,
                (
                    f"{model_family} requires an origin-specific "
                    "structured input dimension."
                ),
            )

            component_id = (
                "startup_structured_base_encoder"
                if self.family_spec.structured_mode
                == "BASE"
                else
                "startup_structured_extended_encoder"
            )

            self.startup_structured_encoder = (
                StructuredStartupEncoder(
                    input_dim=int(
                        startup_structured_input_dim
                    ),
                    semantic_component_id=component_id,
                )
            )

        # ---------------------------------------------------------------------
        # Trend.
        # ---------------------------------------------------------------------

        if self.family_spec.trend_active:

            self.trend_extractor = (
                TrendExtractor()
            )

        else:

            self.trend_extractor = None

        # ---------------------------------------------------------------------
        # Heterogeneous graph.
        # ---------------------------------------------------------------------

        if self.family_spec.graph_active:

            self.startup_graph_input_projector = (
                StartupGraphInputProjector()
            )

            self.preference_propagation = (
                HeterogeneousPreferencePropagation()
            )

        else:

            self.startup_graph_input_projector = None
            self.preference_propagation = None

        # ---------------------------------------------------------------------
        # Shared frozen-topology scorer.
        # ---------------------------------------------------------------------

        self.scorer = (
            ScoringMLP()
        )

    # =========================================================================
    # INVESTOR ENCODING
    # =========================================================================

    def encode_investor_latent(
        self,
        investor_indices: Tensor,
    ) -> Tensor:

        require(
            investor_indices.ndim == 1,
            (
                "investor_indices must be rank-1; "
                f"got {tuple(investor_indices.shape)}."
            ),
        )

        require(
            investor_indices.dtype
            == torch.long,
            (
                "investor_indices must use torch.long; "
                f"got {investor_indices.dtype}."
            ),
        )

        output = self.investor_embedding(
            investor_indices
        )

        require_width(
            output,
            LATENT_DIM,
            "Investor latent",
        )

        return output

    def encode_investor_description(
        self,
        investor_doc2vec32: Tensor,
        investor_labels: Tensor,
    ) -> Tensor:

        return self.investor_description_encoder(
            investor_doc2vec32,
            investor_labels,
        )

    # =========================================================================
    # STARTUP ENCODING
    # =========================================================================

    def encode_startup_content(
        self,
        text_svd32: Tensor,
        categories: Tensor,
        visibility_mask: Optional[Tensor] = None,
    ) -> Tensor:

        return self.startup_content_encoder(
            text_svd32=text_svd32,
            categories=categories,
            visibility_mask=visibility_mask,
        )

    def encode_startup_structured(
        self,
        startup_content_40: Tensor,
        structured_input: Optional[Tensor],
        visibility_mask: Optional[Tensor] = None,
    ) -> Tensor:

        require_width(
            startup_content_40,
            STARTUP_CONTENT_DIM,
            "Startup content reference",
        )

        if (
            self.family_spec.structured_mode
            == "ZERO"
        ):

            require(
                structured_input
                is None,
                (
                    "B1a must not receive a structured input."
                ),
            )

            return zero_40_like(
                startup_content_40
            )

        require(
            self.startup_structured_encoder
            is not None,
            "Structured encoder unexpectedly absent.",
        )

        require(
            structured_input
            is not None,
            (
                f"{self.model_family} requires "
                "structured_input."
            ),
        )

        return self.startup_structured_encoder(
            structured_input=structured_input,
            visibility_mask=visibility_mask,
        )

    # =========================================================================
    # TREND
    # =========================================================================

    def compose_trend_query(
        self,
        investor_latent_40: Tensor,
        investor_description_40: Tensor,
    ) -> Tensor:

        require(
            self.family_spec.trend_active,
            (
                f"{self.model_family} has ZERO_40 trend; "
                "trend query construction is inactive."
            ),
        )

        require(
            self.trend_extractor
            is not None,
            "Trend extractor unexpectedly absent.",
        )

        return self.trend_extractor.compose_query(
            investor_latent_40,
            investor_description_40,
        )

    def compose_history_items(
        self,
        startup_structured_40: Tensor,
        startup_content_40: Tensor,
    ) -> Tensor:

        require(
            self.family_spec.trend_active,
            (
                f"{self.model_family} has ZERO_40 trend; "
                "history-item construction is inactive."
            ),
        )

        require(
            self.trend_extractor
            is not None,
            "Trend extractor unexpectedly absent.",
        )

        return self.trend_extractor.compose_history_items(
            startup_structured_40,
            startup_content_40,
        )

    def encode_trend_sequence(
        self,
        sequence_80: Tensor,
    ) -> Tuple[Tensor, Tensor]:

        require(
            self.family_spec.trend_active,
            (
                f"{self.model_family} has ZERO_40 trend."
            ),
        )

        require(
            self.trend_extractor
            is not None,
            "Trend extractor unexpectedly absent.",
        )

        return self.trend_extractor.encode_sequence(
            sequence_80
        )

    # =========================================================================
    # GRAPH
    # =========================================================================

    def encode_startup_graph_input(
        self,
        startup_content_40: Tensor,
        startup_structured_40: Tensor,
        visibility_mask: Optional[Tensor] = None,
    ) -> Tensor:

        require(
            self.family_spec.graph_active,
            (
                f"{self.model_family} is a no-graph comparator."
            ),
        )

        require(
            self.startup_graph_input_projector
            is not None,
            "Startup graph projector unexpectedly absent.",
        )

        return self.startup_graph_input_projector(
            content_40=startup_content_40,
            structured_40=startup_structured_40,
            visibility_mask=visibility_mask,
        )

    def propagate_graph(
        self,
        global_node_input_40: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
    ) -> Dict[str, Tensor]:

        require(
            self.family_spec.graph_active,
            (
                f"{self.model_family} is a no-graph comparator."
            ),
        )

        require(
            self.preference_propagation
            is not None,
            "R-GCN unexpectedly absent.",
        )

        return self.preference_propagation(
            node_input_40=global_node_input_40,
            edge_index=edge_index,
            edge_type=edge_type,
        )

    # =========================================================================
    # PAIR SLOTS
    # =========================================================================

    def resolve_trend_slot(
        self,
        investor_latent_40: Tensor,
        investor_trend_40: Optional[Tensor],
    ) -> Tensor:

        require_width(
            investor_latent_40,
            LATENT_DIM,
            "Investor latent reference",
        )

        if not self.family_spec.trend_active:

            require(
                investor_trend_40
                is None,
                (
                    "M1 trend slot is ZERO_40 and must not "
                    "receive a learned trend tensor."
                ),
            )

            return zero_40_like(
                investor_latent_40
            )

        require(
            investor_trend_40
            is not None,
            (
                f"{self.model_family} requires "
                "investor_trend_40."
            ),
        )

        require_width(
            investor_trend_40,
            TREND_DIM,
            "Investor trend",
        )

        return investor_trend_40

    def resolve_investor_structural_slot(
        self,
        investor_latent_40: Tensor,
        investor_structural_40: Optional[Tensor],
    ) -> Tensor:

        require_width(
            investor_latent_40,
            LATENT_DIM,
            "Investor latent reference",
        )

        if not self.family_spec.graph_active:

            require(
                investor_structural_40
                is None,
                (
                    f"{self.model_family} is no-graph; "
                    "investor_structural_40 must be None."
                ),
            )

            return zero_40_like(
                investor_latent_40
            )

        require(
            investor_structural_40
            is not None,
            (
                f"{self.model_family} requires "
                "investor_structural_40."
            ),
        )

        require_width(
            investor_structural_40,
            GRAPH_OUTPUT_DIM,
            "Investor structural",
        )

        return investor_structural_40

    def resolve_startup_structural_slot(
        self,
        startup_content_40: Tensor,
        startup_structural_40: Optional[Tensor],
    ) -> Tensor:

        require_width(
            startup_content_40,
            STARTUP_CONTENT_DIM,
            "Startup content reference",
        )

        if not self.family_spec.graph_active:

            require(
                startup_structural_40
                is None,
                (
                    f"{self.model_family} is no-graph; "
                    "startup_structural_40 must be None."
                ),
            )

            return zero_40_like(
                startup_content_40
            )

        require(
            startup_structural_40
            is not None,
            (
                f"{self.model_family} requires "
                "startup_structural_40."
            ),
        )

        require_width(
            startup_structural_40,
            GRAPH_OUTPUT_DIM,
            "Startup structural",
        )

        return startup_structural_40

    # =========================================================================
    # SCORING
    # =========================================================================

    def assemble_pair(
        self,
        investor_trend_40: Tensor,
        investor_latent_40: Tensor,
        investor_description_40: Tensor,
        investor_structural_40: Tensor,
        startup_structured_40: Tensor,
        startup_content_40: Tensor,
        startup_structural_40: Tensor,
    ) -> Tensor:

        return concatenate_pair_features(
            investor_trend_40=investor_trend_40,
            investor_latent_40=investor_latent_40,
            investor_description_40=investor_description_40,
            investor_structural_40=investor_structural_40,
            startup_structured_40=startup_structured_40,
            startup_content_40=startup_content_40,
            startup_structural_40=startup_structural_40,
        )

    def score_pair(
        self,
        pair_features_280: Tensor,
    ) -> Tensor:

        return self.scorer(
            pair_features_280
        )


# =============================================================================
# STATIC CONTRACT DESCRIPTION
# =============================================================================

def module_contract_summary() -> Dict[str, object]:
    """
    Pure metadata helper.

    No model is instantiated.
    """

    return {
        "phase":
            "8.2.3",

        "version":
            "V1",

        "candidate_families":
            tuple(
                FAMILY_SPECS.keys()
            ),

        "pair_component_order":
            PAIR_COMPONENT_ORDER,

        "pair_dim":
            PAIR_DIM,

        "investor_scoring_dim":
            INVESTOR_SCORING_DIM,

        "startup_scoring_dim":
            STARTUP_SCORING_DIM,

        "typed_relations":
            TYPED_RELATION_VOCABULARY,

        "num_typed_relations":
            NUM_TYPED_RELATIONS,

        "rgcn_layers":
            NUM_RGCN_LAYERS,

        "rgcn_bases":
            NUM_RGCN_BASES,

        "rgcn_width":
            GRAPH_OUTPUT_DIM,

        "scorer_hidden_widths":
            SCORER_HIDDEN_WIDTHS,

        "trend_query":
            (
                "investor_latent_40",
                "investor_description_40",
            ),

        "trend_history_item":
            (
                "startup_structured_inductive_40",
                "startup_content_inductive_40",
            ),

        "startup_id_embedding":
            False,

        "person_id_embedding":
            False,

        "phase4_startup_doc2vec":
            False,

        "training_performed":
            False,
    }


if __name__ == "__main__":

    print(
        "=" * 120
    )

    print(
        "PHASE 8.2.3 — CANDIDATE MODEL MODULES V1"
    )

    print(
        "=" * 120
    )

    print(
        "Definition module only."
    )

    print(
        "Model instantiated:     NO"
    )

    print(
        "Initialization applied: NO"
    )

    print(
        "Optimizer created:      NO"
    )

    print(
        "Training performed:     NO"
    )

    print(
        "Performance inspected:  NO"
    )

    print(
        "T60 scored:             NO"
    )

    print(
        "April analyzed:         NO"
    )

    print(
        "=" * 120
    )
