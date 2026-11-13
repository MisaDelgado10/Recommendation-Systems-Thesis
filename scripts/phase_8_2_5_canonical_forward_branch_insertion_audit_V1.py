from pathlib import Path
import gc
import hashlib
import importlib.util
import json
import sys

import torch


# =============================================================================
# PATHS
# =============================================================================

MODEL_MODULE_PATH = Path(
    "scripts/"
    "phase_8_2_3_candidate_model_modules_V1.py"
)

INITIALIZER_PATH = Path(
    "scripts/"
    "phase_8_2_4_canonical_initialization_audit_V1.py"
)

OUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_5_forward_branch_insertion"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SUMMARY_PATH = (
    OUT_DIR
    / "phase_8_2_5_forward_branch_insertion_summary_V1.json"
)

HASH_PATH = (
    OUT_DIR
    / "phase_8_2_5_artifact_sha256_V1.json"
)


# =============================================================================
# HELPERS
# =============================================================================

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256_file(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def tensor_hash(tensor):
    return hashlib.sha256(
        tensor
        .detach()
        .cpu()
        .contiguous()
        .numpy()
        .tobytes()
    ).hexdigest()


def load_python_module(
    path,
    module_name,
):
    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    require(
        spec is not None
        and spec.loader is not None,
        f"Could not import {path}",
    )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def require_zero_row(
    tensor,
    row,
    name,
):
    require(
        bool(
            torch.count_nonzero(
                tensor[row]
            )
            == 0
        ),
        f"{name} row {row} is not exactly ZERO.",
    )


def require_same_tensor(
    left,
    right,
    name,
):
    require(
        torch.equal(
            left,
            right,
        ),
        f"{name} is not bitwise identical.",
    )


def require_finite(
    tensor,
    name,
):
    require(
        bool(
            torch.isfinite(
                tensor
            ).all()
        ),
        f"{name} contains non-finite values.",
    )


# =============================================================================
# SYNTHETIC INPUT FACTORY
# =============================================================================

def make_inputs(
    *,
    category_dim,
    structured_dim,
):
    batch = 3

    investor_indices = torch.tensor(
        [0, 1, 2],
        dtype=torch.long,
    )

    investor_doc2vec = (
        torch.arange(
            batch * 32,
            dtype=torch.float32,
        )
        .reshape(
            batch,
            32,
        )
        / 100.0
    )

    investor_labels = torch.zeros(
        batch,
        802,
        dtype=torch.float32,
    )

    investor_labels[0, 0] = 1.0
    investor_labels[1, 10] = 1.0
    investor_labels[2, 100] = 1.0

    startup_text = (
        torch.arange(
            batch * 32,
            dtype=torch.float32,
        )
        .reshape(
            batch,
            32,
        )
        / 200.0
    )

    category_indices = torch.tensor(
        [
            [0, 1, 2, 2],
            [0, 10, 20, category_dim - 1],
        ],
        dtype=torch.long,
    )

    category_values = torch.ones(
        category_indices.shape[1],
        dtype=torch.float32,
    )

    startup_categories = (
        torch.sparse_coo_tensor(
            category_indices,
            category_values,
            size=(
                batch,
                category_dim,
            ),
            dtype=torch.float32,
        )
        .coalesce()
    )

    if structured_dim is None:

        structured_input = None

    else:

        structured_input = torch.zeros(
            batch,
            structured_dim,
            dtype=torch.float32,
        )

        structured_input[0, 0] = 1.0
        structured_input[1, 1] = 0.5
        structured_input[2, structured_dim - 1] = 1.0

    # Row 1 is the synthetic COLD_t Startup.
    fit_visibility = torch.tensor(
        [True, False, True],
        dtype=torch.bool,
    )

    inference_visibility = torch.tensor(
        [True, True, True],
        dtype=torch.bool,
    )

    return {
        "investor_indices":
            investor_indices,

        "investor_doc2vec":
            investor_doc2vec,

        "investor_labels":
            investor_labels,

        "startup_text":
            startup_text,

        "startup_categories":
            startup_categories,

        "structured_input":
            structured_input,

        "fit_visibility":
            fit_visibility,

        "inference_visibility":
            inference_visibility,
    }


# =============================================================================
# SYNTHETIC GRAPH
# =============================================================================

def make_graph():
    """
    Tiny typed graph.

    Node roles:
        0,1,2 = Investor
        3,4,5 = Startup
        6     = Person

    Startup node 4 is synthetic COLD_t.

    The base edge set contains all 10 relation IDs exactly once.

    fit_mask:
        removes every edge incident to node 4.

    inference_mask:
        restores incoming non-cold -> node 4 edges,
        but suppresses node 4 -> * outgoing edges.
    """

    edge_index = torch.tensor(
        [
            # sources
            [
                3,  # r0 startup -> person
                6,  # r1 person -> startup
                0,  # r2 investor -> investor
                0,  # r3 investor -> startup
                3,  # r4 startup -> investor
                3,  # r5 startup -> startup(COLD)
                1,  # r6 investor -> investor
                1,  # r7 investor -> startup(COLD)
                4,  # r8 COLD startup -> investor
                4,  # r9 COLD startup -> startup
            ],

            # destinations
            [
                6,
                3,
                1,
                3,
                0,
                4,
                0,
                4,
                0,
                5,
            ],
        ],
        dtype=torch.long,
    )

    edge_type = torch.arange(
        10,
        dtype=torch.long,
    )

    fit_mask = torch.tensor(
        [
            True,
            True,
            True,
            True,
            True,
            False,  # incident to COLD
            True,
            False,  # incoming to COLD hidden at fit
            False,  # outgoing from COLD
            False,  # outgoing from COLD
        ],
        dtype=torch.bool,
    )

    inference_mask = torch.tensor(
        [
            True,
            True,
            True,
            True,
            True,
            True,   # incoming to COLD restored
            True,
            True,   # incoming to COLD restored
            False,  # outgoing from COLD suppressed
            False,  # outgoing from COLD suppressed
        ],
        dtype=torch.bool,
    )

    return (
        edge_index,
        edge_type,
        fit_mask,
        inference_mask,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 120)
    print("PHASE 8.2.5 — CANONICAL FORWARD / BRANCH INSERTION AUDIT V1")
    print("=" * 120)

    require(
        MODEL_MODULE_PATH.exists(),
        "Missing Phase-8.2.3 model module.",
    )

    require(
        INITIALIZER_PATH.exists(),
        "Missing Phase-8.2.4 initializer.",
    )

    model_module = load_python_module(
        MODEL_MODULE_PATH,
        "phase8_candidate_model_forward_audit",
    )

    init_module = load_python_module(
        INITIALIZER_PATH,
        "phase8_initializer_forward_audit",
    )

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

    dimensions = (
        init_module
        .load_origin_dimensions(
            "t57"
        )
    )

    require(
        dimensions["category_dim"] == 803,
        "T57 category width changed.",
    )

    require(
        dimensions["base_dim"] == 1896,
        "T57 base structured width changed.",
    )

    require(
        dimensions["extended_dim"] == 2744,
        "T57 extended structured width changed.",
    )

    (
        full_edge_index,
        full_edge_type,
        fit_edge_mask,
        inference_edge_mask,
    ) = make_graph()

    fit_edge_index, fit_edge_type = (
        model_module.apply_edge_mask(
            full_edge_index,
            full_edge_type,
            fit_edge_mask,
        )
    )

    inference_edge_index, inference_edge_type = (
        model_module.apply_edge_mask(
            full_edge_index,
            full_edge_type,
            inference_edge_mask,
        )
    )

    require(
        fit_edge_index.shape[1] == 6,
        "Synthetic fit graph edge count changed.",
    )

    require(
        inference_edge_index.shape[1] == 8,
        "Synthetic inference graph edge count changed.",
    )

    require(
        not bool(
            torch.any(
                inference_edge_index[0]
                == 4
            )
        ),
        "COLD_t outgoing edge survived inference mask.",
    )

    require(
        bool(
            torch.any(
                inference_edge_index[1]
                == 4
            )
        ),
        "COLD_t incoming edge was not restored.",
    )

    print("Synthetic fit/inference graph semantics: PASS")

    family_results = {}

    common_hashes = {
        "investor_latent": {},
        "investor_description": {},
        "startup_content_fit": {},
    }

    graph_outputs = {}

    for family in (
        "B1a",
        "B1b",
        "M1",
        "M2",
    ):
        print()
        print("-" * 120)
        print(f"MODEL {family}")
        print("-" * 120)

        if family == "B1a":
            structured_dim = None

        elif family == "B1b":
            structured_dim = (
                dimensions[
                    "base_dim"
                ]
            )

        else:
            structured_dim = (
                dimensions[
                    "extended_dim"
                ]
            )

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
            scope="DEV_T57",
            namespace_lookup=namespace_lookup,
            reverse_order=False,
        )

        model.eval()

        data = make_inputs(
            category_dim=(
                dimensions[
                    "category_dim"
                ]
            ),
            structured_dim=structured_dim,
        )

        with torch.no_grad():

            # -----------------------------------------------------------------
            # Investor side.
            # -----------------------------------------------------------------

            investor_latent = (
                model
                .encode_investor_latent(
                    data[
                        "investor_indices"
                    ]
                )
            )

            investor_description = (
                model
                .encode_investor_description(
                    data[
                        "investor_doc2vec"
                    ],
                    data[
                        "investor_labels"
                    ],
                )
            )

            require(
                investor_latent.shape
                == (3, 40),
                f"{family}: Investor latent shape mismatch.",
            )

            require(
                investor_description.shape
                == (3, 40),
                f"{family}: Investor description shape mismatch.",
            )

            # -----------------------------------------------------------------
            # Startup content:
            # fit => row 1 must be exactly zero.
            # inference => all rows visible.
            # -----------------------------------------------------------------

            startup_content_fit = (
                model
                .encode_startup_content(
                    data[
                        "startup_text"
                    ],
                    data[
                        "startup_categories"
                    ],
                    visibility_mask=(
                        data[
                            "fit_visibility"
                        ]
                    ),
                )
            )

            startup_content_inference = (
                model
                .encode_startup_content(
                    data[
                        "startup_text"
                    ],
                    data[
                        "startup_categories"
                    ],
                    visibility_mask=(
                        data[
                            "inference_visibility"
                        ]
                    ),
                )
            )

            require_zero_row(
                startup_content_fit,
                1,
                f"{family} fit Startup content",
            )

            require(
                startup_content_fit.shape
                == (3, 40),
                f"{family}: Startup content shape mismatch.",
            )

            # -----------------------------------------------------------------
            # Startup structured branch.
            # -----------------------------------------------------------------

            startup_structured_fit = (
                model
                .encode_startup_structured(
                    startup_content_40=(
                        startup_content_fit
                    ),
                    structured_input=(
                        data[
                            "structured_input"
                        ]
                    ),
                    visibility_mask=(
                        data[
                            "fit_visibility"
                        ]
                    ),
                )
            )

            startup_structured_inference = (
                model
                .encode_startup_structured(
                    startup_content_40=(
                        startup_content_inference
                    ),
                    structured_input=(
                        data[
                            "structured_input"
                        ]
                    ),
                    visibility_mask=(
                        data[
                            "inference_visibility"
                        ]
                    ),
                )
            )

            require(
                startup_structured_fit.shape
                == (3, 40),
                f"{family}: structured shape mismatch.",
            )

            if family == "B1a":

                require(
                    bool(
                        torch.count_nonzero(
                            startup_structured_fit
                        )
                        == 0
                    ),
                    "B1a structured slot is not ZERO_40.",
                )

                require(
                    bool(
                        torch.count_nonzero(
                            startup_structured_inference
                        )
                        == 0
                    ),
                    "B1a inference structured slot is not ZERO_40.",
                )

            else:

                require_zero_row(
                    startup_structured_fit,
                    1,
                    f"{family} fit Startup structured",
                )

            # -----------------------------------------------------------------
            # Trend branch.
            # -----------------------------------------------------------------

            if (
                model
                .family_spec
                .trend_active
            ):

                trend_query = (
                    model
                    .compose_trend_query(
                        investor_latent,
                        investor_description,
                    )
                )

                require(
                    trend_query.shape
                    == (3, 80),
                    f"{family}: trend query shape mismatch.",
                )

                # Historical items are visible/warm by construction.
                history_items = (
                    model
                    .compose_history_items(
                        startup_structured_inference,
                        startup_content_inference,
                    )
                )

                require(
                    history_items.shape
                    == (3, 80),
                    f"{family}: trend history item shape mismatch.",
                )

                period_vector, alpha = (
                    model
                    .trend_extractor
                    .attend_period(
                        trend_query[0],
                        history_items,
                    )
                )

                require(
                    period_vector.shape
                    == (80,),
                    f"{family}: period vector shape mismatch.",
                )

                require(
                    alpha.shape
                    == (3,),
                    f"{family}: attention shape mismatch.",
                )

                require(
                    torch.allclose(
                        alpha.sum(),
                        torch.tensor(
                            1.0,
                            dtype=alpha.dtype,
                        ),
                        atol=1e-6,
                        rtol=0.0,
                    ),
                    f"{family}: attention does not sum to one.",
                )

                sequence = torch.stack(
                    [
                        period_vector,
                        torch.zeros_like(
                            period_vector
                        ),
                        period_vector * 0.5,
                    ],
                    dim=0,
                ).unsqueeze(0)

                trend_single, gru_output = (
                    model
                    .encode_trend_sequence(
                        sequence
                    )
                )

                require(
                    trend_single.shape
                    == (1, 40),
                    f"{family}: trend output shape mismatch.",
                )

                require(
                    gru_output.shape
                    == (1, 3, 40),
                    f"{family}: GRU output shape mismatch.",
                )

                investor_trend = (
                    trend_single
                    .repeat(
                        3,
                        1,
                    )
                )

            else:

                require(
                    family == "M1",
                    "Unexpected trend-inactive family.",
                )

                investor_trend = (
                    model
                    .resolve_trend_slot(
                        investor_latent,
                        None,
                    )
                )

                require(
                    bool(
                        torch.count_nonzero(
                            investor_trend
                        )
                        == 0
                    ),
                    "M1 trend slot is not ZERO_40.",
                )

            # -----------------------------------------------------------------
            # Graph branch.
            # -----------------------------------------------------------------

            if (
                model
                .family_spec
                .graph_active
            ):

                startup_graph_fit = (
                    model
                    .encode_startup_graph_input(
                        startup_content_fit,
                        startup_structured_fit,
                        visibility_mask=(
                            data[
                                "fit_visibility"
                            ]
                        ),
                    )
                )

                startup_graph_inference = (
                    model
                    .encode_startup_graph_input(
                        startup_content_inference,
                        startup_structured_inference,
                        visibility_mask=(
                            data[
                                "inference_visibility"
                            ]
                        ),
                    )
                )

                require_zero_row(
                    startup_graph_fit,
                    1,
                    f"{family} fit Startup graph input",
                )

                person_zero = torch.zeros(
                    1,
                    40,
                    dtype=torch.float32,
                )

                node_input_fit = torch.cat(
                    [
                        investor_latent,
                        startup_graph_fit,
                        person_zero,
                    ],
                    dim=0,
                )

                node_input_inference = torch.cat(
                    [
                        investor_latent,
                        startup_graph_inference,
                        person_zero,
                    ],
                    dim=0,
                )

                require(
                    node_input_fit.shape
                    == (7, 40),
                    f"{family}: fit graph input shape mismatch.",
                )

                # -------------------------------------------------------------
                # FIT graph:
                # cold node has zero input + no incident edges.
                # It must remain exactly zero after both R-GCN layers.
                # -------------------------------------------------------------

                graph_fit = (
                    model
                    .propagate_graph(
                        node_input_fit,
                        fit_edge_index,
                        fit_edge_type,
                    )
                )

                require(
                    graph_fit["h2"].shape
                    == (7, 40),
                    f"{family}: graph output shape mismatch.",
                )

                require_zero_row(
                    graph_fit["h2"],
                    4,
                    f"{family} fit COLD_t structural output",
                )

                # -------------------------------------------------------------
                # INFERENCE sink-only invariance:
                #
                # Restore COLD features and incoming edges.
                # Compare against same inference graph with COLD input zeroed.
                #
                # Because COLD has no outgoing edges, every NON-COLD node
                # must remain bitwise identical.
                # -------------------------------------------------------------

                graph_inference_restored = (
                    model
                    .propagate_graph(
                        node_input_inference,
                        inference_edge_index,
                        inference_edge_type,
                    )
                )

                node_input_inference_zero_cold = (
                    node_input_inference
                    .clone()
                )

                node_input_inference_zero_cold[
                    4
                ] = 0.0

                graph_inference_zero_cold = (
                    model
                    .propagate_graph(
                        node_input_inference_zero_cold,
                        inference_edge_index,
                        inference_edge_type,
                    )
                )

                noncold_indices = torch.tensor(
                    [
                        0,
                        1,
                        2,
                        3,
                        5,
                        6,
                    ],
                    dtype=torch.long,
                )

                require_same_tensor(
                    graph_inference_restored[
                        "h2"
                    ][
                        noncold_indices
                    ],
                    graph_inference_zero_cold[
                        "h2"
                    ][
                        noncold_indices
                    ],
                    (
                        f"{family} sink-only non-COLD "
                        "structural representations"
                    ),
                )

                investor_structural = (
                    graph_fit[
                        "h2"
                    ][
                        0:3
                    ]
                )

                startup_structural = (
                    graph_fit[
                        "h2"
                    ][
                        3:6
                    ]
                )

                require_zero_row(
                    startup_structural,
                    1,
                    f"{family} pair Startup structural fit slot",
                )

                graph_outputs[
                    family
                ] = {
                    "fit_h2":
                        graph_fit[
                            "h2"
                        ].clone(),

                    "inference_restored_h2":
                        graph_inference_restored[
                            "h2"
                        ].clone(),
                }

            else:

                investor_structural = (
                    model
                    .resolve_investor_structural_slot(
                        investor_latent,
                        None,
                    )
                )

                startup_structural = (
                    model
                    .resolve_startup_structural_slot(
                        startup_content_fit,
                        None,
                    )
                )

                require(
                    bool(
                        torch.count_nonzero(
                            investor_structural
                        )
                        == 0
                    ),
                    f"{family}: Investor graph slot not ZERO.",
                )

                require(
                    bool(
                        torch.count_nonzero(
                            startup_structural
                        )
                        == 0
                    ),
                    f"{family}: Startup graph slot not ZERO.",
                )

            # -----------------------------------------------------------------
            # Resolve trend slot for trend-active models too.
            # -----------------------------------------------------------------

            if (
                model
                .family_spec
                .trend_active
            ):

                investor_trend = (
                    model
                    .resolve_trend_slot(
                        investor_latent,
                        investor_trend,
                    )
                )

            # -----------------------------------------------------------------
            # Pair insertion.
            # -----------------------------------------------------------------

            pair = (
                model
                .assemble_pair(
                    investor_trend_40=(
                        investor_trend
                    ),
                    investor_latent_40=(
                        investor_latent
                    ),
                    investor_description_40=(
                        investor_description
                    ),
                    investor_structural_40=(
                        investor_structural
                    ),
                    startup_structured_40=(
                        startup_structured_fit
                    ),
                    startup_content_40=(
                        startup_content_fit
                    ),
                    startup_structural_40=(
                        startup_structural
                    ),
                )
            )

            require(
                pair.shape
                == (3, 280),
                f"{family}: pair shape mismatch.",
            )

            expected_slots = [
                investor_trend,
                investor_latent,
                investor_description,
                investor_structural,
                startup_structured_fit,
                startup_content_fit,
                startup_structural,
            ]

            for slot_idx, expected_tensor in enumerate(
                expected_slots
            ):

                start = (
                    slot_idx
                    * 40
                )

                end = (
                    start
                    + 40
                )

                require_same_tensor(
                    pair[
                        :,
                        start:end
                    ],
                    expected_tensor,
                    (
                        f"{family} pair slot "
                        f"{slot_idx}"
                    ),
                )

            logits = (
                model
                .score_pair(
                    pair
                )
            )

            require(
                logits.shape
                == (3, 1),
                f"{family}: scorer output shape mismatch.",
            )

            require_finite(
                logits,
                f"{family} logits",
            )

        # ---------------------------------------------------------------------
        # Common-component hashes.
        # ---------------------------------------------------------------------

        common_hashes[
            "investor_latent"
        ][family] = tensor_hash(
            investor_latent
        )

        common_hashes[
            "investor_description"
        ][family] = tensor_hash(
            investor_description
        )

        common_hashes[
            "startup_content_fit"
        ][family] = tensor_hash(
            startup_content_fit
        )

        family_results[
            family
        ] = {
            "pair_shape":
                list(
                    pair.shape
                ),

            "logit_shape":
                list(
                    logits.shape
                ),

            "pair_sha256":
                tensor_hash(
                    pair
                ),

            "logit_sha256":
                tensor_hash(
                    logits
                ),

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

            "cold_content_fit_zero":
                True,

            "cold_structured_fit_zero":
                bool(
                    torch.count_nonzero(
                        startup_structured_fit[
                            1
                        ]
                    )
                    == 0
                ),

            "cold_structural_fit_zero":
                bool(
                    torch.count_nonzero(
                        startup_structural[
                            1
                        ]
                    )
                    == 0
                ),
        }

        print("Canonical initialization:              PASS")
        print("Investor latent/description:            PASS")
        print("Startup content insertion:              PASS")
        print("COLD_t fit feature zeroing:             PASS")
        print("Trend branch semantics:                 PASS")

        if (
            model
            .family_spec
            .graph_active
        ):
            print("Fit graph cold isolation:              PASS")
            print("Inference sink-only invariance:         PASS")
        else:
            print("No-graph ZERO_40 slots:                PASS")

        print("280-D pair slot insertion:              PASS")
        print("Scorer forward [3,280] -> [3,1]:       PASS")

        del model
        gc.collect()

    # =========================================================================
    # CROSS-FAMILY SHARED COMPONENT OUTPUTS
    # =========================================================================

    for component, family_hashes in common_hashes.items():

        require(
            len(
                set(
                    family_hashes.values()
                )
            )
            == 1,
            (
                "Shared initialized component produced "
                "different output across families: "
                f"{component}: {family_hashes}"
            ),
        )

    require(
        "M1" in graph_outputs
        and "M2" in graph_outputs,
        "Missing M1/M2 graph outputs.",
    )

    require_same_tensor(
        graph_outputs[
            "M1"
        ][
            "fit_h2"
        ],
        graph_outputs[
            "M2"
        ][
            "fit_h2"
        ],
        "M1/M2 shared fit heterogeneous graph",
    )

    require_same_tensor(
        graph_outputs[
            "M1"
        ][
            "inference_restored_h2"
        ],
        graph_outputs[
            "M2"
        ][
            "inference_restored_h2"
        ],
        "M1/M2 shared inference heterogeneous graph",
    )

    print()
    print("-" * 120)
    print("CROSS-FAMILY INVARIANTS")
    print("-" * 120)
    print("Investor latent B1a/B1b/M1/M2:         BITWISE MATCH")
    print("Investor description B1a/B1b/M1/M2:    BITWISE MATCH")
    print("Startup content B1a/B1b/M1/M2:         BITWISE MATCH")
    print("M1/M2 heterogeneous graph fit output:   BITWISE MATCH")
    print("M1/M2 heterogeneous graph inference:    BITWISE MATCH")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    summary = {
        "schema_version":
            "PHASE_8_2_5_CANONICAL_FORWARD_BRANCH_INSERTION_AUDIT_V1",

        "phase":
            "8.2.5",

        "status":
            "FORWARD_AND_BRANCH_INSERTION_AUDITED",

        "scope":
            "DEV_T57",

        "data_role":
            "SYNTHETIC_CONTROLLED_MICROCASE_ONLY",

        "real_evaluation_cases_loaded":
            False,

        "candidate_matrix_loaded":
            False,

        "labels_loaded":
            False,

        "performance_metric_computed":
            False,

        "family_results":
            family_results,

        "graph_audit": {
            "typed_relation_ids_present":
                10,

            "fit_edges":
                6,

            "inference_edges":
                8,

            "cold_node":
                4,

            "fit_all_incident_edges_masked":
                True,

            "inference_incoming_restored":
                True,

            "inference_outgoing_suppressed":
                True,

            "sink_only_noncold_invariance":
                True,
        },

        "pair_audit": {
            "dimension":
                280,

            "slot_width":
                40,

            "exact_slot_order_verified":
                True,

            "scorer_output_dimension":
                1,
        },

        "cross_family_invariants": {
            "investor_latent":
                "BITWISE_MATCH",

            "investor_description":
                "BITWISE_MATCH",

            "startup_content":
                "BITWISE_MATCH",

            "M1_M2_graph_fit":
                "BITWISE_MATCH",

            "M1_M2_graph_inference":
                "BITWISE_MATCH",
        },

        "execution_state": {
            "canonical_initialization_applied":
                True,

            "forward_executed":
                True,

            "graph_forward_executed":
                True,

            "synthetic_logits_computed":
                True,

            "checkpoint_written":
                False,

            "optimizer_created":
                False,

            "backward_executed":
                False,

            "training_performed":
                False,

            "performance_inspected":
                False,

            "T60_scored":
                False,

            "April_analyzed":
                False,
        },

        "training_allowed_now":
            False,

        "next_step":
            (
                "Phase 8.2.6 — training-stream dry run "
                "without optimizer/backward."
            ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    HASH_PATH.write_text(
        json.dumps(
            {
                str(MODEL_MODULE_PATH):
                    sha256_file(
                        MODEL_MODULE_PATH
                    ),

                str(INITIALIZER_PATH):
                    sha256_file(
                        INITIALIZER_PATH
                    ),

                str(SUMMARY_PATH):
                    sha256_file(
                        SUMMARY_PATH
                    ),
            },
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("PHASE 8.2.5 V1 — FORWARD / BRANCH INSERTION AUDIT READY")
    print("=" * 120)
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Hashes:  {HASH_PATH}")
    print()
    print("Canonical initialization:               YES")
    print("Synthetic forward executed:             YES")
    print("Graph forward executed:                 YES")
    print("Real evaluation cases loaded:           NO")
    print("Candidate matrix loaded:                NO")
    print("Performance metric computed:            NO")
    print("Checkpoint written:                     NO")
    print("Optimizer created:                      NO")
    print("Backward executed:                      NO")
    print("Training performed:                     NO")
    print("Performance inspected:                  NO")
    print("T60 scored:                             NO")
    print("April analyzed:                         NO")
    print("Training allowed now:                   NO")
    print("=" * 120)


if __name__ == "__main__":
    main()
