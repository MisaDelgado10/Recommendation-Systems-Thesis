from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PHASE 3.2.11 — ITRS -> CRUNCHBASE RELATION SUPPORT MATRIX
#                & PHASE-3.2 AUDIT CLOSURE
# =============================================================================
#
# PURPOSE
# -------
#
# Synthesize all Phase-3.2 audits into a definitive relationship-support
# matrix.
#
# NO raw source scan is performed.
# NO graph node or graph edge is materialized.
# NO new relationship is inferred.
#
# This script distinguishes:
#
#   - literal ITRS primitive relations,
#   - ITRS-principle-aligned derived relations,
#   - Crunchbase-specific adaptations,
#   - temporal investment interactions,
#   - separate feature sources,
#   - deferred / excluded sources.
#
# =============================================================================


AUDIT_DIR = Path(
    "data/experimental/phase_3/audits"
)

FIGURE_DIR = Path(
    "data/experimental/phase_3/figures"
)

HOLDOUT_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "t60_holdout_pair_manifest.parquet"
)


# =============================================================================
# INPUTS
# =============================================================================


SHARED_FOUNDER_SUMMARY_PATH = (
    AUDIT_DIR
    / "shared_founder_strict_candidate_summary.csv"
)

SHARED_FOUNDER_ROLE_PATH = (
    AUDIT_DIR
    / "shared_founder_strict_role_pair_summary.csv"
)

ACQUISITION_PAIR_PATH = (
    AUDIT_DIR
    / "acquisition_strict_underlying_pair_summary.csv"
)

ACQUISITION_ROLE_PATH = (
    AUDIT_DIR
    / "acquisition_strict_role_projection_summary.csv"
)

BRIDGE_PATH = (
    AUDIT_DIR
    / "cross_dataset_reference_bridge_summary.csv"
)

CONTACT_PATH = (
    AUDIT_DIR
    / "contact_reference_integrity_summary.csv"
)

SCHOOL_PATH = (
    AUDIT_DIR
    / "school_identity_namespace_overlap.csv"
)


# =============================================================================
# OUTPUTS
# =============================================================================


SUPPORT_MATRIX_OUTPUT = (
    AUDIT_DIR
    / "itrs_crunchbase_relation_support_matrix.csv"
)

SOURCE_DECISION_OUTPUT = (
    AUDIT_DIR
    / "crunchbase_relation_source_decision_matrix.csv"
)

CLOSURE_SUMMARY_OUTPUT = (
    AUDIT_DIR
    / "phase_3_2_closure_summary.csv"
)

HOLDOUT_MASK_OUTPUT = (
    AUDIT_DIR
    / "phase_3_2_underlying_holdout_mask_audit.csv"
)


RELATION_COUNT_FIGURE = (
    FIGURE_DIR
    / "phase_3_2_supported_relation_candidate_counts.png"
)

SUPPORT_STATUS_FIGURE = (
    FIGURE_DIR
    / "phase_3_2_relation_support_status_counts.png"
)


def separator(char="=", width=120):
    print(char * width)


def metric_value(
    dataframe,
    metric,
):

    row = dataframe[
        dataframe[
            "metric"
        ]
        == metric
    ]

    if len(row) != 1:

        raise ValueError(
            f"Expected one row for metric '{metric}', "
            f"found {len(row)}."
        )

    return row.iloc[0][
        "value"
    ]


def ordered_pair(
    value_a,
    value_b,
):

    return tuple(
        sorted(
            [
                str(value_a),
                str(value_b),
            ]
        )
    )


def main():

    separator()

    print(
        "PHASE 3.2.11 — "
        "ITRS -> CRUNCHBASE RELATION SUPPORT MATRIX "
        "& PHASE-3.2 AUDIT CLOSURE"
    )

    separator()

    # =========================================================================
    # 1. Load audited summaries
    # =========================================================================

    shared_summary = pd.read_csv(
        SHARED_FOUNDER_SUMMARY_PATH
    )

    shared_roles = pd.read_csv(
        SHARED_FOUNDER_ROLE_PATH
    )

    acquisition_summary = pd.read_csv(
        ACQUISITION_PAIR_PATH
    )

    acquisition_roles = pd.read_csv(
        ACQUISITION_ROLE_PATH
    )

    bridge_summary = pd.read_csv(
        BRIDGE_PATH
    )

    contact_summary = pd.read_csv(
        CONTACT_PATH
    )

    school_summary = pd.read_csv(
        SCHOOL_PATH
    )

    # =========================================================================
    # 2. Extract final audited counts
    # =========================================================================

    shared_founder_pairs = int(
        metric_value(
            shared_summary,
            "final_shared_founder_relation_candidates",
        )
    )

    shared_founder_directed_if_bidirectional = int(
        metric_value(
            shared_summary,
            "potential_directed_edges_if_stored_bidirectionally",
        )
    )

    shared_founder_holdout_removed = int(
        metric_value(
            shared_summary,
            "T60_holdout_role_pairs_removed",
        )
    )

    acquisition_underlying_pairs = int(
        metric_value(
            acquisition_summary,
            "unique_directed_underlying_pairs",
        )
    )

    acquisition_role_relations = int(
        metric_value(
            acquisition_summary,
            "potential_role_specific_directed_relations",
        )
    )

    # =========================================================================
    # 3. Integrity checks
    # =========================================================================

    if (
        shared_roles[
            "unique_role_pair_candidates"
        ]
        .sum()
        !=
        shared_founder_pairs
    ):

        raise ValueError(
            "Shared-founder role distribution does not "
            "sum to final candidate count."
        )

    if (
        acquisition_roles[
            "projected_directed_relation_candidates"
        ]
        .sum()
        !=
        acquisition_role_relations
    ):

        raise ValueError(
            "Acquisition role projection does not "
            "sum to audited projected total."
        )

    # =========================================================================
    # 4. Held-out directed-pair -> unordered underlying-pair audit
    # =========================================================================

    holdout = pd.read_parquet(
        HOLDOUT_PATH,
        columns=[
            "investor_id",
            "startup_id",
        ],
    )

    holdout[
        "investor_id"
    ] = (
        holdout[
            "investor_id"
        ]
        .astype("string")
        .str.strip()
    )

    holdout[
        "startup_id"
    ] = (
        holdout[
            "startup_id"
        ]
        .astype("string")
        .str.strip()
    )

    directed_pairs = set(
        zip(
            holdout[
                "investor_id"
            ],
            holdout[
                "startup_id"
            ],
        )
    )

    unordered_pairs = {
        ordered_pair(
            investor_id,
            startup_id,
        )
        for investor_id, startup_id
        in directed_pairs
    }

    reciprocal_directed_pairs = 0

    seen = set()

    for pair in directed_pairs:

        if pair in seen:
            continue

        reverse = (
            pair[1],
            pair[0],
        )

        if (
            reverse in directed_pairs
            and reverse != pair
        ):

            reciprocal_directed_pairs += 1

            seen.add(
                pair
            )

            seen.add(
                reverse
            )

    holdout_mask_audit = pd.DataFrame(
        [
            {
                "metric":
                    "T60_unique_directed_holdout_pairs",

                "value":
                    len(
                        directed_pairs
                    ),
            },

            {
                "metric":
                    "T60_unique_unordered_underlying_pairs",

                "value":
                    len(
                        unordered_pairs
                    ),
            },

            {
                "metric":
                    "directed_pair_reduction_after_unordered_canonicalization",

                "value":
                    (
                        len(
                            directed_pairs
                        )
                        -
                        len(
                            unordered_pairs
                        )
                    ),
            },

            {
                "metric":
                    "reciprocal_underlying_pair_groups",

                "value":
                    reciprocal_directed_pairs,
            },
        ]
    )

    # =========================================================================
    # 5. ITRS -> Crunchbase support matrix
    #
    # ITRS primitive relation names/counts were frozen in Phase 3.1
    # from the paper's Table I.
    # =========================================================================

    primitive_relations = [
        (
            "Ten major shareholders",
            "Company or human",
            25137,
        ),

        (
            "Shareholder",
            "Company or human",
            35984,
        ),

        (
            "Former shareholders",
            "Company or human",
            13238,
        ),

        (
            "Controlling",
            "Company",
            10014,
        ),

        (
            "Supplier",
            "Company",
            17932,
        ),

        (
            "Client",
            "Company",
            15981,
        ),

        (
            "Competitor",
            "Company",
            107767,
        ),

        (
            "Holding shares now",
            "Company",
            26254,
        ),

        (
            "Once held shares",
            "Company",
            12234,
        ),

        (
            "Business",
            "Brand",
            45238,
        ),
    ]

    support_rows = []

    unsupported_relations = {
        "Ten major shareholders",
        "Shareholder",
        "Former shareholders",
        "Controlling",
        "Supplier",
        "Client",
        "Competitor",
        "Holding shares now",
        "Once held shares",
    }

    for (
        relation_name,
        intermediary_type,
        reported_count,
    ) in primitive_relations:

        if (
            relation_name
            in unsupported_relations
        ):

            support_status = (
                "unsupported_in_available_crunchbase_exports"
            )

            crunchbase_evidence = (
                "No audited entity-level source field "
                "provides this relationship."
            )

            phase_3_3_decision = (
                "exclude"
            )

        else:

            # Business relation:
            # Crunchbase supports shared underlying organizational identity,
            # but this is used as provenance / role mapping rather than a
            # standalone structural graph edge.
            support_status = (
                "partially_supported_as_identity_mapping"
            )

            crunchbase_evidence = (
                "Startup role IDs map to companies.id; "
                "shared Crunchbase UUIDs preserve underlying "
                "entity identity across semantic roles."
            )

            phase_3_3_decision = (
                "identity_layer_not_relation_edge"
            )

        support_rows.append(
            {
                "itrs_relation_or_principle":
                    relation_name,

                "itrs_relation_class":
                    "published_primitive",

                "itrs_intermediary_or_target_type":
                    intermediary_type,

                "itrs_reported_relation_count":
                    reported_count,

                "crunchbase_relation":
                    "",

                "crunchbase_evidence":
                    crunchbase_evidence,

                "support_status":
                    support_status,

                "candidate_count":
                    np.nan,

                "temporal_provenance":
                    "",

                "leakage_policy":
                    "",

                "phase_3_3_decision":
                    phase_3_3_decision,
            }
        )

    # =========================================================================
    # 6. Add principle-level / adaptation rows
    # =========================================================================

    founder_bridge_row = (
        bridge_summary[
            bridge_summary[
                "bridge"
            ]
            ==
            "companies.founder_links -> people.link"
        ]
    )

    if len(
        founder_bridge_row
    ) != 1:

        raise ValueError(
            "Could not locate founder-link bridge audit."
        )

    founder_resolution_pct = float(
        founder_bridge_row.iloc[0][
            "coverage_pct"
        ]
    )

    support_rows.append(
        {
            "itrs_relation_or_principle":
                (
                    "Human-intermediary-derived relationship"
                ),

            "itrs_relation_class":
                "published_construction_principle",

            "itrs_intermediary_or_target_type":
                "Human",

            "itrs_reported_relation_count":
                np.nan,

            "crunchbase_relation":
                "SHARED_FOUNDER",

            "crunchbase_evidence":
                (
                    "companies.founder_links resolves to "
                    "people.link with "
                    f"{founder_resolution_pct:.4f}% exact coverage."
                ),

            "support_status":
                "supported_principle_aligned_derived_relation",

            "candidate_count":
                shared_founder_pairs,

            "temporal_provenance":
                "current_snapshot_unversioned",

            "leakage_policy":
                (
                    "Remove direct relation for any T60 "
                    "held-out underlying Investor-Startup pair."
                ),

            "phase_3_3_decision":
                "candidate_for_graph_schema",
        }
    )

    support_rows.append(
        {
            "itrs_relation_or_principle":
                "Company structural relationships",

            "itrs_relation_class":
                "crunchbase_adaptation",

            "itrs_intermediary_or_target_type":
                "Company",

            "itrs_reported_relation_count":
                np.nan,

            "crunchbase_relation":
                "ACQUIRED",

            "crunchbase_evidence":
                (
                    f"{acquisition_underlying_pairs:,} "
                    "leakage-safe unique directed underlying "
                    "acquisition pairs; "
                    f"{acquisition_role_relations:,} "
                    "potential role-specific directed relations."
                ),

            "support_status":
                "supported_crunchbase_adaptation",

            "candidate_count":
                acquisition_role_relations,

            "temporal_provenance":
                "timestamped_pre_T60",

            "leakage_policy":
                (
                    "Use only announced_on < 2026-01-01 "
                    "and remove T60 held-out underlying pairs."
                ),

            "phase_3_3_decision":
                "candidate_for_graph_schema",
        }
    )

    support_rows.append(
        {
            "itrs_relation_or_principle":
                "Investment event E=(investor,startup,time)",

            "itrs_relation_class":
                "temporal_interaction_layer",

            "itrs_intermediary_or_target_type":
                "Investor -> Startup",

            "itrs_reported_relation_count":
                np.nan,

            "crunchbase_relation":
                "INVESTMENT_EVENT",

            "crunchbase_evidence":
                (
                    "Canonical Phase-1 interactions and "
                    "frozen Phase-2 temporal segmentation."
                ),

            "support_status":
                "supported_but_not_structural_RGCN_relation",

            "candidate_count":
                np.nan,

            "temporal_provenance":
                "event_timestamped",

            "leakage_policy":
                "Phase-2 temporal split remains authoritative.",

            "phase_3_3_decision":
                "keep_separate_from_structural_graph",
        }
    )

    support_rows.append(
        {
            "itrs_relation_or_principle":
                "Description / label information",

            "itrs_relation_class":
                "feature_layer",

            "itrs_intermediary_or_target_type":
                "Investor / Startup attributes",

            "itrs_reported_relation_count":
                np.nan,

            "crunchbase_relation":
                "",

            "crunchbase_evidence":
                (
                    "Descriptions, categories, category groups "
                    "and related attributes are available."
                ),

            "support_status":
                "supported_as_features_not_relations",

            "candidate_count":
                np.nan,

            "temporal_provenance":
                "current_snapshot_mixed",

            "leakage_policy":
                "Feature-time validity must be treated separately.",

            "phase_3_3_decision":
                "feature_layer_not_structural_edge",
        }
    )

    support_matrix = pd.DataFrame(
        support_rows
    )

    # =========================================================================
    # 7. Crunchbase auxiliary-source decision matrix
    # =========================================================================

    contacts_people_overlap = int(
        metric_value(
            contact_summary,
            "contact_ids_found_in_people",
        )
    )

    contact_org_uuid_nonmissing = int(
        metric_value(
            contact_summary,
            "organization_uuid_nonmissing_rows",
        )
    )

    source_decisions = pd.DataFrame(
        [
            {
                "source":
                    "companies.founder_links + people.link",

                "candidate_relation":
                    "SHARED_FOUNDER",

                "status":
                    "supported",

                "reason":
                    (
                        f"Exact Person bridge coverage "
                        f"{founder_resolution_pct:.4f}%."
                    ),

                "phase_3_3_action":
                    "evaluate_for_core_graph",
            },

            {
                "source":
                    "acquisitions",

                "candidate_relation":
                    "ACQUIRED",

                "status":
                    "supported_adaptation",

                "reason":
                    (
                        "Conservative endpoint resolution, "
                        "explicit announced_on timestamp, "
                        "canonical-role filtering and leakage masking."
                    ),

                "phase_3_3_action":
                    "evaluate_for_core_graph",
            },

            {
                "source":
                    "people.schools + school registry",

                "candidate_relation":
                    "SHARED_SCHOOL",

                "status":
                    "deferred",

                "reason":
                    (
                        "People stores school names rather than IDs; "
                        "safe canonical name-list segmentation has "
                        "not been implemented."
                    ),

                "phase_3_3_action":
                    "exclude_from_initial_graph",
            },

            {
                "source":
                    "contacts",

                "candidate_relation":
                    "EMPLOYMENT_OR_CONTACT",

                "status":
                    "excluded_strict",

                "reason":
                    (
                        f"contacts.id -> people.id overlap = "
                        f"{contacts_people_overlap}; "
                        f"organization_uuid populated rows = "
                        f"{contact_org_uuid_nonmissing}."
                    ),

                "phase_3_3_action":
                    "exclude",
            },

            {
                "source":
                    "hubs",

                "candidate_relation":
                    "HUB_MEMBERSHIP",

                "status":
                    "unsupported",

                "reason":
                    (
                        "Hub export contains aggregates but no "
                        "audited entity-level membership references."
                    ),

                "phase_3_3_action":
                    "exclude",
            },

            {
                "source":
                    "events",

                "candidate_relation":
                    "EVENT_PARTICIPATION",

                "status":
                    "unsupported",

                "reason":
                    (
                        "Event export contains aggregate participant "
                        "counts but no audited participant references."
                    ),

                "phase_3_3_action":
                    "exclude",
            },

            {
                "source":
                    "funding / investment-derived fields",

                "candidate_relation":
                    "INVESTED_IN",

                "status":
                    "separate_interaction_layer",

                "reason":
                    (
                        "Investment activity is the recommendation "
                        "target and temporal interaction sequence."
                    ),

                "phase_3_3_action":
                    "do_not_use_as_structural_edge",
            },

            {
                "source":
                    "categories / category_groups",

                "candidate_relation":
                    "SAME_CATEGORY",

                "status":
                    "extension_only",

                "reason":
                    (
                        "Better aligned with description/attribute "
                        "features; pair projection can create broad "
                        "similarity cliques."
                    ),

                "phase_3_3_action":
                    "exclude_from_initial_graph",
            },

            {
                "source":
                    "locations / city / country",

                "candidate_relation":
                    "SAME_LOCATION",

                "status":
                    "extension_only",

                "reason":
                    (
                        "Attribute similarity rather than observed "
                        "business relationship; high-cardinality "
                        "projection risks large artificial cliques."
                    ),

                "phase_3_3_action":
                    "exclude_from_initial_graph",
            },
        ]
    )

    # =========================================================================
    # 8. Phase-3.2 closure summary
    # =========================================================================

    closure_summary = pd.DataFrame(
        [
            {
                "metric":
                    "canonical_investor_nodes",

                "value":
                    165975,
            },

            {
                "metric":
                    "canonical_startup_nodes",

                "value":
                    311589,
            },

            {
                "metric":
                    "shared_founder_unique_symmetric_candidates",

                "value":
                    shared_founder_pairs,
            },

            {
                "metric":
                    "shared_founder_potential_directed_edges",

                "value":
                    shared_founder_directed_if_bidirectional,
            },

            {
                "metric":
                    "shared_founder_holdout_pairs_removed",

                "value":
                    shared_founder_holdout_removed,
            },

            {
                "metric":
                    "acquisition_unique_directed_underlying_pairs",

                "value":
                    acquisition_underlying_pairs,
            },

            {
                "metric":
                    "acquisition_potential_role_specific_directed_relations",

                "value":
                    acquisition_role_relations,
            },

            {
                "metric":
                    "supported_structural_relation_candidates",

                "value":
                    2,
            },

            {
                "metric":
                    "T60_unique_directed_holdout_pairs",

                "value":
                    len(
                        directed_pairs
                    ),
            },

            {
                "metric":
                    "T60_unique_unordered_underlying_mask_pairs",

                "value":
                    len(
                        unordered_pairs
                    ),
            },
        ]
    )

    # =========================================================================
    # 9. Save
    # =========================================================================

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    support_matrix.to_csv(
        SUPPORT_MATRIX_OUTPUT,
        index=False,
    )

    source_decisions.to_csv(
        SOURCE_DECISION_OUTPUT,
        index=False,
    )

    closure_summary.to_csv(
        CLOSURE_SUMMARY_OUTPUT,
        index=False,
    )

    holdout_mask_audit.to_csv(
        HOLDOUT_MASK_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 10. Figures
    # =========================================================================

    relation_plot = pd.DataFrame(
        [
            {
                "relation":
                    "SHARED_FOUNDER\n(if bidirectional)",

                "potential_directed_edge_records":
                    shared_founder_directed_if_bidirectional,
            },

            {
                "relation":
                    "ACQUIRED\n(role projected)",

                "potential_directed_edge_records":
                    acquisition_role_relations,
            },
        ]
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        relation_plot[
            "relation"
        ],
        relation_plot[
            "potential_directed_edge_records"
        ],
    )

    ax.set_ylabel(
        "Potential directed edge records"
    )

    ax.set_title(
        "Audited Structural Relation Candidates"
    )

    fig.tight_layout()

    fig.savefig(
        RELATION_COUNT_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # -------------------------------------------------------------------------

    status_counts = (
        support_matrix[
            "support_status"
        ]
        .value_counts()
        .rename_axis(
            "support_status"
        )
        .reset_index(
            name="row_count"
        )
        .sort_values(
            "row_count",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        status_counts[
            "support_status"
        ],
        status_counts[
            "row_count"
        ],
    )

    ax.set_xlabel(
        "Relation/principle entries"
    )

    ax.set_title(
        "ITRS-to-Crunchbase Relation Support Status"
    )

    fig.tight_layout()

    fig.savefig(
        SUPPORT_STATUS_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =========================================================================
    # 11. Terminal output
    # =========================================================================

    separator("-")

    print(
        "PHASE-3.2 CLOSURE SUMMARY"
    )

    separator("-")

    print(
        closure_summary.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "T60 UNDERLYING HOLDOUT MASK AUDIT"
    )

    separator("-")

    print(
        holdout_mask_audit.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "ITRS -> CRUNCHBASE RELATION SUPPORT MATRIX"
    )

    separator("-")

    print(
        support_matrix[
            [
                "itrs_relation_or_principle",
                "itrs_relation_class",
                "crunchbase_relation",
                "support_status",
                "candidate_count",
                "phase_3_3_decision",
            ]
        ]
        .to_string(
            index=False
        )
    )

    separator("-")

    print(
        "CRUNCHBASE SOURCE DECISION MATRIX"
    )

    separator("-")

    print(
        source_decisions.to_string(
            index=False
        )
    )

    separator()

    print(
        "PHASE 3.2.11 COMPLETE"
    )

    separator()

    print(
        f"""
Outputs written to:

{SUPPORT_MATRIX_OUTPUT}
{SOURCE_DECISION_OUTPUT}
{CLOSURE_SUMMARY_OUTPUT}
{HOLDOUT_MASK_OUTPUT}

Figures written to:

{RELATION_COUNT_FIGURE}
{SUPPORT_STATUS_FIGURE}


PHASE 3.2 STATUS:

    READY FOR CLOSURE

No final graph has been materialized.

NEXT:

    Phase 3.3 — Experimental Graph-Schema Selection

The next phase will decide:

1. whether SHARED_FOUNDER enters the initial graph;
2. whether ACQUIRED enters the initial graph;
3. exact relation directionality;
4. whether reverse relation IDs are required;
5. node-ID representation;
6. relation-ID vocabulary;
7. static graph snapshot policy;
8. final graph edge-table schema.
"""
    )


if __name__ == "__main__":
    main()