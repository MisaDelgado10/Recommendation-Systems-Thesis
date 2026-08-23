from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PHASE 3.2.9 — CONSERVATIVE ACQUISITION ACCEPTANCE &
#               LEAKAGE-SAFE CANDIDATE AUDIT
# =============================================================================
#
# PURPOSE
# -------
#
# Apply the acquisition-relation acceptance rules decided after Phase 3.2.8.
#
# This script:
#
#   1. restricts endpoint resolution to accepted evidence statuses;
#   2. uses only acquisitions observable before T60;
#   3. removes same-underlying-entity acquisitions;
#   4. requires both endpoints to belong to the canonical Investor/Startup
#      role universe;
#   5. removes any underlying entity pair that corresponds to a T60 held-out
#      Investor–Startup target;
#   6. deduplicates repeated acquisition events into underlying directed
#      relation candidates;
#   7. estimates eventual role-specific edge counts.
#
# NO final heterogeneous graph is created.
# =============================================================================


RESOLVED_PATH = Path(
    "data/experimental/phase_3/audits/"
    "acquisition_resolved_relation_candidates.parquet"
)

INTERACTIONS_PATH = Path(
    "data/processed/interactions.parquet"
)

HOLDOUT_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "t60_holdout_pair_manifest.parquet"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_3/audits"
)

FIGURE_DIR = Path(
    "data/experimental/phase_3/figures"
)


WATERFALL_OUTPUT = (
    OUTPUT_DIR
    / "acquisition_strict_filter_waterfall.csv"
)

ROLE_OUTPUT = (
    OUTPUT_DIR
    / "acquisition_strict_role_projection_summary.csv"
)

HOLDOUT_OUTPUT = (
    OUTPUT_DIR
    / "acquisition_strict_holdout_mask_summary.csv"
)

PAIR_OUTPUT = (
    OUTPUT_DIR
    / "acquisition_strict_underlying_pair_summary.csv"
)

CANDIDATES_OUTPUT = (
    OUTPUT_DIR
    / "acquisition_strict_underlying_relation_candidates.parquet"
)


WATERFALL_FIGURE = (
    FIGURE_DIR
    / "acquisition_strict_filter_waterfall.png"
)

ROLE_FIGURE = (
    FIGURE_DIR
    / "acquisition_strict_role_projection.png"
)


T60_START = pd.Timestamp(
    "2026-01-01"
)


ACCEPTED_ENDPOINT_STATUSES = {
    "name_website_agree",
    "ambiguous_name_resolved_by_website",
    "unique_name_only",
}


def separator(char="=", width=120):
    print(char * width)


def pct(num, den):

    if den == 0:
        return np.nan

    return (
        num
        / den
        * 100
    )


def clean_string(series):

    return (
        series
        .astype("string")
        .str.strip()
    )


def role_memberships(
    entity_id,
    investor_ids,
    startup_ids,
):

    roles = []

    if entity_id in investor_ids:

        roles.append(
            "investor"
        )

    if entity_id in startup_ids:

        roles.append(
            "startup"
        )

    return roles


def role_label(
    entity_id,
    investor_ids,
    startup_ids,
):

    roles = role_memberships(
        entity_id,
        investor_ids,
        startup_ids,
    )

    if len(roles) == 2:
        return "investor+startup"

    if len(roles) == 1:
        return roles[0]

    return "outside_canonical_roles"


def main():

    separator()

    print(
        "PHASE 3.2.9 — "
        "CONSERVATIVE ACQUISITION ACCEPTANCE & "
        "LEAKAGE-SAFE CANDIDATE AUDIT"
    )

    separator()

    # =========================================================================
    # 1. Load resolved acquisition candidates
    # =========================================================================

    resolved = pd.read_parquet(
        RESOLVED_PATH
    )

    required = {
        "id",
        "announced_on",
        "acquirer_resolved_id",
        "acquiree_resolved_id",
        "acquirer_resolution_status",
        "acquiree_resolution_status",
    }

    missing = (
        required
        - set(
            resolved.columns
        )
    )

    if missing:

        raise ValueError(
            f"Resolved acquisition file missing columns: "
            f"{sorted(missing)}"
        )

    resolved[
        "announced_on"
    ] = pd.to_datetime(
        resolved[
            "announced_on"
        ],
        errors="coerce",
    )

    resolved[
        "acquirer_resolved_id"
    ] = clean_string(
        resolved[
            "acquirer_resolved_id"
        ]
    )

    resolved[
        "acquiree_resolved_id"
    ] = clean_string(
        resolved[
            "acquiree_resolved_id"
        ]
    )

    # =========================================================================
    # 2. Canonical role universes
    # =========================================================================

    interactions = pd.read_parquet(
        INTERACTIONS_PATH,
        columns=[
            "investor_id",
            "startup_id",
        ],
    )

    investor_ids = set(
        clean_string(
            interactions[
                "investor_id"
            ]
        )
        .dropna()
        .unique()
    )

    startup_ids = set(
        clean_string(
            interactions[
                "startup_id"
            ]
        )
        .dropna()
        .unique()
    )

    canonical_ids = (
        investor_ids
        |
        startup_ids
    )

    print(
        f"\nCanonical investors: "
        f"{len(investor_ids):,}"
    )

    print(
        f"Canonical startups:  "
        f"{len(startup_ids):,}"
    )

    # =========================================================================
    # 3. T60 held-out target pairs
    # =========================================================================

    holdout = pd.read_parquet(
        HOLDOUT_PATH
    )

    required_holdout = {
        "investor_id",
        "startup_id",
    }

    if not (
        required_holdout
        .issubset(
            holdout.columns
        )
    ):

        raise ValueError(
            "T60 manifest does not contain "
            "investor_id and startup_id."
        )

    holdout[
        "investor_id"
    ] = clean_string(
        holdout[
            "investor_id"
        ]
    )

    holdout[
        "startup_id"
    ] = clean_string(
        holdout[
            "startup_id"
        ]
    )

    holdout_pairs = set(
        zip(
            holdout[
                "investor_id"
            ],
            holdout[
                "startup_id"
            ],
        )
    )

    print(
        f"T60 held-out unique pairs: "
        f"{len(holdout_pairs):,}"
    )

    # =========================================================================
    # 4. Filtering waterfall
    # =========================================================================

    waterfall_rows = []

    initial_count = len(
        resolved
    )

    def record_stage(
        stage_number,
        stage,
        dataframe,
        previous_count,
    ):

        count = len(
            dataframe
        )

        waterfall_rows.append(
            {
                "stage_number":
                    stage_number,

                "stage":
                    stage,

                "rows_remaining":
                    count,

                "rows_removed_from_previous":
                    (
                        previous_count
                        - count
                    ),

                "retained_from_previous_pct":
                    pct(
                        count,
                        previous_count,
                    ),

                "retained_from_initial_pct":
                    pct(
                        count,
                        initial_count,
                    ),
            }
        )

        return count

    work = (
        resolved.copy()
    )

    previous_count = (
        record_stage(
            0,
            "fully_resolved_input",
            work,
            len(
                work
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Accepted endpoint evidence
    # -------------------------------------------------------------------------

    work = (
        work[
            work[
                "acquirer_resolution_status"
            ]
            .isin(
                ACCEPTED_ENDPOINT_STATUSES
            )
            &
            work[
                "acquiree_resolution_status"
            ]
            .isin(
                ACCEPTED_ENDPOINT_STATUSES
            )
        ]
        .copy()
    )

    previous_count = (
        record_stage(
            1,
            "accepted_endpoint_evidence",
            work,
            previous_count,
        )
    )

    # -------------------------------------------------------------------------
    # Available before T60
    # -------------------------------------------------------------------------

    work = (
        work[
            work[
                "announced_on"
            ]
            .notna()
            &
            (
                work[
                    "announced_on"
                ]
                < T60_START
            )
        ]
        .copy()
    )

    previous_count = (
        record_stage(
            2,
            "observable_before_T60",
            work,
            previous_count,
        )
    )

    # -------------------------------------------------------------------------
    # Remove underlying self-relations
    # -------------------------------------------------------------------------

    work = (
        work[
            ~work[
                "acquirer_resolved_id"
            ]
            .eq(
                work[
                    "acquiree_resolved_id"
                ]
            )
        ]
        .copy()
    )

    previous_count = (
        record_stage(
            3,
            "remove_self_acquisitions",
            work,
            previous_count,
        )
    )

    # -------------------------------------------------------------------------
    # Require both endpoints in final canonical node universe
    # -------------------------------------------------------------------------

    work = (
        work[
            work[
                "acquirer_resolved_id"
            ]
            .isin(
                canonical_ids
            )
            &
            work[
                "acquiree_resolved_id"
            ]
            .isin(
                canonical_ids
            )
        ]
        .copy()
    )

    previous_count = (
        record_stage(
            4,
            "both_endpoints_in_canonical_roles",
            work,
            previous_count,
        )
    )

    # =========================================================================
    # 5. Held-out target-pair masking
    #
    # Conservative rule:
    #
    # If the underlying entity pair corresponds to a held-out
    # Investor -> Startup pair in either acquisition direction,
    # remove the direct structural relationship entirely.
    # =========================================================================

    direct_holdout = []

    reverse_holdout = []

    for acquirer_id, acquiree_id in zip(
        work[
            "acquirer_resolved_id"
        ],
        work[
            "acquiree_resolved_id"
        ],
    ):

        direct_holdout.append(
            (
                acquirer_id,
                acquiree_id,
            )
            in holdout_pairs
        )

        reverse_holdout.append(
            (
                acquiree_id,
                acquirer_id,
            )
            in holdout_pairs
        )

    work[
        "matches_holdout_pair_direct"
    ] = direct_holdout

    work[
        "matches_holdout_pair_reverse"
    ] = reverse_holdout

    work[
        "matches_any_holdout_underlying_pair"
    ] = (
        work[
            "matches_holdout_pair_direct"
        ]
        |
        work[
            "matches_holdout_pair_reverse"
        ]
    )

    holdout_rows_removed = int(
        work[
            "matches_any_holdout_underlying_pair"
        ]
        .sum()
    )

    holdout_unique_pairs_removed = int(
        work.loc[
            work[
                "matches_any_holdout_underlying_pair"
            ],
            [
                "acquirer_resolved_id",
                "acquiree_resolved_id",
            ],
        ]
        .drop_duplicates()
        .shape[0]
    )

    holdout_summary = pd.DataFrame(
        [
            {
                "metric":
                    "candidate_rows_before_holdout_mask",

                "value":
                    len(
                        work
                    ),
            },

            {
                "metric":
                    "rows_matching_direct_holdout_pair",

                "value":
                    int(
                        work[
                            "matches_holdout_pair_direct"
                        ]
                        .sum()
                    ),
            },

            {
                "metric":
                    "rows_matching_reverse_holdout_pair",

                "value":
                    int(
                        work[
                            "matches_holdout_pair_reverse"
                        ]
                        .sum()
                    ),
            },

            {
                "metric":
                    "rows_removed_by_conservative_holdout_mask",

                "value":
                    holdout_rows_removed,
            },

            {
                "metric":
                    "unique_underlying_pairs_removed",

                "value":
                    holdout_unique_pairs_removed,
            },
        ]
    )

    work = (
        work[
            ~work[
                "matches_any_holdout_underlying_pair"
            ]
        ]
        .copy()
    )

    previous_count = (
        record_stage(
            5,
            "remove_T60_holdout_underlying_pairs",
            work,
            previous_count,
        )
    )

    waterfall = pd.DataFrame(
        waterfall_rows
    )

    # =========================================================================
    # 6. Deduplicate repeated acquisition events
    #
    # One final structural relation candidate per directed underlying pair.
    # =========================================================================

    aggregation = {
        "id":
            "size",

        "announced_on":
            [
                "min",
                "max",
            ],
    }

    if (
        "acquisition_type"
        in work.columns
    ):

        aggregation[
            "acquisition_type"
        ] = (
            lambda values:
                " | ".join(
                    sorted(
                        set(
                            values
                            .dropna()
                            .astype(str)
                        )
                    )
                )
        )

    pair_candidates = (
        work.groupby(
            [
                "acquirer_resolved_id",
                "acquiree_resolved_id",
            ],
            observed=True,
        )
        .agg(
            aggregation
        )
        .reset_index()
    )

    # Flatten aggregation column names.
    pair_candidates.columns = [
        "_".join(
            [
                str(part)
                for part in column
                if str(part)
                not in {
                    "",
                    "<lambda>",
                }
            ]
        )
        if isinstance(
            column,
            tuple
        )
        else column
        for column in (
            pair_candidates.columns
        )
    ]

    rename_map = {
        "id_size":
            "acquisition_event_count",

        "announced_on_min":
            "first_acquisition_date",

        "announced_on_max":
            "last_acquisition_date",
    }

    pair_candidates = (
        pair_candidates.rename(
            columns=rename_map
        )
    )

    # Depending on pandas aggregation naming, make sure endpoint names survive.
    endpoint_rename = {}

    for column in (
        pair_candidates.columns
    ):

        if column.startswith(
            "acquirer_resolved_id"
        ):

            endpoint_rename[
                column
            ] = (
                "acquirer_resolved_id"
            )

        if column.startswith(
            "acquiree_resolved_id"
        ):

            endpoint_rename[
                column
            ] = (
                "acquiree_resolved_id"
            )

    pair_candidates = (
        pair_candidates.rename(
            columns=endpoint_rename
        )
    )

    # =========================================================================
    # 7. Underlying role memberships
    # =========================================================================

    pair_candidates[
        "acquirer_role_membership"
    ] = (
        pair_candidates[
            "acquirer_resolved_id"
        ]
        .apply(
            lambda entity_id:
                role_label(
                    entity_id,
                    investor_ids,
                    startup_ids,
                )
        )
    )

    pair_candidates[
        "acquiree_role_membership"
    ] = (
        pair_candidates[
            "acquiree_resolved_id"
        ]
        .apply(
            lambda entity_id:
                role_label(
                    entity_id,
                    investor_ids,
                    startup_ids,
                )
        )
    )

    pair_candidates[
        "underlying_role_pair"
    ] = (
        pair_candidates[
            "acquirer_role_membership"
        ]
        +
        " -> "
        +
        pair_candidates[
            "acquiree_role_membership"
        ]
    )

    # =========================================================================
    # 8. Potential role-specific graph projections
    #
    # Still diagnostic only.
    # =========================================================================

    projected_role_counts = {
        (
            "investor",
            "investor",
        ): 0,

        (
            "investor",
            "startup",
        ): 0,

        (
            "startup",
            "investor",
        ): 0,

        (
            "startup",
            "startup",
        ): 0,
    }

    projected_total = 0

    for acquirer_id, acquiree_id in zip(
        pair_candidates[
            "acquirer_resolved_id"
        ],
        pair_candidates[
            "acquiree_resolved_id"
        ],
    ):

        source_roles = (
            role_memberships(
                acquirer_id,
                investor_ids,
                startup_ids,
            )
        )

        target_roles = (
            role_memberships(
                acquiree_id,
                investor_ids,
                startup_ids,
            )
        )

        for source_role in (
            source_roles
        ):

            for target_role in (
                target_roles
            ):

                projected_role_counts[
                    (
                        source_role,
                        target_role,
                    )
                ] += 1

                projected_total += 1

    role_projection_rows = []

    for (
        source_role,
        target_role,
    ), count in (
        projected_role_counts.items()
    ):

        role_projection_rows.append(
            {
                "source_role":
                    source_role,

                "target_role":
                    target_role,

                "relation_type":
                    (
                        f"{source_role}"
                        f" -> "
                        f"{target_role}"
                    ),

                "projected_directed_relation_candidates":
                    count,

                "share_pct":
                    pct(
                        count,
                        projected_total,
                    ),
            }
        )

    role_projection = pd.DataFrame(
        role_projection_rows
    )

    # =========================================================================
    # 9. Pair-level summary
    # =========================================================================

    pair_summary = pd.DataFrame(
        [
            {
                "metric":
                    "accepted_acquisition_event_rows",

                "value":
                    len(
                        work
                    ),
            },

            {
                "metric":
                    "unique_directed_underlying_pairs",

                "value":
                    len(
                        pair_candidates
                    ),
            },

            {
                "metric":
                    "repeat_event_occurrences_above_unique_pairs",

                "value":
                    (
                        len(
                            work
                        )
                        -
                        len(
                            pair_candidates
                        )
                    ),
            },

            {
                "metric":
                    "pairs_with_multiple_acquisition_events",

                "value":
                    int(
                        (
                            pair_candidates[
                                "acquisition_event_count"
                            ]
                            > 1
                        )
                        .sum()
                    ),
            },

            {
                "metric":
                    "max_acquisition_events_for_one_pair",

                "value":
                    int(
                        pair_candidates[
                            "acquisition_event_count"
                        ]
                        .max()
                        if len(
                            pair_candidates
                        ) > 0
                        else 0
                    ),
            },

            {
                "metric":
                    "potential_role_specific_directed_relations",

                "value":
                    projected_total,
            },
        ]
    )

    # =========================================================================
    # 10. Save audit outputs
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    waterfall.to_csv(
        WATERFALL_OUTPUT,
        index=False,
    )

    role_projection.to_csv(
        ROLE_OUTPUT,
        index=False,
    )

    holdout_summary.to_csv(
        HOLDOUT_OUTPUT,
        index=False,
    )

    pair_summary.to_csv(
        PAIR_OUTPUT,
        index=False,
    )

    pair_candidates.to_parquet(
        CANDIDATES_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 11. Figures
    # =========================================================================

    figure_data = (
        waterfall
        .sort_values(
            "stage_number"
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        figure_data[
            "stage"
        ],
        figure_data[
            "rows_remaining"
        ],
    )

    ax.set_xlabel(
        "Acquisition rows remaining"
    )

    ax.set_title(
        "Conservative Acquisition Relation Filtering"
    )

    fig.tight_layout()

    fig.savefig(
        WATERFALL_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # -------------------------------------------------------------------------

    role_plot = (
        role_projection
        .sort_values(
            "projected_directed_relation_candidates",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.barh(
        role_plot[
            "relation_type"
        ],
        role_plot[
            "projected_directed_relation_candidates"
        ],
    )

    ax.set_xlabel(
        "Potential directed relation candidates"
    )

    ax.set_title(
        "Acquisition Relations by Role Projection"
    )

    fig.tight_layout()

    fig.savefig(
        ROLE_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =========================================================================
    # 12. Terminal report
    # =========================================================================

    separator("-")

    print(
        "CONSERVATIVE FILTER WATERFALL"
    )

    separator("-")

    print(
        waterfall.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "T60 HOLDOUT MASK"
    )

    separator("-")

    print(
        holdout_summary.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "FINAL UNDERLYING ACQUISITION PAIRS"
    )

    separator("-")

    print(
        pair_summary.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "POTENTIAL ROLE-SPECIFIC PROJECTION"
    )

    separator("-")

    print(
        role_projection.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator()

    print(
        "PHASE 3.2.9 AUDIT COMPLETE"
    )

    separator()

    print(
        f"""
Outputs written to:

{WATERFALL_OUTPUT}
{ROLE_OUTPUT}
{HOLDOUT_OUTPUT}
{PAIR_OUTPUT}
{CANDIDATES_OUTPUT}

Figures written to:

{WATERFALL_FIGURE}
{ROLE_FIGURE}


IMPORTANT

1. acquisition_strict_underlying_relation_candidates.parquet is an
   AUDITED RELATION-CANDIDATE artifact, not the final graph.

2. No reverse R-GCN relation has been added.

3. No role-specific graph edge has been materialized.

4. No SHARED_FOUNDER edge has yet been materialized.

5. T60 held-out target pairs have been conservatively masked from the
   acquisition candidate relation set.

6. The acquisition relation remains a Crunchbase-specific adaptation
   of the ITRS structural-relation principle.

NEXT:

Phase 3.2.10 — ITRS-to-Crunchbase Relation Support Matrix & Audit Closure
"""
    )


if __name__ == "__main__":
    main()