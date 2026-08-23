from pathlib import Path
from itertools import combinations
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PHASE 3.2.10 — SHARED-FOUNDER DEDUPLICATION &
#                LEAKAGE-SAFE CANDIDATE AUDIT
# =============================================================================
#
# PURPOSE
# -------
#
# Convert the validated:
#
#     Company -> Person <- Company
#
# bridge into audited SHARED_FOUNDER relation candidates.
#
# This script:
#
#   1. reconstructs exact founder -> People mappings;
#   2. generates role-node pair occurrences through shared founders;
#   3. deduplicates pairs sharing multiple founders;
#   4. removes pairs representing the same underlying Crunchbase entity;
#   5. removes any underlying pair corresponding to a T60 held-out target;
#   6. reports the final symmetric relation-candidate universe.
#
# NO final graph edge is materialized.
# =============================================================================


RAW_DIR = Path("data/raw")

INTERACTIONS_PATH = Path(
    "data/processed/interactions.parquet"
)

HOLDOUT_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "t60_holdout_pair_manifest.parquet"
)

PHASE_327_PROJECTION_PATH = Path(
    "data/experimental/phase_3/audits/"
    "shared_founder_projection_size_summary.csv"
)


OUTPUT_DIR = Path(
    "data/experimental/phase_3/audits"
)

FIGURE_DIR = Path(
    "data/experimental/phase_3/figures"
)


WATERFALL_OUTPUT = (
    OUTPUT_DIR
    / "shared_founder_strict_filter_waterfall.csv"
)

ROLE_OUTPUT = (
    OUTPUT_DIR
    / "shared_founder_strict_role_pair_summary.csv"
)

MULTIPLICITY_OUTPUT = (
    OUTPUT_DIR
    / "shared_founder_multiplicity_summary.csv"
)

HOLDOUT_OUTPUT = (
    OUTPUT_DIR
    / "shared_founder_holdout_mask_summary.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "shared_founder_strict_candidate_summary.csv"
)

CANDIDATES_OUTPUT = (
    OUTPUT_DIR
    / "shared_founder_strict_role_pair_candidates.parquet"
)


WATERFALL_FIGURE = (
    FIGURE_DIR
    / "shared_founder_strict_filter_waterfall.png"
)

ROLE_FIGURE = (
    FIGURE_DIR
    / "shared_founder_strict_role_pair_distribution.png"
)

MULTIPLICITY_FIGURE = (
    FIGURE_DIR
    / "shared_founder_count_distribution.png"
)


PERSON_URL_RE = re.compile(
    r"https?://(?:www\.)?crunchbase\.com/person/[^\s,;|]+",
    flags=re.IGNORECASE,
)


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


def normalize_url(value):

    if pd.isna(value):
        return None

    value = str(
        value
    ).strip()

    if not value:
        return None

    value = (
        value
        .split("?")[0]
        .split("#")[0]
        .rstrip("/")
        .lower()
    )

    value = value.replace(
        "http://www.crunchbase.com/",
        "https://www.crunchbase.com/",
    )

    value = value.replace(
        "http://crunchbase.com/",
        "https://www.crunchbase.com/",
    )

    value = value.replace(
        "https://crunchbase.com/",
        "https://www.crunchbase.com/",
    )

    return value


def extract_person_urls(value):

    if pd.isna(value):
        return []

    matches = (
        PERSON_URL_RE.findall(
            str(value)
        )
    )

    values = []

    for url in matches:

        normalized = (
            normalize_url(
                url
            )
        )

        if normalized is not None:

            values.append(
                normalized
            )

    # Preserve order but remove duplicate URLs in one company record.
    return list(
        dict.fromkeys(
            values
        )
    )


def role_from_node_id(
    node_id,
):

    if node_id.startswith(
        "investor::"
    ):
        return "investor"

    if node_id.startswith(
        "startup::"
    ):
        return "startup"

    raise ValueError(
        f"Unknown role-node namespace: {node_id}"
    )


def role_pair_label(
    role_a,
    role_b,
):

    roles = sorted(
        [
            role_a,
            role_b,
        ]
    )

    return (
        f"{roles[0]}"
        f" -- "
        f"{roles[1]}"
    )


def ordered_underlying_pair(
    entity_a,
    entity_b,
):

    return tuple(
        sorted(
            [
                str(entity_a),
                str(entity_b),
            ]
        )
    )


def main():

    separator()

    print(
        "PHASE 3.2.10 — "
        "SHARED-FOUNDER DEDUPLICATION & "
        "LEAKAGE-SAFE CANDIDATE AUDIT"
    )

    separator()

    # =========================================================================
    # 1. Canonical role universes
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

    canonical_underlying_ids = (
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
    # 2. T60 held-out underlying pair universe
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

    holdout_underlying_pairs = {
        ordered_underlying_pair(
            investor_id,
            startup_id,
        )
        for investor_id, startup_id in zip(
            holdout[
                "investor_id"
            ],
            holdout[
                "startup_id"
            ],
        )
    }

    print(
        f"T60 held-out underlying pairs: "
        f"{len(holdout_underlying_pairs):,}"
    )

    # =========================================================================
    # 3. Extract founder URLs from canonical company records
    # =========================================================================

    company_files = sorted(
        RAW_DIR.glob(
            "companies*.csv"
        )
    )

    company_frames = []

    for path in company_files:

        chunk = pd.read_csv(
            path,
            usecols=[
                "id",
                "founder_links",
            ],
            dtype="string",
            low_memory=False,
        )

        chunk[
            "id"
        ] = clean_string(
            chunk[
                "id"
            ]
        )

        subset = (
            chunk[
                chunk[
                    "id"
                ]
                .isin(
                    canonical_underlying_ids
                )
            ]
            .copy()
        )

        if len(
            subset
        ) == 0:
            continue

        subset[
            "source_file"
        ] = (
            path.name
        )

        company_frames.append(
            subset
        )

    canonical_companies = pd.concat(
        company_frames,
        ignore_index=True,
    )

    # Cross-chunk company duplication was already audited.
    canonical_companies = (
        canonical_companies
        .sort_values(
            [
                "id",
                "source_file",
            ]
        )
        .drop_duplicates(
            subset=[
                "id"
            ],
            keep="last",
        )
        .copy()
    )

    canonical_companies[
        "founder_urls"
    ] = (
        canonical_companies[
            "founder_links"
        ]
        .apply(
            extract_person_urls
        )
    )

    # =========================================================================
    # 4. Create role -> founder URL references
    # =========================================================================

    role_founder_rows = []

    for row in (
        canonical_companies.itertuples(
            index=False
        )
    ):

        entity_id = (
            row.id
        )

        founder_urls = (
            row.founder_urls
        )

        if not founder_urls:
            continue

        roles = []

        if entity_id in investor_ids:

            roles.append(
                "investor"
            )

        if entity_id in startup_ids:

            roles.append(
                "startup"
            )

        for role in roles:

            role_node_id = (
                f"{role}::{entity_id}"
            )

            for person_url in (
                founder_urls
            ):

                role_founder_rows.append(
                    {
                        "role_node_id":
                            role_node_id,

                        "role":
                            role,

                        "underlying_entity_id":
                            entity_id,

                        "person_url":
                            person_url,
                    }
                )

    role_founders = (
        pd.DataFrame(
            role_founder_rows
        )
        .drop_duplicates()
    )

    founder_urls_needed = set(
        role_founders[
            "person_url"
        ]
        .unique()
    )

    print(
        f"Unique founder URLs to resolve: "
        f"{len(founder_urls_needed):,}"
    )

    # =========================================================================
    # 5. Exact People URL -> UUID mapping
    # =========================================================================

    people_files = sorted(
        RAW_DIR.glob(
            "people*.csv"
        )
    )

    matched_people_frames = []

    for path in people_files:

        people = pd.read_csv(
            path,
            usecols=[
                "id",
                "link",
            ],
            dtype="string",
            low_memory=False,
        )

        people[
            "id"
        ] = clean_string(
            people[
                "id"
            ]
        )

        people[
            "person_url"
        ] = (
            people[
                "link"
            ]
            .apply(
                normalize_url
            )
        )

        matched = (
            people[
                people[
                    "person_url"
                ]
                .isin(
                    founder_urls_needed
                )
            ][
                [
                    "id",
                    "person_url",
                ]
            ]
            .copy()
        )

        if len(
            matched
        ) > 0:

            matched_people_frames.append(
                matched
            )

    matched_people = pd.concat(
        matched_people_frames,
        ignore_index=True,
    )

    # Duplicate export rows with same URL + same ID are harmless.
    person_link_resolution = (
        matched_people.groupby(
            "person_url",
            observed=True,
        )
        .agg(
            unique_person_ids=(
                "id",
                "nunique",
            )
        )
        .reset_index()
    )

    unique_links = set(
        person_link_resolution.loc[
            person_link_resolution[
                "unique_person_ids"
            ]
            == 1,
            "person_url",
        ]
    )

    unique_people_map = (
        matched_people[
            matched_people[
                "person_url"
            ]
            .isin(
                unique_links
            )
        ]
        .drop_duplicates(
            subset=[
                "person_url",
                "id",
            ]
        )
        .drop_duplicates(
            subset=[
                "person_url"
            ],
            keep="first",
        )
        .rename(
            columns={
                "id":
                    "person_id"
            }
        )[
            [
                "person_url",
                "person_id",
            ]
        ]
    )

    resolved_role_founders = (
        role_founders.merge(
            unique_people_map,
            on="person_url",
            how="inner",
            validate="many_to_one",
        )
        .drop_duplicates(
            subset=[
                "role_node_id",
                "person_id",
            ]
        )
    )

    # =========================================================================
    # 6. Generate founder-induced role-pair OCCURRENCES
    # =========================================================================

    pair_occurrence_rows = []

    grouped = (
        resolved_role_founders.groupby(
            "person_id",
            observed=True,
        )
    )

    for person_id, group in grouped:

        role_nodes = (
            group[
                [
                    "role_node_id",
                    "role",
                    "underlying_entity_id",
                ]
            ]
            .drop_duplicates()
            .to_dict(
                "records"
            )
        )

        if len(
            role_nodes
        ) < 2:
            continue

        for left, right in combinations(
            role_nodes,
            2,
        ):

            # Canonicalize symmetric role-node pair.
            if (
                left[
                    "role_node_id"
                ]
                <=
                right[
                    "role_node_id"
                ]
            ):

                a = left
                b = right

            else:

                a = right
                b = left

            pair_occurrence_rows.append(
                {
                    "role_node_a":
                        a[
                            "role_node_id"
                        ],

                    "role_node_b":
                        b[
                            "role_node_id"
                        ],

                    "underlying_entity_a":
                        a[
                            "underlying_entity_id"
                        ],

                    "underlying_entity_b":
                        b[
                            "underlying_entity_id"
                        ],

                    "role_a":
                        a[
                            "role"
                        ],

                    "role_b":
                        b[
                            "role"
                        ],

                    "person_id":
                        person_id,
                }
            )

    pair_occurrences = pd.DataFrame(
        pair_occurrence_rows
    )

    # -------------------------------------------------------------------------
    # Cross-check against Phase 3.2.7.
    # -------------------------------------------------------------------------

    if (
        PHASE_327_PROJECTION_PATH.exists()
    ):

        old_summary = pd.read_csv(
            PHASE_327_PROJECTION_PATH
        )

        expected_row = (
            old_summary[
                old_summary[
                    "metric"
                ]
                ==
                "projected_all_shared_founder_pairs"
            ]
        )

        if len(
            expected_row
        ) == 1:

            expected_occurrences = int(
                expected_row.iloc[0][
                    "value"
                ]
            )

            if (
                len(
                    pair_occurrences
                )
                !=
                expected_occurrences
            ):

                raise ValueError(
                    "Shared-founder occurrence count does not "
                    "match Phase 3.2.7. "
                    f"Expected {expected_occurrences:,}, "
                    f"found {len(pair_occurrences):,}."
                )

    # =========================================================================
    # 7. Deduplicate pairs sharing multiple founders
    # =========================================================================

    pair_candidates = (
        pair_occurrences.groupby(
            [
                "role_node_a",
                "role_node_b",
                "underlying_entity_a",
                "underlying_entity_b",
                "role_a",
                "role_b",
            ],
            observed=True,
        )
        .agg(
            shared_founder_count=(
                "person_id",
                "nunique",
            )
        )
        .reset_index()
    )

    pair_candidates[
        "relation_type"
    ] = [
        role_pair_label(
            role_a,
            role_b,
        )
        for role_a, role_b in zip(
            pair_candidates[
                "role_a"
            ],
            pair_candidates[
                "role_b"
            ],
        )
    ]

    pair_candidates[
        "same_underlying_entity"
    ] = (
        pair_candidates[
            "underlying_entity_a"
        ]
        .eq(
            pair_candidates[
                "underlying_entity_b"
            ]
        )
    )

    pair_candidates[
        "underlying_pair_key"
    ] = [
        ordered_underlying_pair(
            entity_a,
            entity_b,
        )
        for entity_a, entity_b in zip(
            pair_candidates[
                "underlying_entity_a"
            ],
            pair_candidates[
                "underlying_entity_b"
            ],
        )
    ]

    pair_candidates[
        "matches_T60_holdout_underlying_pair"
    ] = (
        pair_candidates[
            "underlying_pair_key"
        ]
        .isin(
            holdout_underlying_pairs
        )
    )

    # =========================================================================
    # 8. Filtering waterfall
    # =========================================================================

    occurrence_count = len(
        pair_occurrences
    )

    unique_role_pairs = len(
        pair_candidates
    )

    same_entity_count = int(
        pair_candidates[
            "same_underlying_entity"
        ]
        .sum()
    )

    work = (
        pair_candidates[
            ~pair_candidates[
                "same_underlying_entity"
            ]
        ]
        .copy()
    )

    before_holdout = len(
        work
    )

    holdout_count = int(
        work[
            "matches_T60_holdout_underlying_pair"
        ]
        .sum()
    )

    work = (
        work[
            ~work[
                "matches_T60_holdout_underlying_pair"
            ]
        ]
        .copy()
    )

    final_count = len(
        work
    )

    waterfall = pd.DataFrame(
        [
            {
                "stage_number":
                    0,

                "stage":
                    "founder_induced_pair_occurrences",

                "rows_remaining":
                    occurrence_count,
            },

            {
                "stage_number":
                    1,

                "stage":
                    "deduplicate_role_node_pairs",

                "rows_remaining":
                    unique_role_pairs,
            },

            {
                "stage_number":
                    2,

                "stage":
                    "remove_same_underlying_entity_pairs",

                "rows_remaining":
                    before_holdout,
            },

            {
                "stage_number":
                    3,

                "stage":
                    "remove_T60_holdout_underlying_pairs",

                "rows_remaining":
                    final_count,
            },
        ]
    )

    waterfall[
        "removed_from_previous"
    ] = (
        waterfall[
            "rows_remaining"
        ]
        .shift(1)
        -
        waterfall[
            "rows_remaining"
        ]
    )

    waterfall.loc[
        0,
        "removed_from_previous",
    ] = 0

    waterfall[
        "retained_from_initial_pct"
    ] = (
        waterfall[
            "rows_remaining"
        ]
        /
        occurrence_count
        * 100
    )

    # =========================================================================
    # 9. Multiplicity
    # =========================================================================

    multiplicity = (
        pair_candidates[
            "shared_founder_count"
        ]
        .value_counts()
        .sort_index()
        .rename_axis(
            "shared_founder_count"
        )
        .reset_index(
            name="role_pair_count"
        )
    )

    # =========================================================================
    # 10. Final role-pair distribution
    # =========================================================================

    role_summary = (
        work[
            "relation_type"
        ]
        .value_counts()
        .rename_axis(
            "relation_type"
        )
        .reset_index(
            name="unique_role_pair_candidates"
        )
    )

    role_summary[
        "share_pct"
    ] = (
        role_summary[
            "unique_role_pair_candidates"
        ]
        /
        len(
            work
        )
        * 100
    )

    # =========================================================================
    # 11. Holdout summary
    # =========================================================================

    holdout_summary = pd.DataFrame(
        [
            {
                "metric":
                    "unique_role_pairs_before_holdout_mask",

                "value":
                    before_holdout,
            },

            {
                "metric":
                    "role_pairs_removed_by_holdout_mask",

                "value":
                    holdout_count,
            },

            {
                "metric":
                    "final_leakage_safe_role_pairs",

                "value":
                    final_count,
            },
        ]
    )

    # =========================================================================
    # 12. Final candidate summary
    # =========================================================================

    summary = pd.DataFrame(
        [
            {
                "metric":
                    "founder_induced_pair_occurrences",

                "value":
                    occurrence_count,
            },

            {
                "metric":
                    "unique_role_node_pairs_before_filters",

                "value":
                    unique_role_pairs,
            },

            {
                "metric":
                    "pairs_with_multiple_shared_founders",

                "value":
                    int(
                        (
                            pair_candidates[
                                "shared_founder_count"
                            ]
                            > 1
                        )
                        .sum()
                    ),
            },

            {
                "metric":
                    "max_shared_founders_for_one_role_pair",

                "value":
                    int(
                        pair_candidates[
                            "shared_founder_count"
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
                    "same_underlying_entity_pairs_removed",

                "value":
                    same_entity_count,
            },

            {
                "metric":
                    "T60_holdout_role_pairs_removed",

                "value":
                    holdout_count,
            },

            {
                "metric":
                    "final_shared_founder_relation_candidates",

                "value":
                    final_count,
            },

            {
                "metric":
                    "potential_directed_edges_if_stored_bidirectionally",

                "value":
                    final_count * 2,
            },
        ]
    )

    # Founder relation has no historical relation timestamp.
    work[
        "temporal_provenance"
    ] = (
        "current_snapshot_unversioned"
    )

    # Drop Python tuple helper before Parquet output.
    work = work.drop(
        columns=[
            "underlying_pair_key"
        ]
    )

    # =========================================================================
    # 13. Save
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

    role_summary.to_csv(
        ROLE_OUTPUT,
        index=False,
    )

    multiplicity.to_csv(
        MULTIPLICITY_OUTPUT,
        index=False,
    )

    holdout_summary.to_csv(
        HOLDOUT_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    work.to_parquet(
        CANDIDATES_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 14. Figures
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    plot_data = (
        waterfall
        .sort_values(
            "stage_number",
            ascending=False,
        )
    )

    ax.barh(
        plot_data[
            "stage"
        ],
        plot_data[
            "rows_remaining"
        ],
    )

    ax.set_xlabel(
        "Pair candidates remaining"
    )

    ax.set_title(
        "Shared-Founder Candidate Filtering"
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

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    role_plot = (
        role_summary
        .sort_values(
            "unique_role_pair_candidates",
            ascending=True,
        )
    )

    ax.barh(
        role_plot[
            "relation_type"
        ],
        role_plot[
            "unique_role_pair_candidates"
        ],
    )

    ax.set_xlabel(
        "Unique leakage-safe role-node pairs"
    )

    ax.set_title(
        "SHARED_FOUNDER Relations by Role Pair"
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

    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        multiplicity[
            "shared_founder_count"
        ].astype(str),
        multiplicity[
            "role_pair_count"
        ],
    )

    ax.set_xlabel(
        "Number of shared founders"
    )

    ax.set_ylabel(
        "Role-node pairs"
    )

    ax.set_title(
        "Shared-Founder Multiplicity"
    )

    fig.tight_layout()

    fig.savefig(
        MULTIPLICITY_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =========================================================================
    # 15. Terminal report
    # =========================================================================

    separator("-")

    print(
        "SHARED-FOUNDER FILTER WATERFALL"
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
        "SHARED-FOUNDER MULTIPLICITY"
    )

    separator("-")

    print(
        multiplicity.to_string(
            index=False
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
        "FINAL ROLE-PAIR DISTRIBUTION"
    )

    separator("-")

    print(
        role_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "FINAL SHARED-FOUNDER SUMMARY"
    )

    separator("-")

    print(
        summary.to_string(
            index=False
        )
    )

    separator()

    print(
        "PHASE 3.2.10 AUDIT COMPLETE"
    )

    separator()

    print(
        f"""
Outputs written to:

{WATERFALL_OUTPUT}
{ROLE_OUTPUT}
{MULTIPLICITY_OUTPUT}
{HOLDOUT_OUTPUT}
{SUMMARY_OUTPUT}
{CANDIDATES_OUTPUT}

Figures written to:

{WATERFALL_FIGURE}
{ROLE_FIGURE}
{MULTIPLICITY_FIGURE}


IMPORTANT

1. SHARED_FOUNDER is treated as a symmetric structural relation.

2. The candidate Parquet stores ONE unordered role-node pair per relation.

3. No forward/reverse R-GCN expansion has yet been performed.

4. Pairs representing two roles of the SAME underlying Crunchbase UUID
   have been excluded.

5. All direct structural pairs matching a T60 held-out underlying
   Investor-Startup pair have been removed.

6. Founder relationships have no historical relationship timestamp.
   Their provenance is explicitly recorded as:

       current_snapshot_unversioned

7. This remains a RELATION-CANDIDATE artifact, not the final graph.

NEXT:

Phase 3.2.11 — ITRS-to-Crunchbase Relation Support Matrix &
               Phase-3.2 Audit Closure
"""
    )


if __name__ == "__main__":
    main()