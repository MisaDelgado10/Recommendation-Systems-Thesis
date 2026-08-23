from pathlib import Path
from collections import defaultdict
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PHASE 3.2.8 — ACQUISITION ENDPOINT RESOLUTION & STRUCTURAL-RELATION AUDIT
# =============================================================================
#
# PURPOSE
# -------
#
# Resolve Crunchbase acquisition acquirer/acquiree endpoints to company UUIDs
# conservatively using independently observed name and website evidence.
#
# NO graph edge is materialized.
# NO ambiguous endpoint is assigned arbitrarily.
#
# Also audits temporal availability relative to the frozen Phase-2 T60 window.
# =============================================================================


RAW_DIR = Path("data/raw")

INTERACTIONS_PATH = Path(
    "data/processed/interactions.parquet"
)

T60_MANIFEST_PATH = Path(
    "data/experimental/phase_2/model_ready/"
    "t60_holdout_pair_manifest.parquet"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_3/audits"
)

FIGURE_DIR = Path(
    "data/experimental/phase_3/figures"
)


ENDPOINT_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "acquisition_endpoint_resolution_summary.csv"
)

RELATION_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "acquisition_resolved_relation_summary.csv"
)

TEMPORAL_OUTPUT = (
    OUTPUT_DIR
    / "acquisition_temporal_availability_summary.csv"
)

ROLE_OUTPUT = (
    OUTPUT_DIR
    / "acquisition_canonical_role_overlap.csv"
)

HOLDOUT_OUTPUT = (
    OUTPUT_DIR
    / "acquisition_t60_holdout_overlap_summary.csv"
)

UNRESOLVED_OUTPUT = (
    OUTPUT_DIR
    / "acquisition_unresolved_endpoint_examples.csv"
)

CONFLICT_OUTPUT = (
    OUTPUT_DIR
    / "acquisition_resolution_conflict_examples.csv"
)

RESOLVED_RELATIONS_OUTPUT = (
    OUTPUT_DIR
    / "acquisition_resolved_relation_candidates.parquet"
)


RESOLUTION_FIGURE = (
    FIGURE_DIR
    / "acquisition_endpoint_resolution_status.png"
)

TEMPORAL_FIGURE = (
    FIGURE_DIR
    / "acquisition_relation_temporal_availability.png"
)

ROLE_FIGURE = (
    FIGURE_DIR
    / "acquisition_canonical_role_pair_types.png"
)


T60_START = pd.Timestamp(
    "2026-01-01"
)

T60_END = pd.Timestamp(
    "2026-03-31"
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


def normalize_name(value):

    if pd.isna(value):
        return None

    value = str(
        value
    ).strip()

    if not value:
        return None

    # Conservative normalization:
    # case-insensitive and whitespace-normalized only.
    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.casefold()


def normalize_website(value):

    if pd.isna(value):
        return None

    value = str(
        value
    ).strip()

    if not value:
        return None

    value = value.casefold()

    value = re.sub(
        r"^https?://",
        "",
        value,
    )

    value = re.sub(
        r"^www\.",
        "",
        value,
    )

    value = value.split(
        "#",
        1,
    )[0]

    value = value.split(
        "?",
        1,
    )[0]

    value = value.rstrip(
        "/"
    )

    return (
        value
        if value
        else None
    )


def add_candidate(
    mapping,
    key,
    entity_id,
):

    if key is None:
        return

    mapping[
        key
    ].add(
        entity_id
    )


def resolve_endpoint(
    normalized_name,
    normalized_website,
    name_map,
    website_map,
):

    name_candidates = (
        name_map.get(
            normalized_name,
            set(),
        )
        if normalized_name
        is not None
        else set()
    )

    website_candidates = (
        website_map.get(
            normalized_website,
            set(),
        )
        if normalized_website
        is not None
        else set()
    )

    # -------------------------------------------------------------------------
    # Strongest case:
    # both independently identify exactly one same UUID.
    # -------------------------------------------------------------------------

    if (
        len(
            name_candidates
        )
        == 1
        and
        len(
            website_candidates
        )
        == 1
    ):

        name_id = next(
            iter(
                name_candidates
            )
        )

        website_id = next(
            iter(
                website_candidates
            )
        )

        if (
            name_id
            == website_id
        ):

            return (
                name_id,
                "name_website_agree",
                len(
                    name_candidates
                ),
                len(
                    website_candidates
                ),
            )

        return (
            None,
            "name_website_conflict",
            len(
                name_candidates
            ),
            len(
                website_candidates
            ),
        )

    # -------------------------------------------------------------------------
    # Ambiguous name rescued by a unique website ONLY if website result
    # is one of the exact-name candidates.
    # -------------------------------------------------------------------------

    if (
        len(
            name_candidates
        )
        > 1
        and
        len(
            website_candidates
        )
        == 1
    ):

        website_id = next(
            iter(
                website_candidates
            )
        )

        if (
            website_id
            in name_candidates
        ):

            return (
                website_id,
                "ambiguous_name_resolved_by_website",
                len(
                    name_candidates
                ),
                len(
                    website_candidates
                ),
            )

        return (
            None,
            "ambiguous_name_website_conflict",
            len(
                name_candidates
            ),
            len(
                website_candidates
            ),
        )

    # -------------------------------------------------------------------------
    # Unique exact name with no usable website evidence.
    # -------------------------------------------------------------------------

    if (
        len(
            name_candidates
        )
        == 1
        and
        len(
            website_candidates
        )
        == 0
    ):

        return (
            next(
                iter(
                    name_candidates
                )
            ),
            "unique_name_only",
            1,
            0,
        )

    # -------------------------------------------------------------------------
    # Unique website but name did not resolve.
    # Keep as AUDIT CANDIDATE rather than automatically accepting later.
    # -------------------------------------------------------------------------

    if (
        len(
            name_candidates
        )
        == 0
        and
        len(
            website_candidates
        )
        == 1
    ):

        return (
            next(
                iter(
                    website_candidates
                )
            ),
            "unique_website_only",
            0,
            1,
        )

    # -------------------------------------------------------------------------

    if (
        len(
            name_candidates
        )
        > 1
    ):

        return (
            None,
            "ambiguous_name_unresolved",
            len(
                name_candidates
            ),
            len(
                website_candidates
            ),
        )

    if (
        len(
            website_candidates
        )
        > 1
    ):

        return (
            None,
            "ambiguous_website_unresolved",
            len(
                name_candidates
            ),
            len(
                website_candidates
            ),
        )

    return (
        None,
        "no_match",
        len(
            name_candidates
        ),
        len(
            website_candidates
        ),
    )


def role_label(
    entity_id,
    investor_ids,
    startup_ids,
):

    if entity_id is None:
        return "unresolved"

    is_investor = (
        entity_id
        in investor_ids
    )

    is_startup = (
        entity_id
        in startup_ids
    )

    if (
        is_investor
        and is_startup
    ):
        return "investor+startup"

    if is_investor:
        return "investor"

    if is_startup:
        return "startup"

    return "outside_canonical_roles"


def main():

    separator()

    print(
        "PHASE 3.2.8 — "
        "ACQUISITION ENDPOINT RESOLUTION & STRUCTURAL-RELATION AUDIT"
    )

    separator()

    # =========================================================================
    # 1. Locate acquisition source
    # =========================================================================

    acquisition_files = sorted(
        RAW_DIR.glob(
            "CRUNCHBASE_acquisition*.csv"
        )
    )

    if len(
        acquisition_files
    ) != 1:

        raise ValueError(
            "Expected exactly one acquisition CSV."
        )

    acquisition_path = (
        acquisition_files[0]
    )

    acquisitions = pd.read_csv(
        acquisition_path,
        dtype="string",
        low_memory=False,
    )

    required_columns = [
        "id",
        "announced_on",
        "acquirer_name",
        "acquirer_website",
        "acquiree_name",
        "acquiree_website",
    ]

    missing = [
        column
        for column in required_columns
        if column
        not in acquisitions.columns
    ]

    if missing:

        raise ValueError(
            f"Acquisition file missing required columns: {missing}"
        )

    acquisitions[
        "announced_on"
    ] = pd.to_datetime(
        acquisitions[
            "announced_on"
        ],
        errors="coerce",
    )

    # =========================================================================
    # 2. Normalize endpoint evidence
    # =========================================================================

    for side in [
        "acquirer",
        "acquiree",
    ]:

        acquisitions[
            f"{side}_normalized_name"
        ] = (
            acquisitions[
                f"{side}_name"
            ]
            .apply(
                normalize_name
            )
        )

        acquisitions[
            f"{side}_normalized_website"
        ] = (
            acquisitions[
                f"{side}_website"
            ]
            .apply(
                normalize_website
            )
        )

    target_names = set(
        acquisitions[
            "acquirer_normalized_name"
        ]
        .dropna()
    ) | set(
        acquisitions[
            "acquiree_normalized_name"
        ]
        .dropna()
    )

    target_websites = set(
        acquisitions[
            "acquirer_normalized_website"
        ]
        .dropna()
    ) | set(
        acquisitions[
            "acquiree_normalized_website"
        ]
        .dropna()
    )

    print(
        f"\nUnique endpoint names to resolve: "
        f"{len(target_names):,}"
    )

    print(
        f"Unique endpoint websites to resolve: "
        f"{len(target_websites):,}"
    )

    # =========================================================================
    # 3. Stream company registry and collect only relevant candidates
    # =========================================================================

    name_map = defaultdict(
        set
    )

    website_map = defaultdict(
        set
    )

    company_files = sorted(
        RAW_DIR.glob(
            "companies*.csv"
        )
    )

    print(
        "\nScanning company registry for acquisition endpoints..."
    )

    for file_index, path in enumerate(
        company_files,
        start=1,
    ):

        print(
            f"  [{file_index}/{len(company_files)}] "
            f"{path.name}"
        )

        chunk = pd.read_csv(
            path,
            usecols=[
                "id",
                "name",
                "website",
            ],
            dtype="string",
            low_memory=False,
        )

        chunk[
            "normalized_name"
        ] = (
            chunk[
                "name"
            ]
            .apply(
                normalize_name
            )
        )

        chunk[
            "normalized_website"
        ] = (
            chunk[
                "website"
            ]
            .apply(
                normalize_website
            )
        )

        candidate_mask = (
            chunk[
                "normalized_name"
            ]
            .isin(
                target_names
            )
            |
            chunk[
                "normalized_website"
            ]
            .isin(
                target_websites
            )
        )

        relevant = (
            chunk[
                candidate_mask
            ]
        )

        for row in (
            relevant.itertuples(
                index=False
            )
        ):

            entity_id = str(
                row.id
            ).strip()

            if (
                row.normalized_name
                in target_names
            ):

                add_candidate(
                    name_map,
                    row.normalized_name,
                    entity_id,
                )

            if (
                row.normalized_website
                in target_websites
            ):

                add_candidate(
                    website_map,
                    row.normalized_website,
                    entity_id,
                )

    # =========================================================================
    # 4. Resolve each acquisition endpoint
    # =========================================================================

    for side in [
        "acquirer",
        "acquiree",
    ]:

        results = [
            resolve_endpoint(
                normalized_name=row[
                    f"{side}_normalized_name"
                ],
                normalized_website=row[
                    f"{side}_normalized_website"
                ],
                name_map=name_map,
                website_map=website_map,
            )
            for _, row in (
                acquisitions.iterrows()
            )
        ]

        acquisitions[
            f"{side}_resolved_id"
        ] = [
            result[0]
            for result in results
        ]

        acquisitions[
            f"{side}_resolution_status"
        ] = [
            result[1]
            for result in results
        ]

        acquisitions[
            f"{side}_name_candidate_count"
        ] = [
            result[2]
            for result in results
        ]

        acquisitions[
            f"{side}_website_candidate_count"
        ] = [
            result[3]
            for result in results
        ]

    # =========================================================================
    # 5. Endpoint resolution summaries
    # =========================================================================

    endpoint_summaries = []

    for side in [
        "acquirer",
        "acquiree",
    ]:

        counts = (
            acquisitions[
                f"{side}_resolution_status"
            ]
            .value_counts()
        )

        for status, count in (
            counts.items()
        ):

            endpoint_summaries.append(
                {
                    "endpoint":
                        side,

                    "resolution_status":
                        status,

                    "count":
                        int(
                            count
                        ),

                    "share_pct":
                        pct(
                            count,
                            len(
                                acquisitions
                            ),
                        ),
                }
            )

    endpoint_summary = pd.DataFrame(
        endpoint_summaries
    )

    # =========================================================================
    # 6. Fully resolved relation candidates
    # =========================================================================

    fully_resolved = (
        acquisitions[
            acquisitions[
                "acquirer_resolved_id"
            ]
            .notna()
            &
            acquisitions[
                "acquiree_resolved_id"
            ]
            .notna()
        ]
        .copy()
    )

    fully_resolved[
        "same_underlying_entity"
    ] = (
        fully_resolved[
            "acquirer_resolved_id"
        ]
        .eq(
            fully_resolved[
                "acquiree_resolved_id"
            ]
        )
    )

    unique_underlying_pairs = (
        fully_resolved[
            [
                "acquirer_resolved_id",
                "acquiree_resolved_id",
            ]
        ]
        .drop_duplicates()
    )

    relation_summary = pd.DataFrame(
        [
            {
                "metric":
                    "acquisition_rows",

                "value":
                    len(
                        acquisitions
                    ),
            },

            {
                "metric":
                    "fully_resolved_rows",

                "value":
                    len(
                        fully_resolved
                    ),
            },

            {
                "metric":
                    "fully_resolved_rows_pct",

                "value":
                    pct(
                        len(
                            fully_resolved
                        ),
                        len(
                            acquisitions
                        ),
                    ),
            },

            {
                "metric":
                    "unique_resolved_underlying_pairs",

                "value":
                    len(
                        unique_underlying_pairs
                    ),
            },

            {
                "metric":
                    "self_acquisition_rows",

                "value":
                    int(
                        fully_resolved[
                            "same_underlying_entity"
                        ]
                        .sum()
                    ),
            },
        ]
    )

    # =========================================================================
    # 7. Canonical investor/startup role overlap
    # =========================================================================

    interactions = pd.read_parquet(
        INTERACTIONS_PATH,
        columns=[
            "investor_id",
            "startup_id",
        ],
    )

    investor_ids = set(
        interactions[
            "investor_id"
        ]
        .astype("string")
        .dropna()
        .unique()
    )

    startup_ids = set(
        interactions[
            "startup_id"
        ]
        .astype("string")
        .dropna()
        .unique()
    )

    fully_resolved[
        "acquirer_role"
    ] = (
        fully_resolved[
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

    fully_resolved[
        "acquiree_role"
    ] = (
        fully_resolved[
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

    fully_resolved[
        "role_pair_type"
    ] = (
        fully_resolved[
            "acquirer_role"
        ]
        +
        " -> "
        +
        fully_resolved[
            "acquiree_role"
        ]
    )

    role_summary = (
        fully_resolved[
            "role_pair_type"
        ]
        .value_counts()
        .rename_axis(
            "role_pair_type"
        )
        .reset_index(
            name="acquisition_rows"
        )
    )

    role_summary[
        "share_of_resolved_pct"
    ] = (
        role_summary[
            "acquisition_rows"
        ]
        /
        len(
            fully_resolved
        )
        * 100
    )

    # =========================================================================
    # 8. Temporal availability relative to T60
    # =========================================================================

    def temporal_bucket(value):

        if pd.isna(
            value
        ):
            return "missing_date"

        if (
            value
            < T60_START
        ):
            return "pre_T60"

        if (
            value
            <= T60_END
        ):
            return "during_T60"

        return "post_T60_endpoint"

    fully_resolved[
        "temporal_bucket"
    ] = (
        fully_resolved[
            "announced_on"
        ]
        .apply(
            temporal_bucket
        )
    )

    temporal_summary = (
        fully_resolved[
            "temporal_bucket"
        ]
        .value_counts()
        .rename_axis(
            "temporal_bucket"
        )
        .reset_index(
            name="resolved_acquisition_rows"
        )
    )

    temporal_summary[
        "share_pct"
    ] = (
        temporal_summary[
            "resolved_acquisition_rows"
        ]
        /
        len(
            fully_resolved
        )
        * 100
    )

    # =========================================================================
    # 9. T60 held-out investor/startup pair overlap
    # =========================================================================

    holdout_summary_rows = []

    if (
        T60_MANIFEST_PATH.exists()
    ):

        holdout = pd.read_parquet(
            T60_MANIFEST_PATH
        )

        required_holdout = {
            "investor_id",
            "startup_id",
        }

        if (
            required_holdout
            .issubset(
                holdout.columns
            )
        ):

            holdout_pairs = set(
                zip(
                    holdout[
                        "investor_id"
                    ]
                    .astype("string"),

                    holdout[
                        "startup_id"
                    ]
                    .astype("string"),
                )
            )

            investor_startup_acquisitions = (
                fully_resolved[
                    fully_resolved[
                        "acquirer_resolved_id"
                    ]
                    .isin(
                        investor_ids
                    )
                    &
                    fully_resolved[
                        "acquiree_resolved_id"
                    ]
                    .isin(
                        startup_ids
                    )
                ]
                .copy()
            )

            investor_startup_acquisitions[
                "is_t60_holdout_pair"
            ] = [
                (
                    investor_id,
                    startup_id,
                )
                in holdout_pairs
                for investor_id, startup_id in zip(
                    investor_startup_acquisitions[
                        "acquirer_resolved_id"
                    ],
                    investor_startup_acquisitions[
                        "acquiree_resolved_id"
                    ],
                )
            ]

            for bucket in [
                "pre_T60",
                "during_T60",
                "post_T60_endpoint",
                "missing_date",
            ]:

                subset = (
                    investor_startup_acquisitions[
                        investor_startup_acquisitions[
                            "temporal_bucket"
                        ]
                        == bucket
                    ]
                )

                holdout_summary_rows.append(
                    {
                        "temporal_bucket":
                            bucket,

                        "investor_to_startup_acquisition_rows":
                            len(
                                subset
                            ),

                        "rows_matching_t60_holdout_pairs":
                            int(
                                subset[
                                    "is_t60_holdout_pair"
                                ]
                                .sum()
                            ),
                    }
                )

    holdout_summary = pd.DataFrame(
        holdout_summary_rows
    )

    # =========================================================================
    # 10. Audit examples
    # =========================================================================

    unresolved_mask = (
        acquisitions[
            "acquirer_resolved_id"
        ]
        .isna()
        |
        acquisitions[
            "acquiree_resolved_id"
        ]
        .isna()
    )

    unresolved_examples = (
        acquisitions[
            unresolved_mask
        ][
            [
                "id",
                "announced_on",
                "acquirer_name",
                "acquirer_website",
                "acquirer_resolution_status",
                "acquirer_name_candidate_count",
                "acquirer_website_candidate_count",
                "acquiree_name",
                "acquiree_website",
                "acquiree_resolution_status",
                "acquiree_name_candidate_count",
                "acquiree_website_candidate_count",
            ]
        ]
        .head(
            500
        )
    )

    conflict_statuses = {
        "name_website_conflict",
        "ambiguous_name_website_conflict",
    }

    conflict_mask = (
        acquisitions[
            "acquirer_resolution_status"
        ]
        .isin(
            conflict_statuses
        )
        |
        acquisitions[
            "acquiree_resolution_status"
        ]
        .isin(
            conflict_statuses
        )
    )

    conflict_examples = (
        acquisitions[
            conflict_mask
        ][
            [
                "id",
                "acquirer_name",
                "acquirer_website",
                "acquirer_resolution_status",
                "acquiree_name",
                "acquiree_website",
                "acquiree_resolution_status",
            ]
        ]
        .head(
            500
        )
    )

    # =========================================================================
    # 11. Save
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    endpoint_summary.to_csv(
        ENDPOINT_SUMMARY_OUTPUT,
        index=False,
    )

    relation_summary.to_csv(
        RELATION_SUMMARY_OUTPUT,
        index=False,
    )

    temporal_summary.to_csv(
        TEMPORAL_OUTPUT,
        index=False,
    )

    role_summary.to_csv(
        ROLE_OUTPUT,
        index=False,
    )

    holdout_summary.to_csv(
        HOLDOUT_OUTPUT,
        index=False,
    )

    unresolved_examples.to_csv(
        UNRESOLVED_OUTPUT,
        index=False,
    )

    conflict_examples.to_csv(
        CONFLICT_OUTPUT,
        index=False,
    )

    fully_resolved.to_parquet(
        RESOLVED_RELATIONS_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 12. Figures
    # =========================================================================

    figure_data = (
        endpoint_summary
        .pivot(
            index="resolution_status",
            columns="endpoint",
            values="count",
        )
        .fillna(0)
    )

    ax = figure_data.plot(
        kind="bar",
        figsize=(11, 6),
    )

    ax.set_ylabel(
        "Endpoint rows"
    )

    ax.set_title(
        "Acquisition Endpoint Resolution Status"
    )

    ax.figure.tight_layout()

    ax.figure.savefig(
        RESOLUTION_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        ax.figure
    )

    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        temporal_summary[
            "temporal_bucket"
        ],
        temporal_summary[
            "resolved_acquisition_rows"
        ],
    )

    ax.set_ylabel(
        "Resolved acquisition rows"
    )

    ax.set_title(
        "Temporal Availability of Acquisition Relations"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    fig.tight_layout()

    fig.savefig(
        TEMPORAL_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # -------------------------------------------------------------------------

    role_plot = (
        role_summary
        .head(
            15
        )
        .sort_values(
            "acquisition_rows",
            ascending=True,
        )
    )

    if len(
        role_plot
    ) > 0:

        fig, ax = plt.subplots(
            figsize=(
                10,
                max(
                    5,
                    len(
                        role_plot
                    )
                    * 0.42,
                ),
            )
        )

        ax.barh(
            role_plot[
                "role_pair_type"
            ],
            role_plot[
                "acquisition_rows"
            ],
        )

        ax.set_xlabel(
            "Resolved acquisition rows"
        )

        ax.set_title(
            "Acquisition Relations by Canonical Role Pair"
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
    # 13. Terminal report
    # =========================================================================

    separator("-")

    print(
        "ACQUISITION ENDPOINT RESOLUTION"
    )

    separator("-")

    print(
        endpoint_summary
        .sort_values(
            [
                "endpoint",
                "count",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "RESOLVED ACQUISITION RELATIONS"
    )

    separator("-")

    print(
        relation_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "CANONICAL ROLE PAIR TYPES"
    )

    separator("-")

    print(
        role_summary
        .head(
            30
        )
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "TEMPORAL AVAILABILITY"
    )

    separator("-")

    print(
        temporal_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "T60 HOLDOUT-PAIR OVERLAP"
    )

    separator("-")

    if len(
        holdout_summary
    ) > 0:

        print(
            holdout_summary.to_string(
                index=False
            )
        )

    else:

        print(
            "No compatible T60 manifest columns were available."
        )

    separator()

    print(
        "PHASE 3.2.8 AUDIT COMPLETE"
    )

    separator()

    print(
        f"""
Outputs written to:

{ENDPOINT_SUMMARY_OUTPUT}
{RELATION_SUMMARY_OUTPUT}
{TEMPORAL_OUTPUT}
{ROLE_OUTPUT}
{HOLDOUT_OUTPUT}
{UNRESOLVED_OUTPUT}
{CONFLICT_OUTPUT}
{RESOLVED_RELATIONS_OUTPUT}

Figures written to:

{RESOLUTION_FIGURE}
{TEMPORAL_FIGURE}
{ROLE_FIGURE}


IMPORTANT

1. These are relation CANDIDATES, not graph edges.

2. 'unique_website_only' cases are intentionally reported separately.
   They have not yet been accepted as conservative resolutions.

3. No ambiguous endpoint was assigned arbitrarily.

4. No temporal cutoff has yet been imposed.

5. Acquisition relations occurring during T60 or after 2026-03-31
   are explicitly separated for leakage analysis.

6. Historical investment interactions remain separate from the
   structural relation audit.

NEXT:

Interpret acquisition resolution quality and decide which endpoint
resolution statuses are safe enough for graph construction.
"""
    )


if __name__ == "__main__":
    main()