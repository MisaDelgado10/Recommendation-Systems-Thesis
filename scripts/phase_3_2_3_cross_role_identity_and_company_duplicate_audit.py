from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PHASE 3.2.3 — CROSS-ROLE IDENTITY & COMPANY-DUPLICATE AUDIT
# =============================================================================
#
# PURPOSE
# -------
#
# Resolve two identity questions discovered in Phase 3.2.2:
#
# 1. The 24 companies export chunks contain cross-file duplicate IDs.
#    Are those duplicate records consistent?
#
# 2. Some investor IDs also appear in the companies universe.
#    Do those IDs represent the same underlying Crunchbase entity?
#
# This audit also measures actual dual-role entities inside the Phase-1
# experimental investor/startup universe.
#
# NO graph node or edge is created here.
# =============================================================================


RAW_DIR = Path(
    "data/raw"
)

PHASE1_PATH = Path(
    "data/processed/interactions.parquet"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_3/audits"
)

FIGURE_DIR = Path(
    "data/experimental/phase_3/figures"
)


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------

DUPLICATE_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "companies_cross_chunk_duplicate_summary.csv"
)

DUPLICATE_DETAILS_OUTPUT = (
    OUTPUT_DIR
    / "companies_cross_chunk_duplicate_details.csv"
)

SHARED_ID_ATTRIBUTE_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "investor_company_shared_id_attribute_agreement.csv"
)

SHARED_ID_DISAGREEMENTS_OUTPUT = (
    OUTPUT_DIR
    / "investor_company_shared_id_disagreements_sample.csv"
)

INVESTOR_TYPE_OUTPUT = (
    OUTPUT_DIR
    / "investor_company_overlap_by_investor_type.csv"
)

CROSS_ROLE_OUTPUT = (
    OUTPUT_DIR
    / "phase1_cross_role_identity_summary.csv"
)

DUAL_ROLE_IDS_OUTPUT = (
    OUTPUT_DIR
    / "phase1_dual_role_entity_ids.csv"
)


DUPLICATE_FIGURE = (
    FIGURE_DIR
    / "companies_duplicate_id_multiplicity.png"
)

INVESTOR_TYPE_FIGURE = (
    FIGURE_DIR
    / "investor_company_overlap_by_investor_type.png"
)

CROSS_ROLE_FIGURE = (
    FIGURE_DIR
    / "phase1_cross_role_identity_counts.png"
)


KEY_COMPANY_ATTRIBUTES = [
    "id",
    "name",
    "link",
    "website",
    "company_type",
]

KEY_INVESTOR_ATTRIBUTES = [
    "id",
    "name",
    "link",
    "website",
    "investor_type",
    "company_type",
]


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


def normalized_compare(series):

    return (
        clean_string(series)
        .str.lower()
        .replace(
            {
                "": pd.NA
            }
        )
    )


def nunique_nonmissing(series):

    values = (
        clean_string(series)
        .dropna()
    )

    values = values[
        values.ne("")
    ]

    return int(
        values.nunique()
    )


def main():

    separator()

    print(
        "PHASE 3.2.3 — "
        "CROSS-ROLE IDENTITY & COMPANY-DUPLICATE AUDIT"
    )

    separator()

    # =========================================================================
    # 1. Discover source files
    # =========================================================================

    company_files = sorted(
        RAW_DIR.glob(
            "companies*.csv"
        )
    )

    investor_files = sorted(
        RAW_DIR.glob(
            "CRUNCHBASE_investor*.csv"
        )
    )

    if not company_files:

        raise FileNotFoundError(
            "No companies*.csv files found."
        )

    if len(investor_files) != 1:

        raise ValueError(
            "Expected exactly one investor file; "
            f"found {len(investor_files)}."
        )

    investor_path = (
        investor_files[0]
    )

    print(
        f"\nCompany chunks: "
        f"{len(company_files)}"
    )

    print(
        f"Investor file:  "
        f"{investor_path.name}"
    )

    # =========================================================================
    # 2. Load company ID + source-file index
    # =========================================================================

    print(
        "\nLoading company IDs for "
        "cross-chunk duplicate audit..."
    )

    company_id_frames = []

    for file_index, path in enumerate(
        company_files
    ):

        chunk = pd.read_csv(
            path,
            usecols=["id"],
            dtype={
                "id": "string"
            },
        )

        chunk["id"] = (
            clean_string(
                chunk["id"]
            )
        )

        chunk = (
            chunk[
                chunk["id"]
                .notna()
                &
                chunk["id"]
                .ne("")
            ]
            .copy()
        )

        chunk[
            "source_file_index"
        ] = np.int16(
            file_index
        )

        company_id_frames.append(
            chunk
        )

    company_ids = pd.concat(
        company_id_frames,
        ignore_index=True,
    )

    company_id_frames.clear()

    total_company_rows = len(
        company_ids
    )

    unique_company_ids = (
        company_ids["id"]
        .nunique()
    )

    # =========================================================================
    # 3. Identify true cross-file duplicate company IDs
    # =========================================================================

    id_counts = (
        company_ids[
            "id"
        ]
        .value_counts()
    )

    duplicate_id_counts = (
        id_counts[
            id_counts > 1
        ]
    )

    duplicate_ids = set(
        duplicate_id_counts.index
    )

    duplicate_company_id_count = (
        len(
            duplicate_id_counts
        )
    )

    duplicate_rows_involved = int(
        duplicate_id_counts.sum()
    )

    excess_duplicate_occurrences = (
        total_company_rows
        - unique_company_ids
    )

    max_multiplicity = int(
        duplicate_id_counts.max()
        if len(
            duplicate_id_counts
        ) > 0
        else 1
    )

    # -------------------------------------------------------------------------
    # Multiplicity summary
    # -------------------------------------------------------------------------

    multiplicity_summary = (
        duplicate_id_counts
        .value_counts()
        .sort_index()
        .rename_axis(
            "occurrences_per_company_id"
        )
        .reset_index(
            name="company_id_count"
        )
    )

    multiplicity_summary[
        "rows_represented"
    ] = (
        multiplicity_summary[
            "occurrences_per_company_id"
        ]
        *
        multiplicity_summary[
            "company_id_count"
        ]
    )

    # =========================================================================
    # 4. Reload only duplicate company records with key attributes
    # =========================================================================

    duplicate_records = []

    for file_index, path in enumerate(
        company_files
    ):

        header = (
            pd.read_csv(
                path,
                nrows=0,
            )
            .columns
            .tolist()
        )

        usecols = [
            c
            for c in (
                KEY_COMPANY_ATTRIBUTES
            )
            if c in header
        ]

        chunk = pd.read_csv(
            path,
            usecols=usecols,
            dtype="string",
            low_memory=False,
        )

        chunk["id"] = (
            clean_string(
                chunk["id"]
            )
        )

        subset = (
            chunk[
                chunk["id"]
                .isin(
                    duplicate_ids
                )
            ]
            .copy()
        )

        if len(subset) > 0:

            subset[
                "source_file"
            ] = path.name

            duplicate_records.append(
                subset
            )

    if duplicate_records:

        duplicate_records = (
            pd.concat(
                duplicate_records,
                ignore_index=True,
            )
        )

    else:

        duplicate_records = pd.DataFrame(
            columns=(
                KEY_COMPANY_ATTRIBUTES
                + [
                    "source_file"
                ]
            )
        )

    # =========================================================================
    # 5. Attribute consistency of duplicate company records
    # =========================================================================

    duplicate_detail_rows = []

    for company_id, group in (
        duplicate_records.groupby(
            "id",
            observed=True,
        )
    ):

        row = {
            "company_id": (
                company_id
            ),

            "occurrences": (
                len(group)
            ),

            "source_file_count": (
                group[
                    "source_file"
                ]
                .nunique()
            ),

            "source_files": (
                " | ".join(
                    sorted(
                        group[
                            "source_file"
                        ]
                        .dropna()
                        .unique()
                    )
                )
            ),
        }

        for attribute in [
            "name",
            "link",
            "website",
            "company_type",
        ]:

            if attribute in (
                group.columns
            ):

                distinct_count = (
                    nunique_nonmissing(
                        group[
                            attribute
                        ]
                    )
                )

                row[
                    f"{attribute}_distinct_nonmissing"
                ] = (
                    distinct_count
                )

                row[
                    f"{attribute}_consistent"
                ] = (
                    distinct_count <= 1
                )

            else:

                row[
                    f"{attribute}_distinct_nonmissing"
                ] = np.nan

                row[
                    f"{attribute}_consistent"
                ] = np.nan

        duplicate_detail_rows.append(
            row
        )

    duplicate_details = pd.DataFrame(
        duplicate_detail_rows
    )

    # =========================================================================
    # 6. Load investor registry
    # =========================================================================

    investor_header = (
        pd.read_csv(
            investor_path,
            nrows=0,
        )
        .columns
        .tolist()
    )

    investor_usecols = [
        c
        for c in (
            KEY_INVESTOR_ATTRIBUTES
        )
        if c in investor_header
    ]

    investors = pd.read_csv(
        investor_path,
        usecols=investor_usecols,
        dtype="string",
        low_memory=False,
    )

    investors["id"] = (
        clean_string(
            investors["id"]
        )
    )

    investors = (
        investors[
            investors["id"]
            .notna()
            &
            investors["id"]
            .ne("")
        ]
        .copy()
    )

    investor_id_set = set(
        investors[
            "id"
        ]
        .unique()
    )

    # =========================================================================
    # 7. Extract company records whose ID exists in investor registry
    # =========================================================================

    shared_company_records = []

    for path in company_files:

        header = (
            pd.read_csv(
                path,
                nrows=0,
            )
            .columns
            .tolist()
        )

        usecols = [
            c
            for c in (
                KEY_COMPANY_ATTRIBUTES
            )
            if c in header
        ]

        chunk = pd.read_csv(
            path,
            usecols=usecols,
            dtype="string",
            low_memory=False,
        )

        chunk["id"] = (
            clean_string(
                chunk["id"]
            )
        )

        subset = (
            chunk[
                chunk["id"]
                .isin(
                    investor_id_set
                )
            ]
            .copy()
        )

        if len(subset) > 0:

            subset[
                "source_file"
            ] = (
                path.name
            )

            shared_company_records.append(
                subset
            )

    shared_companies = pd.concat(
        shared_company_records,
        ignore_index=True,
    )

    # -------------------------------------------------------------------------
    # Multiple company-export copies may exist.
    #
    # Attribute disagreement across those copies has already been audited.
    # For cross-registry matching, use one row per underlying company ID.
    # -------------------------------------------------------------------------

    shared_companies_one = (
        shared_companies
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
            keep="first",
        )
        .copy()
    )

    shared = investors.merge(
        shared_companies_one,
        on="id",
        how="inner",
        suffixes=(
            "_investor",
            "_company",
        ),
        validate="one_to_one",
    )

    # =========================================================================
    # 8. Shared-ID attribute agreement
    # =========================================================================

    agreement_rows = []

    disagreement_mask = pd.Series(
        False,
        index=shared.index,
    )

    for attribute in [
        "name",
        "link",
        "website",
        "company_type",
    ]:

        inv_col = (
            f"{attribute}_investor"
        )

        comp_col = (
            f"{attribute}_company"
        )

        if (
            inv_col
            not in shared.columns
            or
            comp_col
            not in shared.columns
        ):

            continue

        left = (
            normalized_compare(
                shared[
                    inv_col
                ]
            )
        )

        right = (
            normalized_compare(
                shared[
                    comp_col
                ]
            )
        )

        both_nonmissing = (
            left.notna()
            &
            right.notna()
        )

        exact_match = (
            both_nonmissing
            &
            left.eq(
                right
            )
        )

        disagreement = (
            both_nonmissing
            &
            ~left.eq(
                right
            )
        )

        disagreement_mask = (
            disagreement_mask
            |
            disagreement
        )

        agreement_rows.append(
            {
                "attribute": (
                    attribute
                ),

                "shared_ids": (
                    len(shared)
                ),

                "both_nonmissing": int(
                    both_nonmissing.sum()
                ),

                "exact_matches": int(
                    exact_match.sum()
                ),

                "disagreements": int(
                    disagreement.sum()
                ),

                "exact_match_pct_when_both_nonmissing": pct(
                    exact_match.sum(),
                    both_nonmissing.sum(),
                ),
            }
        )

    agreement_summary = (
        pd.DataFrame(
            agreement_rows
        )
    )

    disagreement_columns = [
        c
        for c in [
            "id",
            "name_investor",
            "name_company",
            "link_investor",
            "link_company",
            "website_investor",
            "website_company",
            "investor_type",
            "company_type_investor",
            "company_type_company",
            "source_file",
        ]
        if c in shared.columns
    ]

    disagreements = (
        shared.loc[
            disagreement_mask,
            disagreement_columns,
        ]
        .head(
            500
        )
        .copy()
    )

    # =========================================================================
    # 9. Investor-type overlap with companies
    # =========================================================================

    company_id_set = set(
        company_ids[
            "id"
        ]
        .unique()
    )

    investors[
        "appears_in_companies"
    ] = (
        investors[
            "id"
        ]
        .isin(
            company_id_set
        )
    )

    if (
        "investor_type"
        in investors.columns
    ):

        investor_type_series = (
            investors[
                "investor_type"
            ]
            .fillna(
                "<missing>"
            )
            .replace(
                "",
                "<missing>",
            )
        )

        temp = (
            investors
            .assign(
                investor_type_group=(
                    investor_type_series
                )
            )
        )

        investor_type_summary = (
            temp.groupby(
                "investor_type_group",
                observed=True,
            )
            .agg(
                total_investor_ids=(
                    "id",
                    "nunique",
                ),

                ids_in_companies=(
                    "appears_in_companies",
                    "sum",
                ),
            )
            .reset_index()
        )

        investor_type_summary[
            "ids_not_in_companies"
        ] = (
            investor_type_summary[
                "total_investor_ids"
            ]
            -
            investor_type_summary[
                "ids_in_companies"
            ]
        )

        investor_type_summary[
            "company_overlap_pct"
        ] = (
            investor_type_summary[
                "ids_in_companies"
            ]
            /
            investor_type_summary[
                "total_investor_ids"
            ]
            * 100
        )

        investor_type_summary = (
            investor_type_summary
            .sort_values(
                "total_investor_ids",
                ascending=False,
            )
        )

    else:

        investor_type_summary = (
            pd.DataFrame()
        )

    # =========================================================================
    # 10. Phase-1 experimental cross-role identity
    # =========================================================================

    phase1 = pd.read_parquet(
        PHASE1_PATH,
        columns=[
            "interaction_id",
            "investor_id",
            "startup_id",
        ],
    )

    phase1[
        "investor_id"
    ] = (
        clean_string(
            phase1[
                "investor_id"
            ]
        )
    )

    phase1[
        "startup_id"
    ] = (
        clean_string(
            phase1[
                "startup_id"
            ]
        )
    )

    canonical_investors = set(
        phase1[
            "investor_id"
        ]
        .dropna()
        .unique()
    )

    canonical_startups = set(
        phase1[
            "startup_id"
        ]
        .dropna()
        .unique()
    )

    dual_role_ids = (
        canonical_investors
        &
        canonical_startups
    )

    self_interactions = (
        phase1[
            "investor_id"
        ]
        .eq(
            phase1[
                "startup_id"
            ]
        )
    )

    cross_role_summary = pd.DataFrame(
        [
            {
                "metric": (
                    "canonical_investor_ids"
                ),
                "value": (
                    len(
                        canonical_investors
                    )
                ),
            },
            {
                "metric": (
                    "canonical_startup_ids"
                ),
                "value": (
                    len(
                        canonical_startups
                    )
                ),
            },
            {
                "metric": (
                    "ids_with_both_investor_and_startup_role"
                ),
                "value": (
                    len(
                        dual_role_ids
                    )
                ),
            },
            {
                "metric": (
                    "dual_role_share_of_investors_pct"
                ),
                "value": pct(
                    len(
                        dual_role_ids
                    ),
                    len(
                        canonical_investors
                    ),
                ),
            },
            {
                "metric": (
                    "dual_role_share_of_startups_pct"
                ),
                "value": pct(
                    len(
                        dual_role_ids
                    ),
                    len(
                        canonical_startups
                    ),
                ),
            },
            {
                "metric": (
                    "events_where_investor_id_equals_startup_id"
                ),
                "value": int(
                    self_interactions.sum()
                ),
            },
        ]
    )

    dual_role_output = pd.DataFrame(
        {
            "crunchbase_id":
                sorted(
                    dual_role_ids
                )
        }
    )

    # =========================================================================
    # 11. Save audit outputs
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    multiplicity_summary.to_csv(
        DUPLICATE_SUMMARY_OUTPUT,
        index=False,
    )

    duplicate_details.to_csv(
        DUPLICATE_DETAILS_OUTPUT,
        index=False,
    )

    agreement_summary.to_csv(
        SHARED_ID_ATTRIBUTE_SUMMARY_OUTPUT,
        index=False,
    )

    disagreements.to_csv(
        SHARED_ID_DISAGREEMENTS_OUTPUT,
        index=False,
    )

    investor_type_summary.to_csv(
        INVESTOR_TYPE_OUTPUT,
        index=False,
    )

    cross_role_summary.to_csv(
        CROSS_ROLE_OUTPUT,
        index=False,
    )

    dual_role_output.to_csv(
        DUAL_ROLE_IDS_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 12. Figure — duplicate multiplicity
    # =========================================================================

    if len(
        multiplicity_summary
    ) > 0:

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.bar(
            multiplicity_summary[
                "occurrences_per_company_id"
            ]
            .astype(str),
            multiplicity_summary[
                "company_id_count"
            ],
        )

        ax.set_xlabel(
            "Occurrences per duplicated company ID"
        )

        ax.set_ylabel(
            "Number of company IDs"
        )

        ax.set_title(
            "Cross-Chunk Company-ID Multiplicity"
        )

        fig.tight_layout()

        fig.savefig(
            DUPLICATE_FIGURE,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    # =========================================================================
    # 13. Figure — investor overlap by investor type
    # =========================================================================

    if len(
        investor_type_summary
    ) > 0:

        plot_data = (
            investor_type_summary
            .head(20)
            .sort_values(
                "company_overlap_pct",
                ascending=True,
            )
        )

        fig, ax = plt.subplots(
            figsize=(
                10,
                max(
                    5,
                    len(plot_data)
                    * 0.42,
                ),
            )
        )

        ax.barh(
            plot_data[
                "investor_type_group"
            ],
            plot_data[
                "company_overlap_pct"
            ],
        )

        ax.set_xlabel(
            "Investor IDs also found in companies (%)"
        )

        ax.set_title(
            "Investor → Companies ID Overlap by Investor Type"
        )

        fig.tight_layout()

        fig.savefig(
            INVESTOR_TYPE_FIGURE,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    # =========================================================================
    # 14. Figure — Phase-1 cross-role counts
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    labels = [
        "Investor IDs",
        "Startup IDs",
        "Dual-role IDs",
    ]

    values = [
        len(
            canonical_investors
        ),
        len(
            canonical_startups
        ),
        len(
            dual_role_ids
        ),
    ]

    ax.bar(
        labels,
        values,
    )

    ax.set_ylabel(
        "Unique IDs"
    )

    ax.set_title(
        "Phase-1 Investor / Startup Cross-Role Identity"
    )

    fig.tight_layout()

    fig.savefig(
        CROSS_ROLE_FIGURE,
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
        "COMPANY CROSS-CHUNK DUPLICATION"
    )

    separator("-")

    print(
        f"Company ID rows:                  "
        f"{total_company_rows:,}"
    )

    print(
        f"Globally unique company IDs:      "
        f"{unique_company_ids:,}"
    )

    print(
        f"Distinct duplicated company IDs:  "
        f"{duplicate_company_id_count:,}"
    )

    print(
        f"Rows involving duplicated IDs:    "
        f"{duplicate_rows_involved:,}"
    )

    print(
        f"Excess duplicate occurrences:     "
        f"{excess_duplicate_occurrences:,}"
    )

    print(
        f"Maximum ID multiplicity:           "
        f"{max_multiplicity:,}"
    )

    print(
        "\nMultiplicity distribution:"
    )

    print(
        multiplicity_summary.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------

    separator("-")

    print(
        "DUPLICATE-RECORD ATTRIBUTE CONSISTENCY"
    )

    separator("-")

    if len(
        duplicate_details
    ) > 0:

        for attribute in [
            "name",
            "link",
            "website",
            "company_type",
        ]:

            col = (
                f"{attribute}_consistent"
            )

            if col in (
                duplicate_details.columns
            ):

                valid = (
                    duplicate_details[
                        col
                    ]
                    .dropna()
                )

                inconsistent = int(
                    (
                        valid
                        == False
                    )
                    .sum()
                )

                print(
                    f"{attribute:15s} "
                    f"inconsistent duplicate IDs: "
                    f"{inconsistent:,}"
                )

    # -------------------------------------------------------------------------

    separator("-")

    print(
        "INVESTOR / COMPANY SHARED-ID ATTRIBUTE AGREEMENT"
    )

    separator("-")

    print(
        f"Shared IDs audited: "
        f"{len(shared):,}"
    )

    if len(
        agreement_summary
    ) > 0:

        print(
            agreement_summary.to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    # -------------------------------------------------------------------------

    separator("-")

    print(
        "INVESTOR-TYPE OVERLAP"
    )

    separator("-")

    if len(
        investor_type_summary
    ) > 0:

        print(
            investor_type_summary
            .head(30)
            .to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.2f}",
            )
        )

    # -------------------------------------------------------------------------

    separator("-")

    print(
        "PHASE-1 CROSS-ROLE IDENTITY"
    )

    separator("-")

    print(
        f"Canonical investor IDs:        "
        f"{len(canonical_investors):,}"
    )

    print(
        f"Canonical startup IDs:         "
        f"{len(canonical_startups):,}"
    )

    print(
        f"IDs appearing in BOTH roles:   "
        f"{len(dual_role_ids):,}"
    )

    print(
        f"Share of investor universe:    "
        f"{pct(len(dual_role_ids), len(canonical_investors)):.4f}%"
    )

    print(
        f"Share of startup universe:     "
        f"{pct(len(dual_role_ids), len(canonical_startups)):.4f}%"
    )

    print(
        f"Self-ID investment events:     "
        f"{int(self_interactions.sum()):,}"
    )

    separator()

    print(
        "PHASE 3.2.3 AUDIT COMPLETE"
    )

    separator()

    print(
        f"""
Outputs written to:

{DUPLICATE_SUMMARY_OUTPUT}
{DUPLICATE_DETAILS_OUTPUT}
{SHARED_ID_ATTRIBUTE_SUMMARY_OUTPUT}
{SHARED_ID_DISAGREEMENTS_OUTPUT}
{INVESTOR_TYPE_OUTPUT}
{CROSS_ROLE_OUTPUT}
{DUAL_ROLE_IDS_OUTPUT}

Figures written to:

{DUPLICATE_FIGURE}
{INVESTOR_TYPE_FIGURE}
{CROSS_ROLE_FIGURE}


IMPORTANT

This audit does NOT merge investor and startup nodes.

The results will determine whether shared Crunchbase IDs should be treated as:

1. one underlying real-world identity with two role-specific graph nodes, or
2. unrelated/ambiguous records requiring separate treatment.

No graph has been constructed.
"""
    )


if __name__ == "__main__":
    main()