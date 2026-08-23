from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PHASE 3.2.7 — CROSS-DATASET REFERENCE & BRIDGE INTEGRITY AUDIT
# =============================================================================
#
# PURPOSE
# -------
#
# Test the strongest candidate cross-dataset references discovered in 3.2.6.
#
# Primary questions:
#
# 1. Do companies.founder_links resolve exactly to people.link?
# 2. How many canonical investor/startup role nodes have founder information?
# 3. What graph-pair volume would shared-founder projection create?
# 4. Do contacts IDs correspond to people IDs?
# 5. Are organization_uuid / organization_permalink truly empty in contacts?
# 6. Do School IDs share the Company / Investor identity namespace?
# 7. Are the two duplicated People IDs harmless cross-chunk overlap?
#
# NO GRAPH EDGES ARE MATERIALIZED.
# =============================================================================


RAW_DIR = Path("data/raw")

INTERACTIONS_PATH = Path(
    "data/processed/interactions.parquet"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_3/audits"
)

FIGURE_DIR = Path(
    "data/experimental/phase_3/figures"
)


# =============================================================================
# OUTPUTS
# =============================================================================


BRIDGE_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "cross_dataset_reference_bridge_summary.csv"
)

FOUNDER_COVERAGE_OUTPUT = (
    OUTPUT_DIR
    / "canonical_role_founder_link_coverage.csv"
)

FOUNDER_MATCH_OUTPUT = (
    OUTPUT_DIR
    / "founder_link_people_resolution_summary.csv"
)

FOUNDER_DEGREE_OUTPUT = (
    OUTPUT_DIR
    / "founder_role_degree_distribution.csv"
)

FOUNDER_PROJECTION_OUTPUT = (
    OUTPUT_DIR
    / "shared_founder_projection_size_summary.csv"
)

PEOPLE_DUPLICATE_OUTPUT = (
    OUTPUT_DIR
    / "people_cross_chunk_duplicate_details.csv"
)

CONTACT_REFERENCE_OUTPUT = (
    OUTPUT_DIR
    / "contact_reference_integrity_summary.csv"
)

SCHOOL_NAMESPACE_OUTPUT = (
    OUTPUT_DIR
    / "school_identity_namespace_overlap.csv"
)


FOUNDER_COVERAGE_FIGURE = (
    FIGURE_DIR
    / "canonical_role_founder_link_coverage.png"
)

FOUNDER_DEGREE_FIGURE = (
    FIGURE_DIR
    / "founder_role_degree_distribution.png"
)

BRIDGE_FIGURE = (
    FIGURE_DIR
    / "cross_dataset_reference_bridge_coverage.png"
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


def normalize_url_value(value):

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

    normalized = []

    for url in matches:

        url = (
            normalize_url_value(
                url
            )
        )

        if url is not None:

            normalized.append(
                url
            )

    return list(
        dict.fromkeys(
            normalized
        )
    )


# =============================================================================
# MAIN
# =============================================================================


def main():

    separator()

    print(
        "PHASE 3.2.7 — "
        "CROSS-DATASET REFERENCE & BRIDGE INTEGRITY AUDIT"
    )

    separator()

    # =========================================================================
    # 1. Canonical experimental role universes
    # =========================================================================

    interactions = pd.read_parquet(
        INTERACTIONS_PATH,
        columns=[
            "investor_id",
            "startup_id",
        ],
    )

    canonical_investor_ids = set(
        clean_string(
            interactions[
                "investor_id"
            ]
        )
        .dropna()
        .unique()
    )

    canonical_startup_ids = set(
        clean_string(
            interactions[
                "startup_id"
            ]
        )
        .dropna()
        .unique()
    )

    canonical_underlying_ids = (
        canonical_investor_ids
        |
        canonical_startup_ids
    )

    print(
        f"\nCanonical investors: "
        f"{len(canonical_investor_ids):,}"
    )

    print(
        f"Canonical startups:  "
        f"{len(canonical_startup_ids):,}"
    )

    # =========================================================================
    # 2. Extract founder links only for relevant Company records
    # =========================================================================

    company_files = sorted(
        RAW_DIR.glob(
            "companies*.csv"
        )
    )

    founder_rows = []

    for path in company_files:

        chunk = pd.read_csv(
            path,
            usecols=[
                "id",
                "founders",
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

        founder_rows.append(
            subset
        )

    canonical_company_records = (
        pd.concat(
            founder_rows,
            ignore_index=True,
        )
    )

    # Overlapping company chunks can duplicate the same record.
    canonical_company_records = (
        canonical_company_records
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

    canonical_company_records[
        "is_investor_role"
    ] = (
        canonical_company_records[
            "id"
        ]
        .isin(
            canonical_investor_ids
        )
    )

    canonical_company_records[
        "is_startup_role"
    ] = (
        canonical_company_records[
            "id"
        ]
        .isin(
            canonical_startup_ids
        )
    )

    canonical_company_records[
        "founder_urls"
    ] = (
        canonical_company_records[
            "founder_links"
        ]
        .apply(
            extract_person_urls
        )
    )

    canonical_company_records[
        "founder_url_count"
    ] = (
        canonical_company_records[
            "founder_urls"
        ]
        .apply(
            len
        )
    )

    canonical_company_records[
        "has_founder_url"
    ] = (
        canonical_company_records[
            "founder_url_count"
        ]
        > 0
    )

    # =========================================================================
    # 3. Role coverage
    # =========================================================================

    investor_company_records = (
        canonical_company_records[
            canonical_company_records[
                "is_investor_role"
            ]
        ]
    )

    startup_company_records = (
        canonical_company_records[
            canonical_company_records[
                "is_startup_role"
            ]
        ]
    )

    founder_coverage = pd.DataFrame(
        [
            {
                "role":
                    "investor",

                "canonical_role_ids":
                    len(
                        canonical_investor_ids
                    ),

                "ids_with_company_record":
                    len(
                        investor_company_records
                    ),

                "ids_with_founder_url":
                    int(
                        investor_company_records[
                            "has_founder_url"
                        ]
                        .sum()
                    ),
            },

            {
                "role":
                    "startup",

                "canonical_role_ids":
                    len(
                        canonical_startup_ids
                    ),

                "ids_with_company_record":
                    len(
                        startup_company_records
                    ),

                "ids_with_founder_url":
                    int(
                        startup_company_records[
                            "has_founder_url"
                        ]
                        .sum()
                    ),
            },
        ]
    )

    founder_coverage[
        "company_record_coverage_pct"
    ] = (
        founder_coverage[
            "ids_with_company_record"
        ]
        /
        founder_coverage[
            "canonical_role_ids"
        ]
        * 100
    )

    founder_coverage[
        "founder_url_coverage_of_role_pct"
    ] = (
        founder_coverage[
            "ids_with_founder_url"
        ]
        /
        founder_coverage[
            "canonical_role_ids"
        ]
        * 100
    )

    founder_coverage[
        "founder_url_coverage_when_company_exists_pct"
    ] = (
        founder_coverage[
            "ids_with_founder_url"
        ]
        /
        founder_coverage[
            "ids_with_company_record"
        ]
        * 100
    )

    # =========================================================================
    # 4. Explode founder URLs into role-node references
    # =========================================================================

    role_founder_rows = []

    for _, row in (
        canonical_company_records[
            canonical_company_records[
                "has_founder_url"
            ]
        ]
        .iterrows()
    ):

        underlying_id = (
            row[
                "id"
            ]
        )

        urls = (
            row[
                "founder_urls"
            ]
        )

        if row[
            "is_investor_role"
        ]:

            for url in urls:

                role_founder_rows.append(
                    {
                        "role_node_id":
                            (
                                "investor::"
                                + underlying_id
                            ),

                        "role":
                            "investor",

                        "underlying_entity_id":
                            underlying_id,

                        "person_url":
                            url,
                    }
                )

        if row[
            "is_startup_role"
        ]:

            for url in urls:

                role_founder_rows.append(
                    {
                        "role_node_id":
                            (
                                "startup::"
                                + underlying_id
                            ),

                        "role":
                            "startup",

                        "underlying_entity_id":
                            underlying_id,

                        "person_url":
                            url,
                    }
                )

    role_founders = (
        pd.DataFrame(
            role_founder_rows
        )
        .drop_duplicates()
    )

    unique_founder_urls = set(
        role_founders[
            "person_url"
        ]
        .unique()
    )

    # =========================================================================
    # 5. People registry + exact founder URL matching
    # =========================================================================

    people_files = sorted(
        RAW_DIR.glob(
            "people*.csv"
        )
    )

    people_id_frames = []

    matched_people_frames = []

    for path in people_files:

        people = pd.read_csv(
            path,
            usecols=[
                "id",
                "link",
                "name",
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
            "normalized_link"
        ] = (
            people[
                "link"
            ]
            .apply(
                normalize_url_value
            )
        )

        people[
            "source_file"
        ] = (
            path.name
        )

        people_id_frames.append(
            people[
                [
                    "id",
                    "name",
                    "normalized_link",
                    "source_file",
                ]
            ]
        )

        matched = (
            people[
                people[
                    "normalized_link"
                ]
                .isin(
                    unique_founder_urls
                )
            ]
            .copy()
        )

        if len(
            matched
        ) > 0:

            matched_people_frames.append(
                matched
            )

    people_all = pd.concat(
        people_id_frames,
        ignore_index=True,
    )

    # =========================================================================
    # 6. People duplicate-ID integrity
    # =========================================================================

    people_id_counts = (
        people_all[
            "id"
        ]
        .value_counts()
    )

    duplicated_people_ids = set(
        people_id_counts[
            people_id_counts
            > 1
        ]
        .index
    )

    people_duplicate_details = (
        people_all[
            people_all[
                "id"
            ]
            .isin(
                duplicated_people_ids
            )
        ]
        .sort_values(
            [
                "id",
                "source_file",
            ]
        )
        .copy()
    )

    # =========================================================================
    # 7. Founder URL -> People resolution
    # =========================================================================

    if matched_people_frames:

        matched_people = pd.concat(
            matched_people_frames,
            ignore_index=True,
        )

    else:

        matched_people = pd.DataFrame(
            columns=[
                "id",
                "link",
                "name",
                "normalized_link",
                "source_file",
            ]
        )

    person_link_resolution = (
        matched_people.groupby(
            "normalized_link",
            observed=True,
        )
        .agg(
            matched_rows=(
                "id",
                "size",
            ),

            unique_person_ids=(
                "id",
                "nunique",
            ),
        )
        .reset_index()
    )

    resolved_unique_links = set(
        person_link_resolution[
            person_link_resolution[
                "unique_person_ids"
            ]
            == 1
        ][
            "normalized_link"
        ]
    )

    ambiguous_links = set(
        person_link_resolution[
            person_link_resolution[
                "unique_person_ids"
            ]
            > 1
        ][
            "normalized_link"
        ]
    )

    unresolved_links = (
        unique_founder_urls
        -
        set(
            person_link_resolution[
                "normalized_link"
            ]
        )
    )

    founder_match_summary = pd.DataFrame(
        [
            {
                "metric":
                    "unique_founder_urls",

                "value":
                    len(
                        unique_founder_urls
                    ),
            },

            {
                "metric":
                    "exactly_resolved_to_one_people_id",

                "value":
                    len(
                        resolved_unique_links
                    ),
            },

            {
                "metric":
                    "ambiguous_people_links",

                "value":
                    len(
                        ambiguous_links
                    ),
            },

            {
                "metric":
                    "unresolved_people_links",

                "value":
                    len(
                        unresolved_links
                    ),
            },

            {
                "metric":
                    "exact_resolution_pct",

                "value":
                    pct(
                        len(
                            resolved_unique_links
                        ),
                        len(
                            unique_founder_urls
                        ),
                    ),
            },
        ]
    )

    # =========================================================================
    # 8. Attach resolved Person IDs to role-founder references
    # =========================================================================

    unique_people_map = (
        matched_people[
            matched_people[
                "normalized_link"
            ]
            .isin(
                resolved_unique_links
            )
        ]
        .sort_values(
            [
                "normalized_link",
                "source_file",
            ]
        )
        .drop_duplicates(
            subset=[
                "normalized_link"
            ],
            keep="first",
        )[
            [
                "normalized_link",
                "id",
                "name",
            ]
        ]
        .rename(
            columns={
                "normalized_link":
                    "person_url",

                "id":
                    "person_id",

                "name":
                    "person_name",
            }
        )
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
    # 9. Founder degree and projected pair counts
    # =========================================================================

    founder_degree = (
        resolved_role_founders.groupby(
            "person_id",
            observed=True,
        )
        .agg(
            role_node_degree=(
                "role_node_id",
                "nunique",
            ),

            investor_role_degree=(
                "role",
                lambda x:
                    int(
                        (
                            x
                            == "investor"
                        )
                        .sum()
                    ),
            ),

            startup_role_degree=(
                "role",
                lambda x:
                    int(
                        (
                            x
                            == "startup"
                        )
                        .sum()
                    ),
            ),
        )
        .reset_index()
    )

    founder_degree[
        "projected_all_pairs"
    ] = (
        founder_degree[
            "role_node_degree"
        ]
        *
        (
            founder_degree[
                "role_node_degree"
            ]
            - 1
        )
        // 2
    )

    founder_degree[
        "projected_investor_investor_pairs"
    ] = (
        founder_degree[
            "investor_role_degree"
        ]
        *
        (
            founder_degree[
                "investor_role_degree"
            ]
            - 1
        )
        // 2
    )

    founder_degree[
        "projected_startup_startup_pairs"
    ] = (
        founder_degree[
            "startup_role_degree"
        ]
        *
        (
            founder_degree[
                "startup_role_degree"
            ]
            - 1
        )
        // 2
    )

    founder_degree[
        "projected_investor_startup_pairs"
    ] = (
        founder_degree[
            "investor_role_degree"
        ]
        *
        founder_degree[
            "startup_role_degree"
        ]
    )

    projection_summary = pd.DataFrame(
        [
            {
                "metric":
                    "resolved_people_with_at_least_one_role_node",

                "value":
                    len(
                        founder_degree
                    ),
            },

            {
                "metric":
                    "people_linking_at_least_two_role_nodes",

                "value":
                    int(
                        (
                            founder_degree[
                                "role_node_degree"
                            ]
                            >= 2
                        )
                        .sum()
                    ),
            },

            {
                "metric":
                    "max_role_node_degree_of_one_person",

                "value":
                    int(
                        founder_degree[
                            "role_node_degree"
                        ]
                        .max()
                        if len(
                            founder_degree
                        ) > 0
                        else 0
                    ),
            },

            {
                "metric":
                    "projected_all_shared_founder_pairs",

                "value":
                    int(
                        founder_degree[
                            "projected_all_pairs"
                        ]
                        .sum()
                    ),
            },

            {
                "metric":
                    "projected_investor_investor_pairs",

                "value":
                    int(
                        founder_degree[
                            "projected_investor_investor_pairs"
                        ]
                        .sum()
                    ),
            },

            {
                "metric":
                    "projected_investor_startup_pairs",

                "value":
                    int(
                        founder_degree[
                            "projected_investor_startup_pairs"
                        ]
                        .sum()
                    ),
            },

            {
                "metric":
                    "projected_startup_startup_pairs",

                "value":
                    int(
                        founder_degree[
                            "projected_startup_startup_pairs"
                        ]
                        .sum()
                    ),
            },
        ]
    )

    # =========================================================================
    # 10. Contacts -> People / Organization reference audit
    # =========================================================================

    contact_files = sorted(
        RAW_DIR.glob(
            "CRUNCHBASE_contact*.csv"
        )
    )

    if len(
        contact_files
    ) != 1:

        raise ValueError(
            "Expected exactly one contact file."
        )

    contacts = pd.read_csv(
        contact_files[0],
        usecols=[
            "id",
            "organization_uuid",
            "organization_permalink",
            "organization_name",
        ],
        dtype="string",
        low_memory=False,
    )

    for column in [
        "id",
        "organization_uuid",
        "organization_permalink",
        "organization_name",
    ]:

        contacts[
            column
        ] = (
            clean_string(
                contacts[
                    column
                ]
            )
            .replace(
                "",
                pd.NA,
            )
        )

    people_id_set = set(
        people_all[
            "id"
        ]
        .dropna()
        .unique()
    )

    contact_id_set = set(
        contacts[
            "id"
        ]
        .dropna()
        .unique()
    )

    contact_people_overlap = (
        contact_id_set
        &
        people_id_set
    )

    organization_uuid_nonmissing = int(
        contacts[
            "organization_uuid"
        ]
        .notna()
        .sum()
    )

    organization_permalink_nonmissing = int(
        contacts[
            "organization_permalink"
        ]
        .notna()
        .sum()
    )

    contact_reference_summary = pd.DataFrame(
        [
            {
                "metric":
                    "contact_rows",

                "value":
                    len(
                        contacts
                    ),
            },

            {
                "metric":
                    "unique_contact_ids",

                "value":
                    len(
                        contact_id_set
                    ),
            },

            {
                "metric":
                    "contact_ids_found_in_people",

                "value":
                    len(
                        contact_people_overlap
                    ),
            },

            {
                "metric":
                    "contact_id_to_people_coverage_pct",

                "value":
                    pct(
                        len(
                            contact_people_overlap
                        ),
                        len(
                            contact_id_set
                        ),
                    ),
            },

            {
                "metric":
                    "organization_uuid_nonmissing_rows",

                "value":
                    organization_uuid_nonmissing,
            },

            {
                "metric":
                    "organization_permalink_nonmissing_rows",

                "value":
                    organization_permalink_nonmissing,
            },

            {
                "metric":
                    "organization_name_nonmissing_rows",

                "value":
                    int(
                        contacts[
                            "organization_name"
                        ]
                        .notna()
                        .sum()
                    ),
            },
        ]
    )

    # =========================================================================
    # 11. Schools namespace overlap
    # =========================================================================

    school_files = sorted(
        RAW_DIR.glob(
            "CRUNCHBASE_school*.csv"
        )
    )

    if len(
        school_files
    ) != 1:

        raise ValueError(
            "Expected exactly one school file."
        )

    schools = pd.read_csv(
        school_files[0],
        usecols=[
            "id",
            "name",
        ],
        dtype="string",
        low_memory=False,
    )

    schools[
        "id"
    ] = clean_string(
        schools[
            "id"
        ]
    )

    school_ids = set(
        schools[
            "id"
        ]
        .dropna()
        .unique()
    )

    # Scan companies without storing all 4.7M IDs.
    school_ids_found_in_companies = set()

    for path in company_files:

        chunk = pd.read_csv(
            path,
            usecols=[
                "id"
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

        matches = set(
            chunk.loc[
                chunk[
                    "id"
                ]
                .isin(
                    school_ids
                ),
                "id",
            ]
            .dropna()
            .unique()
        )

        school_ids_found_in_companies.update(
            matches
        )

    investor_file = sorted(
        RAW_DIR.glob(
            "CRUNCHBASE_investor*.csv"
        )
    )[0]

    investor_registry = pd.read_csv(
        investor_file,
        usecols=[
            "id"
        ],
        dtype="string",
        low_memory=False,
    )

    investor_registry_ids = set(
        clean_string(
            investor_registry[
                "id"
            ]
        )
        .dropna()
        .unique()
    )

    school_ids_found_in_investors = (
        school_ids
        &
        investor_registry_ids
    )

    school_namespace_summary = pd.DataFrame(
        [
            {
                "reference_population":
                    "companies",

                "school_ids":
                    len(
                        school_ids
                    ),

                "matched_school_ids":
                    len(
                        school_ids_found_in_companies
                    ),

                "coverage_pct":
                    pct(
                        len(
                            school_ids_found_in_companies
                        ),
                        len(
                            school_ids
                        ),
                    ),
            },

            {
                "reference_population":
                    "investors",

                "school_ids":
                    len(
                        school_ids
                    ),

                "matched_school_ids":
                    len(
                        school_ids_found_in_investors
                    ),

                "coverage_pct":
                    pct(
                        len(
                            school_ids_found_in_investors
                        ),
                        len(
                            school_ids
                        ),
                    ),
            },
        ]
    )

    # =========================================================================
    # 12. Bridge summary
    # =========================================================================

    bridge_summary = pd.DataFrame(
        [
            {
                "bridge":
                    "companies.founder_links -> people.link",

                "reference_type":
                    "exact Crunchbase URL",

                "candidate_status":
                    "high_priority",

                "coverage_pct":
                    pct(
                        len(
                            resolved_unique_links
                        ),
                        len(
                            unique_founder_urls
                        ),
                    ),
            },

            {
                "bridge":
                    "contacts.id -> people.id",

                "reference_type":
                    "exact UUID",

                "candidate_status":
                    "audit_result",

                "coverage_pct":
                    pct(
                        len(
                            contact_people_overlap
                        ),
                        len(
                            contact_id_set
                        ),
                    ),
            },

            {
                "bridge":
                    "schools.id -> companies.id",

                "reference_type":
                    "exact UUID",

                "candidate_status":
                    "audit_result",

                "coverage_pct":
                    pct(
                        len(
                            school_ids_found_in_companies
                        ),
                        len(
                            school_ids
                        ),
                    ),
            },

            {
                "bridge":
                    "schools.id -> investors.id",

                "reference_type":
                    "exact UUID",

                "candidate_status":
                    "audit_result",

                "coverage_pct":
                    pct(
                        len(
                            school_ids_found_in_investors
                        ),
                        len(
                            school_ids
                        ),
                    ),
            },
        ]
    )

    # =========================================================================
    # 13. Save outputs
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    bridge_summary.to_csv(
        BRIDGE_SUMMARY_OUTPUT,
        index=False,
    )

    founder_coverage.to_csv(
        FOUNDER_COVERAGE_OUTPUT,
        index=False,
    )

    founder_match_summary.to_csv(
        FOUNDER_MATCH_OUTPUT,
        index=False,
    )

    founder_degree.to_csv(
        FOUNDER_DEGREE_OUTPUT,
        index=False,
    )

    projection_summary.to_csv(
        FOUNDER_PROJECTION_OUTPUT,
        index=False,
    )

    people_duplicate_details.to_csv(
        PEOPLE_DUPLICATE_OUTPUT,
        index=False,
    )

    contact_reference_summary.to_csv(
        CONTACT_REFERENCE_OUTPUT,
        index=False,
    )

    school_namespace_summary.to_csv(
        SCHOOL_NAMESPACE_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 14. Figures
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    ax.bar(
        founder_coverage[
            "role"
        ],
        founder_coverage[
            "founder_url_coverage_of_role_pct"
        ],
    )

    ax.set_ylabel(
        "Canonical role nodes with founder URL (%)"
    )

    ax.set_title(
        "Founder-Link Coverage of Canonical Graph Roles"
    )

    fig.tight_layout()

    fig.savefig(
        FOUNDER_COVERAGE_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # -------------------------------------------------------------------------

    if len(
        founder_degree
    ) > 0:

        degree_counts = (
            founder_degree[
                "role_node_degree"
            ]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        degree_counts.columns = [
            "role_node_degree",
            "people_count",
        ]

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        ax.scatter(
            degree_counts[
                "role_node_degree"
            ],
            degree_counts[
                "people_count"
            ],
        )

        if (
            degree_counts[
                "role_node_degree"
            ]
            .max()
            > 20
        ):

            ax.set_xscale(
                "log"
            )

        if (
            degree_counts[
                "people_count"
            ]
            .max()
            /
            max(
                1,
                degree_counts[
                    "people_count"
                ]
                .min()
            )
            > 100
        ):

            ax.set_yscale(
                "log"
            )

        ax.set_xlabel(
            "Canonical role nodes linked to person"
        )

        ax.set_ylabel(
            "People count"
        )

        ax.set_title(
            "Founder Intermediary Degree Distribution"
        )

        fig.tight_layout()

        fig.savefig(
            FOUNDER_DEGREE_FIGURE,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    # -------------------------------------------------------------------------

    bridge_plot = (
        bridge_summary
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.barh(
        bridge_plot[
            "bridge"
        ],
        bridge_plot[
            "coverage_pct"
        ],
    )

    ax.set_xlim(
        0,
        105,
    )

    ax.set_xlabel(
        "Exact-reference coverage (%)"
    )

    ax.set_title(
        "Cross-Dataset Reference Bridge Coverage"
    )

    fig.tight_layout()

    fig.savefig(
        BRIDGE_FIGURE,
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
        "FOUNDER-LINK ROLE COVERAGE"
    )

    separator("-")

    print(
        founder_coverage.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "FOUNDER-LINK -> PEOPLE RESOLUTION"
    )

    separator("-")

    print(
        founder_match_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "SHARED-FOUNDER PROJECTION SIZE"
    )

    separator("-")

    print(
        projection_summary.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "PEOPLE DUPLICATE IDs"
    )

    separator("-")

    print(
        f"Distinct duplicated People IDs: "
        f"{len(duplicated_people_ids):,}"
    )

    if len(
        people_duplicate_details
    ) > 0:

        print(
            people_duplicate_details.to_string(
                index=False
            )
        )

    separator("-")

    print(
        "CONTACT REFERENCE INTEGRITY"
    )

    separator("-")

    print(
        contact_reference_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "SCHOOL IDENTITY NAMESPACE"
    )

    separator("-")

    print(
        school_namespace_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "CROSS-DATASET BRIDGE SUMMARY"
    )

    separator("-")

    print(
        bridge_summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator()

    print(
        "PHASE 3.2.7 AUDIT COMPLETE"
    )

    separator()

    print(
        f"""
Outputs written to:

{BRIDGE_SUMMARY_OUTPUT}
{FOUNDER_COVERAGE_OUTPUT}
{FOUNDER_MATCH_OUTPUT}
{FOUNDER_DEGREE_OUTPUT}
{FOUNDER_PROJECTION_OUTPUT}
{PEOPLE_DUPLICATE_OUTPUT}
{CONTACT_REFERENCE_OUTPUT}
{SCHOOL_NAMESPACE_OUTPUT}

Figures written to:

{FOUNDER_COVERAGE_FIGURE}
{FOUNDER_DEGREE_FIGURE}
{BRIDGE_FIGURE}


IMPORTANT

1. No shared-founder graph edge has been materialized.

2. Projected pair counts are diagnostics only.

3. Acquisitions have not yet been resolved to canonical company IDs.

4. School name lists have not been parsed.

5. Investment-derived fields remain quarantined.

NEXT:

Phase 3.2.8 — Acquisition Endpoint Resolution & Structural-Relation Audit
"""
    )


if __name__ == "__main__":
    main()