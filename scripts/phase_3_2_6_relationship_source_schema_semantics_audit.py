from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PHASE 3.2.6 — RELATIONSHIP-SOURCE SCHEMA & SEMANTICS AUDIT
# =============================================================================
#
# PURPOSE
# -------
#
# Audit every available Crunchbase source dataset that could contribute
# structural information to the ITRS-compatible heterogeneous graph.
#
# THIS SCRIPT DOES NOT:
#
#   - create graph nodes,
#   - create graph edges,
#   - assume foreign-key semantics,
#   - infer missing IDs,
#   - merge datasets,
#   - alter Phase-1 or Phase-2 outputs.
#
# It discovers the actual schema and profiles potentially relational fields.
#
# =============================================================================


RAW_DIR = Path("data/raw")

OUTPUT_DIR = Path(
    "data/experimental/phase_3/audits"
)

FIGURE_DIR = Path(
    "data/experimental/phase_3/figures"
)


DATASET_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "relationship_source_dataset_summary.csv"
)

SCHEMA_OUTPUT = (
    OUTPUT_DIR
    / "relationship_source_schema_inventory.csv"
)

CANDIDATE_FIELDS_OUTPUT = (
    OUTPUT_DIR
    / "relationship_source_candidate_fields.csv"
)

FIELD_PROFILES_OUTPUT = (
    OUTPUT_DIR
    / "relationship_source_candidate_field_profiles.csv"
)

CHUNK_INTEGRITY_OUTPUT = (
    OUTPUT_DIR
    / "relationship_source_chunk_integrity.csv"
)


ROW_COUNT_FIGURE = (
    FIGURE_DIR
    / "relationship_source_row_counts.png"
)

FIELD_COVERAGE_FIGURE = (
    FIGURE_DIR
    / "relationship_source_candidate_field_coverage.png"
)


# =============================================================================
# DATASET DEFINITIONS
# =============================================================================


DATASET_PATTERNS = {
    "companies": "companies*.csv",

    "investors":
        "CRUNCHBASE_investor*.csv",

    "funding":
        "CRUNCHBASE_funding*.csv",

    "acquisitions":
        "CRUNCHBASE_acquisition*.csv",

    "people":
        "people*.csv",

    "hubs":
        "CRUNCHBASE_hub*.csv",

    "schools":
        "CRUNCHBASE_school*.csv",

    "events":
        "CRUNCHBASE_event*.csv",

    "contacts":
        "CRUNCHBASE_contact*.csv",
}


# =============================================================================
# RELATIONSHIP-DISCOVERY TOKENS
#
# These are only heuristics to identify fields worth inspecting.
# A field matching a token is NOT automatically considered an edge.
# =============================================================================


RELATION_TOKENS = [
    "id",
    "uuid",

    "name",
    "link",

    "company",
    "organization",
    "org",

    "investor",
    "investment",

    "acquirer",
    "acquiree",
    "acquisition",

    "person",
    "people",
    "founder",
    "employee",
    "partner",

    "school",
    "university",
    "education",

    "hub",

    "event",

    "contact",

    "location",
    "city",
    "region",
    "country",

    "category",
    "industry",

    "portfolio",

    "parent",
    "subsidiary",

    "owner",
    "shareholder",

    "competitor",
    "supplier",
    "customer",
    "client",
]


SAMPLE_PER_FILE = 20_000

CSV_CHUNK_SIZE = 250_000


# =============================================================================
# UTILITIES
# =============================================================================


def separator(char="=", width=120):
    print(char * width)


def normalize_column(column):

    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def clean_string(series):

    return (
        series
        .astype("string")
        .str.strip()
    )


def safe_pct(num, den):

    if den == 0:
        return np.nan

    return (
        num
        / den
        * 100
    )


def human_size(num_bytes):

    size = float(
        num_bytes
    )

    for unit in [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]:

        if size < 1024:

            return (
                f"{size:.2f} {unit}"
            )

        size /= 1024

    return (
        f"{size:.2f} PB"
    )


def is_candidate_field(column):

    c = normalize_column(
        column
    )

    return any(
        token in c
        for token in (
            RELATION_TOKENS
        )
    )


def semantic_hint(column):

    c = normalize_column(
        column
    )

    hints = []

    if (
        c == "id"
        or c.endswith("_id")
        or c.endswith("_ids")
        or "uuid" in c
    ):

        hints.append(
            "identifier_like"
        )

    if (
        c == "name"
        or c.endswith("_name")
        or c.endswith("_names")
    ):

        hints.append(
            "name_like"
        )

    if (
        "link" in c
        or "url" in c
        or "website" in c
    ):

        hints.append(
            "link_like"
        )

    if any(
        token in c
        for token in [
            "acquirer",
            "acquiree",
            "acquisition",
        ]
    ):

        hints.append(
            "acquisition_related"
        )

    if any(
        token in c
        for token in [
            "investor",
            "investment",
            "portfolio",
        ]
    ):

        hints.append(
            "investment_related"
        )

    if any(
        token in c
        for token in [
            "person",
            "people",
            "founder",
            "employee",
            "partner",
        ]
    ):

        hints.append(
            "person_related"
        )

    if any(
        token in c
        for token in [
            "school",
            "university",
            "education",
        ]
    ):

        hints.append(
            "education_related"
        )

    if "hub" in c:

        hints.append(
            "hub_related"
        )

    if "event" in c:

        hints.append(
            "event_related"
        )

    if "contact" in c:

        hints.append(
            "contact_related"
        )

    if any(
        token in c
        for token in [
            "location",
            "city",
            "region",
            "country",
        ]
    ):

        hints.append(
            "location_related"
        )

    if any(
        token in c
        for token in [
            "category",
            "industry",
        ]
    ):

        hints.append(
            "category_related"
        )

    if any(
        token in c
        for token in [
            "parent",
            "subsidiary",
            "owner",
            "shareholder",
            "competitor",
            "supplier",
            "customer",
            "client",
        ]
    ):

        hints.append(
            "structural_relation_candidate"
        )

    if not hints:

        hints.append(
            "general_candidate"
        )

    return "|".join(
        sorted(
            set(
                hints
            )
        )
    )


# =============================================================================
# VALUE PROFILING
# =============================================================================


def profile_sample_field(
    series,
):

    total = len(
        series
    )

    if total == 0:

        return {
            "sample_rows": 0,
            "nonmissing_count": 0,
            "nonmissing_pct": np.nan,
            "sample_unique_nonmissing": 0,
            "contains_crunchbase_url_pct": np.nan,
            "contains_comma_pct": np.nan,
            "contains_pipe_pct": np.nan,
            "contains_semicolon_pct": np.nan,
            "example_1": "",
            "example_2": "",
            "example_3": "",
            "top_value_1": "",
            "top_value_1_count": 0,
        }

    values = (
        clean_string(
            series
        )
        .replace(
            "",
            pd.NA,
        )
    )

    nonmissing = (
        values.dropna()
    )

    nonmissing_count = len(
        nonmissing
    )

    if nonmissing_count == 0:

        return {
            "sample_rows": total,
            "nonmissing_count": 0,
            "nonmissing_pct": 0.0,
            "sample_unique_nonmissing": 0,
            "contains_crunchbase_url_pct": np.nan,
            "contains_comma_pct": np.nan,
            "contains_pipe_pct": np.nan,
            "contains_semicolon_pct": np.nan,
            "example_1": "",
            "example_2": "",
            "example_3": "",
            "top_value_1": "",
            "top_value_1_count": 0,
        }

    unique_examples = (
        nonmissing
        .drop_duplicates()
        .head(3)
        .tolist()
    )

    while len(
        unique_examples
    ) < 3:

        unique_examples.append(
            ""
        )

    top_counts = (
        nonmissing
        .value_counts()
    )

    top_value = (
        str(
            top_counts.index[0]
        )
        if len(
            top_counts
        ) > 0
        else ""
    )

    top_count = (
        int(
            top_counts.iloc[0]
        )
        if len(
            top_counts
        ) > 0
        else 0
    )

    return {
        "sample_rows":
            total,

        "nonmissing_count":
            nonmissing_count,

        "nonmissing_pct":
            safe_pct(
                nonmissing_count,
                total,
            ),

        "sample_unique_nonmissing":
            int(
                nonmissing.nunique()
            ),

        "contains_crunchbase_url_pct":
            safe_pct(
                nonmissing
                .str.contains(
                    "crunchbase.com",
                    case=False,
                    regex=False,
                )
                .sum(),
                nonmissing_count,
            ),

        "contains_comma_pct":
            safe_pct(
                nonmissing
                .str.contains(
                    ",",
                    regex=False,
                )
                .sum(),
                nonmissing_count,
            ),

        "contains_pipe_pct":
            safe_pct(
                nonmissing
                .str.contains(
                    "|",
                    regex=False,
                )
                .sum(),
                nonmissing_count,
            ),

        "contains_semicolon_pct":
            safe_pct(
                nonmissing
                .str.contains(
                    ";",
                    regex=False,
                )
                .sum(),
                nonmissing_count,
            ),

        "example_1":
            str(
                unique_examples[0]
            ),

        "example_2":
            str(
                unique_examples[1]
            ),

        "example_3":
            str(
                unique_examples[2]
            ),

        "top_value_1":
            top_value,

        "top_value_1_count":
            top_count,
    }


# =============================================================================
# DATASET AUDIT
# =============================================================================


def audit_dataset(
    family,
    files,
):

    dataset_rows = []

    schema_rows = []

    candidate_rows = []

    profile_frames = []

    chunk_rows = []

    all_id_hashes = []

    reference_header = None

    # =========================================================================
    # Per-file audit
    # =========================================================================

    for file_index, path in enumerate(
        files,
        start=1,
    ):

        header = (
            pd.read_csv(
                path,
                nrows=0,
            )
            .columns
            .tolist()
        )

        if (
            reference_header
            is None
        ):

            reference_header = (
                header
            )

        schema_matches_reference = (
            header
            ==
            reference_header
        )

        row_count = 0

        id_missing = np.nan

        id_nonmissing = np.nan

        id_unique = np.nan

        id_duplicates = np.nan

        has_exact_id = (
            "id" in header
        )

        # ---------------------------------------------------------------------
        # Count rows and audit exact id
        # ---------------------------------------------------------------------

        usecol = (
            "id"
            if has_exact_id
            else header[0]
        )

        id_missing_count = 0

        id_nonmissing_count = 0

        file_id_hashes = []

        for chunk in pd.read_csv(
            path,
            usecols=[
                usecol
            ],
            chunksize=(
                CSV_CHUNK_SIZE
            ),
            low_memory=False,
        ):

            row_count += len(
                chunk
            )

            if has_exact_id:

                values = (
                    clean_string(
                        chunk[
                            "id"
                        ]
                    )
                    .replace(
                        "",
                        pd.NA,
                    )
                )

                missing = (
                    values.isna()
                )

                id_missing_count += int(
                    missing.sum()
                )

                valid = (
                    values[
                        ~missing
                    ]
                )

                id_nonmissing_count += len(
                    valid
                )

                if len(
                    valid
                ) > 0:

                    hashes = (
                        pd.util
                        .hash_pandas_object(
                            valid,
                            index=False,
                        )
                        .to_numpy(
                            dtype=np.uint64
                        )
                    )

                    file_id_hashes.append(
                        hashes
                    )

                    all_id_hashes.append(
                        hashes
                    )

        if has_exact_id:

            if file_id_hashes:

                file_hashes = (
                    np.concatenate(
                        file_id_hashes
                    )
                )

                id_unique = int(
                    np.unique(
                        file_hashes
                    )
                    .size
                )

            else:

                id_unique = 0

            id_missing = (
                id_missing_count
            )

            id_nonmissing = (
                id_nonmissing_count
            )

            id_duplicates = (
                id_nonmissing_count
                - id_unique
            )

        # ---------------------------------------------------------------------
        # Schema
        # ---------------------------------------------------------------------

        for position, column in enumerate(
            header
        ):

            row = {
                "dataset_family":
                    family,

                "file_name":
                    path.name,

                "column_position":
                    position,

                "column_name":
                    column,

                "normalized_column_name":
                    normalize_column(
                        column
                    ),

                "candidate_relationship_field":
                    is_candidate_field(
                        column
                    ),

                "semantic_hint":
                    semantic_hint(
                        column
                    ),
            }

            schema_rows.append(
                row
            )

            if (
                row[
                    "candidate_relationship_field"
                ]
            ):

                candidate_rows.append(
                    row.copy()
                )

        # ---------------------------------------------------------------------
        # Sample candidate columns
        # ---------------------------------------------------------------------

        candidate_columns = [
            column
            for column in header
            if is_candidate_field(
                column
            )
        ]

        if candidate_columns:

            sample = pd.read_csv(
                path,
                usecols=(
                    candidate_columns
                ),
                nrows=(
                    SAMPLE_PER_FILE
                ),
                low_memory=False,
            )

            sample[
                "__source_file"
            ] = (
                path.name
            )

            profile_frames.append(
                sample
            )

        chunk_rows.append(
            {
                "dataset_family":
                    family,

                "file_name":
                    path.name,

                "row_count":
                    row_count,

                "column_count":
                    len(
                        header
                    ),

                "schema_matches_first_chunk":
                    schema_matches_reference,

                "has_exact_id":
                    has_exact_id,

                "id_missing_count":
                    id_missing,

                "id_nonmissing_count":
                    id_nonmissing,

                "id_unique_nonmissing_count":
                    id_unique,

                "id_duplicate_nonmissing_count":
                    id_duplicates,
            }
        )

    # =========================================================================
    # Family-level ID integrity
    # =========================================================================

    total_rows = sum(
        row[
            "row_count"
        ]
        for row in (
            chunk_rows
        )
    )

    if all_id_hashes:

        all_hashes = (
            np.concatenate(
                all_id_hashes
            )
        )

        globally_unique_ids = int(
            np.unique(
                all_hashes
            )
            .size
        )

        total_nonmissing_ids = int(
            len(
                all_hashes
            )
        )

        global_duplicate_occurrences = (
            total_nonmissing_ids
            - globally_unique_ids
        )

    else:

        globally_unique_ids = np.nan

        total_nonmissing_ids = np.nan

        global_duplicate_occurrences = np.nan

    dataset_rows.append(
        {
            "dataset_family":
                family,

            "file_count":
                len(
                    files
                ),

            "total_rows":
                total_rows,

            "column_count":
                len(
                    reference_header
                ),

            "all_chunks_same_schema":
                all(
                    row[
                        "schema_matches_first_chunk"
                    ]
                    for row in (
                        chunk_rows
                    )
                ),

            "has_exact_id":
                (
                    "id"
                    in (
                        reference_header
                    )
                ),

            "total_nonmissing_id_rows":
                total_nonmissing_ids,

            "globally_unique_ids":
                globally_unique_ids,

            "global_duplicate_id_occurrences":
                global_duplicate_occurrences,

            "total_file_size_bytes":
                sum(
                    path.stat()
                    .st_size
                    for path in files
                ),

            "total_file_size_human":
                human_size(
                    sum(
                        path.stat()
                        .st_size
                        for path in files
                    )
                ),
        }
    )

    # =========================================================================
    # Aggregate sample field profiles across family
    # =========================================================================

    profile_rows = []

    if profile_frames:

        samples = pd.concat(
            profile_frames,
            ignore_index=True,
            sort=False,
        )

        candidate_columns = sorted(
            {
                row[
                    "column_name"
                ]
                for row in (
                    candidate_rows
                )
            }
        )

        for column in (
            candidate_columns
        ):

            if (
                column
                not in samples.columns
            ):

                continue

            stats = (
                profile_sample_field(
                    samples[
                        column
                    ]
                )
            )

            profile_rows.append(
                {
                    "dataset_family":
                        family,

                    "column_name":
                        column,

                    "semantic_hint":
                        semantic_hint(
                            column
                        ),

                    **stats,
                }
            )

    return (
        pd.DataFrame(
            dataset_rows
        ),

        pd.DataFrame(
            schema_rows
        ),

        pd.DataFrame(
            candidate_rows
        ),

        pd.DataFrame(
            profile_rows
        ),

        pd.DataFrame(
            chunk_rows
        ),
    )


# =============================================================================
# MAIN
# =============================================================================


def main():

    separator()

    print(
        "PHASE 3.2.6 — "
        "RELATIONSHIP-SOURCE SCHEMA & SEMANTICS AUDIT"
    )

    separator()

    all_dataset_summaries = []

    all_schema = []

    all_candidates = []

    all_profiles = []

    all_chunk_integrity = []

    # =========================================================================
    # Audit all families
    # =========================================================================

    for family, pattern in (
        DATASET_PATTERNS.items()
    ):

        files = sorted(
            RAW_DIR.glob(
                pattern
            )
        )

        separator("-")

        print(
            f"{family.upper()}"
        )

        separator("-")

        print(
            f"Pattern: {pattern}"
        )

        print(
            f"Files found: "
            f"{len(files)}"
        )

        if not files:

            print(
                "SKIPPED — no files found."
            )

            continue

        (
            dataset_summary,
            schema,
            candidates,
            profiles,
            chunk_integrity,
        ) = audit_dataset(
            family=family,
            files=files,
        )

        all_dataset_summaries.append(
            dataset_summary
        )

        all_schema.append(
            schema
        )

        all_candidates.append(
            candidates
        )

        all_profiles.append(
            profiles
        )

        all_chunk_integrity.append(
            chunk_integrity
        )

        # ---------------------------------------------------------------------
        # Terminal schema
        # ---------------------------------------------------------------------

        first_file = (
            files[0]
        )

        header = (
            pd.read_csv(
                first_file,
                nrows=0,
            )
            .columns
            .tolist()
        )

        print(
            f"\nRows: "
            f"{int(dataset_summary.iloc[0]['total_rows']):,}"
        )

        print(
            f"Columns: "
            f"{len(header):,}"
        )

        print(
            f"Globally unique exact IDs: "
            f"{dataset_summary.iloc[0]['globally_unique_ids']}"
        )

        print(
            f"Global duplicate ID occurrences: "
            f"{dataset_summary.iloc[0]['global_duplicate_id_occurrences']}"
        )

        print(
            "\nFULL SCHEMA:"
        )

        for column in (
            header
        ):

            flag = (
                "  <-- candidate"
                if is_candidate_field(
                    column
                )
                else ""
            )

            print(
                f"  - {column}"
                f"{flag}"
            )

        print(
            "\nCANDIDATE FIELD PROFILES:"
        )

        if len(
            profiles
        ) > 0:

            print(
                profiles[
                    [
                        "column_name",
                        "semantic_hint",
                        "nonmissing_pct",
                        "sample_unique_nonmissing",
                        "contains_crunchbase_url_pct",
                        "contains_comma_pct",
                        "example_1",
                    ]
                ]
                .to_string(
                    index=False,
                    float_format=lambda x:
                        f"{x:.2f}",
                )
            )

    # =========================================================================
    # Combine outputs
    # =========================================================================

    dataset_summary = pd.concat(
        all_dataset_summaries,
        ignore_index=True,
    )

    schema = pd.concat(
        all_schema,
        ignore_index=True,
    )

    candidates = pd.concat(
        all_candidates,
        ignore_index=True,
    )

    profiles = pd.concat(
        all_profiles,
        ignore_index=True,
    )

    chunk_integrity = pd.concat(
        all_chunk_integrity,
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_summary.to_csv(
        DATASET_SUMMARY_OUTPUT,
        index=False,
    )

    schema.to_csv(
        SCHEMA_OUTPUT,
        index=False,
    )

    candidates.to_csv(
        CANDIDATE_FIELDS_OUTPUT,
        index=False,
    )

    profiles.to_csv(
        FIELD_PROFILES_OUTPUT,
        index=False,
    )

    chunk_integrity.to_csv(
        CHUNK_INTEGRITY_OUTPUT,
        index=False,
    )

    # =========================================================================
    # FIGURE 1 — DATASET ROW COUNTS
    # =========================================================================

    plot_data = (
        dataset_summary
        .sort_values(
            "total_rows",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        plot_data[
            "dataset_family"
        ],
        plot_data[
            "total_rows"
        ],
    )

    positive = (
        plot_data[
            plot_data[
                "total_rows"
            ]
            > 0
        ][
            "total_rows"
        ]
    )

    if (
        len(
            positive
        ) > 1
        and
        (
            positive.max()
            /
            positive.min()
            >= 100
        )
    ):

        ax.set_xscale(
            "log"
        )

        ax.set_xlabel(
            "Rows (log scale)"
        )

    else:

        ax.set_xlabel(
            "Rows"
        )

    ax.set_title(
        "Crunchbase Relationship-Source Dataset Sizes"
    )

    fig.tight_layout()

    fig.savefig(
        ROW_COUNT_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =========================================================================
    # FIGURE 2 — CANDIDATE FIELD COVERAGE
    # =========================================================================

    coverage_plot = (
        profiles[
            profiles[
                "nonmissing_pct"
            ]
            .notna()
        ]
        .copy()
    )

    # Focus on the most populated potentially relational fields.
    coverage_plot = (
        coverage_plot
        .sort_values(
            "nonmissing_pct",
            ascending=False,
        )
        .head(40)
    )

    coverage_plot[
        "label"
    ] = (
        coverage_plot[
            "dataset_family"
        ]
        +
        "."
        +
        coverage_plot[
            "column_name"
        ]
    )

    coverage_plot = (
        coverage_plot
        .sort_values(
            "nonmissing_pct",
            ascending=True,
        )
    )

    if len(
        coverage_plot
    ) > 0:

        fig, ax = plt.subplots(
            figsize=(
                11,
                max(
                    6,
                    0.32
                    * len(
                        coverage_plot
                    ),
                ),
            )
        )

        ax.barh(
            coverage_plot[
                "label"
            ],
            coverage_plot[
                "nonmissing_pct"
            ],
        )

        ax.set_xlabel(
            "Non-missing values in audit sample (%)"
        )

        ax.set_title(
            "Coverage of Candidate Relationship Fields"
        )

        fig.tight_layout()

        fig.savefig(
            FIELD_COVERAGE_FIGURE,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    # =========================================================================
    # FINAL TERMINAL SUMMARY
    # =========================================================================

    separator()

    print(
        "RELATIONSHIP-SOURCE DATASET SUMMARY"
    )

    separator()

    print(
        dataset_summary[
            [
                "dataset_family",
                "file_count",
                "total_rows",
                "column_count",
                "all_chunks_same_schema",
                "has_exact_id",
                "globally_unique_ids",
                "global_duplicate_id_occurrences",
                "total_file_size_human",
            ]
        ]
        .to_string(
            index=False
        )
    )

    separator("-")

    print(
        "MULTI-FILE SCHEMA / ID INTEGRITY"
    )

    separator("-")

    multi_file = (
        dataset_summary[
            dataset_summary[
                "file_count"
            ]
            > 1
        ]
    )

    if len(
        multi_file
    ) > 0:

        print(
            multi_file[
                [
                    "dataset_family",
                    "file_count",
                    "all_chunks_same_schema",
                    "global_duplicate_id_occurrences",
                ]
            ]
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No multi-file dataset families."
        )

    separator()

    print(
        "PHASE 3.2.6 AUDIT COMPLETE"
    )

    separator()

    print(
        f"""
Outputs written to:

{DATASET_SUMMARY_OUTPUT}
{SCHEMA_OUTPUT}
{CANDIDATE_FIELDS_OUTPUT}
{FIELD_PROFILES_OUTPUT}
{CHUNK_INTEGRITY_OUTPUT}

Figures written to:

{ROW_COUNT_FIGURE}
{FIELD_COVERAGE_FIGURE}


IMPORTANT

1. Candidate fields are detected by column-name heuristics only.

2. No candidate field has yet been accepted as a graph relation.

3. High coverage does not imply usefulness.

4. Fields derived from investments remain quarantined until leakage is
   explicitly evaluated.

5. Phase 3.2.7 will map actual entity/reference fields across datasets
   and test which relationships can be constructed with reliable IDs.

NO GRAPH HAS BEEN CREATED.
"""
    )


if __name__ == "__main__":
    main()