from pathlib import Path
import json
import re

import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PHASE 3.2.5 — RELATIONSHIP-SOURCE DATASET DISCOVERY
# =============================================================================
#
# PURPOSE
# -------
#
# Locate the datasets that could provide structural relationships for the
# ITRS-compatible heterogeneous graph.
#
# We already know data/raw contains:
#
#   - companies
#   - investors
#   - funding rounds
#
# Previous project analysis also used datasets corresponding to:
#
#   - acquisitions
#   - people
#   - hubs
#   - schools
#   - events
#   - contacts
#
# This script searches:
#
#   A. data files throughout the repository
#   B. Python / notebook / Markdown source references
#
# No graph relationships are created.
# =============================================================================


REPO_ROOT = Path(".")

OUTPUT_DIR = Path(
    "data/experimental/phase_3/audits"
)

FIGURE_DIR = Path(
    "data/experimental/phase_3/figures"
)


FILE_DISCOVERY_OUTPUT = (
    OUTPUT_DIR
    / "relationship_source_dataset_discovery.csv"
)

SCHEMA_OUTPUT = (
    OUTPUT_DIR
    / "relationship_source_candidate_schema.csv"
)

CODE_REFERENCE_OUTPUT = (
    OUTPUT_DIR
    / "relationship_source_code_references.csv"
)

FAMILY_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "relationship_source_family_summary.csv"
)

FIGURE_OUTPUT = (
    FIGURE_DIR
    / "relationship_source_family_discovery.png"
)


# =============================================================================
# DATASET FAMILIES
# =============================================================================


FAMILY_KEYWORDS = {
    "companies": [
        "companies",
        "company",
        "organizations",
        "organization",
    ],

    "investors": [
        "investor",
        "investors",
    ],

    "funding": [
        "funding",
        "funding_round",
        "funding_rounds",
    ],

    "acquisitions": [
        "acquisition",
        "acquisitions",
        "acquirer",
        "acquiree",
    ],

    "people": [
        "people",
        "person",
        "persons",
    ],

    "hubs": [
        "hub",
        "hubs",
    ],

    "schools": [
        "school",
        "schools",
        "university",
        "universities",
    ],

    "events": [
        "event",
        "events",
    ],

    "contacts": [
        "contact",
        "contacts",
    ],
}


DATA_EXTENSIONS = {
    ".csv",
    ".parquet",
    ".xlsx",
    ".xls",
}

SOURCE_EXTENSIONS = {
    ".py",
    ".ipynb",
    ".md",
    ".txt",
}


# Directories we should never recursively scan.
EXCLUDED_DIR_NAMES = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "node_modules",
    "venv",
    ".venv",
    "env",
    ".conda",
}


# Generated data should not be mistaken for an original relationship source.
EXCLUDED_PATH_FRAGMENTS = {
    "data/experimental",
}


# =============================================================================
# UTILITIES
# =============================================================================


def separator(char="=", width=120):
    print(char * width)


def normalized_path(path):

    return str(
        path
        .as_posix()
    ).lower()


def should_exclude(path):

    path_text = (
        normalized_path(
            path
        )
    )

    if any(
        fragment
        in path_text
        for fragment in (
            EXCLUDED_PATH_FRAGMENTS
        )
    ):
        return True

    if any(
        part
        in EXCLUDED_DIR_NAMES
        for part in path.parts
    ):
        return True

    return False


def classify_filename(path):

    text = (
        normalized_path(
            path
        )
    )

    matches = []

    for family, keywords in (
        FAMILY_KEYWORDS.items()
    ):

        if any(
            keyword in text
            for keyword in keywords
        ):
            matches.append(
                family
            )

    return matches


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


# =============================================================================
# HEADER READERS
# =============================================================================


def read_header(path):

    suffix = (
        path.suffix.lower()
    )

    if suffix == ".csv":

        return (
            pd.read_csv(
                path,
                nrows=0,
            )
            .columns
            .tolist()
        )

    if suffix == ".parquet":

        import pyarrow.parquet as pq

        parquet = (
            pq.ParquetFile(
                path
            )
        )

        return (
            parquet
            .schema_arrow
            .names
        )

    if suffix in {
        ".xlsx",
        ".xls",
    }:

        return (
            pd.read_excel(
                path,
                nrows=0,
            )
            .columns
            .tolist()
        )

    return []


# =============================================================================
# DATA-FILE DISCOVERY
# =============================================================================


def discover_data_files():

    results = []

    for path in (
        REPO_ROOT.rglob("*")
    ):

        if not path.is_file():
            continue

        if should_exclude(
            path
        ):
            continue

        suffix = (
            path.suffix.lower()
        )

        if suffix not in (
            DATA_EXTENSIONS
        ):
            continue

        families = (
            classify_filename(
                path
            )
        )

        if not families:
            continue

        try:

            columns = (
                read_header(
                    path
                )
            )

            schema_status = (
                "PASS"
            )

            schema_error = ""

        except Exception as exc:

            columns = []

            schema_status = (
                "ERROR"
            )

            schema_error = str(
                exc
            )

        results.append(
            {
                "relative_path": str(
                    path
                ),

                "file_name": (
                    path.name
                ),

                "extension": (
                    suffix
                ),

                "file_size_bytes": (
                    path.stat()
                    .st_size
                ),

                "file_size_human": (
                    human_size(
                        path.stat()
                        .st_size
                    )
                ),

                "candidate_families": (
                    "|".join(
                        families
                    )
                ),

                "column_count": (
                    len(
                        columns
                    )
                ),

                "columns": (
                    " | ".join(
                        columns
                    )
                ),

                "schema_status": (
                    schema_status
                ),

                "schema_error": (
                    schema_error
                ),
            }
        )

    return pd.DataFrame(
        results
    )


# =============================================================================
# LONG SCHEMA
# =============================================================================


def build_schema_table(
    discovery,
):

    rows = []

    for _, record in (
        discovery.iterrows()
    ):

        path = Path(
            record[
                "relative_path"
            ]
        )

        try:

            columns = (
                read_header(
                    path
                )
            )

        except Exception:

            continue

        for position, column in enumerate(
            columns
        ):

            rows.append(
                {
                    "relative_path": str(
                        path
                    ),

                    "file_name": (
                        path.name
                    ),

                    "candidate_families": (
                        record[
                            "candidate_families"
                        ]
                    ),

                    "column_position": (
                        position
                    ),

                    "column_name": (
                        column
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# SOURCE-CODE SEARCH
# =============================================================================


SEARCH_TERMS = [
    "acquisitions_df",
    "acquisition_df",
    "people_df",
    "person_df",
    "hubs_df",
    "hub_df",
    "schools_df",
    "school_df",
    "events_df",
    "event_df",
    "contacts_df",
    "contact_df",
    "funding_rounds_df",
    "investors_df",
    "companies_df",
]


def extract_notebook_text(path):

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:

            notebook = json.load(
                f
            )

    except Exception:

        return ""

    parts = []

    for cell in (
        notebook.get(
            "cells",
            []
        )
    ):

        source = (
            cell.get(
                "source",
                []
            )
        )

        if isinstance(
            source,
            list,
        ):

            parts.extend(
                source
            )

        elif isinstance(
            source,
            str,
        ):

            parts.append(
                source
            )

    return "\n".join(
        parts
    )


def read_source_text(path):

    if (
        path.suffix.lower()
        == ".ipynb"
    ):

        return (
            extract_notebook_text(
                path
            )
        )

    try:

        return (
            path.read_text(
                encoding="utf-8",
                errors="ignore",
            )
        )

    except Exception:

        return ""


def discover_code_references():

    rows = []

    for path in (
        REPO_ROOT.rglob("*")
    ):

        if not path.is_file():
            continue

        if should_exclude(
            path
        ):
            continue

        if (
            path.suffix.lower()
            not in SOURCE_EXTENSIONS
        ):
            continue

        # Skip very large text/source files.
        if (
            path.stat().st_size
            > 20 * 1024 * 1024
        ):
            continue

        text = (
            read_source_text(
                path
            )
        )

        if not text:
            continue

        lines = (
            text.splitlines()
        )

        for line_number, line in enumerate(
            lines,
            start=1,
        ):

            line_lower = (
                line.lower()
            )

            matched_terms = [
                term
                for term in (
                    SEARCH_TERMS
                )
                if (
                    term.lower()
                    in line_lower
                )
            ]

            if not matched_terms:
                continue

            start = max(
                0,
                line_number
                - 3
            )

            end = min(
                len(lines),
                line_number
                + 2
            )

            context = (
                "\n".join(
                    lines[
                        start:end
                    ]
                )
            )

            rows.append(
                {
                    "source_file": str(
                        path
                    ),

                    "line_number": (
                        line_number
                    ),

                    "matched_terms": (
                        "|".join(
                            matched_terms
                        )
                    ),

                    "line_text": (
                        line.strip()
                    ),

                    "context": (
                        context
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# FAMILY SUMMARY
# =============================================================================


def build_family_summary(
    discovery,
    code_refs,
):

    rows = []

    for family in (
        FAMILY_KEYWORDS.keys()
    ):

        if len(
            discovery
        ) > 0:

            file_matches = (
                discovery[
                    discovery[
                        "candidate_families"
                    ]
                    .str.split("|")
                    .apply(
                        lambda values:
                            family
                            in values
                    )
                ]
            )

        else:

            file_matches = (
                pd.DataFrame()
            )

        family_terms = [
            term
            for term in (
                SEARCH_TERMS
            )
            if (
                family.rstrip("s")
                in term
                or family
                in term
            )
        ]

        if len(
            code_refs
        ) > 0:

            ref_count = int(
                code_refs[
                    "matched_terms"
                ]
                .str.contains(
                    "|".join(
                        re.escape(
                            term
                        )
                        for term in (
                            family_terms
                        )
                    ),
                    case=False,
                    regex=True,
                    na=False,
                )
                .sum()
                if family_terms
                else 0
            )

        else:

            ref_count = 0

        rows.append(
            {
                "dataset_family": (
                    family
                ),

                "candidate_data_files": (
                    len(
                        file_matches
                    )
                ),

                "source_code_references": (
                    ref_count
                ),

                "candidate_paths": (
                    " | ".join(
                        file_matches[
                            "relative_path"
                        ]
                        .astype(str)
                        .tolist()
                    )
                    if len(
                        file_matches
                    ) > 0
                    else ""
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MAIN
# =============================================================================


def main():

    separator()

    print(
        "PHASE 3.2.5 — "
        "RELATIONSHIP-SOURCE DATASET DISCOVERY"
    )

    separator()

    print(
        "\nSearching repository for "
        "candidate relationship-source datasets..."
    )

    discovery = (
        discover_data_files()
    )

    schema = (
        build_schema_table(
            discovery
        )
    )

    print(
        "Searching source files / notebooks "
        "for dataframe references..."
    )

    code_refs = (
        discover_code_references()
    )

    family_summary = (
        build_family_summary(
            discovery,
            code_refs,
        )
    )

    # =========================================================================
    # Save
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    discovery.to_csv(
        FILE_DISCOVERY_OUTPUT,
        index=False,
    )

    schema.to_csv(
        SCHEMA_OUTPUT,
        index=False,
    )

    code_refs.to_csv(
        CODE_REFERENCE_OUTPUT,
        index=False,
    )

    family_summary.to_csv(
        FAMILY_SUMMARY_OUTPUT,
        index=False,
    )

    # =========================================================================
    # Figure
    # =========================================================================

    plot_data = (
        family_summary
        .sort_values(
            "candidate_data_files",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        plot_data[
            "dataset_family"
        ],
        plot_data[
            "candidate_data_files"
        ],
    )

    ax.set_xlabel(
        "Candidate data files discovered"
    )

    ax.set_title(
        "Relationship-Source Dataset Discovery"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_OUTPUT,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =========================================================================
    # Terminal report
    # =========================================================================

    separator("-")

    print(
        "DATASET FAMILY SUMMARY"
    )

    separator("-")

    print(
        family_summary.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "DISCOVERED CANDIDATE DATA FILES"
    )

    separator("-")

    if len(
        discovery
    ) > 0:

        print(
            discovery[
                [
                    "candidate_families",
                    "relative_path",
                    "file_size_human",
                    "column_count",
                    "schema_status",
                ]
            ]
            .sort_values(
                [
                    "candidate_families",
                    "relative_path",
                ]
            )
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No candidate files discovered."
        )

    separator("-")

    print(
        "SOURCE-CODE / NOTEBOOK REFERENCES"
    )

    separator("-")

    if len(
        code_refs
    ) > 0:

        display = (
            code_refs[
                [
                    "source_file",
                    "line_number",
                    "matched_terms",
                    "line_text",
                ]
            ]
            .drop_duplicates()
        )

        print(
            display
            .head(200)
            .to_string(
                index=False
            )
        )

    else:

        print(
            "No dataframe-name references discovered."
        )

    separator()

    print(
        "PHASE 3.2.5 DISCOVERY COMPLETE"
    )

    separator()

    print(
        f"""
Outputs written to:

{FILE_DISCOVERY_OUTPUT}
{SCHEMA_OUTPUT}
{CODE_REFERENCE_OUTPUT}
{FAMILY_SUMMARY_OUTPUT}

Figure written to:

{FIGURE_OUTPUT}


NO GRAPH RELATIONSHIPS HAVE BEEN CREATED.

Next step depends on which source datasets are actually located.

If the acquisitions / people / hubs / schools / events / contacts
files are found, Phase 3.2.6 will audit their exact fields and
relationship semantics.

If they are not found, the code-reference output should help identify
where they were originally loaded from.
"""
    )


if __name__ == "__main__":
    main()