from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


# =============================================================================
# PHASE 3.2.1 — RAW DATASET & SCHEMA INVENTORY
# =============================================================================
#
# PURPOSE
# -------
# Discover and profile the original Crunchbase datasets before constructing
# any heterogeneous graph.
#
# This phase DOES NOT:
#   - assume companies is the master entity table,
#   - create graph nodes,
#   - create graph edges,
#   - resolve entity identities,
#   - infer foreign-key semantics,
#   - filter datasets,
#   - modify Phase-1 or Phase-2 data.
#
# The results of this audit will be used in:
#
#   Phase 3.2.2 — companies_df Master-Entity Hypothesis Audit
#
# =============================================================================


RAW_DIR = Path("data/raw")

OUTPUT_DIR = Path(
    "data/experimental/phase_3/audits"
)

FIGURE_DIR = Path(
    "data/experimental/phase_3/figures"
)


DATASET_INVENTORY_OUTPUT = (
    OUTPUT_DIR
    / "raw_dataset_inventory.csv"
)

SCHEMA_OUTPUT = (
    OUTPUT_DIR
    / "raw_schema_inventory_long.csv"
)

CANDIDATE_COLUMNS_OUTPUT = (
    OUTPUT_DIR
    / "candidate_identifier_reference_columns.csv"
)


ROW_COUNT_FIGURE = (
    FIGURE_DIR
    / "raw_dataset_row_counts.png"
)

SCHEMA_WIDTH_FIGURE = (
    FIGURE_DIR
    / "raw_dataset_schema_width.png"
)


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".parquet",
}

SAMPLE_ROWS = 5_000

CSV_CHUNK_SIZE = 250_000

PARQUET_BATCH_SIZE = 250_000


# =============================================================================
# Utilities
# =============================================================================


def separator(char="=", width=120):
    print(char * width)


def human_file_size(num_bytes):

    size = float(num_bytes)

    for unit in [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]:

        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def normalize_column_name(column):

    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


# =============================================================================
# Column-semantic heuristic
#
# IMPORTANT:
#
# These labels are ONLY discovery aids.
#
# A field being tagged "entity_reference_like" does NOT mean that we have
# established that it is a valid foreign key.
#
# That will be audited later.
# =============================================================================


ENTITY_REFERENCE_TOKENS = {
    "investor",
    "company",
    "organization",
    "org",
    "startup",
    "acquirer",
    "acquiree",
    "person",
    "people",
    "founder",
    "school",
    "hub",
    "event",
    "contact",
    "partner",
    "lead",
}


def classify_column(column):

    c = normalize_column_name(
        column
    )

    labels = []

    # -------------------------------------------------------------------------
    # Strong identifier patterns
    # -------------------------------------------------------------------------

    if c == "id":
        labels.append(
            "primary_id_candidate"
        )

    if (
        c.endswith("_id")
        or c.endswith("_ids")
        or "uuid" in c
    ):
        labels.append(
            "id_like"
        )

    # -------------------------------------------------------------------------
    # Name-like fields
    # -------------------------------------------------------------------------

    if (
        c == "name"
        or c.endswith("_name")
        or c.endswith("_names")
        or c in {
            "org_name",
            "company_name",
        }
    ):
        labels.append(
            "name_like"
        )

    # -------------------------------------------------------------------------
    # Entity/reference-like terminology
    # -------------------------------------------------------------------------

    for token in ENTITY_REFERENCE_TOKENS:

        if token in c:

            labels.append(
                "entity_reference_like"
            )

            break

    # -------------------------------------------------------------------------
    # URL / website fields can be useful for entity resolution later.
    # -------------------------------------------------------------------------

    if (
        "website" in c
        or c.endswith("_url")
        or c == "link"
    ):
        labels.append(
            "resolution_attribute"
        )

    if not labels:
        labels.append(
            "other"
        )

    return "|".join(
        sorted(set(labels))
    )


# =============================================================================
# CSV PROFILING
# =============================================================================


def read_csv_header(path):

    header = pd.read_csv(
        path,
        nrows=0,
    )

    return list(
        header.columns
    )


def read_csv_sample(
    path,
    columns,
):

    sample = pd.read_csv(
        path,
        nrows=SAMPLE_ROWS,
        low_memory=False,
    )

    # Ensure original header ordering.
    sample = sample[
        [
            c
            for c in columns
            if c in sample.columns
        ]
    ]

    return sample


def profile_csv_rows_and_id(
    path,
    columns,
):

    # -------------------------------------------------------------------------
    # If an exact "id" column exists, use it while streaming the file.
    #
    # Otherwise use the first column merely to count parsed records.
    # -------------------------------------------------------------------------

    has_exact_id = (
        "id" in columns
    )

    count_column = (
        "id"
        if has_exact_id
        else columns[0]
    )

    total_rows = 0

    id_missing = np.nan
    id_unique = np.nan
    id_duplicates = np.nan

    hash_arrays = []

    id_missing_count = 0
    id_nonmissing_count = 0

    for chunk in pd.read_csv(
        path,
        usecols=[count_column],
        chunksize=CSV_CHUNK_SIZE,
        low_memory=False,
    ):

        total_rows += len(
            chunk
        )

        if has_exact_id:

            values = (
                chunk["id"]
                .astype("string")
            )

            missing_mask = (
                values.isna()
            )

            id_missing_count += int(
                missing_mask.sum()
            )

            nonmissing = (
                values[
                    ~missing_mask
                ]
            )

            id_nonmissing_count += len(
                nonmissing
            )

            if len(nonmissing) > 0:

                hashes = (
                    pd.util
                    .hash_pandas_object(
                        nonmissing,
                        index=False,
                    )
                    .to_numpy(
                        dtype=np.uint64
                    )
                )

                hash_arrays.append(
                    hashes
                )

    if has_exact_id:

        if hash_arrays:

            all_hashes = np.concatenate(
                hash_arrays
            )

            id_unique = int(
                np.unique(
                    all_hashes
                ).size
            )

        else:

            id_unique = 0

        id_missing = int(
            id_missing_count
        )

        id_duplicates = int(
            id_nonmissing_count
            - id_unique
        )

    return {
        "row_count": int(
            total_rows
        ),

        "has_exact_id": (
            has_exact_id
        ),

        "id_missing_count": (
            id_missing
        ),

        "id_unique_nonmissing_count": (
            id_unique
        ),

        "id_duplicate_nonmissing_count": (
            id_duplicates
        ),
    }


# =============================================================================
# PARQUET PROFILING
# =============================================================================


def profile_parquet(
    path,
):

    if pq is None:

        raise ImportError(
            "pyarrow is required to audit parquet files."
        )

    parquet = pq.ParquetFile(
        path
    )

    columns = (
        parquet
        .schema_arrow
        .names
    )

    total_rows = int(
        parquet.metadata.num_rows
    )

    # -------------------------------------------------------------------------
    # Sample from the first row group.
    # -------------------------------------------------------------------------

    if parquet.num_row_groups > 0:

        sample_table = (
            parquet
            .read_row_group(
                0
            )
            .slice(
                0,
                SAMPLE_ROWS,
            )
        )

        sample = (
            sample_table
            .to_pandas()
        )

    else:

        sample = pd.DataFrame(
            columns=columns
        )

    has_exact_id = (
        "id" in columns
    )

    id_missing = np.nan
    id_unique = np.nan
    id_duplicates = np.nan

    if has_exact_id:

        hash_arrays = []

        id_missing_count = 0
        id_nonmissing_count = 0

        for batch in (
            parquet.iter_batches(
                columns=["id"],
                batch_size=(
                    PARQUET_BATCH_SIZE
                ),
            )
        ):

            chunk = (
                batch
                .to_pandas()
            )

            values = (
                chunk["id"]
                .astype("string")
            )

            missing_mask = (
                values.isna()
            )

            id_missing_count += int(
                missing_mask.sum()
            )

            nonmissing = (
                values[
                    ~missing_mask
                ]
            )

            id_nonmissing_count += len(
                nonmissing
            )

            if len(nonmissing) > 0:

                hashes = (
                    pd.util
                    .hash_pandas_object(
                        nonmissing,
                        index=False,
                    )
                    .to_numpy(
                        dtype=np.uint64
                    )
                )

                hash_arrays.append(
                    hashes
                )

        if hash_arrays:

            all_hashes = np.concatenate(
                hash_arrays
            )

            id_unique = int(
                np.unique(
                    all_hashes
                ).size
            )

        else:

            id_unique = 0

        id_missing = int(
            id_missing_count
        )

        id_duplicates = int(
            id_nonmissing_count
            - id_unique
        )

    arrow_types = {
        field.name: str(
            field.type
        )
        for field in (
            parquet
            .schema_arrow
        )
    }

    return {
        "columns": columns,
        "sample": sample,
        "row_count": total_rows,
        "has_exact_id": (
            has_exact_id
        ),
        "id_missing_count": (
            id_missing
        ),
        "id_unique_nonmissing_count": (
            id_unique
        ),
        "id_duplicate_nonmissing_count": (
            id_duplicates
        ),
        "schema_types": (
            arrow_types
        ),
    }


# =============================================================================
# SCHEMA ROWS
# =============================================================================


def build_schema_rows(
    path,
    columns,
    sample,
    schema_types=None,
):

    schema_rows = []

    for position, column in enumerate(
        columns
    ):

        if column in sample.columns:

            series = (
                sample[column]
            )

            sample_missing = (
                series.isna()
                .mean()
                * 100
                if len(sample) > 0
                else np.nan
            )

            sample_nunique = (
                series.nunique(
                    dropna=True
                )
                if len(sample) > 0
                else 0
            )

            sample_dtype = str(
                series.dtype
            )

        else:

            sample_missing = np.nan
            sample_nunique = np.nan
            sample_dtype = ""

        if schema_types is not None:

            stored_dtype = (
                schema_types.get(
                    column,
                    sample_dtype,
                )
            )

        else:

            stored_dtype = (
                sample_dtype
            )

        semantic_class = (
            classify_column(
                column
            )
        )

        schema_rows.append(
            {
                "file_name": (
                    path.name
                ),

                "relative_path": str(
                    path
                ),

                "column_position": (
                    position
                ),

                "column_name": (
                    column
                ),

                "normalized_column_name": (
                    normalize_column_name(
                        column
                    )
                ),

                "observed_or_stored_dtype": (
                    stored_dtype
                ),

                "sample_rows_used": (
                    len(sample)
                ),

                "sample_missing_pct": (
                    sample_missing
                ),

                "sample_nonmissing_nunique": (
                    sample_nunique
                ),

                "semantic_discovery_class": (
                    semantic_class
                ),
            }
        )

    return schema_rows


# =============================================================================
# FIGURES
# =============================================================================


def create_figures(
    inventory,
):

    if len(inventory) == 0:
        return

    figure_data = (
        inventory[
            inventory[
                "audit_status"
            ]
            == "PASS"
        ]
        .copy()
    )

    if len(figure_data) == 0:
        return

    # -------------------------------------------------------------------------
    # Figure 1 — row counts
    # -------------------------------------------------------------------------

    rows_plot = (
        figure_data
        .sort_values(
            "row_count",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            11,
            max(
                5,
                0.48
                * len(rows_plot),
            ),
        )
    )

    ax.barh(
        rows_plot[
            "file_name"
        ],
        rows_plot[
            "row_count"
        ],
    )

    positive_rows = (
        rows_plot[
            rows_plot[
                "row_count"
            ]
            > 0
        ][
            "row_count"
        ]
    )

    if len(
        positive_rows
    ) > 1:

        ratio = (
            positive_rows.max()
            / positive_rows.min()
        )

        if ratio >= 100:

            ax.set_xscale(
                "log"
            )

            ax.set_xlabel(
                "Row count (log scale)"
            )

        else:

            ax.set_xlabel(
                "Row count"
            )

    ax.set_title(
        "Crunchbase Raw Dataset Row Counts"
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

    # -------------------------------------------------------------------------
    # Figure 2 — number of columns
    # -------------------------------------------------------------------------

    width_plot = (
        figure_data
        .sort_values(
            "column_count",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(
            11,
            max(
                5,
                0.48
                * len(width_plot),
            ),
        )
    )

    ax.barh(
        width_plot[
            "file_name"
        ],
        width_plot[
            "column_count"
        ],
    )

    ax.set_xlabel(
        "Number of columns"
    )

    ax.set_title(
        "Crunchbase Raw Dataset Schema Width"
    )

    fig.tight_layout()

    fig.savefig(
        SCHEMA_WIDTH_FIGURE,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# MAIN
# =============================================================================


def main():

    separator()

    print(
        "PHASE 3.2.1 — "
        "RAW DATASET & SCHEMA INVENTORY"
    )

    separator()

    if not RAW_DIR.exists():

        raise FileNotFoundError(
            f"Raw data directory not found: "
            f"{RAW_DIR}"
        )

    # -------------------------------------------------------------------------
    # 1. Discover data files
    # -------------------------------------------------------------------------

    data_files = sorted(
        [
            path
            for path in (
                RAW_DIR.rglob("*")
            )
            if (
                path.is_file()
                and
                path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ]
    )

    print(
        f"\nRaw directory: "
        f"{RAW_DIR.resolve()}"
    )

    print(
        f"Supported data files discovered: "
        f"{len(data_files)}"
    )

    if not data_files:

        raise RuntimeError(
            "No CSV or Parquet files were found "
            "under data/raw."
        )

    for path in data_files:

        print(
            f"  - {path}"
        )

    inventory_rows = []
    schema_rows = []

    # -------------------------------------------------------------------------
    # 2. Audit every file
    # -------------------------------------------------------------------------

    for index, path in enumerate(
        data_files,
        start=1,
    ):

        separator("-")

        print(
            f"[{index}/{len(data_files)}] "
            f"Auditing: {path.name}"
        )

        extension = (
            path.suffix
            .lower()
        )

        file_size_bytes = (
            path.stat()
            .st_size
        )

        try:

            # ================================================================
            # CSV
            # ================================================================

            if extension == ".csv":

                columns = (
                    read_csv_header(
                        path
                    )
                )

                if len(columns) == 0:

                    raise ValueError(
                        "CSV contains no columns."
                    )

                sample = (
                    read_csv_sample(
                        path,
                        columns,
                    )
                )

                row_id_stats = (
                    profile_csv_rows_and_id(
                        path,
                        columns,
                    )
                )

                schema_types = None

            # ================================================================
            # PARQUET
            # ================================================================

            elif extension == ".parquet":

                result = (
                    profile_parquet(
                        path
                    )
                )

                columns = (
                    result["columns"]
                )

                sample = (
                    result["sample"]
                )

                row_id_stats = {
                    "row_count": (
                        result[
                            "row_count"
                        ]
                    ),

                    "has_exact_id": (
                        result[
                            "has_exact_id"
                        ]
                    ),

                    "id_missing_count": (
                        result[
                            "id_missing_count"
                        ]
                    ),

                    "id_unique_nonmissing_count": (
                        result[
                            "id_unique_nonmissing_count"
                        ]
                    ),

                    "id_duplicate_nonmissing_count": (
                        result[
                            "id_duplicate_nonmissing_count"
                        ]
                    ),
                }

                schema_types = (
                    result[
                        "schema_types"
                    ]
                )

            else:

                continue

            # -----------------------------------------------------------------
            # ID uniqueness rate
            # -----------------------------------------------------------------

            row_count = int(
                row_id_stats[
                    "row_count"
                ]
            )

            if (
                row_id_stats[
                    "has_exact_id"
                ]
            ):

                nonmissing_id_count = (
                    row_count
                    - int(
                        row_id_stats[
                            "id_missing_count"
                        ]
                    )
                )

                if nonmissing_id_count > 0:

                    id_uniqueness_pct = (
                        row_id_stats[
                            "id_unique_nonmissing_count"
                        ]
                        / nonmissing_id_count
                        * 100
                    )

                else:

                    id_uniqueness_pct = (
                        np.nan
                    )

            else:

                id_uniqueness_pct = (
                    np.nan
                )

            # -----------------------------------------------------------------
            # Dataset-level inventory
            # -----------------------------------------------------------------

            candidate_columns = [
                col
                for col in columns
                if (
                    classify_column(
                        col
                    )
                    != "other"
                )
            ]

            inventory_rows.append(
                {
                    "file_name": (
                        path.name
                    ),

                    "relative_path": str(
                        path
                    ),

                    "extension": (
                        extension
                    ),

                    "file_size_bytes": (
                        file_size_bytes
                    ),

                    "file_size_human": (
                        human_file_size(
                            file_size_bytes
                        )
                    ),

                    "row_count": (
                        row_count
                    ),

                    "column_count": (
                        len(columns)
                    ),

                    "has_exact_id": (
                        row_id_stats[
                            "has_exact_id"
                        ]
                    ),

                    "id_missing_count": (
                        row_id_stats[
                            "id_missing_count"
                        ]
                    ),

                    "id_unique_nonmissing_count": (
                        row_id_stats[
                            "id_unique_nonmissing_count"
                        ]
                    ),

                    "id_duplicate_nonmissing_count": (
                        row_id_stats[
                            "id_duplicate_nonmissing_count"
                        ]
                    ),

                    "id_uniqueness_pct": (
                        id_uniqueness_pct
                    ),

                    "candidate_identifier_reference_column_count": (
                        len(
                            candidate_columns
                        )
                    ),

                    "audit_status": (
                        "PASS"
                    ),

                    "audit_error": "",
                }
            )

            # -----------------------------------------------------------------
            # Long schema
            # -----------------------------------------------------------------

            schema_rows.extend(
                build_schema_rows(
                    path=path,
                    columns=columns,
                    sample=sample,
                    schema_types=(
                        schema_types
                    ),
                )
            )

            print(
                f"Rows:             "
                f"{row_count:,}"
            )

            print(
                f"Columns:          "
                f"{len(columns):,}"
            )

            print(
                f"File size:        "
                f"{human_file_size(file_size_bytes)}"
            )

            print(
                f"Exact id column:  "
                f"{row_id_stats['has_exact_id']}"
            )

            if row_id_stats[
                "has_exact_id"
            ]:

                print(
                    f"id missing:       "
                    f"{int(row_id_stats['id_missing_count']):,}"
                )

                print(
                    f"id unique:        "
                    f"{int(row_id_stats['id_unique_nonmissing_count']):,}"
                )

                print(
                    f"id duplicates:    "
                    f"{int(row_id_stats['id_duplicate_nonmissing_count']):,}"
                )

            print(
                "\nCandidate identifier / "
                "reference columns:"
            )

            if candidate_columns:

                for column in (
                    candidate_columns
                ):

                    print(
                        f"  - {column}"
                        f" [{classify_column(column)}]"
                    )

            else:

                print(
                    "  (none identified by "
                    "name heuristic)"
                )

        except Exception as exc:

            warnings.warn(
                f"Could not audit {path}: {exc}"
            )

            inventory_rows.append(
                {
                    "file_name": (
                        path.name
                    ),

                    "relative_path": str(
                        path
                    ),

                    "extension": (
                        extension
                    ),

                    "file_size_bytes": (
                        file_size_bytes
                    ),

                    "file_size_human": (
                        human_file_size(
                            file_size_bytes
                        )
                    ),

                    "row_count": (
                        np.nan
                    ),

                    "column_count": (
                        np.nan
                    ),

                    "has_exact_id": (
                        np.nan
                    ),

                    "id_missing_count": (
                        np.nan
                    ),

                    "id_unique_nonmissing_count": (
                        np.nan
                    ),

                    "id_duplicate_nonmissing_count": (
                        np.nan
                    ),

                    "id_uniqueness_pct": (
                        np.nan
                    ),

                    "candidate_identifier_reference_column_count": (
                        np.nan
                    ),

                    "audit_status": (
                        "ERROR"
                    ),

                    "audit_error": str(
                        exc
                    ),
                }
            )

    # -------------------------------------------------------------------------
    # 3. Materialize audit tables
    # -------------------------------------------------------------------------

    inventory = (
        pd.DataFrame(
            inventory_rows
        )
    )

    schema = (
        pd.DataFrame(
            schema_rows
        )
    )

    if len(schema) > 0:

        candidate_schema = (
            schema[
                schema[
                    "semantic_discovery_class"
                ]
                != "other"
            ]
            .copy()
        )

    else:

        candidate_schema = (
            pd.DataFrame()
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory.to_csv(
        DATASET_INVENTORY_OUTPUT,
        index=False,
    )

    schema.to_csv(
        SCHEMA_OUTPUT,
        index=False,
    )

    candidate_schema.to_csv(
        CANDIDATE_COLUMNS_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 4. Generate first Phase-3 graphics
    # -------------------------------------------------------------------------

    create_figures(
        inventory
    )

    # -------------------------------------------------------------------------
    # 5. Global summary
    # -------------------------------------------------------------------------

    separator()

    print(
        "RAW DATASET INVENTORY SUMMARY"
    )

    separator()

    summary_columns = [
        "file_name",
        "row_count",
        "column_count",
        "file_size_human",
        "has_exact_id",
        "id_missing_count",
        "id_duplicate_nonmissing_count",
        "audit_status",
    ]

    print(
        inventory[
            summary_columns
        ]
        .sort_values(
            "row_count",
            ascending=False,
            na_position="last",
        )
        .to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # 6. Files containing exact ID
    # -------------------------------------------------------------------------

    separator("-")

    print(
        "FILES WITH AN EXACT 'id' COLUMN"
    )

    separator("-")

    exact_id_files = (
        inventory[
            inventory[
                "has_exact_id"
            ]
            == True
        ]
    )

    if len(
        exact_id_files
    ) > 0:

        print(
            exact_id_files[
                [
                    "file_name",
                    "row_count",
                    "id_missing_count",
                    "id_unique_nonmissing_count",
                    "id_duplicate_nonmissing_count",
                    "id_uniqueness_pct",
                ]
            ]
            .to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.4f}",
            )
        )

    else:

        print(
            "No files contain an exact "
            "'id' column."
        )

    # -------------------------------------------------------------------------
    # 7. Candidate reference structure
    # -------------------------------------------------------------------------

    separator("-")

    print(
        "CANDIDATE IDENTIFIER / REFERENCE "
        "COLUMNS BY DATASET"
    )

    separator("-")

    if len(
        candidate_schema
    ) > 0:

        for file_name, group in (
            candidate_schema
            .groupby(
                "file_name",
                sort=True,
            )
        ):

            print(
                f"\n{file_name}"
            )

            for _, row in (
                group.iterrows()
            ):

                print(
                    f"  - "
                    f"{row['column_name']}"
                    f"  "
                    f"[{row['semantic_discovery_class']}]"
                )

    else:

        print(
            "No candidate columns detected."
        )

    # -------------------------------------------------------------------------
    # 8. Errors
    # -------------------------------------------------------------------------

    errors = (
        inventory[
            inventory[
                "audit_status"
            ]
            != "PASS"
        ]
    )

    separator("-")

    print(
        "AUDIT ERRORS"
    )

    separator("-")

    if len(errors) == 0:

        print(
            "None."
        )

    else:

        print(
            errors[
                [
                    "file_name",
                    "audit_error",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # -------------------------------------------------------------------------
    # 9. Final status
    # -------------------------------------------------------------------------

    separator()

    print(
        "PHASE 3.2.1 AUDIT COMPLETE"
    )

    separator()

    print(
        f"""
Outputs written to:

{DATASET_INVENTORY_OUTPUT}
{SCHEMA_OUTPUT}
{CANDIDATE_COLUMNS_OUTPUT}

Figures written to:

{ROW_COUNT_FIGURE}
{SCHEMA_WIDTH_FIGURE}


IMPORTANT INTERPRETATION RULES

1. This audit has NOT concluded that any field is a foreign key.

2. A column marked "entity_reference_like" is only a candidate based on
   its column name.

3. An exact "id" field is only being audited as a possible table-level
   identifier.

4. No assumption has been made that the file containing companies is
   the master entity table.

5. The next phase will explicitly test the companies_df master-entity
   hypothesis against the actual discovered schemas.

6. No graph nodes or graph edges have been created.

7. Phase-1 and Phase-2 outputs have not been modified.


NEXT:

Phase 3.2.2 — companies_df Master-Entity Hypothesis Audit
"""
    )


if __name__ == "__main__":
    main()