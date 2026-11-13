from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PHASE 8.1.6A — MULTI-ENTITY DATA FOUNDATION EDA V1
# =============================================================================
#
# Scientific role:
#
#   DATA QUALITY / SCALE / COMPLETENESS / TEMPORAL EDA
#
# PURPOSE
# -------
#
# Perform systematic foundational EDA across every Crunchbase entity group
# available in the project.
#
# This phase DOES NOT:
#
#   - construct relations;
#   - modify the graph;
#   - select predictive features;
#   - choose architecture;
#   - train;
#   - score rankings;
#   - inspect April performance.
#
# All statistics are aggregate.
# No personal contact values are exported.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

RAW_DIR = Path(
    "data/raw"
)

PHASE814_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_4_multi_entity_schema_inventory"
)

PHASE814_FILE_INVENTORY = (
    PHASE814_DIR
    / "phase_8_1_4_raw_file_inventory_V1.csv"
)

PHASE815_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_5_relation_semantics"
)

PHASE815_SUMMARY = (
    PHASE815_DIR
    / "phase_8_1_5_relation_semantics_summary_V1.json"
)


OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_6_data_foundations/"
    "phase_8_1_6a_entity_foundation_eda"
)

CHART_DIR = (
    OUTPUT_DIR
    / "charts"
)

ENTITY_PROFILE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6a_entity_profile_V1.csv"
)

COLUMN_COVERAGE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6a_column_completeness_V1.csv"
)

TEMPORAL_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6a_temporal_field_profile_V1.csv"
)

CATEGORICAL_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6a_categorical_top_values_V1.csv"
)

COVERAGE_BIN_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6a_column_coverage_distribution_V1.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6a_eda_summary_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6a_analysis_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6a_derived_artifact_sha256_V1.json"
)


# =============================================================================
# DATASET GROUPS
# =============================================================================

FILE_GLOBS = {
    "companies":
        "companies*.csv",

    "people":
        "people*.csv",

    "contacts":
        "CRUNCHBASE_contact*.csv",

    "funding":
        "CRUNCHBASE_funding*.csv",

    "acquisitions":
        "CRUNCHBASE_acquisition*.csv",

    "investors":
        "CRUNCHBASE_investor*.csv",

    "events":
        "CRUNCHBASE_event*.csv",

    "hubs":
        "CRUNCHBASE_hub*.csv",

    "schools":
        "CRUNCHBASE_school*.csv",
}


# =============================================================================
# SAFE LOW-CARDINALITY / INTERPRETABLE FIELDS
# =============================================================================
#
# We deliberately exclude names, emails, phone numbers, Linkedin profiles,
# etc. from value-level exports.
# =============================================================================

CATEGORICAL_FIELDS = {

    "companies": [
        "country",
        "region",
        "operating_status",
        "company_type",
        "funding_stage",
        "last_funding_type",
        "last_equity_funding_type",
        "num_employees_enum",
        "revenue_range",
        "ipo_status",
        "acquisition_status",
    ],

    "people": [
        "country",
        "region",
    ],

    "contacts": [
        "organization_funding_stage",
        "organization_operating_status",
        "organization_last_funding_type",
        "organization_num_employees_enum",
        "organization_revenue_range",
        "organization_acquisition_status",
        "organization_ipo_status",
    ],

    "funding": [
        "country",
        "region",
        "investment_stage",
        "investment_type",
        "funding_stage",
        "revenue_range",
    ],

    "acquisitions": [
        "acquisition_type",
        "acquirer_country",
        "acquiree_country",
        "acquiree_funding_stage",
        "acquiree_last_funding_type",
    ],

    "investors": [
        "country",
        "region",
        "operating_status",
        "company_type",
        "investor_type",
        "investor_stage",
        "num_employees_enum",
        "ipo_status",
    ],

    "events": [
        "country",
        "region",
        "event_type",
    ],

    "hubs": [
        "country",
        "country_group",
        "auto_types",
        "employee_range",
        "category_groups",
    ],

    "schools": [
        "country",
        "region",
        "operating_status",
        "num_employees_enum",
        "ipo_status",
        "revenue_range",
        "category_groups",
    ],
}


# =============================================================================
# SETTINGS
# =============================================================================

CHUNK_SIZE = 100_000
TOP_K = 20

COVERAGE_BINS = [
    (-0.000001, 1.0, "0-1%"),
    (1.0, 10.0, "1-10%"),
    (10.0, 25.0, "10-25%"),
    (25.0, 50.0, "25-50%"),
    (50.0, 75.0, "50-75%"),
    (75.0, 90.0, "75-90%"),
    (90.0, 99.0, "90-99%"),
    (99.0, 100.000001, "99-100%"),
]


# =============================================================================
# HELPERS
# =============================================================================

def separator(
    char="=",
    width=118,
):
    print(
        char * width
    )


def require(
    condition,
    message,
):
    if not condition:

        raise ValueError(
            message
        )


def sha256_file(
    path,
    chunk_size=1024 * 1024,
):
    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def read_json(
    path,
):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def write_json(
    path,
    obj,
):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            default=str,
        )


def pct(
    numerator,
    denominator,
):
    if denominator == 0:
        return float("nan")

    return (
        100.0
        * float(
            numerator
        )
        /
        float(
            denominator
        )
    )


def normalize_nonempty_mask(
    series,
):
    mask = series.notna()

    if (
        pd.api.types.is_object_dtype(
            series.dtype
        )
        or
        pd.api.types.is_string_dtype(
            series.dtype
        )
    ):

        normalized = (
            series.astype(
                "string"
            )
            .str.strip()
        )

        mask = (
            mask
            &
            normalized.ne("")
        )

    return mask


def is_temporal_column(
    column,
):
    c = column.lower()

    if c.endswith(
        "_on"
    ):
        return True

    if c.endswith(
        "_at"
    ):
        return True

    if "date" in c:
        return True

    if c in {
        "starts_on",
        "ends_on",
        "founded_on",
        "announced_on",
        "went_public_on",
        "exited_on",
    }:
        return True

    return False


# =============================================================================
# MAIN
# =============================================================================

def main():

    separator()

    print(
        "PHASE 8.1.6A — "
        "MULTI-ENTITY DATA FOUNDATION EDA V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "DATA FOUNDATION EDA"
    )

    print(
        "All raw entity rows scanned: YES"
    )

    print(
        "Relations constructed:       NO"
    )

    print(
        "Features selected:           NO"
    )

    print(
        "Graph modified:              NO"
    )

    print(
        "Architecture selected:       NO"
    )

    print(
        "Model trained:               NO"
    )

    print(
        "April performance inspected: NO"
    )

    # =========================================================================
    # 1. Prerequisite gates
    # =========================================================================

    separator("-")

    print(
        "8.1.6A.1 — PREREQUISITE GATES"
    )

    separator("-")

    require(
        PHASE814_FILE_INVENTORY.exists(),
        "Missing Phase-8.1.4 file inventory.",
    )

    require(
        PHASE815_SUMMARY.exists(),
        "Missing Phase-8.1.5 summary.",
    )

    phase815 = read_json(
        PHASE815_SUMMARY
    )

    require(
        phase815.get(
            "status"
        )
        ==
        "COMPLETE",
        "Phase 8.1.5 is not complete.",
    )

    require(
        phase815.get(
            "edges_constructed"
        )
        is False,
        "Phase 8.1.5 unexpectedly constructed edges.",
    )

    inventory_hash = sha256_file(
        PHASE814_FILE_INVENTORY
    )

    relation_summary_hash = sha256_file(
        PHASE815_SUMMARY
    )

    print(
        "Phase 8.1.4 inventory: PASS"
    )

    print(
        "Phase 8.1.5 relation audit: PASS"
    )

    # =========================================================================
    # 2. Resolve raw files
    # =========================================================================

    separator("-")

    print(
        "8.1.6A.2 — ENTITY FILE RESOLUTION"
    )

    separator("-")

    entity_files = {}

    for entity, pattern in (
        FILE_GLOBS.items()
    ):

        files = sorted(
            RAW_DIR.glob(
                pattern
            )
        )

        require(
            len(
                files
            )
            >
            0,
            (
                f"No files found for "
                f"{entity}: {pattern}"
            ),
        )

        entity_files[
            entity
        ] = files

        print(
            f"{entity:<14} "
            f"{len(files):>2} file(s)"
        )

    # =========================================================================
    # 3. Full entity scan
    # =========================================================================

    separator("-")

    print(
        "8.1.6A.3 — FULL ENTITY SCAN"
    )

    separator("-")

    entity_rows = []
    column_rows = []
    temporal_rows = []
    categorical_rows = []

    for entity, files in (
        entity_files.items()
    ):

        print()

        separator(".")

        print(
            f"ENTITY: {entity.upper()}"
        )

        separator(".")

        total_rows = 0

        column_nonempty = Counter()
        column_presence_rows = Counter()
        column_files = Counter()

        all_columns = set()

        temporal_nonempty = Counter()
        temporal_parseable = Counter()
        temporal_min = {}
        temporal_max = {}

        categorical_counters = defaultdict(
            Counter
        )

        id_hash_parts = []
        id_nonempty_rows = 0

        file_column_map = {}

        # ---------------------------------------------------------------------
        # Read headers first.
        # ---------------------------------------------------------------------

        for path in files:

            header = pd.read_csv(
                path,
                nrows=0,
            )

            columns = header.columns.tolist()

            file_column_map[
                path
            ] = columns

            all_columns.update(
                columns
            )

            for column in columns:

                column_files[
                    column
                ] += 1

        temporal_fields = sorted(
            [
                c
                for c
                in all_columns
                if is_temporal_column(
                    c
                )
            ]
        )

        safe_categorical_fields = [
            c
            for c
            in CATEGORICAL_FIELDS.get(
                entity,
                []
            )
            if c
            in all_columns
        ]

        # ---------------------------------------------------------------------
        # Chunk through every row.
        # ---------------------------------------------------------------------

        for file_i, path in enumerate(
            files,
            start=1,
        ):

            print(
                f"[{file_i:02d}/{len(files):02d}] "
                f"{path.name}"
            )

            file_columns = (
                file_column_map[
                    path
                ]
            )

            for chunk in pd.read_csv(
                path,
                chunksize=CHUNK_SIZE,
                low_memory=False,
            ):

                n = len(
                    chunk
                )

                total_rows += n

                # -------------------------------------------------------------
                # Column completeness
                # -------------------------------------------------------------

                for column in (
                    file_columns
                ):

                    column_presence_rows[
                        column
                    ] += n

                    mask = (
                        normalize_nonempty_mask(
                            chunk[
                                column
                            ]
                        )
                    )

                    column_nonempty[
                        column
                    ] += int(
                        mask.sum()
                    )

                # -------------------------------------------------------------
                # ID integrity via deterministic uint64 hashing.
                #
                # Used only for EDA duplicate detection. Resolution audits
                # later operate on exact raw identifiers.
                # -------------------------------------------------------------

                if "id" in chunk.columns:

                    id_series = (
                        chunk[
                            "id"
                        ]
                        .astype(
                            "string"
                        )
                        .str.strip()
                    )

                    id_series = (
                        id_series[
                            id_series.notna()
                            &
                            id_series.ne("")
                        ]
                    )

                    id_nonempty_rows += len(
                        id_series
                    )

                    if len(
                        id_series
                    ):

                        hashes = (
                            pd.util.hash_pandas_object(
                                id_series,
                                index=False,
                            )
                            .to_numpy(
                                dtype=np.uint64
                            )
                        )

                        id_hash_parts.append(
                            hashes
                        )

                # -------------------------------------------------------------
                # Temporal field quality
                # -------------------------------------------------------------

                for column in (
                    temporal_fields
                ):

                    if column not in (
                        chunk.columns
                    ):
                        continue

                    raw = chunk[
                        column
                    ]

                    mask = (
                        normalize_nonempty_mask(
                            raw
                        )
                    )

                    raw = raw[
                        mask
                    ]

                    temporal_nonempty[
                        column
                    ] += len(
                        raw
                    )

                    if len(
                        raw
                    ) == 0:
                        continue

                    parsed = pd.to_datetime(
                        raw,
                        errors="coerce",
                        format="mixed",
                    )

                    valid = (
                        parsed.dropna()
                    )

                    temporal_parseable[
                        column
                    ] += len(
                        valid
                    )

                    if len(
                        valid
                    ):

                        local_min = (
                            valid.min()
                        )

                        local_max = (
                            valid.max()
                        )

                        if (
                            column
                            not in
                            temporal_min
                            or
                            local_min
                            <
                            temporal_min[
                                column
                            ]
                        ):

                            temporal_min[
                                column
                            ] = (
                                local_min
                            )

                        if (
                            column
                            not in
                            temporal_max
                            or
                            local_max
                            >
                            temporal_max[
                                column
                            ]
                        ):

                            temporal_max[
                                column
                            ] = (
                                local_max
                            )

                # -------------------------------------------------------------
                # Safe categorical distributions
                # -------------------------------------------------------------

                for column in (
                    safe_categorical_fields
                ):

                    if column not in (
                        chunk.columns
                    ):
                        continue

                    series = (
                        chunk[
                            column
                        ]
                        .astype(
                            "string"
                        )
                        .str.strip()
                    )

                    series = (
                        series[
                            series.notna()
                            &
                            series.ne("")
                        ]
                    )

                    if len(
                        series
                    ):

                        counts = (
                            series.value_counts(
                                dropna=True
                            )
                        )

                        categorical_counters[
                            column
                        ].update(
                            counts.to_dict()
                        )

        # ---------------------------------------------------------------------
        # Entity-level ID diagnostics
        # ---------------------------------------------------------------------

        if id_hash_parts:

            all_id_hashes = (
                np.concatenate(
                    id_hash_parts
                )
            )

            unique_hashes = int(
                np.unique(
                    all_id_hashes
                ).size
            )

            duplicate_hash_rows = (
                id_nonempty_rows
                -
                unique_hashes
            )

        else:

            unique_hashes = None
            duplicate_hash_rows = None

        # ---------------------------------------------------------------------
        # Completeness table
        # ---------------------------------------------------------------------

        coverages = []

        for column in sorted(
            all_columns
        ):

            nonempty = int(
                column_nonempty[
                    column
                ]
            )

            coverage = pct(
                nonempty,
                total_rows,
            )

            coverages.append(
                coverage
            )

            column_rows.append(
                {
                    "entity":
                        entity,

                    "column":
                        column,

                    "entity_rows":
                        total_rows,

                    "files_with_column":
                        int(
                            column_files[
                                column
                            ]
                        ),

                    "entity_file_count":
                        len(
                            files
                        ),

                    "rows_from_files_with_column":
                        int(
                            column_presence_rows[
                                column
                            ]
                        ),

                    "nonempty_rows":
                        nonempty,

                    "coverage_pct":
                        coverage,

                    "missing_pct":
                        (
                            100.0
                            -
                            coverage
                        ),
                }
            )

        coverage_array = np.asarray(
            coverages,
            dtype=float,
        )

        entity_rows.append(
            {
                "entity":
                    entity,

                "file_count":
                    len(
                        files
                    ),

                "rows":
                    total_rows,

                "columns_union":
                    len(
                        all_columns
                    ),

                "id_nonempty_rows":
                    id_nonempty_rows,

                "id_unique_hash64":
                    unique_hashes,

                "id_duplicate_rows_hash64":
                    duplicate_hash_rows,

                "median_column_coverage_pct":
                    float(
                        np.median(
                            coverage_array
                        )
                    ),

                "columns_coverage_ge_99pct":
                    int(
                        (
                            coverage_array
                            >=
                            99.0
                        ).sum()
                    ),

                "columns_coverage_lt_10pct":
                    int(
                        (
                            coverage_array
                            <
                            10.0
                        ).sum()
                    ),

                "temporal_field_count":
                    len(
                        temporal_fields
                    ),
            }
        )

        # ---------------------------------------------------------------------
        # Temporal output
        # ---------------------------------------------------------------------

        for column in (
            temporal_fields
        ):

            nonempty = int(
                temporal_nonempty[
                    column
                ]
            )

            parseable = int(
                temporal_parseable[
                    column
                ]
            )

            temporal_rows.append(
                {
                    "entity":
                        entity,

                    "column":
                        column,

                    "entity_rows":
                        total_rows,

                    "nonempty_rows":
                        nonempty,

                    "coverage_pct":
                        pct(
                            nonempty,
                            total_rows,
                        ),

                    "parseable_rows":
                        parseable,

                    "parseable_pct_of_nonempty":
                        pct(
                            parseable,
                            nonempty,
                        ),

                    "min_date":
                        (
                            str(
                                temporal_min[
                                    column
                                ].date()
                            )
                            if
                            column
                            in
                            temporal_min
                            else
                            None
                        ),

                    "max_date":
                        (
                            str(
                                temporal_max[
                                    column
                                ].date()
                            )
                            if
                            column
                            in
                            temporal_max
                            else
                            None
                        ),
                }
            )

        # ---------------------------------------------------------------------
        # Categorical top values
        # ---------------------------------------------------------------------

        for column, counter in (
            categorical_counters.items()
        ):

            total_nonempty = sum(
                counter.values()
            )

            for rank, (
                value,
                count,
            ) in enumerate(
                counter.most_common(
                    TOP_K
                ),
                start=1,
            ):

                categorical_rows.append(
                    {
                        "entity":
                            entity,

                        "column":
                            column,

                        "rank":
                            rank,

                        "value":
                            value,

                        "count":
                            int(
                                count
                            ),

                        "pct_of_nonempty":
                            pct(
                                count,
                                total_nonempty,
                            ),
                    }
                )

        print()

        print(
            f"Rows:                       "
            f"{total_rows:,}"
        )

        print(
            f"Columns:                    "
            f"{len(all_columns):,}"
        )

        print(
            f"Median column completeness: "
            f"{np.median(coverage_array):.2f}%"
        )

        print(
            f"Columns >=99% complete:     "
            f"{int((coverage_array >= 99).sum()):,}"
        )

        print(
            f"Columns <10% complete:      "
            f"{int((coverage_array < 10).sum()):,}"
        )

        print(
            f"Temporal fields:            "
            f"{len(temporal_fields):,}"
        )

        if duplicate_hash_rows is not None:

            print(
                f"ID duplicate rows (hash64): "
                f"{duplicate_hash_rows:,}"
            )

    # =========================================================================
    # 4. Build EDA tables
    # =========================================================================

    separator("-")

    print(
        "8.1.6A.4 — CONSOLIDATED EDA TABLES"
    )

    separator("-")

    entity_df = pd.DataFrame(
        entity_rows
    )

    column_df = pd.DataFrame(
        column_rows
    )

    temporal_df = pd.DataFrame(
        temporal_rows
    )

    categorical_df = pd.DataFrame(
        categorical_rows
    )

    coverage_bin_rows = []

    for entity in (
        entity_df[
            "entity"
        ]
    ):

        subset = column_df[
            column_df[
                "entity"
            ]
            ==
            entity
        ]

        for lower, upper, label in (
            COVERAGE_BINS
        ):

            count = int(
                (
                    (
                        subset[
                            "coverage_pct"
                        ]
                        >
                        lower
                    )
                    &
                    (
                        subset[
                            "coverage_pct"
                        ]
                        <=
                        upper
                    )
                ).sum()
            )

            coverage_bin_rows.append(
                {
                    "entity":
                        entity,

                    "coverage_bin":
                        label,

                    "column_count":
                        count,
                }
            )

    coverage_bin_df = pd.DataFrame(
        coverage_bin_rows
    )

    print()

    print(
        entity_df[
            [
                "entity",
                "rows",
                "columns_union",
                "median_column_coverage_pct",
                "columns_coverage_lt_10pct",
                "temporal_field_count",
            ]
        ]
        .sort_values(
            "rows",
            ascending=False,
        )
        .to_string(
            index=False
        )
    )

    # =========================================================================
    # 5. Persist tabular artifacts
    # =========================================================================

    separator("-")

    print(
        "8.1.6A.5 — PERSIST EDA TABLES"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    entity_df.to_csv(
        ENTITY_PROFILE_OUTPUT,
        index=False,
    )

    column_df.to_csv(
        COLUMN_COVERAGE_OUTPUT,
        index=False,
    )

    temporal_df.to_csv(
        TEMPORAL_OUTPUT,
        index=False,
    )

    categorical_df.to_csv(
        CATEGORICAL_OUTPUT,
        index=False,
    )

    coverage_bin_df.to_csv(
        COVERAGE_BIN_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 6. Charts
    # =========================================================================

    separator("-")

    print(
        "8.1.6A.6 — FOUNDATION CHARTS"
    )

    separator("-")

    # -------------------------------------------------------------------------
    # Chart 1: entity scale
    # -------------------------------------------------------------------------

    scale = (
        entity_df
        .sort_values(
            "rows",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        scale[
            "entity"
        ],
        scale[
            "rows"
        ],
    )

    ax.set_title(
        "Crunchbase entity scale"
    )

    ax.set_xlabel(
        "Rows"
    )

    fig.tight_layout()

    scale_chart = (
        CHART_DIR
        / "phase_8_1_6a_entity_row_counts_V1.png"
    )

    fig.savefig(
        scale_chart,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"WROTE {scale_chart}"
    )

    # -------------------------------------------------------------------------
    # Chart 2+: largest missingness fields per entity
    # -------------------------------------------------------------------------

    missing_charts = []

    for entity in (
        entity_df[
            "entity"
        ]
    ):

        subset = (
            column_df[
                column_df[
                    "entity"
                ]
                ==
                entity
            ]
            .sort_values(
                "missing_pct",
                ascending=False,
            )
            .head(
                15
            )
            .sort_values(
                "missing_pct",
                ascending=True,
            )
        )

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        ax.barh(
            subset[
                "column"
            ],
            subset[
                "missing_pct"
            ],
        )

        ax.set_xlim(
            0,
            100,
        )

        ax.set_xlabel(
            "Missing (%)"
        )

        ax.set_title(
            f"{entity}: 15 sparsest fields"
        )

        fig.tight_layout()

        path = (
            CHART_DIR
            /
            (
                "phase_8_1_6a_"
                f"{entity}_sparsest_fields_V1.png"
            )
        )

        fig.savefig(
            path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

        missing_charts.append(
            str(
                path
            )
        )

        print(
            f"WROTE {path}"
        )

    # =========================================================================
    # 7. Automatic foundation insights
    # =========================================================================

    separator("-")

    print(
        "8.1.6A.7 — AUTOMATIC FOUNDATION INSIGHTS"
    )

    separator("-")

    insight_rows = []

    for row in (
        entity_df.itertuples(
            index=False
        )
    ):

        insight = {
            "entity":
                row.entity,

            "rows":
                int(
                    row.rows
                ),

            "columns":
                int(
                    row.columns_union
                ),

            "median_completeness_pct":
                float(
                    row.median_column_coverage_pct
                ),

            "very_sparse_columns_lt10pct":
                int(
                    row.columns_coverage_lt_10pct
                ),

            "near_complete_columns_ge99pct":
                int(
                    row.columns_coverage_ge_99pct
                ),

            "temporal_fields":
                int(
                    row.temporal_field_count
                ),

            "id_duplicate_rows_hash64":
                (
                    int(
                        row.id_duplicate_rows_hash64
                    )
                    if
                    not pd.isna(
                        row.id_duplicate_rows_hash64
                    )
                    else
                    None
                ),
        }

        insight_rows.append(
            insight
        )

        print()

        print(
            f"{row.entity.upper()}"
        )

        print(
            f"  rows:                   "
            f"{int(row.rows):,}"
        )

        print(
            f"  median completeness:    "
            f"{row.median_column_coverage_pct:.2f}%"
        )

        print(
            f"  <10% populated fields:  "
            f"{int(row.columns_coverage_lt_10pct):,}"
        )

        print(
            f"  >=99% populated fields: "
            f"{int(row.columns_coverage_ge_99pct):,}"
        )

        print(
            f"  temporal fields:        "
            f"{int(row.temporal_field_count):,}"
        )

    # =========================================================================
    # 8. Summary / manifest
    # =========================================================================

    summary = {
        "schema_version":
            "PHASE_8_1_6A_MULTI_ENTITY_DATA_FOUNDATION_EDA_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            (
                "DATA_QUALITY_SCALE_COMPLETENESS_"
                "TEMPORAL_EDA"
            ),

        "entity_count":
            len(
                entity_df
            ),

        "entities":
            entity_df[
                "entity"
            ].tolist(),

        "total_raw_rows_across_entity_exports":
            int(
                entity_df[
                    "rows"
                ].sum()
            ),

        "phase_8_1_4_inventory_sha256":
            inventory_hash,

        "phase_8_1_5_summary_sha256":
            relation_summary_hash,

        "entity_foundation_insights":
            insight_rows,

        "charts": {
            "entity_scale":
                str(
                    scale_chart
                ),

            "sparsest_field_charts":
                missing_charts,
        },

        "privacy_policy":
            (
                "No raw personal contact values "
                "exported or printed."
            ),

        "relations_constructed":
            False,

        "features_selected":
            False,

        "graph_modified":
            False,

        "architecture_selected":
            False,

        "model_trained":
            False,

        "April_performance_inspected":
            False,

        "next_step":
            (
                "Entity-specific EDA focused on "
                "startup features and heterogeneous "
                "relationship resolution."
            ),
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    manifest = {
        "phase":
            "8.1.6A",

        "status":
            "COMPLETE",

        "inputs": {
            "phase_8_1_4_file_inventory":
                str(
                    PHASE814_FILE_INVENTORY
                ),

            "phase_8_1_5_summary":
                str(
                    PHASE815_SUMMARY
                ),
        },

        "outputs": [
            str(
                ENTITY_PROFILE_OUTPUT
            ),
            str(
                COLUMN_COVERAGE_OUTPUT
            ),
            str(
                TEMPORAL_OUTPUT
            ),
            str(
                CATEGORICAL_OUTPUT
            ),
            str(
                COVERAGE_BIN_OUTPUT
            ),
            str(
                SUMMARY_OUTPUT
            ),
            str(
                scale_chart
            ),
            *missing_charts,
        ],

        "prohibited_operations": [
            "edge_construction",
            "feature_selection",
            "graph_modification",
            "architecture_selection",
            "training",
            "inference",
            "ranking_scoring",
            "April_performance_inspection",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    hash_targets = [
        ENTITY_PROFILE_OUTPUT,
        COLUMN_COVERAGE_OUTPUT,
        TEMPORAL_OUTPUT,
        CATEGORICAL_OUTPUT,
        COVERAGE_BIN_OUTPUT,
        SUMMARY_OUTPUT,
        MANIFEST_OUTPUT,
        scale_chart,
        *[
            Path(
                path
            )
            for path
            in missing_charts
        ],
    ]

    hashes = {
        str(path):
            sha256_file(
                path
            )
        for path
        in hash_targets
    }

    write_json(
        HASH_OUTPUT,
        hashes,
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    separator()

    print(
        "PHASE 8.1.6A MULTI-ENTITY "
        "DATA FOUNDATION EDA COMPLETE"
    )

    separator()

    print()

    print(
        "ENTITIES FULLY SCANNED:       "
        f"{len(entity_df)}"
    )

    print(
        "COLUMN COMPLETENESS PROFILED: YES"
    )

    print(
        "TEMPORAL FIELDS PROFILED:     YES"
    )

    print(
        "CATEGORICAL DISTRIBUTIONS:    YES"
    )

    print(
        "FOUNDATION CHARTS CREATED:    YES"
    )

    print()

    print(
        "RELATIONS CONSTRUCTED:        NO"
    )

    print(
        "FEATURES SELECTED:            NO"
    )

    print(
        "GRAPH MODIFIED:               NO"
    )

    print(
        "ARCHITECTURE SELECTED:        NO"
    )

    print(
        "MODEL TRAINED:                NO"
    )

    print(
        "APRIL PERFORMANCE INSPECTED:  NO"
    )

    print()

    print(
        "Next:"
    )

    print(
        "Phase 8.1.6B — deep EDA of "
        "Company/startup attributes and their "
        "value for inductive representation."
    )

    print()

    print(
        f"Outputs written to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()