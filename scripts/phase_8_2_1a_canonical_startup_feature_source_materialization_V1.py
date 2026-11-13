from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.2.1A — CANONICAL STARTUP FEATURE-SOURCE MATERIALIZATION V1
# =============================================================================
#
# Scientific role:
#
#   FEATURE-MATERIALIZATION FOUNDATION
#
# PURPOSE
# -------
#
# Materialize ONE canonical, identity-aligned source table for the frozen
# 311,589 Startup-role nodes.
#
# This is deliberately BEFORE:
#
#   - TF-IDF fitting;
#   - SVD fitting;
#   - categorical vocabulary fitting;
#   - numeric scaling;
#   - founder-history derivation;
#   - graph materialization;
#   - model instantiation;
#   - training.
#
# Sources:
#
#   1. Phase-3 frozen Startup node universe/order.
#   2. Phase-4 audited resolved description + parsed categories.
#   3. Raw Companies fields approved by Phase-8.1 contracts.
#
# No outcome/current-performance field is loaded.
#
# =============================================================================


RAW_DIR = Path("data/raw")

COMPANY_PATHS = sorted(
    RAW_DIR.glob("companies*.csv")
)

NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/"
    "model_ready/"
    "node_index.parquet"
)

DESCRIPTION_MANIFEST_PATH = Path(
    "data/experimental/phase_4/"
    "description_contract/"
    "description_input_manifest.parquet"
)

PREFLIGHT_CONTRACT = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_0_preflight/"
    "phase_8_2_0_preflight_contract_V1.json"
)

EXECUTION_CONTRACT = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_12_execution_contract/"
    "phase_8_1_12_execution_contract_V3.json"
)


OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_1_feature_materialization"
)

SOURCE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1a_canonical_startup_source_V1.parquet"
)

DUPLICATE_AUDIT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1a_duplicate_consistency_audit_V1.csv"
)

COVERAGE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1a_source_coverage_V1.csv"
)

RAW_FILE_INVENTORY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1a_raw_company_file_inventory_V1.csv"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1a_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1a_artifact_sha256_V1.json"
)


EXPECTED_STARTUPS = 311_589
EXPECTED_COMPANY_FILES = 24

# Frozen result from comprehensive Startup EDA.
EXPECTED_MATCHED_COMPANY_ROWS = 312_130
EXPECTED_DUPLICATED_STARTUP_IDS = 541

# Frozen Phase-4 content coverage.
EXPECTED_STARTUPS_WITH_TEXT = 311_586
EXPECTED_STARTUPS_WITH_CATEGORIES = 303_688

CHUNK_SIZE = 200_000


APPROVED_RAW_FIELDS = [
    "id",
    "category_groups",
    "country",
    "region",
    "location_groups",
    "company_type",
    "founded_on",
    "founder_links",
]

STABLE_DUPLICATE_CHECK_FIELDS = [
    "category_groups",
    "country",
    "region",
    "location_groups",
    "company_type",
    "founded_on",
    "founder_links",
]


def separator(char="=", width=118):
    print(char * width)


def require(condition, message):

    if not bool(condition):
        raise ValueError(message)


def read_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def write_json(path, obj):

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


def sha256_file(path, chunk_size=1024 * 1024):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

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


def normalize_scalar(value):

    if pd.isna(value):
        return None

    value = str(
        value
    ).strip()

    if not value:
        return None

    if value.casefold() in {
        "nan",
        "none",
        "null",
    }:
        return None

    return value


def main():

    separator()

    print(
        "PHASE 8.2.1A — "
        "CANONICAL STARTUP FEATURE-SOURCE "
        "MATERIALIZATION V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "FEATURE-MATERIALIZATION FOUNDATION"
    )

    print(
        "Method design reopened:      NO"
    )

    print(
        "Learned preprocessing fit:   NO"
    )

    print(
        "Founder history derived:     NO"
    )

    print(
        "Graph materialized:          NO"
    )

    print(
        "Model instantiated:          NO"
    )

    print(
        "Training performed:          NO"
    )

    print(
        "Performance inspected:       NO"
    )

    print(
        "T60 scored:                  NO"
    )

    print(
        "April analyzed:              NO"
    )

    # =========================================================================
    # 1. Prerequisite gates
    # =========================================================================

    separator("-")

    print(
        "8.2.1A.1 — PREREQUISITE GATES"
    )

    separator("-")

    required = [
        NODE_INDEX_PATH,
        DESCRIPTION_MANIFEST_PATH,
        PREFLIGHT_CONTRACT,
        EXECUTION_CONTRACT,
    ]

    for path in required:

        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

        print(
            f"FOUND {path}"
        )

    require(
        len(COMPANY_PATHS)
        ==
        EXPECTED_COMPANY_FILES,
        (
            f"Expected {EXPECTED_COMPANY_FILES} Companies files; "
            f"found {len(COMPANY_PATHS)}."
        ),
    )

    print(
        f"FOUND {len(COMPANY_PATHS)} Companies shards"
    )

    preflight = read_json(
        PREFLIGHT_CONTRACT
    )

    execution = read_json(
        EXECUTION_CONTRACT
    )

    require(
        preflight.get("status")
        ==
        "FROZEN",
        "Phase 8.2.0 preflight is not frozen.",
    )

    require(
        preflight.get(
            "method_design_closed"
        )
        is True,
        "Method design is not closed.",
    )

    require(
        preflight.get(
            "training_allowed_now"
        )
        is False,
        "Training unexpectedly authorized.",
    )

    require(
        execution.get("status")
        ==
        "FROZEN",
        "Phase 8.1.12 V3 is not frozen.",
    )

    require(
        execution.get(
            "training_performed"
        )
        is False,
        "Training unexpectedly occurred.",
    )

    require(
        execution.get(
            "T60_scored"
        )
        is False,
        "T60 unexpectedly scored.",
    )

    require(
        execution.get(
            "April_analyzed"
        )
        is False,
        "April unexpectedly analyzed.",
    )

    print()

    print(
        "Framework-development entry gate: PASS"
    )

    # =========================================================================
    # 2. Frozen Startup role universe
    # =========================================================================

    separator("-")

    print(
        "8.2.1A.2 — FROZEN STARTUP ROLE UNIVERSE"
    )

    separator("-")

    nodes = pd.read_parquet(
        NODE_INDEX_PATH
    )

    required_node_columns = {
        "node_index",
        "node_id",
        "node_type",
        "raw_entity_id",
    }

    require(
        required_node_columns.issubset(
            nodes.columns
        ),
        (
            "Phase-3 node index is missing "
            "required identity columns."
        ),
    )

    startup_nodes = (
        nodes.loc[
            nodes[
                "node_type"
            ].astype(str)
            ==
            "startup",
            [
                "node_index",
                "node_id",
                "raw_entity_id",
            ],
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    startup_nodes[
        "raw_entity_id"
    ] = (
        startup_nodes[
            "raw_entity_id"
        ]
        .astype(str)
    )

    require(
        len(startup_nodes)
        ==
        EXPECTED_STARTUPS,
        (
            "Frozen Startup role count changed: "
            f"{len(startup_nodes):,}"
        ),
    )

    require(
        startup_nodes[
            "raw_entity_id"
        ].nunique()
        ==
        EXPECTED_STARTUPS,
        "Startup raw_entity_id is not unique.",
    )

    require(
        startup_nodes[
            "node_id"
        ].astype(str)
        .nunique()
        ==
        EXPECTED_STARTUPS,
        "Startup node_id is not unique.",
    )

    startup_id_set = set(
        startup_nodes[
            "raw_entity_id"
        ].tolist()
    )

    print(
        f"Frozen Startup role nodes: "
        f"{len(startup_nodes):,}"
    )

    print(
        "Candidate universe expanded: NO"
    )

    # =========================================================================
    # 3. Reuse audited Phase-4 text/category source
    # =========================================================================

    separator("-")

    print(
        "8.2.1A.3 — AUDITED CONTENT SOURCE"
    )

    separator("-")

    description = pd.read_parquet(
        DESCRIPTION_MANIFEST_PATH
    )

    required_description_columns = {
        "node_id",
        "node_type",
        "raw_entity_id",
        "text",
        "text_source",
        "has_text",
        "has_labels",
        "label_count",
        "labels_json",
    }

    require(
        required_description_columns.issubset(
            description.columns
        ),
        (
            "Description manifest is missing "
            "required columns."
        ),
    )

    description = (
        description.loc[
            description[
                "node_type"
            ].astype(str)
            ==
            "startup",
            [
                "node_id",
                "raw_entity_id",
                "text",
                "text_source",
                "has_text",
                "has_labels",
                "label_count",
                "labels_json",
            ],
        ]
        .copy()
    )

    description[
        "raw_entity_id"
    ] = (
        description[
            "raw_entity_id"
        ]
        .astype(str)
    )

    require(
        len(description)
        ==
        EXPECTED_STARTUPS,
        "Startup description manifest count drift.",
    )

    require(
        description[
            "raw_entity_id"
        ].nunique()
        ==
        EXPECTED_STARTUPS,
        (
            "Startup content manifest contains "
            "duplicate entity IDs."
        ),
    )

    text_count = int(
        description[
            "has_text"
        ].sum()
    )

    category_count = int(
        description[
            "has_labels"
        ].sum()
    )

    require(
        text_count
        ==
        EXPECTED_STARTUPS_WITH_TEXT,
        "Phase-4 Startup text coverage drift.",
    )

    require(
        category_count
        ==
        EXPECTED_STARTUPS_WITH_CATEGORIES,
        "Phase-4 Startup category coverage drift.",
    )

    print(
        f"Startups with text:       "
        f"{text_count:,}"
    )

    print(
        f"Startups with categories: "
        f"{category_count:,}"
    )

    print(
        "Phase-4 audited parser reused: YES"
    )

    print(
        "Doc2Vec reused:                NO"
    )

    # =========================================================================
    # 4. Raw Companies file inventory
    # =========================================================================

    separator("-")

    print(
        "8.2.1A.4 — RAW COMPANIES INVENTORY"
    )

    separator("-")

    raw_inventory_rows = []

    for i, path in enumerate(
        COMPANY_PATHS,
        start=1,
    ):

        raw_inventory_rows.append(
            {
                "file_index":
                    i,

                "filename":
                    path.name,

                "bytes":
                    int(
                        path.stat().st_size
                    ),
            }
        )

        print(
            f"[{i:02d}/{len(COMPANY_PATHS):02d}] "
            f"{path.name}"
        )

    raw_inventory_df = pd.DataFrame(
        raw_inventory_rows
    )

    # =========================================================================
    # 5. Scan ONLY approved structured fields
    # =========================================================================

    separator("-")

    print(
        "8.2.1A.5 — LOAD APPROVED STARTUP STRUCTURED SOURCE"
    )

    separator("-")

    matched_parts = []

    for i, path in enumerate(
        COMPANY_PATHS,
        start=1,
    ):

        print(
            f"[{i:02d}/{len(COMPANY_PATHS):02d}] "
            f"{path.name}"
        )

        for chunk in pd.read_csv(
            path,
            usecols=APPROVED_RAW_FIELDS,
            dtype={
                "id":
                    "string",
            },
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):

            chunk[
                "id"
            ] = (
                chunk[
                    "id"
                ]
                .astype(str)
            )

            matched = chunk.loc[
                chunk[
                    "id"
                ].isin(
                    startup_id_set
                )
            ].copy()

            if not matched.empty:

                matched_parts.append(
                    matched
                )

    require(
        len(matched_parts)
        >
        0,
        "No frozen Startup rows recovered from Companies.",
    )

    raw_startups = pd.concat(
        matched_parts,
        ignore_index=True,
    )

    print()

    print(
        f"Matched Companies rows: "
        f"{len(raw_startups):,}"
    )

    print(
        f"Unique Startup IDs:      "
        f"{raw_startups['id'].nunique():,}"
    )

    require(
        len(raw_startups)
        ==
        EXPECTED_MATCHED_COMPANY_ROWS,
        (
            "Matched Companies row count differs from "
            "Phase-8.1 comprehensive EDA."
        ),
    )

    require(
        raw_startups[
            "id"
        ].nunique()
        ==
        EXPECTED_STARTUPS,
        (
            "Raw Companies scan did not recover "
            "all frozen Startup IDs."
        ),
    )

    # =========================================================================
    # 6. Duplicate consistency
    # =========================================================================

    separator("-")

    print(
        "8.2.1A.6 — DUPLICATE STARTUP CONSISTENCY"
    )

    separator("-")

    duplicated_ids = set(
        raw_startups.loc[
            raw_startups[
                "id"
            ].duplicated(
                keep=False
            ),
            "id",
        ].astype(str)
    )

    print(
        f"Duplicated Startup IDs: "
        f"{len(duplicated_ids):,}"
    )

    require(
        len(duplicated_ids)
        ==
        EXPECTED_DUPLICATED_STARTUP_IDS,
        (
            "Duplicated Startup-ID count differs "
            "from frozen Phase-8.1 EDA."
        ),
    )

    duplicate_audit_rows = []

    for field in (
        STABLE_DUPLICATE_CHECK_FIELDS
    ):

        subset = (
            raw_startups.loc[
                raw_startups[
                    "id"
                ].isin(
                    duplicated_ids
                ),
                [
                    "id",
                    field,
                ],
            ]
            .copy()
        )

        subset[
            "_normalized"
        ] = (
            subset[
                field
            ]
            .map(
                normalize_scalar
            )
        )

        conflicts = int(
            subset
            .dropna(
                subset=[
                    "_normalized"
                ]
            )
            .groupby(
                "id"
            )[
                "_normalized"
            ]
            .nunique(
                dropna=True
            )
            .gt(1)
            .sum()
        )

        duplicate_audit_rows.append(
            {
                "field":
                    field,

                "duplicated_startup_ids":
                    len(
                        duplicated_ids
                    ),

                "conflicting_startup_ids":
                    conflicts,

                "status":
                    (
                        "PASS"
                        if conflicts == 0
                        else "FAIL"
                    ),
            }
        )

        print(
            f"{field:<28} "
            f"conflicting IDs = {conflicts:,}"
        )

    duplicate_audit_df = pd.DataFrame(
        duplicate_audit_rows
    )

    require(
        duplicate_audit_df[
            "conflicting_startup_ids"
        ].eq(
            0
        ).all(),
        (
            "At least one approved stable Startup field "
            "has conflicting duplicate values. "
            "Do not silently choose a row."
        ),
    )

    # After proving equality on every approved field,
    # first-row retention is deterministic and semantically harmless.

    raw_startups = (
        raw_startups
        .drop_duplicates(
            subset=[
                "id"
            ],
            keep="first",
        )
        .copy()
    )

    require(
        len(raw_startups)
        ==
        EXPECTED_STARTUPS,
        "Canonical raw Startup table size mismatch.",
    )

    # =========================================================================
    # 7. Normalize source scalars
    # =========================================================================

    separator("-")

    print(
        "8.2.1A.7 — NORMALIZE SOURCE REPRESENTATION"
    )

    separator("-")

    for field in [
        "category_groups",
        "country",
        "region",
        "location_groups",
        "company_type",
        "founder_links",
    ]:

        raw_startups[
            field
        ] = (
            raw_startups[
                field
            ]
            .map(
                normalize_scalar
            )
        )

    founded_raw = (
        raw_startups[
            "founded_on"
        ]
        .map(
            normalize_scalar
        )
    )

    raw_startups[
        "founded_on_raw"
    ] = founded_raw

    raw_startups[
        "founded_on"
    ] = pd.to_datetime(
        founded_raw,
        errors="coerce",
    )

    nonmissing_founded = int(
        founded_raw
        .notna()
        .sum()
    )

    parsed_founded = int(
        raw_startups[
            "founded_on"
        ]
        .notna()
        .sum()
    )

    founded_parse_failures = (
        nonmissing_founded
        -
        parsed_founded
    )

    print(
        f"founded_on nonmissing: "
        f"{nonmissing_founded:,}"
    )

    print(
        f"founded_on parsed:     "
        f"{parsed_founded:,}"
    )

    print(
        f"parse failures:        "
        f"{founded_parse_failures:,}"
    )

    # =========================================================================
    # 8. Exact identity-aligned merge
    # =========================================================================

    separator("-")

    print(
        "8.2.1A.8 — EXACT STARTUP NODE ALIGNMENT"
    )

    separator("-")

    structured = (
        raw_startups
        .rename(
            columns={
                "id":
                    "raw_entity_id",
            }
        )
    )

    content = (
        description
        .rename(
            columns={
                "labels_json":
                    "categories_json",
            }
        )
    )

    source = (
        startup_nodes
        .merge(
            content,
            on=[
                "node_id",
                "raw_entity_id",
            ],
            how="left",
            validate="one_to_one",
        )
        .merge(
            structured,
            on=[
                "raw_entity_id",
            ],
            how="left",
            validate="one_to_one",
        )
    )

    require(
        len(source)
        ==
        EXPECTED_STARTUPS,
        "Aligned source table row count mismatch.",
    )

    require(
        source[
            "raw_entity_id"
        ].nunique()
        ==
        EXPECTED_STARTUPS,
        "Aligned source entity IDs are not unique.",
    )

    require(
        source[
            "text_source"
        ].notna()
        .all(),
        (
            "At least one frozen Startup failed "
            "Phase-4 content attachment."
        ),
    )

    require(
        source[
            "founded_on_raw"
        ].isna()
        .sum()
        <
        EXPECTED_STARTUPS,
        "Structured Companies attachment appears empty.",
    )

    source = source.rename(
        columns={
            "node_index":
                "startup_node_index",

            "node_id":
                "startup_node_id",

            "raw_entity_id":
                "startup_id",

            "category_groups":
                "category_groups_raw",

            "country":
                "country_raw",

            "region":
                "region_raw",

            "location_groups":
                "location_groups_raw",

            "company_type":
                "company_type_raw",

            "founder_links":
                "founder_links_raw",
        }
    )

    source[
        "source_row"
    ] = range(
        len(source)
    )

    source = source[
        [
            "source_row",
            "startup_node_index",
            "startup_node_id",
            "startup_id",

            "text",
            "text_source",
            "has_text",

            "categories_json",
            "label_count",
            "has_labels",

            "category_groups_raw",
            "country_raw",
            "region_raw",
            "location_groups_raw",
            "company_type_raw",

            "founded_on",
            "founded_on_raw",
            "founder_links_raw",
        ]
    ].copy()

    require(
        source[
            "source_row"
        ].tolist()
        ==
        list(
            range(
                EXPECTED_STARTUPS
            )
        ),
        "Source rows are not contiguous.",
    )

    print(
        f"Aligned source rows: "
        f"{len(source):,}"
    )

    print(
        "One row per Startup: PASS"
    )

    print(
        "Phase-3 ordering preserved: PASS"
    )

    # =========================================================================
    # 9. Coverage audit
    # =========================================================================

    separator("-")

    print(
        "8.2.1A.9 — APPROVED SOURCE COVERAGE"
    )

    separator("-")

    coverage_fields = [
        ("text", "text"),
        ("categories", "categories_json"),
        ("category_groups", "category_groups_raw"),
        ("country", "country_raw"),
        ("region", "region_raw"),
        ("location_groups", "location_groups_raw"),
        ("company_type", "company_type_raw"),
        ("founded_on", "founded_on"),
        ("founder_links", "founder_links_raw"),
    ]

    coverage_rows = []

    for channel, column in (
        coverage_fields
    ):

        if channel == "categories":

            available = int(
                source[
                    "has_labels"
                ].sum()
            )

        else:

            available = int(
                source[
                    column
                ]
                .notna()
                .sum()
            )

        missing = (
            EXPECTED_STARTUPS
            -
            available
        )

        coverage = (
            available
            /
            EXPECTED_STARTUPS
        )

        coverage_rows.append(
            {
                "channel":
                    channel,

                "column":
                    column,

                "available":
                    available,

                "missing":
                    missing,

                "coverage_fraction":
                    coverage,

                "coverage_percent":
                    coverage
                    * 100.0,
            }
        )

        print(
            f"{channel:<20} "
            f"{available:>9,} / "
            f"{EXPECTED_STARTUPS:,} "
            f"({coverage * 100.0:6.2f}%)"
        )

    coverage_df = pd.DataFrame(
        coverage_rows
    )

    # =========================================================================
    # 10. Leakage boundary
    # =========================================================================

    separator("-")

    print(
        "8.2.1A.10 — LEAKAGE / SCOPE BOUNDARY"
    )

    separator("-")

    boundary = {
        "startup_id_used_as_predictor":
            False,

        "startup_node_index_used_as_predictor":
            False,

        "current_funding_totals_loaded":
            False,

        "current_funding_stage_loaded":
            False,

        "current_employee_counts_loaded":
            False,

        "current_revenue_loaded":
            False,

        "current_rank_loaded":
            False,

        "ipo_or_acquisition_outcome_loaded":
            False,

        "current_operating_status_loaded":
            False,

        "focal_startup_investment_history_loaded":
            False,

        "doc2vec_vector_loaded":
            False,

        "evaluation_matrix_loaded":
            False,

        "performance_metric_loaded":
            False,

        "learned_preprocessing_fit":
            False,

        "role_of_identifiers":
            (
                "alignment/audit only; never predictive input"
            ),
    }

    for key, value in (
        boundary.items()
    ):

        print(
            f"{key:<44} {value}"
        )

    # =========================================================================
    # 11. Persist source substrate
    # =========================================================================

    separator("-")

    print(
        "8.2.1A.11 — PERSIST CANONICAL SOURCE SUBSTRATE"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source.to_parquet(
        SOURCE_OUTPUT,
        index=False,
    )

    duplicate_audit_df.to_csv(
        DUPLICATE_AUDIT_OUTPUT,
        index=False,
    )

    coverage_df.to_csv(
        COVERAGE_OUTPUT,
        index=False,
    )

    raw_inventory_df.to_csv(
        RAW_FILE_INVENTORY_OUTPUT,
        index=False,
    )

    manifest = {
        "schema_version":
            "PHASE_8_2_1A_CANONICAL_STARTUP_SOURCE_V1",

        "phase":
            "8.2.1A",

        "status":
            "MATERIALIZED_AND_AUDITED",

        "scientific_role":
            "FEATURE_MATERIALIZATION_FOUNDATION",

        "startup_rows":
            int(
                len(source)
            ),

        "candidate_universe_expanded":
            False,

        "phase3_startup_order_preserved":
            True,

        "phase4_audited_text_reused":
            True,

        "phase4_audited_category_parsing_reused":
            True,

        "phase4_doc2vec_reused":
            False,

        "approved_raw_fields":
            APPROVED_RAW_FIELDS,

        "founded_on_nonmissing":
            nonmissing_founded,

        "founded_on_parseable":
            parsed_founded,

        "founded_on_parse_failures":
            founded_parse_failures,

        "duplicate_ids":
            len(
                duplicated_ids
            ),

        "duplicate_stable_field_conflicts":
            int(
                duplicate_audit_df[
                    "conflicting_startup_ids"
                ].sum()
            ),

        "learned_preprocessing_fit":
            False,

        "founder_history_derived":
            False,

        "graph_materialized":
            False,

        "model_instantiated":
            False,

        "training_performed":
            False,

        "performance_inspected":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,

        "source_output":
            str(
                SOURCE_OUTPUT
            ),

        "next_step":
            (
                "Phase 8.2.1B — exact founder identity "
                "resolution and pre-focal founder-history "
                "source materialization."
            ),
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    outputs = [
        SOURCE_OUTPUT,
        DUPLICATE_AUDIT_OUTPUT,
        COVERAGE_OUTPUT,
        RAW_FILE_INVENTORY_OUTPUT,
        MANIFEST_OUTPUT,
    ]

    write_json(
        HASH_OUTPUT,
        {
            str(path):
                sha256_file(path)
            for path
            in outputs
        },
    )

    separator()

    print(
        "PHASE 8.2.1A CANONICAL STARTUP "
        "SOURCE MATERIALIZED"
    )

    separator()

    print()

    print(
        f"STARTUP SOURCE ROWS:        "
        f"{len(source):,}"
    )

    print(
        "CANDIDATE UNIVERSE EXPANDED:NO"
    )

    print(
        "PHASE-3 ORDER PRESERVED:    YES"
    )

    print(
        "AUDITED RAW TEXT REUSED:    YES"
    )

    print(
        "AUDITED CATEGORIES REUSED:  YES"
    )

    print(
        "PHASE-4 DOC2VEC REUSED:     NO"
    )

    print(
        "LEARNED PREPROCESSING FIT:  NO"
    )

    print(
        "FOUNDER HISTORY DERIVED:    NO"
    )

    print(
        "GRAPH MATERIALIZED:         NO"
    )

    print(
        "MODEL INSTANTIATED:         NO"
    )

    print(
        "TRAINING PERFORMED:         NO"
    )

    print(
        "PERFORMANCE INSPECTED:      NO"
    )

    print(
        "T60 SCORED:                 NO"
    )

    print(
        "APRIL ANALYZED:             NO"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 8.2.1B — exact founder identity "
        "resolution and pre-focal founder-history "
        "source materialization."
    )


if __name__ == "__main__":
    main()
