from pathlib import Path
import hashlib
import json
import re

import pandas as pd


# =============================================================================
# PHASE 8.1.4 — CRUNCHBASE MULTI-ENTITY SCHEMA INVENTORY V1
# =============================================================================

RAW_DIR = Path("data/raw")

OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_4_multi_entity_schema_inventory"
)

FILE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_4_raw_file_inventory_V1.csv"
)

COLUMN_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_4_column_inventory_V1.csv"
)

ENTITY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_4_entity_schema_summary_V1.csv"
)

OPPORTUNITY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_4_candidate_field_opportunities_V1.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_4_schema_inventory_summary_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_4_analysis_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_4_derived_artifact_sha256_V1.json"
)


EXPECTED_ENTITY_GROUPS = [
    "companies",
    "acquisitions",
    "contacts",
    "events",
    "funding",
    "hubs",
    "investors",
    "schools",
    "people",
]


def separator(char="=", width=118):
    print(char * width)


def sha256_file(path, chunk_size=1024 * 1024):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


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


def classify_file(filename):

    name = filename.lower()

    # Ordering matters.
    rules = [
        ("acquisitions", ["acquisition"]),
        ("contacts", ["contact"]),
        ("funding", ["funding"]),
        ("hubs", ["hub"]),
        ("investors", ["investor"]),
        ("schools", ["school"]),
        ("people", ["people", "person"]),
        ("events", ["event"]),
        ("companies", ["compan"]),
    ]

    for entity, patterns in rules:

        if any(
            pattern in name
            for pattern in patterns
        ):
            return entity

    return "unknown"


def classify_column(column):

    c = column.lower()

    roles = []

    # -------------------------------------------------------------------------
    # Identifiers
    # -------------------------------------------------------------------------

    if (
        c == "id"
        or c.endswith("_id")
        or c.endswith("_ids")
        or "uuid" in c
        or "identifier" in c
    ):
        roles.append("IDENTIFIER")

    # -------------------------------------------------------------------------
    # Potential relational fields
    # -------------------------------------------------------------------------

    relationship_terms = [
        "link",
        "founder",
        "investor",
        "company",
        "organization",
        "person",
        "people",
        "school",
        "hub",
        "event",
        "contact",
        "acquir",
        "acquis",
        "participant",
        "attend",
        "member",
        "partner",
        "employee",
        "funding",
    ]

    if any(
        term in c
        for term in relationship_terms
    ):
        roles.append(
            "POTENTIAL_RELATION"
        )

    # -------------------------------------------------------------------------
    # Temporal fields
    # -------------------------------------------------------------------------

    temporal_terms = [
        "date",
        "_on",
        "_at",
        "year",
        "month",
        "founded",
        "announced",
        "closed",
        "created",
        "updated",
    ]

    if any(
        term in c
        for term in temporal_terms
    ):
        roles.append(
            "TEMPORAL"
        )

    # -------------------------------------------------------------------------
    # Content / attribute fields
    # -------------------------------------------------------------------------

    content_terms = [
        "description",
        "category",
        "industr",
        "location",
        "city",
        "country",
        "state",
        "region",
        "employee",
        "status",
        "type",
        "stage",
        "website",
        "domain",
        "revenue",
        "money",
        "amount",
        "rank",
    ]

    if any(
        term in c
        for term in content_terms
    ):
        roles.append(
            "POTENTIAL_FEATURE"
        )

    if not roles:
        roles.append(
            "OTHER"
        )

    return "|".join(
        sorted(
            set(
                roles
            )
        )
    )


def main():

    separator()

    print(
        "PHASE 8.1.4 — "
        "CRUNCHBASE MULTI-ENTITY SCHEMA INVENTORY V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "SCHEMA DISCOVERY ONLY"
    )

    print(
        "Relations constructed:       NO"
    )

    print(
        "Features selected:           NO"
    )

    print(
        "Graph changed:               NO"
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
    # 1. Discover CSV files
    # =========================================================================

    separator("-")

    print(
        "8.1.4.1 — RAW CSV DISCOVERY"
    )

    separator("-")

    csv_files = sorted(
        RAW_DIR.glob("*.csv")
    )

    if not csv_files:

        raise FileNotFoundError(
            "No CSV files found under data/raw/"
        )

    print(
        f"CSV files discovered: "
        f"{len(csv_files):,}"
    )

    file_rows = []
    column_rows = []

    # =========================================================================
    # 2. Header-only schema inspection
    # =========================================================================

    separator("-")

    print(
        "8.1.4.2 — HEADER-ONLY SCHEMA INVENTORY"
    )

    separator("-")

    for i, path in enumerate(
        csv_files,
        start=1,
    ):

        entity = classify_file(
            path.name
        )

        header = pd.read_csv(
            path,
            nrows=0,
            low_memory=False,
        )

        columns = header.columns.tolist()

        file_rows.append(
            {
                "filename":
                    path.name,

                "entity_group":
                    entity,

                "file_size_bytes":
                    path.stat().st_size,

                "column_count":
                    len(columns),

                "sha256":
                    sha256_file(path),
            }
        )

        for position, column in enumerate(
            columns
        ):

            column_rows.append(
                {
                    "filename":
                        path.name,

                    "entity_group":
                        entity,

                    "column_position":
                        position,

                    "column":
                        column,

                    "candidate_roles":
                        classify_column(
                            column
                        ),
                }
            )

        print(
            f"[{i:02d}/{len(csv_files):02d}] "
            f"{entity:<14} "
            f"{len(columns):>3} cols  "
            f"{path.name}"
        )

    files_df = pd.DataFrame(
        file_rows
    )

    columns_df = pd.DataFrame(
        column_rows
    )

    # =========================================================================
    # 3. Entity-level schema union / intersection
    # =========================================================================

    separator("-")

    print(
        "8.1.4.3 — ENTITY SCHEMA SUMMARY"
    )

    separator("-")

    entity_rows = []

    all_groups = sorted(
        files_df[
            "entity_group"
        ].unique()
    )

    for entity in all_groups:

        entity_files = files_df.loc[
            files_df[
                "entity_group"
            ]
            ==
            entity,
            "filename",
        ].tolist()

        entity_columns = (
            columns_df[
                columns_df[
                    "entity_group"
                ]
                ==
                entity
            ]
        )

        file_column_sets = []

        for filename in entity_files:

            cols = set(
                entity_columns.loc[
                    entity_columns[
                        "filename"
                    ]
                    ==
                    filename,
                    "column",
                ]
            )

            file_column_sets.append(
                cols
            )

        union_columns = (
            set().union(
                *file_column_sets
            )
            if file_column_sets
            else set()
        )

        intersection_columns = (
            set.intersection(
                *file_column_sets
            )
            if file_column_sets
            else set()
        )

        relation_columns = sorted(
            set(
                entity_columns.loc[
                    entity_columns[
                        "candidate_roles"
                    ].str.contains(
                        "POTENTIAL_RELATION"
                    ),
                    "column",
                ]
            )
        )

        temporal_columns = sorted(
            set(
                entity_columns.loc[
                    entity_columns[
                        "candidate_roles"
                    ].str.contains(
                        "TEMPORAL"
                    ),
                    "column",
                ]
            )
        )

        feature_columns = sorted(
            set(
                entity_columns.loc[
                    entity_columns[
                        "candidate_roles"
                    ].str.contains(
                        "POTENTIAL_FEATURE"
                    ),
                    "column",
                ]
            )
        )

        entity_rows.append(
            {
                "entity_group":
                    entity,

                "file_count":
                    len(
                        entity_files
                    ),

                "union_column_count":
                    len(
                        union_columns
                    ),

                "intersection_column_count":
                    len(
                        intersection_columns
                    ),

                "potential_relation_column_count":
                    len(
                        relation_columns
                    ),

                "potential_temporal_column_count":
                    len(
                        temporal_columns
                    ),

                "potential_feature_column_count":
                    len(
                        feature_columns
                    ),

                "potential_relation_columns":
                    json.dumps(
                        relation_columns,
                        ensure_ascii=False,
                    ),

                "potential_temporal_columns":
                    json.dumps(
                        temporal_columns,
                        ensure_ascii=False,
                    ),

                "potential_feature_columns":
                    json.dumps(
                        feature_columns,
                        ensure_ascii=False,
                    ),
            }
        )

    entity_df = pd.DataFrame(
        entity_rows
    )

    print(
        entity_df[
            [
                "entity_group",
                "file_count",
                "union_column_count",
                "potential_relation_column_count",
                "potential_temporal_column_count",
                "potential_feature_column_count",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 4. Candidate field opportunity register
    # =========================================================================

    separator("-")

    print(
        "8.1.4.4 — CANDIDATE FIELD OPPORTUNITIES"
    )

    separator("-")

    opportunity_df = (
        columns_df[
            ~columns_df[
                "candidate_roles"
            ].eq("OTHER")
        ]
        .drop_duplicates(
            subset=[
                "entity_group",
                "column",
                "candidate_roles",
            ]
        )
        .sort_values(
            [
                "entity_group",
                "candidate_roles",
                "column",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    for entity in all_groups:

        subset = opportunity_df[
            opportunity_df[
                "entity_group"
            ]
            ==
            entity
        ]

        print()

        print(
            f"{entity.upper()}"
        )

        if subset.empty:

            print(
                "  No automatically flagged "
                "candidate fields."
            )

            continue

        for row in subset.itertuples(
            index=False
        ):

            print(
                f"  {row.column:<40} "
                f"{row.candidate_roles}"
            )

    # =========================================================================
    # 5. Expected entity-group presence
    # =========================================================================

    separator("-")

    print(
        "8.1.4.5 — EXPECTED ENTITY COVERAGE"
    )

    separator("-")

    discovered_groups = set(
        files_df[
            "entity_group"
        ]
    )

    entity_presence = {}

    for entity in EXPECTED_ENTITY_GROUPS:

        present = (
            entity
            in
            discovered_groups
        )

        entity_presence[
            entity
        ] = present

        print(
            f"{entity:<14} "
            f"{'FOUND' if present else 'NOT FOUND'}"
        )

    # Do not fail if naming prevented automatic classification.
    # Unknown files are surfaced for inspection.
    unknown_files = files_df[
        files_df[
            "entity_group"
        ]
        ==
        "unknown"
    ]

    if not unknown_files.empty:

        print()

        print(
            "UNCLASSIFIED FILES:"
        )

        for filename in unknown_files[
            "filename"
        ]:

            print(
                f"  {filename}"
            )

    # =========================================================================
    # 6. Persist
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files_df.to_csv(
        FILE_OUTPUT,
        index=False,
    )

    columns_df.to_csv(
        COLUMN_OUTPUT,
        index=False,
    )

    entity_df.to_csv(
        ENTITY_OUTPUT,
        index=False,
    )

    opportunity_df.to_csv(
        OPPORTUNITY_OUTPUT,
        index=False,
    )

    summary = {
        "schema_version":
            "PHASE_8_1_4_MULTI_ENTITY_SCHEMA_INVENTORY_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            "SCHEMA_DISCOVERY_ONLY",

        "raw_csv_files":
            len(
                csv_files
            ),

        "expected_entity_groups":
            EXPECTED_ENTITY_GROUPS,

        "entity_presence":
            entity_presence,

        "discovered_entity_groups":
            sorted(
                discovered_groups
            ),

        "unknown_files":
            unknown_files[
                "filename"
            ].tolist(),

        "relations_constructed":
            False,

        "features_selected":
            False,

        "graph_changed":
            False,

        "architecture_selected":
            False,

        "model_trained":
            False,

        "April_performance_inspected":
            False,

        "next_step":
            (
                "Inspect actual relationship and "
                "feature semantics, coverage, and "
                "temporal provenance before adding "
                "any Phase-8 relation."
            ),
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    manifest = {
        "phase":
            "8.1.4",

        "status":
            "COMPLETE",

        "outputs": [
            str(
                FILE_OUTPUT
            ),
            str(
                COLUMN_OUTPUT
            ),
            str(
                ENTITY_OUTPUT
            ),
            str(
                OPPORTUNITY_OUTPUT
            ),
            str(
                SUMMARY_OUTPUT
            ),
        ],

        "prohibited_operations": [
            "relation_construction",
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

    hashes = {
        str(FILE_OUTPUT):
            sha256_file(FILE_OUTPUT),

        str(COLUMN_OUTPUT):
            sha256_file(COLUMN_OUTPUT),

        str(ENTITY_OUTPUT):
            sha256_file(ENTITY_OUTPUT),

        str(OPPORTUNITY_OUTPUT):
            sha256_file(OPPORTUNITY_OUTPUT),

        str(SUMMARY_OUTPUT):
            sha256_file(SUMMARY_OUTPUT),

        str(MANIFEST_OUTPUT):
            sha256_file(MANIFEST_OUTPUT),
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
        "PHASE 8.1.4 MULTI-ENTITY "
        "SCHEMA INVENTORY COMPLETE"
    )

    separator()

    print()

    print(
        "RAW DATASETS INVENTORIED:      YES"
    )

    print(
        "RELATIONS CONSTRUCTED:         NO"
    )

    print(
        "FEATURES SELECTED:             NO"
    )

    print(
        "GRAPH MODIFIED:                NO"
    )

    print(
        "ARCHITECTURE SELECTED:         NO"
    )

    print(
        "MODEL TRAINED:                 NO"
    )

    print(
        "APRIL PERFORMANCE INSPECTED:   NO"
    )

    print()

    print(
        "Next:"
    )

    print(
        "Audit which discovered fields "
        "actually define usable features "
        "or cross-entity relations."
    )

    print()

    print(
        f"Outputs written to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()