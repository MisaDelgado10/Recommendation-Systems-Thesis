from pathlib import Path
import hashlib
import json
import re

import pandas as pd


# =============================================================================
# PHASE 8.0.5 — FOUNDER TEMPORAL OBSERVABILITY AUDIT V1
# =============================================================================
#
# Scientific role:
#
#   OUTCOME-BLIND SIDE-INFORMATION TEMPORAL-PROVENANCE AUDIT
#
# PURPOSE
# -------
#
# Phase 3 established that SHARED_FOUNDER has:
#
#     temporal_provenance = current_snapshot_unversioned
#
# Before freezing Phase-8 rolling-origin development folds, determine
# whether the raw Crunchbase Company records contain any temporal fields
# that can constrain founder-edge availability.
#
# IMPORTANT:
#
#   founded_on  != founder-relation timestamp
#   created_at  != founder-relation timestamp
#   updated_at  != founder-relation timestamp
#
# These may provide entity-existence / record-existence evidence only.
# They must NOT be silently reinterpreted as historical founder-edge
# observation times.
#
# This script:
#
#   - does NOT alter Phase 3;
#   - does NOT create final founder masks;
#   - does NOT load a model;
#   - does NOT score any outcome;
#   - does NOT inspect ranking performance;
#   - does NOT select any architecture.
#
# =============================================================================


RAW_DIR = Path(
    "data/raw"
)

PHASE3_EDGES_PATH = Path(
    "data/experimental/phase_3/graph/edges.parquet"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_5_founder_temporal_observability"
)


SCHEMA_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_5_company_temporal_schema_V1.csv"
)

ENTITY_COVERAGE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_5_founder_entity_temporal_coverage_V1.csv"
)

ORIGIN_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_5_founder_origin_eligibility_V1.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_5_founder_temporal_observability_summary_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_5_analysis_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_5_derived_artifact_sha256_V1.json"
)


EXPECTED_PHASE3_EDGES = 158_818
EXPECTED_FOUNDER_EDGES = 94_818

EXPECTED_PHASE3_SHA256 = (
    "895871308de416b12d001550838cdb276"
    "3c87cd897774c6a55539a0ebb9a1ee2"
)


ORIGINS = [
    {
        "label": "T57",
        "history_end": "2025-03-31",
    },
    {
        "label": "T58",
        "history_end": "2025-06-30",
    },
    {
        "label": "T59",
        "history_end": "2025-09-30",
    },
    {
        "label": "APRIL_CONFIRMATORY",
        "history_end": "2026-03-31",
    },
]


TEMPORAL_CANDIDATE_COLUMNS = [
    "founded_on",
    "created_at",
    "updated_at",
    "closed_on",
]


def separator(char="=", width=118):
    print(char * width)


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


def clean_string(series):

    result = (
        series
        .astype("string")
        .str.strip()
    )

    result = result.mask(
        result.eq("")
    )

    return result


def temporal_founder_field(column_name):

    name = column_name.lower()

    contains_founder = (
        "founder" in name
    )

    contains_temporal_token = any(
        token in name
        for token in [
            "date",
            "time",
            "created",
            "updated",
            "start",
            "since",
            "_on",
            "_at",
        ]
    )

    return (
        contains_founder
        and
        contains_temporal_token
    )


def main():

    separator()

    print(
        "PHASE 8.0.5 — "
        "FOUNDER TEMPORAL OBSERVABILITY AUDIT V1"
    )

    separator()

    print()
    print(
        "Scientific role:             "
        "OUTCOME-BLIND SIDE-INFORMATION "
        "TEMPORAL-PROVENANCE AUDIT"
    )

    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Inference executed:          NO")
    print("Ranking metrics scored:      NO")
    print("Negatives generated:         NO")
    print("Model selection performed:   NO")
    print("Phase-3 graph modified:      NO")
    print("Founder mask frozen:         NO")
    print("April outcomes inspected:    NO")

    # =========================================================================
    # 1. Source integrity
    # =========================================================================

    separator("-")
    print(
        "8.0.5.1 — SOURCE EXISTENCE "
        "AND INTEGRITY"
    )
    separator("-")

    if not PHASE3_EDGES_PATH.exists():

        raise FileNotFoundError(
            PHASE3_EDGES_PATH
        )

    graph_sha = sha256_file(
        PHASE3_EDGES_PATH
    )

    print(
        f"FOUND  {PHASE3_EDGES_PATH}"
    )

    print(
        f"SHA256 {graph_sha}"
    )

    if (
        graph_sha
        != EXPECTED_PHASE3_SHA256
    ):

        raise ValueError(
            "Frozen Phase-3 graph fingerprint drift."
        )

    company_files = sorted(
        RAW_DIR.glob(
            "companies*.csv"
        )
    )

    if not company_files:

        raise FileNotFoundError(
            "No data/raw/companies*.csv files found."
        )

    print(
        f"Company CSV files: "
        f"{len(company_files):,}"
    )

    # =========================================================================
    # 2. Inspect raw Company schemas
    # =========================================================================

    separator("-")
    print(
        "8.0.5.2 — RAW COMPANY TEMPORAL SCHEMA AUDIT"
    )
    separator("-")

    schema_rows = []

    all_columns = set()

    for path in company_files:

        header = pd.read_csv(
            path,
            nrows=0,
        )

        columns = list(
            header.columns
        )

        all_columns.update(
            columns
        )

        lower_lookup = {
            c.lower(): c
            for c in columns
        }

        row = {
            "source_file":
                path.name,

            "column_count":
                len(columns),

            "has_id":
                "id"
                in lower_lookup,

            "has_founder_links":
                "founder_links"
                in lower_lookup,
        }

        for candidate in (
            TEMPORAL_CANDIDATE_COLUMNS
        ):

            row[
                f"has_{candidate}"
            ] = (
                candidate
                in lower_lookup
            )

        schema_rows.append(
            row
        )

    schema_table = pd.DataFrame(
        schema_rows
    )

    print(
        schema_table.to_string(
            index=False
        )
    )

    founder_temporal_columns = sorted(
        [
            column
            for column in all_columns
            if temporal_founder_field(
                column
            )
        ]
    )

    print()
    print(
        "Founder-specific temporal-looking columns:"
    )

    if founder_temporal_columns:

        for column in (
            founder_temporal_columns
        ):
            print(
                f"  {column}"
            )

    else:

        print(
            "  NONE"
        )

    exact_founder_timestamp_available = bool(
        founder_temporal_columns
    )

    # =========================================================================
    # 3. Load frozen shared-founder graph endpoints
    # =========================================================================

    separator("-")
    print(
        "8.0.5.3 — FROZEN SHARED-FOUNDER "
        "ENDPOINT UNIVERSE"
    )
    separator("-")

    edges = pd.read_parquet(
        PHASE3_EDGES_PATH,
        columns=[
            "edge_id",
            "relation",
            "underlying_src_id",
            "underlying_dst_id",
            "temporal_provenance",
        ],
    )

    if (
        len(edges)
        != EXPECTED_PHASE3_EDGES
    ):

        raise ValueError(
            "Phase-3 edge count drift."
        )

    founder_edges = edges[
        edges[
            "relation"
        ]
        ==
        "shared_founder"
    ].copy()

    if (
        len(founder_edges)
        != EXPECTED_FOUNDER_EDGES
    ):

        raise ValueError(
            "Shared-founder edge count drift."
        )

    founder_entity_ids = set(
        founder_edges[
            "underlying_src_id"
        ].astype(str)
    )

    founder_entity_ids.update(
        founder_edges[
            "underlying_dst_id"
        ].astype(str)
    )

    print(
        f"Shared-founder edges:       "
        f"{len(founder_edges):,}"
    )

    print(
        f"Unique endpoint entities:   "
        f"{len(founder_entity_ids):,}"
    )

    provenance_values = sorted(
        founder_edges[
            "temporal_provenance"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        "Founder temporal provenance: "
        f"{provenance_values}"
    )

    # =========================================================================
    # 4. Extract temporal metadata for founder-participating companies
    # =========================================================================

    separator("-")
    print(
        "8.0.5.4 — EXTRACT ENDPOINT "
        "ENTITY TEMPORAL METADATA"
    )
    separator("-")

    company_rows = []

    for file_index, path in enumerate(
        company_files,
        start=1,
    ):

        header = pd.read_csv(
            path,
            nrows=0,
        )

        available_columns = list(
            header.columns
        )

        required = [
            "id",
        ]

        if (
            "id"
            not in available_columns
        ):

            raise ValueError(
                f"{path} has no id column."
            )

        optional = [
            column
            for column in (
                TEMPORAL_CANDIDATE_COLUMNS
            )
            if column
            in available_columns
        ]

        usecols = (
            required
            +
            optional
        )

        print(
            f"[{file_index}/{len(company_files)}] "
            f"{path.name} | "
            f"reading {usecols}"
        )

        for chunk in pd.read_csv(
            path,
            usecols=usecols,
            dtype="string",
            chunksize=200_000,
            low_memory=False,
        ):

            chunk[
                "id"
            ] = clean_string(
                chunk[
                    "id"
                ]
            )

            subset = chunk[
                chunk[
                    "id"
                ]
                .isin(
                    founder_entity_ids
                )
            ].copy()

            if len(subset) == 0:
                continue

            for column in (
                TEMPORAL_CANDIDATE_COLUMNS
            ):

                if (
                    column
                    not in subset.columns
                ):

                    subset[
                        column
                    ] = pd.NA

            subset[
                "source_file"
            ] = (
                path.name
            )

            company_rows.append(
                subset[
                    [
                        "id",
                        "founded_on",
                        "created_at",
                        "updated_at",
                        "closed_on",
                        "source_file",
                    ]
                ]
            )

    if not company_rows:

        raise ValueError(
            "No founder endpoint entities "
            "resolved to Company records."
        )

    companies = pd.concat(
        company_rows,
        ignore_index=True,
    )

    # Preserve Phase-3 duplicate handling:
    # deterministic file ordering + keep last.
    companies = (
        companies
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
        .reset_index(
            drop=True
        )
    )

    for column in [
        "founded_on",
        "created_at",
        "updated_at",
        "closed_on",
    ]:

        companies[
            f"{column}_parsed"
        ] = pd.to_datetime(
            companies[
                column
            ],
            errors="coerce",
            utc=True,
        ).dt.tz_localize(
            None
        )

    resolved_ids = set(
        companies[
            "id"
        ].astype(str)
    )

    unresolved_ids = (
        founder_entity_ids
        -
        resolved_ids
    )

    print()
    print(
        f"Founder entities expected:  "
        f"{len(founder_entity_ids):,}"
    )

    print(
        f"Resolved Company records:   "
        f"{len(resolved_ids):,}"
    )

    print(
        f"Unresolved entities:        "
        f"{len(unresolved_ids):,}"
    )

    # =========================================================================
    # 5. Temporal-field coverage
    # =========================================================================

    separator("-")
    print(
        "8.0.5.5 — TEMPORAL FIELD COVERAGE"
    )
    separator("-")

    coverage_rows = []

    total_entities = len(
        founder_entity_ids
    )

    for column in [
        "founded_on",
        "created_at",
        "updated_at",
        "closed_on",
    ]:

        parsed_column = (
            f"{column}_parsed"
        )

        available = int(
            companies[
                parsed_column
            ]
            .notna()
            .sum()
        )

        coverage_rows.append(
            {
                "field":
                    column,

                "entities_with_parseable_value":
                    available,

                "founder_endpoint_entities":
                    total_entities,

                "coverage_pct":
                    (
                        100.0
                        * available
                        / total_entities
                        if total_entities
                        else 0.0
                    ),

                "scientific_interpretation":
                    {
                        "founded_on":
                            "entity-existence proxy only",

                        "created_at":
                            "Crunchbase record-existence proxy only",

                        "updated_at":
                            "record update metadata; "
                            "NOT relation-start time",

                        "closed_on":
                            "entity lifecycle metadata; "
                            "NOT founder-relation time",
                    }[
                        column
                    ],
            }
        )

    coverage_table = pd.DataFrame(
        coverage_rows
    )

    print(
        coverage_table.to_string(
            index=False
        )
    )

    # =========================================================================
    # 6. Attach entity temporal metadata to founder edges
    # =========================================================================

    company_dates = (
        companies[
            [
                "id",
                "founded_on_parsed",
                "created_at_parsed",
                "updated_at_parsed",
            ]
        ]
        .copy()
    )

    src_dates = (
        company_dates.rename(
            columns={
                "id":
                    "underlying_src_id",

                "founded_on_parsed":
                    "src_founded_on",

                "created_at_parsed":
                    "src_created_at",

                "updated_at_parsed":
                    "src_updated_at",
            }
        )
    )

    dst_dates = (
        company_dates.rename(
            columns={
                "id":
                    "underlying_dst_id",

                "founded_on_parsed":
                    "dst_founded_on",

                "created_at_parsed":
                    "dst_created_at",

                "updated_at_parsed":
                    "dst_updated_at",
            }
        )
    )

    founder_augmented = (
        founder_edges.merge(
            src_dates,
            on="underlying_src_id",
            how="left",
            validate="many_to_one",
        )
        .merge(
            dst_dates,
            on="underlying_dst_id",
            how="left",
            validate="many_to_one",
        )
    )

    if (
        len(founder_augmented)
        != EXPECTED_FOUNDER_EDGES
    ):

        raise ValueError(
            "Founder-edge augmentation "
            "changed edge count."
        )

    # =========================================================================
    # 7. Origin-specific observability constraints
    # =========================================================================

    separator("-")
    print(
        "8.0.5.6 — ORIGIN-SPECIFIC "
        "FOUNDER TEMPORAL CONSTRAINTS"
    )
    separator("-")

    origin_rows = []

    for origin_spec in ORIGINS:

        label = (
            origin_spec[
                "label"
            ]
        )

        origin = pd.Timestamp(
            origin_spec[
                "history_end"
            ]
        )

        src_founded_known = (
            founder_augmented[
                "src_founded_on"
            ].notna()
        )

        dst_founded_known = (
            founder_augmented[
                "dst_founded_on"
            ].notna()
        )

        both_founded_known = (
            src_founded_known
            &
            dst_founded_known
        )

        future_by_founded = (
            (
                src_founded_known
                &
                (
                    founder_augmented[
                        "src_founded_on"
                    ]
                    > origin
                )
            )
            |
            (
                dst_founded_known
                &
                (
                    founder_augmented[
                        "dst_founded_on"
                    ]
                    > origin
                )
            )
        )

        both_founded_pre_origin = (
            both_founded_known
            &
            (
                founder_augmented[
                    "src_founded_on"
                ]
                <= origin
            )
            &
            (
                founder_augmented[
                    "dst_founded_on"
                ]
                <= origin
            )
        )

        src_created_known = (
            founder_augmented[
                "src_created_at"
            ].notna()
        )

        dst_created_known = (
            founder_augmented[
                "dst_created_at"
            ].notna()
        )

        both_created_known = (
            src_created_known
            &
            dst_created_known
        )

        future_by_record_creation = (
            (
                src_created_known
                &
                (
                    founder_augmented[
                        "src_created_at"
                    ]
                    > origin
                )
            )
            |
            (
                dst_created_known
                &
                (
                    founder_augmented[
                        "dst_created_at"
                    ]
                    > origin
                )
            )
        )

        both_created_pre_origin = (
            both_created_known
            &
            (
                founder_augmented[
                    "src_created_at"
                ]
                <= origin
            )
            &
            (
                founder_augmented[
                    "dst_created_at"
                ]
                <= origin
            )
        )

        definite_future_proxy = (
            future_by_founded
            |
            future_by_record_creation
        )

        # This is deliberately NOT called "historically observed".
        #
        # Passing these necessary-condition checks means only that the
        # available entity/record metadata does not prove the edge was
        # impossible at that origin.
        not_disproven_by_proxy = (
            ~definite_future_proxy
        )

        row = {
            "origin":
                label,

            "history_end":
                origin.date(),

            "founder_edges_total":
                len(
                    founder_augmented
                ),

            "both_endpoints_founded_on_known":
                int(
                    both_founded_known.sum()
                ),

            "both_endpoints_founded_pre_origin":
                int(
                    both_founded_pre_origin.sum()
                ),

            "edges_with_endpoint_founded_after_origin":
                int(
                    future_by_founded.sum()
                ),

            "both_endpoints_created_at_known":
                int(
                    both_created_known.sum()
                ),

            "both_endpoint_records_created_pre_origin":
                int(
                    both_created_pre_origin.sum()
                ),

            "edges_with_endpoint_record_created_after_origin":
                int(
                    future_by_record_creation.sum()
                ),

            "definite_future_by_available_proxy":
                int(
                    definite_future_proxy.sum()
                ),

            "not_disproven_by_available_proxy":
                int(
                    not_disproven_by_proxy.sum()
                ),

            "historically_observed_founder_edges_proven":
                0,

            "exact_relation_timestamp_available":
                exact_founder_timestamp_available,
        }

        origin_rows.append(
            row
        )

        print()
        print(
            f"{label} | origin "
            f"{origin.date()}"
        )

        print(
            f"Founder edges:                         "
            f"{len(founder_augmented):,}"
        )

        print(
            f"Both endpoint founded_on known:        "
            f"{int(both_founded_known.sum()):,}"
        )

        print(
            f"Endpoint founded after origin:         "
            f"{int(future_by_founded.sum()):,}"
        )

        print(
            f"Both endpoint created_at known:        "
            f"{int(both_created_known.sum()):,}"
        )

        print(
            f"Endpoint record created after origin:  "
            f"{int(future_by_record_creation.sum()):,}"
        )

        print(
            f"Definite future by available proxy:    "
            f"{int(definite_future_proxy.sum()):,}"
        )

        print(
            f"Not disproven by available proxy:      "
            f"{int(not_disproven_by_proxy.sum()):,}"
        )

        print(
            "Historically observed founder edges "
            "proven:                              0"
        )

    origin_table = pd.DataFrame(
        origin_rows
    )

    # =========================================================================
    # 8. Interpretation
    # =========================================================================

    separator("-")
    print(
        "8.0.5.7 — SCIENTIFIC INTERPRETATION"
    )
    separator("-")

    if exact_founder_timestamp_available:

        relation_timestamp_status = (
            "TEMPORAL-LOOKING FOUNDER FIELD FOUND; "
            "SEMANTICS REQUIRE MANUAL REVIEW"
        )

    else:

        relation_timestamp_status = (
            "NO EXACT FOUNDER-RELATION "
            "TIMESTAMP IDENTIFIED"
        )

    print(
        relation_timestamp_status
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "founded_on can establish that an entity "
        "did not yet exist, but cannot establish "
        "when a founder relation became observable."
    )

    print(
        "created_at can establish when a Crunchbase "
        "entity record existed in the database only "
        "if its semantics are appropriate; it is not "
        "a founder-relation start timestamp."
    )

    print(
        "updated_at must not be used as founder-edge "
        "start time."
    )

    # =========================================================================
    # 9. Persist
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    schema_table.to_csv(
        SCHEMA_OUTPUT,
        index=False,
    )

    coverage_table.to_csv(
        ENTITY_COVERAGE_OUTPUT,
        index=False,
    )

    origin_table.to_csv(
        ORIGIN_OUTPUT,
        index=False,
    )

    summary = {
        "schema_version":
            "PHASE_8_0_5_FOUNDER_TEMPORAL_OBSERVABILITY_V1",

        "scientific_role":
            "OUTCOME_BLIND_SIDE_INFORMATION_TEMPORAL_PROVENANCE_AUDIT",

        "phase3_graph_sha256":
            graph_sha,

        "shared_founder_edges":
            len(
                founder_edges
            ),

        "founder_endpoint_entities":
            len(
                founder_entity_ids
            ),

        "company_records_resolved":
            len(
                resolved_ids
            ),

        "company_records_unresolved":
            len(
                unresolved_ids
            ),

        "founder_specific_temporal_looking_columns":
            founder_temporal_columns,

        "exact_founder_relationship_timestamp_available":
            exact_founder_timestamp_available,

        "founder_relation_timestamp_status":
            relation_timestamp_status,

        "allowed_interpretation": (
            "Entity founding and record metadata may "
            "identify edges that are definitely "
            "incompatible with a historical origin. "
            "They do not prove historical observation "
            "of the founder relation."
        ),

        "prohibited_interpretation": (
            "Do not treat founded_on, created_at, "
            "or updated_at as an exact timestamp for "
            "the SHARED_FOUNDER relation."
        ),

        "founder_mask_frozen":
            False,

        "development_folds_frozen":
            False,

        "april_confirmatory_outcomes_inspected":
            False,
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    manifest = {
        "phase":
            "8.0.5",

        "authoritative_inputs": {
            str(PHASE3_EDGES_PATH):
                graph_sha,

            "company_raw_files": [
                path.name
                for path in company_files
            ],
        },

        "outputs": [
            str(
                SCHEMA_OUTPUT
            ),
            str(
                ENTITY_COVERAGE_OUTPUT
            ),
            str(
                ORIGIN_OUTPUT
            ),
            str(
                SUMMARY_OUTPUT
            ),
        ],

        "prohibited_operations": [
            "model_loading",
            "checkpoint_loading",
            "inference",
            "ranking_scoring",
            "negative_generation",
            "model_selection",
            "phase3_graph_modification",
            "treat_proxy_as_exact_relation_timestamp",
            "april_confirmatory_scoring",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    derived_hashes = {
        str(SCHEMA_OUTPUT):
            sha256_file(
                SCHEMA_OUTPUT
            ),

        str(ENTITY_COVERAGE_OUTPUT):
            sha256_file(
                ENTITY_COVERAGE_OUTPUT
            ),

        str(ORIGIN_OUTPUT):
            sha256_file(
                ORIGIN_OUTPUT
            ),

        str(SUMMARY_OUTPUT):
            sha256_file(
                SUMMARY_OUTPUT
            ),

        str(MANIFEST_OUTPUT):
            sha256_file(
                MANIFEST_OUTPUT
            ),
    }

    write_json(
        HASH_OUTPUT,
        derived_hashes,
    )

    # =========================================================================
    # 10. Final summary
    # =========================================================================

    separator()
    print(
        "PHASE 8.0.5 AUDIT COMPLETE"
    )
    separator()

    print()
    print(
        f"Shared-founder edges:                 "
        f"{len(founder_edges):,}"
    )

    print(
        f"Founder endpoint entities:            "
        f"{len(founder_entity_ids):,}"
    )

    print(
        f"Resolved Company records:             "
        f"{len(resolved_ids):,}"
    )

    print()

    print(
        "Exact founder relation timestamp:     "
        f"{'FOUND' if exact_founder_timestamp_available else 'NOT FOUND'}"
    )

    print()

    print(
        "FOUNDER TEMPORAL MASK FROZEN:         NO"
    )

    print(
        "DEVELOPMENT FOLDS FROZEN:             NO"
    )

    print(
        "APRIL CONFIRMATORY OUTCOMES TOUCHED:  NO"
    )

    print()
    print(
        "Next decision:"
    )

    print(
        "Choose the scientifically defensible "
        "Phase-8 policy for unversioned founder "
        "side information based on this audit."
    )

    print()

    print(
        f"Outputs written to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()