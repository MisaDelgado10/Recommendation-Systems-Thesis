from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 8.2.1B — ORIGIN-SPECIFIC FOUNDER-HISTORY REMATERIALIZATION V2
# =============================================================================
#
# Scientific role:
#
#   CORRECTED ORIGIN-SPECIFIC FOUNDER-HISTORY MATERIALIZATION
#
# IMPORTANT
# ---------
#
# V1 identity artifacts remain authoritative:
#
#   - Founder URL -> Person resolution
#   - Company -> Founder Person edges
#   - Startup -> Founder Person edges
#   - Person node universe
#
# V1 predictive founder-history artifacts are superseded.
#
# V2 DOES NOT rescan raw Companies or People.
#
# For Startup s and prediction origin t:
#
#   H(s,t) =
#       unique non-focal Companies c
#       founded by at least one resolved founder of s
#       with parseable founded_on(c) < t.
#
# P11:
#   founder_prior_startup_count(s,t) = |H(s,t)|
#
# P12 source:
#   categories/category_groups from those same Companies H(s,t).
#
# No vocabulary/scaler/model is fit here.
#
# =============================================================================


PHASE8_FRAMEWORK = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development"
)

FEATURE_DIR = (
    PHASE8_FRAMEWORK
    / "phase_8_2_1_feature_materialization"
)

STARTUP_SOURCE = (
    FEATURE_DIR
    / "phase_8_2_1a_canonical_startup_source_V1.parquet"
)

V1_MANIFEST = (
    FEATURE_DIR
    / "phase_8_2_1b_manifest_V1.json"
)

COMPANY_FOUNDER_EDGES = (
    FEATURE_DIR
    / "phase_8_2_1b_resolved_company_founder_edges_V1.parquet"
)

STARTUP_FOUNDER_EDGES = (
    FEATURE_DIR
    / "phase_8_2_1b_startup_founder_edges_V1.parquet"
)

PERSON_NODE_SOURCE = (
    FEATURE_DIR
    / "phase_8_2_1b_person_node_source_V1.parquet"
)

TEMPORAL_METADATA = Path(
    "data/experimental/phase_2/"
    "temporal/"
    "itrs_segment_metadata.csv"
)

AMENDMENT_8_1_12B = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_12b_origin_referenced_founder_history/"
    "phase_8_1_12b_origin_referenced_founder_history_contract_V1.json"
)

MASKING_AMENDMENT = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_12a_outcome_independent_cold_start_masking/"
    "phase_8_1_12a_outcome_independent_masking_contract_V1.json"
)


OUTPUT_DIR = FEATURE_DIR

ORIGIN_TABLE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1b_origin_table_V2.csv"
)

HISTORY_SOURCE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1b_origin_specific_founder_history_source_V2.parquet"
)

COVERAGE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1b_origin_specific_founder_history_coverage_V2.csv"
)

MONOTONICITY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1b_origin_history_monotonicity_audit_V2.csv"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1b_manifest_V2.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1b_artifact_sha256_V2.json"
)


EXPECTED_STARTUPS = 311_589
EXPECTED_STARTUPS_WITH_RESOLVED_FOUNDER = 184_388

EXPECTED_ORIGINS = {
    "T57": "2025-04-01",
    "T58": "2025-07-01",
    "T59": "2025-10-01",
    "T60": "2026-01-01",
    "APRIL_CONFIRMATORY": "2026-04-01",
}


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

            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def first_nonmissing(series):

    values = series.dropna()

    if len(values) == 0:
        return None

    return values.iloc[0]


def main():

    separator()

    print(
        "PHASE 8.2.1B — "
        "ORIGIN-SPECIFIC FOUNDER-HISTORY "
        "REMATERIALIZATION V2"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "CORRECTED ORIGIN-SPECIFIC HISTORY MATERIALIZATION"
    )

    print("Raw Companies rescanned:     NO")
    print("Raw People rescanned:        NO")
    print("Identity resolution changed: NO")
    print("Feature discovery reopened:  NO")
    print("Learned preprocessing fit:   NO")
    print("Graph materialized:          NO")
    print("Model instantiated:          NO")
    print("Training performed:          NO")
    print("Performance inspected:       NO")
    print("T60 scored:                  NO")
    print("April analyzed:              NO")

    # =========================================================================
    # 1. Prerequisite gates
    # =========================================================================

    separator("-")

    print(
        "8.2.1B-V2.1 — PREREQUISITE GATES"
    )

    separator("-")

    required = [
        STARTUP_SOURCE,
        V1_MANIFEST,
        COMPANY_FOUNDER_EDGES,
        STARTUP_FOUNDER_EDGES,
        PERSON_NODE_SOURCE,
        TEMPORAL_METADATA,
        AMENDMENT_8_1_12B,
        MASKING_AMENDMENT,
    ]

    for path in required:

        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

        print(
            f"FOUND {path}"
        )

    v1_manifest = read_json(
        V1_MANIFEST
    )

    amendment = read_json(
        AMENDMENT_8_1_12B
    )

    masking = read_json(
        MASKING_AMENDMENT
    )

    require(
        v1_manifest.get("status")
        ==
        "MATERIALIZED_AND_AUDITED",
        "8.2.1B V1 identity substrate is not authoritative.",
    )

    require(
        amendment.get("status")
        ==
        "FROZEN",
        "8.1.12B amendment is not frozen.",
    )

    require(
        masking.get("status")
        ==
        "FROZEN",
        "8.1.12A masking amendment is not frozen.",
    )

    require(
        v1_manifest.get(
            "startups_with_resolved_founder"
        )
        ==
        EXPECTED_STARTUPS_WITH_RESOLVED_FOUNDER,
        "V1 resolved-founder count drift.",
    )

    print()

    print(
        "V1 identity artifacts reusable: PASS"
    )

    print(
        "V1 predictive history artifacts: SUPERSEDED"
    )

    # =========================================================================
    # 2. Derive exact prediction origins
    # =========================================================================

    separator("-")

    print(
        "8.2.1B-V2.2 — PREDICTION ORIGIN TABLE"
    )

    separator("-")

    temporal = pd.read_csv(
        TEMPORAL_METADATA
    )

    temporal[
        "segment_start_date"
    ] = pd.to_datetime(
        temporal[
            "segment_start_date"
        ],
        errors="coerce",
    )

    temporal[
        "segment_end_date"
    ] = pd.to_datetime(
        temporal[
            "segment_end_date"
        ],
        errors="coerce",
    )

    origin_rows = []

    for label in [
        "T57",
        "T58",
        "T59",
        "T60",
    ]:

        row = temporal.loc[
            temporal[
                "segment_label"
            ].astype(str)
            ==
            label
        ]

        require(
            len(row)
            ==
            1,
            f"Expected exactly one metadata row for {label}.",
        )

        origin = (
            row.iloc[0][
                "segment_start_date"
            ]
        )

        require(
            pd.notna(origin),
            f"{label} start date is missing.",
        )

        origin_rows.append(
            {
                "origin_label":
                    label,

                "prediction_origin":
                    origin,

                "origin_source":
                    (
                        "Phase-2 frozen target-segment start date"
                    ),
            }
        )

    t60_row = temporal.loc[
        temporal[
            "segment_label"
        ].astype(str)
        ==
        "T60"
    ].iloc[0]

    april_origin = (
        t60_row[
            "segment_end_date"
        ]
        +
        pd.Timedelta(
            days=1
        )
    )

    origin_rows.append(
        {
            "origin_label":
                "APRIL_CONFIRMATORY",

            "prediction_origin":
                april_origin,

            "origin_source":
                (
                    "first calendar day after frozen T60 endpoint"
                ),
        }
    )

    origin_df = pd.DataFrame(
        origin_rows
    )

    for row in origin_df.itertuples(
        index=False
    ):

        observed = (
            pd.Timestamp(
                row.prediction_origin
            )
            .strftime(
                "%Y-%m-%d"
            )
        )

        expected = EXPECTED_ORIGINS[
            row.origin_label
        ]

        require(
            observed
            ==
            expected,
            (
                f"{row.origin_label} origin drift: "
                f"expected {expected}, observed {observed}."
            ),
        )

    origin_df.to_csv(
        ORIGIN_TABLE_OUTPUT,
        index=False,
    )

    print(
        origin_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # 3. Load frozen Startup and founder identity substrates
    # =========================================================================

    separator("-")

    print(
        "8.2.1B-V2.3 — LOAD AUTHORITATIVE IDENTITY SUBSTRATES"
    )

    separator("-")

    startups = pd.read_parquet(
        STARTUP_SOURCE,
        columns=[
            "startup_node_index",
            "startup_node_id",
            "startup_id",
        ],
    )

    startup_founders = pd.read_parquet(
        STARTUP_FOUNDER_EDGES,
        columns=[
            "startup_id",
            "person_id",
        ],
    ).drop_duplicates()

    company_founders = pd.read_parquet(
        COMPANY_FOUNDER_EDGES,
        columns=[
            "company_id",
            "person_id",
            "company_founded_on",
            "categories_raw",
            "category_groups_raw",
            "company_is_frozen_startup",
        ],
    ).drop_duplicates()

    person_nodes = pd.read_parquet(
        PERSON_NODE_SOURCE
    )

    require(
        len(startups)
        ==
        EXPECTED_STARTUPS,
        "Frozen Startup count drift.",
    )

    require(
        startups[
            "startup_id"
        ].nunique()
        ==
        EXPECTED_STARTUPS,
        "Startup IDs are not unique.",
    )

    require(
        startup_founders[
            "startup_id"
        ].nunique()
        ==
        EXPECTED_STARTUPS_WITH_RESOLVED_FOUNDER,
        "Resolved-founder Startup coverage drift.",
    )

    require(
        not person_nodes[
            "identity_trainable"
        ].astype(bool).any(),
        "Person identity unexpectedly trainable.",
    )

    require(
        person_nodes[
            "initial_feature_policy"
        ]
        .astype(str)
        .eq(
            "ZERO_40"
        )
        .all(),
        "Person initialization policy drift.",
    )

    company_founders[
        "company_founded_on"
    ] = pd.to_datetime(
        company_founders[
            "company_founded_on"
        ],
        errors="coerce",
    )

    print(
        f"Frozen Startups:             {len(startups):,}"
    )

    print(
        f"Startup-Founder edges:       {len(startup_founders):,}"
    )

    print(
        f"Company-Founder edges:       {len(company_founders):,}"
    )

    print(
        f"Person nodes:                {len(person_nodes):,}"
    )

    # =========================================================================
    # 4. Construct founder-linked non-focal Company candidate substrate ONCE
    # =========================================================================

    separator("-")

    print(
        "8.2.1B-V2.4 — FOUNDER-LINKED COMPANY CANDIDATE SUBSTRATE"
    )

    separator("-")

    joined = (
        startup_founders
        .merge(
            company_founders,
            on="person_id",
            how="inner",
            validate="many_to_many",
        )
    )

    joined = joined.loc[
        joined[
            "company_id"
        ].astype(str)
        !=
        joined[
            "startup_id"
        ].astype(str)
    ].copy()

    joined = joined.loc[
        joined[
            "company_founded_on"
        ].notna()
    ].copy()

    require(
        not (
            joined[
                "company_id"
            ].astype(str)
            ==
            joined[
                "startup_id"
            ].astype(str)
        ).any(),
        "Focal Startup entered founder-history substrate.",
    )

    print(
        f"Founder-linked non-focal references: "
        f"{len(joined):,}"
    )

    print(
        "Focal Startup excluded: YES"
    )

    print(
        "Unparseable prior founded_on excluded: YES"
    )

    # =========================================================================
    # 5. Materialize one history snapshot per origin
    # =========================================================================

    separator("-")

    print(
        "8.2.1B-V2.5 — ORIGIN-SPECIFIC HISTORY MATERIALIZATION"
    )

    separator("-")

    history_sources = []
    coverage_rows = []
    pair_output_paths = []

    for origin_row in origin_df.itertuples(
        index=False
    ):

        label = (
            origin_row.origin_label
        )

        origin = pd.Timestamp(
            origin_row.prediction_origin
        )

        print()

        print(
            f"[{label}] prediction origin = "
            f"{origin.date()}"
        )

        eligible = joined.loc[
            joined[
                "company_founded_on"
            ]
            <
            origin
        ].copy()

        require(
            (
                eligible[
                    "company_founded_on"
                ]
                <
                origin
            ).all(),
            f"{label}: temporal eligibility invariant failed.",
        )

        # ---------------------------------------------------------------------
        # A Startup-priorCompany pair may appear through multiple founders.
        # Before aggregation, prove Company attributes agree.
        # ---------------------------------------------------------------------

        pair_key = [
            "startup_id",
            "company_id",
        ]

        for field in [
            "company_founded_on",
            "categories_raw",
            "category_groups_raw",
            "company_is_frozen_startup",
        ]:

            conflicts = int(
                eligible
                .groupby(
                    pair_key,
                    observed=True,
                )[
                    field
                ]
                .nunique(
                    dropna=True
                )
                .gt(1)
                .sum()
            )

            require(
                conflicts
                ==
                0,
                (
                    f"{label}: conflicting {field} values "
                    "within focal/prior Company pair."
                ),
            )

        pairs = (
            eligible
            .groupby(
                pair_key,
                observed=True,
                as_index=False,
            )
            .agg(
                prior_company_founded_on=(
                    "company_founded_on",
                    "first",
                ),

                shared_founder_count=(
                    "person_id",
                    "nunique",
                ),

                prior_categories_raw=(
                    "categories_raw",
                    first_nonmissing,
                ),

                prior_category_groups_raw=(
                    "category_groups_raw",
                    first_nonmissing,
                ),

                prior_company_is_frozen_startup=(
                    "company_is_frozen_startup",
                    "first",
                ),
            )
            .rename(
                columns={
                    "company_id":
                        "prior_company_id",
                }
            )
        )

        require(
            not pairs.duplicated(
                subset=[
                    "startup_id",
                    "prior_company_id",
                ]
            ).any(),
            f"{label}: duplicate prior Company pairs remain.",
        )

        require(
            (
                pairs[
                    "prior_company_founded_on"
                ]
                <
                origin
            ).all(),
            f"{label}: post-origin Company entered history.",
        )

        pairs[
            "origin_label"
        ] = label

        pairs[
            "prediction_origin"
        ] = origin

        pairs[
            "prior_company_external_context_only"
        ] = (
            ~pairs[
                "prior_company_is_frozen_startup"
            ].astype(bool)
        )

        pair_path = (
            OUTPUT_DIR
            /
            (
                "phase_8_2_1b_"
                f"{label.lower()}_founder_history_pairs_V2.parquet"
            )
        )

        pairs.to_parquet(
            pair_path,
            index=False,
        )

        pair_output_paths.append(
            pair_path
        )

        # ---------------------------------------------------------------------
        # Startup-level P11/P12 raw-source substrate.
        # ---------------------------------------------------------------------

        prior_count = (
            pairs
            .groupby(
                "startup_id",
                observed=True,
            )[
                "prior_company_id"
            ]
            .nunique()
            .rename(
                "founder_prior_startup_count"
            )
        )

        sector_company_count = (
            pairs.assign(
                _has_sector=(
                    pairs[
                        "prior_categories_raw"
                    ].notna()
                    |
                    pairs[
                        "prior_category_groups_raw"
                    ].notna()
                )
            )
            .loc[
                lambda x:
                    x[
                        "_has_sector"
                    ]
            ]
            .groupby(
                "startup_id",
                observed=True,
            )[
                "prior_company_id"
            ]
            .nunique()
            .rename(
                "founder_prior_sector_company_count"
            )
        )

        history = (
            startups
            .merge(
                prior_count,
                on="startup_id",
                how="left",
                validate="one_to_one",
            )
            .merge(
                sector_company_count,
                on="startup_id",
                how="left",
                validate="one_to_one",
            )
        )

        history[
            "founder_prior_startup_count"
        ] = (
            history[
                "founder_prior_startup_count"
            ]
            .fillna(0)
            .astype(
                "int32"
            )
        )

        history[
            "founder_prior_sector_company_count"
        ] = (
            history[
                "founder_prior_sector_company_count"
            ]
            .fillna(0)
            .astype(
                "int32"
            )
        )

        history[
            "founder_prior_sector_source_available"
        ] = (
            history[
                "founder_prior_sector_company_count"
            ]
            >
            0
        )

        history[
            "has_origin_founder_history"
        ] = (
            history[
                "founder_prior_startup_count"
            ]
            >
            0
        )

        history[
            "origin_label"
        ] = label

        history[
            "prediction_origin"
        ] = origin

        require(
            len(history)
            ==
            EXPECTED_STARTUPS,
            f"{label}: Startup history row count drift.",
        )

        history_sources.append(
            history
        )

        coverage_rows.append(
            {
                "origin_label":
                    label,

                "prediction_origin":
                    origin,

                "unique_prior_company_pairs":
                    int(
                        len(
                            pairs
                        )
                    ),

                "startups_with_prior_history":
                    int(
                        history[
                            "has_origin_founder_history"
                        ].sum()
                    ),

                "startups_with_prior_sector_source":
                    int(
                        history[
                            "founder_prior_sector_source_available"
                        ].sum()
                    ),

                "external_context_prior_company_pairs":
                    int(
                        pairs[
                            "prior_company_external_context_only"
                        ].sum()
                    ),

                "maximum_prior_company_founded_on":
                    (
                        pairs[
                            "prior_company_founded_on"
                        ].max()
                        if len(pairs)
                        else pd.NaT
                    ),
            }
        )

        print(
            f"  unique prior Company pairs: "
            f"{len(pairs):,}"
        )

        print(
            f"  Startups with history:      "
            f"{int(history['has_origin_founder_history'].sum()):,}"
        )

        print(
            f"  Startups with sector source:"
            f" {int(history['founder_prior_sector_source_available'].sum()):,}"
        )

    # =========================================================================
    # 6. Consolidated one-row-per-startup-per-origin source
    # =========================================================================

    separator("-")

    print(
        "8.2.1B-V2.6 — CONSOLIDATED ORIGIN-SPECIFIC SOURCE"
    )

    separator("-")

    history_all = pd.concat(
        history_sources,
        ignore_index=True,
    )

    expected_rows = (
        EXPECTED_STARTUPS
        *
        len(
            EXPECTED_ORIGINS
        )
    )

    require(
        len(history_all)
        ==
        expected_rows,
        (
            "Consolidated founder-history row count mismatch: "
            f"expected {expected_rows:,}, "
            f"found {len(history_all):,}."
        ),
    )

    require(
        not history_all.duplicated(
            subset=[
                "origin_label",
                "startup_id",
            ]
        ).any(),
        (
            "Duplicate origin/startup history rows detected."
        ),
    )

    history_all.to_parquet(
        HISTORY_SOURCE_OUTPUT,
        index=False,
    )

    coverage_df = pd.DataFrame(
        coverage_rows
    )

    coverage_df.to_csv(
        COVERAGE_OUTPUT,
        index=False,
    )

    print(
        f"Rows: {len(history_all):,}"
    )

    print(
        f"Origins: {history_all['origin_label'].nunique()}"
    )

    # =========================================================================
    # 7. Temporal monotonicity audit
    # =========================================================================

    separator("-")

    print(
        "8.2.1B-V2.7 — TEMPORAL MONOTONICITY AUDIT"
    )

    separator("-")

    order = [
        "T57",
        "T58",
        "T59",
        "T60",
        "APRIL_CONFIRMATORY",
    ]

    counts = (
        history_all[
            [
                "startup_id",
                "origin_label",
                "founder_prior_startup_count",
            ]
        ]
        .pivot(
            index="startup_id",
            columns="origin_label",
            values="founder_prior_startup_count",
        )
        .reindex(
            columns=order
        )
    )

    audit_rows = []

    for earlier, later in zip(
        order[:-1],
        order[1:],
    ):

        violations = int(
            (
                counts[
                    later
                ]
                <
                counts[
                    earlier
                ]
            ).sum()
        )

        audit_rows.append(
            {
                "earlier_origin":
                    earlier,

                "later_origin":
                    later,

                "startup_count":
                    EXPECTED_STARTUPS,

                "count_decrease_violations":
                    violations,

                "status":
                    (
                        "PASS"
                        if violations == 0
                        else "FAIL"
                    ),
            }
        )

        print(
            f"{earlier:>20} -> {later:<20} "
            f"decreases = {violations:,}"
        )

    monotonicity_df = pd.DataFrame(
        audit_rows
    )

    require(
        monotonicity_df[
            "count_decrease_violations"
        ].eq(
            0
        ).all(),
        (
            "Founder-history counts decreased across "
            "chronologically later origins."
        ),
    )

    monotonicity_df.to_csv(
        MONOTONICITY_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 8. Confirm superseded V1 semantics are absent from V2
    # =========================================================================

    separator("-")

    print(
        "8.2.1B-V2.8 — SUPERSESSION / LEAKAGE BOUNDARY"
    )

    separator("-")

    boundary = {
        "history_reference":
            "prediction_origin",

        "focal_founded_on_used_as_history_cutoff":
            False,

        "future_positive_ids_used":
            False,

        "future_negative_ids_used":
            False,

        "evaluation_candidate_matrix_used":
            False,

        "external_companies_used_as_candidates":
            False,

        "external_companies_added_as_graph_nodes":
            False,

        "external_companies_used_as_history_context":
            True,

        "person_identity_embedding":
            False,

        "person_initial_feature_policy":
            "ZERO_40",

        "sector_vocabulary_fit":
            False,

        "learned_preprocessing_fit":
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
    }

    for key, value in boundary.items():

        print(
            f"{key:<50} {value}"
        )

    # =========================================================================
    # 9. Persist manifest and hashes
    # =========================================================================

    separator("-")

    print(
        "8.2.1B-V2.9 — PERSIST CORRECTED MATERIALIZATION"
    )

    separator("-")

    manifest = {
        "schema_version":
            "PHASE_8_2_1B_ORIGIN_HISTORY_V2",

        "phase":
            "8.2.1B",

        "version":
            "V2",

        "status":
            "MATERIALIZED_AND_AUDITED",

        "scientific_role":
            (
                "CORRECTED_ORIGIN_SPECIFIC_FOUNDER_HISTORY"
            ),

        "authoritative_temporal_contract":
            str(
                AMENDMENT_8_1_12B
            ),

        "V1_identity_resolution_reused":
            True,

        "V1_company_founder_edges_reused":
            True,

        "V1_startup_founder_edges_reused":
            True,

        "V1_person_nodes_reused":
            True,

        "V1_prefocal_history_superseded":
            True,

        "history_reference":
            "prediction_origin",

        "history_rule":
            (
                "prior_company.founded_on < prediction_origin"
            ),

        "focal_startup_excluded":
            True,

        "deduplicate_prior_company_by_uuid":
            True,

        "origins":
            {
                row.origin_label:
                    pd.Timestamp(
                        row.prediction_origin
                    ).strftime(
                        "%Y-%m-%d"
                    )
                for row
                in origin_df.itertuples(
                    index=False
                )
            },

        "startup_rows_per_origin":
            EXPECTED_STARTUPS,

        "total_history_source_rows":
            int(
                len(
                    history_all
                )
            ),

        "learned_preprocessing_fit":
            False,

        "sector_vocabulary_fit":
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

        "outputs": {
            "origin_table":
                str(
                    ORIGIN_TABLE_OUTPUT
                ),

            "history_source":
                str(
                    HISTORY_SOURCE_OUTPUT
                ),

            "coverage":
                str(
                    COVERAGE_OUTPUT
                ),

            "monotonicity":
                str(
                    MONOTONICITY_OUTPUT
                ),

            "origin_pair_files":
                [
                    str(path)
                    for path
                    in pair_output_paths
                ],
        },

        "next_step":
            (
                "Phase 8.2.1C — outcome-independent origin "
                "cold/warm masks and training-only preprocessing "
                "population materialization."
            ),
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    outputs = [
        ORIGIN_TABLE_OUTPUT,
        HISTORY_SOURCE_OUTPUT,
        COVERAGE_OUTPUT,
        MONOTONICITY_OUTPUT,
        MANIFEST_OUTPUT,
    ] + pair_output_paths

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
        "PHASE 8.2.1B V2 ORIGIN-SPECIFIC "
        "FOUNDER HISTORY MATERIALIZED"
    )

    separator()

    print()

    print(
        "V1 IDENTITY RESOLUTION REUSED:       YES"
    )

    print(
        "V1 COMPANY-FOUNDER EDGES REUSED:     YES"
    )

    print(
        "V1 PERSON NODES REUSED:              YES"
    )

    print(
        "V1 PRE-FOCAL HISTORY:                SUPERSEDED"
    )

    print()

    print(
        "AUTHORITATIVE HISTORY RULE:"
    )

    print(
        "  prior_company.founded_on "
        "< prediction_origin"
    )

    print()

    print(
        "ORIGINS:"
    )

    for row in origin_df.itertuples(
        index=False
    ):

        print(
            f"  {row.origin_label:<20} "
            f"{pd.Timestamp(row.prediction_origin).date()}"
        )

    print()

    print(
        "FOCAL STARTUP EXCLUDED:              YES"
    )

    print(
        "PRIOR COMPANY UUID DEDUP:            YES"
    )

    print(
        "TEMPORAL MONOTONICITY:               PASS"
    )

    print(
        "EXTERNAL COMPANY CANDIDATES ADDED:   NO"
    )

    print(
        "SECTOR VOCABULARY FIT:               NO"
    )

    print(
        "TRAINING PERFORMED:                  NO"
    )

    print(
        "PERFORMANCE INSPECTED:               NO"
    )

    print(
        "T60 SCORED:                          NO"
    )

    print(
        "APRIL ANALYZED:                      NO"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 8.2.1C — outcome-independent "
        "cold/warm masks + training-only "
        "preprocessing populations."
    )


if __name__ == "__main__":
    main()
