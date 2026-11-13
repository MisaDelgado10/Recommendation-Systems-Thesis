from pathlib import Path
from collections import Counter
import hashlib
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PHASE 8.1.6B — STARTUP ATTRIBUTE DEEP EDA V1
# =============================================================================
#
# Scientific role:
#
#   STARTUP-UNIVERSE DATA QUALITY / COVERAGE / LEAKAGE-RISK EDA
#
# SCOPE
# -----
#
# Focus specifically on the 311,589 frozen Phase-3 Startup-role nodes.
#
# Questions:
#
#   1. How many raw Company rows resolve to the frozen Startup universe?
#   2. How common are duplicate Company IDs inside that universe?
#   3. When duplicate IDs exist, do their attribute values conflict?
#   4. Which structured Company attributes are actually populated?
#   5. Which attributes are available for development cold-start cases?
#   6. Which attributes carry obvious post-origin leakage risk?
#
# DEVELOPMENT cohorts:
#
#   T57 / T58 / T59
#
# April feature distributions are NOT inspected.
#
# NO:
#
#   - feature selection
#   - graph modification
#   - architecture selection
#   - training
#   - inference
#   - ranking scoring
#   - April feature-distribution analysis
#   - April performance inspection
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

RAW_DIR = Path(
    "data/raw"
)

NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/"
    "model_ready/"
    "node_index.parquet"
)

TARGETS_PATH = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_9_origin_eligibility_candidate_universe/"
    "phase_8_0_9_origin_eligible_targets_V5.parquet"
)

PHASE816A_SUMMARY = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_6_data_foundations/"
    "phase_8_1_6a_entity_foundation_eda/"
    "phase_8_1_6a_eda_summary_V1.json"
)


OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_6_data_foundations/"
    "phase_8_1_6b_startup_attribute_eda"
)

CHART_DIR = (
    OUTPUT_DIR
    / "charts"
)

COVERAGE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6b_startup_attribute_coverage_V1.csv"
)

PRIMARY_COVERAGE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6b_primary_attribute_coverage_V1.csv"
)

DUPLICATE_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6b_startup_duplicate_summary_V1.csv"
)

CONFLICT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6b_duplicate_field_conflicts_V1.csv"
)

RISK_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6b_attribute_temporal_risk_register_V1.csv"
)

TOP_VALUES_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6b_categorical_top_values_V1.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6b_eda_summary_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6b_analysis_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6b_derived_artifact_sha256_V1.json"
)


# =============================================================================
# CONSTANTS
# =============================================================================

EXPECTED_STARTUPS = 311_589

DEVELOPMENT_ORIGINS = [
    "T57",
    "T58",
    "T59",
]

EXPECTED_PRIMARY = {
    "T57": 6_850,
    "T58": 6_872,
    "T59": 6_764,
}

CHUNK_SIZE = 100_000
TOP_K = 20


# =============================================================================
# STRUCTURED COMPANY ATTRIBUTES TO AUDIT
#
# Description/categories already have dedicated Phase-4/8.1 audits.
# Here we focus on the additional structured Company surface.
# =============================================================================

REQUESTED_FIELDS = [
    "id",

    # Geography / organization semantics
    "country",
    "region",
    "city",
    "locations",
    "location_groups",
    "company_type",
    "category_groups",

    # Current snapshot operating characteristics
    "operating_status",
    "num_employees_enum",
    "revenue_range",

    # Funding / financing state
    "funding_stage",
    "last_funding_type",
    "last_equity_funding_type",
    "funding_total_usd",
    "equity_funding_total_usd",
    "last_funding_total_usd",
    "num_funding_rounds",
    "num_investors",
    "num_lead_investors",

    # Outcome-like fields
    "acquisition_status",
    "ipo_status",
    "num_acquisitions",

    # Structural / relation surfaces
    "founder_links",
    "founders",
    "num_founders",
    "hub_tags",
    "num_contacts",

    # Dates
    "founded_on",
    "last_funding_at",
    "last_key_employee_change_date",
    "last_layoff_date",
    "went_public_on",
    "exited_on",

    # Rankings / external current-state signals
    "rank_org",
    "rank_org_company",
    "rank_delta_d7",
    "rank_delta_d30",
    "rank_delta_d90",
    "semrush_global_rank",
    "semrush_visits_latest_month",
    "semrush_visits_latest_6_months_avg",

    # Operational / identity surfaces
    "website",
    "linkedin",
    "contact_job_departments",
]


# =============================================================================
# TEMPORAL / SCIENTIFIC RISK REGISTER
# =============================================================================

RISK_CLASSIFICATION = {

    # -------------------------------------------------------------------------
    # Potential matched-snapshot semantic extensions.
    # Historical value still NOT proven.
    # -------------------------------------------------------------------------

    "country": (
        "POTENTIAL_SNAPSHOT_EXTENSION",
        "SNAPSHOT_UNVERSIONED",
        "Geographic side information; historical value at each origin not proven.",
    ),

    "region": (
        "POTENTIAL_SNAPSHOT_EXTENSION",
        "SNAPSHOT_UNVERSIONED",
        "Geographic side information; historical value at each origin not proven.",
    ),

    "city": (
        "POTENTIAL_SNAPSHOT_EXTENSION",
        "SNAPSHOT_UNVERSIONED",
        "Geographic side information; location may change through time.",
    ),

    "locations": (
        "POTENTIAL_SNAPSHOT_EXTENSION",
        "SNAPSHOT_UNVERSIONED",
        "Potentially multi-location current snapshot.",
    ),

    "location_groups": (
        "POTENTIAL_SNAPSHOT_EXTENSION",
        "SNAPSHOT_UNVERSIONED",
        "Current snapshot geography grouping.",
    ),

    "company_type": (
        "POTENTIAL_SNAPSHOT_EXTENSION",
        "SNAPSHOT_UNVERSIONED",
        "Company classification snapshot; historical value not versioned.",
    ),

    "category_groups": (
        "POTENTIAL_SNAPSHOT_EXTENSION",
        "SNAPSHOT_UNVERSIONED",
        "Broader category grouping; potentially redundant with frozen categories.",
    ),

    # -------------------------------------------------------------------------
    # Current-state variables — high historical leakage risk.
    # -------------------------------------------------------------------------

    "operating_status": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT",
        "May reflect shutdown/acquisition/exit after prediction origin.",
    ),

    "num_employees_enum": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT",
        "Current company size can reflect post-origin growth.",
    ),

    "revenue_range": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT",
        "Current revenue state can reflect post-origin success.",
    ),

    "funding_stage": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT",
        "Directly summarizes financing that may occur after prediction origin.",
    ),

    "last_funding_type": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT",
        "Can encode future funding rounds relative to historical origins.",
    ),

    "last_equity_funding_type": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT",
        "Can encode future equity funding.",
    ),

    "funding_total_usd": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT",
        "Direct cumulative financing outcome.",
    ),

    "equity_funding_total_usd": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT",
        "Direct cumulative financing outcome.",
    ),

    "last_funding_total_usd": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT",
        "Potential post-origin financing information.",
    ),

    "num_funding_rounds": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT",
        "Current total includes rounds after historical origins.",
    ),

    "num_investors": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT",
        "Current investor count can encode future investment outcomes.",
    ),

    "num_lead_investors": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT",
        "Current lead-investor count can encode future investment outcomes.",
    ),

    "acquisition_status": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT",
        "Potential direct acquisition-outcome leakage.",
    ),

    "ipo_status": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT",
        "Potential direct IPO-outcome leakage.",
    ),

    "num_acquisitions": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT",
        "Current acquisition behavior may be post-origin.",
    ),

    "last_funding_at": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT_EVENT_SUMMARY",
        "Can occur after prediction origin.",
    ),

    "last_key_employee_change_date": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT_EVENT_SUMMARY",
        "May occur after prediction origin.",
    ),

    "last_layoff_date": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT_EVENT_SUMMARY",
        "May occur after prediction origin.",
    ),

    "went_public_on": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "OUTCOME_DATE",
        "Direct IPO-related outcome signal.",
    ),

    "exited_on": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "OUTCOME_DATE",
        "Direct exit-outcome signal.",
    ),

    "rank_org": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT_RANK",
        "Current Crunchbase rank is not historically reconstructed.",
    ),

    "rank_org_company": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT_RANK",
        "Current Crunchbase rank is not historically reconstructed.",
    ),

    "rank_delta_d7": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT_RANK",
        "Current short-horizon ranking signal.",
    ),

    "rank_delta_d30": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT_RANK",
        "Current short-horizon ranking signal.",
    ),

    "rank_delta_d90": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT_RANK",
        "Current short-horizon ranking signal.",
    ),

    "semrush_global_rank": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT_WEB_SIGNAL",
        "Current web prominence can reflect future growth.",
    ),

    "semrush_visits_latest_month": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT_WEB_SIGNAL",
        "Explicit current-period traffic measurement.",
    ),

    "semrush_visits_latest_6_months_avg": (
        "EDA_ONLY_HIGH_LEAKAGE_RISK",
        "CURRENT_SNAPSHOT_WEB_SIGNAL",
        "Explicit recent-period traffic measurement.",
    ),

    # -------------------------------------------------------------------------
    # Relation candidates
    # -------------------------------------------------------------------------

    "founder_links": (
        "RELATION_CANDIDATE",
        "SNAPSHOT_UNVERSIONED",
        "Company→Person candidate; audited separately for temporal eligibility.",
    ),

    "founders": (
        "RELATION_CANDIDATE_SECONDARY",
        "SNAPSHOT_UNVERSIONED",
        "Name representation; founder_links preferred.",
    ),

    "hub_tags": (
        "RELATION_CANDIDATE",
        "SNAPSHOT_UNVERSIONED",
        "Potential Company→Hub/tag relation; very sparse globally.",
    ),

    "num_founders": (
        "EDA_ONLY_STRUCTURAL_SUMMARY",
        "SNAPSHOT_UNVERSIONED",
        "Potentially redundant with explicit founder relations.",
    ),

    "num_contacts": (
        "EDA_ONLY_STRUCTURAL_SUMMARY",
        "SNAPSHOT_UNVERSIONED",
        "Operational count; unclear research semantics.",
    ),

    # -------------------------------------------------------------------------
    # Already frozen for eligibility, not predictive use.
    # -------------------------------------------------------------------------

    "founded_on": (
        "ELIGIBILITY_ONLY_FROZEN",
        "SEMANTIC_EVENT_DATE",
        "Already used for existence/relation eligibility; predictive use remains prohibited.",
    ),

    # -------------------------------------------------------------------------
    # Operational / identity fields.
    # -------------------------------------------------------------------------

    "website": (
        "IDENTITY_OPERATIONAL",
        "SNAPSHOT_UNVERSIONED",
        "Not proposed as a main predictive feature.",
    ),

    "linkedin": (
        "IDENTITY_OPERATIONAL",
        "SNAPSHOT_UNVERSIONED",
        "Not proposed as a main predictive feature.",
    ),

    "contact_job_departments": (
        "EDA_ONLY",
        "SNAPSHOT_UNVERSIONED",
        "Operational metadata; unclear startup-quality semantics.",
    ),
}


CATEGORICAL_DETAIL_FIELDS = [
    "country",
    "region",
    "company_type",
    "operating_status",
    "num_employees_enum",
    "revenue_range",
    "funding_stage",
    "last_funding_type",
    "acquisition_status",
    "ipo_status",
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


def normalize_string_series(
    series,
):
    s = (
        series
        .astype(
            "string"
        )
        .str.strip()
    )

    return s.mask(
        s.eq("")
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    separator()

    print(
        "PHASE 8.1.6B — "
        "STARTUP ATTRIBUTE DEEP EDA V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "STARTUP ATTRIBUTE / LEAKAGE-RISK EDA"
    )

    print(
        "Startup universe:            "
        "FROZEN PHASE-3 STARTUP ROLE"
    )

    print(
        "Development cohorts:         "
        "T57 / T58 / T59"
    )

    print(
        "April attribute EDA:         NO"
    )

    print(
        "Features selected:           NO"
    )

    print(
        "Input contract changed:      NO"
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
    # 1. Phase 8.1.6A gate
    # =========================================================================

    separator("-")

    print(
        "8.1.6B.1 — FOUNDATION EDA GATE"
    )

    separator("-")

    require(
        PHASE816A_SUMMARY.exists(),
        "Missing Phase-8.1.6A summary.",
    )

    foundation = read_json(
        PHASE816A_SUMMARY
    )

    require(
        foundation.get(
            "status"
        )
        ==
        "COMPLETE",
        "Phase 8.1.6A is not complete.",
    )

    print(
        "Phase 8.1.6A: COMPLETE PASS"
    )

    # =========================================================================
    # 2. Frozen Startup universe
    # =========================================================================

    separator("-")

    print(
        "8.1.6B.2 — FROZEN STARTUP UNIVERSE"
    )

    separator("-")

    nodes = pd.read_parquet(
        NODE_INDEX_PATH,
        columns=[
            "node_type",
            "raw_entity_id",
        ],
    )

    startup_nodes = (
        nodes[
            nodes[
                "node_type"
            ]
            ==
            "startup"
        ]
        .copy()
    )

    startup_nodes[
        "raw_entity_id"
    ] = (
        startup_nodes[
            "raw_entity_id"
        ].astype(str)
    )

    require(
        len(
            startup_nodes
        )
        ==
        EXPECTED_STARTUPS,
        "Frozen startup population drift.",
    )

    require(
        not startup_nodes[
            "raw_entity_id"
        ].duplicated().any(),
        "Duplicate Startup IDs in node index.",
    )

    startup_ids = set(
        startup_nodes[
            "raw_entity_id"
        ]
    )

    print(
        f"Frozen Startup IDs: "
        f"{len(startup_ids):,} PASS"
    )

    # =========================================================================
    # 3. Resolve Company raw schema
    # =========================================================================

    separator("-")

    print(
        "8.1.6B.3 — COMPANY SOURCE RESOLUTION"
    )

    separator("-")

    company_files = sorted(
        RAW_DIR.glob(
            "companies*.csv"
        )
    )

    require(
        len(
            company_files
        )
        ==
        24,
        "Expected exactly 24 company CSV files.",
    )

    common_columns = None

    for path in (
        company_files
    ):

        header = pd.read_csv(
            path,
            nrows=0,
        )

        columns = set(
            header.columns
        )

        if common_columns is None:

            common_columns = columns

        else:

            common_columns = (
                common_columns
                &
                columns
            )

    available_fields = [
        field
        for field
        in REQUESTED_FIELDS
        if field
        in common_columns
    ]

    missing_requested = [
        field
        for field
        in REQUESTED_FIELDS
        if field
        not in common_columns
    ]

    require(
        "id"
        in
        available_fields,
        "Company id field unavailable.",
    )

    print(
        f"Requested structured fields: "
        f"{len(REQUESTED_FIELDS):,}"
    )

    print(
        f"Available in all 24 files:   "
        f"{len(available_fields):,}"
    )

    print(
        f"Unavailable requested fields:"
        f" {len(missing_requested):,}"
    )

    if missing_requested:

        for field in (
            missing_requested
        ):

            print(
                f"  MISSING {field}"
            )

    # =========================================================================
    # 4. Extract only frozen Startup entities
    # =========================================================================

    separator("-")

    print(
        "8.1.6B.4 — EXTRACT FROZEN "
        "STARTUP COMPANY ROWS"
    )

    separator("-")

    matched_parts = []

    raw_company_rows = 0

    for i, path in enumerate(
        company_files,
        start=1,
    ):

        file_matched = 0

        for chunk in pd.read_csv(
            path,
            usecols=available_fields,
            dtype="string",
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):

            raw_company_rows += len(
                chunk
            )

            chunk[
                "id"
            ] = (
                normalize_string_series(
                    chunk[
                        "id"
                    ]
                )
            )

            matched = chunk[
                chunk[
                    "id"
                ].isin(
                    startup_ids
                )
            ].copy()

            if not matched.empty:

                matched_parts.append(
                    matched
                )

                file_matched += len(
                    matched
                )

        print(
            f"[{i:02d}/24] "
            f"{path.name:<24} "
            f"startup rows={file_matched:,}"
        )

    raw_startup_rows = pd.concat(
        matched_parts,
        ignore_index=True,
    )

    recovered_ids = set(
        raw_startup_rows[
            "id"
        ].dropna()
        .astype(str)
    )

    require(
        recovered_ids
        ==
        startup_ids,
        (
            "Raw Company exports do not recover "
            "the exact frozen Startup universe."
        ),
    )

    row_counts = (
        raw_startup_rows[
            "id"
        ]
        .value_counts()
    )

    duplicate_ids = (
        row_counts[
            row_counts
            >
            1
        ]
    )

    duplicate_id_count = len(
        duplicate_ids
    )

    duplicate_extra_rows = int(
        (
            duplicate_ids
            -
            1
        ).sum()
    )

    print()

    print(
        f"All raw Company rows scanned: "
        f"{raw_company_rows:,}"
    )

    print(
        f"Matched Startup rows:         "
        f"{len(raw_startup_rows):,}"
    )

    print(
        f"Unique Startup IDs:           "
        f"{len(recovered_ids):,}"
    )

    print(
        f"Startup IDs with >1 row:      "
        f"{duplicate_id_count:,}"
    )

    print(
        f"Extra duplicate rows:         "
        f"{duplicate_extra_rows:,}"
    )

    # =========================================================================
    # 5. Attribute coverage and duplicate conflicts
    # =========================================================================

    separator("-")

    print(
        "8.1.6B.5 — ATTRIBUTE COVERAGE "
        "& DUPLICATE CONSISTENCY"
    )

    separator("-")

    audited_fields = [
        field
        for field
        in available_fields
        if field != "id"
    ]

    coverage_rows = []
    conflict_rows = []

    # Availability map:
    # field -> startup_id-indexed boolean
    availability_maps = {}

    # Canonical values are used for EDA only when all non-null duplicate
    # rows agree for the ID.
    canonical_maps = {}

    any_conflict_ids = set()

    for field in (
        audited_fields
    ):

        tmp = raw_startup_rows[
            [
                "id",
                field,
            ]
        ].copy()

        tmp[
            field
        ] = normalize_string_series(
            tmp[
                field
            ]
        )

        nunique = (
            tmp
            .groupby(
                "id",
                sort=False,
            )[
                field
            ]
            .nunique(
                dropna=True
            )
        )

        # All 311,589 IDs must be represented.
        nunique = nunique.reindex(
            startup_nodes[
                "raw_entity_id"
            ],
            fill_value=0,
        )

        populated = (
            nunique
            >
            0
        )

        conflict = (
            nunique
            >
            1
        )

        populated_count = int(
            populated.sum()
        )

        conflict_count = int(
            conflict.sum()
        )

        availability_maps[
            field
        ] = populated

        field_conflict_ids = set(
            nunique[
                conflict
            ].index
        )

        any_conflict_ids.update(
            field_conflict_ids
        )

        # First non-null value; retained only for IDs with exactly one
        # distinct non-null value across duplicate rows.
        non_null_tmp = (
            tmp.dropna(
                subset=[
                    field
                ]
            )
        )

        first_values = (
            non_null_tmp
            .groupby(
                "id",
                sort=False,
            )[
                field
            ]
            .first()
        )

        unique_ids = (
            nunique[
                nunique
                ==
                1
            ].index
        )

        canonical = (
            first_values.reindex(
                unique_ids
            )
        )

        canonical_maps[
            field
        ] = canonical

        risk = RISK_CLASSIFICATION.get(
            field,
            (
                "EDA_ONLY_UNCLASSIFIED",
                "UNKNOWN",
                "No frozen scientific classification yet.",
            ),
        )

        coverage_rows.append(
            {
                "field":
                    field,

                "startup_population":
                    EXPECTED_STARTUPS,

                "populated_startups":
                    populated_count,

                "coverage_pct":
                    pct(
                        populated_count,
                        EXPECTED_STARTUPS,
                    ),

                "conflicting_startup_ids":
                    conflict_count,

                "conflict_pct_all_startups":
                    pct(
                        conflict_count,
                        EXPECTED_STARTUPS,
                    ),

                "conflict_pct_populated":
                    pct(
                        conflict_count,
                        populated_count,
                    ),

                "scientific_role":
                    risk[
                        0
                    ],

                "temporal_class":
                    risk[
                        1
                    ],
            }
        )

        conflict_rows.append(
            {
                "field":
                    field,

                "populated_startups":
                    populated_count,

                "conflicting_startup_ids":
                    conflict_count,

                "conflict_pct_populated":
                    pct(
                        conflict_count,
                        populated_count,
                    ),
            }
        )

    coverage_df = (
        pd.DataFrame(
            coverage_rows
        )
        .sort_values(
            "coverage_pct",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    conflict_df = (
        pd.DataFrame(
            conflict_rows
        )
        .sort_values(
            [
                "conflicting_startup_ids",
                "field",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    print()

    print(
        "TOP STRUCTURED ATTRIBUTE COVERAGE"
    )

    print(
        coverage_df[
            [
                "field",
                "coverage_pct",
                "scientific_role",
                "temporal_class",
            ]
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )

    print()

    print(
        "TOP DUPLICATE-VALUE CONFLICTS"
    )

    print(
        conflict_df[
            [
                "field",
                "conflicting_startup_ids",
                "conflict_pct_populated",
            ]
        ]
        .head(
            15
        )
        .to_string(
            index=False
        )
    )

    # =========================================================================
    # 6. Development primary cold-start coverage
    # =========================================================================

    separator("-")

    print(
        "8.1.6B.6 — DEVELOPMENT PRIMARY "
        "COLD-START ATTRIBUTE COVERAGE"
    )

    separator("-")

    targets = pd.read_parquet(
        TARGETS_PATH,
        columns=[
            "origin",
            "startup_id_str",
            "is_primary_stratum",
        ],
    )

    targets[
        "startup_id_str"
    ] = (
        targets[
            "startup_id_str"
        ].astype(str)
    )

    primary = targets[
        targets[
            "origin"
        ].isin(
            DEVELOPMENT_ORIGINS
        )
        &
        targets[
            "is_primary_stratum"
        ]
    ].copy()

    for origin, expected in (
        EXPECTED_PRIMARY.items()
    ):

        n = int(
            (
                primary[
                    "origin"
                ]
                ==
                origin
            ).sum()
        )

        require(
            n
            ==
            expected,
            (
                f"{origin}: primary population drift."
            ),
        )

    primary_coverage_rows = []

    for origin in (
        DEVELOPMENT_ORIGINS
    ):

        origin_cases = primary[
            primary[
                "origin"
            ]
            ==
            origin
        ]

        startup_case_ids = (
            origin_cases[
                "startup_id_str"
            ]
            .tolist()
        )

        for field in (
            audited_fields
        ):

            availability = (
                availability_maps[
                    field
                ]
            )

            flags = availability.reindex(
                startup_case_ids,
                fill_value=False,
            )

            available_n = int(
                flags.sum()
            )

            primary_coverage_rows.append(
                {
                    "origin":
                        origin,

                    "field":
                        field,

                    "primary_cases":
                        len(
                            startup_case_ids
                        ),

                    "available_cases":
                        available_n,

                    "coverage_pct":
                        pct(
                            available_n,
                            len(
                                startup_case_ids
                            ),
                        ),
                }
            )

    primary_coverage_df = pd.DataFrame(
        primary_coverage_rows
    )

    mean_primary_coverage = (
        primary_coverage_df
        .groupby(
            "field",
            as_index=False,
        )[
            "coverage_pct"
        ]
        .mean()
        .rename(
            columns={
                "coverage_pct":
                    "mean_primary_dev_coverage_pct"
            }
        )
    )

    coverage_df = coverage_df.merge(
        mean_primary_coverage,
        on="field",
        how="left",
        validate="one_to_one",
    )

    print()

    print(
        coverage_df[
            [
                "field",
                "coverage_pct",
                "mean_primary_dev_coverage_pct",
                "scientific_role",
            ]
        ]
        .sort_values(
            "mean_primary_dev_coverage_pct",
            ascending=False,
        )
        .head(
            25
        )
        .to_string(
            index=False
        )
    )

    # =========================================================================
    # 7. Temporal-risk register
    # =========================================================================

    separator("-")

    print(
        "8.1.6B.7 — TEMPORAL / LEAKAGE "
        "RISK REGISTER"
    )

    separator("-")

    risk_rows = []

    for field in (
        audited_fields
    ):

        role, temporal_class, rationale = (
            RISK_CLASSIFICATION.get(
                field,
                (
                    "EDA_ONLY_UNCLASSIFIED",
                    "UNKNOWN",
                    (
                        "No frozen classification; "
                        "cannot enter model without "
                        "explicit review."
                    ),
                ),
            )
        )

        risk_rows.append(
            {
                "field":
                    field,

                "scientific_role":
                    role,

                "temporal_class":
                    temporal_class,

                "main_model_currently_allowed":
                    False,

                "requires_contract_amendment":
                    True,

                "rationale":
                    rationale,
            }
        )

    risk_df = pd.DataFrame(
        risk_rows
    )

    print(
        risk_df[
            [
                "field",
                "scientific_role",
                "temporal_class",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 8. Categorical EDA
    # =========================================================================

    separator("-")

    print(
        "8.1.6B.8 — CATEGORICAL "
        "STARTUP DISTRIBUTIONS"
    )

    separator("-")

    top_value_rows = []

    dev_primary_unique_ids = set(
        primary[
            "startup_id_str"
        ]
    )

    for field in (
        CATEGORICAL_DETAIL_FIELDS
    ):

        if field not in (
            canonical_maps
        ):
            continue

        canonical = (
            canonical_maps[
                field
            ]
        )

        # ---------------------------------------------------------------------
        # Global Startup-role population
        # ---------------------------------------------------------------------

        global_counts = (
            canonical.value_counts()
        )

        global_total = int(
            global_counts.sum()
        )

        for rank, (
            value,
            count,
        ) in enumerate(
            global_counts.head(
                TOP_K
            ).items(),
            start=1,
        ):

            top_value_rows.append(
                {
                    "population":
                        "GLOBAL_STARTUP_ROLE",

                    "field":
                        field,

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
                            global_total,
                        ),
                }
            )

        # ---------------------------------------------------------------------
        # Unique development primary startups
        # ---------------------------------------------------------------------

        dev_values = canonical.reindex(
            list(
                dev_primary_unique_ids
            )
        ).dropna()

        dev_counts = (
            dev_values.value_counts()
        )

        dev_total = int(
            dev_counts.sum()
        )

        for rank, (
            value,
            count,
        ) in enumerate(
            dev_counts.head(
                TOP_K
            ).items(),
            start=1,
        ):

            top_value_rows.append(
                {
                    "population":
                        "DEV_PRIMARY_UNIQUE_STARTUPS",

                    "field":
                        field,

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
                            dev_total,
                        ),
                }
            )

    top_values_df = pd.DataFrame(
        top_value_rows
    )

    # =========================================================================
    # 9. Persist tables
    # =========================================================================

    separator("-")

    print(
        "8.1.6B.9 — PERSIST EDA TABLES"
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

    coverage_df.to_csv(
        COVERAGE_OUTPUT,
        index=False,
    )

    primary_coverage_df.to_csv(
        PRIMARY_COVERAGE_OUTPUT,
        index=False,
    )

    duplicate_summary_df = pd.DataFrame(
        [
            {
                "frozen_startup_ids":
                    EXPECTED_STARTUPS,

                "matched_raw_company_rows":
                    len(
                        raw_startup_rows
                    ),

                "startup_ids_with_multiple_rows":
                    duplicate_id_count,

                "extra_duplicate_rows":
                    duplicate_extra_rows,

                "startup_ids_with_any_audited_field_conflict":
                    len(
                        any_conflict_ids
                    ),

                "any_conflict_pct":
                    pct(
                        len(
                            any_conflict_ids
                        ),
                        EXPECTED_STARTUPS,
                    ),
            }
        ]
    )

    duplicate_summary_df.to_csv(
        DUPLICATE_SUMMARY_OUTPUT,
        index=False,
    )

    conflict_df.to_csv(
        CONFLICT_OUTPUT,
        index=False,
    )

    risk_df.to_csv(
        RISK_OUTPUT,
        index=False,
    )

    top_values_df.to_csv(
        TOP_VALUES_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 10. Charts
    # =========================================================================

    separator("-")

    print(
        "8.1.6B.10 — STARTUP FOUNDATION CHARTS"
    )

    separator("-")

    # -------------------------------------------------------------------------
    # Chart 1 — Global structured attribute coverage
    # -------------------------------------------------------------------------

    chart_coverage = (
        coverage_df
        .sort_values(
            "coverage_pct",
            ascending=True,
        )
    )

    fig, ax = plt.subplots(
        figsize=(11, 10)
    )

    ax.barh(
        chart_coverage[
            "field"
        ],
        chart_coverage[
            "coverage_pct"
        ],
    )

    ax.set_xlim(
        0,
        100,
    )

    ax.set_xlabel(
        "Startup coverage (%)"
    )

    ax.set_title(
        "Frozen Startup universe: structured attribute coverage"
    )

    fig.tight_layout()

    coverage_chart = (
        CHART_DIR
        / "phase_8_1_6b_startup_attribute_coverage_V1.png"
    )

    fig.savefig(
        coverage_chart,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # -------------------------------------------------------------------------
    # Chart 2 — Global vs development cold-start coverage
    # -------------------------------------------------------------------------

    compare = (
        coverage_df
        .sort_values(
            "mean_primary_dev_coverage_pct",
            ascending=False,
        )
        .head(
            25
        )
        .sort_values(
            "mean_primary_dev_coverage_pct",
            ascending=True,
        )
    )

    y = np.arange(
        len(
            compare
        )
    )

    height = 0.38

    fig, ax = plt.subplots(
        figsize=(11, 9)
    )

    ax.barh(
        y - height / 2,
        compare[
            "coverage_pct"
        ],
        height=height,
        label="All Startup nodes",
    )

    ax.barh(
        y + height / 2,
        compare[
            "mean_primary_dev_coverage_pct"
        ],
        height=height,
        label="Primary cold-start (T57-T59 mean)",
    )

    ax.set_yticks(
        y
    )

    ax.set_yticklabels(
        compare[
            "field"
        ]
    )

    ax.set_xlim(
        0,
        100,
    )

    ax.set_xlabel(
        "Coverage (%)"
    )

    ax.set_title(
        "Structured attribute coverage: global vs development cold-start"
    )

    ax.legend()

    fig.tight_layout()

    compare_chart = (
        CHART_DIR
        / "phase_8_1_6b_global_vs_primary_coverage_V1.png"
    )

    fig.savefig(
        compare_chart,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # -------------------------------------------------------------------------
    # Chart 3 — Duplicate field conflicts
    # -------------------------------------------------------------------------

    conflict_chart_df = (
        conflict_df[
            conflict_df[
                "conflicting_startup_ids"
            ]
            >
            0
        ]
        .head(
            15
        )
        .sort_values(
            "conflicting_startup_ids",
            ascending=True,
        )
    )

    conflict_chart = None

    if not conflict_chart_df.empty:

        fig, ax = plt.subplots(
            figsize=(10, 7)
        )

        ax.barh(
            conflict_chart_df[
                "field"
            ],
            conflict_chart_df[
                "conflicting_startup_ids"
            ],
        )

        ax.set_xlabel(
            "Startup IDs with conflicting duplicate values"
        )

        ax.set_title(
            "Duplicate Company-ID attribute conflicts"
        )

        fig.tight_layout()

        conflict_chart = (
            CHART_DIR
            / "phase_8_1_6b_duplicate_field_conflicts_V1.png"
        )

        fig.savefig(
            conflict_chart,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    # -------------------------------------------------------------------------
    # Chart 4 — Development-primary geography
    # -------------------------------------------------------------------------

    geography_chart = None

    country_dev = top_values_df[
        (
            top_values_df[
                "population"
            ]
            ==
            "DEV_PRIMARY_UNIQUE_STARTUPS"
        )
        &
        (
            top_values_df[
                "field"
            ]
            ==
            "country"
        )
    ].copy()

    if not country_dev.empty:

        country_dev = (
            country_dev
            .head(
                15
            )
            .sort_values(
                "count",
                ascending=True,
            )
        )

        fig, ax = plt.subplots(
            figsize=(9, 7)
        )

        ax.barh(
            country_dev[
                "value"
            ],
            country_dev[
                "count"
            ],
        )

        ax.set_xlabel(
            "Unique development-primary startups"
        )

        ax.set_title(
            "Primary cold-start startups by current snapshot country"
        )

        fig.tight_layout()

        geography_chart = (
            CHART_DIR
            / "phase_8_1_6b_primary_country_distribution_V1.png"
        )

        fig.savefig(
            geography_chart,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    print(
        f"WROTE {coverage_chart}"
    )

    print(
        f"WROTE {compare_chart}"
    )

    if conflict_chart is not None:

        print(
            f"WROTE {conflict_chart}"
        )

    if geography_chart is not None:

        print(
            f"WROTE {geography_chart}"
        )

    # =========================================================================
    # 11. Summary
    # =========================================================================

    separator("-")

    print(
        "8.1.6B.11 — STARTUP DATA INSIGHTS"
    )

    separator("-")

    high_leakage_fields = risk_df[
        risk_df[
            "scientific_role"
        ]
        ==
        "EDA_ONLY_HIGH_LEAKAGE_RISK"
    ][
        "field"
    ].tolist()

    potential_extensions = risk_df[
        risk_df[
            "scientific_role"
        ]
        ==
        "POTENTIAL_SNAPSHOT_EXTENSION"
    ][
        "field"
    ].tolist()

    relation_candidates = risk_df[
        risk_df[
            "scientific_role"
        ].str.startswith(
            "RELATION_CANDIDATE"
        )
    ][
        "field"
    ].tolist()

    print()

    print(
        f"Startup IDs with duplicate rows: "
        f"{duplicate_id_count:,}"
    )

    print(
        f"Startup IDs with any audited "
        f"value conflict: "
        f"{len(any_conflict_ids):,}"
    )

    print()

    print(
        "Potential matched-snapshot extensions:"
    )

    for field in (
        potential_extensions
    ):

        row = coverage_df[
            coverage_df[
                "field"
            ]
            ==
            field
        ]

        if len(
            row
        ):

            row = row.iloc[0]

            print(
                f"  {field:<24} "
                f"global={row['coverage_pct']:.2f}% "
                f"primary={row['mean_primary_dev_coverage_pct']:.2f}%"
            )

    print()

    print(
        "Explicit relation candidates:"
    )

    for field in (
        relation_candidates
    ):

        print(
            f"  {field}"
        )

    print()

    print(
        f"High leakage-risk fields identified: "
        f"{len(high_leakage_fields):,}"
    )

    summary = {
        "schema_version":
            "PHASE_8_1_6B_STARTUP_ATTRIBUTE_DEEP_EDA_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            (
                "STARTUP_UNIVERSE_ATTRIBUTE_"
                "QUALITY_COVERAGE_LEAKAGE_EDA"
            ),

        "frozen_startup_population":
            EXPECTED_STARTUPS,

        "raw_company_rows_scanned":
            raw_company_rows,

        "matched_startup_rows":
            len(
                raw_startup_rows
            ),

        "startup_ids_with_duplicate_rows":
            duplicate_id_count,

        "extra_duplicate_rows":
            duplicate_extra_rows,

        "startup_ids_with_any_attribute_conflict":
            len(
                any_conflict_ids
            ),

        "audited_structured_fields":
            len(
                audited_fields
            ),

        "potential_snapshot_extensions":
            potential_extensions,

        "relation_candidates":
            relation_candidates,

        "high_leakage_risk_fields":
            high_leakage_fields,

        "April_attribute_distributions_analyzed":
            False,

        "features_selected":
            False,

        "input_contract_changed":
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
                "Phase 8.1.6C — deep People / founder / "
                "School EDA and explicit relation-resolution audit."
            ),
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    output_paths = [
        COVERAGE_OUTPUT,
        PRIMARY_COVERAGE_OUTPUT,
        DUPLICATE_SUMMARY_OUTPUT,
        CONFLICT_OUTPUT,
        RISK_OUTPUT,
        TOP_VALUES_OUTPUT,
        SUMMARY_OUTPUT,
        coverage_chart,
        compare_chart,
    ]

    if conflict_chart is not None:
        output_paths.append(
            conflict_chart
        )

    if geography_chart is not None:
        output_paths.append(
            geography_chart
        )

    manifest = {
        "phase":
            "8.1.6B",

        "status":
            "COMPLETE",

        "development_origins":
            DEVELOPMENT_ORIGINS,

        "April_attribute_distributions_analyzed":
            False,

        "outputs": [
            str(
                path
            )
            for path
            in output_paths
        ],

        "prohibited_operations": [
            "feature_selection",
            "input_contract_change",
            "graph_modification",
            "architecture_selection",
            "training",
            "inference",
            "ranking_scoring",
            "April_feature_distribution_analysis",
            "April_performance_inspection",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    output_paths.append(
        MANIFEST_OUTPUT
    )

    hashes = {
        str(
            path
        ):
            sha256_file(
                path
            )
        for path
        in output_paths
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
        "PHASE 8.1.6B STARTUP ATTRIBUTE "
        "DEEP EDA COMPLETE"
    )

    separator()

    print()

    print(
        "FROZEN STARTUP POPULATION:        "
        f"{EXPECTED_STARTUPS:,}"
    )

    print(
        "DUPLICATE CONSISTENCY AUDITED:    YES"
    )

    print(
        "STRUCTURED COVERAGE AUDITED:      YES"
    )

    print(
        "COLD-START COVERAGE AUDITED:      YES"
    )

    print(
        "TEMPORAL LEAKAGE RISKS LABELED:  YES"
    )

    print(
        "EDA CHARTS CREATED:               YES"
    )

    print()

    print(
        "FEATURES SELECTED:                NO"
    )

    print(
        "INPUT CONTRACT CHANGED:           NO"
    )

    print(
        "GRAPH MODIFIED:                   NO"
    )

    print(
        "ARCHITECTURE SELECTED:            NO"
    )

    print(
        "MODEL TRAINED:                    NO"
    )

    print(
        "APRIL ATTRIBUTE EDA:              NO"
    )

    print(
        "APRIL PERFORMANCE INSPECTED:      NO"
    )

    print()

    print(
        "Next:"
    )

    print(
        "Phase 8.1.6C — deep People / founder / "
        "School EDA and explicit relation resolution."
    )

    print()

    print(
        f"Outputs written to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()