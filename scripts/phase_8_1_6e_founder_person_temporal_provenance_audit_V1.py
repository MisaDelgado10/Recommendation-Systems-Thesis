from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re

import pandas as pd


# =============================================================================
# PHASE 8.1.6E — FOUNDER-PERSON TEMPORAL PROVENANCE AUDIT V1
# =============================================================================
#
# Scientific role:
#
#   DEVELOPMENT-ONLY TEMPORAL-PROVENANCE AUDIT
#
# PURPOSE
# -------
#
# Determine whether explicit Startup -> Founder Person context can be assigned
# a defensible semantic effective date using startup.founded_on.
#
# IMPORTANT DISTINCTION
# ---------------------
#
#   semantic eligibility != database observation timestamp
#
# Person attributes and Person->School remain snapshot-unversioned unless
# explicit temporal evidence exists.
#
# No graph modification.
# No relation approval.
# No training.
# April untouched.
#
# =============================================================================


RAW_DIR = Path("data/raw")

TARGETS_PATH = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_9_origin_eligibility_candidate_universe/"
    "phase_8_0_9_origin_eligible_targets_V5.parquet"
)

PHASE816D_SUMMARY = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_6_data_foundations/"
    "phase_8_1_6d_explicit_founder_incremental_coverage/"
    "phase_8_1_6d_summary_V2.json"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_6_data_foundations/"
    "phase_8_1_6e_founder_person_temporal_provenance"
)

CASE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6e_case_level_founder_temporal_coverage_V1.csv"
)

PROVENANCE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6e_person_information_provenance_register_V1.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6e_summary_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6e_analysis_manifest_V1.json"
)


DEVELOPMENT_ORIGINS = {
    "T57": pd.Timestamp("2025-03-31"),
    "T58": pd.Timestamp("2025-06-30"),
    "T59": pd.Timestamp("2025-09-30"),
}

EXPECTED_PRIMARY = {
    "T57": 6_850,
    "T58": 6_872,
    "T59": 6_764,
}

EXPECTED_RAW_PERSON_CASES = {
    "T57": 3_379,
    "T58": 3_271,
    "T59": 2_817,
}

CHUNK_SIZE = 100_000


PERSON_URL_RE = re.compile(
    r"https?://(?:www\.)?crunchbase\.com/person/[^\s,;|]+",
    flags=re.IGNORECASE,
)


def separator(char="=", width=118):
    print(char * width)


def require(condition, message):

    if not condition:
        raise ValueError(message)


def pct(num, den):

    if den == 0:
        return float("nan")

    return (
        100.0
        * float(num)
        / float(den)
    )


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


def normalize_url(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

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

    urls = []

    for url in PERSON_URL_RE.findall(
        str(value)
    ):

        url = normalize_url(url)

        if url is not None:
            urls.append(url)

    return list(
        dict.fromkeys(urls)
    )


def main():

    separator()

    print(
        "PHASE 8.1.6E — "
        "FOUNDER-PERSON TEMPORAL PROVENANCE AUDIT V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "DEVELOPMENT-ONLY TEMPORAL-PROVENANCE AUDIT"
    )

    print("April analyzed:              NO")
    print("Relation approved:           NO")
    print("Graph modified:              NO")
    print("Architecture selected:       NO")
    print("Model trained:               NO")
    print("April performance inspected: NO")

    # =========================================================================
    # 1. Gate
    # =========================================================================

    separator("-")

    print(
        "8.1.6E.1 — PREREQUISITE GATE"
    )

    separator("-")

    require(
        PHASE816D_SUMMARY.exists(),
        "Missing Phase-8.1.6D V2 summary.",
    )

    previous = read_json(
        PHASE816D_SUMMARY
    )

    require(
        previous.get("status")
        ==
        "COMPLETE",
        "Phase 8.1.6D V2 not complete.",
    )

    require(
        previous.get(
            "April_analyzed"
        )
        is False,
        "April unexpectedly analyzed.",
    )

    print(
        "Phase 8.1.6D V2: COMPLETE PASS"
    )

    # =========================================================================
    # 2. Development-primary cases
    # =========================================================================

    separator("-")

    print(
        "8.1.6E.2 — DEVELOPMENT PRIMARY CASES"
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
            DEVELOPMENT_ORIGINS.keys()
        )
        &
        targets[
            "is_primary_stratum"
        ]
    ].copy()

    primary_ids = set(
        primary[
            "startup_id_str"
        ]
    )

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
            n == expected,
            f"{origin}: primary count drift.",
        )

        print(
            f"{origin}: {n:,} PASS"
        )

    # =========================================================================
    # 3. Startup founded_on + founder URLs
    # =========================================================================

    separator("-")

    print(
        "8.1.6E.3 — STARTUP FOUNDING DATE "
        "& FOUNDER RELATION"
    )

    separator("-")

    company_files = sorted(
        RAW_DIR.glob(
            "companies*.csv"
        )
    )

    startup_founded_values = defaultdict(
        set
    )

    startup_founder_urls = defaultdict(
        set
    )

    for i, path in enumerate(
        company_files,
        start=1,
    ):

        matched = 0

        for chunk in pd.read_csv(
            path,
            usecols=[
                "id",
                "founded_on",
                "founder_links",
            ],
            dtype="string",
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):

            chunk[
                "id"
            ] = (
                chunk[
                    "id"
                ]
                .astype("string")
                .str.strip()
            )

            subset = chunk[
                chunk[
                    "id"
                ].isin(
                    primary_ids
                )
            ]

            matched += len(
                subset
            )

            for row in subset.itertuples(
                index=False
            ):

                startup_id = str(
                    row.id
                )

                if pd.notna(
                    row.founded_on
                ):

                    value = str(
                        row.founded_on
                    ).strip()

                    if value:

                        startup_founded_values[
                            startup_id
                        ].add(
                            value
                        )

                for url in extract_person_urls(
                    row.founder_links
                ):

                    startup_founder_urls[
                        startup_id
                    ].add(
                        url
                    )

        print(
            f"[{i:02d}/24] "
            f"{path.name:<24} "
            f"primary rows={matched:,}"
        )

    # Duplicate Company rows must not disagree on founded_on.
    conflicting_founded = {
        startup_id:
            values
        for startup_id, values
        in startup_founded_values.items()
        if len(values) > 1
    }

    require(
        not conflicting_founded,
        (
            "Primary startup founded_on conflicts "
            "detected across duplicate Company rows."
        ),
    )

    startup_founded_on = {}

    for startup_id, values in (
        startup_founded_values.items()
    ):

        if not values:
            continue

        raw_value = next(
            iter(values)
        )

        parsed = pd.to_datetime(
            raw_value,
            errors="coerce",
        )

        if pd.notna(parsed):

            startup_founded_on[
                startup_id
            ] = parsed

    # =========================================================================
    # 4. Resolve founder URLs to People
    # =========================================================================

    separator("-")

    print(
        "8.1.6E.4 — EXACT PERSON RESOLUTION"
    )

    separator("-")

    founder_urls_needed = set()

    for urls in (
        startup_founder_urls.values()
    ):

        founder_urls_needed.update(
            urls
        )

    people_files = sorted(
        RAW_DIR.glob(
            "people*.csv"
        )
    )

    url_to_ids = defaultdict(
        set
    )

    person_snapshot_fields = {
        "schools",
        "current_organizations",
        "description",
        "country",
        "region",
        "locations",
    }

    person_values = defaultdict(
        lambda:
            defaultdict(
                set
            )
    )

    for i, path in enumerate(
        people_files,
        start=1,
    ):

        header = pd.read_csv(
            path,
            nrows=0,
        )

        available_fields = [
            field
            for field
            in [
                "id",
                "link",
                *sorted(
                    person_snapshot_fields
                ),
            ]
            if field
            in
            header.columns
        ]

        matched = 0

        for chunk in pd.read_csv(
            path,
            usecols=available_fields,
            dtype="string",
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):

            normalized_links = (
                chunk[
                    "link"
                ]
                .map(
                    normalize_url
                )
            )

            relevant = normalized_links.isin(
                founder_urls_needed
            )

            if not relevant.any():
                continue

            subset = chunk.loc[
                relevant
            ].copy()

            subset[
                "_normalized_link"
            ] = (
                normalized_links[
                    relevant
                ]
                .to_numpy()
            )

            matched += len(
                subset
            )

            for row in subset.itertuples(
                index=False
            ):

                person_id = str(
                    row.id
                ).strip()

                url = str(
                    row._normalized_link
                )

                url_to_ids[
                    url
                ].add(
                    person_id
                )

                for field in (
                    person_snapshot_fields
                ):

                    if not hasattr(
                        row,
                        field
                    ):
                        continue

                    value = getattr(
                        row,
                        field
                    )

                    if pd.isna(
                        value
                    ):
                        continue

                    value = str(
                        value
                    ).strip()

                    if value:

                        person_values[
                            person_id
                        ][
                            field
                        ].add(
                            value
                        )

        print(
            f"[{i:02d}/11] "
            f"{path.name:<20} "
            f"matched={matched:,}"
        )

    url_to_person = {
        url:
            next(
                iter(ids)
            )
        for url, ids
        in url_to_ids.items()
        if len(ids) == 1
    }

    startup_people = defaultdict(
        set
    )

    for startup_id, urls in (
        startup_founder_urls.items()
    ):

        for url in urls:

            person_id = (
                url_to_person.get(
                    url
                )
            )

            if person_id is not None:

                startup_people[
                    startup_id
                ].add(
                    person_id
                )

    # =========================================================================
    # 5. Case-level temporal eligibility
    # =========================================================================

    separator("-")

    print(
        "8.1.6E.5 — CASE-LEVEL SEMANTIC "
        "TEMPORAL ELIGIBILITY"
    )

    separator("-")

    rows = []

    for origin, origin_date in (
        DEVELOPMENT_ORIGINS.items()
    ):

        cases = primary[
            primary[
                "origin"
            ]
            ==
            origin
        ]

        ids = cases[
            "startup_id_str"
        ].tolist()

        raw_person = 0
        founded_parseable = 0
        semantic_eligible_person = 0
        person_with_school = 0
        person_with_description = 0
        person_with_country = 0
        person_with_current_org = 0

        for startup_id in ids:

            people = startup_people.get(
                startup_id,
                set(),
            )

            if not people:
                continue

            raw_person += 1

            founded_on = (
                startup_founded_on.get(
                    startup_id
                )
            )

            if founded_on is None:
                continue

            founded_parseable += 1

            if founded_on > origin_date:
                continue

            semantic_eligible_person += 1

            if any(
                person_values[p]["schools"]
                for p in people
            ):
                person_with_school += 1

            if any(
                person_values[p]["description"]
                for p in people
            ):
                person_with_description += 1

            if any(
                person_values[p]["country"]
                for p in people
            ):
                person_with_country += 1

            if any(
                person_values[p][
                    "current_organizations"
                ]
                for p in people
            ):
                person_with_current_org += 1

        require(
            raw_person
            ==
            EXPECTED_RAW_PERSON_CASES[
                origin
            ],
            (
                f"{origin}: raw Person case coverage "
                "does not reconcile with 8.1.6C."
            ),
        )

        rows.append(
            {
                "origin":
                    origin,

                "primary_cases":
                    len(ids),

                "raw_founder_person_cases":
                    raw_person,

                "raw_founder_person_pct":
                    pct(
                        raw_person,
                        len(ids),
                    ),

                "founder_person_with_parseable_founded_on":
                    founded_parseable,

                "semantic_origin_eligible_founder_person":
                    semantic_eligible_person,

                "semantic_origin_eligible_pct":
                    pct(
                        semantic_eligible_person,
                        len(ids),
                    ),

                "semantic_retention_pct_of_raw_person":
                    pct(
                        semantic_eligible_person,
                        raw_person,
                    ),

                "eligible_person_with_school_snapshot":
                    person_with_school,

                "eligible_person_with_description_snapshot":
                    person_with_description,

                "eligible_person_with_country_snapshot":
                    person_with_country,

                "eligible_person_with_current_org_snapshot":
                    person_with_current_org,
            }
        )

        print()

        print(
            f"{origin}"
        )

        print(
            f"  primary cases:                   "
            f"{len(ids):,}"
        )

        print(
            f"  raw founder Person:              "
            f"{raw_person:,} "
            f"({pct(raw_person, len(ids)):.4f}%)"
        )

        print(
            f"  with parseable founded_on:       "
            f"{founded_parseable:,}"
        )

        print(
            f"  semantic eligible founder edge:  "
            f"{semantic_eligible_person:,} "
            f"({pct(semantic_eligible_person, len(ids)):.4f}%)"
        )

        print(
            f"  retention of raw Person context: "
            f"{pct(semantic_eligible_person, raw_person):.4f}%"
        )

    case_df = pd.DataFrame(
        rows
    )

    # =========================================================================
    # 6. Information provenance register
    # =========================================================================

    separator("-")

    print(
        "8.1.6E.6 — PERSON INFORMATION "
        "PROVENANCE REGISTER"
    )

    separator("-")

    provenance_df = pd.DataFrame(
        [
            {
                "information":
                    "startup_to_founder_person",

                "temporal_basis":
                    "startup.founded_on",

                "origin_gate_available":
                    True,

                "approved_claim":
                    (
                        "semantically origin-eligible "
                        "founder relation"
                    ),

                "strict_database_observation_claim":
                    False,

                "risk":
                    "DATABASE_OBSERVATION_TIME_UNKNOWN",
            },
            {
                "information":
                    "person_description",

                "temporal_basis":
                    "none",

                "origin_gate_available":
                    False,

                "approved_claim":
                    "current-snapshot Person text only",

                "strict_database_observation_claim":
                    False,

                "risk":
                    "POST_ORIGIN_BIOGRAPHY_POSSIBLE",
            },
            {
                "information":
                    "person_country_region_locations",

                "temporal_basis":
                    "none",

                "origin_gate_available":
                    False,

                "approved_claim":
                    "current-snapshot Person geography only",

                "strict_database_observation_claim":
                    False,

                "risk":
                    "SNAPSHOT_UNVERSIONED",
            },
            {
                "information":
                    "person_current_organizations",

                "temporal_basis":
                    "none",

                "origin_gate_available":
                    False,

                "approved_claim":
                    "EDA only",

                "strict_database_observation_claim":
                    False,

                "risk":
                    "HIGH_POST_ORIGIN_LEAKAGE_RISK",
            },
            {
                "information":
                    "person_to_school",

                "temporal_basis":
                    "none",

                "origin_gate_available":
                    False,

                "approved_claim":
                    "current-snapshot education relation only",

                "strict_database_observation_claim":
                    False,

                "risk":
                    (
                        "EDUCATION_DATE_UNKNOWN_"
                        "SNAPSHOT_UNVERSIONED"
                    ),
            },
        ]
    )

    print(
        provenance_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # 7. Persist
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    case_df.to_csv(
        CASE_OUTPUT,
        index=False,
    )

    provenance_df.to_csv(
        PROVENANCE_OUTPUT,
        index=False,
    )

    mean_raw = float(
        case_df[
            "raw_founder_person_pct"
        ].mean()
    )

    mean_semantic = float(
        case_df[
            "semantic_origin_eligible_pct"
        ].mean()
    )

    mean_retention = float(
        case_df[
            "semantic_retention_pct_of_raw_person"
        ].mean()
    )

    summary = {
        "schema_version":
            "PHASE_8_1_6E_FOUNDER_PERSON_TEMPORAL_PROVENANCE_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            "DEVELOPMENT_ONLY_TEMPORAL_PROVENANCE_AUDIT",

        "mean_raw_founder_person_coverage_pct":
            mean_raw,

        "mean_semantic_origin_eligible_founder_person_pct":
            mean_semantic,

        "mean_semantic_retention_pct":
            mean_retention,

        "founder_relation_temporal_class":
            (
                "SEMANTIC_EVENT_DATE_PROXY_"
                "DATABASE_OBSERVATION_UNKNOWN"
            ),

        "person_school_temporal_class":
            (
                "CURRENT_SNAPSHOT_UNVERSIONED_"
                "NO_EDUCATION_DATES"
            ),

        "person_current_organizations_allowed":
            False,

        "relation_approved":
            False,

        "input_contract_changed":
            False,

        "graph_modified":
            False,

        "architecture_selected":
            False,

        "model_trained":
            False,

        "April_analyzed":
            False,

        "April_performance_inspected":
            False,

        "next_step":
            (
                "Use provenance + coverage evidence to define "
                "the candidate heterogeneous relation tracks "
                "before any architecture freeze."
            ),
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    manifest = {
        "phase":
            "8.1.6E",

        "status":
            "COMPLETE",

        "outputs": [
            str(CASE_OUTPUT),
            str(PROVENANCE_OUTPUT),
            str(SUMMARY_OUTPUT),
        ],

        "prohibited_operations": [
            "relation_approval",
            "input_contract_change",
            "graph_modification",
            "architecture_selection",
            "training",
            "inference",
            "ranking_scoring",
            "April_analysis",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    separator()

    print(
        "PHASE 8.1.6E FOUNDER-PERSON "
        "TEMPORAL PROVENANCE AUDIT COMPLETE"
    )

    separator()

    print()

    print(
        f"MEAN RAW PERSON COVERAGE:       "
        f"{mean_raw:.4f}%"
    )

    print(
        f"MEAN SEMANTIC PERSON COVERAGE:  "
        f"{mean_semantic:.4f}%"
    )

    print(
        f"MEAN SEMANTIC RETENTION:        "
        f"{mean_retention:.4f}%"
    )

    print()

    print(
        "FOUNDER RELATION:"
    )

    print(
        "  semantic origin gate available"
    )

    print(
        "  database observation time unknown"
    )

    print()

    print(
        "PERSON -> SCHOOL:"
    )

    print(
        "  snapshot relation only"
    )

    print(
        "  education dates unavailable"
    )

    print()

    print(
        "RELATION APPROVED:              NO"
    )

    print(
        "GRAPH MODIFIED:                 NO"
    )

    print(
        "ARCHITECTURE SELECTED:          NO"
    )

    print(
        "MODEL TRAINED:                  NO"
    )

    print(
        "APRIL ANALYZED:                 NO"
    )


if __name__ == "__main__":
    main()