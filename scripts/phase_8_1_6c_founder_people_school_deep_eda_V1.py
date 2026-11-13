from pathlib import Path
from collections import defaultdict, Counter
from functools import lru_cache
import hashlib
import json
import re
import unicodedata

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PHASE 8.1.6C — FOUNDER / PEOPLE / SCHOOL DEEP EDA V1
# =============================================================================
#
# Scientific role:
#
#   HETEROGENEOUS SIDE-INFORMATION EDA / RESOLUTION FEASIBILITY AUDIT
#
# PURPOSE
# -------
#
# 1. Reconstruct Startup -> Founder URL using the already audited Phase-3 rule.
# 2. Resolve Founder URL -> People UUID by exact normalized Crunchbase URL.
# 3. Measure founder-Person coverage specifically for cold-start startups.
# 4. Audit attributes available on resolved founder People.
# 5. Evaluate whether people.schools can be deterministically resolved
#    against the Crunchbase School registry.
# 6. Measure potential Startup -> Person -> School coverage.
#
# IMPORTANT
# ---------
#
# This script does NOT modify the graph and does NOT approve Person->School
# as a model relation.
#
# people.schools is snapshot-unversioned and has no education dates.
#
# April remains untouched.
#
# =============================================================================


RAW_DIR = Path("data/raw")

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

PHASE816B_SUMMARY = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_6_data_foundations/"
    "phase_8_1_6b_startup_attribute_eda/"
    "phase_8_1_6b_eda_summary_V1.json"
)


OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_6_data_foundations/"
    "phase_8_1_6c_founder_people_school_eda"
)

CHART_DIR = OUTPUT_DIR / "charts"

FOUNDER_RESOLUTION_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6c_founder_people_resolution_summary_V1.csv"
)

PRIMARY_FOUNDER_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6c_primary_founder_people_coverage_V1.csv"
)

FOUNDER_ATTRIBUTE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6c_founder_people_attribute_coverage_V1.csv"
)

SCHOOL_REGISTRY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6c_school_registry_resolution_surface_V1.csv"
)

SCHOOL_VALUE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6c_school_value_resolution_summary_V1.csv"
)

PERSON_SCHOOL_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6c_founder_person_school_resolution_V1.csv"
)

PRIMARY_SCHOOL_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6c_primary_founder_school_coverage_V1.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6c_eda_summary_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6c_analysis_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6c_derived_artifact_sha256_V1.json"
)


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


PERSON_URL_RE = re.compile(
    r"https?://(?:www\.)?crunchbase\.com/person/[^\s,;|]+",
    flags=re.IGNORECASE,
)


# =============================================================================
# HELPERS
# =============================================================================

def separator(char="=", width=118):
    print(char * width)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path, chunk_size=1024 * 1024):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


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


def pct(num, den):

    if den == 0:
        return float("nan")

    return 100.0 * float(num) / float(den)


def clean_value(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if not value:
        return None

    return value


def normalize_url_value(value):

    value = clean_value(value)

    if value is None:
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

    matches = PERSON_URL_RE.findall(
        str(value)
    )

    normalized = []

    for url in matches:

        url = normalize_url_value(url)

        if url is not None:
            normalized.append(url)

    return list(
        dict.fromkeys(normalized)
    )


def normalize_name(value):

    value = clean_value(value)

    if value is None:
        return None

    value = unicodedata.normalize(
        "NFKC",
        value,
    )

    value = value.casefold()

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    value = re.sub(
        r"\s*,\s*",
        ", ",
        value,
    )

    value = re.sub(
        r"\s*;\s*",
        "; ",
        value,
    )

    value = re.sub(
        r"\s*\|\s*",
        "|",
        value,
    )

    return value.strip()


def unique_nonempty_values(series):

    values = []

    for value in series:

        value = clean_value(value)

        if value is not None:
            values.append(value)

    return list(
        dict.fromkeys(values)
    )


# =============================================================================
# SCHOOL DICTIONARY SEGMENTATION
# =============================================================================

def segment_school_component(
    component,
    school_name_to_ids,
    max_fragments_per_school,
):

    component = normalize_name(component)

    if component is None:
        return []

    # Whole-string match first.
    if component in school_name_to_ids:
        return [
            (
                component,
            )
        ]

    fragments = tuple(
        part.strip()
        for part in component.split(",")
        if part.strip()
    )

    if not fragments:
        return []

    @lru_cache(maxsize=None)
    def solve(start):

        if start == len(fragments):
            return (
                tuple(),
            )

        solutions = []

        max_end = min(
            len(fragments),
            start + max_fragments_per_school,
        )

        for end in range(
            start + 1,
            max_end + 1,
        ):

            candidate = ", ".join(
                fragments[
                    start:end
                ]
            )

            if candidate not in school_name_to_ids:
                continue

            suffixes = solve(end)

            for suffix in suffixes:

                solutions.append(
                    (
                        candidate,
                        *suffix,
                    )
                )

                # We only need to distinguish unique vs ambiguous.
                if len(solutions) >= 2:
                    return tuple(
                        solutions[:2]
                    )

        return tuple(solutions)

    return list(
        solve(0)
    )


def resolve_school_value(
    value,
    school_name_to_ids,
    max_fragments_per_school,
):

    normalized = normalize_name(value)

    if normalized is None:

        return {
            "status": "EMPTY",
            "school_ids": set(),
            "school_names": tuple(),
            "method": None,
        }

    # Exact whole-field school name.
    if normalized in school_name_to_ids:

        ids = school_name_to_ids[
            normalized
        ]

        if len(ids) == 1:

            return {
                "status": "RESOLVED_UNIQUE",
                "school_ids": set(ids),
                "school_names": (
                    normalized,
                ),
                "method": "EXACT_WHOLE_FIELD",
            }

        return {
            "status": "AMBIGUOUS_SCHOOL_NAME",
            "school_ids": set(),
            "school_names": (
                normalized,
            ),
            "method": "EXACT_WHOLE_FIELD",
        }

    # Semicolon and pipe are treated as hard separators.
    hard_components = [
        part.strip()
        for part in re.split(
            r"[;|]",
            normalized,
        )
        if part.strip()
    ]

    all_names = []

    for component in hard_components:

        solutions = segment_school_component(
            component,
            school_name_to_ids,
            max_fragments_per_school,
        )

        if len(solutions) == 0:

            return {
                "status": "UNRESOLVED",
                "school_ids": set(),
                "school_names": tuple(),
                "method": "DICTIONARY_SEGMENTATION",
            }

        if len(solutions) > 1:

            return {
                "status": "AMBIGUOUS_SEGMENTATION",
                "school_ids": set(),
                "school_names": tuple(),
                "method": "DICTIONARY_SEGMENTATION",
            }

        all_names.extend(
            solutions[0]
        )

    school_ids = set()

    for name in all_names:

        ids = school_name_to_ids[
            name
        ]

        if len(ids) != 1:

            return {
                "status": "AMBIGUOUS_SCHOOL_NAME",
                "school_ids": set(),
                "school_names": tuple(
                    all_names
                ),
                "method": "DICTIONARY_SEGMENTATION",
            }

        school_ids.update(ids)

    if not school_ids:

        return {
            "status": "UNRESOLVED",
            "school_ids": set(),
            "school_names": tuple(),
            "method": "DICTIONARY_SEGMENTATION",
        }

    return {
        "status": "RESOLVED_UNIQUE",
        "school_ids": school_ids,
        "school_names": tuple(
            all_names
        ),
        "method": "DICTIONARY_SEGMENTATION",
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    separator()

    print(
        "PHASE 8.1.6C — "
        "FOUNDER / PEOPLE / SCHOOL DEEP EDA V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "HETEROGENEOUS SIDE-INFORMATION EDA"
    )

    print(
        "Development cohorts:         "
        "T57 / T58 / T59"
    )

    print(
        "April relation EDA:          NO"
    )

    print(
        "Relations approved:          NO"
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
    # 1. Prerequisite gate
    # =========================================================================

    separator("-")

    print(
        "8.1.6C.1 — PREREQUISITE GATE"
    )

    separator("-")

    require(
        PHASE816B_SUMMARY.exists(),
        "Missing Phase-8.1.6B summary.",
    )

    previous = read_json(
        PHASE816B_SUMMARY
    )

    require(
        previous.get("status")
        ==
        "COMPLETE",
        "Phase 8.1.6B is not complete.",
    )

    require(
        previous.get(
            "April_attribute_distributions_analyzed"
        )
        is False,
        "April attribute EDA unexpectedly occurred.",
    )

    print(
        "Phase 8.1.6B: COMPLETE PASS"
    )

    # =========================================================================
    # 2. Frozen Startup universe
    # =========================================================================

    separator("-")

    print(
        "8.1.6C.2 — FROZEN STARTUP UNIVERSE"
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
        len(startup_nodes)
        ==
        EXPECTED_STARTUPS,
        "Startup population drift.",
    )

    startup_ids = set(
        startup_nodes[
            "raw_entity_id"
        ]
    )

    print(
        f"Startup IDs: "
        f"{len(startup_ids):,} PASS"
    )

    # =========================================================================
    # 3. Startup -> Founder URL extraction
    # =========================================================================

    separator("-")

    print(
        "8.1.6C.3 — STARTUP -> FOUNDER URL EXTRACTION"
    )

    separator("-")

    company_files = sorted(
        RAW_DIR.glob(
            "companies*.csv"
        )
    )

    require(
        len(company_files) == 24,
        "Expected 24 Company files.",
    )

    startup_founder_urls = defaultdict(
        set
    )

    for i, path in enumerate(
        company_files,
        start=1,
    ):

        matched_rows = 0

        for chunk in pd.read_csv(
            path,
            usecols=[
                "id",
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
                    startup_ids
                )
            ]

            matched_rows += len(
                subset
            )

            for startup_id, founder_links in zip(
                subset["id"],
                subset["founder_links"],
            ):

                for url in extract_person_urls(
                    founder_links
                ):

                    startup_founder_urls[
                        str(startup_id)
                    ].add(url)

        print(
            f"[{i:02d}/24] "
            f"{path.name:<24} "
            f"startup rows={matched_rows:,}"
        )

    startups_with_founder_url = {
        startup_id
        for startup_id, urls
        in startup_founder_urls.items()
        if urls
    }

    unique_founder_urls = set()

    for urls in (
        startup_founder_urls.values()
    ):

        unique_founder_urls.update(
            urls
        )

    print()

    print(
        f"Startups with founder URL: "
        f"{len(startups_with_founder_url):,} "
        f"({pct(len(startups_with_founder_url), EXPECTED_STARTUPS):.4f}%)"
    )

    print(
        f"Unique founder URLs:       "
        f"{len(unique_founder_urls):,}"
    )

    # =========================================================================
    # 4. Exact Founder URL -> People UUID resolution
    # =========================================================================

    separator("-")

    print(
        "8.1.6C.4 — FOUNDER URL -> PEOPLE UUID RESOLUTION"
    )

    separator("-")

    people_files = sorted(
        RAW_DIR.glob(
            "people*.csv"
        )
    )

    require(
        len(people_files) == 11,
        "Expected 11 People files.",
    )

    people_header = pd.read_csv(
        people_files[0],
        nrows=0,
    )

    desired_people_fields = [
        "id",
        "link",
        "schools",
        "current_organizations",
        "description",
        "country",
        "region",
        "locations",
    ]

    people_fields = [
        field
        for field
        in desired_people_fields
        if field
        in people_header.columns
    ]

    require(
        "id" in people_fields
        and
        "link" in people_fields,
        "People id/link fields unavailable.",
    )

    matched_people_parts = []

    for i, path in enumerate(
        people_files,
        start=1,
    ):

        matched_count = 0

        for chunk in pd.read_csv(
            path,
            usecols=people_fields,
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

            chunk[
                "normalized_link"
            ] = (
                chunk[
                    "link"
                ]
                .map(
                    normalize_url_value
                )
            )

            matched = chunk[
                chunk[
                    "normalized_link"
                ].isin(
                    unique_founder_urls
                )
            ].copy()

            if not matched.empty:

                matched_people_parts.append(
                    matched
                )

                matched_count += len(
                    matched
                )

        print(
            f"[{i:02d}/11] "
            f"{path.name:<20} "
            f"matched founder rows={matched_count:,}"
        )

    if matched_people_parts:

        matched_people = pd.concat(
            matched_people_parts,
            ignore_index=True,
        )

    else:

        matched_people = pd.DataFrame(
            columns=[
                *people_fields,
                "normalized_link",
            ]
        )

    url_to_person_ids = defaultdict(
        set
    )

    for url, person_id in zip(
        matched_people[
            "normalized_link"
        ],
        matched_people[
            "id"
        ],
    ):

        if (
            clean_value(url) is not None
            and
            clean_value(person_id) is not None
        ):

            url_to_person_ids[
                str(url)
            ].add(
                str(person_id)
            )

    resolved_unique_urls = {
        url
        for url, ids
        in url_to_person_ids.items()
        if len(ids) == 1
    }

    ambiguous_urls = {
        url
        for url, ids
        in url_to_person_ids.items()
        if len(ids) > 1
    }

    unresolved_urls = (
        unique_founder_urls
        -
        set(
            url_to_person_ids
        )
    )

    url_to_person = {
        url:
            next(
                iter(
                    url_to_person_ids[
                        url
                    ]
                )
            )
        for url
        in resolved_unique_urls
    }

    print()

    print(
        f"Founder URLs total:        "
        f"{len(unique_founder_urls):,}"
    )

    print(
        f"Exactly resolved:          "
        f"{len(resolved_unique_urls):,} "
        f"({pct(len(resolved_unique_urls), len(unique_founder_urls)):.4f}%)"
    )

    print(
        f"Ambiguous URLs:            "
        f"{len(ambiguous_urls):,}"
    )

    print(
        f"Unresolved URLs:           "
        f"{len(unresolved_urls):,}"
    )

    # =========================================================================
    # 5. Startup -> resolved Person coverage
    # =========================================================================

    separator("-")

    print(
        "8.1.6C.5 — STARTUP -> RESOLVED PERSON COVERAGE"
    )

    separator("-")

    startup_person_ids = defaultdict(
        set
    )

    startup_founder_url_counts = {}

    startup_resolved_url_counts = {}

    for startup_id, urls in (
        startup_founder_urls.items()
    ):

        startup_founder_url_counts[
            startup_id
        ] = len(
            urls
        )

        resolved_count = 0

        for url in urls:

            person_id = (
                url_to_person.get(
                    url
                )
            )

            if person_id is None:
                continue

            resolved_count += 1

            startup_person_ids[
                startup_id
            ].add(
                person_id
            )

        startup_resolved_url_counts[
            startup_id
        ] = resolved_count

    startups_with_resolved_person = {
        startup_id
        for startup_id, people
        in startup_person_ids.items()
        if people
    }

    fully_resolved_startups = {
        startup_id
        for startup_id
        in startups_with_founder_url
        if (
            startup_founder_url_counts[
                startup_id
            ]
            ==
            startup_resolved_url_counts.get(
                startup_id,
                0,
            )
        )
    }

    all_resolved_person_ids = set()

    for people in (
        startup_person_ids.values()
    ):

        all_resolved_person_ids.update(
            people
        )

    print(
        f"Startups with >=1 resolved founder Person: "
        f"{len(startups_with_resolved_person):,} "
        f"({pct(len(startups_with_resolved_person), EXPECTED_STARTUPS):.4f}%)"
    )

    print(
        f"Startups with all founder URLs resolved:   "
        f"{len(fully_resolved_startups):,}"
    )

    print(
        f"Unique resolved founder People:            "
        f"{len(all_resolved_person_ids):,}"
    )

    founder_resolution_df = pd.DataFrame(
        [
            {
                "startup_population":
                    EXPECTED_STARTUPS,

                "startups_with_founder_url":
                    len(
                        startups_with_founder_url
                    ),

                "founder_url_coverage_pct":
                    pct(
                        len(
                            startups_with_founder_url
                        ),
                        EXPECTED_STARTUPS,
                    ),

                "unique_founder_urls":
                    len(
                        unique_founder_urls
                    ),

                "exactly_resolved_founder_urls":
                    len(
                        resolved_unique_urls
                    ),

                "exact_resolution_pct":
                    pct(
                        len(
                            resolved_unique_urls
                        ),
                        len(
                            unique_founder_urls
                        ),
                    ),

                "ambiguous_urls":
                    len(
                        ambiguous_urls
                    ),

                "unresolved_urls":
                    len(
                        unresolved_urls
                    ),

                "startups_with_resolved_person":
                    len(
                        startups_with_resolved_person
                    ),

                "resolved_person_startup_coverage_pct":
                    pct(
                        len(
                            startups_with_resolved_person
                        ),
                        EXPECTED_STARTUPS,
                    ),

                "unique_founder_people":
                    len(
                        all_resolved_person_ids
                    ),
            }
        ]
    )

    # =========================================================================
    # 6. Development-primary founder coverage
    # =========================================================================

    separator("-")

    print(
        "8.1.6C.6 — DEVELOPMENT PRIMARY "
        "FOUNDER-PERSON COVERAGE"
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

    primary_rows = []

    primary_unique_ids = set()

    for origin in (
        DEVELOPMENT_ORIGINS
    ):

        origin_ids = (
            primary.loc[
                primary[
                    "origin"
                ]
                ==
                origin,
                "startup_id_str",
            ]
            .tolist()
        )

        require(
            len(origin_ids)
            ==
            EXPECTED_PRIMARY[
                origin
            ],
            f"{origin}: primary population drift.",
        )

        primary_unique_ids.update(
            origin_ids
        )

        with_url = sum(
            startup_id
            in
            startups_with_founder_url
            for startup_id
            in origin_ids
        )

        with_person = sum(
            startup_id
            in
            startups_with_resolved_person
            for startup_id
            in origin_ids
        )

        primary_rows.append(
            {
                "origin":
                    origin,

                "primary_cases":
                    len(
                        origin_ids
                    ),

                "with_founder_url":
                    with_url,

                "founder_url_coverage_pct":
                    pct(
                        with_url,
                        len(
                            origin_ids
                        ),
                    ),

                "with_resolved_founder_person":
                    with_person,

                "resolved_founder_person_coverage_pct":
                    pct(
                        with_person,
                        len(
                            origin_ids
                        ),
                    ),
            }
        )

        print()

        print(
            f"{origin}:"
        )

        print(
            f"  primary cases:             "
            f"{len(origin_ids):,}"
        )

        print(
            f"  with founder URL:          "
            f"{with_url:,} "
            f"({pct(with_url, len(origin_ids)):.4f}%)"
        )

        print(
            f"  with resolved Person:      "
            f"{with_person:,} "
            f"({pct(with_person, len(origin_ids)):.4f}%)"
        )

    primary_founder_df = pd.DataFrame(
        primary_rows
    )

    # =========================================================================
    # 7. Resolved founder-People attribute coverage
    # =========================================================================

    separator("-")

    print(
        "8.1.6C.7 — FOUNDER PEOPLE "
        "ATTRIBUTE COVERAGE"
    )

    separator("-")

    person_field_values = defaultdict(
        lambda:
            defaultdict(
                set
            )
    )

    founder_attribute_fields = [
        field
        for field
        in [
            "schools",
            "current_organizations",
            "description",
            "country",
            "region",
            "locations",
        ]
        if field
        in matched_people.columns
    ]

    for row in matched_people.itertuples(
        index=False
    ):

        person_id = clean_value(
            getattr(
                row,
                "id"
            )
        )

        if (
            person_id is None
            or
            person_id
            not in
            all_resolved_person_ids
        ):
            continue

        for field in (
            founder_attribute_fields
        ):

            value = clean_value(
                getattr(
                    row,
                    field
                )
            )

            if value is not None:

                person_field_values[
                    person_id
                ][
                    field
                ].add(
                    value
                )

    founder_attribute_rows = []

    for field in (
        founder_attribute_fields
    ):

        covered = sum(
            bool(
                person_field_values[
                    person_id
                ][
                    field
                ]
            )
            for person_id
            in all_resolved_person_ids
        )

        conflicting = sum(
            len(
                person_field_values[
                    person_id
                ][
                    field
                ]
            )
            >
            1
            for person_id
            in all_resolved_person_ids
        )

        founder_attribute_rows.append(
            {
                "field":
                    field,

                "resolved_founder_people":
                    len(
                        all_resolved_person_ids
                    ),

                "people_with_value":
                    covered,

                "coverage_pct":
                    pct(
                        covered,
                        len(
                            all_resolved_person_ids
                        ),
                    ),

                "people_with_multiple_distinct_values":
                    conflicting,
            }
        )

        print(
            f"{field:<24} "
            f"{covered:,}/"
            f"{len(all_resolved_person_ids):,} "
            f"({pct(covered, len(all_resolved_person_ids)):.4f}%)"
        )

    founder_attribute_df = pd.DataFrame(
        founder_attribute_rows
    )

    # =========================================================================
    # 8. School registry
    # =========================================================================

    separator("-")

    print(
        "8.1.6C.8 — SCHOOL REGISTRY "
        "NAME RESOLUTION SURFACE"
    )

    separator("-")

    school_files = sorted(
        RAW_DIR.glob(
            "CRUNCHBASE_school*.csv"
        )
    )

    require(
        len(school_files) == 1,
        "Expected exactly one School registry.",
    )

    school_header = pd.read_csv(
        school_files[0],
        nrows=0,
    )

    require(
        "id" in school_header.columns
        and
        "name" in school_header.columns,
        "School id/name unavailable.",
    )

    schools = pd.read_csv(
        school_files[0],
        usecols=[
            "id",
            "name",
        ],
        dtype="string",
        low_memory=False,
    )

    schools[
        "id"
    ] = (
        schools[
            "id"
        ]
        .astype("string")
        .str.strip()
    )

    schools[
        "normalized_name"
    ] = (
        schools[
            "name"
        ]
        .map(
            normalize_name
        )
    )

    schools = schools[
        schools[
            "id"
        ].notna()
        &
        schools[
            "normalized_name"
        ].notna()
    ].copy()

    school_name_to_ids = defaultdict(
        set
    )

    for school_id, name in zip(
        schools[
            "id"
        ],
        schools[
            "normalized_name"
        ],
    ):

        school_name_to_ids[
            name
        ].add(
            str(
                school_id
            )
        )

    unique_school_names = sum(
        len(ids) == 1
        for ids
        in school_name_to_ids.values()
    )

    ambiguous_school_names = sum(
        len(ids) > 1
        for ids
        in school_name_to_ids.values()
    )

    max_fragments_per_school = max(
        (
            name.count(",")
            +
            1
        )
        for name
        in school_name_to_ids
    )

    school_registry_df = pd.DataFrame(
        [
            {
                "school_rows":
                    len(
                        schools
                    ),

                "normalized_school_names":
                    len(
                        school_name_to_ids
                    ),

                "unique_name_to_one_id":
                    unique_school_names,

                "ambiguous_name_to_multiple_ids":
                    ambiguous_school_names,

                "max_comma_fragments_in_school_name":
                    max_fragments_per_school,
            }
        ]
    )

    print(
        f"School registry rows:             "
        f"{len(schools):,}"
    )

    print(
        f"Distinct normalized names:        "
        f"{len(school_name_to_ids):,}"
    )

    print(
        f"Unique name -> one School ID:     "
        f"{unique_school_names:,}"
    )

    print(
        f"Ambiguous school names:           "
        f"{ambiguous_school_names:,}"
    )

    # =========================================================================
    # 9. Founder People -> School resolution
    # =========================================================================

    separator("-")

    print(
        "8.1.6C.9 — FOUNDER PERSON -> "
        "SCHOOL RESOLUTION"
    )

    separator("-")

    school_value_status_counter = Counter()
    school_value_method_counter = Counter()

    person_school_ids = defaultdict(
        set
    )

    person_school_status = {}

    people_with_school_field = 0

    for person_id in (
        all_resolved_person_ids
    ):

        values = person_field_values[
            person_id
        ][
            "schools"
        ]

        if not values:

            person_school_status[
                person_id
            ] = "NO_SCHOOL_FIELD"

            continue

        people_with_school_field += 1

        value_results = []

        for value in values:

            result = resolve_school_value(
                value,
                school_name_to_ids,
                max_fragments_per_school,
            )

            value_results.append(
                result
            )

            school_value_status_counter[
                result[
                    "status"
                ]
            ] += 1

            if result[
                "method"
            ] is not None:

                school_value_method_counter[
                    result[
                        "method"
                    ]
                ] += 1

        statuses = {
            result[
                "status"
            ]
            for result
            in value_results
        }

        if statuses == {
            "RESOLVED_UNIQUE"
        }:

            person_school_status[
                person_id
            ] = "RESOLVED_UNIQUE"

            for result in value_results:

                person_school_ids[
                    person_id
                ].update(
                    result[
                        "school_ids"
                    ]
                )

        elif (
            "AMBIGUOUS_SEGMENTATION"
            in
            statuses
        ):

            person_school_status[
                person_id
            ] = "AMBIGUOUS_SEGMENTATION"

        elif (
            "AMBIGUOUS_SCHOOL_NAME"
            in
            statuses
        ):

            person_school_status[
                person_id
            ] = "AMBIGUOUS_SCHOOL_NAME"

        elif (
            "RESOLVED_UNIQUE"
            in
            statuses
        ):

            person_school_status[
                person_id
            ] = "PARTIAL_RESOLUTION"

        else:

            person_school_status[
                person_id
            ] = "UNRESOLVED"

    person_status_counter = Counter(
        person_school_status.values()
    )

    resolved_school_people = {
        person_id
        for person_id, status
        in person_school_status.items()
        if status
        ==
        "RESOLVED_UNIQUE"
    }

    candidate_person_school_pairs = sum(
        len(
            person_school_ids[
                person_id
            ]
        )
        for person_id
        in resolved_school_people
    )

    print(
        f"Resolved founder People:          "
        f"{len(all_resolved_person_ids):,}"
    )

    print(
        f"Founder People with schools:      "
        f"{people_with_school_field:,} "
        f"({pct(people_with_school_field, len(all_resolved_person_ids)):.4f}%)"
    )

    print(
        f"Strictly resolved Person->School: "
        f"{len(resolved_school_people):,} "
        f"({pct(len(resolved_school_people), len(all_resolved_person_ids)):.4f}%)"
    )

    print(
        f"Candidate Person-School pairs:    "
        f"{candidate_person_school_pairs:,}"
    )

    print()

    print(
        "Person-level school resolution statuses:"
    )

    for status, count in (
        person_status_counter.most_common()
    ):

        print(
            f"  {status:<26} "
            f"{count:,}"
        )

    school_value_rows = []

    all_value_statuses = sorted(
        set(
            school_value_status_counter
        )
    )

    for status in (
        all_value_statuses
    ):

        school_value_rows.append(
            {
                "status":
                    status,

                "value_count":
                    school_value_status_counter[
                        status
                    ],
            }
        )

    school_value_df = pd.DataFrame(
        school_value_rows
    )

    person_school_df = pd.DataFrame(
        [
            {
                "resolved_founder_people":
                    len(
                        all_resolved_person_ids
                    ),

                "people_with_school_field":
                    people_with_school_field,

                "school_field_coverage_pct":
                    pct(
                        people_with_school_field,
                        len(
                            all_resolved_person_ids
                        ),
                    ),

                "people_with_strict_school_resolution":
                    len(
                        resolved_school_people
                    ),

                "strict_school_resolution_pct_all_founders":
                    pct(
                        len(
                            resolved_school_people
                        ),
                        len(
                            all_resolved_person_ids
                        ),
                    ),

                "strict_school_resolution_pct_of_people_with_schools":
                    pct(
                        len(
                            resolved_school_people
                        ),
                        people_with_school_field,
                    ),

                "candidate_person_school_pairs":
                    candidate_person_school_pairs,
            }
        ]
    )

    # =========================================================================
    # 10. Startup -> Founder -> School coverage
    # =========================================================================

    separator("-")

    print(
        "8.1.6C.10 — STARTUP -> FOUNDER -> "
        "SCHOOL COVERAGE"
    )

    separator("-")

    startups_with_school_context = set()

    for startup_id, people in (
        startup_person_ids.items()
    ):

        if any(
            person_id
            in
            resolved_school_people
            for person_id
            in people
        ):

            startups_with_school_context.add(
                startup_id
            )

    print(
        f"All Startup nodes with strict "
        f"founder-school context: "
        f"{len(startups_with_school_context):,} "
        f"({pct(len(startups_with_school_context), EXPECTED_STARTUPS):.4f}%)"
    )

    primary_school_rows = []

    for origin in (
        DEVELOPMENT_ORIGINS
    ):

        origin_ids = (
            primary.loc[
                primary[
                    "origin"
                ]
                ==
                origin,
                "startup_id_str",
            ]
            .tolist()
        )

        with_person = sum(
            startup_id
            in
            startups_with_resolved_person
            for startup_id
            in origin_ids
        )

        with_school = sum(
            startup_id
            in
            startups_with_school_context
            for startup_id
            in origin_ids
        )

        primary_school_rows.append(
            {
                "origin":
                    origin,

                "primary_cases":
                    len(
                        origin_ids
                    ),

                "with_resolved_founder_person":
                    with_person,

                "founder_person_coverage_pct":
                    pct(
                        with_person,
                        len(
                            origin_ids
                        ),
                    ),

                "with_strict_founder_school_context":
                    with_school,

                "founder_school_coverage_pct":
                    pct(
                        with_school,
                        len(
                            origin_ids
                        ),
                    ),
            }
        )

        print()

        print(
            f"{origin}:"
        )

        print(
            f"  with founder Person: "
            f"{with_person:,} "
            f"({pct(with_person, len(origin_ids)):.4f}%)"
        )

        print(
            f"  with founder School: "
            f"{with_school:,} "
            f"({pct(with_school, len(origin_ids)):.4f}%)"
        )

    primary_school_df = pd.DataFrame(
        primary_school_rows
    )

    # =========================================================================
    # 11. Persist
    # =========================================================================

    separator("-")

    print(
        "8.1.6C.11 — PERSIST EDA ARTIFACTS"
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

    founder_resolution_df.to_csv(
        FOUNDER_RESOLUTION_OUTPUT,
        index=False,
    )

    primary_founder_df.to_csv(
        PRIMARY_FOUNDER_OUTPUT,
        index=False,
    )

    founder_attribute_df.to_csv(
        FOUNDER_ATTRIBUTE_OUTPUT,
        index=False,
    )

    school_registry_df.to_csv(
        SCHOOL_REGISTRY_OUTPUT,
        index=False,
    )

    school_value_df.to_csv(
        SCHOOL_VALUE_OUTPUT,
        index=False,
    )

    person_school_df.to_csv(
        PERSON_SCHOOL_OUTPUT,
        index=False,
    )

    primary_school_df.to_csv(
        PRIMARY_SCHOOL_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 12. Charts
    # =========================================================================

    separator("-")

    print(
        "8.1.6C.12 — FOUNDER / SCHOOL CHARTS"
    )

    separator("-")

    chart_df = primary_school_df[
        [
            "origin",
            "founder_person_coverage_pct",
            "founder_school_coverage_pct",
        ]
    ].copy()

    x = np.arange(
        len(
            chart_df
        )
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        x - width / 2,
        chart_df[
            "founder_person_coverage_pct"
        ],
        width=width,
        label="Resolved founder Person",
    )

    ax.bar(
        x + width / 2,
        chart_df[
            "founder_school_coverage_pct"
        ],
        width=width,
        label="Resolved founder→School",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        chart_df[
            "origin"
        ]
    )

    ax.set_ylabel(
        "Primary cold-start coverage (%)"
    )

    ax.set_title(
        "Inductive heterogeneous context coverage"
    )

    ax.legend()

    fig.tight_layout()

    coverage_chart = (
        CHART_DIR
        / "phase_8_1_6c_primary_founder_school_coverage_V1.png"
    )

    fig.savefig(
        coverage_chart,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    status_chart_df = pd.DataFrame(
        [
            {
                "status":
                    status,
                "count":
                    count,
            }
            for status, count
            in person_status_counter.items()
        ]
    ).sort_values(
        "count",
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.barh(
        status_chart_df[
            "status"
        ],
        status_chart_df[
            "count"
        ],
    )

    ax.set_xlabel(
        "Resolved founder People"
    )

    ax.set_title(
        "Founder Person → School resolution status"
    )

    fig.tight_layout()

    school_chart = (
        CHART_DIR
        / "phase_8_1_6c_school_resolution_status_V1.png"
    )

    fig.savefig(
        school_chart,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"WROTE {coverage_chart}"
    )

    print(
        f"WROTE {school_chart}"
    )

    # =========================================================================
    # 13. Summary
    # =========================================================================

    separator("-")

    print(
        "8.1.6C.13 — SCIENTIFIC INTERPRETATION"
    )

    separator("-")

    mean_primary_person_coverage = float(
        primary_school_df[
            "founder_person_coverage_pct"
        ].mean()
    )

    mean_primary_school_coverage = float(
        primary_school_df[
            "founder_school_coverage_pct"
        ].mean()
    )

    print()

    print(
        f"Mean primary founder-Person coverage: "
        f"{mean_primary_person_coverage:.4f}%"
    )

    print(
        f"Mean primary founder-School coverage: "
        f"{mean_primary_school_coverage:.4f}%"
    )

    print()

    print(
        "Interpretation:"
    )

    print(
        "  Explicit Person nodes may provide "
        "heterogeneous context even when founders "
        "are not shared between two startups."
    )

    print(
        "  Person->School remains a candidate "
        "snapshot relation only; education dates "
        "are unavailable."
    )

    summary = {
        "schema_version":
            "PHASE_8_1_6C_FOUNDER_PEOPLE_SCHOOL_EDA_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            (
                "HETEROGENEOUS_SIDE_INFORMATION_"
                "RESOLUTION_FEASIBILITY_EDA"
            ),

        "startup_population":
            EXPECTED_STARTUPS,

        "startups_with_founder_url":
            len(
                startups_with_founder_url
            ),

        "unique_founder_urls":
            len(
                unique_founder_urls
            ),

        "exactly_resolved_founder_urls":
            len(
                resolved_unique_urls
            ),

        "unique_resolved_founder_people":
            len(
                all_resolved_person_ids
            ),

        "startups_with_resolved_founder_person":
            len(
                startups_with_resolved_person
            ),

        "founder_people_with_school_field":
            people_with_school_field,

        "strictly_resolved_founder_people_with_school":
            len(
                resolved_school_people
            ),

        "candidate_person_school_pairs":
            candidate_person_school_pairs,

        "startups_with_strict_founder_school_context":
            len(
                startups_with_school_context
            ),

        "mean_primary_founder_person_coverage_pct":
            mean_primary_person_coverage,

        "mean_primary_founder_school_coverage_pct":
            mean_primary_school_coverage,

        "school_temporal_provenance":
            "CURRENT_SNAPSHOT_UNVERSIONED_NO_EDUCATION_DATES",

        "relation_selected":
            False,

        "graph_modified":
            False,

        "architecture_selected":
            False,

        "model_trained":
            False,

        "April_relation_EDA":
            False,

        "April_performance_inspected":
            False,

        "next_step":
            (
                "Compare explicit Founder-Person / School "
                "coverage against the existing projected "
                "shared-founder graph and audit incremental "
                "cold-start connectivity."
            ),
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    outputs = [
        FOUNDER_RESOLUTION_OUTPUT,
        PRIMARY_FOUNDER_OUTPUT,
        FOUNDER_ATTRIBUTE_OUTPUT,
        SCHOOL_REGISTRY_OUTPUT,
        SCHOOL_VALUE_OUTPUT,
        PERSON_SCHOOL_OUTPUT,
        PRIMARY_SCHOOL_OUTPUT,
        SUMMARY_OUTPUT,
        coverage_chart,
        school_chart,
    ]

    manifest = {
        "phase":
            "8.1.6C",

        "status":
            "COMPLETE",

        "development_origins":
            DEVELOPMENT_ORIGINS,

        "April_relation_EDA":
            False,

        "relations_approved":
            False,

        "graph_modified":
            False,

        "architecture_selected":
            False,

        "model_trained":
            False,

        "outputs": [
            str(path)
            for path
            in outputs
        ],

        "prohibited_operations": [
            "relation_selection",
            "graph_modification",
            "architecture_selection",
            "training",
            "inference",
            "ranking_scoring",
            "April_relation_EDA",
            "April_performance_inspection",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    outputs.append(
        MANIFEST_OUTPUT
    )

    hashes = {
        str(path):
            sha256_file(path)
        for path
        in outputs
    }

    write_json(
        HASH_OUTPUT,
        hashes,
    )

    separator()

    print(
        "PHASE 8.1.6C FOUNDER / PEOPLE / "
        "SCHOOL DEEP EDA COMPLETE"
    )

    separator()

    print()

    print(
        "STARTUP -> PERSON RESOLUTION:     AUDITED"
    )

    print(
        "PERSON -> SCHOOL RESOLUTION:      AUDITED"
    )

    print(
        "COLD-START COVERAGE:              AUDITED"
    )

    print(
        "RELATIONS APPROVED:               NO"
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
        "APRIL RELATION EDA:               NO"
    )

    print(
        "APRIL PERFORMANCE INSPECTED:      NO"
    )

    print()

    print(
        "Next:"
    )

    print(
        "Measure how much explicit Person / School "
        "context reduces cold-start structural isolation "
        "relative to the existing shared-founder projection."
    )

    print()

    print(
        f"Outputs written to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()