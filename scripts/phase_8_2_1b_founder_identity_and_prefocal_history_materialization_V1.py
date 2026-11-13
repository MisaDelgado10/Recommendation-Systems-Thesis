from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 8.2.1B — FOUNDER IDENTITY + PRE-FOCAL HISTORY MATERIALIZATION V1
# =============================================================================
#
# Scientific role:
#
#   IDENTITY-RESOLVED HUMAN-CAPITAL / HISTORY SOURCE MATERIALIZATION
#
# PURPOSE
# -------
#
# Materialize:
#
#   1. exact Crunchbase Founder URL -> Person UUID resolution;
#   2. explicit frozen-Startup -> Founder Person relations;
#   3. the Person node universe used by the Phase-8 heterogeneous graph;
#   4. all resolved Company -> Founder Person relations needed ONLY as
#      historical aggregation context;
#   5. focal-Startup -> prior-founder-company pairs under the frozen rule:
#
#         prior_company != focal_startup
#         AND
#         prior_company.founded_on < focal_startup.founded_on
#
#   6. founder_prior_startup_count for every frozen Startup.
#
# IMPORTANT
# ---------
#
# The prior-history cutoff is the FOCAL STARTUP FOUNDING DATE, not prediction
# origin. This implements Phase 8.1.10A exactly.
#
# External Companies are permitted as aggregation/context records but:
#
#   - are NOT candidate Startups;
#   - are NOT added to the Phase-8 graph as Company nodes;
#   - are NOT scored;
#   - do NOT expand the 311,589 frozen Startup candidate universe.
#
# No learned preprocessing is fit here.
# No sector vocabulary is fit here.
# No model is instantiated.
# No performance is inspected.
#
# =============================================================================


RAW_DIR = Path("data/raw")

COMPANY_PATHS = sorted(
    RAW_DIR.glob("companies*.csv")
)

PEOPLE_PATHS = sorted(
    RAW_DIR.glob("people*.csv")
)


PHASE8_ROOT = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development"
)

SOURCE_PATH = (
    PHASE8_ROOT
    / "phase_8_2_1_feature_materialization"
    / "phase_8_2_1a_canonical_startup_source_V1.parquet"
)

SOURCE_MANIFEST = (
    PHASE8_ROOT
    / "phase_8_2_1_feature_materialization"
    / "phase_8_2_1a_manifest_V1.json"
)

CONSISTENCY_CONTRACT = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_10a_inductive_consistency_closure/"
    "phase_8_1_10a_consistency_amendment_V1.json"
)

ARCHITECTURE_CONTRACT = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_11_exact_neural_architecture/"
    "phase_8_1_11_architecture_contract_V1.json"
)


OUTPUT_DIR = (
    PHASE8_ROOT
    / "phase_8_2_1_feature_materialization"
)

URL_RESOLUTION_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1b_founder_url_resolution_V1.parquet"
)

COMPANY_FOUNDER_EDGE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1b_resolved_company_founder_edges_V1.parquet"
)

STARTUP_FOUNDER_EDGE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1b_startup_founder_edges_V1.parquet"
)

PERSON_NODE_SOURCE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1b_person_node_source_V1.parquet"
)

PRIOR_COMPANY_PAIR_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1b_prefocal_founder_company_pairs_V1.parquet"
)

FOUNDER_HISTORY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1b_founder_history_source_V1.parquet"
)

RESOLUTION_AUDIT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1b_resolution_audit_V1.csv"
)

EDGE_DUPLICATE_AUDIT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1b_company_founder_duplicate_audit_V1.csv"
)

COVERAGE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1b_founder_history_coverage_V1.csv"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1b_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1b_artifact_sha256_V1.json"
)


EXPECTED_STARTUPS = 311_589
EXPECTED_COMPANY_FILES = 24
EXPECTED_PEOPLE_FILES = 11

# Frozen results already reproduced in Phase 8.2.1A / Phase 8.1 EDA.
EXPECTED_STARTUPS_WITH_FOUNDER_LINKS = 184_393
EXPECTED_STARTUPS_WITH_RESOLVED_FOUNDER = 184_388

CHUNK_SIZE = 200_000


PERSON_URL_RE = re.compile(
    r"https?://(?:www\.)?crunchbase\.com/person/[^\s,;|]+",
    flags=re.IGNORECASE,
)


COMPANY_USECOLS = [
    "id",
    "founder_links",
    "founded_on",
    "categories",
    "category_groups",
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

            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

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


def normalize_url(value):

    value = normalize_scalar(
        value
    )

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

    value = normalize_scalar(
        value
    )

    if value is None:
        return []

    matches = PERSON_URL_RE.findall(
        value
    )

    normalized = []

    for url in matches:

        url = normalize_url(
            url
        )

        if url is not None:
            normalized.append(
                url
            )

    # One founder URL may not contribute twice to the same Company.
    return list(
        dict.fromkeys(
            normalized
        )
    )


def first_nonmissing(series):

    values = series.dropna()

    if len(values) == 0:
        return None

    return values.iloc[0]


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    separator()

    print(
        "PHASE 8.2.1B — "
        "FOUNDER IDENTITY + PRE-FOCAL HISTORY "
        "MATERIALIZATION V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "IDENTITY-RESOLVED FOUNDER-HISTORY SOURCE"
    )

    print("Method design reopened:      NO")
    print("Learned preprocessing fit:   NO")
    print("Sector vocabulary fit:       NO")
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
        "8.2.1B.1 — PREREQUISITE GATES"
    )

    separator("-")

    required = [
        SOURCE_PATH,
        SOURCE_MANIFEST,
        CONSISTENCY_CONTRACT,
        ARCHITECTURE_CONTRACT,
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
            f"Expected {EXPECTED_COMPANY_FILES} Companies shards; "
            f"found {len(COMPANY_PATHS)}."
        ),
    )

    require(
        len(PEOPLE_PATHS)
        ==
        EXPECTED_PEOPLE_FILES,
        (
            f"Expected {EXPECTED_PEOPLE_FILES} People shards; "
            f"found {len(PEOPLE_PATHS)}."
        ),
    )

    print(
        f"FOUND {len(COMPANY_PATHS)} Companies shards"
    )

    print(
        f"FOUND {len(PEOPLE_PATHS)} People shards"
    )

    source_manifest = read_json(
        SOURCE_MANIFEST
    )

    consistency = read_json(
        CONSISTENCY_CONTRACT
    )

    architecture = read_json(
        ARCHITECTURE_CONTRACT
    )

    require(
        source_manifest.get("status")
        ==
        "MATERIALIZED_AND_AUDITED",
        "Phase 8.2.1A source is not authoritative.",
    )

    require(
        consistency.get("status")
        ==
        "FROZEN",
        "Phase 8.1.10A is not frozen.",
    )

    require(
        architecture.get("status")
        ==
        "FROZEN",
        "Phase 8.1.11 is not frozen.",
    )

    require(
        consistency[
            "person_initialization_effective_policy"
        ][
            "founder_history_stored_on_person"
        ]
        is False,
        (
            "Founder history unexpectedly stored "
            "on Person nodes."
        ),
    )

    print()

    print(
        "Founder-history semantic gate: PASS"
    )

    # =========================================================================
    # 2. Load canonical Startup substrate
    # =========================================================================

    separator("-")

    print(
        "8.2.1B.2 — CANONICAL STARTUP SOURCE"
    )

    separator("-")

    startups = pd.read_parquet(
        SOURCE_PATH,
        columns=[
            "startup_node_index",
            "startup_node_id",
            "startup_id",
            "founded_on",
            "founder_links_raw",
        ],
    )

    startups[
        "startup_id"
    ] = (
        startups[
            "startup_id"
        ]
        .astype(str)
    )

    require(
        len(startups)
        ==
        EXPECTED_STARTUPS,
        "Canonical Startup count drift.",
    )

    require(
        startups[
            "startup_id"
        ].nunique()
        ==
        EXPECTED_STARTUPS,
        "Canonical Startup IDs are not unique.",
    )

    startup_id_set = set(
        startups[
            "startup_id"
        ].tolist()
    )

    startups_with_founder_links = int(
        startups[
            "founder_links_raw"
        ]
        .notna()
        .sum()
    )

    require(
        startups_with_founder_links
        ==
        EXPECTED_STARTUPS_WITH_FOUNDER_LINKS,
        (
            "Startup founder-link coverage differs "
            "from the frozen source audit."
        ),
    )

    print(
        f"Frozen Startups:              "
        f"{len(startups):,}"
    )

    print(
        f"Startups with founder_links:  "
        f"{startups_with_founder_links:,}"
    )

    # =========================================================================
    # 3. Extract focal Startup founder URL universe
    # =========================================================================

    separator("-")

    print(
        "8.2.1B.3 — FOCAL STARTUP FOUNDER URL UNIVERSE"
    )

    separator("-")

    startup_founder_url_rows = []

    for row in startups.loc[
        startups[
            "founder_links_raw"
        ].notna()
    ].itertuples(
        index=False
    ):

        urls = extract_person_urls(
            row.founder_links_raw
        )

        for url in urls:

            startup_founder_url_rows.append(
                {
                    "startup_id":
                        row.startup_id,

                    "startup_node_index":
                        int(
                            row.startup_node_index
                        ),

                    "person_url":
                        url,
                }
            )

    startup_founder_urls = (
        pd.DataFrame(
            startup_founder_url_rows
        )
        .drop_duplicates()
    )

    require(
        len(startup_founder_urls)
        >
        0,
        "No Startup founder URLs were parsed.",
    )

    focal_founder_url_set = set(
        startup_founder_urls[
            "person_url"
        ].unique()
    )

    print(
        f"Startup founder URL references: "
        f"{len(startup_founder_urls):,}"
    )

    print(
        f"Unique Startup founder URLs:     "
        f"{len(focal_founder_url_set):,}"
    )

    # =========================================================================
    # 4. First Companies pass: complete founder URL universe
    #
    # All Companies may contribute ONLY as prior-founder context.
    # No external Company becomes a candidate or graph node.
    # =========================================================================

    separator("-")

    print(
        "8.2.1B.4 — ALL-COMPANY FOUNDER URL DISCOVERY"
    )

    separator("-")

    all_founder_urls = set()

    company_rows_with_founder_links = 0

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
            usecols=[
                "id",
                "founder_links",
            ],
            dtype={
                "id":
                    "string",
                "founder_links":
                    "string",
            },
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):

            founder_values = (
                chunk[
                    "founder_links"
                ]
                .dropna()
            )

            company_rows_with_founder_links += int(
                len(
                    founder_values
                )
            )

            for value in founder_values:

                all_founder_urls.update(
                    extract_person_urls(
                        value
                    )
                )

    require(
        focal_founder_url_set.issubset(
            all_founder_urls
        ),
        (
            "Canonical Startup founder URLs are not "
            "a subset of the all-Company URL universe."
        ),
    )

    print()

    print(
        f"Companies rows with founder_links: "
        f"{company_rows_with_founder_links:,}"
    )

    print(
        f"Unique founder URLs required:      "
        f"{len(all_founder_urls):,}"
    )

    # =========================================================================
    # 5. Exact People URL -> Person UUID resolution
    # =========================================================================

    separator("-")

    print(
        "8.2.1B.5 — EXACT PEOPLE URL RESOLUTION"
    )

    separator("-")

    matched_people_parts = []

    for i, path in enumerate(
        PEOPLE_PATHS,
        start=1,
    ):

        print(
            f"[{i:02d}/{len(PEOPLE_PATHS):02d}] "
            f"{path.name}"
        )

        for chunk in pd.read_csv(
            path,
            usecols=[
                "id",
                "link",
            ],
            dtype={
                "id":
                    "string",
                "link":
                    "string",
            },
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):

            chunk[
                "person_id"
            ] = (
                chunk[
                    "id"
                ]
                .astype(str)
                .str.strip()
            )

            chunk[
                "person_url"
            ] = (
                chunk[
                    "link"
                ]
                .map(
                    normalize_url
                )
            )

            matched = (
                chunk.loc[
                    chunk[
                        "person_url"
                    ].isin(
                        all_founder_urls
                    ),
                    [
                        "person_url",
                        "person_id",
                    ],
                ]
                .dropna()
                .drop_duplicates()
            )

            if not matched.empty:

                matched_people_parts.append(
                    matched
                )

    require(
        len(matched_people_parts)
        >
        0,
        "No Founder URLs resolved to People.",
    )

    matched_people = pd.concat(
        matched_people_parts,
        ignore_index=True,
    ).drop_duplicates()

    resolution_counts = (
        matched_people
        .groupby(
            "person_url",
            observed=True,
        )[
            "person_id"
        ]
        .nunique()
    )

    unique_urls = set(
        resolution_counts[
            resolution_counts
            ==
            1
        ].index
    )

    ambiguous_urls = set(
        resolution_counts[
            resolution_counts
            >
            1
        ].index
    )

    matched_urls = set(
        resolution_counts.index
    )

    unresolved_urls = (
        all_founder_urls
        -
        matched_urls
    )

    exact_people = (
        matched_people.loc[
            matched_people[
                "person_url"
            ].isin(
                unique_urls
            )
        ]
        .drop_duplicates(
            subset=[
                "person_url",
                "person_id",
            ]
        )
        .drop_duplicates(
            subset=[
                "person_url"
            ],
            keep="first",
        )
        .copy()
    )

    exact_map = dict(
        zip(
            exact_people[
                "person_url"
            ],
            exact_people[
                "person_id"
            ],
        )
    )

    require(
        set(
            exact_map
        )
        ==
        unique_urls,
        "Exact People resolution map drift.",
    )

    print()

    print(
        f"Exact unique URLs:   "
        f"{len(unique_urls):,}"
    )

    print(
        f"Ambiguous URLs:      "
        f"{len(ambiguous_urls):,}"
    )

    print(
        f"Unresolved URLs:     "
        f"{len(unresolved_urls):,}"
    )

    print(
        "Ambiguous URLs mapped: 0"
    )

    # -------------------------------------------------------------------------
    # Focal Startup resolution-specific audit.
    # -------------------------------------------------------------------------

    focal_exact_urls = (
        focal_founder_url_set
        &
        unique_urls
    )

    focal_ambiguous_urls = (
        focal_founder_url_set
        &
        ambiguous_urls
    )

    focal_unresolved_urls = (
        focal_founder_url_set
        -
        matched_urls
    )

    focal_resolution_rate = (
        len(
            focal_exact_urls
        )
        /
        len(
            focal_founder_url_set
        )
    )

    require(
        focal_resolution_rate
        >
        0.999,
        (
            "Focal Startup founder URL exact-resolution "
            "rate fell below 99.9%."
        ),
    )

    print()

    print(
        f"Focal exact URLs:      "
        f"{len(focal_exact_urls):,}"
    )

    print(
        f"Focal ambiguous URLs:  "
        f"{len(focal_ambiguous_urls):,}"
    )

    print(
        f"Focal unresolved URLs: "
        f"{len(focal_unresolved_urls):,}"
    )

    print(
        f"Focal exact rate:      "
        f"{focal_resolution_rate * 100.0:.4f}%"
    )

    # =========================================================================
    # 6. Persist URL resolution registry
    # =========================================================================

    resolution_rows = []

    for url in sorted(
        all_founder_urls
    ):

        if url in unique_urls:

            status = (
                "EXACT_UNIQUE"
            )

            person_id = (
                exact_map[
                    url
                ]
            )

        elif url in ambiguous_urls:

            status = (
                "AMBIGUOUS_MULTIPLE_PERSON_IDS"
            )

            person_id = None

        else:

            status = (
                "UNRESOLVED"
            )

            person_id = None

        resolution_rows.append(
            {
                "person_url":
                    url,

                "resolution_status":
                    status,

                "person_id":
                    person_id,

                "used_by_frozen_startup":
                    url
                    in
                    focal_founder_url_set,
            }
        )

    resolution_df = pd.DataFrame(
        resolution_rows
    )

    resolution_df.to_parquet(
        URL_RESOLUTION_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 7. Second Companies pass: exact resolved Company -> Person edges
    # =========================================================================

    separator("-")

    print(
        "8.2.1B.7 — MATERIALIZE RESOLVED COMPANY -> PERSON EDGES"
    )

    separator("-")

    edge_parts = []

    total_parsed_url_references = 0
    total_exact_edge_rows_before_dedup = 0

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
            usecols=COMPANY_USECOLS,
            dtype={
                "id":
                    "string",
                "founder_links":
                    "string",
                "founded_on":
                    "string",
                "categories":
                    "string",
                "category_groups":
                    "string",
            },
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):

            chunk[
                "company_id"
            ] = (
                chunk[
                    "id"
                ]
                .astype(str)
                .str.strip()
            )

            chunk = chunk.loc[
                chunk[
                    "founder_links"
                ].notna()
            ].copy()

            if chunk.empty:
                continue

            chunk[
                "person_url"
            ] = (
                chunk[
                    "founder_links"
                ]
                .map(
                    extract_person_urls
                )
            )

            exploded = (
                chunk[
                    [
                        "company_id",
                        "founded_on",
                        "categories",
                        "category_groups",
                        "person_url",
                    ]
                ]
                .explode(
                    "person_url"
                )
                .dropna(
                    subset=[
                        "person_url"
                    ]
                )
                .copy()
            )

            total_parsed_url_references += int(
                len(
                    exploded
                )
            )

            exploded[
                "person_id"
            ] = (
                exploded[
                    "person_url"
                ]
                .map(
                    exact_map
                )
            )

            resolved = exploded.loc[
                exploded[
                    "person_id"
                ].notna()
            ].copy()

            if resolved.empty:
                continue

            resolved[
                "company_founded_on_raw"
            ] = (
                resolved[
                    "founded_on"
                ]
                .map(
                    normalize_scalar
                )
            )

            resolved[
                "company_founded_on"
            ] = pd.to_datetime(
                resolved[
                    "company_founded_on_raw"
                ],
                errors="coerce",
            )

            resolved[
                "categories_raw"
            ] = (
                resolved[
                    "categories"
                ]
                .map(
                    normalize_scalar
                )
            )

            resolved[
                "category_groups_raw"
            ] = (
                resolved[
                    "category_groups"
                ]
                .map(
                    normalize_scalar
                )
            )

            resolved[
                "source_file"
            ] = path.name

            resolved = resolved[
                [
                    "company_id",
                    "person_id",
                    "person_url",
                    "company_founded_on",
                    "company_founded_on_raw",
                    "categories_raw",
                    "category_groups_raw",
                    "source_file",
                ]
            ]

            total_exact_edge_rows_before_dedup += int(
                len(
                    resolved
                )
            )

            edge_parts.append(
                resolved
            )

    require(
        len(edge_parts)
        >
        0,
        "No exact Company-Founder edges materialized.",
    )

    company_founders_raw = pd.concat(
        edge_parts,
        ignore_index=True,
    )

    print()

    print(
        f"Parsed Company-founder URL references: "
        f"{total_parsed_url_references:,}"
    )

    print(
        f"Exact edge rows before dedup:           "
        f"{len(company_founders_raw):,}"
    )

    # =========================================================================
    # 8. Duplicate edge consistency audit
    # =========================================================================

    separator("-")

    print(
        "8.2.1B.8 — COMPANY-FOUNDER EDGE DUPLICATE CONSISTENCY"
    )

    separator("-")

    edge_key = [
        "company_id",
        "person_id",
    ]

    duplicate_edge_mask = (
        company_founders_raw
        .duplicated(
            subset=edge_key,
            keep=False,
        )
    )

    duplicate_edge_rows = int(
        duplicate_edge_mask.sum()
    )

    duplicate_edge_keys = int(
        company_founders_raw.loc[
            duplicate_edge_mask,
            edge_key,
        ]
        .drop_duplicates()
        .shape[0]
    )

    duplicate_audit_rows = []

    for field in [
        "person_url",
        "company_founded_on",
        "categories_raw",
        "category_groups_raw",
    ]:

        conflicts = int(
            company_founders_raw.loc[
                duplicate_edge_mask
            ]
            .groupby(
                edge_key,
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

        duplicate_audit_rows.append(
            {
                "field":
                    field,

                "duplicate_edge_keys":
                    duplicate_edge_keys,

                "conflicting_edge_keys":
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
            f"{field:<26} "
            f"conflicting edge keys = {conflicts:,}"
        )

    duplicate_audit_df = pd.DataFrame(
        duplicate_audit_rows
    )

    require(
        duplicate_audit_df[
            "conflicting_edge_keys"
        ]
        .eq(
            0
        )
        .all(),
        (
            "Duplicate Company-Founder edges contain "
            "conflicting source attributes."
        ),
    )

    # Prefer the most complete duplicate row.
    company_founders_raw[
        "_completeness"
    ] = (
        company_founders_raw[
            [
                "company_founded_on",
                "categories_raw",
                "category_groups_raw",
            ]
        ]
        .notna()
        .sum(
            axis=1
        )
    )

    company_founders = (
        company_founders_raw
        .sort_values(
            edge_key
            +
            [
                "_completeness",
                "source_file",
            ],
            ascending=[
                True,
                True,
                False,
                True,
            ],
            kind="mergesort",
        )
        .drop_duplicates(
            subset=edge_key,
            keep="first",
        )
        .drop(
            columns=[
                "_completeness"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    require(
        not company_founders.duplicated(
            subset=edge_key
        ).any(),
        "Company-Founder edge key remains duplicated.",
    )

    print()

    print(
        f"Duplicate raw edge rows: "
        f"{duplicate_edge_rows:,}"
    )

    print(
        f"Duplicate edge keys:     "
        f"{duplicate_edge_keys:,}"
    )

    print(
        f"Canonical exact edges:   "
        f"{len(company_founders):,}"
    )

    company_founders[
        "company_is_frozen_startup"
    ] = (
        company_founders[
            "company_id"
        ]
        .isin(
            startup_id_set
        )
    )

    company_founders.to_parquet(
        COMPANY_FOUNDER_EDGE_OUTPUT,
        index=False,
    )

    duplicate_audit_df.to_csv(
        EDGE_DUPLICATE_AUDIT_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 9. Frozen Startup -> Founder Person edges
    # =========================================================================

    separator("-")

    print(
        "8.2.1B.9 — FROZEN STARTUP -> FOUNDER PERSON EDGES"
    )

    separator("-")

    startup_identity = startups[
        [
            "startup_id",
            "startup_node_index",
            "startup_node_id",
            "founded_on",
        ]
    ].rename(
        columns={
            "founded_on":
                "focal_founded_on",
        }
    )

    startup_founders = (
        company_founders.loc[
            company_founders[
                "company_is_frozen_startup"
            ],
            [
                "company_id",
                "person_id",
                "person_url",
            ],
        ]
        .rename(
            columns={
                "company_id":
                    "startup_id",
            }
        )
        .merge(
            startup_identity,
            on="startup_id",
            how="inner",
            validate="many_to_one",
        )
        .drop_duplicates(
            subset=[
                "startup_id",
                "person_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    resolved_startup_count = int(
        startup_founders[
            "startup_id"
        ].nunique()
    )

    require(
        resolved_startup_count
        ==
        EXPECTED_STARTUPS_WITH_RESOLVED_FOUNDER,
        (
            "Resolved Startup-Founder coverage differs "
            "from frozen Phase-8.1 evidence. "
            f"Expected {EXPECTED_STARTUPS_WITH_RESOLVED_FOUNDER:,}; "
            f"found {resolved_startup_count:,}."
        ),
    )

    startup_founders[
        "relation_type"
    ] = "founded_by"

    startup_founders.to_parquet(
        STARTUP_FOUNDER_EDGE_OUTPUT,
        index=False,
    )

    print(
        f"Resolved Startup-Founder edges: "
        f"{len(startup_founders):,}"
    )

    print(
        f"Startups with >=1 Person:       "
        f"{resolved_startup_count:,}"
    )

    # =========================================================================
    # 10. Exact Person graph-node universe
    # =========================================================================

    separator("-")

    print(
        "8.2.1B.10 — PERSON GRAPH-NODE SOURCE"
    )

    separator("-")

    person_nodes = (
        startup_founders[
            [
                "person_id",
                "person_url",
            ]
        ]
        .drop_duplicates(
            subset=[
                "person_id"
            ]
        )
        .sort_values(
            "person_id",
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    person_nodes[
        "person_source_index"
    ] = np.arange(
        len(person_nodes),
        dtype=np.int64,
    )

    person_nodes[
        "identity_trainable"
    ] = False

    person_nodes[
        "initial_feature_policy"
    ] = "ZERO_40"

    person_nodes = person_nodes[
        [
            "person_source_index",
            "person_id",
            "person_url",
            "identity_trainable",
            "initial_feature_policy",
        ]
    ]

    person_nodes.to_parquet(
        PERSON_NODE_SOURCE_OUTPUT,
        index=False,
    )

    print(
        f"Person nodes for G_PRIMARY: "
        f"{len(person_nodes):,}"
    )

    print(
        "Person identity embeddings: NO"
    )

    print(
        "Person initial features:    ZERO_40"
    )

    # =========================================================================
    # 11. Build pre-focal founder-history candidate pairs
    # =========================================================================

    separator("-")

    print(
        "8.2.1B.11 — PRE-FOCAL FOUNDER-HISTORY PAIRS"
    )

    separator("-")

    focal_edges = (
        startup_founders.loc[
            startup_founders[
                "focal_founded_on"
            ].notna(),
            [
                "startup_id",
                "person_id",
                "focal_founded_on",
            ],
        ]
        .drop_duplicates()
    )

    prior_edges = (
        company_founders.loc[
            company_founders[
                "company_founded_on"
            ].notna(),
            [
                "company_id",
                "person_id",
                "company_founded_on",
                "categories_raw",
                "category_groups_raw",
                "company_is_frozen_startup",
            ],
        ]
        .rename(
            columns={
                "company_id":
                    "prior_company_id",

                "company_founded_on":
                    "prior_company_founded_on",

                "categories_raw":
                    "prior_categories_raw",

                "category_groups_raw":
                    "prior_category_groups_raw",

                "company_is_frozen_startup":
                    "prior_company_is_frozen_startup",
            }
        )
    )

    # Shared Person join.
    history_candidates = (
        focal_edges
        .merge(
            prior_edges,
            on="person_id",
            how="inner",
            validate="many_to_many",
        )
    )

    history_candidates = history_candidates.loc[
        (
            history_candidates[
                "prior_company_id"
            ]
            !=
            history_candidates[
                "startup_id"
            ]
        )
        &
        (
            history_candidates[
                "prior_company_founded_on"
            ]
            <
            history_candidates[
                "focal_founded_on"
            ]
        )
    ].copy()

    require(
        not (
            history_candidates[
                "prior_company_id"
            ]
            ==
            history_candidates[
                "startup_id"
            ]
        ).any(),
        "Focal Startup leaked into founder history.",
    )

    require(
        (
            history_candidates[
                "prior_company_founded_on"
            ]
            <
            history_candidates[
                "focal_founded_on"
            ]
        ).all(),
        "Non-pre-focal Company entered founder history.",
    )

    # -------------------------------------------------------------------------
    # A prior Company can be connected to the focal Startup through multiple
    # shared founders. Count the Company ONCE for founder_prior_startup_count.
    # -------------------------------------------------------------------------

    prior_pairs = (
        history_candidates
        .groupby(
            [
                "startup_id",
                "prior_company_id",
            ],
            observed=True,
            as_index=False,
        )
        .agg(
            focal_founded_on=(
                "focal_founded_on",
                "first",
            ),

            prior_company_founded_on=(
                "prior_company_founded_on",
                "first",
            ),

            shared_founder_count=(
                "person_id",
                "nunique",
            ),

            prior_categories_raw=(
                "prior_categories_raw",
                first_nonmissing,
            ),

            prior_category_groups_raw=(
                "prior_category_groups_raw",
                first_nonmissing,
            ),

            prior_company_is_frozen_startup=(
                "prior_company_is_frozen_startup",
                "first",
            ),
        )
    )

    require(
        not prior_pairs.duplicated(
            subset=[
                "startup_id",
                "prior_company_id",
            ]
        ).any(),
        "Prior Company pairs are not unique.",
    )

    require(
        (
            prior_pairs[
                "prior_company_founded_on"
            ]
            <
            prior_pairs[
                "focal_founded_on"
            ]
        ).all(),
        "Prior-company temporal invariant failed.",
    )

    prior_pairs[
        "prior_company_external_context_only"
    ] = (
        ~prior_pairs[
            "prior_company_is_frozen_startup"
        ]
    )

    prior_pairs.to_parquet(
        PRIOR_COMPANY_PAIR_OUTPUT,
        index=False,
    )

    print(
        f"Unique focal/prior-company pairs: "
        f"{len(prior_pairs):,}"
    )

    print(
        f"Focal Startups with prior history: "
        f"{prior_pairs['startup_id'].nunique():,}"
    )

    print(
        "Cutoff rule: prior founded_on < focal founded_on"
    )

    print(
        "Focal Startup excluded: YES"
    )

    # =========================================================================
    # 12. Materialize founder-history source per frozen Startup
    # =========================================================================

    separator("-")

    print(
        "8.2.1B.12 — ONE-ROW-PER-STARTUP FOUNDER-HISTORY SOURCE"
    )

    separator("-")

    founder_count = (
        startup_founders
        .groupby(
            "startup_id",
            observed=True,
        )[
            "person_id"
        ]
        .nunique()
        .rename(
            "startup_founder_count_resolved"
        )
    )

    prior_count = (
        prior_pairs
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

    prior_sector_available = (
        prior_pairs.assign(
            _has_sector=(
                prior_pairs[
                    "prior_categories_raw"
                ].notna()
                |
                prior_pairs[
                    "prior_category_groups_raw"
                ].notna()
            )
        )
        .groupby(
            "startup_id",
            observed=True,
        )[
            "_has_sector"
        ]
        .any()
        .rename(
            "founder_prior_sector_source_available"
        )
    )

    history_source = (
        startups[
            [
                "startup_node_index",
                "startup_node_id",
                "startup_id",
                "founded_on",
            ]
        ]
        .rename(
            columns={
                "founded_on":
                    "focal_founded_on",
            }
        )
        .merge(
            founder_count,
            on="startup_id",
            how="left",
        )
        .merge(
            prior_count,
            on="startup_id",
            how="left",
        )
        .merge(
            prior_sector_available,
            on="startup_id",
            how="left",
        )
    )

    history_source[
        "startup_founder_count_resolved"
    ] = (
        history_source[
            "startup_founder_count_resolved"
        ]
        .fillna(0)
        .astype(
            "int32"
        )
    )

    history_source[
        "founder_prior_startup_count"
    ] = (
        history_source[
            "founder_prior_startup_count"
        ]
        .fillna(0)
        .astype(
            "int32"
        )
    )

    history_source[
        "founder_prior_sector_source_available"
    ] = (
        history_source[
            "founder_prior_sector_source_available"
        ]
        .fillna(False)
        .astype(bool)
    )

    history_source[
        "prefocal_history_temporally_defined"
    ] = (
        history_source[
            "focal_founded_on"
        ]
        .notna()
    )

    history_source[
        "has_resolved_founder"
    ] = (
        history_source[
            "startup_founder_count_resolved"
        ]
        >
        0
    )

    history_source[
        "has_prefocal_founder_history"
    ] = (
        history_source[
            "founder_prior_startup_count"
        ]
        >
        0
    )

    # Missing focal founding date => frozen missing-history behavior = zero.
    undefined_history_rows = (
        ~history_source[
            "prefocal_history_temporally_defined"
        ]
    )

    require(
        history_source.loc[
            undefined_history_rows,
            "founder_prior_startup_count",
        ]
        .eq(0)
        .all(),
        (
            "Startup without focal founded_on received "
            "pre-focal founder history."
        ),
    )

    require(
        len(history_source)
        ==
        EXPECTED_STARTUPS,
        "Founder-history source row count mismatch.",
    )

    history_source.to_parquet(
        FOUNDER_HISTORY_OUTPUT,
        index=False,
    )

    print(
        f"Founder-history rows:        "
        f"{len(history_source):,}"
    )

    print(
        f"Resolved-founder Startups:   "
        f"{int(history_source['has_resolved_founder'].sum()):,}"
    )

    print(
        f"Startups with prior history: "
        f"{int(history_source['has_prefocal_founder_history'].sum()):,}"
    )

    # =========================================================================
    # 13. Coverage / provenance audit
    # =========================================================================

    separator("-")

    print(
        "8.2.1B.13 — COVERAGE / PROVENANCE AUDIT"
    )

    separator("-")

    coverage_rows = [
        {
            "metric":
                "frozen_startups",

            "value":
                EXPECTED_STARTUPS,
        },

        {
            "metric":
                "startups_with_raw_founder_links",

            "value":
                startups_with_founder_links,
        },

        {
            "metric":
                "startups_with_resolved_founder_person",

            "value":
                resolved_startup_count,
        },

        {
            "metric":
                "person_nodes_for_primary_graph",

            "value":
                len(
                    person_nodes
                ),
        },

        {
            "metric":
                "resolved_all_company_founder_edges",

            "value":
                len(
                    company_founders
                ),
        },

        {
            "metric":
                "prefocal_startup_prior_company_pairs",

            "value":
                len(
                    prior_pairs
                ),
        },

        {
            "metric":
                "startups_with_prefocal_founder_history",

            "value":
                int(
                    history_source[
                        "has_prefocal_founder_history"
                    ].sum()
                ),
        },

        {
            "metric":
                "startups_with_prefocal_sector_source",

            "value":
                int(
                    history_source[
                        "founder_prior_sector_source_available"
                    ].sum()
                ),
        },

        {
            "metric":
                "external_context_prior_company_pairs",

            "value":
                int(
                    prior_pairs[
                        "prior_company_external_context_only"
                    ].sum()
                ),
        },
    ]

    coverage_df = pd.DataFrame(
        coverage_rows
    )

    coverage_df.to_csv(
        COVERAGE_OUTPUT,
        index=False,
    )

    print(
        coverage_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # 14. Resolution audit
    # =========================================================================

    resolution_audit = pd.DataFrame(
        [
            {
                "population":
                    "ALL_REQUIRED_FOUNDER_URLS",

                "urls":
                    len(
                        all_founder_urls
                    ),

                "exact_unique":
                    len(
                        unique_urls
                    ),

                "ambiguous":
                    len(
                        ambiguous_urls
                    ),

                "unresolved":
                    len(
                        unresolved_urls
                    ),

                "exact_rate":
                    (
                        len(
                            unique_urls
                        )
                        /
                        len(
                            all_founder_urls
                        )
                    ),
            },

            {
                "population":
                    "FROZEN_STARTUP_FOUNDER_URLS",

                "urls":
                    len(
                        focal_founder_url_set
                    ),

                "exact_unique":
                    len(
                        focal_exact_urls
                    ),

                "ambiguous":
                    len(
                        focal_ambiguous_urls
                    ),

                "unresolved":
                    len(
                        focal_unresolved_urls
                    ),

                "exact_rate":
                    focal_resolution_rate,
            },
        ]
    )

    resolution_audit.to_csv(
        RESOLUTION_AUDIT_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 15. Leakage / identity boundary
    # =========================================================================

    separator("-")

    print(
        "8.2.1B.15 — LEAKAGE / IDENTITY BOUNDARY"
    )

    separator("-")

    boundary = {
        "external_companies_used_as_candidates":
            False,

        "external_companies_added_as_graph_nodes":
            False,

        "external_companies_used_for_history_aggregation":
            True,

        "person_identity_embedding":
            False,

        "person_current_snapshot_features_loaded":
            False,

        "people_current_organization_loaded":
            False,

        "people_current_investment_counts_loaded":
            False,

        "people_current_exit_counts_loaded":
            False,

        "raw_people_num_founded_organizations_loaded":
            False,

        "focal_startup_investment_history_used":
            False,

        "history_cutoff":
            (
                "prior_company.founded_on "
                "< focal_startup.founded_on"
            ),

        "prediction_origin_used_for_founder_history_cutoff":
            False,

        "sector_labels":
            (
                "raw matched-snapshot source only; "
                "vocabulary not yet fit"
            ),

        "learned_preprocessing_fit":
            False,

        "evaluation_candidates_loaded":
            False,

        "performance_inspected":
            False,
    }

    for key, value in boundary.items():

        print(
            f"{key:<48} {value}"
        )

    # =========================================================================
    # 16. Persist manifest
    # =========================================================================

    separator("-")

    print(
        "8.2.1B.16 — PERSIST FOUNDER-HISTORY MATERIALIZATION"
    )

    separator("-")

    manifest = {
        "schema_version":
            "PHASE_8_2_1B_FOUNDER_HISTORY_SOURCE_V1",

        "phase":
            "8.2.1B",

        "status":
            "MATERIALIZED_AND_AUDITED",

        "scientific_role":
            (
                "IDENTITY_RESOLVED_FOUNDER_HISTORY_SOURCE"
            ),

        "startup_rows":
            EXPECTED_STARTUPS,

        "startups_with_founder_links":
            startups_with_founder_links,

        "startups_with_resolved_founder":
            resolved_startup_count,

        "person_graph_nodes":
            int(
                len(
                    person_nodes
                )
            ),

        "resolved_company_founder_edges":
            int(
                len(
                    company_founders
                )
            ),

        "prefocal_prior_company_pairs":
            int(
                len(
                    prior_pairs
                )
            ),

        "startups_with_prefocal_history":
            int(
                history_source[
                    "has_prefocal_founder_history"
                ].sum()
            ),

        "founder_history_cutoff":
            (
                "prior_company.founded_on "
                "< focal_startup.founded_on"
            ),

        "focal_startup_excluded":
            True,

        "prior_company_deduplicated_across_founders":
            True,

        "external_company_context_allowed":
            True,

        "external_company_candidate_expansion":
            False,

        "person_identity_embedding":
            False,

        "person_initial_feature_policy":
            "ZERO_40",

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
            "url_resolution":
                str(
                    URL_RESOLUTION_OUTPUT
                ),

            "company_founder_edges":
                str(
                    COMPANY_FOUNDER_EDGE_OUTPUT
                ),

            "startup_founder_edges":
                str(
                    STARTUP_FOUNDER_EDGE_OUTPUT
                ),

            "person_node_source":
                str(
                    PERSON_NODE_SOURCE_OUTPUT
                ),

            "prefocal_prior_company_pairs":
                str(
                    PRIOR_COMPANY_PAIR_OUTPUT
                ),

            "founder_history_source":
                str(
                    FOUNDER_HISTORY_OUTPUT
                ),
        },

        "next_step":
            (
                "Phase 8.2.1C — origin-specific "
                "training-only preprocessing fit and "
                "inductive feature matrix materialization."
            ),
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    outputs = [
        URL_RESOLUTION_OUTPUT,
        COMPANY_FOUNDER_EDGE_OUTPUT,
        STARTUP_FOUNDER_EDGE_OUTPUT,
        PERSON_NODE_SOURCE_OUTPUT,
        PRIOR_COMPANY_PAIR_OUTPUT,
        FOUNDER_HISTORY_OUTPUT,
        RESOLUTION_AUDIT_OUTPUT,
        EDGE_DUPLICATE_AUDIT_OUTPUT,
        COVERAGE_OUTPUT,
        MANIFEST_OUTPUT,
    ]

    write_json(
        HASH_OUTPUT,
        {
            str(path):
                sha256_file(path)
            for path in outputs
        },
    )

    separator()

    print(
        "PHASE 8.2.1B FOUNDER IDENTITY + "
        "PRE-FOCAL HISTORY MATERIALIZED"
    )

    separator()

    print()

    print(
        f"STARTUPS WITH RAW FOUNDER LINKS: "
        f"{startups_with_founder_links:,}"
    )

    print(
        f"STARTUPS WITH RESOLVED PERSON:    "
        f"{resolved_startup_count:,}"
    )

    print(
        f"PERSON GRAPH NODES:               "
        f"{len(person_nodes):,}"
    )

    print(
        f"ALL COMPANY-FOUNDER EDGES:        "
        f"{len(company_founders):,}"
    )

    print(
        f"PRE-FOCAL PRIOR COMPANY PAIRS:    "
        f"{len(prior_pairs):,}"
    )

    print(
        f"STARTUPS WITH PRIOR HISTORY:       "
        f"{int(history_source['has_prefocal_founder_history'].sum()):,}"
    )

    print()

    print(
        "FOUNDER HISTORY CUTOFF:"
    )

    print(
        "  prior_company.founded_on "
        "< focal_startup.founded_on"
    )

    print()

    print(
        "FOCAL STARTUP EXCLUDED:             YES"
    )

    print(
        "PRIOR COMPANY DEDUP ACROSS FOUNDERS:YES"
    )

    print(
        "EXTERNAL COMPANY CANDIDATES ADDED:  NO"
    )

    print(
        "PERSON ID EMBEDDINGS:               NO"
    )

    print(
        "PERSON INITIAL FEATURES:             ZERO_40"
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
        "Phase 8.2.1C — origin-specific "
        "training-only preprocessing and "
        "feature-matrix materialization."
    )


if __name__ == "__main__":
    main()
