from pathlib import Path
from collections import defaultdict
import hashlib
import json
import re

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# PHASE 8.1.6D — EXPLICIT FOUNDER INCREMENTAL COVERAGE AUDIT V2
# =============================================================================
#
# Scientific role:
#
#   DEVELOPMENT-ONLY HETEROGENEOUS CONTEXT COVERAGE AUDIT
#
# PURPOSE
# -------
#
# Compare, case-by-case:
#
#   1. existing acquisition-only structural connectivity;
#   2. existing founder-proxy structural connectivity;
#   3. explicit Startup -> Founder Person raw context availability;
#   4. union of founder-proxy + explicit founder Person context.
#
# CENTRAL QUESTION
# ----------------
#
# How many primary cold-start startups obtain founder information through
# explicit Person nodes that is lost by the projected shared-founder graph?
#
# IMPORTANT
# ---------
#
# Explicit founder-Person availability remains snapshot-unversioned.
# This script DOES NOT approve that relation for confirmatory modeling.
#
# April is not analyzed.
#
# =============================================================================


RAW_DIR = Path("data/raw")

NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/"
    "model_ready/"
    "node_index.parquet"
)

EDGE_INDEX_PATH = Path(
    "data/experimental/phase_3/"
    "model_ready/"
    "edge_index.npy"
)

GRAPH_MASKS_PATH = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_6_dual_track_graph_policy/"
    "phase_8_0_6_dual_track_graph_masks_V1.npz"
)

TARGETS_PATH = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_9_origin_eligibility_candidate_universe/"
    "phase_8_0_9_origin_eligible_targets_V5.parquet"
)

PHASE816C_SUMMARY = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_6_data_foundations/"
    "phase_8_1_6c_founder_people_school_eda/"
    "phase_8_1_6c_eda_summary_V1.json"
)


OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_6_data_foundations/"
    "phase_8_1_6d_explicit_founder_incremental_coverage"
)

CHART_DIR = OUTPUT_DIR / "charts"

SUMMARY_TABLE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6d_incremental_founder_coverage_V2.csv"
)

OVERLAP_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6d_founder_context_overlap_V2.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6d_summary_V2.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6d_analysis_manifest_V2.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6d_derived_artifact_sha256_V2.json"
)


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


def pct(num, den):

    if den == 0:
        return float("nan")

    return 100.0 * float(num) / float(den)


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


def normalize_url_value(value):

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

        url = normalize_url_value(
            url
        )

        if url is not None:
            urls.append(url)

    return list(
        dict.fromkeys(urls)
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    separator()

    print(
        "PHASE 8.1.6D — "
        "EXPLICIT FOUNDER INCREMENTAL COVERAGE AUDIT V2"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "DEVELOPMENT-ONLY CONTEXT COVERAGE AUDIT"
    )

    print(
        "Development folds:           T57 / T58 / T59"
    )

    print(
        "April analyzed:              NO"
    )

    print(
        "Relation approved:           NO"
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
    # 1. Gates
    # =========================================================================

    separator("-")

    print(
        "8.1.6D.1 — PREREQUISITE GATES"
    )

    separator("-")

    for path in [
        NODE_INDEX_PATH,
        EDGE_INDEX_PATH,
        GRAPH_MASKS_PATH,
        TARGETS_PATH,
        PHASE816C_SUMMARY,
    ]:

        require(
            path.exists(),
            f"Missing prerequisite artifact: {path}",
        )

    previous = read_json(
        PHASE816C_SUMMARY
    )

    require(
        previous.get(
            "status"
        )
        ==
        "COMPLETE",
        "Phase 8.1.6C is not complete.",
    )

    require(
        previous.get(
            "relation_selected"
        )
        is False,
        "A relation was unexpectedly selected in 8.1.6C.",
    )

    require(
        previous.get(
            "April_relation_EDA"
        )
        is False,
        "April relation EDA unexpectedly occurred.",
    )

    print(
        "Phase 8.1.6C: COMPLETE PASS"
    )

    # =========================================================================
    # 2. Development primary cases
    # =========================================================================

    separator("-")

    print(
        "8.1.6D.2 — DEVELOPMENT PRIMARY CASES"
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

    primary_ids_by_origin = {}

    all_primary_ids = set()

    for origin in DEVELOPMENT_ORIGINS:

        ids = (
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
            len(ids)
            ==
            EXPECTED_PRIMARY[
                origin
            ],
            f"{origin}: primary count drift.",
        )

        primary_ids_by_origin[
            origin
        ] = ids

        all_primary_ids.update(
            ids
        )

        print(
            f"{origin}: {len(ids):,} PASS"
        )

    print(
        f"Unique development-primary Startup IDs: "
        f"{len(all_primary_ids):,}"
    )

    # =========================================================================
    # 3. Startup node-index map
    # =========================================================================

    separator("-")

    print(
        "8.1.6D.3 — STARTUP NODE MAP"
    )

    separator("-")

    nodes = pd.read_parquet(
        NODE_INDEX_PATH,
        columns=[
            "node_index",
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

    startup_id_to_node = (
        startup_nodes
        .set_index(
            "raw_entity_id"
        )[
            "node_index"
        ]
        .astype(np.int64)
        .to_dict()
    )

    missing_nodes = (
        all_primary_ids
        -
        set(
            startup_id_to_node
        )
    )

    require(
        not missing_nodes,
        (
            f"{len(missing_nodes):,} primary startups "
            "missing from node index."
        ),
    )

    print(
        "All primary startups mapped to graph nodes: PASS"
    )

    # =========================================================================
    # 4. Existing graph connectivity
    # =========================================================================

    separator("-")

    print(
        "8.1.6D.4 — EXISTING GRAPH CONNECTIVITY"
    )

    separator("-")

    edge_index = np.load(
        EDGE_INDEX_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    if (
        edge_index.ndim == 2
        and
        edge_index.shape[0] == 2
    ):

        src_all = np.asarray(
            edge_index[0],
            dtype=np.int64,
        )

        dst_all = np.asarray(
            edge_index[1],
            dtype=np.int64,
        )

    elif (
        edge_index.ndim == 2
        and
        edge_index.shape[1] == 2
    ):

        src_all = np.asarray(
            edge_index[:, 0],
            dtype=np.int64,
        )

        dst_all = np.asarray(
            edge_index[:, 1],
            dtype=np.int64,
        )

    else:

        raise ValueError(
            f"Unexpected edge_index shape: "
            f"{edge_index.shape}"
        )

    masks = np.load(
        GRAPH_MASKS_PATH,
        allow_pickle=False,
    )

    connected_sets = {}

    for origin in DEVELOPMENT_ORIGINS:

        for track in [
            "strict_temporal",
            "founder_proxy",
        ]:

            key = (
                f"{origin}_{track}"
            )

            require(
                key in masks.files,
                f"Missing graph mask: {key}",
            )

            mask = np.asarray(
                masks[
                    key
                ],
                dtype=bool,
            )

            require(
                len(mask)
                ==
                len(src_all),
                f"{key}: mask length mismatch.",
            )

            active_nodes = set(
                np.unique(
                    np.concatenate(
                        [
                            src_all[
                                mask
                            ],
                            dst_all[
                                mask
                            ],
                        ]
                    )
                ).tolist()
            )

            connected_sets[
                (
                    origin,
                    track,
                )
            ] = active_nodes

            print(
                f"{origin} / {track:<15} "
                f"active nodes={len(active_nodes):,}"
            )

    # =========================================================================
    # 5. Extract founder URLs only for development-primary startups
    # =========================================================================

    separator("-")

    print(
        "8.1.6D.5 — PRIMARY STARTUP "
        "FOUNDER URL EXTRACTION"
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
                    all_primary_ids
                )
            ]

            matched_rows += len(
                subset
            )

            for startup_id, raw_links in zip(
                subset[
                    "id"
                ],
                subset[
                    "founder_links"
                ],
            ):

                for url in extract_person_urls(
                    raw_links
                ):

                    startup_founder_urls[
                        str(
                            startup_id
                        )
                    ].add(
                        url
                    )

        print(
            f"[{i:02d}/24] "
            f"{path.name:<24} "
            f"primary rows={matched_rows:,}"
        )

    founder_urls_needed = set()

    for urls in (
        startup_founder_urls.values()
    ):

        founder_urls_needed.update(
            urls
        )

    print()

    print(
        f"Primary startups with founder URLs: "
        f"{sum(bool(v) for v in startup_founder_urls.values()):,}"
    )

    print(
        f"Unique founder URLs needed:          "
        f"{len(founder_urls_needed):,}"
    )

    # =========================================================================
    # 6. Resolve exact People URLs
    # =========================================================================

    separator("-")

    print(
        "8.1.6D.6 — EXACT FOUNDER URL "
        "RESOLUTION"
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

    url_to_ids = defaultdict(
        set
    )

    for i, path in enumerate(
        people_files,
        start=1,
    ):

        matched = 0

        for chunk in pd.read_csv(
            path,
            usecols=[
                "id",
                "link",
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

            normalized = (
                chunk[
                    "link"
                ]
                .map(
                    normalize_url_value
                )
            )

            relevant = normalized.isin(
                founder_urls_needed
            )

            if not relevant.any():
                continue

            subset = chunk.loc[
                relevant,
                [
                    "id",
                ],
            ].copy()

            subset[
                "normalized_link"
            ] = normalized[
                relevant
            ].to_numpy()

            matched += len(
                subset
            )

            for url, person_id in zip(
                subset[
                    "normalized_link"
                ],
                subset[
                    "id"
                ],
            ):

                if (
                    pd.notna(url)
                    and
                    pd.notna(person_id)
                ):

                    url_to_ids[
                        str(url)
                    ].add(
                        str(person_id)
                    )

        print(
            f"[{i:02d}/11] "
            f"{path.name:<20} "
            f"matched={matched:,}"
        )

    unique_url_to_person = {
        url:
            next(
                iter(ids)
            )
        for url, ids
        in url_to_ids.items()
        if len(ids) == 1
    }

    explicit_person_startups = set()

    for startup_id, urls in (
        startup_founder_urls.items()
    ):

        if any(
            url
            in
            unique_url_to_person
            for url
            in urls
        ):

            explicit_person_startups.add(
                startup_id
            )

    print()

    print(
        f"Primary Startup IDs with >=1 "
        f"resolved founder Person: "
        f"{len(explicit_person_startups):,}"
    )

    # =========================================================================
    # 7. Fold-by-fold incremental coverage
    # =========================================================================

    separator("-")

    print(
        "8.1.6D.7 — INCREMENTAL CONTEXT "
        "COVERAGE — CASE LEVEL"
    )

    separator("-")

    # ---------------------------------------------------------------------
    # Authoritative reconciliation targets from prior completed audits.
    #
    # Phase 8.1.3:
    #   acquisition-only / founder-proxy CASE connectivity
    #
    # Phase 8.1.6C:
    #   explicit founder-Person CASE coverage
    #
    # These gates guarantee that V2 uses the same unit of analysis.
    # ---------------------------------------------------------------------

    EXPECTED_ACQUISITION_CONNECTED_CASES = {
        "T57": 114,
        "T58": 100,
        "T59": 146,
    }

    EXPECTED_FOUNDER_PROXY_CONNECTED_CASES = {
        "T57": 1_006,
        "T58": 947,
        "T59": 946,
    }

    EXPECTED_EXPLICIT_PERSON_CASES = {
        "T57": 3_379,
        "T58": 3_271,
        "T59": 2_817,
    }

    rows = []
    overlap_rows = []

    for origin in DEVELOPMENT_ORIGINS:

        # IMPORTANT:
        #
        # Keep `ids` as the full evaluation-case sequence.
        # Do NOT deduplicate for primary scientific coverage.
        #
        # The same Startup may legitimately occur in multiple investor
        # recommendation cases, and HR/NDCG are also evaluated case-wise.

        ids = primary_ids_by_origin[
            origin
        ]

        unique_ids = set(
            ids
        )

        # -------------------------------------------------------------
        # Unique-ID connectivity registries.
        #
        # These are used only as lookup sets. Scientific coverage below
        # is counted over the full case sequence `ids`.
        # -------------------------------------------------------------

        acquisition_connected_ids = {
            startup_id
            for startup_id
            in unique_ids
            if startup_id_to_node[
                startup_id
            ]
            in
            connected_sets[
                (
                    origin,
                    "strict_temporal",
                )
            ]
        }

        founder_proxy_connected_ids = {
            startup_id
            for startup_id
            in unique_ids
            if startup_id_to_node[
                startup_id
            ]
            in
            connected_sets[
                (
                    origin,
                    "founder_proxy",
                )
            ]
        }

        explicit_person_ids = (
            unique_ids
            &
            explicit_person_startups
        )

        # -------------------------------------------------------------
        # CASE-LEVEL booleans.
        # -------------------------------------------------------------

        acquisition_flags = [
            startup_id
            in
            acquisition_connected_ids
            for startup_id
            in ids
        ]

        founder_proxy_flags = [
            startup_id
            in
            founder_proxy_connected_ids
            for startup_id
            in ids
        ]

        explicit_person_flags = [
            startup_id
            in
            explicit_person_startups
            for startup_id
            in ids
        ]

        acquisition_cases = int(
            sum(
                acquisition_flags
            )
        )

        founder_proxy_cases = int(
            sum(
                founder_proxy_flags
            )
        )

        explicit_person_cases = int(
            sum(
                explicit_person_flags
            )
        )

        both_cases = int(
            sum(
                graph_flag
                and
                person_flag
                for graph_flag, person_flag
                in zip(
                    founder_proxy_flags,
                    explicit_person_flags,
                )
            )
        )

        person_only_cases = int(
            sum(
                (not graph_flag)
                and
                person_flag
                for graph_flag, person_flag
                in zip(
                    founder_proxy_flags,
                    explicit_person_flags,
                )
            )
        )

        graph_only_cases = int(
            sum(
                graph_flag
                and
                (not person_flag)
                for graph_flag, person_flag
                in zip(
                    founder_proxy_flags,
                    explicit_person_flags,
                )
            )
        )

        neither_cases = int(
            sum(
                (not graph_flag)
                and
                (not person_flag)
                for graph_flag, person_flag
                in zip(
                    founder_proxy_flags,
                    explicit_person_flags,
                )
            )
        )

        union_cases = (
            len(
                ids
            )
            -
            neither_cases
        )

        current_isolated_cases = (
            len(
                ids
            )
            -
            founder_proxy_cases
        )

        isolated_after_person_cases = (
            len(
                ids
            )
            -
            union_cases
        )

        recovered_from_current_isolation_cases = (
            person_only_cases
        )

        # -------------------------------------------------------------
        # Reconciliation with prior authoritative audits.
        # -------------------------------------------------------------

        require(
            acquisition_cases
            ==
            EXPECTED_ACQUISITION_CONNECTED_CASES[
                origin
            ],
            (
                f"{origin}: acquisition CASE connectivity "
                f"does not reconcile with Phase 8.1.3. "
                f"Observed={acquisition_cases:,}, "
                f"expected="
                f"{EXPECTED_ACQUISITION_CONNECTED_CASES[origin]:,}"
            ),
        )

        require(
            founder_proxy_cases
            ==
            EXPECTED_FOUNDER_PROXY_CONNECTED_CASES[
                origin
            ],
            (
                f"{origin}: founder-proxy CASE connectivity "
                f"does not reconcile with Phase 8.1.3. "
                f"Observed={founder_proxy_cases:,}, "
                f"expected="
                f"{EXPECTED_FOUNDER_PROXY_CONNECTED_CASES[origin]:,}"
            ),
        )

        require(
            explicit_person_cases
            ==
            EXPECTED_EXPLICIT_PERSON_CASES[
                origin
            ],
            (
                f"{origin}: explicit founder-Person CASE coverage "
                f"does not reconcile with Phase 8.1.6C. "
                f"Observed={explicit_person_cases:,}, "
                f"expected="
                f"{EXPECTED_EXPLICIT_PERSON_CASES[origin]:,}"
            ),
        )

        # -------------------------------------------------------------
        # Unique-startup diagnostics.
        #
        # Useful descriptively, but NOT mixed with the CASE denominator.
        # -------------------------------------------------------------

        both_unique = (
            founder_proxy_connected_ids
            &
            explicit_person_ids
        )

        person_only_unique = (
            explicit_person_ids
            -
            founder_proxy_connected_ids
        )

        graph_only_unique = (
            founder_proxy_connected_ids
            -
            explicit_person_ids
        )

        union_unique = (
            founder_proxy_connected_ids
            |
            explicit_person_ids
        )

        rows.append(
            {
                "origin":
                    origin,

                # -----------------------------------------------------
                # Primary scientific unit: CASES
                # -----------------------------------------------------

                "primary_cases":
                    len(
                        ids
                    ),

                "acquisition_connected":
                    acquisition_cases,

                "acquisition_connected_pct":
                    pct(
                        acquisition_cases,
                        len(
                            ids
                        ),
                    ),

                "current_founder_proxy_connected":
                    founder_proxy_cases,

                "current_founder_proxy_connected_pct":
                    pct(
                        founder_proxy_cases,
                        len(
                            ids
                        ),
                    ),

                "explicit_founder_person_context":
                    explicit_person_cases,

                "explicit_founder_person_context_pct":
                    pct(
                        explicit_person_cases,
                        len(
                            ids
                        ),
                    ),

                "both_current_graph_and_person":
                    both_cases,

                "incremental_person_only":
                    person_only_cases,

                "incremental_person_only_pct":
                    pct(
                        person_only_cases,
                        len(
                            ids
                        ),
                    ),

                "current_graph_only":
                    graph_only_cases,

                "potential_union_context":
                    union_cases,

                "potential_union_context_pct":
                    pct(
                        union_cases,
                        len(
                            ids
                        ),
                    ),

                "current_isolated":
                    current_isolated_cases,

                "potential_isolated_after_person":
                    isolated_after_person_cases,

                "recovered_from_current_isolation":
                    recovered_from_current_isolation_cases,

                "recovered_fraction_of_current_isolated_pct":
                    pct(
                        recovered_from_current_isolation_cases,
                        current_isolated_cases,
                    ),

                # -----------------------------------------------------
                # Secondary descriptive unit: UNIQUE STARTUPS
                # -----------------------------------------------------

                "unique_primary_startups":
                    len(
                        unique_ids
                    ),

                "unique_acquisition_connected":
                    len(
                        acquisition_connected_ids
                    ),

                "unique_founder_proxy_connected":
                    len(
                        founder_proxy_connected_ids
                    ),

                "unique_explicit_founder_person":
                    len(
                        explicit_person_ids
                    ),

                "unique_both_graph_and_person":
                    len(
                        both_unique
                    ),

                "unique_person_only_incremental":
                    len(
                        person_only_unique
                    ),

                "unique_graph_only":
                    len(
                        graph_only_unique
                    ),

                "unique_union_context":
                    len(
                        union_unique
                    ),
            }
        )

        overlap_rows.extend(
            [
                {
                    "origin":
                        origin,
                    "set":
                        "BOTH_CURRENT_GRAPH_AND_PERSON",
                    "case_count":
                        both_cases,
                    "unique_startup_count":
                        len(
                            both_unique
                        ),
                },
                {
                    "origin":
                        origin,
                    "set":
                        "PERSON_ONLY_INCREMENTAL",
                    "case_count":
                        person_only_cases,
                    "unique_startup_count":
                        len(
                            person_only_unique
                        ),
                },
                {
                    "origin":
                        origin,
                    "set":
                        "CURRENT_GRAPH_ONLY",
                    "case_count":
                        graph_only_cases,
                    "unique_startup_count":
                        len(
                            graph_only_unique
                        ),
                },
                {
                    "origin":
                        origin,
                    "set":
                        "NEITHER",
                    "case_count":
                        neither_cases,
                    "unique_startup_count":
                        (
                            len(
                                unique_ids
                            )
                            -
                            len(
                                union_unique
                            )
                        ),
                },
            ]
        )

        print()

        print(
            f"{origin}"
        )

        print(
            f"  Primary cases:                 "
            f"{len(ids):,}"
        )

        print(
            f"  Unique primary startups:       "
            f"{len(unique_ids):,}"
        )

        print(
            f"  Acquisition-only connected:    "
            f"{acquisition_cases:,} "
            f"({pct(acquisition_cases, len(ids)):.4f}%)"
        )

        print(
            f"  Current founder-proxy:         "
            f"{founder_proxy_cases:,} "
            f"({pct(founder_proxy_cases, len(ids)):.4f}%)"
        )

        print(
            f"  Explicit founder Person:       "
            f"{explicit_person_cases:,} "
            f"({pct(explicit_person_cases, len(ids)):.4f}%)"
        )

        print(
            f"  Both graph + Person:           "
            f"{both_cases:,} "
            f"({pct(both_cases, len(ids)):.4f}%)"
        )

        print(
            f"  Person-only incremental:       "
            f"{person_only_cases:,} "
            f"({pct(person_only_cases, len(ids)):.4f}%)"
        )

        print(
            f"  Potential graph+Person union:  "
            f"{union_cases:,} "
            f"({pct(union_cases, len(ids)):.4f}%)"
        )

        print(
            f"  Current isolated recovered:    "
            f"{recovered_from_current_isolation_cases:,}/"
            f"{current_isolated_cases:,} "
            f"({pct(recovered_from_current_isolation_cases, current_isolated_cases):.4f}%)"
        )

    result_df = pd.DataFrame(
        rows
    )

    overlap_df = pd.DataFrame(
        overlap_rows
    )

    # =========================================================================
    # 8. Aggregate interpretation
    # =========================================================================

    separator("-")

    print(
        "8.1.6D.8 — AGGREGATE INTERPRETATION"
    )

    separator("-")

    mean_current = float(
        result_df[
            "current_founder_proxy_connected_pct"
        ].mean()
    )

    mean_person = float(
        result_df[
            "explicit_founder_person_context_pct"
        ].mean()
    )

    mean_union = float(
        result_df[
            "potential_union_context_pct"
        ].mean()
    )

    mean_incremental = float(
        result_df[
            "incremental_person_only_pct"
        ].mean()
    )

    mean_recovery = float(
        result_df[
            "recovered_fraction_of_current_isolated_pct"
        ].mean()
    )

    print(
        f"Mean current founder-proxy coverage: "
        f"{mean_current:.4f}%"
    )

    print(
        f"Mean explicit founder-Person context: "
        f"{mean_person:.4f}%"
    )

    print(
        f"Mean incremental Person-only coverage: "
        f"{mean_incremental:.4f}%"
    )

    print(
        f"Mean potential union context: "
        f"{mean_union:.4f}%"
    )

    print(
        f"Mean fraction of currently isolated "
        f"cases recovered by Person context: "
        f"{mean_recovery:.4f}%"
    )

    # =========================================================================
    # 9. Persist + chart
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CHART_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        SUMMARY_TABLE_OUTPUT,
        index=False,
    )

    overlap_df.to_csv(
        OVERLAP_OUTPUT,
        index=False,
    )

    x = np.arange(
        len(
            result_df
        )
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width,
        result_df[
            "acquisition_connected_pct"
        ],
        width=width,
        label="Acquisition only",
    )

    ax.bar(
        x,
        result_df[
            "current_founder_proxy_connected_pct"
        ],
        width=width,
        label="Current founder-proxy",
    )

    ax.bar(
        x + width,
        result_df[
            "explicit_founder_person_context_pct"
        ],
        width=width,
        label="Explicit founder Person",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        result_df[
            "origin"
        ]
    )

    ax.set_ylabel(
        "Primary cold-start coverage (%)"
    )

    ax.set_title(
        "Existing structural connectivity vs explicit founder context"
    )

    ax.legend()

    fig.tight_layout()

    chart_path = (
        CHART_DIR
        / "phase_8_1_6d_founder_context_comparison_V2.png"
    )

    fig.savefig(
        chart_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    summary = {
        "schema_version":
            "PHASE_8_1_6D_EXPLICIT_FOUNDER_INCREMENTAL_COVERAGE_V2",

        "status":
            "COMPLETE",

        "scientific_role":
            (
                "DEVELOPMENT_ONLY_HETEROGENEOUS_"
                "CONTEXT_COVERAGE_AUDIT"
            ),

        "development_origins":
            DEVELOPMENT_ORIGINS,

        "mean_current_founder_proxy_coverage_pct":
            mean_current,

        "mean_explicit_founder_person_context_pct":
            mean_person,

        "mean_incremental_person_only_coverage_pct":
            mean_incremental,

        "mean_potential_union_context_pct":
            mean_union,

        "mean_fraction_current_isolation_recovered_pct":
            mean_recovery,

        "interpretation":
            (
                "Explicit founder Person nodes preserve founder "
                "information that may be discarded when founder "
                "relations are projected only into shared-founder "
                "Company-to-Company edges."
            ),

        "important_caveat":
            (
                "Explicit founder-Person context remains "
                "snapshot-unversioned and is not yet an "
                "approved origin-safe Phase-8 relation."
            ),

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
                "Temporal-provenance and eligibility audit for "
                "explicit founder-Person and Person-School context."
            ),
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    outputs = [
        SUMMARY_TABLE_OUTPUT,
        OVERLAP_OUTPUT,
        SUMMARY_OUTPUT,
        chart_path,
    ]

    manifest = {
        "phase":
            "8.1.6D",

        "status":
            "COMPLETE",

        "April_analyzed":
            False,

        "relation_approved":
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
        "PHASE 8.1.6D EXPLICIT FOUNDER "
        "INCREMENTAL COVERAGE AUDIT COMPLETE"
    )

    separator()

    print()

    print(
        "DEVELOPMENT FOLDS:              T57 / T58 / T59"
    )

    print(
        "EXPLICIT PERSON CONTEXT:        AUDITED"
    )

    print(
        "INCREMENTAL COVERAGE:           AUDITED"
    )

    print(
        "RELATION APPROVED:              NO"
    )

    print(
        "INPUT CONTRACT CHANGED:         NO"
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

    print(
        "APRIL PERFORMANCE INSPECTED:    NO"
    )

    print()

    print(
        "Next:"
    )

    print(
        "Audit temporal eligibility and provenance "
        "before deciding whether Person/School "
        "relations enter the Phase-8 model."
    )


if __name__ == "__main__":
    main()