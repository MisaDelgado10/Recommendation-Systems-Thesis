from pathlib import Path
from collections import Counter, defaultdict
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 8.1.6F — FOUNDER NODE STRUCTURAL UTILITY EDA V1
# =============================================================================

RAW_DIR = Path("data/raw")

NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/"
    "model_ready/node_index.parquet"
)

EDGE_INDEX_PATH = Path(
    "data/experimental/phase_3/"
    "model_ready/edge_index.npy"
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

PREVIOUS_SUMMARY = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_6_data_foundations/"
    "phase_8_1_6e_founder_person_temporal_provenance/"
    "phase_8_1_6e_summary_V2.json"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_6_data_foundations/"
    "phase_8_1_6f_founder_node_structural_utility"
)

CHART_DIR = OUTPUT_DIR / "charts"

DEGREE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6f_founder_degree_distribution_V1.csv"
)

CASE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6f_primary_founder_utility_V1.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6f_summary_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_6f_manifest_V1.json"
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


def separator(char="=", width=118):
    print(char * width)


def require(condition, message):

    if not condition:
        raise ValueError(message)


def pct(num, den):

    if den == 0:
        return float("nan")

    return 100.0 * float(num) / float(den)


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

        url = normalize_url(
            url
        )

        if url is not None:
            urls.append(url)

    return list(
        dict.fromkeys(urls)
    )


def main():

    separator()

    print(
        "PHASE 8.1.6F — "
        "FOUNDER NODE STRUCTURAL UTILITY EDA V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "DEVELOPMENT-ONLY STRUCTURAL UTILITY EDA"
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

    # =========================================================================
    # 1. Gate
    # =========================================================================

    separator("-")

    print(
        "8.1.6F.1 — PREREQUISITE GATE"
    )

    separator("-")

    require(
        PREVIOUS_SUMMARY.exists(),
        "Missing Phase-8.1.6E V2 summary.",
    )

    previous = read_json(
        PREVIOUS_SUMMARY
    )

    require(
        previous.get("status")
        ==
        "COMPLETE",
        "Phase 8.1.6E V2 incomplete.",
    )

    require(
        previous.get("April_analyzed")
        is False,
        "April unexpectedly analyzed.",
    )

    print(
        "Phase 8.1.6E V2: COMPLETE PASS"
    )

    # =========================================================================
    # 2. Startup universe
    # =========================================================================

    separator("-")

    print(
        "8.1.6F.2 — STARTUP ROLE UNIVERSE"
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
            nodes["node_type"]
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

    startup_ids = set(
        startup_nodes[
            "raw_entity_id"
        ]
    )

    print(
        f"Startup-role nodes: "
        f"{len(startup_ids):,}"
    )

    # =========================================================================
    # 3. Build Startup -> Founder URL relation over entire Startup universe
    # =========================================================================

    separator("-")

    print(
        "8.1.6F.3 — STARTUP -> FOUNDER "
        "RELATION RECONSTRUCTION"
    )

    separator("-")

    startup_founder_urls = defaultdict(
        set
    )

    company_files = sorted(
        RAW_DIR.glob(
            "companies*.csv"
        )
    )

    require(
        len(company_files) == 24,
        "Expected 24 Company files.",
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

            matched += len(
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
                        str(startup_id)
                    ].add(url)

        print(
            f"[{i:02d}/24] "
            f"{path.name:<24} "
            f"startup rows={matched:,}"
        )

    founder_to_startups = defaultdict(
        set
    )

    for startup_id, urls in (
        startup_founder_urls.items()
    ):

        for url in urls:

            founder_to_startups[
                url
            ].add(
                startup_id
            )

    unique_founder_urls = set(
        founder_to_startups
    )

    print()

    print(
        f"Startups with founder URL: "
        f"{len(startup_founder_urls):,}"
    )

    print(
        f"Unique founder URLs:       "
        f"{len(unique_founder_urls):,}"
    )

    # =========================================================================
    # 4. Founder degree distribution
    # =========================================================================

    separator("-")

    print(
        "8.1.6F.4 — FOUNDER STARTUP-DEGREE "
        "DISTRIBUTION"
    )

    separator("-")

    founder_degree = {
        url:
            len(startups)
        for url, startups
        in founder_to_startups.items()
    }

    degree_counter = Counter(
        founder_degree.values()
    )

    degree_rows = []

    for degree in sorted(
        degree_counter
    ):

        degree_rows.append(
            {
                "startup_degree":
                    degree,

                "founder_count":
                    degree_counter[
                        degree
                    ],

                "founder_pct":
                    pct(
                        degree_counter[
                            degree
                        ],
                        len(
                            founder_degree
                        ),
                    ),
            }
        )

    degree_df = pd.DataFrame(
        degree_rows
    )

    leaf_founders = sum(
        degree == 1
        for degree
        in founder_degree.values()
    )

    shared_founders = sum(
        degree >= 2
        for degree
        in founder_degree.values()
    )

    print(
        f"Degree-1 founder URLs: "
        f"{leaf_founders:,} "
        f"({pct(leaf_founders, len(founder_degree)):.4f}%)"
    )

    print(
        f"Degree>=2 founder URLs: "
        f"{shared_founders:,} "
        f"({pct(shared_founders, len(founder_degree)):.4f}%)"
    )

    # =========================================================================
    # 5. Resolve founder-People and audit raw side-information availability
    # =========================================================================

    separator("-")

    print(
        "8.1.6F.5 — FOUNDER PERSON "
        "SIDE-INFORMATION SURFACE"
    )

    separator("-")

    people_files = sorted(
        RAW_DIR.glob(
            "people*.csv"
        )
    )

    url_to_ids = defaultdict(
        set
    )

    person_has_description = defaultdict(
        bool
    )

    person_has_school = defaultdict(
        bool
    )

    person_has_geo = defaultdict(
        bool
    )

    for i, path in enumerate(
        people_files,
        start=1,
    ):

        header = pd.read_csv(
            path,
            nrows=0,
        )

        usecols = [
            field
            for field
            in [
                "id",
                "link",
                "description",
                "schools",
                "country",
                "region",
                "locations",
            ]
            if field
            in header.columns
        ]

        matched = 0

        for chunk in pd.read_csv(
            path,
            usecols=usecols,
            dtype="string",
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ):

            normalized = (
                chunk[
                    "link"
                ]
                .map(
                    normalize_url
                )
            )

            relevant = normalized.isin(
                unique_founder_urls
            )

            if not relevant.any():
                continue

            subset = chunk.loc[
                relevant
            ].copy()

            subset[
                "normalized_link"
            ] = (
                normalized[
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

                url_to_ids[
                    row.normalized_link
                ].add(
                    person_id
                )

                if hasattr(
                    row,
                    "description"
                ):

                    value = getattr(
                        row,
                        "description"
                    )

                    if (
                        pd.notna(value)
                        and
                        str(value).strip()
                    ):

                        person_has_description[
                            person_id
                        ] = True

                if hasattr(
                    row,
                    "schools"
                ):

                    value = getattr(
                        row,
                        "schools"
                    )

                    if (
                        pd.notna(value)
                        and
                        str(value).strip()
                    ):

                        person_has_school[
                            person_id
                        ] = True

                geo_present = False

                for field in [
                    "country",
                    "region",
                    "locations",
                ]:

                    if not hasattr(
                        row,
                        field
                    ):
                        continue

                    value = getattr(
                        row,
                        field
                    )

                    if (
                        pd.notna(value)
                        and
                        str(value).strip()
                    ):

                        geo_present = True

                if geo_present:

                    person_has_geo[
                        person_id
                    ] = True

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

    # =========================================================================
    # 6. Existing founder-proxy connectivity
    # =========================================================================

    separator("-")

    print(
        "8.1.6F.6 — CURRENT FOUNDER-PROXY "
        "REFERENCE"
    )

    separator("-")

    edge_index = np.load(
        EDGE_INDEX_PATH,
        mmap_mode="r",
        allow_pickle=False,
    )

    if edge_index.shape[0] == 2:

        src_all = np.asarray(
            edge_index[0],
            dtype=np.int64,
        )

        dst_all = np.asarray(
            edge_index[1],
            dtype=np.int64,
        )

    elif edge_index.shape[1] == 2:

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

    masks = np.load(
        GRAPH_MASKS_PATH,
        allow_pickle=False,
    )

    founder_proxy_active = {}

    for origin in (
        DEVELOPMENT_ORIGINS
    ):

        key = (
            f"{origin}_founder_proxy"
        )

        require(
            key in masks.files,
            f"Missing mask: {key}",
        )

        mask = np.asarray(
            masks[
                key
            ],
            dtype=bool,
        )

        founder_proxy_active[
            origin
        ] = set(
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

    # =========================================================================
    # 7. Development-primary utility
    # =========================================================================

    separator("-")

    print(
        "8.1.6F.7 — DEVELOPMENT PRIMARY "
        "FOUNDER UTILITY"
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

    result_rows = []

    for origin in (
        DEVELOPMENT_ORIGINS
    ):

        ids = primary.loc[
            primary[
                "origin"
            ]
            ==
            origin,
            "startup_id_str",
        ].tolist()

        require(
            len(ids)
            ==
            EXPECTED_PRIMARY[
                origin
            ],
            f"{origin}: primary count drift.",
        )

        with_founder = 0
        with_shared_founder = 0
        leaf_only = 0

        with_description = 0
        with_school = 0
        with_geo = 0

        current_proxy = 0
        proxy_isolated_with_founder = 0

        for startup_id in ids:

            urls = (
                startup_founder_urls.get(
                    startup_id,
                    set(),
                )
            )

            resolved_urls = [
                url
                for url in urls
                if url
                in
                url_to_person
            ]

            if not resolved_urls:
                continue

            with_founder += 1

            degrees = [
                founder_degree[
                    url
                ]
                for url
                in resolved_urls
            ]

            if any(
                degree >= 2
                for degree
                in degrees
            ):

                with_shared_founder += 1

            else:

                leaf_only += 1

            people = [
                url_to_person[
                    url
                ]
                for url
                in resolved_urls
            ]

            if any(
                person_has_description[
                    person_id
                ]
                for person_id
                in people
            ):

                with_description += 1

            if any(
                person_has_school[
                    person_id
                ]
                for person_id
                in people
            ):

                with_school += 1

            if any(
                person_has_geo[
                    person_id
                ]
                for person_id
                in people
            ):

                with_geo += 1

            node_index = (
                startup_id_to_node[
                    startup_id
                ]
            )

            in_proxy = (
                node_index
                in
                founder_proxy_active[
                    origin
                ]
            )

            if in_proxy:

                current_proxy += 1

            else:

                proxy_isolated_with_founder += 1

        result_rows.append(
            {
                "origin":
                    origin,

                "primary_cases":
                    len(ids),

                "with_resolved_founder":
                    with_founder,

                "resolved_founder_pct":
                    pct(
                        with_founder,
                        len(ids),
                    ),

                "with_founder_degree_ge2":
                    with_shared_founder,

                "founder_degree_ge2_pct":
                    pct(
                        with_shared_founder,
                        len(ids),
                    ),

                "founder_leaf_only":
                    leaf_only,

                "founder_leaf_only_pct":
                    pct(
                        leaf_only,
                        len(ids),
                    ),

                "founder_with_description":
                    with_description,

                "founder_description_pct":
                    pct(
                        with_description,
                        len(ids),
                    ),

                "founder_with_school_field":
                    with_school,

                "founder_school_field_pct":
                    pct(
                        with_school,
                        len(ids),
                    ),

                "founder_with_geo":
                    with_geo,

                "founder_geo_pct":
                    pct(
                        with_geo,
                        len(ids),
                    ),

                "current_founder_proxy_connected":
                    current_proxy,

                "proxy_isolated_but_has_founder":
                    proxy_isolated_with_founder,

                "proxy_isolated_but_has_founder_pct":
                    pct(
                        proxy_isolated_with_founder,
                        len(ids),
                    ),
            }
        )

        print()

        print(
            f"{origin}"
        )

        print(
            f"  resolved founder:          "
            f"{with_founder:,} "
            f"({pct(with_founder, len(ids)):.4f}%)"
        )

        print(
            f"  founder degree >=2:        "
            f"{with_shared_founder:,} "
            f"({pct(with_shared_founder, len(ids)):.4f}%)"
        )

        print(
            f"  founder leaf-only:         "
            f"{leaf_only:,} "
            f"({pct(leaf_only, len(ids)):.4f}%)"
        )

        print(
            f"  founder description:       "
            f"{with_description:,} "
            f"({pct(with_description, len(ids)):.4f}%)"
        )

        print(
            f"  founder school field:      "
            f"{with_school:,} "
            f"({pct(with_school, len(ids)):.4f}%)"
        )

        print(
            f"  founder geography:         "
            f"{with_geo:,} "
            f"({pct(with_geo, len(ids)):.4f}%)"
        )

        print(
            f"  graph-isolated + founder:  "
            f"{proxy_isolated_with_founder:,} "
            f"({pct(proxy_isolated_with_founder, len(ids)):.4f}%)"
        )

    result_df = pd.DataFrame(
        result_rows
    )

    # =========================================================================
    # 8. Interpretation
    # =========================================================================

    separator("-")

    print(
        "8.1.6F.8 — STRUCTURAL INTERPRETATION"
    )

    separator("-")

    mean_founder = float(
        result_df[
            "resolved_founder_pct"
        ].mean()
    )

    mean_shared = float(
        result_df[
            "founder_degree_ge2_pct"
        ].mean()
    )

    mean_leaf = float(
        result_df[
            "founder_leaf_only_pct"
        ].mean()
    )

    mean_school = float(
        result_df[
            "founder_school_field_pct"
        ].mean()
    )

    mean_description = float(
        result_df[
            "founder_description_pct"
        ].mean()
    )

    print(
        f"Mean founder coverage:             "
        f"{mean_founder:.4f}%"
    )

    print(
        f"Mean founder degree>=2 coverage:   "
        f"{mean_shared:.4f}%"
    )

    print(
        f"Mean leaf-only founder coverage:   "
        f"{mean_leaf:.4f}%"
    )

    print(
        f"Mean founder-description coverage: "
        f"{mean_description:.4f}%"
    )

    print(
        f"Mean founder-school field coverage:"
        f" {mean_school:.4f}%"
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

    degree_df.to_csv(
        DEGREE_OUTPUT,
        index=False,
    )

    result_df.to_csv(
        CASE_OUTPUT,
        index=False,
    )

    chart_df = result_df[
        [
            "origin",
            "resolved_founder_pct",
            "founder_degree_ge2_pct",
            "founder_leaf_only_pct",
            "founder_school_field_pct",
        ]
    ]

    x = np.arange(
        len(chart_df)
    )

    width = 0.20

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        x - 1.5 * width,
        chart_df[
            "resolved_founder_pct"
        ],
        width=width,
        label="Any founder",
    )

    ax.bar(
        x - 0.5 * width,
        chart_df[
            "founder_degree_ge2_pct"
        ],
        width=width,
        label="Founder degree ≥2",
    )

    ax.bar(
        x + 0.5 * width,
        chart_df[
            "founder_leaf_only_pct"
        ],
        width=width,
        label="Leaf-only founder",
    )

    ax.bar(
        x + 1.5 * width,
        chart_df[
            "founder_school_field_pct"
        ],
        width=width,
        label="Founder has school field",
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
        "Primary cold-start cases (%)"
    )

    ax.set_title(
        "What information does an explicit Founder Person node provide?"
    )

    ax.legend()

    fig.tight_layout()

    chart_path = (
        CHART_DIR
        / "phase_8_1_6f_founder_structural_utility_V1.png"
    )

    fig.savefig(
        chart_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    summary = {
        "schema_version":
            "PHASE_8_1_6F_FOUNDER_STRUCTURAL_UTILITY_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            "DEVELOPMENT_ONLY_STRUCTURAL_UTILITY_EDA",

        "mean_founder_coverage_pct":
            mean_founder,

        "mean_founder_degree_ge2_pct":
            mean_shared,

        "mean_founder_leaf_only_pct":
            mean_leaf,

        "mean_founder_description_pct":
            mean_description,

        "mean_founder_school_field_pct":
            mean_school,

        "interpretation":
            (
                "Distinguishes genuinely relational founder "
                "structure from explicit founder nodes that "
                "would otherwise behave as leaf context."
            ),

        "relation_approved":
            False,

        "graph_modified":
            False,

        "architecture_selected":
            False,

        "model_trained":
            False,

        "April_analyzed":
            False,

        "next_step":
            (
                "Close the multi-entity Data Foundations stage "
                "and freeze research roles for each available dataset."
            ),
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    write_json(
        MANIFEST_OUTPUT,
        {
            "phase":
                "8.1.6F",

            "status":
                "COMPLETE",

            "outputs": [
                str(DEGREE_OUTPUT),
                str(CASE_OUTPUT),
                str(SUMMARY_OUTPUT),
                str(chart_path),
            ],

            "prohibited_operations": [
                "relation_approval",
                "graph_modification",
                "architecture_selection",
                "training",
                "April_analysis",
            ],
        },
    )

    separator()

    print(
        "PHASE 8.1.6F FOUNDER NODE "
        "STRUCTURAL UTILITY EDA COMPLETE"
    )

    separator()

    print()

    print(
        "FOUNDER DEGREE AUDITED:          YES"
    )

    print(
        "LEAF VS RELATIONAL CONTEXT:      AUDITED"
    )

    print(
        "PERSON SIDE INFO AVAILABILITY:   AUDITED"
    )

    print(
        "RELATION APPROVED:               NO"
    )

    print(
        "GRAPH MODIFIED:                  NO"
    )

    print(
        "ARCHITECTURE SELECTED:           NO"
    )

    print(
        "MODEL TRAINED:                   NO"
    )

    print(
        "APRIL ANALYZED:                  NO"
    )


if __name__ == "__main__":
    main()