from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.0.8 — EVIDENCE-BASED STARTUP EXISTENCE AUDIT V1
# =============================================================================
#
# Scientific role:
#
#   OUTCOME-BLIND ENTITY-EXISTENCE / CANDIDATE-ELIGIBILITY AUDIT
#
# A startup is considered PROVEN TO EXIST by a prediction origin if at
# least one of the following is true:
#
#   1. founded_on <= origin
#   2. an investment event involving that startup occurred <= origin
#   3. a timestamped acquisition relation involving that startup occurred
#      <= origin
#
# This script does NOT:
#
#   - score any model;
#   - generate negatives;
#   - select a model;
#   - remove evaluation cases;
#   - freeze the candidate universe;
#   - inspect April ranking performance.
#
# =============================================================================


RAW_DIR = Path(
    "data/raw"
)

CANONICAL_PATH = Path(
    "data/processed/interactions.parquet"
)

TEMPORAL_PATH = Path(
    "data/experimental/phase_2/"
    "temporal/interactions_itrs_temporal.parquet"
)

NODES_PATH = Path(
    "data/experimental/phase_3/graph/nodes.parquet"
)

EDGES_PATH = Path(
    "data/experimental/phase_3/graph/edges.parquet"
)


OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_8_evidence_based_startup_existence"
)

ORIGIN_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_8_origin_existence_summary_V1.csv"
)

PRIMARY_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_8_primary_stratum_existence_summary_V1.csv"
)

CANDIDATE_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_8_candidate_universe_existence_summary_V1.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_8_existence_audit_summary_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_8_analysis_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_8_derived_artifact_sha256_V1.json"
)


EXPECTED_STARTUP_NODES = 311_589


ORIGINS = [
    {
        "label": "T57",
        "history_end": "2025-03-31",
        "target_source": "temporal",
        "segment_number": 57,
        "expected_events": 22_742,
        "expected_primary": 7_470,
    },
    {
        "label": "T58",
        "history_end": "2025-06-30",
        "target_source": "temporal",
        "segment_number": 58,
        "expected_events": 23_337,
        "expected_primary": 7_550,
    },
    {
        "label": "T59",
        "history_end": "2025-09-30",
        "target_source": "temporal",
        "segment_number": 59,
        "expected_events": 23_644,
        "expected_primary": 7_538,
    },
    {
        "label": "APRIL_CONFIRMATORY",
        "history_end": "2026-03-31",
        "target_source": "april",
        "segment_number": None,
        "expected_events": 6_622,
        "expected_primary": 1_931,
    },
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

    return result.mask(
        result.eq("")
    )


def reconstruct_primary_mask(
    targets,
    history,
):

    historical_investors = set(
        history[
            "investor_id"
        ].astype(str)
    )

    historical_startups = set(
        history[
            "startup_id"
        ].astype(str)
    )

    historical_pairs = set(
        zip(
            history[
                "investor_id"
            ].astype(str),
            history[
                "startup_id"
            ].astype(str),
        )
    )

    investor = (
        targets[
            "investor_id"
        ].astype(str)
    )

    startup = (
        targets[
            "startup_id"
        ].astype(str)
    )

    new_pair = pd.Series(
        [
            (
                investor_id,
                startup_id,
            )
            not in historical_pairs
            for investor_id, startup_id
            in zip(
                investor,
                startup,
            )
        ],
        index=targets.index,
    )

    warm_investor = (
        investor.isin(
            historical_investors
        )
    )

    cold_startup = (
        ~startup.isin(
            historical_startups
        )
    )

    return (
        new_pair
        &
        warm_investor
        &
        cold_startup
    )


def main():

    separator()

    print(
        "PHASE 8.0.8 — "
        "EVIDENCE-BASED STARTUP EXISTENCE AUDIT V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "OUTCOME-BLIND ENTITY-EXISTENCE AUDIT"
    )

    print("Model loaded:                NO")
    print("Inference executed:          NO")
    print("Ranking metrics scored:      NO")
    print("Negatives generated:         NO")
    print("Model selection performed:   NO")
    print("Evaluation cases removed:    NO")
    print("Candidate universe frozen:   NO")
    print("April rankings inspected:    NO")

    # =========================================================================
    # 1. Source integrity
    # =========================================================================

    separator("-")
    print("8.0.8.1 — SOURCE EXISTENCE")
    separator("-")

    required_paths = [
        CANONICAL_PATH,
        TEMPORAL_PATH,
        NODES_PATH,
        EDGES_PATH,
    ]

    source_hashes = {}

    for path in required_paths:

        if not path.exists():
            raise FileNotFoundError(
                path
            )

        source_hashes[
            str(path)
        ] = sha256_file(
            path
        )

        print(
            f"FOUND  {path}"
        )

        print(
            f"SHA256 "
            f"{source_hashes[str(path)]}"
        )

    # =========================================================================
    # 2. Startup universe
    # =========================================================================

    nodes = pd.read_parquet(
        NODES_PATH,
        columns=[
            "node_type",
            "raw_entity_id",
        ],
    )

    startup_ids = set(
        nodes.loc[
            nodes[
                "node_type"
            ]
            ==
            "startup",
            "raw_entity_id",
        ]
        .astype(str)
        .unique()
    )

    if (
        len(startup_ids)
        != EXPECTED_STARTUP_NODES
    ):

        raise ValueError(
            "Startup universe drift."
        )

    print()
    print(
        f"Startup role nodes: "
        f"{len(startup_ids):,}"
    )

    # =========================================================================
    # 3. founded_on lookup
    # =========================================================================

    separator("-")
    print(
        "8.0.8.3 — FOUNDING-DATE RECONSTRUCTION"
    )
    separator("-")

    company_files = sorted(
        RAW_DIR.glob(
            "companies*.csv"
        )
    )

    if not company_files:
        raise FileNotFoundError(
            "No companies*.csv files."
        )

    frames = []

    for path in company_files:

        for chunk in pd.read_csv(
            path,
            usecols=[
                "id",
                "founded_on",
            ],
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
                ].isin(
                    startup_ids
                )
            ].copy()

            if len(subset) == 0:
                continue

            subset[
                "source_file"
            ] = path.name

            frames.append(
                subset
            )

    companies = pd.concat(
        frames,
        ignore_index=True,
    )

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
        .copy()
    )

    companies[
        "founded_on_parsed"
    ] = pd.to_datetime(
        companies[
            "founded_on"
        ],
        errors="coerce",
    )

    founded_lookup = (
        companies
        .set_index(
            "id"
        )[
            "founded_on_parsed"
        ]
        .to_dict()
    )

    # =========================================================================
    # 4. Earliest investment evidence
    # =========================================================================

    separator("-")
    print(
        "8.0.8.4 — EARLIEST INVESTMENT "
        "EXISTENCE EVIDENCE"
    )
    separator("-")

    canonical = pd.read_parquet(
        CANONICAL_PATH,
        columns=[
            "interaction_id",
            "announced_on",
            "investor_id",
            "startup_id",
        ],
    )

    canonical[
        "announced_on"
    ] = pd.to_datetime(
        canonical[
            "announced_on"
        ],
        errors="raise",
    )

    earliest_investment = (
        canonical
        .groupby(
            "startup_id"
        )[
            "announced_on"
        ]
        .min()
        .to_dict()
    )

    print(
        f"Startups with investment evidence: "
        f"{len(earliest_investment):,}"
    )

    # =========================================================================
    # 5. Earliest acquisition evidence
    # =========================================================================

    separator("-")
    print(
        "8.0.8.5 — EARLIEST ACQUISITION "
        "EXISTENCE EVIDENCE"
    )
    separator("-")

    edges = pd.read_parquet(
        EDGES_PATH,
        columns=[
            "relation",
            "underlying_src_id",
            "underlying_dst_id",
            "first_observed_on",
        ],
    )

    acquisition_edges = edges[
        edges[
            "relation"
        ]
        .isin(
            [
                "acquired",
                "acquired_by",
            ]
        )
    ].copy()

    acquisition_edges[
        "first_observed_on"
    ] = pd.to_datetime(
        acquisition_edges[
            "first_observed_on"
        ],
        errors="coerce",
    )

    acquisition_rows = []

    for row in (
        acquisition_edges
        .itertuples(
            index=False
        )
    ):

        if pd.isna(
            row.first_observed_on
        ):
            continue

        acquisition_rows.append(
            (
                str(
                    row.underlying_src_id
                ),
                row.first_observed_on,
            )
        )

        acquisition_rows.append(
            (
                str(
                    row.underlying_dst_id
                ),
                row.first_observed_on,
            )
        )

    acquisition_df = pd.DataFrame(
        acquisition_rows,
        columns=[
            "startup_id",
            "observed_on",
        ],
    )

    acquisition_df = (
        acquisition_df[
            acquisition_df[
                "startup_id"
            ].isin(
                startup_ids
            )
        ]
    )

    earliest_acquisition = (
        acquisition_df
        .groupby(
            "startup_id"
        )[
            "observed_on"
        ]
        .min()
        .to_dict()
    )

    print(
        f"Startups with acquisition evidence: "
        f"{len(earliest_acquisition):,}"
    )

    # =========================================================================
    # 6. Build startup existence registry
    # =========================================================================

    registry = pd.DataFrame(
        {
            "startup_id":
                sorted(
                    startup_ids
                )
        }
    )

    registry[
        "founded_on"
    ] = (
        registry[
            "startup_id"
        ]
        .map(
            founded_lookup
        )
    )

    registry[
        "first_investment_on"
    ] = (
        registry[
            "startup_id"
        ]
        .map(
            earliest_investment
        )
    )

    registry[
        "first_acquisition_evidence_on"
    ] = (
        registry[
            "startup_id"
        ]
        .map(
            earliest_acquisition
        )
    )

    # earliest available real-world evidence
    registry[
        "earliest_observed_existence"
    ] = registry[
        [
            "founded_on",
            "first_investment_on",
            "first_acquisition_evidence_on",
        ]
    ].min(
        axis=1
    )

    # =========================================================================
    # 7. Load temporal data
    # =========================================================================

    temporal = pd.read_parquet(
        TEMPORAL_PATH,
        columns=[
            "interaction_id",
            "announced_on",
            "investor_id",
            "startup_id",
            "segment_number",
        ],
    )

    temporal[
        "announced_on"
    ] = pd.to_datetime(
        temporal[
            "announced_on"
        ],
        errors="raise",
    )

    # =========================================================================
    # 8. Origin-specific analysis
    # =========================================================================

    origin_rows = []
    primary_rows = []
    candidate_rows = []

    registry_lookup = (
        registry.set_index(
            "startup_id"
        )
    )

    for spec in ORIGINS:

        label = spec[
            "label"
        ]

        origin = pd.Timestamp(
            spec[
                "history_end"
            ]
        )

        separator("-")
        print(
            f"8.0.8 — {label}"
        )
        separator("-")

        if (
            spec[
                "target_source"
            ]
            ==
            "temporal"
        ):

            targets = temporal[
                temporal[
                    "segment_number"
                ]
                ==
                spec[
                    "segment_number"
                ]
            ].copy()

            history = temporal[
                temporal[
                    "announced_on"
                ]
                <= origin
            ].copy()

        else:

            targets = canonical[
                (
                    canonical[
                        "announced_on"
                    ]
                    >=
                    pd.Timestamp(
                        "2026-04-01"
                    )
                )
                &
                (
                    canonical[
                        "announced_on"
                    ]
                    <=
                    pd.Timestamp(
                        "2026-04-30"
                    )
                )
            ].copy()

            history = canonical[
                canonical[
                    "announced_on"
                ]
                <= origin
            ].copy()

        if (
            len(targets)
            != spec[
                "expected_events"
            ]
        ):

            raise ValueError(
                f"{label} event-count drift."
            )

        primary_mask = (
            reconstruct_primary_mask(
                targets,
                history,
            )
        )

        if (
            int(
                primary_mask.sum()
            )
            !=
            spec[
                "expected_primary"
            ]
        ):

            raise ValueError(
                f"{label} primary-count drift."
            )

        targets[
            "startup_id_str"
        ] = (
            targets[
                "startup_id"
            ].astype(str)
        )

        targets = targets.join(
            registry_lookup[
                [
                    "founded_on",
                    "first_investment_on",
                    "first_acquisition_evidence_on",
                    "earliest_observed_existence",
                ]
            ],
            on="startup_id_str",
        )

        founded_proves = (
            targets[
                "founded_on"
            ].notna()
            &
            (
                targets[
                    "founded_on"
                ]
                <= origin
            )
        )

        investment_proves = (
            targets[
                "first_investment_on"
            ].notna()
            &
            (
                targets[
                    "first_investment_on"
                ]
                <= origin
            )
        )

        acquisition_proves = (
            targets[
                "first_acquisition_evidence_on"
            ].notna()
            &
            (
                targets[
                    "first_acquisition_evidence_on"
                ]
                <= origin
            )
        )

        proven_by_origin = (
            founded_proves
            |
            investment_proves
            |
            acquisition_proves
        )

        founded_only_proven = (
            founded_proves
        )

        rescued_by_event_evidence = (
            proven_by_origin
            &
            ~founded_only_proven
        )

        unresolved = (
            ~proven_by_origin
        )

        primary_proven = int(
            (
                primary_mask
                &
                proven_by_origin
            ).sum()
        )

        primary_rescued = int(
            (
                primary_mask
                &
                rescued_by_event_evidence
            ).sum()
        )

        primary_unresolved = int(
            (
                primary_mask
                &
                unresolved
            ).sum()
        )

        print(
            f"Target events:                     "
            f"{len(targets):,}"
        )

        print(
            f"Primary events:                    "
            f"{int(primary_mask.sum()):,}"
        )

        print()

        print(
            f"Existence proven by origin:         "
            f"{int(proven_by_origin.sum()):,}"
        )

        print(
            f"Rescued beyond founded_on:          "
            f"{int(rescued_by_event_evidence.sum()):,}"
        )

        print(
            f"Existence unresolved:               "
            f"{int(unresolved.sum()):,}"
        )

        print()

        print(
            f"PRIMARY proven by origin:           "
            f"{primary_proven:,}"
        )

        print(
            f"PRIMARY rescued by event evidence:  "
            f"{primary_rescued:,}"
        )

        print(
            f"PRIMARY unresolved:                 "
            f"{primary_unresolved:,}"
        )

        origin_rows.append(
            {
                "origin":
                    label,

                "target_events":
                    len(
                        targets
                    ),

                "existence_proven_by_origin":
                    int(
                        proven_by_origin.sum()
                    ),

                "rescued_beyond_founded_on":
                    int(
                        rescued_by_event_evidence.sum()
                    ),

                "existence_unresolved":
                    int(
                        unresolved.sum()
                    ),
            }
        )

        primary_rows.append(
            {
                "origin":
                    label,

                "primary_events":
                    int(
                        primary_mask.sum()
                    ),

                "primary_existence_proven_by_origin":
                    primary_proven,

                "primary_rescued_beyond_founded_on":
                    primary_rescued,

                "primary_existence_unresolved":
                    primary_unresolved,

                "primary_proven_share_pct":
                    (
                        100.0
                        * primary_proven
                        / int(
                            primary_mask.sum()
                        )
                    ),
            }
        )

        # ---------------------------------------------------------------------
        # Candidate universe
        # ---------------------------------------------------------------------

        candidate_founded = (
            registry[
                "founded_on"
            ].notna()
            &
            (
                registry[
                    "founded_on"
                ]
                <= origin
            )
        )

        candidate_investment = (
            registry[
                "first_investment_on"
            ].notna()
            &
            (
                registry[
                    "first_investment_on"
                ]
                <= origin
            )
        )

        candidate_acquisition = (
            registry[
                "first_acquisition_evidence_on"
            ].notna()
            &
            (
                registry[
                    "first_acquisition_evidence_on"
                ]
                <= origin
            )
        )

        candidate_proven = (
            candidate_founded
            |
            candidate_investment
            |
            candidate_acquisition
        )

        candidate_rescued = (
            candidate_proven
            &
            ~candidate_founded
        )

        candidate_rows.append(
            {
                "origin":
                    label,

                "global_startup_nodes":
                    len(
                        registry
                    ),

                "founded_on_only_eligible":
                    int(
                        candidate_founded.sum()
                    ),

                "evidence_based_eligible":
                    int(
                        candidate_proven.sum()
                    ),

                "rescued_by_historical_event":
                    int(
                        candidate_rescued.sum()
                    ),

                "existence_unresolved":
                    int(
                        (
                            ~candidate_proven
                        ).sum()
                    ),
            }
        )

    # =========================================================================
    # 9. Persist
    # =========================================================================

    origin_summary = pd.DataFrame(
        origin_rows
    )

    primary_summary = pd.DataFrame(
        primary_rows
    )

    candidate_summary = pd.DataFrame(
        candidate_rows
    )

    separator("-")
    print(
        "8.0.8.9 — SUMMARY"
    )
    separator("-")

    print(
        primary_summary.to_string(
            index=False
        )
    )

    print()

    print(
        candidate_summary.to_string(
            index=False
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    origin_summary.to_csv(
        ORIGIN_SUMMARY_OUTPUT,
        index=False,
    )

    primary_summary.to_csv(
        PRIMARY_SUMMARY_OUTPUT,
        index=False,
    )

    candidate_summary.to_csv(
        CANDIDATE_SUMMARY_OUTPUT,
        index=False,
    )

    summary = {
        "schema_version":
            "PHASE_8_0_8_EVIDENCE_BASED_STARTUP_EXISTENCE_V1",

        "scientific_role":
            "OUTCOME_BLIND_ENTITY_EXISTENCE_AUDIT",

        "existence_rule_tested": [
            "founded_on <= origin",
            "OR first investment <= origin",
            "OR first timestamped acquisition evidence <= origin",
        ],

        "evaluation_cases_removed":
            False,

        "candidate_universe_frozen":
            False,

        "model_selection_performed":
            False,

        "april_ranking_performance_inspected":
            False,

        "decision_status":
            "AUDIT_ONLY_NOT_FROZEN",
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    manifest = {
        "phase":
            "8.0.8",

        "authoritative_inputs":
            source_hashes,

        "outputs": [
            str(
                ORIGIN_SUMMARY_OUTPUT
            ),
            str(
                PRIMARY_SUMMARY_OUTPUT
            ),
            str(
                CANDIDATE_SUMMARY_OUTPUT
            ),
            str(
                SUMMARY_OUTPUT
            ),
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    derived_hashes = {
        str(ORIGIN_SUMMARY_OUTPUT):
            sha256_file(
                ORIGIN_SUMMARY_OUTPUT
            ),

        str(PRIMARY_SUMMARY_OUTPUT):
            sha256_file(
                PRIMARY_SUMMARY_OUTPUT
            ),

        str(CANDIDATE_SUMMARY_OUTPUT):
            sha256_file(
                CANDIDATE_SUMMARY_OUTPUT
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

    separator()

    print(
        "PHASE 8.0.8 EVIDENCE-BASED "
        "STARTUP EXISTENCE AUDIT COMPLETE"
    )

    separator()

    print()

    print(
        "EVALUATION CASES REMOVED:          NO"
    )

    print(
        "CANDIDATE UNIVERSE FROZEN:         NO"
    )

    print(
        "MODEL SELECTION PERFORMED:         NO"
    )

    print(
        "APRIL RANKING PERFORMANCE TOUCHED: NO"
    )

    print()

    print(
        "Next decision:"
    )

    print(
        "Freeze the origin-eligibility rule "
        "and candidate universe using all "
        "available pre-origin existence evidence."
    )


if __name__ == "__main__":
    main()