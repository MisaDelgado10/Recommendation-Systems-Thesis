from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.0.7 — STARTUP ORIGIN-ELIGIBILITY AUDIT V1
# =============================================================================
#
# Scientific role:
#
#   OUTCOME-BLIND TARGET / CANDIDATE TEMPORAL-ELIGIBILITY AUDIT
#
# PURPOSE
# -------
#
# For each candidate Phase-8 evaluation origin:
#
#   T57   origin = 2025-03-31
#   T58   origin = 2025-06-30
#   T59   origin = 2025-09-30
#   April origin = 2026-03-31
#
# determine whether focal startups:
#
#   1. were already founded by the prediction origin;
#   2. were founded during the evaluation window;
#   3. have unknown founded_on;
#   4. have founded_on after their investment event (temporal anomaly).
#
# Also measure how large the Startup candidate universe would be if
# candidates were restricted to organizations known to be founded by
# the prediction origin.
#
# NO performance metrics are inspected.
#
# =============================================================================


RAW_DIR = Path(
    "data/raw"
)

TEMPORAL_PATH = Path(
    "data/experimental/phase_2/"
    "temporal/interactions_itrs_temporal.parquet"
)

CANONICAL_PATH = Path(
    "data/processed/interactions.parquet"
)

NODES_PATH = Path(
    "data/experimental/phase_3/graph/nodes.parquet"
)


OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_7_startup_origin_eligibility"
)

TARGET_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_7_target_origin_eligibility_summary_V1.csv"
)

CANDIDATE_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_7_candidate_universe_summary_V1.csv"
)

DETAIL_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_7_non_origin_eligible_target_detail_V1.parquet"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_7_startup_origin_eligibility_summary_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_7_analysis_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_7_derived_artifact_sha256_V1.json"
)


EXPECTED_GRAPH_NODES = 477_564
EXPECTED_STARTUP_NODES = 311_589


ORIGINS = [
    {
        "label": "T57",
        "history_end": "2025-03-31",
        "target_source": "temporal",
        "segment_number": 57,
        "expected_events": 22_742,
        "expected_primary_events": 7_470,
    },
    {
        "label": "T58",
        "history_end": "2025-06-30",
        "target_source": "temporal",
        "segment_number": 58,
        "expected_events": 23_337,
        "expected_primary_events": 7_550,
    },
    {
        "label": "T59",
        "history_end": "2025-09-30",
        "target_source": "temporal",
        "segment_number": 59,
        "expected_events": 23_644,
        "expected_primary_events": 7_538,
    },
    {
        "label": "APRIL_CONFIRMATORY",
        "history_end": "2026-03-31",
        "target_source": "canonical_april",
        "segment_number": None,
        "expected_events": 6_622,
        "expected_primary_events": 1_931,
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
        "PHASE 8.0.7 — "
        "STARTUP ORIGIN-ELIGIBILITY AUDIT V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "OUTCOME-BLIND TARGET/CANDIDATE "
        "TEMPORAL-ELIGIBILITY AUDIT"
    )

    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Inference executed:          NO")
    print("Ranking metrics scored:      NO")
    print("Negatives generated:         NO")
    print("Model selection performed:   NO")
    print("Evaluation cases removed:    NO")
    print("April outcomes scored:       NO")

    # =========================================================================
    # 1. Source existence
    # =========================================================================

    separator("-")

    print(
        "8.0.7.1 — SOURCE EXISTENCE "
        "AND FINGERPRINTS"
    )

    separator("-")

    required_paths = [
        TEMPORAL_PATH,
        CANONICAL_PATH,
        NODES_PATH,
    ]

    source_hashes = {}

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        observed_hash = sha256_file(
            path
        )

        source_hashes[
            str(path)
        ] = observed_hash

        print(
            f"FOUND  {path}"
        )

        print(
            f"SHA256 {observed_hash}"
        )

    company_files = sorted(
        RAW_DIR.glob(
            "companies*.csv"
        )
    )

    if not company_files:

        raise FileNotFoundError(
            "No companies*.csv files found."
        )

    print(
        f"Company CSV files: "
        f"{len(company_files):,}"
    )

    # =========================================================================
    # 2. Load Startup graph universe
    # =========================================================================

    separator("-")

    print(
        "8.0.7.2 — STARTUP ROLE UNIVERSE"
    )

    separator("-")

    nodes = pd.read_parquet(
        NODES_PATH,
        columns=[
            "node_type",
            "raw_entity_id",
        ],
    )

    if len(
        nodes
    ) != EXPECTED_GRAPH_NODES:

        raise ValueError(
            "Frozen graph node count drift."
        )

    startup_nodes = (
        nodes[
            nodes[
                "node_type"
            ]
            ==
            "startup"
        ][
            "raw_entity_id"
        ]
        .astype(str)
        .drop_duplicates()
    )

    if (
        len(
            startup_nodes
        )
        != EXPECTED_STARTUP_NODES
    ):

        raise ValueError(
            "Frozen Startup role count drift."
        )

    startup_ids = set(
        startup_nodes
    )

    print(
        f"Startup role nodes: "
        f"{len(startup_ids):,}"
    )

    # =========================================================================
    # 3. Reconstruct founded_on for Startup universe
    # =========================================================================

    separator("-")

    print(
        "8.0.7.3 — STARTUP FOUNDING-DATE "
        "RECONSTRUCTION"
    )

    separator("-")

    company_frames = []

    for file_number, path in enumerate(
        company_files,
        start=1,
    ):

        header = pd.read_csv(
            path,
            nrows=0,
        )

        if (
            "id"
            not in header.columns
            or
            "founded_on"
            not in header.columns
        ):

            raise ValueError(
                f"{path} missing id/founded_on."
            )

        print(
            f"[{file_number}/{len(company_files)}] "
            f"{path.name}"
        )

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

            if len(
                subset
            ) == 0:

                continue

            subset[
                "source_file"
            ] = path.name

            company_frames.append(
                subset
            )

    if not company_frames:

        raise ValueError(
            "No Startup nodes resolved "
            "to Company records."
        )

    companies = pd.concat(
        company_frames,
        ignore_index=True,
    )

    # Preserve frozen deterministic duplicate policy.
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

    companies[
        "founded_on_parsed"
    ] = pd.to_datetime(
        companies[
            "founded_on"
        ],
        errors="coerce",
    )

    founding_lookup = (
        companies.set_index(
            "id"
        )[
            "founded_on_parsed"
        ]
        .to_dict()
    )

    resolved_startups = set(
        companies[
            "id"
        ].astype(str)
    )

    print()
    print(
        f"Startup nodes expected:       "
        f"{len(startup_ids):,}"
    )

    print(
        f"Resolved Company records:     "
        f"{len(resolved_startups):,}"
    )

    print(
        f"Unresolved Startup nodes:     "
        f"{len(startup_ids - resolved_startups):,}"
    )

    founded_known = int(
        companies[
            "founded_on_parsed"
        ]
        .notna()
        .sum()
    )

    print(
        f"Parseable founded_on:         "
        f"{founded_known:,}"
    )

    print(
        f"Missing/unparseable founded:  "
        f"{len(startup_ids) - founded_known:,}"
    )

    # =========================================================================
    # 4. Load temporal/canonical targets
    # =========================================================================

    separator("-")

    print(
        "8.0.7.4 — LOAD TARGET "
        "INTERACTION UNIVERSES"
    )

    separator("-")

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

    print(
        f"Temporal rows:  "
        f"{len(temporal):,}"
    )

    print(
        f"Canonical rows: "
        f"{len(canonical):,}"
    )

    # =========================================================================
    # 5. Origin-specific target eligibility
    # =========================================================================

    target_rows = []
    candidate_rows = []
    detail_frames = []

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
            f"8.0.7 — {label} "
            "ORIGIN-ELIGIBILITY AUDIT"
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
            len(
                targets
            )
            !=
            spec[
                "expected_events"
            ]
        ):

            raise ValueError(
                f"{label} target count drift."
            )

        targets[
            "startup_id_str"
        ] = (
            targets[
                "startup_id"
            ].astype(str)
        )

        targets[
            "startup_founded_on"
        ] = (
            targets[
                "startup_id_str"
            ]
            .map(
                founding_lookup
            )
        )

        founded_known_mask = (
            targets[
                "startup_founded_on"
            ].notna()
        )

        founded_by_origin = (
            founded_known_mask
            &
            (
                targets[
                    "startup_founded_on"
                ]
                <= origin
            )
        )

        founded_after_origin = (
            founded_known_mask
            &
            (
                targets[
                    "startup_founded_on"
                ]
                > origin
            )
        )

        founded_by_event = (
            founded_known_mask
            &
            (
                targets[
                    "startup_founded_on"
                ]
                <=
                targets[
                    "announced_on"
                ]
            )
        )

        founded_during_window_before_event = (
            founded_after_origin
            &
            founded_by_event
        )

        founded_after_event = (
            founded_known_mask
            &
            (
                targets[
                    "startup_founded_on"
                ]
                >
                targets[
                    "announced_on"
                ]
            )
        )

        unknown_founded = (
            ~founded_known_mask
        )

        primary_mask = (
            reconstruct_primary_mask(
                targets,
                history,
            )
        )

        primary_count = int(
            primary_mask.sum()
        )

        if (
            primary_count
            !=
            spec[
                "expected_primary_events"
            ]
        ):

            raise ValueError(
                f"{label} primary-stratum "
                "count drift."
            )

        primary_founded_by_origin = int(
            (
                primary_mask
                &
                founded_by_origin
            ).sum()
        )

        primary_founded_during_window = int(
            (
                primary_mask
                &
                founded_during_window_before_event
            ).sum()
        )

        primary_unknown_founded = int(
            (
                primary_mask
                &
                unknown_founded
            ).sum()
        )

        primary_founded_after_event = int(
            (
                primary_mask
                &
                founded_after_event
            ).sum()
        )

        print(
            f"Prediction origin:                 "
            f"{origin.date()}"
        )

        print(
            f"Target events:                     "
            f"{len(targets):,}"
        )

        print(
            f"Primary-stratum events:            "
            f"{primary_count:,}"
        )

        print()
        print(
            f"Startup founded by origin:          "
            f"{int(founded_by_origin.sum()):,}"
        )

        print(
            f"Startup founded during eval window: "
            f"{int(founded_during_window_before_event.sum()):,}"
        )

        print(
            f"Startup founded_on unknown:         "
            f"{int(unknown_founded.sum()):,}"
        )

        print(
            f"Startup founded after event:        "
            f"{int(founded_after_event.sum()):,}"
        )

        print()
        print(
            f"PRIMARY founded by origin:          "
            f"{primary_founded_by_origin:,}"
        )

        print(
            f"PRIMARY founded during window:      "
            f"{primary_founded_during_window:,}"
        )

        print(
            f"PRIMARY founded_on unknown:         "
            f"{primary_unknown_founded:,}"
        )

        print(
            f"PRIMARY founded after event:        "
            f"{primary_founded_after_event:,}"
        )

        target_rows.append(
            {
                "origin":
                    label,

                "history_end":
                    origin.date(),

                "target_events":
                    len(
                        targets
                    ),

                "primary_stratum_events":
                    primary_count,

                "startup_founded_by_origin":
                    int(
                        founded_by_origin.sum()
                    ),

                "startup_founded_during_eval_window":
                    int(
                        founded_during_window_before_event.sum()
                    ),

                "startup_founded_on_unknown":
                    int(
                        unknown_founded.sum()
                    ),

                "startup_founded_after_event":
                    int(
                        founded_after_event.sum()
                    ),

                "primary_founded_by_origin":
                    primary_founded_by_origin,

                "primary_founded_during_eval_window":
                    primary_founded_during_window,

                "primary_founded_on_unknown":
                    primary_unknown_founded,

                "primary_founded_after_event":
                    primary_founded_after_event,

                "primary_origin_eligible_share_pct":
                    (
                        100.0
                        * primary_founded_by_origin
                        / primary_count
                        if primary_count
                        else 0.0
                    ),
            }
        )

        # ---------------------------------------------------------------------
        # Candidate universe at origin
        # ---------------------------------------------------------------------

        startup_universe = pd.DataFrame(
            {
                "startup_id":
                    sorted(
                        startup_ids
                    )
            }
        )

        startup_universe[
            "founded_on"
        ] = (
            startup_universe[
                "startup_id"
            ]
            .map(
                founding_lookup
            )
        )

        candidate_known = (
            startup_universe[
                "founded_on"
            ].notna()
        )

        candidate_founded_by_origin = (
            candidate_known
            &
            (
                startup_universe[
                    "founded_on"
                ]
                <= origin
            )
        )

        candidate_future = (
            candidate_known
            &
            (
                startup_universe[
                    "founded_on"
                ]
                > origin
            )
        )

        candidate_unknown = (
            ~candidate_known
        )

        candidate_rows.append(
            {
                "origin":
                    label,

                "history_end":
                    origin.date(),

                "global_startup_nodes":
                    len(
                        startup_universe
                    ),

                "startup_nodes_founded_by_origin":
                    int(
                        candidate_founded_by_origin.sum()
                    ),

                "startup_nodes_founded_after_origin":
                    int(
                        candidate_future.sum()
                    ),

                "startup_nodes_founded_on_unknown":
                    int(
                        candidate_unknown.sum()
                    ),

                "strict_origin_known_candidate_pool":
                    int(
                        candidate_founded_by_origin.sum()
                    ),
            }
        )

        # ---------------------------------------------------------------------
        # Persist only cases requiring interpretation.
        # ---------------------------------------------------------------------

        non_origin_eligible = targets[
            (
                ~founded_by_origin
            )
        ].copy()

        non_origin_eligible[
            "origin"
        ] = label

        non_origin_eligible[
            "history_end"
        ] = origin

        non_origin_eligible[
            "is_primary_stratum"
        ] = (
            primary_mask.loc[
                non_origin_eligible.index
            ]
            .to_numpy(
                dtype=bool
            )
        )

        non_origin_eligible[
            "eligibility_status"
        ] = "unknown"

        non_origin_eligible.loc[
            non_origin_eligible[
                "startup_founded_on"
            ].isna(),
            "eligibility_status",
        ] = (
            "founded_on_unknown"
        )

        non_origin_eligible.loc[
            (
                non_origin_eligible[
                    "startup_founded_on"
                ].notna()
            )
            &
            (
                non_origin_eligible[
                    "startup_founded_on"
                ]
                >
                non_origin_eligible[
                    "announced_on"
                ]
            ),
            "eligibility_status",
        ] = (
            "founded_after_investment_event"
        )

        non_origin_eligible.loc[
            (
                non_origin_eligible[
                    "startup_founded_on"
                ].notna()
            )
            &
            (
                non_origin_eligible[
                    "startup_founded_on"
                ]
                > origin
            )
            &
            (
                non_origin_eligible[
                    "startup_founded_on"
                ]
                <=
                non_origin_eligible[
                    "announced_on"
                ]
            ),
            "eligibility_status",
        ] = (
            "founded_during_evaluation_window"
        )

        detail_frames.append(
            non_origin_eligible[
                [
                    "origin",
                    "history_end",
                    "interaction_id",
                    "announced_on",
                    "investor_id",
                    "startup_id",
                    "startup_founded_on",
                    "is_primary_stratum",
                    "eligibility_status",
                ]
            ]
        )

    # =========================================================================
    # 6. Summaries
    # =========================================================================

    target_summary = pd.DataFrame(
        target_rows
    )

    candidate_summary = pd.DataFrame(
        candidate_rows
    )

    detail = pd.concat(
        detail_frames,
        ignore_index=True,
    )

    separator("-")

    print(
        "8.0.7.6 — TARGET ELIGIBILITY SUMMARY"
    )

    separator("-")

    print(
        target_summary.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "8.0.7.7 — CANDIDATE UNIVERSE SUMMARY"
    )

    separator("-")

    print(
        candidate_summary.to_string(
            index=False
        )
    )

    separator("-")

    print(
        "8.0.7.8 — NON-ORIGIN-ELIGIBLE "
        "STATUS SUMMARY"
    )

    separator("-")

    status_summary = (
        detail.groupby(
            [
                "origin",
                "eligibility_status",
            ],
            dropna=False,
        )
        .agg(
            events=(
                "interaction_id",
                "size",
            ),
            primary_events=(
                "is_primary_stratum",
                "sum",
            ),
        )
        .reset_index()
    )

    print(
        status_summary.to_string(
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

    target_summary.to_csv(
        TARGET_SUMMARY_OUTPUT,
        index=False,
    )

    candidate_summary.to_csv(
        CANDIDATE_SUMMARY_OUTPUT,
        index=False,
    )

    detail.to_parquet(
        DETAIL_OUTPUT,
        index=False,
    )

    summary = {
        "schema_version":
            "PHASE_8_0_7_STARTUP_ORIGIN_ELIGIBILITY_V1",

        "scientific_role":
            "OUTCOME_BLIND_TARGET_CANDIDATE_TEMPORAL_ELIGIBILITY_AUDIT",

        "evaluation_cases_removed":
            False,

        "candidate_universe_frozen":
            False,

        "development_protocol_frozen":
            False,

        "april_confirmatory_outcomes_scored":
            False,

        "key_question": (
            "Was each focal startup already founded "
            "at the prediction origin?"
        ),

        "interpretation": {
            "founded_by_origin":
                (
                    "Startup is temporally eligible "
                    "for a recommendation made at the "
                    "fold origin."
                ),

            "founded_during_evaluation_window":
                (
                    "Startup exists by its investment "
                    "event but did not exist at the "
                    "fold origin; cannot be recommended "
                    "under a single origin-time ranking."
                ),

            "founded_on_unknown":
                (
                    "Temporal startup existence cannot "
                    "be established from available data."
                ),

            "founded_after_investment_event":
                (
                    "Temporal inconsistency requiring "
                    "manual/data-quality interpretation."
                ),
        },

        "decision_status":
            "AUDIT_ONLY_NOT_FROZEN",
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    manifest = {
        "phase":
            "8.0.7",

        "authoritative_inputs":
            source_hashes,

        "company_raw_files": [
            path.name
            for path in company_files
        ],

        "origins":
            ORIGINS,

        "outputs": [
            str(
                TARGET_SUMMARY_OUTPUT
            ),
            str(
                CANDIDATE_SUMMARY_OUTPUT
            ),
            str(
                DETAIL_OUTPUT
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
            "case_removal",
            "candidate_universe_freeze",
            "april_confirmatory_scoring",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    derived_hashes = {
        str(TARGET_SUMMARY_OUTPUT):
            sha256_file(
                TARGET_SUMMARY_OUTPUT
            ),

        str(CANDIDATE_SUMMARY_OUTPUT):
            sha256_file(
                CANDIDATE_SUMMARY_OUTPUT
            ),

        str(DETAIL_OUTPUT):
            sha256_file(
                DETAIL_OUTPUT
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
    # 8. Final
    # =========================================================================

    separator()

    print(
        "PHASE 8.0.7 STARTUP "
        "ORIGIN-ELIGIBILITY AUDIT COMPLETE"
    )

    separator()

    print()

    for row in target_rows:

        print(
            f"{row['origin']}: "
            f"{row['target_events']:,} events | "
            f"{row['primary_stratum_events']:,} primary | "
            f"{row['primary_founded_by_origin']:,} "
            f"primary founded by origin "
            f"({row['primary_origin_eligible_share_pct']:.2f}%)"
        )

    print()

    print(
        "EVALUATION CASES REMOVED:          NO"
    )

    print(
        "CANDIDATE UNIVERSE FROZEN:         NO"
    )

    print(
        "DEVELOPMENT PROTOCOL FROZEN:       NO"
    )

    print(
        "APRIL CONFIRMATORY OUTCOMES SCORED:NO"
    )

    print()

    print(
        "Next decision:"
    )

    print(
        "Freeze whether Phase-8 ranking is "
        "defined over startups that already "
        "exist at prediction origin, and define "
        "handling of unknown founded_on cases."
    )

    print()

    print(
        f"Outputs written to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()