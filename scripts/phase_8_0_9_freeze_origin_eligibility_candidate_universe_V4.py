from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 8.0.9 — FREEZE ORIGIN ELIGIBILITY & CANDIDATE UNIVERSE V2
# =============================================================================
#
# Scientific role:
#
#   OUTCOME-BLIND EVALUATION-UNIVERSE CONTRACT FREEZE
#
# This script freezes:
#
#   1. the rule for proving that a Startup existed at a prediction origin;
#   2. the origin-specific Startup candidate universe;
#   3. origin-eligible target-event membership;
#   4. the origin-eligible primary thesis stratum.
#
# EXISTENCE RULE
# --------------
#
# A Startup is proven to exist by origin t if:
#
#   founded_on <= t
#
#       OR
#
#   first observed investment event <= t
#
#       OR
#
#   first frozen timestamped acquisition evidence <= t
#
# IMPORTANT
# ---------
#
# "Unresolved" means:
#
#   existence is not established by the audited evidence sources.
#
# It does NOT mean:
#
#   the Startup definitely did not exist.
#
# This script:
#
#   - does NOT train a model;
#   - does NOT score a model;
#   - does NOT generate evaluation negatives;
#   - does NOT inspect ranking performance;
#   - does NOT perform model selection.
#
# =============================================================================


# =============================================================================
# PATHS
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
    "phase_8_0_9_origin_eligibility_candidate_universe"
)

REGISTRY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_9_startup_existence_registry_V4.parquet"
)

CANDIDATE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_9_origin_candidate_universe_V4.parquet"
)

TARGET_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_9_origin_eligible_targets_V4.parquet"
)

EXCLUDED_TARGET_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_9_noneligible_target_register_V4.parquet"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_9_origin_contract_summary_V4.csv"
)

CONTRACT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_9_origin_eligibility_contract_V4.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_9_manifest_V4.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_9_derived_artifact_sha256_V4.json"
)


# =============================================================================
# FROZEN EXPECTATIONS FROM PHASE 8.0.8
# =============================================================================

EXPECTED_STARTUP_NODES = 311_589

EXPECTED = {
    "T57": {
        "events": 22_742,
        "eligible_targets": 21_924,
        "primary": 7_470,
        "eligible_primary": 6_850,
        "candidate_universe": 308_027,
    },
    "T58": {
        "events": 23_337,
        "eligible_targets": 22_384,
        "primary": 7_550,
        "eligible_primary": 6_872,
        "candidate_universe": 308_854,
    },
    "T59": {
        "events": 23_644,
        "eligible_targets": 22_650,
        "primary": 7_538,
        "eligible_primary": 6_764,
        "candidate_universe": 309_611,
    },
    "APRIL_CONFIRMATORY": {
        "events": 6_622,
        "eligible_targets": 6_313,
        "primary": 1_931,
        "eligible_primary": 1_690,
        "candidate_universe": 311_266,
    },
}


ORIGINS = [
    {
        "origin": "T57",
        "history_end": "2025-03-31",
        "source": "temporal",
        "segment_number": 57,
    },
    {
        "origin": "T58",
        "history_end": "2025-06-30",
        "source": "temporal",
        "segment_number": 58,
    },
    {
        "origin": "T59",
        "history_end": "2025-09-30",
        "source": "temporal",
        "segment_number": 59,
    },
    {
        "origin": "APRIL_CONFIRMATORY",
        "history_end": "2026-03-31",
        "source": "canonical_april",
        "segment_number": None,
    },
]


# =============================================================================
# UTILITIES
# =============================================================================

def separator(char="=", width=118):
    print(char * width)


def sha256_file(path, chunk_size=1024 * 1024):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

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


def primary_mask(
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


# =============================================================================
# MAIN
# =============================================================================

def main():

    # Create the Phase-8.0.9 output directory before the first
    # persistence operation. V1 attempted to persist the existence
    # registry before this directory had been materialized.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    separator()

    print(
        "PHASE 8.0.9 — "
        "FREEZE ORIGIN ELIGIBILITY "
        "& CANDIDATE UNIVERSE V4"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "OUTCOME-BLIND EVALUATION-UNIVERSE FREEZE"
    )

    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Inference executed:          NO")
    print("Ranking metrics scored:      NO")
    print("Negatives generated:         NO")
    print("Model selection performed:   NO")
    print("Candidate universe frozen:   YES")
    print("Eligible targets frozen:     YES")
    print("April ranking scored:        NO")

    # =========================================================================
    # 1. Input integrity
    # =========================================================================

    separator("-")
    print(
        "8.0.9.1 — INPUT EXISTENCE "
        "AND FINGERPRINTS"
    )
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

    # =========================================================================
    # 2. Startup role universe
    # =========================================================================

    separator("-")
    print(
        "8.0.9.2 — STARTUP ROLE UNIVERSE"
    )
    separator("-")

    nodes = pd.read_parquet(
        NODES_PATH,
        columns=[
            "node_type",
            "raw_entity_id",
        ],
    )

    startup_ids = sorted(
        set(
            nodes.loc[
                nodes[
                    "node_type"
                ]
                ==
                "startup",
                "raw_entity_id",
            ]
            .astype(str)
        )
    )

    if (
        len(startup_ids)
        != EXPECTED_STARTUP_NODES
    ):

        raise ValueError(
            "Startup role universe drift."
        )

    print(
        f"Startup nodes: "
        f"{len(startup_ids):,}"
    )

    # =========================================================================
    # 3. founded_on reconstruction
    # =========================================================================

    separator("-")
    print(
        "8.0.9.3 — FOUNDING-DATE "
        "RECONSTRUCTION"
    )
    separator("-")

    company_frames = []

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

            company_frames.append(
                subset
            )

    companies = pd.concat(
        company_frames,
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

    if (
        companies[
            "id"
        ].nunique()
        != EXPECTED_STARTUP_NODES
    ):

        raise ValueError(
            "Not all Startup nodes resolved "
            "to Company records."
        )

    companies[
        "founded_on"
    ] = pd.to_datetime(
        companies[
            "founded_on"
        ],
        errors="coerce",
    )

    # =========================================================================
    # 4. Investment evidence
    # =========================================================================

    separator("-")
    print(
        "8.0.9.4 — INVESTMENT EXISTENCE EVIDENCE"
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

    # Explicit Phase-8.0.9 V3 key normalization.
    #
    # Phase 8.0.8 used dictionary/map-style lookup semantics.
    # V1/V2 switched to merge semantics. Normalize identity keys
    # before grouping/merging so both implementations operate on
    # the same canonical string namespace.
    canonical[
        "startup_id"
    ] = canonical[
        "startup_id"
    ].astype(str)

    canonical[
        "investor_id"
    ] = canonical[
        "investor_id"
    ].astype(str)

    earliest_investment = (
        canonical
        .groupby(
            "startup_id"
        )[
            "announced_on"
        ]
        .min()
        .rename(
            "first_investment_on"
        )
    )

    # =========================================================================
    # 5. Frozen acquisition existence evidence
    # =========================================================================

    separator("-")
    print(
        "8.0.9.5 — ACQUISITION EXISTENCE EVIDENCE"
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

    acquisition_evidence_rows = []

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

        acquisition_evidence_rows.append(
            {
                "startup_id":
                    str(
                        row.underlying_src_id
                    ),

                "observed_on":
                    row.first_observed_on,
            }
        )

        acquisition_evidence_rows.append(
            {
                "startup_id":
                    str(
                        row.underlying_dst_id
                    ),

                "observed_on":
                    row.first_observed_on,
            }
        )

    acquisition_evidence = (
        pd.DataFrame(
            acquisition_evidence_rows
        )
    )

    acquisition_evidence = (
        acquisition_evidence[
            acquisition_evidence[
                "startup_id"
            ].isin(
                startup_ids
            )
        ]
    )

    earliest_acquisition = (
        acquisition_evidence
        .groupby(
            "startup_id"
        )[
            "observed_on"
        ]
        .min()
        .rename(
            "first_acquisition_evidence_on"
        )
    )

    # =========================================================================
    # 6. Freeze Startup existence registry
    # =========================================================================

    separator("-")
    print(
        "8.0.9.6 — STARTUP EXISTENCE REGISTRY"
    )
    separator("-")

    registry = pd.DataFrame(
        {
            "startup_id":
                startup_ids
        }
    )

    registry = registry.merge(
        companies[
            [
                "id",
                "founded_on",
            ]
        ].rename(
            columns={
                "id":
                    "startup_id"
            }
        ),
        on="startup_id",
        how="left",
        validate="one_to_one",
    )

    registry = registry.merge(
        earliest_investment,
        left_on="startup_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )

    registry = registry.merge(
        earliest_acquisition,
        left_on="startup_id",
        right_index=True,
        how="left",
        validate="one_to_one",
    )

    registry[
        "earliest_existence_evidence_on"
    ] = registry[
        [
            "founded_on",
            "first_investment_on",
            "first_acquisition_evidence_on",
        ]
    ].min(
        axis=1
    )

    registry[
        "has_founded_on_evidence"
    ] = registry[
        "founded_on"
    ].notna()

    registry[
        "has_investment_evidence"
    ] = registry[
        "first_investment_on"
    ].notna()

    registry[
        "has_acquisition_evidence"
    ] = registry[
        "first_acquisition_evidence_on"
    ].notna()

    registry.to_parquet(
        REGISTRY_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 7. Temporal interaction layer
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

    temporal[
        "startup_id"
    ] = temporal[
        "startup_id"
    ].astype(str)

    temporal[
        "investor_id"
    ] = temporal[
        "investor_id"
    ].astype(str)

    registry_lookup = (
        registry.set_index(
            "startup_id"
        )
    )

    # =========================================================================
    # 8. Freeze each origin
    # =========================================================================

    candidate_frames = []
    eligible_target_frames = []
    excluded_target_frames = []
    summary_rows = []

    for spec in ORIGINS:

        label = spec[
            "origin"
        ]

        origin = pd.Timestamp(
            spec[
                "history_end"
            ]
        )

        separator("-")
        print(
            f"8.0.9 — {label} FREEZE"
        )
        separator("-")

        # ---------------------------------------------------------------------
        # Candidate universe:
        #
        # determined ONLY by information <= origin.
        # ---------------------------------------------------------------------

        candidates = registry[
            registry[
                "earliest_existence_evidence_on"
            ].notna()
            &
            (
                registry[
                    "earliest_existence_evidence_on"
                ]
                <= origin
            )
        ].copy()

        candidates[
            "origin"
        ] = label

        candidates[
            "history_end"
        ] = origin

        candidates[
            "candidate_eligible"
        ] = True

        candidate_ids = set(
            candidates[
                "startup_id"
            ]
        )

        # ---------------------------------------------------------------------
        # V3 RECONSTRUCTION DIAGNOSTICS
        #
        # Reconstruct the union explicitly from each evidence source.
        # These diagnostics occur before the frozen count assertion so a
        # mismatch can be explained rather than merely reported.
        # ---------------------------------------------------------------------

        founded_eligible = (
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

        investment_eligible = (
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

        acquisition_eligible = (
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

        explicit_union = (
            founded_eligible
            |
            investment_eligible
            |
            acquisition_eligible
        )

        earliest_evidence_eligible = (
            registry[
                "earliest_existence_evidence_on"
            ].notna()
            &
            (
                registry[
                    "earliest_existence_evidence_on"
                ]
                <= origin
            )
        )

        if not (
            explicit_union
            .equals(
                earliest_evidence_eligible
            )
        ):

            disagreement_mask = (
                explicit_union
                !=
                earliest_evidence_eligible
            )

            disagreement = int(
                disagreement_mask.sum()
            )

            diagnostic = registry.loc[
                disagreement_mask,
                [
                    "startup_id",
                    "founded_on",
                    "first_investment_on",
                    "first_acquisition_evidence_on",
                    "earliest_existence_evidence_on",
                ],
            ].copy()

            diagnostic[
                "founded_eligible"
            ] = (
                founded_eligible.loc[
                    disagreement_mask
                ]
                .to_numpy(
                    dtype=bool
                )
            )

            diagnostic[
                "investment_eligible"
            ] = (
                investment_eligible.loc[
                    disagreement_mask
                ]
                .to_numpy(
                    dtype=bool
                )
            )

            diagnostic[
                "acquisition_eligible"
            ] = (
                acquisition_eligible.loc[
                    disagreement_mask
                ]
                .to_numpy(
                    dtype=bool
                )
            )

            diagnostic[
                "explicit_union_eligible"
            ] = (
                explicit_union.loc[
                    disagreement_mask
                ]
                .to_numpy(
                    dtype=bool
                )
            )

            diagnostic[
                "earliest_rule_eligible"
            ] = (
                earliest_evidence_eligible.loc[
                    disagreement_mask
                ]
                .to_numpy(
                    dtype=bool
                )
            )

            print()
            print(
                "!!! EVIDENCE-RULE DISAGREEMENT DETECTED !!!"
            )

            print(
                f"Origin: {label}"
            )

            print(
                f"Prediction origin: {origin.date()}"
            )

            print(
                f"Disagreeing startups: "
                f"{disagreement:,}"
            )

            print()

            print(
                diagnostic.to_string(
                    index=False
                )
            )

            print()

            print(
                "Column dtypes:"
            )

            print(
                registry[
                    [
                        "founded_on",
                        "first_investment_on",
                        "first_acquisition_evidence_on",
                        "earliest_existence_evidence_on",
                    ]
                ]
                .dtypes
                .to_string()
            )

            raise RuntimeError(
                f"{label}: stopped intentionally after "
                f"printing {disagreement:,} evidence-rule "
                "disagreements. No contract frozen."
            )

        print()
        print(
            "Candidate evidence reconstruction:"
        )

        print(
            f"  founded_on <= origin:       "
            f"{int(founded_eligible.sum()):,}"
        )

        print(
            f"  investment <= origin:       "
            f"{int(investment_eligible.sum()):,}"
        )

        print(
            f"  acquisition <= origin:      "
            f"{int(acquisition_eligible.sum()):,}"
        )

        print(
            f"  evidence union:              "
            f"{int(explicit_union.sum()):,}"
        )

        print(
            f"  earliest-evidence selection: "
            f"{len(candidates):,}"
        )

        print(
            f"  frozen Phase-8.0.8 expected: "
            f"{EXPECTED[label]['candidate_universe']:,}"
        )

        # ---------------------------------------------------------------------
        # Target universe
        # ---------------------------------------------------------------------

        if (
            spec[
                "source"
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
                    "earliest_existence_evidence_on",
                ]
            ],
            on="startup_id_str",
        )

        targets[
            "is_primary_stratum"
        ] = (
            primary_mask(
                targets,
                history,
            )
            .to_numpy(
                dtype=bool
            )
        )

        targets[
            "origin_eligible"
        ] = (
            targets[
                "startup_id_str"
            ]
            .isin(
                candidate_ids
            )
        )

        eligible = targets[
            targets[
                "origin_eligible"
            ]
        ].copy()

        excluded = targets[
            ~targets[
                "origin_eligible"
            ]
        ].copy()

        eligible[
            "origin"
        ] = label

        eligible[
            "history_end"
        ] = origin

        excluded[
            "origin"
        ] = label

        excluded[
            "history_end"
        ] = origin

        # ---------------------------------------------------------------------
        # Assertions against Phase 8.0.8
        # ---------------------------------------------------------------------

        expected = EXPECTED[
            label
        ]

        if (
            len(targets)
            != expected[
                "events"
            ]
        ):

            raise ValueError(
                f"{label}: event count drift."
            )

        if (
            int(
                targets[
                    "is_primary_stratum"
                ].sum()
            )
            != expected[
                "primary"
            ]
        ):

            raise ValueError(
                f"{label}: primary count drift."
            )

        if (
            len(candidates)
            != expected[
                "candidate_universe"
            ]
        ):

            raise ValueError(
                f"{label}: candidate-universe "
                "count drift."
            )

        if (
            len(eligible)
            != expected[
                "eligible_targets"
            ]
        ):

            raise ValueError(
                f"{label}: eligible-target "
                "count drift."
            )

        eligible_primary = int(
            eligible[
                "is_primary_stratum"
            ].sum()
        )

        if (
            eligible_primary
            != expected[
                "eligible_primary"
            ]
        ):

            raise ValueError(
                f"{label}: eligible-primary "
                "count drift."
            )

        # Every eligible focal positive must be in the frozen
        # candidate universe.
        if not (
            set(
                eligible[
                    "startup_id_str"
                ]
            )
            <= candidate_ids
        ):

            raise ValueError(
                f"{label}: eligible focal positive "
                "outside candidate universe."
            )

        print(
            f"Prediction origin:          "
            f"{origin.date()}"
        )

        print(
            f"Candidate startups:         "
            f"{len(candidates):,}"
        )

        print(
            f"All target events:          "
            f"{len(targets):,}"
        )

        print(
            f"Origin-eligible targets:     "
            f"{len(eligible):,}"
        )

        print(
            f"Excluded/unresolved targets: "
            f"{len(excluded):,}"
        )

        print(
            f"Primary events:             "
            f"{int(targets['is_primary_stratum'].sum()):,}"
        )

        print(
            f"Eligible primary events:    "
            f"{eligible_primary:,}"
        )

        candidate_frames.append(
            candidates
        )

        eligible_target_frames.append(
            eligible
        )

        excluded_target_frames.append(
            excluded
        )

        summary_rows.append(
            {
                "origin":
                    label,

                "history_end":
                    origin.date(),

                "candidate_startups":
                    len(
                        candidates
                    ),

                "all_target_events":
                    len(
                        targets
                    ),

                "origin_eligible_targets":
                    len(
                        eligible
                    ),

                "noneligible_targets":
                    len(
                        excluded
                    ),

                "all_primary_events":
                    int(
                        targets[
                            "is_primary_stratum"
                        ].sum()
                    ),

                "eligible_primary_events":
                    eligible_primary,

                "noneligible_primary_events":
                    int(
                        excluded[
                            "is_primary_stratum"
                        ].sum()
                    ),
            }
        )

    # =========================================================================
    # 9. Persist frozen universes
    # =========================================================================

    separator("-")
    print(
        "8.0.9.9 — PERSIST FROZEN "
        "EVALUATION UNIVERSES"
    )
    separator("-")

    candidate_universe = pd.concat(
        candidate_frames,
        ignore_index=True,
    )

    eligible_targets = pd.concat(
        eligible_target_frames,
        ignore_index=True,
    )

    excluded_targets = pd.concat(
        excluded_target_frames,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    candidate_universe.to_parquet(
        CANDIDATE_OUTPUT,
        index=False,
    )

    eligible_targets.to_parquet(
        TARGET_OUTPUT,
        index=False,
    )

    excluded_targets.to_parquet(
        EXCLUDED_TARGET_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # =========================================================================
    # 10. Freeze contract
    # =========================================================================

    contract = {
        "schema_version":
            "PHASE_8_0_9_ORIGIN_ELIGIBILITY_CONTRACT_V4",

        "status":
            "FROZEN",

        "prediction_semantics":
            (
                "Fixed-origin ranking. "
                "All recommendations for a fold are "
                "conditioned only on information available "
                "at that fold's history_end."
            ),

        "startup_origin_eligibility_rule": {
            "operator":
                "OR",

            "evidence": [
                "founded_on <= prediction_origin",
                (
                    "first canonical investment event "
                    "<= prediction_origin"
                ),
                (
                    "first frozen timestamped acquisition "
                    "evidence <= prediction_origin"
                ),
            ],
        },

        "unresolved_policy":
            (
                "Exclude from the primary fixed-origin "
                "candidate universe and target evaluation. "
                "Interpret as existence not established, "
                "not as proven nonexistent."
            ),

        "primary_thesis_stratum":
            (
                "new-to-investor pair + warm investor + "
                "startup with no prior investment history + "
                "startup existence proven by prediction origin"
            ),

        "candidate_universe_membership_uses_fold_outcomes":
            False,

        "other_within_fold_positive_labels_used":
            False,

        "phase6_metric_direct_comparability":
            False,

        "phase6_comparison_policy":
            (
                "Any Phase-8 comparison against reproduced ITRS "
                "must score the baseline under the same Phase-8 "
                "eligible cases, candidate universe, candidate "
                "lists, and graph-information policy."
            ),

        "april_confirmatory_performance_inspected":
            False,

        "model_selection_performed":
            False,
    }

    write_json(
        CONTRACT_OUTPUT,
        contract,
    )

    # =========================================================================
    # 11. Manifest and hashes
    # =========================================================================

    manifest = {
        "phase":
            "8.0.9",

        "authoritative_inputs":
            source_hashes,

        "company_files": [
            path.name
            for path in company_files
        ],

        "origins":
            ORIGINS,

        "outputs": [
            str(
                REGISTRY_OUTPUT
            ),
            str(
                CANDIDATE_OUTPUT
            ),
            str(
                TARGET_OUTPUT
            ),
            str(
                EXCLUDED_TARGET_OUTPUT
            ),
            str(
                SUMMARY_OUTPUT
            ),
            str(
                CONTRACT_OUTPUT
            ),
        ],

        "prohibited_operations": [
            "model_training",
            "model_inference",
            "ranking_metric_scoring",
            "negative_generation",
            "model_selection",
            "april_confirmatory_scoring",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    derived_hashes = {
        str(REGISTRY_OUTPUT):
            sha256_file(
                REGISTRY_OUTPUT
            ),

        str(CANDIDATE_OUTPUT):
            sha256_file(
                CANDIDATE_OUTPUT
            ),

        str(TARGET_OUTPUT):
            sha256_file(
                TARGET_OUTPUT
            ),

        str(EXCLUDED_TARGET_OUTPUT):
            sha256_file(
                EXCLUDED_TARGET_OUTPUT
            ),

        str(SUMMARY_OUTPUT):
            sha256_file(
                SUMMARY_OUTPUT
            ),

        str(CONTRACT_OUTPUT):
            sha256_file(
                CONTRACT_OUTPUT
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
    # 12. Final
    # =========================================================================

    separator()

    print(
        "PHASE 8.0.9 ORIGIN-ELIGIBILITY "
        "CONTRACT FREEZE COMPLETE"
    )

    separator()

    print()

    print(
        "STARTUP EXISTENCE RULE:             FROZEN"
    )

    print(
        "ORIGIN-SPECIFIC CANDIDATE UNIVERSE:FROZEN"
    )

    print(
        "ELIGIBLE TARGET MEMBERSHIP:         FROZEN"
    )

    print(
        "PRIMARY THESIS STRATUM:             FROZEN"
    )

    print(
        "NEGATIVE CANDIDATES GENERATED:      NO"
    )

    print(
        "MODEL PERFORMANCE SCORED:           NO"
    )

    print(
        "MODEL SELECTION PERFORMED:          NO"
    )

    print(
        "APRIL CONFIRMATORY PERFORMANCE:     UNTOUCHED"
    )

    print()

    print(
        "Next:"
    )

    print(
        "Freeze the complete Phase-8 "
        "development / model-selection / "
        "confirmatory evaluation protocol."
    )

    print()

    print(
        f"Outputs written to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()