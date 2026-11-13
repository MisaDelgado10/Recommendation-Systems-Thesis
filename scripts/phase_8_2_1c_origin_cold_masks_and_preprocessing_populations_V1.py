from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 8.2.1C — ORIGIN COLD/WARM MASKS + PREPROCESSING POPULATIONS V1
# =============================================================================
#
# Scientific role:
#
#   OUTCOME-INDEPENDENT FIT-POPULATION MATERIALIZATION
#
# PURPOSE
# -------
#
# For each frozen prediction origin:
#
#   T57
#   T58
#   T59
#   T60
#   APRIL_CONFIRMATORY
#
# materialize, for every one of the frozen 311,589 Startup-role nodes:
#
#   1. origin eligibility;
#   2. prior investment count strictly before origin;
#   3. investment-cold / investment-warm status;
#   4. whether Startup covariates/edges are visible during parameter fitting;
#   5. whether the Startup belongs to the supervised preprocessing-fit
#      population.
#
# AUTHORITATIVE COLD DEFINITION
# -----------------------------
#
#   origin_eligible
#   AND
#   ZERO canonical investment interactions strictly before origin.
#
# Critically, this definition does NOT use:
#
#   - evaluation positive IDs;
#   - evaluation negative IDs;
#   - candidate matrices;
#   - future evaluation participation.
#
# PREPROCESSING FIT POPULATION
# ----------------------------
#
# The learned preprocessor is fit ONLY on Startup entities appearing in the
# supervised training interaction stream:
#
#   T57:   T1-T56
#   T58:   T1-T57
#   T59:   T1-T58
#   T60:   T1-T59
#   April: T1-T60
#
# T0 counts as PRIOR INVESTMENT HISTORY for cold/warm classification,
# but T0 does NOT enter the supervised preprocessing-fit population.
#
# No TF-IDF/SVD/vocabulary/scaler is fit in this phase.
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

FOUNDER_HISTORY_V2_MANIFEST = (
    FEATURE_DIR
    / "phase_8_2_1b_manifest_V2.json"
)

ORIGIN_TABLE = (
    FEATURE_DIR
    / "phase_8_2_1b_origin_table_V2.csv"
)

TEMPORAL_INTERACTIONS = Path(
    "data/experimental/phase_2/"
    "temporal/"
    "interactions_itrs_temporal.parquet"
)

ACQUISITION_RESOLVED = Path(
    "data/experimental/phase_3/"
    "audits/"
    "acquisition_resolved_relation_candidates.parquet"
)

MASKING_CONTRACT = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_12a_outcome_independent_cold_start_masking/"
    "phase_8_1_12a_outcome_independent_masking_contract_V1.json"
)

FOUNDER_TEMPORAL_CONTRACT = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_12b_origin_referenced_founder_history/"
    "phase_8_1_12b_origin_referenced_founder_history_contract_V1.json"
)

EXECUTION_CONTRACT = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_12_execution_contract/"
    "phase_8_1_12_execution_contract_V3.json"
)


OUTPUT_DIR = FEATURE_DIR

MASK_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1c_origin_startup_masks_V1.parquet"
)

PREPROCESSING_POPULATION_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1c_preprocessing_fit_population_V1.parquet"
)

EXISTENCE_EVIDENCE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1c_startup_existence_evidence_V1.parquet"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1c_origin_population_summary_V1.csv"
)

MONOTONICITY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1c_temporal_mask_monotonicity_audit_V1.csv"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1c_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_1c_artifact_sha256_V1.json"
)


EXPECTED_STARTUPS = 311_589


# Frozen origin-specific supervised training streams.
ORIGIN_CONFIG = {
    "T57": {
        "max_training_segment": 56,
        "expected_training_events": 1_003_526,
        "expected_origin_eligible": 308_027,
    },

    "T58": {
        "max_training_segment": 57,
        "expected_training_events": 1_026_268,
        "expected_origin_eligible": 308_854,
    },

    "T59": {
        "max_training_segment": 58,
        "expected_training_events": 1_049_605,
        "expected_origin_eligible": 309_611,
    },

    "T60": {
        "max_training_segment": 59,
        "expected_training_events": 1_073_249,
        "expected_origin_eligible": None,
    },

    "APRIL_CONFIRMATORY": {
        "max_training_segment": 60,
        "expected_training_events": 1_095_764,
        "expected_origin_eligible": 311_266,
    },
}


ORIGIN_ORDER = [
    "T57",
    "T58",
    "T59",
    "T60",
    "APRIL_CONFIRMATORY",
]


# Same conservative acquisition endpoint policy audited in Phase 3.2.9.
ACCEPTED_ACQUISITION_ENDPOINT_STATUSES = {
    "name_website_agree",
    "ambiguous_name_resolved_by_website",
    "unique_name_only",
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


def clean_string(series):

    return (
        series
        .astype("string")
        .str.strip()
        .replace(
            "",
            pd.NA,
        )
    )


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    separator()

    print(
        "PHASE 8.2.1C — "
        "ORIGIN COLD/WARM MASKS + "
        "PREPROCESSING POPULATIONS V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "OUTCOME-INDEPENDENT FIT-POPULATION MATERIALIZATION"
    )

    print("Feature discovery reopened:  NO")
    print("TF-IDF fit:                  NO")
    print("SVD fit:                     NO")
    print("Vocabulary fit:              NO")
    print("Scaler fit:                  NO")
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
        "8.2.1C.1 — PREREQUISITE GATES"
    )

    separator("-")

    required = [
        STARTUP_SOURCE,
        FOUNDER_HISTORY_V2_MANIFEST,
        ORIGIN_TABLE,
        TEMPORAL_INTERACTIONS,
        ACQUISITION_RESOLVED,
        MASKING_CONTRACT,
        FOUNDER_TEMPORAL_CONTRACT,
        EXECUTION_CONTRACT,
    ]

    for path in required:

        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

        print(
            f"FOUND {path}"
        )

    founder_manifest = read_json(
        FOUNDER_HISTORY_V2_MANIFEST
    )

    masking_contract = read_json(
        MASKING_CONTRACT
    )

    founder_contract = read_json(
        FOUNDER_TEMPORAL_CONTRACT
    )

    execution = read_json(
        EXECUTION_CONTRACT
    )

    require(
        founder_manifest.get("status")
        ==
        "MATERIALIZED_AND_AUDITED",
        "8.2.1B V2 is not authoritative.",
    )

    require(
        masking_contract.get("status")
        ==
        "FROZEN",
        "8.1.12A masking contract is not frozen.",
    )

    require(
        founder_contract.get("status")
        ==
        "FROZEN",
        "8.1.12B temporal contract is not frozen.",
    )

    require(
        execution.get("status")
        ==
        "FROZEN",
        "8.1.12 V3 is not frozen.",
    )

    require(
        execution.get(
            "training_performed"
        )
        is False,
        "Training unexpectedly occurred.",
    )

    require(
        execution.get(
            "T60_scored"
        )
        is False,
        "T60 unexpectedly scored.",
    )

    require(
        execution.get(
            "April_analyzed"
        )
        is False,
        "April unexpectedly analyzed.",
    )

    print()

    print(
        "Outcome-independent masking gate: PASS"
    )

    # =========================================================================
    # 2. Load frozen Startup universe
    # =========================================================================

    separator("-")

    print(
        "8.2.1C.2 — FROZEN STARTUP UNIVERSE"
    )

    separator("-")

    startups = pd.read_parquet(
        STARTUP_SOURCE,
        columns=[
            "startup_node_index",
            "startup_node_id",
            "startup_id",
            "founded_on",
        ],
    )

    startups[
        "startup_id"
    ] = clean_string(
        startups[
            "startup_id"
        ]
    )

    startups[
        "founded_on"
    ] = pd.to_datetime(
        startups[
            "founded_on"
        ],
        errors="coerce",
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
        "Frozen Startup IDs are not unique.",
    )

    startup_ids = set(
        startups[
            "startup_id"
        ]
        .dropna()
        .astype(str)
    )

    print(
        f"Frozen Startup nodes: "
        f"{len(startups):,}"
    )

    # =========================================================================
    # 3. Load exact prediction origins
    # =========================================================================

    separator("-")

    print(
        "8.2.1C.3 — PREDICTION ORIGINS"
    )

    separator("-")

    origins = pd.read_csv(
        ORIGIN_TABLE
    )

    origins[
        "prediction_origin"
    ] = pd.to_datetime(
        origins[
            "prediction_origin"
        ],
        errors="raise",
    )

    require(
        origins[
            "origin_label"
        ].tolist()
        ==
        ORIGIN_ORDER,
        (
            "Origin order differs from the frozen "
            "T57/T58/T59/T60/April sequence."
        ),
    )

    print(
        origins.to_string(
            index=False
        )
    )

    # =========================================================================
    # 4. Canonical temporal investment history
    # =========================================================================

    separator("-")

    print(
        "8.2.1C.4 — CANONICAL INVESTMENT HISTORY"
    )

    separator("-")

    interactions = pd.read_parquet(
        TEMPORAL_INTERACTIONS,
        columns=[
            "startup_id",
            "announced_on",
            "segment_number",
        ],
    )

    interactions[
        "startup_id"
    ] = clean_string(
        interactions[
            "startup_id"
        ]
    )

    interactions[
        "announced_on"
    ] = pd.to_datetime(
        interactions[
            "announced_on"
        ],
        errors="raise",
    )

    interactions[
        "segment_number"
    ] = (
        interactions[
            "segment_number"
        ]
        .astype(
            "int16"
        )
    )

    require(
        interactions[
            "startup_id"
        ].isin(
            startup_ids
        ).all(),
        (
            "Temporal interactions contain Startup IDs outside "
            "the frozen Startup-role universe."
        ),
    )

    first_investment = (
        interactions
        .groupby(
            "startup_id",
            observed=True,
        )[
            "announced_on"
        ]
        .min()
        .rename(
            "first_investment_on"
        )
        .reset_index()
    )

    print(
        f"Temporal interaction rows: "
        f"{len(interactions):,}"
    )

    print(
        f"Startups with <=T60 interaction: "
        f"{first_investment['startup_id'].nunique():,}"
    )

    # =========================================================================
    # 5. Conservative timestamped acquisition existence evidence
    #
    # IMPORTANT:
    # Use the pre-holdout-mask resolved acquisition artifact and reproduce
    # Phase-3.2.9 endpoint acceptance. This makes acquisition evidence an
    # existence signal independent of evaluation-pair membership.
    # =========================================================================

    separator("-")

    print(
        "8.2.1C.5 — TIMESTAMPED ACQUISITION EXISTENCE EVIDENCE"
    )

    separator("-")

    acquisitions = pd.read_parquet(
        ACQUISITION_RESOLVED,
        columns=[
            "announced_on",
            "acquirer_resolved_id",
            "acquiree_resolved_id",
            "acquirer_resolution_status",
            "acquiree_resolution_status",
        ],
    )

    acquisitions[
        "announced_on"
    ] = pd.to_datetime(
        acquisitions[
            "announced_on"
        ],
        errors="coerce",
    )

    acquisitions[
        "acquirer_resolved_id"
    ] = clean_string(
        acquisitions[
            "acquirer_resolved_id"
        ]
    )

    acquisitions[
        "acquiree_resolved_id"
    ] = clean_string(
        acquisitions[
            "acquiree_resolved_id"
        ]
    )

    accepted = acquisitions.loc[
        acquisitions[
            "announced_on"
        ].notna()
        &
        acquisitions[
            "acquirer_resolution_status"
        ].isin(
            ACCEPTED_ACQUISITION_ENDPOINT_STATUSES
        )
        &
        acquisitions[
            "acquiree_resolution_status"
        ].isin(
            ACCEPTED_ACQUISITION_ENDPOINT_STATUSES
        )
    ].copy()

    acquirer_evidence = (
        accepted.loc[
            accepted[
                "acquirer_resolved_id"
            ].isin(
                startup_ids
            ),
            [
                "acquirer_resolved_id",
                "announced_on",
            ],
        ]
        .rename(
            columns={
                "acquirer_resolved_id":
                    "startup_id",
            }
        )
    )

    acquiree_evidence = (
        accepted.loc[
            accepted[
                "acquiree_resolved_id"
            ].isin(
                startup_ids
            ),
            [
                "acquiree_resolved_id",
                "announced_on",
            ],
        ]
        .rename(
            columns={
                "acquiree_resolved_id":
                    "startup_id",
            }
        )
    )

    acquisition_evidence = pd.concat(
        [
            acquirer_evidence,
            acquiree_evidence,
        ],
        ignore_index=True,
    )

    first_acquisition = (
        acquisition_evidence
        .groupby(
            "startup_id",
            observed=True,
        )[
            "announced_on"
        ]
        .min()
        .rename(
            "first_acquisition_on"
        )
        .reset_index()
    )

    print(
        f"Accepted acquisition rows: "
        f"{len(accepted):,}"
    )

    print(
        f"Frozen Startups with timestamped "
        f"acquisition evidence: "
        f"{first_acquisition['startup_id'].nunique():,}"
    )

    # =========================================================================
    # 6. One-row-per-Startup existence evidence substrate
    # =========================================================================

    separator("-")

    print(
        "8.2.1C.6 — STARTUP EXISTENCE EVIDENCE SUBSTRATE"
    )

    separator("-")

    existence = (
        startups
        .merge(
            first_investment,
            on="startup_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            first_acquisition,
            on="startup_id",
            how="left",
            validate="one_to_one",
        )
    )

    require(
        len(existence)
        ==
        EXPECTED_STARTUPS,
        "Existence substrate row count drift.",
    )

    existence[
        "has_founded_on"
    ] = (
        existence[
            "founded_on"
        ].notna()
    )

    existence[
        "has_investment_evidence"
    ] = (
        existence[
            "first_investment_on"
        ].notna()
    )

    existence[
        "has_acquisition_evidence"
    ] = (
        existence[
            "first_acquisition_on"
        ].notna()
    )

    existence.to_parquet(
        EXISTENCE_EVIDENCE_OUTPUT,
        index=False,
    )

    print(
        f"founded_on evidence:   "
        f"{int(existence['has_founded_on'].sum()):,}"
    )

    print(
        f"investment evidence:   "
        f"{int(existence['has_investment_evidence'].sum()):,}"
    )

    print(
        f"acquisition evidence:  "
        f"{int(existence['has_acquisition_evidence'].sum()):,}"
    )

    # =========================================================================
    # 7. Materialize origin-specific masks
    # =========================================================================

    separator("-")

    print(
        "8.2.1C.7 — ORIGIN-SPECIFIC COLD/WARM MASKS"
    )

    separator("-")

    mask_parts = []
    fit_population_parts = []
    summary_rows = []

    for origin_row in origins.itertuples(
        index=False
    ):

        label = (
            origin_row.origin_label
        )

        origin = pd.Timestamp(
            origin_row.prediction_origin
        )

        config = (
            ORIGIN_CONFIG[
                label
            ]
        )

        max_train_segment = int(
            config[
                "max_training_segment"
            ]
        )

        print()

        print(
            f"[{label}] prediction origin = "
            f"{origin.date()}"
        )

        # ---------------------------------------------------------------------
        # Frozen existence policy:
        #
        #   founded_on <= origin
        #   OR first investment <= origin
        #   OR first accepted timestamped acquisition <= origin
        #
        # This is EXISTENCE evidence only.
        #
        # Cold/warm itself remains STRICTLY based on investments < origin.
        # ---------------------------------------------------------------------

        frame = existence[
            [
                "startup_node_index",
                "startup_node_id",
                "startup_id",
                "founded_on",
                "first_investment_on",
                "first_acquisition_on",
            ]
        ].copy()

        frame[
            "origin_label"
        ] = label

        frame[
            "prediction_origin"
        ] = origin

        frame[
            "eligible_by_founded_on"
        ] = (
            frame[
                "founded_on"
            ].notna()
            &
            (
                frame[
                    "founded_on"
                ]
                <=
                origin
            )
        )

        frame[
            "eligible_by_investment"
        ] = (
            frame[
                "first_investment_on"
            ].notna()
            &
            (
                frame[
                    "first_investment_on"
                ]
                <=
                origin
            )
        )

        frame[
            "eligible_by_acquisition"
        ] = (
            frame[
                "first_acquisition_on"
            ].notna()
            &
            (
                frame[
                    "first_acquisition_on"
                ]
                <=
                origin
            )
        )

        frame[
            "origin_eligible"
        ] = (
            frame[
                "eligible_by_founded_on"
            ]
            |
            frame[
                "eligible_by_investment"
            ]
            |
            frame[
                "eligible_by_acquisition"
            ]
        )

        # ---------------------------------------------------------------------
        # Outcome-independent prior investment history:
        # strictly before prediction origin.
        # ---------------------------------------------------------------------

        prior = (
            interactions.loc[
                interactions[
                    "announced_on"
                ]
                <
                origin
            ]
            .groupby(
                "startup_id",
                observed=True,
            )
            .size()
            .rename(
                "prior_investment_count"
            )
        )

        frame = frame.merge(
            prior,
            on="startup_id",
            how="left",
            validate="one_to_one",
        )

        frame[
            "prior_investment_count"
        ] = (
            frame[
                "prior_investment_count"
            ]
            .fillna(0)
            .astype(
                "int32"
            )
        )

        frame[
            "investment_cold"
        ] = (
            frame[
                "origin_eligible"
            ]
            &
            frame[
                "prior_investment_count"
            ].eq(0)
        )

        frame[
            "investment_warm"
        ] = (
            frame[
                "origin_eligible"
            ]
            &
            frame[
                "prior_investment_count"
            ].gt(0)
        )

        require(
            not (
                frame[
                    "investment_cold"
                ]
                &
                frame[
                    "investment_warm"
                ]
            ).any(),
            f"{label}: Startup is both cold and warm.",
        )

        require(
            (
                frame[
                    "investment_cold"
                ]
                |
                frame[
                    "investment_warm"
                ]
            ).eq(
                frame[
                    "origin_eligible"
                ]
            ).all(),
            (
                f"{label}: eligible Startup does not partition "
                "exactly into cold/warm."
            ),
        )

        # ---------------------------------------------------------------------
        # Parameter-fit visibility from 8.1.12A:
        #
        #   all origin-cold Startup covariates + incident graph edges masked;
        #   warm eligible Startup covariates may remain visible.
        # ---------------------------------------------------------------------

        frame[
            "fit_startup_features_visible"
        ] = (
            frame[
                "origin_eligible"
            ]
            &
            frame[
                "investment_warm"
            ]
        )

        frame[
            "fit_startup_structural_edges_visible"
        ] = (
            frame[
                "fit_startup_features_visible"
            ]
        )

        frame[
            "inference_startup_features_restorable"
        ] = (
            frame[
                "origin_eligible"
            ]
        )

        frame[
            "cold_inference_sink_only"
        ] = (
            frame[
                "investment_cold"
            ]
        )

        # ---------------------------------------------------------------------
        # Supervised preprocessing-fit population.
        #
        # IMPORTANT:
        #   T0 is intentionally excluded here.
        # ---------------------------------------------------------------------

        training_events = interactions.loc[
            interactions[
                "segment_number"
            ].between(
                1,
                max_train_segment,
            )
        ].copy()

        expected_training_events = int(
            config[
                "expected_training_events"
            ]
        )

        require(
            len(training_events)
            ==
            expected_training_events,
            (
                f"{label}: supervised training-event count drift. "
                f"Expected {expected_training_events:,}; "
                f"found {len(training_events):,}."
            ),
        )

        supervised_ids = set(
            training_events[
                "startup_id"
            ]
            .dropna()
            .astype(str)
            .unique()
        )

        frame[
            "supervised_training_startup"
        ] = (
            frame[
                "startup_id"
            ].isin(
                supervised_ids
            )
        )

        frame[
            "preprocessing_fit_startup"
        ] = (
            frame[
                "supervised_training_startup"
            ]
        )

        # Every supervised-training Startup must necessarily be warm at origin.
        require(
            frame.loc[
                frame[
                    "preprocessing_fit_startup"
                ],
                "investment_warm",
            ].all(),
            (
                f"{label}: preprocessing-fit population contains "
                "an investment-cold Startup."
            ),
        )

        require(
            frame.loc[
                frame[
                    "preprocessing_fit_startup"
                ],
                "origin_eligible",
            ].all(),
            (
                f"{label}: preprocessing-fit population contains "
                "an origin-ineligible Startup."
            ),
        )

        # Every preprocessing-fit Startup is visible during fitting.
        require(
            frame.loc[
                frame[
                    "preprocessing_fit_startup"
                ],
                "fit_startup_features_visible",
            ].all(),
            (
                f"{label}: preprocessing-fit Startup is masked "
                "during parameter fitting."
            ),
        )

        # ---------------------------------------------------------------------
        # Frozen origin-eligibility count cross-checks where available.
        # ---------------------------------------------------------------------

        origin_eligible_count = int(
            frame[
                "origin_eligible"
            ].sum()
        )

        expected_origin_eligible = (
            config[
                "expected_origin_eligible"
            ]
        )

        if expected_origin_eligible is not None:

            require(
                origin_eligible_count
                ==
                int(
                    expected_origin_eligible
                ),
                (
                    f"{label}: origin-eligible count differs from "
                    "the frozen Phase-8.0 existence audit. "
                    f"Expected {int(expected_origin_eligible):,}; "
                    f"found {origin_eligible_count:,}."
                ),
            )

        cold_count = int(
            frame[
                "investment_cold"
            ].sum()
        )

        warm_count = int(
            frame[
                "investment_warm"
            ].sum()
        )

        fit_count = int(
            frame[
                "preprocessing_fit_startup"
            ].sum()
        )

        fit_population = (
            frame.loc[
                frame[
                    "preprocessing_fit_startup"
                ],
                [
                    "origin_label",
                    "prediction_origin",
                    "startup_node_index",
                    "startup_node_id",
                    "startup_id",
                    "prior_investment_count",
                ],
            ]
            .copy()
        )

        fit_population[
            "preprocessing_fit_reason"
        ] = (
            f"startup appears in supervised interactions "
            f"T1-T{max_train_segment}"
        )

        fit_population_parts.append(
            fit_population
        )

        mask_parts.append(
            frame
        )

        summary_rows.append(
            {
                "origin_label":
                    label,

                "prediction_origin":
                    origin,

                "max_training_segment":
                    max_train_segment,

                "supervised_training_events":
                    len(
                        training_events
                    ),

                "supervised_training_unique_startups":
                    fit_count,

                "origin_eligible_startups":
                    origin_eligible_count,

                "investment_warm_startups":
                    warm_count,

                "investment_cold_startups":
                    cold_count,

                "fit_feature_visible_startups":
                    int(
                        frame[
                            "fit_startup_features_visible"
                        ].sum()
                    ),

                "masked_cold_startups":
                    cold_count,

                "outside_origin_eligibility":
                    int(
                        (
                            ~frame[
                                "origin_eligible"
                            ]
                        ).sum()
                    ),
            }
        )

        print(
            f"  origin eligible:             "
            f"{origin_eligible_count:,}"
        )

        print(
            f"  investment warm:             "
            f"{warm_count:,}"
        )

        print(
            f"  investment cold:             "
            f"{cold_count:,}"
        )

        print(
            f"  supervised training events:  "
            f"{len(training_events):,}"
        )

        print(
            f"  preprocessing-fit Startups:  "
            f"{fit_count:,}"
        )

    # =========================================================================
    # 8. Consolidate masks and populations
    # =========================================================================

    separator("-")

    print(
        "8.2.1C.8 — CONSOLIDATED ORIGIN POPULATIONS"
    )

    separator("-")

    masks = pd.concat(
        mask_parts,
        ignore_index=True,
    )

    preprocessing_population = pd.concat(
        fit_population_parts,
        ignore_index=True,
    )

    expected_mask_rows = (
        EXPECTED_STARTUPS
        *
        len(
            ORIGIN_ORDER
        )
    )

    require(
        len(masks)
        ==
        expected_mask_rows,
        (
            "Consolidated mask row count mismatch: "
            f"expected {expected_mask_rows:,}, "
            f"found {len(masks):,}."
        ),
    )

    require(
        not masks.duplicated(
            subset=[
                "origin_label",
                "startup_id",
            ]
        ).any(),
        (
            "Duplicate origin/Startup rows in "
            "mask artifact."
        ),
    )

    require(
        not preprocessing_population.duplicated(
            subset=[
                "origin_label",
                "startup_id",
            ]
        ).any(),
        (
            "Duplicate origin/Startup rows in "
            "preprocessing population."
        ),
    )

    masks.to_parquet(
        MASK_OUTPUT,
        index=False,
    )

    preprocessing_population.to_parquet(
        PREPROCESSING_POPULATION_OUTPUT,
        index=False,
    )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # 9. Temporal monotonicity / irreversibility audits
    # =========================================================================

    separator("-")

    print(
        "8.2.1C.9 — TEMPORAL MASK MONOTONICITY"
    )

    separator("-")

    eligibility_wide = (
        masks[
            [
                "startup_id",
                "origin_label",
                "origin_eligible",
            ]
        ]
        .pivot(
            index="startup_id",
            columns="origin_label",
            values="origin_eligible",
        )
        .reindex(
            columns=ORIGIN_ORDER
        )
    )

    warm_wide = (
        masks[
            [
                "startup_id",
                "origin_label",
                "investment_warm",
            ]
        ]
        .pivot(
            index="startup_id",
            columns="origin_label",
            values="investment_warm",
        )
        .reindex(
            columns=ORIGIN_ORDER
        )
    )

    prior_count_wide = (
        masks[
            [
                "startup_id",
                "origin_label",
                "prior_investment_count",
            ]
        ]
        .pivot(
            index="startup_id",
            columns="origin_label",
            values="prior_investment_count",
        )
        .reindex(
            columns=ORIGIN_ORDER
        )
    )

    fit_wide = (
        masks[
            [
                "startup_id",
                "origin_label",
                "preprocessing_fit_startup",
            ]
        ]
        .pivot(
            index="startup_id",
            columns="origin_label",
            values="preprocessing_fit_startup",
        )
        .reindex(
            columns=ORIGIN_ORDER
        )
    )

    monotonicity_rows = []

    for earlier, later in zip(
        ORIGIN_ORDER[:-1],
        ORIGIN_ORDER[1:],
    ):

        eligibility_loss = int(
            (
                eligibility_wide[
                    earlier
                ].astype(bool)
                &
                ~eligibility_wide[
                    later
                ].astype(bool)
            ).sum()
        )

        warm_to_nonwarm = int(
            (
                warm_wide[
                    earlier
                ].astype(bool)
                &
                ~warm_wide[
                    later
                ].astype(bool)
            ).sum()
        )

        prior_count_decrease = int(
            (
                prior_count_wide[
                    later
                ]
                <
                prior_count_wide[
                    earlier
                ]
            ).sum()
        )

        preprocessing_loss = int(
            (
                fit_wide[
                    earlier
                ].astype(bool)
                &
                ~fit_wide[
                    later
                ].astype(bool)
            ).sum()
        )

        monotonicity_rows.append(
            {
                "earlier_origin":
                    earlier,

                "later_origin":
                    later,

                "eligibility_loss_violations":
                    eligibility_loss,

                "warm_to_nonwarm_violations":
                    warm_to_nonwarm,

                "prior_investment_count_decrease_violations":
                    prior_count_decrease,

                "preprocessing_population_loss_violations":
                    preprocessing_loss,

                "status":
                    (
                        "PASS"
                        if (
                            eligibility_loss
                            ==
                            0
                            and
                            warm_to_nonwarm
                            ==
                            0
                            and
                            prior_count_decrease
                            ==
                            0
                            and
                            preprocessing_loss
                            ==
                            0
                        )
                        else
                        "FAIL"
                    ),
            }
        )

        print(
            f"{earlier:>20} -> {later:<20} "
            f"eligibility_loss={eligibility_loss:,}  "
            f"warm_reversal={warm_to_nonwarm:,}  "
            f"history_decrease={prior_count_decrease:,}  "
            f"fit_population_loss={preprocessing_loss:,}"
        )

    monotonicity_df = pd.DataFrame(
        monotonicity_rows
    )

    require(
        monotonicity_df[
            "status"
        ].eq(
            "PASS"
        ).all(),
        (
            "At least one temporal cold/warm or preprocessing "
            "population monotonicity invariant failed."
        ),
    )

    monotonicity_df.to_csv(
        MONOTONICITY_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 10. Outcome-independence / leakage boundary
    # =========================================================================

    separator("-")

    print(
        "8.2.1C.10 — OUTCOME-INDEPENDENCE / LEAKAGE BOUNDARY"
    )

    separator("-")

    boundary = {
        "candidate_universe":
            "FROZEN_311589_STARTUP_ROLE_NODES",

        "origin_eligibility_rule":
            (
                "founded_on <= origin OR "
                "first_investment_on <= origin OR "
                "first_accepted_acquisition_on <= origin"
            ),

        "cold_definition":
            (
                "origin_eligible AND zero investment "
                "interactions strictly before origin"
            ),

        "T0_counts_for_cold_warm_history":
            True,

        "T0_used_for_supervised_preprocessing_fit":
            False,

        "preprocessing_fit_population":
            (
                "unique Startups appearing in supervised "
                "training events T1 through origin-1 segment"
            ),

        "evaluation_positive_ids_loaded":
            False,

        "evaluation_negative_ids_loaded":
            False,

        "evaluation_candidate_matrix_loaded":
            False,

        "future_evaluation_membership_loaded":
            False,

        "post_origin_investments_used_for_cold_mask":
            False,

        "cold_startup_features_visible_during_fit":
            False,

        "cold_startup_edges_visible_during_fit":
            False,

        "cold_startup_features_restorable_at_inference":
            True,

        "cold_startup_outgoing_messages_at_inference":
            False,

        "TFIDF_fit":
            False,

        "SVD_fit":
            False,

        "category_vocabulary_fit":
            False,

        "structured_vocabulary_fit":
            False,

        "numeric_scaler_fit":
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
            f"{key:<52} {value}"
        )

    # =========================================================================
    # 11. Persist manifest
    # =========================================================================

    separator("-")

    print(
        "8.2.1C.11 — PERSIST MASK / FIT-POPULATION CONTRACT"
    )

    separator("-")

    manifest = {
        "schema_version":
            "PHASE_8_2_1C_ORIGIN_MASKS_AND_PREPROCESSING_POPULATIONS_V1",

        "phase":
            "8.2.1C",

        "status":
            "MATERIALIZED_AND_AUDITED",

        "scientific_role":
            "OUTCOME_INDEPENDENT_FIT_POPULATION_MATERIALIZATION",

        "startup_universe":
            EXPECTED_STARTUPS,

        "origins":
            ORIGIN_ORDER,

        "cold_definition":
            (
                "origin_eligible AND zero canonical investment "
                "interactions strictly before prediction origin"
            ),

        "origin_eligibility_rule":
            (
                "founded_on <= origin OR first investment <= origin "
                "OR first conservative timestamped acquisition <= origin"
            ),

        "T0_counts_as_prior_investment_history":
            True,

        "T0_used_for_supervised_preprocessor_fit":
            False,

        "preprocessing_fit_rule":
            (
                "Startup appears in supervised training interactions "
                "T1 through the origin-specific final training segment."
            ),

        "cold_features_zero_during_fit":
            True,

        "cold_structural_edges_masked_during_fit":
            True,

        "evaluation_membership_used":
            False,

        "evaluation_candidate_matrix_used":
            False,

        "learned_preprocessing_fit":
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
            "origin_masks":
                str(
                    MASK_OUTPUT
                ),

            "preprocessing_fit_population":
                str(
                    PREPROCESSING_POPULATION_OUTPUT
                ),

            "existence_evidence":
                str(
                    EXISTENCE_EVIDENCE_OUTPUT
                ),

            "origin_summary":
                str(
                    SUMMARY_OUTPUT
                ),

            "monotonicity_audit":
                str(
                    MONOTONICITY_OUTPUT
                ),
        },

        "next_step":
            (
                "Phase 8.2.1D — fit origin-specific preprocessing "
                "artifacts using ONLY the frozen preprocessing-fit "
                "populations, then transform all eligible Startup rows."
            ),
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    outputs = [
        MASK_OUTPUT,
        PREPROCESSING_POPULATION_OUTPUT,
        EXISTENCE_EVIDENCE_OUTPUT,
        SUMMARY_OUTPUT,
        MONOTONICITY_OUTPUT,
        MANIFEST_OUTPUT,
    ]

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
        "PHASE 8.2.1C ORIGIN MASKS + "
        "PREPROCESSING POPULATIONS MATERIALIZED"
    )

    separator()

    print()

    print(
        "COLD DEFINITION:"
    )

    print(
        "  origin eligible"
    )

    print(
        "  AND zero investment interactions strictly before origin"
    )

    print()

    print(
        "T0 COUNTS AS PRIOR HISTORY:          YES"
    )

    print(
        "T0 USED FOR PREPROCESSOR FIT:        NO"
    )

    print(
        "EVALUATION POSITIVE IDS USED:        NO"
    )

    print(
        "EVALUATION NEGATIVE IDS USED:        NO"
    )

    print(
        "CANDIDATE MATRICES USED:             NO"
    )

    print(
        "TEMPORAL MONOTONICITY:               PASS"
    )

    print(
        "LEARNED PREPROCESSING FIT:           NO"
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
        "Phase 8.2.1D — origin-specific "
        "training-only preprocessing fit + "
        "feature transformation."
    )


if __name__ == "__main__":
    main()
