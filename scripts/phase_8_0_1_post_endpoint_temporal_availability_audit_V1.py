from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.0.1 — POST-ENDPOINT TEMPORAL AVAILABILITY AUDIT V1
# =============================================================================
#
# Scientific role:
#
#   OUTCOME-BLIND EVALUATION-DESIGN AUDIT ONLY
#
# This script:
#   - does NOT load a model;
#   - does NOT load a checkpoint;
#   - does NOT run inference;
#   - does NOT generate negatives;
#   - does NOT score validation or test data;
#   - does NOT select a holdout;
#   - does NOT select an architecture;
#   - does NOT perform model selection.
#
# Objective:
#
#   Determine what canonical investment-event data exists strictly after the
#   frozen Phase-2/6 reproduction endpoint of 2026-03-31, and whether that
#   future tail contains sufficient thesis-relevant populations to justify
#   designing a new confirmatory temporal holdout.
#
# =============================================================================


CANONICAL_PATH = Path(
    "data/processed/interactions.parquet"
)

FROZEN_PHASE2_POST_ENDPOINT_PATH = Path(
    "data/experimental/phase_2/temporal/"
    "post_endpoint_interactions.parquet"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_1_post_endpoint_availability"
)

MONTHLY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_1_post_endpoint_monthly_availability_V1.csv"
)

STRATUM_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_1_post_endpoint_stratum_counts_V1.csv"
)

WINDOW_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_1_candidate_window_counts_V1.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_1_post_endpoint_audit_summary_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_1_analysis_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_1_derived_artifact_sha256_V1.json"
)


# -----------------------------------------------------------------------------
# Frozen Phase-2 facts
# -----------------------------------------------------------------------------

EXPECTED_CANONICAL_ROWS = 1_208_051

EXPECTED_TEMPORAL_ROWS = 1_195_937

EXPECTED_POST_ENDPOINT_ROWS = 12_114

EXPECTED_CANONICAL_MIN_DATE = pd.Timestamp(
    "1945-01-01"
)

EXPECTED_CANONICAL_MAX_DATE = pd.Timestamp(
    "2026-06-01"
)

FROZEN_EXPERIMENT_END = pd.Timestamp(
    "2026-03-31"
)

FUTURE_START = pd.Timestamp(
    "2026-04-01"
)


# -----------------------------------------------------------------------------
# Canonical fields already audited in Phase 2
# -----------------------------------------------------------------------------

ID_COL = "interaction_id"
DATE_COL = "announced_on"
INVESTOR_COL = "investor_id"
STARTUP_COL = "startup_id"
ROUND_COL = "funding_round_id"

REQUIRED_COLUMNS = [
    ID_COL,
    DATE_COL,
    INVESTOR_COL,
    STARTUP_COL,
    ROUND_COL,
]


def separator(char="=", width=110):
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
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            default=str,
        )


def classify_endpoint_relative_stratum(row):
    if not row["new_to_investor_pair_at_endpoint"]:
        return "prior_pair"

    cold_i = row["cold_start_investor_at_endpoint"]
    cold_s = row["cold_start_startup_at_endpoint"]

    if not cold_i and not cold_s:
        return "new_warm_investor_warm_startup"

    if not cold_i and cold_s:
        return "new_warm_investor_cold_startup"

    if cold_i and not cold_s:
        return "new_cold_investor_warm_startup"

    return "new_dual_cold"


def summarize_window(
    df,
    name,
    start_date,
    end_date,
):
    subset = df[
        (df[DATE_COL] >= start_date)
        & (df[DATE_COL] <= end_date)
    ].copy()

    return {
        "window": name,
        "start_date": start_date.date(),
        "end_date": end_date.date(),
        "interactions": int(len(subset)),
        "unique_investors": int(
            subset[INVESTOR_COL].nunique()
        ),
        "unique_startups": int(
            subset[STARTUP_COL].nunique()
        ),
        "unique_pairs": int(
            subset[
                [INVESTOR_COL, STARTUP_COL]
            ].drop_duplicates().shape[0]
        ),
        "funding_rounds": int(
            subset[ROUND_COL].nunique()
        ),
        "new_to_investor_events": int(
            subset[
                "new_to_investor_pair_at_endpoint"
            ].sum()
        ),
        "new_warm_investor_warm_startup": int(
            (
                subset["analysis_stratum"]
                == "new_warm_investor_warm_startup"
            ).sum()
        ),
        "new_warm_investor_cold_startup": int(
            (
                subset["analysis_stratum"]
                == "new_warm_investor_cold_startup"
            ).sum()
        ),
        "new_cold_investor_warm_startup": int(
            (
                subset["analysis_stratum"]
                == "new_cold_investor_warm_startup"
            ).sum()
        ),
        "new_dual_cold": int(
            (
                subset["analysis_stratum"]
                == "new_dual_cold"
            ).sum()
        ),
        "prior_pair": int(
            (
                subset["analysis_stratum"]
                == "prior_pair"
            ).sum()
        ),
    }


def main():

    separator()
    print(
        "PHASE 8.0.1 — "
        "POST-ENDPOINT TEMPORAL AVAILABILITY AUDIT V1"
    )
    separator()

    print()
    print(
        "Scientific role:             "
        "OUTCOME-BLIND EVALUATION-DESIGN AUDIT ONLY"
    )
    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Inference executed:          NO")
    print("Validation scored:           NO")
    print("Test scored:                 NO")
    print("Negatives generated:         NO")
    print("Model selection performed:   NO")
    print("Holdout selected:            NO")

    # =========================================================================
    # 1. Canonical source integrity
    # =========================================================================

    separator("-")
    print("8.0.1.1 — CANONICAL SOURCE INTEGRITY")
    separator("-")

    if not CANONICAL_PATH.exists():
        raise FileNotFoundError(
            f"Missing canonical interaction file: "
            f"{CANONICAL_PATH}"
        )

    canonical_sha = sha256_file(
        CANONICAL_PATH
    )

    print(
        f"Canonical file:    {CANONICAL_PATH}"
    )
    print(
        f"Canonical SHA256:  {canonical_sha}"
    )

    df = pd.read_parquet(
        CANONICAL_PATH,
        columns=REQUIRED_COLUMNS,
    )

    print(
        f"Canonical rows:    {len(df):,}"
    )

    if len(df) != EXPECTED_CANONICAL_ROWS:
        raise ValueError(
            "Canonical row-count drift detected. "
            f"Expected {EXPECTED_CANONICAL_ROWS:,}, "
            f"found {len(df):,}."
        )

    if df[ID_COL].isna().any():
        raise ValueError(
            "Missing interaction_id detected."
        )

    if df[ID_COL].duplicated().any():
        raise ValueError(
            "Duplicate interaction_id detected."
        )

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="raise",
    )

    if df[DATE_COL].isna().any():
        raise ValueError(
            "Missing announced_on detected."
        )

    canonical_min = df[DATE_COL].min()
    canonical_max = df[DATE_COL].max()

    print(
        f"Canonical range:   "
        f"{canonical_min.date()} -> "
        f"{canonical_max.date()}"
    )

    if canonical_min != EXPECTED_CANONICAL_MIN_DATE:
        raise ValueError(
            "Canonical minimum date differs from "
            "the frozen Phase-2 record."
        )

    if canonical_max != EXPECTED_CANONICAL_MAX_DATE:
        raise ValueError(
            "Canonical maximum date differs from "
            "the frozen Phase-2 record."
        )

    print("Canonical source integrity: PASS")

    # =========================================================================
    # 2. Reconstruct frozen Phase-2 temporal partition
    # =========================================================================

    separator("-")
    print(
        "8.0.1.2 — FROZEN PHASE-2 "
        "ENDPOINT RECONSTRUCTION"
    )
    separator("-")

    historical = df[
        df[DATE_COL]
        <= FROZEN_EXPERIMENT_END
    ].copy()

    post_endpoint = df[
        df[DATE_COL]
        > FROZEN_EXPERIMENT_END
    ].copy()

    print(
        f"Frozen endpoint:             "
        f"{FROZEN_EXPERIMENT_END.date()}"
    )

    print(
        f"Rows <= frozen endpoint:     "
        f"{len(historical):,}"
    )

    print(
        f"Rows > frozen endpoint:      "
        f"{len(post_endpoint):,}"
    )

    if len(historical) != EXPECTED_TEMPORAL_ROWS:
        raise ValueError(
            "Historical temporal row count does not "
            "match frozen Phase-2 value."
        )

    if (
        len(post_endpoint)
        != EXPECTED_POST_ENDPOINT_ROWS
    ):
        raise ValueError(
            "Post-endpoint row count does not match "
            "frozen Phase-2 value."
        )

    if (
        len(historical)
        + len(post_endpoint)
        != len(df)
    ):
        raise ValueError(
            "Historical + post-endpoint partition "
            "does not reconstruct canonical data."
        )

    print(
        f"Post-endpoint range:         "
        f"{post_endpoint[DATE_COL].min().date()} "
        f"-> "
        f"{post_endpoint[DATE_COL].max().date()}"
    )

    print("Frozen endpoint reconstruction: PASS")

    # =========================================================================
    # 3. Optional reconciliation with frozen Phase-2 artifact
    # =========================================================================

    separator("-")
    print(
        "8.0.1.3 — PHASE-2 POST-ENDPOINT "
        "ARTIFACT RECONCILIATION"
    )
    separator("-")

    phase2_artifact_present = (
        FROZEN_PHASE2_POST_ENDPOINT_PATH.exists()
    )

    phase2_artifact_sha = None
    phase2_id_match = None

    if phase2_artifact_present:

        frozen_tail = pd.read_parquet(
            FROZEN_PHASE2_POST_ENDPOINT_PATH,
            columns=[ID_COL],
        )

        phase2_artifact_sha = sha256_file(
            FROZEN_PHASE2_POST_ENDPOINT_PATH
        )

        frozen_ids = set(
            frozen_tail[ID_COL]
        )

        reconstructed_ids = set(
            post_endpoint[ID_COL]
        )

        phase2_id_match = (
            frozen_ids
            == reconstructed_ids
        )

        print(
            f"Frozen artifact:            "
            f"{FROZEN_PHASE2_POST_ENDPOINT_PATH}"
        )

        print(
            f"Artifact rows:              "
            f"{len(frozen_tail):,}"
        )

        print(
            f"Artifact SHA256:            "
            f"{phase2_artifact_sha}"
        )

        print(
            f"Exact interaction-ID match: "
            f"{phase2_id_match}"
        )

        if (
            len(frozen_tail)
            != EXPECTED_POST_ENDPOINT_ROWS
        ):
            raise ValueError(
                "Frozen Phase-2 post-endpoint artifact "
                "has unexpected row count."
            )

        if not phase2_id_match:
            raise ValueError(
                "Reconstructed post-endpoint IDs differ "
                "from frozen Phase-2 artifact."
            )

    else:

        print(
            "Frozen Phase-2 post-endpoint artifact "
            "not present in this workspace."
        )

        print(
            "Canonical reconstruction will remain "
            "the authoritative source for this audit."
        )

    # =========================================================================
    # 4. Endpoint-relative cold-start and novelty status
    # =========================================================================

    separator("-")
    print(
        "8.0.1.4 — ENDPOINT-RELATIVE "
        "THESIS STRATA"
    )
    separator("-")

    historical_investors = set(
        historical[INVESTOR_COL]
    )

    historical_startups = set(
        historical[STARTUP_COL]
    )

    historical_pairs = set(
        zip(
            historical[INVESTOR_COL],
            historical[STARTUP_COL],
        )
    )

    post_endpoint[
        "cold_start_investor_at_endpoint"
    ] = ~post_endpoint[
        INVESTOR_COL
    ].isin(
        historical_investors
    )

    post_endpoint[
        "cold_start_startup_at_endpoint"
    ] = ~post_endpoint[
        STARTUP_COL
    ].isin(
        historical_startups
    )

    tail_pairs = list(
        zip(
            post_endpoint[INVESTOR_COL],
            post_endpoint[STARTUP_COL],
        )
    )

    post_endpoint[
        "new_to_investor_pair_at_endpoint"
    ] = [
        pair not in historical_pairs
        for pair in tail_pairs
    ]

    post_endpoint[
        "analysis_stratum"
    ] = post_endpoint.apply(
        classify_endpoint_relative_stratum,
        axis=1,
    )

    stratum_order = [
        "prior_pair",
        "new_warm_investor_warm_startup",
        "new_warm_investor_cold_startup",
        "new_cold_investor_warm_startup",
        "new_dual_cold",
    ]

    stratum_counts = (
        post_endpoint[
            "analysis_stratum"
        ]
        .value_counts()
        .reindex(
            stratum_order,
            fill_value=0,
        )
        .rename_axis(
            "analysis_stratum"
        )
        .reset_index(
            name="interactions"
        )
    )

    stratum_counts[
        "share_pct"
    ] = (
        stratum_counts["interactions"]
        / len(post_endpoint)
        * 100
    )

    print(
        stratum_counts.to_string(
            index=False,
            formatters={
                "share_pct":
                    lambda x: f"{x:.2f}%"
            },
        )
    )

    # =========================================================================
    # 5. Monthly temporal availability
    # =========================================================================

    separator("-")
    print(
        "8.0.1.5 — POST-ENDPOINT "
        "MONTHLY AVAILABILITY"
    )
    separator("-")

    post_endpoint["month"] = (
        post_endpoint[DATE_COL]
        .dt.to_period("M")
    )

    monthly_base = (
        post_endpoint.groupby(
            "month",
            observed=True,
        )
        .agg(
            interactions=(
                ID_COL,
                "size",
            ),
            unique_investors=(
                INVESTOR_COL,
                "nunique",
            ),
            unique_startups=(
                STARTUP_COL,
                "nunique",
            ),
            funding_rounds=(
                ROUND_COL,
                "nunique",
            ),
            active_event_dates=(
                DATE_COL,
                "nunique",
            ),
            first_event_date=(
                DATE_COL,
                "min",
            ),
            last_event_date=(
                DATE_COL,
                "max",
            ),
        )
        .reset_index()
    )

    unique_pairs = (
        post_endpoint[
            [
                "month",
                INVESTOR_COL,
                STARTUP_COL,
            ]
        ]
        .drop_duplicates()
        .groupby(
            "month",
            observed=True,
        )
        .size()
        .rename(
            "unique_pairs"
        )
        .reset_index()
    )

    monthly = monthly_base.merge(
        unique_pairs,
        on="month",
        how="left",
        validate="one_to_one",
    )

    stratum_month = (
        post_endpoint.groupby(
            [
                "month",
                "analysis_stratum",
            ],
            observed=True,
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            columns=stratum_order,
            fill_value=0,
        )
        .reset_index()
    )

    monthly = monthly.merge(
        stratum_month,
        on="month",
        how="left",
        validate="one_to_one",
    )

    monthly[
        "calendar_days"
    ] = monthly[
        "month"
    ].map(
        lambda p: p.days_in_month
    )

    monthly[
        "snapshot_calendar_complete"
    ] = monthly[
        "month"
    ].map(
        lambda p: (
            p.end_time.normalize()
            <= canonical_max.normalize()
        )
    )

    monthly[
        "month"
    ] = monthly[
        "month"
    ].astype(str)

    print(
        monthly.to_string(
            index=False
        )
    )

    # =========================================================================
    # 6. Candidate temporal windows — DESCRIPTIVE ONLY
    # =========================================================================

    separator("-")
    print(
        "8.0.1.6 — DESCRIPTIVE "
        "CANDIDATE-WINDOW COUNTS"
    )
    separator("-")

    window_rows = []

    window_rows.append(
        summarize_window(
            post_endpoint,
            "APRIL_2026_ONLY",
            pd.Timestamp(
                "2026-04-01"
            ),
            pd.Timestamp(
                "2026-04-30"
            ),
        )
    )

    window_rows.append(
        summarize_window(
            post_endpoint,
            "APRIL_MAY_2026_FULL_MONTHS",
            pd.Timestamp(
                "2026-04-01"
            ),
            pd.Timestamp(
                "2026-05-31"
            ),
        )
    )

    window_rows.append(
        summarize_window(
            post_endpoint,
            "FULL_RAW_POST_ENDPOINT_TAIL",
            pd.Timestamp(
                "2026-04-01"
            ),
            canonical_max,
        )
    )

    windows = pd.DataFrame(
        window_rows
    )

    print(
        windows.to_string(
            index=False
        )
    )

    # =========================================================================
    # 7. Persist outcome-blind audit artifacts
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    monthly.to_csv(
        MONTHLY_OUTPUT,
        index=False,
    )

    stratum_counts.to_csv(
        STRATUM_OUTPUT,
        index=False,
    )

    windows.to_csv(
        WINDOW_OUTPUT,
        index=False,
    )

    summary = {
        "schema_version":
            "PHASE_8_0_1_POST_ENDPOINT_AVAILABILITY_V1",

        "scientific_role":
            "OUTCOME_BLIND_EVALUATION_DESIGN_AUDIT",

        "model_loaded": False,
        "checkpoint_loaded": False,
        "inference_executed": False,
        "validation_scored": False,
        "test_scored": False,
        "negatives_generated": False,
        "model_selection_performed": False,
        "holdout_selected": False,

        "frozen_reproduction_endpoint":
            str(
                FROZEN_EXPERIMENT_END.date()
            ),

        "canonical_min_date":
            str(
                canonical_min.date()
            ),

        "canonical_max_date":
            str(
                canonical_max.date()
            ),

        "canonical_rows":
            int(len(df)),

        "historical_rows_through_endpoint":
            int(len(historical)),

        "post_endpoint_rows":
            int(len(post_endpoint)),

        "post_endpoint_min_date":
            str(
                post_endpoint[
                    DATE_COL
                ].min().date()
            ),

        "post_endpoint_max_date":
            str(
                post_endpoint[
                    DATE_COL
                ].max().date()
            ),

        "phase2_post_endpoint_artifact_present":
            phase2_artifact_present,

        "phase2_post_endpoint_id_match":
            phase2_id_match,

        "decision_status":
            "NO_CONFIRMATORY_HOLDOUT_SELECTED",

        "interpretation_boundary": (
            "Counts and temporal availability only. "
            "No model outcomes or ranking metrics "
            "were inspected. Calendar completeness "
            "does not establish Crunchbase reporting "
            "maturity."
        ),
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    manifest = {
        "phase":
            "8.0.1",

        "script":
            "scripts/"
            "phase_8_0_1_post_endpoint_"
            "temporal_availability_audit_V1.py",

        "authoritative_input":
            str(CANONICAL_PATH),

        "optional_frozen_phase2_input":
            str(
                FROZEN_PHASE2_POST_ENDPOINT_PATH
            ),

        "frozen_experiment_endpoint":
            str(
                FROZEN_EXPERIMENT_END.date()
            ),

        "outputs": [
            str(MONTHLY_OUTPUT),
            str(STRATUM_OUTPUT),
            str(WINDOW_OUTPUT),
            str(SUMMARY_OUTPUT),
        ],

        "prohibited_operations": [
            "model_loading",
            "checkpoint_loading",
            "inference",
            "validation_scoring",
            "test_scoring",
            "negative_generation",
            "model_selection",
            "holdout_selection",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    derived_hashes = {
        str(MONTHLY_OUTPUT):
            sha256_file(
                MONTHLY_OUTPUT
            ),

        str(STRATUM_OUTPUT):
            sha256_file(
                STRATUM_OUTPUT
            ),

        str(WINDOW_OUTPUT):
            sha256_file(
                WINDOW_OUTPUT
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
    # 8. Final audit summary
    # =========================================================================

    separator()
    print(
        "PHASE 8.0.1 AUDIT COMPLETE"
    )
    separator()

    print()
    print(
        f"Frozen reproduction endpoint: "
        f"{FROZEN_EXPERIMENT_END.date()}"
    )

    print(
        f"Post-endpoint events:          "
        f"{len(post_endpoint):,}"
    )

    print(
        f"Available future range:        "
        f"{post_endpoint[DATE_COL].min().date()} "
        f"-> "
        f"{post_endpoint[DATE_COL].max().date()}"
    )

    primary_count = int(
        (
            post_endpoint[
                "analysis_stratum"
            ]
            == "new_warm_investor_cold_startup"
        ).sum()
    )

    print(
        f"Primary-stratum events:        "
        f"{primary_count:,}"
    )

    print()
    print(
        "CONFIRMATORY HOLDOUT SELECTED: NO"
    )

    print()
    print(
        "Next scientific decision must be based "
        "on the observed post-endpoint temporal "
        "coverage and thesis-stratum population."
    )

    print()
    print(
        f"Outputs written to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()