from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# P1 — NEW-TO-INVESTOR + COLD-START DIAGNOSTICS
# =============================================================================
#
# Scientific role:
#   Post-hoc diagnostic analysis of the already-frozen Phase-6 final test.
#
# IMPORTANT:
#   - NO training
#   - NO inference
#   - NO checkpoint selection
#   - NO use of T60 events as history
#   - history = T0-T59 only
#
# =============================================================================


ROOT = Path(__file__).resolve().parents[2]

PROPOSAL_ROOT = (
    ROOT
    / "data"
    / "experimental"
    / "proposal_evidence"
)

TEMPORAL_SPLIT = (
    ROOT
    / "data"
    / "experimental"
    / "phase_2"
    / "model_ready"
    / "interactions_itrs_temporal_split.parquet"
)

T60_SPLIT = (
    ROOT
    / "data"
    / "experimental"
    / "phase_2"
    / "model_ready"
    / "t60_validation_test_split.parquet"
)

FINAL_PREDICTIONS = (
    PROPOSAL_ROOT
    / "05_final_test_event_predictions.csv"
)


# =============================================================================
# Outputs
# =============================================================================

OUT_EVENT_DIAGNOSTICS = (
    PROPOSAL_ROOT
    / "06_test_case_history_diagnostics.csv"
)

OUT_SUBGROUP_PERFORMANCE = (
    PROPOSAL_ROOT
    / "07_subgroup_performance.csv"
)

OUT_HISTORY_PERFORMANCE = (
    PROPOSAL_ROOT
    / "08_startup_history_performance.csv"
)

OUT_INVESTOR_HISTORY_PERFORMANCE = (
    PROPOSAL_ROOT
    / "09_investor_history_performance.csv"
)

OUT_BINARY_COMPARISONS = (
    PROPOSAL_ROOT
    / "10_binary_diagnostic_performance.csv"
)

OUT_SUMMARY = (
    PROPOSAL_ROOT
    / "p1_diagnostic_summary.json"
)


# =============================================================================
# Helpers
# =============================================================================


def banner(text: str) -> None:
    print()
    print("=" * 110)
    print(text)
    print("=" * 110)


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def metric_row(
    df: pd.DataFrame,
    group_name: str,
    total_n: int,
) -> dict:
    n = len(df)

    if n == 0:
        return {
            "group": group_name,
            "n": 0,
            "share_of_test": 0.0,
            "HR@10": np.nan,
            "NDCG@10": np.nan,
            "mean_positive_rank": np.nan,
            "median_positive_rank": np.nan,
        }

    return {
        "group": group_name,
        "n": int(n),
        "share_of_test": float(n / total_n),
        "HR@10": float(df["HR@10"].mean()),
        "NDCG@10": float(df["NDCG@10"].mean()),
        "mean_positive_rank": float(
            df["positive_rank"].mean()
        ),
        "median_positive_rank": float(
            df["positive_rank"].median()
        ),
    }


def history_bin(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 9:
        return "5-9"
    return "10+"


def classify_primary_group(row: pd.Series) -> str:
    """
    Mutually exclusive proposal diagnostic groups.

    History is defined strictly from T0-T59.
    """

    pair_seen = bool(row["pair_seen_before_t60"])
    investor_seen = bool(
        row["investor_seen_before_t60"]
    )
    startup_seen = bool(
        row["startup_seen_before_t60"]
    )

    if pair_seen:
        return "repeat_pair"

    # Everything below is a genuinely novel investor-startup pair.

    if investor_seen and startup_seen:
        return "novel_warm_warm"

    if investor_seen and not startup_seen:
        return "novel_cold_startup"

    if not investor_seen and startup_seen:
        return "novel_cold_investor"

    return "novel_both_cold"


# =============================================================================
# Main
# =============================================================================


def main() -> None:

    banner(
        "P1 — NEW-TO-INVESTOR AND COLD-START DIAGNOSTICS"
    )

    # =========================================================================
    # 1. Load frozen Phase-6 test predictions
    # =========================================================================

    predictions = pd.read_csv(
        FINAL_PREDICTIONS
    )

    require(
        len(predictions) == 20_264,
        (
            "Expected 20,264 final Phase-6 test predictions; "
            f"found {len(predictions):,}."
        ),
    )

    require(
        predictions["interaction_id"].is_unique,
        "Final prediction interaction_id is not unique.",
    )

    required_metric_cols = {
        "interaction_id",
        "positive_rank",
        "HR@10",
        "NDCG@10",
        "split",
    }

    require(
        required_metric_cols.issubset(
            predictions.columns
        ),
        (
            "Final predictions missing required columns: "
            f"{sorted(required_metric_cols - set(predictions.columns))}"
        ),
    )

    require(
        predictions["split"]
        .eq("test")
        .all(),
        "Combined prediction table contains non-test rows.",
    )

    print(
        f"Final predictions:       "
        f"{len(predictions):,}"
    )

    # =========================================================================
    # 2. Load frozen T60 diagnostic labels
    # =========================================================================

    t60 = pd.read_parquet(
        T60_SPLIT
    )

    test_t60 = (
        t60.loc[
            t60["evaluation_split"].eq("test")
        ]
        .copy()
    )

    require(
        len(test_t60) == 20_264,
        (
            "Expected 20,264 Phase-2 test rows; "
            f"found {len(test_t60):,}."
        ),
    )

    require(
        test_t60["interaction_id"].is_unique,
        "T60 test interaction_id is not unique.",
    )

    diagnostic_columns = [
        "interaction_id",
        "funding_round_id",
        "investor_id",
        "investor_name",
        "startup_id",
        "startup_name",
        "announced_on",
        "investment_type",
        "investor_seen_before_t60",
        "investor_seen_in_t1_t59",
        "investor_seen_in_t0",
        "investor_t0_only_history",
        "startup_seen_before_t60",
        "startup_seen_in_t1_t59",
        "startup_seen_in_t0",
        "startup_t0_only_history",
        "pair_seen_before_t60",
        "pair_seen_in_t1_t59",
        "new_to_investor_pair",
        "interaction_cold_start_status",
        "t60_pair_event_count",
        "pair_repeats_within_t60",
        "evaluation_split",
    ]

    test_t60 = test_t60[
        diagnostic_columns
    ].copy()

    # =========================================================================
    # 3. Build strict T0-T59 historical interaction counts
    # =========================================================================

    banner(
        "BUILD STRICT PRE-T60 HISTORY COUNTS"
    )

    temporal = pd.read_parquet(
        TEMPORAL_SPLIT,
        columns=[
            "interaction_id",
            "investor_id",
            "startup_id",
            "segment_number",
            "experiment_split",
        ],
    )

    history = temporal.loc[
        temporal["segment_number"] < 60
    ].copy()

    require(
        not history.empty,
        "T0-T59 history is empty.",
    )

    require(
        history["segment_number"].max() <= 59,
        "T60 leakage detected in history.",
    )

    print(
        f"T0-T59 history events:    "
        f"{len(history):,}"
    )

    investor_counts = (
        history
        .groupby(
            "investor_id",
            observed=True,
        )
        .size()
        .rename(
            "investor_history_count"
        )
    )

    startup_counts = (
        history
        .groupby(
            "startup_id",
            observed=True,
        )
        .size()
        .rename(
            "startup_history_count"
        )
    )

    pair_counts = (
        history
        .groupby(
            [
                "investor_id",
                "startup_id",
            ],
            observed=True,
        )
        .size()
        .rename(
            "pair_history_count"
        )
        .reset_index()
    )

    # =========================================================================
    # 4. Join frozen predictions + labels
    # =========================================================================

    events = test_t60.merge(
        predictions[
            [
                "interaction_id",
                "investor_global_binding",
                "positive_startup_local_binding",
                "positive_rank",
                "HR@10",
                "NDCG@10",
            ]
        ],
        on="interaction_id",
        how="inner",
        validate="one_to_one",
    )

    require(
        len(events) == 20_264,
        "Prediction/T60 join lost or duplicated rows.",
    )

    # Add pre-T60 counts.

    events = events.merge(
        investor_counts,
        left_on="investor_id",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    events = events.merge(
        startup_counts,
        left_on="startup_id",
        right_index=True,
        how="left",
        validate="many_to_one",
    )

    events = events.merge(
        pair_counts,
        on=[
            "investor_id",
            "startup_id",
        ],
        how="left",
        validate="many_to_one",
    )

    count_cols = [
        "investor_history_count",
        "startup_history_count",
        "pair_history_count",
    ]

    for col in count_cols:
        events[col] = (
            events[col]
            .fillna(0)
            .astype("int64")
        )

    # =========================================================================
    # 5. Cross-check frozen flags against independently counted history
    # =========================================================================

    investor_flag_from_count = (
        events["investor_history_count"] > 0
    )

    startup_flag_from_count = (
        events["startup_history_count"] > 0
    )

    pair_flag_from_count = (
        events["pair_history_count"] > 0
    )

    require(
        (
            investor_flag_from_count
            == events["investor_seen_before_t60"]
        ).all(),
        (
            "Investor pre-T60 count disagrees with "
            "frozen Phase-2 flag."
        ),
    )

    require(
        (
            startup_flag_from_count
            == events["startup_seen_before_t60"]
        ).all(),
        (
            "Startup pre-T60 count disagrees with "
            "frozen Phase-2 flag."
        ),
    )

    require(
        (
            pair_flag_from_count
            == events["pair_seen_before_t60"]
        ).all(),
        (
            "Pair pre-T60 count disagrees with "
            "frozen Phase-2 flag."
        ),
    )

    require(
        (
            events["new_to_investor_pair"]
            == ~events["pair_seen_before_t60"]
        ).all(),
        (
            "new_to_investor_pair disagrees with "
            "pair_seen_before_t60."
        ),
    )

    print(
        "Frozen Phase-2 history flags vs "
        "independent T0-T59 counts: PASS"
    )

    # =========================================================================
    # 6. Primary mutually exclusive diagnostic groups
    # =========================================================================

    events["proposal_diagnostic_group"] = (
        events.apply(
            classify_primary_group,
            axis=1,
        )
    )

    group_order = [
        "repeat_pair",
        "novel_warm_warm",
        "novel_cold_startup",
        "novel_cold_investor",
        "novel_both_cold",
    ]

    observed_groups = set(
        events[
            "proposal_diagnostic_group"
        ].unique()
    )

    unexpected_groups = (
        observed_groups
        - set(group_order)
    )

    require(
        not unexpected_groups,
        (
            "Unexpected diagnostic group(s): "
            f"{sorted(unexpected_groups)}"
        ),
    )

    # Useful derived dimensions.

    events["investor_cold"] = (
        events["investor_history_count"] == 0
    )

    events["startup_cold"] = (
        events["startup_history_count"] == 0
    )

    events["pair_repeat"] = (
        events["pair_history_count"] > 0
    )

    events["new_to_investor"] = (
        events["pair_history_count"] == 0
    )

    events["startup_history_bin"] = (
        events["startup_history_count"]
        .map(history_bin)
    )

    events["investor_history_bin"] = (
        events["investor_history_count"]
        .map(history_bin)
    )

    # =========================================================================
    # 7. Save event-level diagnostic dataset
    # =========================================================================

    events = events.sort_values(
        "interaction_id"
    ).reset_index(drop=True)

    events.to_csv(
        OUT_EVENT_DIAGNOSTICS,
        index=False,
    )

    # =========================================================================
    # 8. Main subgroup performance
    # =========================================================================

    total_n = len(events)

    subgroup_rows = []

    subgroup_rows.append(
        metric_row(
            events,
            "all_test",
            total_n,
        )
    )

    subgroup_rows.append(
        metric_row(
            events.loc[
                events["new_to_investor"]
            ],
            "all_new_to_investor",
            total_n,
        )
    )

    subgroup_rows.append(
        metric_row(
            events.loc[
                events["pair_repeat"]
            ],
            "all_repeat_pairs",
            total_n,
        )
    )

    for group in group_order:
        subgroup_rows.append(
            metric_row(
                events.loc[
                    events[
                        "proposal_diagnostic_group"
                    ].eq(group)
                ],
                group,
                total_n,
            )
        )

    subgroup = pd.DataFrame(
        subgroup_rows
    )

    subgroup.to_csv(
        OUT_SUBGROUP_PERFORMANCE,
        index=False,
    )

    # =========================================================================
    # 9. Startup-history performance
    # =========================================================================

    history_order = [
        "0",
        "1",
        "2-4",
        "5-9",
        "10+",
    ]

    startup_history_rows = []

    for history_group in history_order:
        frame = events.loc[
            events[
                "startup_history_bin"
            ].eq(history_group)
        ]

        row = metric_row(
            frame,
            history_group,
            total_n,
        )

        if len(frame):
            row["mean_history_count"] = float(
                frame[
                    "startup_history_count"
                ].mean()
            )
        else:
            row["mean_history_count"] = np.nan

        startup_history_rows.append(row)

    startup_history_df = pd.DataFrame(
        startup_history_rows
    )

    startup_history_df.rename(
        columns={
            "group": "startup_history_bin"
        },
        inplace=True,
    )

    startup_history_df.to_csv(
        OUT_HISTORY_PERFORMANCE,
        index=False,
    )

    # =========================================================================
    # 10. Investor-history performance
    # =========================================================================

    investor_history_rows = []

    for history_group in history_order:
        frame = events.loc[
            events[
                "investor_history_bin"
            ].eq(history_group)
        ]

        row = metric_row(
            frame,
            history_group,
            total_n,
        )

        if len(frame):
            row["mean_history_count"] = float(
                frame[
                    "investor_history_count"
                ].mean()
            )
        else:
            row["mean_history_count"] = np.nan

        investor_history_rows.append(row)

    investor_history_df = pd.DataFrame(
        investor_history_rows
    )

    investor_history_df.rename(
        columns={
            "group": "investor_history_bin"
        },
        inplace=True,
    )

    investor_history_df.to_csv(
        OUT_INVESTOR_HISTORY_PERFORMANCE,
        index=False,
    )

    # =========================================================================
    # 11. Binary diagnostic comparisons
    # =========================================================================

    binary_rows = []

    binary_definitions = [
        (
            "pair_status",
            "repeat",
            events["pair_repeat"],
        ),
        (
            "pair_status",
            "new_to_investor",
            events["new_to_investor"],
        ),
        (
            "startup_status",
            "warm_startup",
            ~events["startup_cold"],
        ),
        (
            "startup_status",
            "cold_startup",
            events["startup_cold"],
        ),
        (
            "investor_status",
            "warm_investor",
            ~events["investor_cold"],
        ),
        (
            "investor_status",
            "cold_investor",
            events["investor_cold"],
        ),
    ]

    for dimension, label, mask in binary_definitions:
        row = metric_row(
            events.loc[mask],
            label,
            total_n,
        )

        row["dimension"] = dimension

        binary_rows.append(row)

    binary_df = pd.DataFrame(
        binary_rows
    )

    binary_df = binary_df[
        [
            "dimension",
            "group",
            "n",
            "share_of_test",
            "HR@10",
            "NDCG@10",
            "mean_positive_rank",
            "median_positive_rank",
        ]
    ]

    binary_df.to_csv(
        OUT_BINARY_COMPARISONS,
        index=False,
    )

    # =========================================================================
    # 12. Internal reproduction metric check
    # =========================================================================

    aggregate_hr = float(
        events["HR@10"].mean()
    )

    aggregate_ndcg = float(
        events["NDCG@10"].mean()
    )

    expected_hr = 0.358813659692
    expected_ndcg = 0.194895356504

    require(
        abs(
            aggregate_hr - expected_hr
        ) < 1e-12,
        (
            "Aggregate HR@10 does not reproduce "
            "frozen Phase-6 final result."
        ),
    )

    require(
        abs(
            aggregate_ndcg - expected_ndcg
        ) < 1e-12,
        (
            "Aggregate NDCG@10 does not reproduce "
            "frozen Phase-6 final result."
        ),
    )

    # =========================================================================
    # 13. Summary JSON
    # =========================================================================

    primary_counts = (
        events[
            "proposal_diagnostic_group"
        ]
        .value_counts()
        .reindex(
            group_order,
            fill_value=0,
        )
        .to_dict()
    )

    summary = {
        "schema_version": (
            "PROPOSAL_EVIDENCE_P1_V1"
        ),
        "status": "P1_COMPLETE",
        "history_definition": (
            "Strictly T0-T59 only; no within-T60 "
            "events treated as prior history."
        ),
        "test_cases": int(total_n),
        "aggregate_metrics": {
            "HR@10": aggregate_hr,
            "NDCG@10": aggregate_ndcg,
            "mean_positive_rank": float(
                events["positive_rank"].mean()
            ),
            "median_positive_rank": float(
                events["positive_rank"].median()
            ),
        },
        "primary_group_counts": {
            str(k): int(v)
            for k, v in primary_counts.items()
        },
        "new_to_investor_cases": int(
            events["new_to_investor"].sum()
        ),
        "repeat_pair_cases": int(
            events["pair_repeat"].sum()
        ),
        "cold_startup_cases": int(
            events["startup_cold"].sum()
        ),
        "cold_investor_cases": int(
            events["investor_cold"].sum()
        ),
        "both_cold_cases": int(
            (
                events["startup_cold"]
                & events["investor_cold"]
            ).sum()
        ),
        "phase2_flag_crosscheck": "PASS",
        "phase6_metric_crosscheck": "PASS",
        "outputs": {
            "event_diagnostics": str(
                OUT_EVENT_DIAGNOSTICS.relative_to(
                    ROOT
                )
            ),
            "subgroup_performance": str(
                OUT_SUBGROUP_PERFORMANCE.relative_to(
                    ROOT
                )
            ),
            "startup_history_performance": str(
                OUT_HISTORY_PERFORMANCE.relative_to(
                    ROOT
                )
            ),
            "investor_history_performance": str(
                OUT_INVESTOR_HISTORY_PERFORMANCE.relative_to(
                    ROOT
                )
            ),
            "binary_comparisons": str(
                OUT_BINARY_COMPARISONS.relative_to(
                    ROOT
                )
            ),
        },
    }

    with OUT_SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    # =========================================================================
    # 14. Console result
    # =========================================================================

    banner(
        "PRIMARY THESIS DIAGNOSTIC — ITRS PERFORMANCE BY GROUP"
    )

    display = subgroup.copy()

    display["share_of_test"] = (
        display["share_of_test"] * 100
    )

    print(
        display.to_string(
            index=False,
            formatters={
                "share_of_test": (
                    lambda x: f"{x:.2f}%"
                ),
                "HR@10": (
                    lambda x: f"{x:.6f}"
                ),
                "NDCG@10": (
                    lambda x: f"{x:.6f}"
                ),
                "mean_positive_rank": (
                    lambda x: f"{x:.2f}"
                ),
                "median_positive_rank": (
                    lambda x: f"{x:.1f}"
                ),
            },
        )
    )

    banner(
        "PERFORMANCE BY STARTUP PRE-T60 HISTORY"
    )

    display_history = (
        startup_history_df.copy()
    )

    display_history["share_of_test"] = (
        display_history["share_of_test"]
        * 100
    )

    print(
        display_history.to_string(
            index=False,
            formatters={
                "share_of_test": (
                    lambda x: f"{x:.2f}%"
                ),
                "HR@10": (
                    lambda x: f"{x:.6f}"
                ),
                "NDCG@10": (
                    lambda x: f"{x:.6f}"
                ),
                "mean_positive_rank": (
                    lambda x: f"{x:.2f}"
                ),
                "median_positive_rank": (
                    lambda x: f"{x:.1f}"
                ),
                "mean_history_count": (
                    lambda x: f"{x:.2f}"
                ),
            },
        )
    )

    banner(
        "P1 OUTPUTS"
    )

    for path in [
        OUT_EVENT_DIAGNOSTICS,
        OUT_SUBGROUP_PERFORMANCE,
        OUT_HISTORY_PERFORMANCE,
        OUT_INVESTOR_HISTORY_PERFORMANCE,
        OUT_BINARY_COMPARISONS,
        OUT_SUMMARY,
    ]:
        print(
            "WROTE ",
            path.relative_to(ROOT),
        )

    banner(
        "P1 COMPLETE — POST-HOC DIAGNOSTICS ONLY / "
        "NO TRAINING OR MODEL SELECTION"
    )


if __name__ == "__main__":
    main()
