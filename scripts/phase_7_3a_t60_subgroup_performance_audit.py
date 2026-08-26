#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# FROZEN PORTABLE INPUTS
# =============================================================================

INPUT_DIR = Path("phase_7_inputs")

METRICS_PATH = INPUT_DIR / "final_t60_test_case_metrics.parquet"
SPLIT_PATH = INPUT_DIR / "t60_validation_test_split.parquet"
RESULT_PATH = INPUT_DIR / "final_t60_test_result.json"
MANIFEST_PATH = INPUT_DIR / "phase_7_handoff_manifest.json"

OUTPUT_DIR = Path("phase_7_outputs/phase_7_3a")

EXPECTED_TEST_CASES = 20_264

EXPECTED_HASHES = {
    METRICS_PATH:
        "2c033f33f62ead31146cdebdb5058f78f438f1b096d81eaa6b80c67cb2eae2a8",

    SPLIT_PATH:
        "8343a37ab552621ec42030784d55e92e6c6dfd7b2195bd8ddef39e028e736f4a",

    RESULT_PATH:
        "3edf2ae281c0bca96399b4e98ec73e48f1324be5c8dd8eed46d8b8f9cb1b0303",

    MANIFEST_PATH:
        "e922dd5c9816486aea32642a795be341b75bab858612e5bf95352dfea23cd904",
}

EXPECTED_HR10 = 0.358813659692
EXPECTED_NDCG10 = 0.194895356504


# =============================================================================
# HELPERS
# =============================================================================

def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)

    return h.hexdigest()


def banner(text: str) -> None:
    print()
    print("=" * 118)
    print(text)
    print("=" * 118)


def summarize(
    frame: pd.DataFrame,
    group_columns: str | list[str],
) -> pd.DataFrame:

    if isinstance(group_columns, str):
        group_columns = [group_columns]

    rows = []

    grouped = frame.groupby(
        group_columns,
        dropna=False,
        sort=True,
        observed=False,
    )

    for group_values, subset in grouped:

        if not isinstance(group_values, tuple):
            group_values = (group_values,)

        row = {
            column: value
            for column, value in zip(
                group_columns,
                group_values,
            )
        }

        row.update(
            {
                "events":
                    int(len(subset)),

                "event_share_pct":
                    float(
                        100.0
                        * len(subset)
                        / len(frame)
                    ),

                "hits_at_10":
                    int(
                        subset["HR@10"].sum()
                    ),

                "HR@10":
                    float(
                        subset["HR@10"].mean()
                    ),

                "NDCG@10":
                    float(
                        subset["NDCG@10"].mean()
                    ),

                "mean_positive_rank":
                    float(
                        subset["positive_rank"].mean()
                    ),

                "median_positive_rank":
                    float(
                        subset["positive_rank"].median()
                    ),
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def print_table(
    title: str,
    table: pd.DataFrame,
) -> None:

    banner(title)

    print(
        table.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    banner(
        "PHASE 7.3a — FINAL T60 SUBGROUP PERFORMANCE AUDIT"
    )

    # -------------------------------------------------------------------------
    # 1. Frozen input integrity
    # -------------------------------------------------------------------------

    for path, expected_hash in EXPECTED_HASHES.items():

        require(
            path.exists(),
            f"Missing frozen input: {path}",
        )

        actual_hash = sha256_file(path)

        require(
            actual_hash == expected_hash,
            (
                f"Frozen input hash drift:\n"
                f"  path:     {path}\n"
                f"  expected: {expected_hash}\n"
                f"  actual:   {actual_hash}"
            ),
        )

        print(f"PASS HASH  {path}")

    # -------------------------------------------------------------------------
    # 2. Load read-only portable Phase-7 inputs
    # -------------------------------------------------------------------------

    metrics = pd.read_parquet(
        METRICS_PATH
    )

    split = pd.read_parquet(
        SPLIT_PATH
    )

    final_result = json.loads(
        RESULT_PATH.read_text(
            encoding="utf-8"
        )
    )

    handoff_manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    # -------------------------------------------------------------------------
    # 3. Portable handoff policy integrity
    # -------------------------------------------------------------------------

    require(
        handoff_manifest[
            "phase_6_status"
        ]
        == "COMPLETE_AND_FROZEN",
        "Phase 6 is not marked complete/frozen.",
    )

    require(
        handoff_manifest[
            "final_t60_result"
        ][
            "test_rescoring_allowed"
        ]
        is False,
        "Portable handoff unexpectedly allows T60 rescoring.",
    )

    require(
        int(
            handoff_manifest[
                "final_t60_result"
            ][
                "test_cases"
            ]
        )
        == EXPECTED_TEST_CASES,
        "Manifest test-case count drift.",
    )

    # -------------------------------------------------------------------------
    # 4. Final metric artifact integrity
    # -------------------------------------------------------------------------

    expected_metric_columns = [
        "test_case_position",
        "matrix_row_index",
        "interaction_id",
        "investor_global",
        "positive_startup_local",
        "positive_rank",
        "HR@10",
        "NDCG@10",
        "chunk_index",
    ]

    require(
        list(metrics.columns)
        == expected_metric_columns,
        "Final metric schema drift.",
    )

    require(
        len(metrics)
        == EXPECTED_TEST_CASES,
        "Final metric row-count drift.",
    )

    require(
        metrics[
            "interaction_id"
        ].is_unique,
        "Final metric interaction_id is not unique.",
    )

    require(
        metrics[
            "test_case_position"
        ].is_unique,
        "test_case_position is not unique.",
    )

    require(
        bool(
            metrics[
                "positive_rank"
            ]
            .between(
                1,
                100,
            )
            .all()
        ),
        "Positive rank outside frozen range 1..100.",
    )

    require(
        bool(
            metrics[
                "HR@10"
            ]
            .isin(
                [0.0, 1.0]
            )
            .all()
        ),
        "HR@10 contains non-binary values.",
    )

    require(
        bool(
            np.isfinite(
                metrics[
                    [
                        "positive_rank",
                        "HR@10",
                        "NDCG@10",
                    ]
                ]
                .to_numpy(
                    dtype=np.float64
                )
            ).all()
        ),
        "Non-finite final test metrics detected.",
    )

    # -------------------------------------------------------------------------
    # 5. Recover frozen T60 TEST metadata
    # -------------------------------------------------------------------------

    require(
        "evaluation_split"
        in split.columns,
        "evaluation_split missing.",
    )

    test_metadata = (
        split.loc[
            split[
                "evaluation_split"
            ]
            == "test"
        ]
        .copy()
    )

    require(
        len(test_metadata)
        == EXPECTED_TEST_CASES,
        "T60 test metadata row-count drift.",
    )

    require(
        test_metadata[
            "interaction_id"
        ].is_unique,
        "T60 test interaction_id is not unique.",
    )

    metadata_columns = [
        "interaction_id",

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
    ]

    for column in metadata_columns:

        require(
            column
            in test_metadata.columns,
            f"Required subgroup column missing: {column}",
        )

    boolean_columns = [
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

        "pair_repeats_within_t60",
    ]

    require(
        not test_metadata[
            boolean_columns
        ].isna().any().any(),
        "Null subgroup boolean detected.",
    )

    # -------------------------------------------------------------------------
    # 6. Exact one-to-one binding
    # -------------------------------------------------------------------------

    merged = metrics.merge(
        test_metadata[
            metadata_columns
        ],
        on="interaction_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    require(
        len(merged)
        == EXPECTED_TEST_CASES,
        "Merged test-case count drift.",
    )

    require(
        bool(
            (
                merged[
                    "_merge"
                ]
                == "both"
            ).all()
        ),
        "At least one final test prediction failed metadata binding.",
    )

    merged = merged.drop(
        columns="_merge"
    )

    # -------------------------------------------------------------------------
    # 7. Derived interpretable subgroup labels
    # -------------------------------------------------------------------------

    merged[
        "pair_novelty_group"
    ] = np.where(
        merged[
            "new_to_investor_pair"
        ].astype(bool),
        "new_to_investor",
        "previous_investor_startup_pair",
    )

    # More detailed pair-history classification.
    merged[
        "pair_history_group"
    ] = np.select(
        [
            merged[
                "new_to_investor_pair"
            ].astype(bool),

            merged[
                "pair_seen_in_t1_t59"
            ].astype(bool),

            (
                merged[
                    "pair_seen_before_t60"
                ].astype(bool)
                &
                ~merged[
                    "pair_seen_in_t1_t59"
                ].astype(bool)
            ),
        ],
        [
            "new_to_investor",
            "previous_pair_seen_t1_t59",
            "previous_pair_t0_only",
        ],
        default="inconsistent_or_other",
    )

    merged[
        "investor_history_group"
    ] = np.select(
        [
            ~merged[
                "investor_seen_before_t60"
            ].astype(bool),

            merged[
                "investor_t0_only_history"
            ].astype(bool),

            merged[
                "investor_seen_in_t1_t59"
            ].astype(bool),
        ],
        [
            "cold_investor_unseen_before_t60",
            "investor_t0_only_history",
            "investor_active_t1_t59",
        ],
        default="inconsistent_or_other",
    )

    merged[
        "startup_history_group"
    ] = np.select(
        [
            ~merged[
                "startup_seen_before_t60"
            ].astype(bool),

            merged[
                "startup_t0_only_history"
            ].astype(bool),

            merged[
                "startup_seen_in_t1_t59"
            ].astype(bool),
        ],
        [
            "cold_startup_unseen_before_t60",
            "startup_t0_only_history",
            "startup_active_t1_t59",
        ],
        default="inconsistent_or_other",
    )

    merged[
        "t60_pair_repeat_group"
    ] = np.where(
        merged[
            "pair_repeats_within_t60"
        ].astype(bool),
        "pair_repeated_within_t60",
        "single_pair_event_in_t60",
    )

    # -------------------------------------------------------------------------
    # 8. Whole-test fingerprint
    # -------------------------------------------------------------------------

    overall_hr = float(
        merged[
            "HR@10"
        ].mean()
    )

    overall_ndcg = float(
        merged[
            "NDCG@10"
        ].mean()
    )

    overall = pd.DataFrame(
        [
            {
                "events":
                    int(
                        len(merged)
                    ),

                "hits_at_10":
                    int(
                        merged[
                            "HR@10"
                        ].sum()
                    ),

                "HR@10":
                    overall_hr,

                "NDCG@10":
                    overall_ndcg,

                "mean_positive_rank":
                    float(
                        merged[
                            "positive_rank"
                        ].mean()
                    ),

                "median_positive_rank":
                    float(
                        merged[
                            "positive_rank"
                        ].median()
                    ),
            }
        ]
    )

    require(
        abs(
            overall_hr
            - EXPECTED_HR10
        )
        <= 5e-12,
        (
            "Whole-test HR@10 fingerprint drift: "
            f"{overall_hr:.15f}"
        ),
    )

    require(
        abs(
            overall_ndcg
            - EXPECTED_NDCG10
        )
        <= 5e-12,
        (
            "Whole-test NDCG@10 fingerprint drift: "
            f"{overall_ndcg:.15f}"
        ),
    )

    require(
        int(
            overall.iloc[0][
                "hits_at_10"
            ]
        )
        == 7_271,
        "Final hit-count fingerprint drift.",
    )

    # -------------------------------------------------------------------------
    # 9. Subgroup analyses
    # -------------------------------------------------------------------------

    pair_novelty = summarize(
        merged,
        "pair_novelty_group",
    )

    pair_history = summarize(
        merged,
        "pair_history_group",
    )

    investor_history = summarize(
        merged,
        "investor_history_group",
    )

    startup_history = summarize(
        merged,
        "startup_history_group",
    )

    cold_start = summarize(
        merged,
        "interaction_cold_start_status",
    )

    t60_repeat = summarize(
        merged,
        "t60_pair_repeat_group",
    )

    discovery_by_investor_history = summarize(
        merged,
        [
            "pair_novelty_group",
            "investor_history_group",
        ],
    )

    # -------------------------------------------------------------------------
    # 10. Persist only aggregate Phase-7 results
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    outputs = {
        "overall_test_performance.csv":
            overall,

        "pair_novelty_performance.csv":
            pair_novelty,

        "pair_history_performance.csv":
            pair_history,

        "investor_history_performance.csv":
            investor_history,

        "startup_history_performance.csv":
            startup_history,

        "cold_start_performance.csv":
            cold_start,

        "t60_pair_repeat_performance.csv":
            t60_repeat,

        "discovery_by_investor_history.csv":
            discovery_by_investor_history,
    }

    for filename, table in outputs.items():

        table.to_csv(
            OUTPUT_DIR / filename,
            index=False,
        )

    result_contract = {
        "schema_version":
            "ITRS_PHASE_7_3A_SUBGROUP_AUDIT_V1",

        "status":
            "PASS",

        "analysis_type":
            "POST_HOC_FROZEN_T60_PREDICTION_ANALYSIS",

        "test_cases_analyzed":
            EXPECTED_TEST_CASES,

        "new_model_inference":
            False,

        "t60_rescoring":
            False,

        "training_performed":
            False,

        "model_selection_performed":
            False,

        "overall_test_metrics": {
            "HR@10":
                overall_hr,

            "NDCG@10":
                overall_ndcg,

            "hits_at_10":
                7271,

            "mean_positive_rank":
                float(
                    merged[
                        "positive_rank"
                    ].mean()
                ),

            "median_positive_rank":
                float(
                    merged[
                        "positive_rank"
                    ].median()
                ),
        },

        "input_hashes": {
            str(path):
                expected_hash
            for path, expected_hash
            in EXPECTED_HASHES.items()
        },

        "output_files":
            sorted(
                outputs.keys()
            ),
    }

    contract_path = (
        OUTPUT_DIR
        / "phase_7_3a_subgroup_audit_result.json"
    )

    contract_path.write_text(
        json.dumps(
            result_contract,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # 11. Human-readable report
    # -------------------------------------------------------------------------

    print_table(
        "OVERALL FINAL T60 TEST PERFORMANCE",
        overall,
    )

    print_table(
        "PAIR NOVELTY — PRIMARY NEW-TO-INVESTOR DISCOVERY QUESTION",
        pair_novelty,
    )

    print_table(
        "PAIR HISTORY — DETAILED",
        pair_history,
    )

    print_table(
        "INVESTOR HISTORY",
        investor_history,
    )

    print_table(
        "STARTUP HISTORY",
        startup_history,
    )

    print_table(
        "COMBINED INTERACTION COLD-START STATUS",
        cold_start,
    )

    print_table(
        "PAIR REPETITION WITHIN T60",
        t60_repeat,
    )

    print_table(
        "NEW-TO-INVESTOR DISCOVERY × INVESTOR HISTORY",
        discovery_by_investor_history,
    )

    banner(
        "PHASE 7.3a FINAL STATUS"
    )

    print(
        f"Final test events analyzed:      "
        f"{len(merged):,} / {EXPECTED_TEST_CASES:,}"
    )

    print(
        "Frozen input hashes:             PASS"
    )

    print(
        "Prediction/metadata binding:     EXACT"
    )

    print(
        "Whole-test metric fingerprint:   PASS"
    )

    print(
        "New model inference:             NO"
    )

    print(
        "T60 rescoring:                   NO"
    )

    print(
        "Training performed:              NO"
    )

    print(
        "Model/configuration selection:   NO"
    )

    print()
    print(
        "PHASE 7.3a: PASS / "
        "FINAL T60 SUBGROUP PERFORMANCE AUDITED"
    )


if __name__ == "__main__":
    main()
