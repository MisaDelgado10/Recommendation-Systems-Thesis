#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_DIR = Path("phase_7_inputs")
OUTPUT_DIR = Path("phase_7_outputs/phase_7_4")

INITIAL_HR10 = 0.091514882275
INITIAL_NDCG10 = 0.040193099163

BUDGETS = {
    "1pct": {
        "path": INPUT_DIR / "1pct_epoch_metrics.csv",
        "positive_examples_per_epoch": 10_732,
        "serialized_examples_per_epoch": 53_660,
        "batches_per_epoch": 105,
        "expected_sha256":
            "44d90dab3a29f418e1fdbdb63f1833f71d0aac8d78d1df11529c795eb9dfd448",
    },
    "5pct": {
        "path": INPUT_DIR / "5pct_epoch_metrics.csv",
        "positive_examples_per_epoch": 53_662,
        "serialized_examples_per_epoch": 268_310,
        "batches_per_epoch": 525,
        "expected_sha256":
            "b28eb0307dc2cf880776d4bb91fe4a5fee7bf73b4facaac6fc0405083f54b635",
    },
    "10pct": {
        "path": INPUT_DIR / "10pct_epoch_metrics.csv",
        "positive_examples_per_epoch": 107_324,
        "serialized_examples_per_epoch": 536_620,
        "batches_per_epoch": 1_049,
        "expected_sha256":
            "e2c0c669c893b0d6894b2877c8dbf811f3827aa3ac713dfd512cbdb56ae64063",
    },
}


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def banner(text):
    print()
    print("=" * 118)
    print(text)
    print("=" * 118)


def select_best(df):
    # Frozen Phase-5 comparator:
    # highest NDCG@10 -> highest HR@10 -> earliest epoch.
    ordered = df.sort_values(
        by=[
            "validation_NDCG@10",
            "validation_HR@10",
            "display_epoch",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="mergesort",
    )

    return ordered.iloc[0]


def main():

    banner(
        "PHASE 7.4 — TRAINING-BUDGET SCALING "
        "AND LEARNING-DYNAMICS ANALYSIS"
    )

    summary_rows = []
    all_curves = []

    for budget, config in BUDGETS.items():

        path = config["path"]

        require(
            path.exists(),
            f"Missing input: {path}",
        )

        actual_hash = sha256_file(path)

        require(
            actual_hash
            == config["expected_sha256"],
            f"Hash drift: {path}",
        )

        print(f"PASS HASH  {path}")

        df = pd.read_csv(path)

        require(
            len(df) == 20,
            f"{budget}: expected 20 epochs.",
        )

        require(
            bool(
                (df["test_cases_scored"] == 0).all()
            ),
            f"{budget}: T60 test unexpectedly accessed.",
        )

        best = select_best(df)
        final = df.loc[
            df["display_epoch"] == 20
        ].iloc[0]

        best_epoch = int(
            best["display_epoch"]
        )

        total_training_seconds = float(
            df["training_seconds"].sum()
        )

        time_to_best_seconds = float(
            df.loc[
                df["display_epoch"]
                <= best_epoch,
                "training_seconds",
            ].sum()
        )

        total_validation_seconds = float(
            df["validation_seconds"].sum()
        )

        best_ndcg = float(
            best["validation_NDCG@10"]
        )

        best_hr = float(
            best["validation_HR@10"]
        )

        final_ndcg = float(
            final["validation_NDCG@10"]
        )

        final_hr = float(
            final["validation_HR@10"]
        )

        summary_rows.append(
            {
                "training_budget":
                    budget,

                "positive_examples_per_epoch":
                    config[
                        "positive_examples_per_epoch"
                    ],

                "serialized_examples_per_epoch":
                    config[
                        "serialized_examples_per_epoch"
                    ],

                "batches_per_epoch":
                    config[
                        "batches_per_epoch"
                    ],

                "optimizer_steps_20_epochs":
                    int(
                        df[
                            "global_optimizer_step"
                        ].iloc[-1]
                    ),

                "best_epoch":
                    best_epoch,

                "best_HR@10":
                    best_hr,

                "best_NDCG@10":
                    best_ndcg,

                "epoch20_HR@10":
                    final_hr,

                "epoch20_NDCG@10":
                    final_ndcg,

                "NDCG_best_to_epoch20_change":
                    final_ndcg
                    - best_ndcg,

                "NDCG_best_to_epoch20_relative_pct":
                    100.0
                    * (
                        final_ndcg
                        - best_ndcg
                    )
                    / best_ndcg,

                "HR_gain_over_initial":
                    best_hr
                    - INITIAL_HR10,

                "NDCG_gain_over_initial":
                    best_ndcg
                    - INITIAL_NDCG10,

                "total_training_hours":
                    total_training_seconds
                    / 3600.0,

                "training_hours_to_best":
                    time_to_best_seconds
                    / 3600.0,

                "total_validation_minutes":
                    total_validation_seconds
                    / 60.0,

                "positive_presentations_20_epochs":
                    (
                        config[
                            "positive_examples_per_epoch"
                        ]
                        * 20
                    ),

                "serialized_presentations_20_epochs":
                    (
                        config[
                            "serialized_examples_per_epoch"
                        ]
                        * 20
                    ),
            }
        )

        curve = df.copy()
        curve.insert(
            0,
            "training_budget",
            budget,
        )

        curve[
            "best_so_far_NDCG@10"
        ] = curve[
            "validation_NDCG@10"
        ].cummax()

        curve[
            "best_so_far_HR@10"
        ] = curve[
            "validation_HR@10"
        ].cummax()

        all_curves.append(curve)

    summary = pd.DataFrame(
        summary_rows
    )

    order = {
        "1pct": 1,
        "5pct": 5,
        "10pct": 10,
    }

    summary[
        "_budget_order"
    ] = summary[
        "training_budget"
    ].map(order)

    summary = (
        summary.sort_values(
            "_budget_order"
        )
        .drop(
            columns="_budget_order"
        )
        .reset_index(drop=True)
    )

    # ----------------------------------------------------------------------
    # Scaling gains between adjacent supervision budgets.
    # ----------------------------------------------------------------------

    scaling_rows = []

    for previous_budget, next_budget in [
        ("1pct", "5pct"),
        ("5pct", "10pct"),
    ]:

        previous = (
            summary.loc[
                summary[
                    "training_budget"
                ]
                == previous_budget
            ]
            .iloc[0]
        )

        nxt = (
            summary.loc[
                summary[
                    "training_budget"
                ]
                == next_budget
            ]
            .iloc[0]
        )

        scaling_rows.append(
            {
                "from_budget":
                    previous_budget,

                "to_budget":
                    next_budget,

                "supervision_multiplier":
                    (
                        nxt[
                            "positive_examples_per_epoch"
                        ]
                        / previous[
                            "positive_examples_per_epoch"
                        ]
                    ),

                "HR_absolute_gain":
                    (
                        nxt["best_HR@10"]
                        - previous["best_HR@10"]
                    ),

                "HR_relative_gain_pct":
                    (
                        100.0
                        * (
                            nxt["best_HR@10"]
                            - previous["best_HR@10"]
                        )
                        / previous["best_HR@10"]
                    ),

                "NDCG_absolute_gain":
                    (
                        nxt["best_NDCG@10"]
                        - previous["best_NDCG@10"]
                    ),

                "NDCG_relative_gain_pct":
                    (
                        100.0
                        * (
                            nxt["best_NDCG@10"]
                            - previous["best_NDCG@10"]
                        )
                        / previous["best_NDCG@10"]
                    ),

                "extra_training_hours_20_epochs":
                    (
                        nxt["total_training_hours"]
                        - previous[
                            "total_training_hours"
                        ]
                    ),
            }
        )

    scaling = pd.DataFrame(
        scaling_rows
    )

    curves = pd.concat(
        all_curves,
        ignore_index=True,
    )

    # ----------------------------------------------------------------------
    # Basic empirical conclusions.
    # ----------------------------------------------------------------------

    ndcg_values = summary[
        "best_NDCG@10"
    ].to_numpy()

    hr_values = summary[
        "best_HR@10"
    ].to_numpy()

    require(
        bool(
            np.all(
                np.diff(ndcg_values) > 0
            )
        ),
        "Best NDCG is not monotonic with budget.",
    )

    require(
        bool(
            np.all(
                np.diff(hr_values) > 0
            )
        ),
        "Best HR is not monotonic with budget.",
    )

    # ----------------------------------------------------------------------
    # Persist
    # ----------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_DIR
        / "training_budget_summary.csv",
        index=False,
    )

    scaling.to_csv(
        OUTPUT_DIR
        / "adjacent_budget_scaling_gains.csv",
        index=False,
    )

    curves.to_csv(
        OUTPUT_DIR
        / "all_epoch_learning_curves.csv",
        index=False,
    )

    result = {
        "schema_version":
            "ITRS_PHASE_7_4_TRAINING_BUDGET_ANALYSIS_V1",

        "status":
            "PASS",

        "analysis_type":
            "POST_HOC_VALIDATION_LEARNING_DYNAMICS",

        "budgets":
            [
                "1pct",
                "5pct",
                "10pct",
            ],

        "epochs_per_budget":
            20,

        "best_metric_monotonic_with_budget": {
            "HR@10":
                True,
            "NDCG@10":
                True,
        },

        "new_model_inference":
            False,

        "training_performed":
            False,

        "t60_test_rescoring":
            False,

        "model_selection_changed":
            False,
    }

    (
        OUTPUT_DIR
        / "phase_7_4_training_budget_analysis_result.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # ----------------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------------

    banner(
        "TRAINING-BUDGET SUMMARY"
    )

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    banner(
        "ADJACENT BUDGET SCALING GAINS"
    )

    print(
        scaling.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    banner(
        "LEARNING-DYNAMICS SNAPSHOT"
    )

    for _, row in summary.iterrows():

        print(
            f"{row['training_budget']:>5} | "
            f"best epoch={int(row['best_epoch']):2d} | "
            f"best NDCG={row['best_NDCG@10']:.6f} | "
            f"epoch20 NDCG={row['epoch20_NDCG@10']:.6f} | "
            f"Δ best→20="
            f"{row['NDCG_best_to_epoch20_relative_pct']:+.2f}% | "
            f"train={row['total_training_hours']:.2f} h | "
            f"time-to-best="
            f"{row['training_hours_to_best']:.2f} h"
        )

    banner(
        "PHASE 7.4 FINAL STATUS"
    )

    print(
        "Budgets analyzed:                1%, 5%, 10%"
    )

    print(
        "Best HR monotonic with budget:   YES"
    )

    print(
        "Best NDCG monotonic with budget: YES"
    )

    print(
        "New model inference:             NO"
    )

    print(
        "New training:                    NO"
    )

    print(
        "T60 rescoring:                   NO"
    )

    print(
        "Final configuration changed:     NO"
    )

    print()
    print(
        "PHASE 7.4: PASS / "
        "TRAINING-BUDGET SCALING AND "
        "LEARNING DYNAMICS AUDITED"
    )


if __name__ == "__main__":
    main()
