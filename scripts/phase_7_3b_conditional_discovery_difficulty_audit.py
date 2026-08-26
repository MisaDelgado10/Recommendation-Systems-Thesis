#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# FROZEN INPUTS
# =============================================================================

INPUT_DIR = Path("phase_7_inputs")
OUTPUT_DIR = Path("phase_7_outputs/phase_7_3b")

METRICS_PATH = INPUT_DIR / "final_t60_test_case_metrics.parquet"
SPLIT_PATH = INPUT_DIR / "t60_validation_test_split.parquet"
RESULT_PATH = INPUT_DIR / "final_t60_test_result.json"
MANIFEST_PATH = INPUT_DIR / "phase_7_handoff_manifest.json"

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

BOOTSTRAP_SEED = 42
BOOTSTRAP_REPLICATES = 5_000


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


def percentile_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)

    return (
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    )


def summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for group_name, subset in frame.groupby(
        "pair_novelty_group",
        sort=True,
        dropna=False,
    ):
        rows.append(
            {
                "pair_novelty_group":
                    group_name,

                "events":
                    int(len(subset)),

                "unique_investors":
                    int(
                        subset[
                            "investor_global"
                        ].nunique()
                    ),

                "unique_startups":
                    int(
                        subset[
                            "startup_id"
                        ].nunique()
                    ),

                "hits_at_10":
                    int(
                        subset[
                            "HR@10"
                        ].sum()
                    ),

                "HR@10":
                    float(
                        subset[
                            "HR@10"
                        ].mean()
                    ),

                "NDCG@10":
                    float(
                        subset[
                            "NDCG@10"
                        ].mean()
                    ),

                "mean_positive_rank":
                    float(
                        subset[
                            "positive_rank"
                        ].mean()
                    ),

                "median_positive_rank":
                    float(
                        subset[
                            "positive_rank"
                        ].median()
                    ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# CLUSTER BOOTSTRAP
# =============================================================================

def investor_cluster_bootstrap(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    # Primary inferential unit = investor.
    #
    # All events belonging to a sampled investor are resampled together.
    # This retains within-investor dependence better than a naive event bootstrap.

    investors = np.array(
        sorted(
            frame[
                "investor_global"
            ].unique()
        ),
        dtype=np.int64,
    )

    investor_position = {
        int(investor): i
        for i, investor
        in enumerate(investors)
    }

    n = len(investors)

    require(
        n > 1,
        "Insufficient investors for cluster bootstrap.",
    )

    groups = [
        "new_to_investor",
        "previous_investor_startup_pair",
    ]

    arrays = {}

    for group in groups:

        subset = frame.loc[
            frame[
                "pair_novelty_group"
            ]
            == group
        ]

        count = np.zeros(
            n,
            dtype=np.float64,
        )

        hr_sum = np.zeros(
            n,
            dtype=np.float64,
        )

        ndcg_sum = np.zeros(
            n,
            dtype=np.float64,
        )

        rank_sum = np.zeros(
            n,
            dtype=np.float64,
        )

        agg = (
            subset
            .groupby(
                "investor_global",
                sort=False,
            )
            .agg(
                count=(
                    "HR@10",
                    "size",
                ),
                hr_sum=(
                    "HR@10",
                    "sum",
                ),
                ndcg_sum=(
                    "NDCG@10",
                    "sum",
                ),
                rank_sum=(
                    "positive_rank",
                    "sum",
                ),
            )
        )

        for investor, row in agg.iterrows():

            i = investor_position[
                int(investor)
            ]

            count[i] = float(
                row["count"]
            )

            hr_sum[i] = float(
                row["hr_sum"]
            )

            ndcg_sum[i] = float(
                row["ndcg_sum"]
            )

            rank_sum[i] = float(
                row["rank_sum"]
            )

        arrays[group] = {
            "count": count,
            "hr_sum": hr_sum,
            "ndcg_sum": ndcg_sum,
            "rank_sum": rank_sum,
        }

    rng = np.random.default_rng(
        BOOTSTRAP_SEED
    )

    records = []

    for replicate in range(
        BOOTSTRAP_REPLICATES
    ):

        sampled = rng.integers(
            0,
            n,
            size=n,
        )

        metrics = {}

        valid = True

        for group in groups:

            data = arrays[group]

            denominator = float(
                data[
                    "count"
                ][
                    sampled
                ].sum()
            )

            if denominator <= 0:
                valid = False
                break

            metrics[group] = {
                "HR@10":
                    float(
                        data[
                            "hr_sum"
                        ][
                            sampled
                        ].sum()
                        / denominator
                    ),

                "NDCG@10":
                    float(
                        data[
                            "ndcg_sum"
                        ][
                            sampled
                        ].sum()
                        / denominator
                    ),

                "mean_positive_rank":
                    float(
                        data[
                            "rank_sum"
                        ][
                            sampled
                        ].sum()
                        / denominator
                    ),
            }

        if not valid:
            continue

        new = metrics[
            "new_to_investor"
        ]

        previous = metrics[
            "previous_investor_startup_pair"
        ]

        records.append(
            {
                "replicate":
                    replicate,

                "new_HR@10":
                    new["HR@10"],

                "previous_HR@10":
                    previous["HR@10"],

                "HR_gap_previous_minus_new":
                    (
                        previous["HR@10"]
                        - new["HR@10"]
                    ),

                "new_NDCG@10":
                    new["NDCG@10"],

                "previous_NDCG@10":
                    previous["NDCG@10"],

                "NDCG_gap_previous_minus_new":
                    (
                        previous["NDCG@10"]
                        - new["NDCG@10"]
                    ),

                "new_mean_positive_rank":
                    new[
                        "mean_positive_rank"
                    ],

                "previous_mean_positive_rank":
                    previous[
                        "mean_positive_rank"
                    ],

                "rank_gap_new_minus_previous":
                    (
                        new[
                            "mean_positive_rank"
                        ]
                        - previous[
                            "mean_positive_rank"
                        ]
                    ),
            }
        )

    bootstrap = pd.DataFrame(
        records
    )

    require(
        len(bootstrap)
        == BOOTSTRAP_REPLICATES,
        (
            "Unexpected invalid bootstrap replicates: "
            f"{len(bootstrap)} / {BOOTSTRAP_REPLICATES}"
        ),
    )

    ci_rows = []

    for column in [
        "new_HR@10",
        "previous_HR@10",
        "HR_gap_previous_minus_new",

        "new_NDCG@10",
        "previous_NDCG@10",
        "NDCG_gap_previous_minus_new",

        "new_mean_positive_rank",
        "previous_mean_positive_rank",
        "rank_gap_new_minus_previous",
    ]:

        lower, upper = percentile_ci(
            bootstrap[column].to_numpy()
        )

        ci_rows.append(
            {
                "quantity":
                    column,

                "bootstrap_mean":
                    float(
                        bootstrap[
                            column
                        ].mean()
                    ),

                "ci95_lower":
                    lower,

                "ci95_upper":
                    upper,
            }
        )

    return (
        bootstrap,
        pd.DataFrame(ci_rows),
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    banner(
        "PHASE 7.3b — CONDITIONAL DISCOVERY DIFFICULTY AUDIT"
    )

    # -------------------------------------------------------------------------
    # 1. Frozen input verification
    # -------------------------------------------------------------------------

    for path, expected_hash in EXPECTED_HASHES.items():

        require(
            path.exists(),
            f"Missing frozen input: {path}",
        )

        actual_hash = sha256_file(
            path
        )

        require(
            actual_hash
            == expected_hash,
            (
                f"Hash drift for {path}\n"
                f"Expected: {expected_hash}\n"
                f"Actual:   {actual_hash}"
            ),
        )

        print(
            f"PASS HASH  {path}"
        )

    # -------------------------------------------------------------------------
    # 2. Load frozen predictions and metadata
    # -------------------------------------------------------------------------

    metrics = pd.read_parquet(
        METRICS_PATH
    )

    split = pd.read_parquet(
        SPLIT_PATH
    )

    result = json.loads(
        RESULT_PATH.read_text(
            encoding="utf-8"
        )
    )

    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    require(
        len(metrics)
        == EXPECTED_TEST_CASES,
        "Final metric row-count drift.",
    )

    require(
        manifest[
            "final_t60_result"
        ][
            "test_rescoring_allowed"
        ]
        is False,
        "Frozen T60 policy drift.",
    )

    # -------------------------------------------------------------------------
    # 3. Bind final predictions to T60 metadata
    # -------------------------------------------------------------------------

    test_metadata = (
        split.loc[
            split[
                "evaluation_split"
            ]
            == "test",
            [
                "interaction_id",
                "investor_id",
                "startup_id",
                "investor_seen_in_t1_t59",
                "startup_seen_in_t1_t59",
                "new_to_investor_pair",
                "pair_seen_before_t60",
                "pair_seen_in_t1_t59",
            ],
        ]
        .copy()
    )

    require(
        len(test_metadata)
        == EXPECTED_TEST_CASES,
        "Frozen test metadata count drift.",
    )

    merged = metrics.merge(
        test_metadata,
        on="interaction_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    require(
        bool(
            (
                merged["_merge"]
                == "both"
            ).all()
        ),
        "Prediction/metadata binding failure.",
    )

    merged = merged.drop(
        columns="_merge"
    )

    # -------------------------------------------------------------------------
    # 4. Strict warm–warm evaluation regime
    # -------------------------------------------------------------------------
    #
    # Both entities must have appeared during the trainable history T1–T59.
    #
    # This removes the entity cold-start explanation and asks:
    #
    #   GIVEN known investor + known startup,
    #   is a previously unseen pair still harder?
    #

    warm_warm = (
        merged.loc[
            (
                merged[
                    "investor_seen_in_t1_t59"
                ].astype(bool)
            )
            &
            (
                merged[
                    "startup_seen_in_t1_t59"
                ].astype(bool)
            )
        ]
        .copy()
    )

    require(
        len(warm_warm) > 0,
        "No warm–warm cases found.",
    )

    warm_warm[
        "pair_novelty_group"
    ] = np.where(
        warm_warm[
            "new_to_investor_pair"
        ].astype(bool),
        "new_to_investor",
        "previous_investor_startup_pair",
    )

    require(
        set(
            warm_warm[
                "pair_novelty_group"
            ].unique()
        )
        == {
            "new_to_investor",
            "previous_investor_startup_pair",
        },
        "Expected both pair-novelty groups.",
    )

    group_summary = summary(
        warm_warm
    )

    summary_by_name = (
        group_summary
        .set_index(
            "pair_novelty_group"
        )
    )

    new = summary_by_name.loc[
        "new_to_investor"
    ]

    previous = summary_by_name.loc[
        "previous_investor_startup_pair"
    ]

    # -------------------------------------------------------------------------
    # 5. Direct conditional discovery gaps
    # -------------------------------------------------------------------------

    comparison = pd.DataFrame(
        [
            {
                "comparison":
                    (
                        "warm_investor + warm_startup: "
                        "previous_pair minus new_pair"
                    ),

                "new_pair_events":
                    int(
                        new["events"]
                    ),

                "previous_pair_events":
                    int(
                        previous["events"]
                    ),

                "new_pair_HR@10":
                    float(
                        new["HR@10"]
                    ),

                "previous_pair_HR@10":
                    float(
                        previous["HR@10"]
                    ),

                "HR_absolute_gap":
                    float(
                        previous["HR@10"]
                        - new["HR@10"]
                    ),

                "HR_relative_discovery_penalty_pct":
                    float(
                        100.0
                        * (
                            previous["HR@10"]
                            - new["HR@10"]
                        )
                        / previous["HR@10"]
                    ),

                "new_pair_NDCG@10":
                    float(
                        new["NDCG@10"]
                    ),

                "previous_pair_NDCG@10":
                    float(
                        previous["NDCG@10"]
                    ),

                "NDCG_absolute_gap":
                    float(
                        previous["NDCG@10"]
                        - new["NDCG@10"]
                    ),

                "NDCG_relative_discovery_penalty_pct":
                    float(
                        100.0
                        * (
                            previous["NDCG@10"]
                            - new["NDCG@10"]
                        )
                        / previous["NDCG@10"]
                    ),

                "new_pair_mean_rank":
                    float(
                        new[
                            "mean_positive_rank"
                        ]
                    ),

                "previous_pair_mean_rank":
                    float(
                        previous[
                            "mean_positive_rank"
                        ]
                    ),

                "mean_rank_penalty":
                    float(
                        new[
                            "mean_positive_rank"
                        ]
                        - previous[
                            "mean_positive_rank"
                        ]
                    ),
            }
        ]
    )

    # -------------------------------------------------------------------------
    # 6. Investor-cluster bootstrap
    # -------------------------------------------------------------------------

    banner(
        "RUNNING DETERMINISTIC INVESTOR-CLUSTER BOOTSTRAP"
    )

    bootstrap, bootstrap_ci = (
        investor_cluster_bootstrap(
            warm_warm
        )
    )

    # -------------------------------------------------------------------------
    # 7. Persist aggregate results
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    group_summary.to_csv(
        OUTPUT_DIR
        / "warm_warm_pair_novelty_performance.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "conditional_discovery_gap.csv",
        index=False,
    )

    bootstrap_ci.to_csv(
        OUTPUT_DIR
        / "investor_cluster_bootstrap_ci.csv",
        index=False,
    )

    # Bootstrap replicates are intentionally not persisted:
    # only the deterministic aggregate confidence intervals are needed.

    result_contract = {
        "schema_version":
            "ITRS_PHASE_7_3B_CONDITIONAL_DISCOVERY_AUDIT_V1",

        "status":
            "PASS",

        "analysis_type":
            (
                "POST_HOC_CONDITIONAL_DISCOVERY_DIFFICULTY"
            ),

        "condition":
            (
                "investor_seen_in_t1_t59 == True AND "
                "startup_seen_in_t1_t59 == True"
            ),

        "comparison":
            (
                "new_to_investor_pair versus "
                "previous_investor_startup_pair"
            ),

        "bootstrap": {
            "unit":
                "investor_global",

            "seed":
                BOOTSTRAP_SEED,

            "replicates":
                BOOTSTRAP_REPLICATES,

            "interval":
                "percentile_95pct",
        },

        "new_model_inference":
            False,

        "t60_rescoring":
            False,

        "training_performed":
            False,

        "model_selection_performed":
            False,

        "input_hashes": {
            str(path):
                expected_hash
            for path, expected_hash
            in EXPECTED_HASHES.items()
        },

        "warm_warm_events":
            int(
                len(warm_warm)
            ),

        "direct_comparison":
            comparison.iloc[
                0
            ].to_dict(),
    }

    (
        OUTPUT_DIR
        / "phase_7_3b_conditional_discovery_audit_result.json"
    ).write_text(
        json.dumps(
            result_contract,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # 8. Report
    # -------------------------------------------------------------------------

    banner(
        "STRICT WARM–WARM PAIR-NOVELTY PERFORMANCE"
    )

    print(
        group_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    banner(
        "CONDITIONAL DISCOVERY GAP"
    )

    print(
        comparison.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    banner(
        "INVESTOR-CLUSTER BOOTSTRAP — 95% CONFIDENCE INTERVALS"
    )

    print(
        bootstrap_ci.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    banner(
        "PHASE 7.3b FINAL STATUS"
    )

    print(
        f"Frozen T60 predictions analyzed: "
        f"{EXPECTED_TEST_CASES:,}"
    )

    print(
        f"Strict warm–warm cases analyzed: "
        f"{len(warm_warm):,}"
    )

    print(
        f"Cluster bootstrap replicates:     "
        f"{BOOTSTRAP_REPLICATES:,}"
    )

    print(
        "Bootstrap unit:                   investor"
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
        "PHASE 7.3b: PASS / "
        "CONDITIONAL DISCOVERY DIFFICULTY AUDITED"
    )


if __name__ == "__main__":
    main()
