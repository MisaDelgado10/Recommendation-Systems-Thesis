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
OUTPUT_DIR = Path("phase_7_outputs/phase_7_3c")

METRICS_PATH = INPUT_DIR / "final_t60_test_case_metrics.parquet"
SPLIT_PATH = INPUT_DIR / "t60_validation_test_split.parquet"
RESULT_PATH = INPUT_DIR / "final_t60_test_result.json"
MANIFEST_PATH = INPUT_DIR / "phase_7_handoff_manifest.json"

EXPECTED_TEST_CASES = 20_264
EXPECTED_DISCOVERY_CASES = 16_446

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


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []

    order = [
        "cold_investor__cold_startup",
        "cold_investor__warm_startup",
        "warm_investor__cold_startup",
        "warm_investor__warm_startup",
    ]

    for group in order:

        subset = frame.loc[
            frame["discovery_history_group"] == group
        ]

        require(
            len(subset) > 0,
            f"Empty discovery subgroup: {group}",
        )

        rows.append(
            {
                "discovery_history_group":
                    group,

                "events":
                    int(len(subset)),

                "event_share_within_strict_discovery_pct":
                    float(
                        100.0
                        * len(subset)
                        / len(frame)
                    ),

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


def metric_by_group(
    frame: pd.DataFrame,
    group: str,
    metric: str,
) -> float:

    subset = frame.loc[
        frame[
            "discovery_history_group"
        ]
        == group
    ]

    return float(
        subset[metric].mean()
    )


# =============================================================================
# INVESTOR-CLUSTER BOOTSTRAP
# =============================================================================

def investor_cluster_bootstrap(
    frame: pd.DataFrame,
) -> pd.DataFrame:

    investors = np.array(
        sorted(
            frame[
                "investor_global"
            ].unique()
        ),
        dtype=np.int64,
    )

    n_investors = len(investors)

    require(
        n_investors > 1,
        "Insufficient investors for bootstrap.",
    )

    investor_position = {
        int(investor): i
        for i, investor
        in enumerate(investors)
    }

    groups = [
        "cold_investor__cold_startup",
        "cold_investor__warm_startup",
        "warm_investor__cold_startup",
        "warm_investor__warm_startup",
    ]

    arrays = {}

    for group in groups:

        subset = frame.loc[
            frame[
                "discovery_history_group"
            ]
            == group
        ]

        count = np.zeros(
            n_investors,
            dtype=np.float64,
        )

        hr_sum = np.zeros(
            n_investors,
            dtype=np.float64,
        )

        ndcg_sum = np.zeros(
            n_investors,
            dtype=np.float64,
        )

        rank_sum = np.zeros(
            n_investors,
            dtype=np.float64,
        )

        aggregate = (
            subset
            .groupby(
                "investor_global",
                sort=False,
            )
            .agg(
                count=("HR@10", "size"),
                hr_sum=("HR@10", "sum"),
                ndcg_sum=("NDCG@10", "sum"),
                rank_sum=("positive_rank", "sum"),
            )
        )

        for investor, row in aggregate.iterrows():

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
            n_investors,
            size=n_investors,
        )

        metrics = {}

        valid = True

        for group in groups:

            data = arrays[group]

            denominator = float(
                data[
                    "count"
                ][sampled].sum()
            )

            if denominator <= 0:
                valid = False
                break

            metrics[group] = {
                "HR@10":
                    float(
                        data[
                            "hr_sum"
                        ][sampled].sum()
                        / denominator
                    ),

                "NDCG@10":
                    float(
                        data[
                            "ndcg_sum"
                        ][sampled].sum()
                        / denominator
                    ),

                "mean_positive_rank":
                    float(
                        data[
                            "rank_sum"
                        ][sampled].sum()
                        / denominator
                    ),
            }

        if not valid:
            continue

        cc = metrics[
            "cold_investor__cold_startup"
        ]

        cw = metrics[
            "cold_investor__warm_startup"
        ]

        wc = metrics[
            "warm_investor__cold_startup"
        ]

        ww = metrics[
            "warm_investor__warm_startup"
        ]

        records.append(
            {
                "replicate":
                    replicate,

                # Startup-history advantage, conditional on investor history.
                "HR_startup_history_effect_given_cold_investor":
                    cw["HR@10"] - cc["HR@10"],

                "HR_startup_history_effect_given_warm_investor":
                    ww["HR@10"] - wc["HR@10"],

                "NDCG_startup_history_effect_given_cold_investor":
                    cw["NDCG@10"] - cc["NDCG@10"],

                "NDCG_startup_history_effect_given_warm_investor":
                    ww["NDCG@10"] - wc["NDCG@10"],

                # Investor-history advantage, conditional on startup history.
                "HR_investor_history_effect_given_cold_startup":
                    wc["HR@10"] - cc["HR@10"],

                "HR_investor_history_effect_given_warm_startup":
                    ww["HR@10"] - cw["HR@10"],

                "NDCG_investor_history_effect_given_cold_startup":
                    wc["NDCG@10"] - cc["NDCG@10"],

                "NDCG_investor_history_effect_given_warm_startup":
                    ww["NDCG@10"] - cw["NDCG@10"],

                # Best-vs-worst discovery regime.
                "HR_warm_warm_minus_cold_cold":
                    ww["HR@10"] - cc["HR@10"],

                "NDCG_warm_warm_minus_cold_cold":
                    ww["NDCG@10"] - cc["NDCG@10"],

                "rank_cold_cold_minus_warm_warm":
                    (
                        cc["mean_positive_rank"]
                        - ww["mean_positive_rank"]
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
            "Invalid bootstrap replicate count: "
            f"{len(bootstrap)} / "
            f"{BOOTSTRAP_REPLICATES}"
        ),
    )

    return bootstrap


def bootstrap_summary(
    bootstrap: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for column in bootstrap.columns:

        if column == "replicate":
            continue

        values = bootstrap[
            column
        ].to_numpy(
            dtype=np.float64
        )

        rows.append(
            {
                "quantity":
                    column,

                "bootstrap_mean":
                    float(
                        np.mean(values)
                    ),

                "ci95_lower":
                    float(
                        np.quantile(
                            values,
                            0.025,
                        )
                    ),

                "ci95_upper":
                    float(
                        np.quantile(
                            values,
                            0.975,
                        )
                    ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    banner(
        "PHASE 7.3c — "
        "DISCOVERY-SPECIFIC COLD-START AUDIT"
    )

    # -------------------------------------------------------------------------
    # 1. Verify frozen input identities.
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
            actual_hash == expected_hash,
            (
                f"Frozen input hash drift:\n"
                f"  path:     {path}\n"
                f"  expected: {expected_hash}\n"
                f"  actual:   {actual_hash}"
            ),
        )

        print(
            f"PASS HASH  {path}"
        )

    # -------------------------------------------------------------------------
    # 2. Load frozen predictions and metadata.
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
    # 3. Bind final TEST predictions to temporal metadata.
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

                "investor_seen_before_t60",
                "investor_seen_in_t1_t59",
                "investor_t0_only_history",

                "startup_seen_before_t60",
                "startup_seen_in_t1_t59",
                "startup_t0_only_history",

                "new_to_investor_pair",
            ],
        ]
        .copy()
    )

    require(
        len(test_metadata)
        == EXPECTED_TEST_CASES,
        "Frozen TEST metadata count drift.",
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
    # 4. Restrict to genuine new-to-investor discovery events.
    # -------------------------------------------------------------------------

    discovery = (
        merged.loc[
            merged[
                "new_to_investor_pair"
            ].astype(bool)
        ]
        .copy()
    )

    require(
        len(discovery)
        == EXPECTED_DISCOVERY_CASES,
        (
            "Discovery event count drift: "
            f"{len(discovery):,}"
        ),
    )

    # -------------------------------------------------------------------------
    # 5. Define strict warm/cold history states.
    #
    # Warm = observed in the trainable T1–T59 history.
    # Cold = completely unseen before T60.
    #
    # T0-only entities are excluded from the strict 2x2 analysis.
    # -------------------------------------------------------------------------

    investor_warm = (
        discovery[
            "investor_seen_in_t1_t59"
        ].astype(bool)
    )

    investor_cold = (
        ~discovery[
            "investor_seen_before_t60"
        ].astype(bool)
    )

    startup_warm = (
        discovery[
            "startup_seen_in_t1_t59"
        ].astype(bool)
    )

    startup_cold = (
        ~discovery[
            "startup_seen_before_t60"
        ].astype(bool)
    )

    valid_investor_state = (
        investor_warm
        | investor_cold
    )

    valid_startup_state = (
        startup_warm
        | startup_cold
    )

    strict = (
        discovery.loc[
            valid_investor_state
            & valid_startup_state
        ]
        .copy()
    )

    excluded = (
        len(discovery)
        - len(strict)
    )

    require(
        bool(
            (
                strict[
                    "investor_t0_only_history"
                ].astype(bool)
                == False
            ).all()
        ),
        "T0-only investor leaked into strict analysis.",
    )

    require(
        bool(
            (
                strict[
                    "startup_t0_only_history"
                ].astype(bool)
                == False
            ).all()
        ),
        "T0-only startup leaked into strict analysis.",
    )

    strict[
        "investor_history_binary"
    ] = np.where(
        strict[
            "investor_seen_in_t1_t59"
        ].astype(bool),
        "warm_investor",
        "cold_investor",
    )

    strict[
        "startup_history_binary"
    ] = np.where(
        strict[
            "startup_seen_in_t1_t59"
        ].astype(bool),
        "warm_startup",
        "cold_startup",
    )

    strict[
        "discovery_history_group"
    ] = (
        strict[
            "investor_history_binary"
        ]
        + "__"
        + strict[
            "startup_history_binary"
        ]
    )

    expected_groups = {
        "cold_investor__cold_startup",
        "cold_investor__warm_startup",
        "warm_investor__cold_startup",
        "warm_investor__warm_startup",
    }

    require(
        set(
            strict[
                "discovery_history_group"
            ].unique()
        )
        == expected_groups,
        "Strict discovery 2x2 groups incomplete.",
    )

    # -------------------------------------------------------------------------
    # 6. Group performance.
    # -------------------------------------------------------------------------

    group_summary = summarize(
        strict
    )

    by_name = (
        group_summary
        .set_index(
            "discovery_history_group"
        )
    )

    cc = by_name.loc[
        "cold_investor__cold_startup"
    ]

    cw = by_name.loc[
        "cold_investor__warm_startup"
    ]

    wc = by_name.loc[
        "warm_investor__cold_startup"
    ]

    ww = by_name.loc[
        "warm_investor__warm_startup"
    ]

    # -------------------------------------------------------------------------
    # 7. Direct descriptive contrasts.
    # -------------------------------------------------------------------------

    contrasts = pd.DataFrame(
        [
            {
                "contrast":
                    "startup_history_effect_given_cold_investor",

                "HR_absolute_difference":
                    float(
                        cw["HR@10"]
                        - cc["HR@10"]
                    ),

                "NDCG_absolute_difference":
                    float(
                        cw["NDCG@10"]
                        - cc["NDCG@10"]
                    ),
            },

            {
                "contrast":
                    "startup_history_effect_given_warm_investor",

                "HR_absolute_difference":
                    float(
                        ww["HR@10"]
                        - wc["HR@10"]
                    ),

                "NDCG_absolute_difference":
                    float(
                        ww["NDCG@10"]
                        - wc["NDCG@10"]
                    ),
            },

            {
                "contrast":
                    "investor_history_effect_given_cold_startup",

                "HR_absolute_difference":
                    float(
                        wc["HR@10"]
                        - cc["HR@10"]
                    ),

                "NDCG_absolute_difference":
                    float(
                        wc["NDCG@10"]
                        - cc["NDCG@10"]
                    ),
            },

            {
                "contrast":
                    "investor_history_effect_given_warm_startup",

                "HR_absolute_difference":
                    float(
                        ww["HR@10"]
                        - cw["HR@10"]
                    ),

                "NDCG_absolute_difference":
                    float(
                        ww["NDCG@10"]
                        - cw["NDCG@10"]
                    ),
            },

            {
                "contrast":
                    "warm_warm_minus_cold_cold",

                "HR_absolute_difference":
                    float(
                        ww["HR@10"]
                        - cc["HR@10"]
                    ),

                "NDCG_absolute_difference":
                    float(
                        ww["NDCG@10"]
                        - cc["NDCG@10"]
                    ),
            },
        ]
    )

    # -------------------------------------------------------------------------
    # 8. Investor-cluster bootstrap.
    # -------------------------------------------------------------------------

    banner(
        "RUNNING DETERMINISTIC "
        "INVESTOR-CLUSTER BOOTSTRAP"
    )

    bootstrap = investor_cluster_bootstrap(
        strict
    )

    bootstrap_ci = bootstrap_summary(
        bootstrap
    )

    # -------------------------------------------------------------------------
    # 9. Persist aggregate analysis outputs.
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    group_summary.to_csv(
        OUTPUT_DIR
        / "discovery_cold_start_2x2_performance.csv",
        index=False,
    )

    contrasts.to_csv(
        OUTPUT_DIR
        / "discovery_history_contrasts.csv",
        index=False,
    )

    bootstrap_ci.to_csv(
        OUTPUT_DIR
        / "investor_cluster_bootstrap_ci.csv",
        index=False,
    )

    result_contract = {
        "schema_version":
            "ITRS_PHASE_7_3C_DISCOVERY_COLD_START_AUDIT_V1",

        "status":
            "PASS",

        "analysis_type":
            "POST_HOC_DISCOVERY_SPECIFIC_COLD_START_ANALYSIS",

        "scope":
            "new_to_investor_pair_only",

        "history_definition": {
            "warm":
                "entity_seen_in_t1_t59 == True",

            "cold":
                "entity_seen_before_t60 == False",

            "t0_only":
                "excluded_from_strict_2x2",
        },

        "test_cases_available":
            EXPECTED_TEST_CASES,

        "discovery_cases_available":
            int(
                len(discovery)
            ),

        "strict_discovery_cases_analyzed":
            int(
                len(strict)
            ),

        "discovery_cases_excluded_from_strict_2x2":
            int(
                excluded
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
    }

    (
        OUTPUT_DIR
        / "phase_7_3c_discovery_cold_start_audit_result.json"
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
    # 10. Human-readable report.
    # -------------------------------------------------------------------------

    banner(
        "DISCOVERY-ONLY STRICT 2×2 PERFORMANCE"
    )

    print(
        group_summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    banner(
        "DISCOVERY HISTORY CONTRASTS"
    )

    print(
        contrasts.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    banner(
        "INVESTOR-CLUSTER BOOTSTRAP — "
        "95% CONFIDENCE INTERVALS"
    )

    print(
        bootstrap_ci.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    banner(
        "PHASE 7.3c FINAL STATUS"
    )

    print(
        f"Frozen T60 predictions available:    "
        f"{EXPECTED_TEST_CASES:,}"
    )

    print(
        f"New-to-investor discovery cases:      "
        f"{len(discovery):,}"
    )

    print(
        f"Strict 2x2 discovery cases analyzed:  "
        f"{len(strict):,}"
    )

    print(
        f"T0-only/ambiguous cases excluded:     "
        f"{excluded:,}"
    )

    print(
        f"Cluster bootstrap replicates:         "
        f"{BOOTSTRAP_REPLICATES:,}"
    )

    print(
        "Bootstrap unit:                       investor"
    )

    print(
        "New model inference:                 NO"
    )

    print(
        "T60 rescoring:                       NO"
    )

    print(
        "Training performed:                  NO"
    )

    print(
        "Model/configuration selection:       NO"
    )

    print()
    print(
        "PHASE 7.3c: PASS / "
        "DISCOVERY-SPECIFIC COLD-START "
        "PERFORMANCE AUDITED"
    )


if __name__ == "__main__":
    main()
