#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# Phase 7.4 — Novelty × Cold-Start Interaction Diagnostic V1
#
# Core thesis questions
# ---------------------
# A) Among WARM investors and WARM startups:
#       Do new-to-investor pairs rank worse than prior relationships?
#
# B) Among NEW-TO-INVESTOR cases only:
#       How does ranking vary across:
#           warm investor + warm startup
#           warm investor + cold startup
#           cold investor + warm startup
#           cold investor + cold startup
#
# C) Where are Top-10 and severe rank>50 failures concentrated?
#
# D) Is there descriptive departure from additivity when BOTH entities are cold?
#
# Scientific boundary
# -------------------
# - immutable Phase-6 one-shot final test
# - post-hoc analysis only
# - no model/checkpoint/logit loading
# - no inference
# - no test rescoring
# - no negative resampling
# - no candidate modification
# - no model selection
#
# Interpretation boundary
# -----------------------
# Stratification reduces important confounding between pair novelty and entity
# cold-start status. It does NOT establish causal effects.
#
# Uncertainty
# -----------
# Percentile bootstrap with investor_id as the resampling unit.
# Startup/multiway clustering is reserved for later robustness analysis.
# =============================================================================


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    REPO_ROOT
    / "data/experimental/phase_6/full_training/100pct/final_test/"
    / "analysis_ready_test_cases.parquet"
)

PHASE_7_0_CONTRACT = (
    REPO_ROOT
    / "data/experimental/phase_7/phase_7_0_analysis_contract/"
    / "phase_7_0_analysis_contract_V2.json"
)

PHASE_7_2_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/phase_7_2_pair_novelty/"
    / "phase_7_2_analysis_manifest_V1.json"
)

PHASE_7_3_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_7/phase_7_3_cold_start/"
    / "phase_7_3_analysis_manifest_V1.json"
)

OUT_DIR = (
    REPO_ROOT
    / "data/experimental/phase_7/"
    / "phase_7_4_novelty_cold_start_interaction"
)

FIG_DIR = OUT_DIR / "figures"

EXPECTED_SOURCE_SHA256 = (
    "d70f21bff0006e094c5d567d307c0664"
    "b41a11e7250a0e69aaec610e1810132d"
)

EXPECTED_ROWS = 20_264

GROUP_LABELS = {
    "prior_ww": "Prior pair | Warm investor + Warm startup",
    "new_ww": "New pair | Warm investor + Warm startup",
    "new_wc": "New pair | Warm investor + Cold startup",
    "new_cw": "New pair | Cold investor + Warm startup",
    "new_cc": "New pair | Cold investor + Cold startup",
}

GROUP_ORDER = [
    "prior_ww",
    "new_ww",
    "new_wc",
    "new_cw",
    "new_cc",
]

EXPECTED_COUNTS = {
    "prior_ww": 3_818,
    "new_ww": 6_889,
    "new_wc": 6_392,
    "new_cw": 1_453,
    "new_cc": 1_712,
}

NEW_GROUPS = [
    "new_ww",
    "new_wc",
    "new_cw",
    "new_cc",
]

RANK_LABELS = [
    "Rank 1",
    "Ranks 2–5",
    "Ranks 6–10",
    "Ranks 11–20",
    "Ranks 21–50",
    "Ranks 51–100",
]

BOOTSTRAP_REPS = 2_000
BOOTSTRAP_SEED = 7_104_001


# =============================================================================
# Helpers
# =============================================================================


def require(condition: bool, message: str) -> None:

    if not bool(condition):
        raise AssertionError(message)


def file_sha256(
    path: Path,
    chunk_size: int = 8 * 1024 * 1024,
) -> str:

    h = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            block = f.read(chunk_size)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def json_converter(value):

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):

        if np.isnan(value):
            return None

        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if pd.isna(value):
        return None

    raise TypeError(
        f"Object of type {type(value).__name__} "
        "is not JSON serializable"
    )


def json_dump(obj, path: Path) -> None:

    with path.open("w", encoding="utf-8") as f:

        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            default=json_converter,
        )


def print_section(title: str) -> None:

    print()
    print("=" * 110)
    print(title)
    print("=" * 110)


def save_figure(fig, stem: str) -> list[Path]:

    png = FIG_DIR / f"{stem}.png"
    pdf = FIG_DIR / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        metadata={
            "Title": stem,
            "Author": "ITRS Phase 7 diagnostic analysis",
        },
    )

    plt.close(fig)

    return [png, pdf]


def calculate_metrics(group: pd.DataFrame) -> dict:

    rank = group["positive_rank"].astype(int)

    return {
        "n": int(len(group)),
        "Hits@1": int((rank <= 1).sum()),
        "Hits@5": int((rank <= 5).sum()),
        "Hits@10": int((rank <= 10).sum()),
        "HR@1": float((rank <= 1).mean()),
        "HR@5": float((rank <= 5).mean()),
        "HR@10": float((rank <= 10).mean()),
        "NDCG@10": float(group["NDCG@10"].mean()),
        "mean_rank": float(rank.mean()),
        "median_rank": float(rank.median()),
        "std_rank": float(rank.std(ddof=1)),
        "P25_rank": float(rank.quantile(0.25)),
        "P75_rank": float(rank.quantile(0.75)),
        "P90_rank": float(rank.quantile(0.90)),
        "P95_rank": float(rank.quantile(0.95)),
        "rank_gt_10_n": int((rank > 10).sum()),
        "rank_gt_10_share": float((rank > 10).mean()),
        "rank_gt_20_n": int((rank > 20).sum()),
        "rank_gt_20_share": float((rank > 20).mean()),
        "rank_gt_50_n": int((rank > 50).sum()),
        "rank_gt_50_share": float((rank > 50).mean()),
        "rank_gt_75_n": int((rank > 75).sum()),
        "rank_gt_75_share": float((rank > 75).mean()),
    }


def percentile_ci(
    values: pd.Series,
) -> tuple[float, float]:

    values = (
        values
        .dropna()
        .to_numpy(dtype=float)
    )

    require(
        len(values) > 0,
        "Cannot compute CI from empty bootstrap values.",
    )

    low, high = np.quantile(
        values,
        [0.025, 0.975],
    )

    return float(low), float(high)


def build_diagnostic_group(df: pd.DataFrame) -> pd.Series:

    new = df["new_to_investor_pair"].astype(bool)
    cold_i = df["cold_start_investor"].astype(bool)
    cold_s = df["cold_start_startup"].astype(bool)

    return pd.Series(
        np.select(
            [
                (~new) & (~cold_i) & (~cold_s),
                new & (~cold_i) & (~cold_s),
                new & (~cold_i) & cold_s,
                new & cold_i & (~cold_s),
                new & cold_i & cold_s,
            ],
            GROUP_ORDER,
            default="INVALID",
        ),
        index=df.index,
    )


def investor_cluster_bootstrap(
    df: pd.DataFrame,
    reps: int,
    seed: int,
) -> pd.DataFrame:

    working = df[
        [
            "investor_id",
            "diagnostic_group",
            "positive_rank",
            "NDCG@10",
        ]
    ].copy()

    working["hit10"] = (
        working["positive_rank"]
        .le(10)
        .astype(int)
    )

    investors = (
        working["investor_id"]
        .drop_duplicates()
        .to_numpy()
    )

    arrays = {}

    for group_key in GROUP_ORDER:

        grouped = (
            working.loc[
                working["diagnostic_group"]
                == group_key
            ]
            .groupby(
                "investor_id",
                sort=False,
            )
            .agg(
                n=("positive_rank", "size"),
                hits10=("hit10", "sum"),
                ndcg_sum=("NDCG@10", "sum"),
                rank_sum=("positive_rank", "sum"),
            )
            .reindex(
                investors,
                fill_value=0,
            )
        )

        arrays[group_key] = {
            col: grouped[col].to_numpy(dtype=float)
            for col in (
                "n",
                "hits10",
                "ndcg_sum",
                "rank_sum",
            )
        }

    rng = np.random.default_rng(seed)

    rows = []

    for bootstrap_index in range(reps):

        sampled = rng.integers(
            0,
            len(investors),
            size=len(investors),
        )

        result = {
            "bootstrap_index": bootstrap_index,
        }

        valid = True

        for group_key in GROUP_ORDER:

            arr = arrays[group_key]

            n = float(
                arr["n"][sampled].sum()
            )

            if n <= 0:

                valid = False
                break

            result[
                f"{group_key}__HR@10"
            ] = float(
                arr["hits10"][sampled].sum()
                / n
            )

            result[
                f"{group_key}__NDCG@10"
            ] = float(
                arr["ndcg_sum"][sampled].sum()
                / n
            )

            result[
                f"{group_key}__mean_rank"
            ] = float(
                arr["rank_sum"][sampled].sum()
                / n
            )

        if not valid:
            continue

        # -------------------------------------------------------------
        # Core pair-novelty contrast among warm/warm entities.
        # -------------------------------------------------------------

        for metric in (
            "HR@10",
            "NDCG@10",
            "mean_rank",
        ):

            result[
                f"pair_novelty_ww__{metric}"
            ] = (
                result[
                    f"new_ww__{metric}"
                ]
                - result[
                    f"prior_ww__{metric}"
                ]
            )

            result[
                f"startup_cold_warm_investor__{metric}"
            ] = (
                result[
                    f"new_wc__{metric}"
                ]
                - result[
                    f"new_ww__{metric}"
                ]
            )

            result[
                f"investor_cold_warm_startup__{metric}"
            ] = (
                result[
                    f"new_cw__{metric}"
                ]
                - result[
                    f"new_ww__{metric}"
                ]
            )

            result[
                f"dual_cold_vs_ww__{metric}"
            ] = (
                result[
                    f"new_cc__{metric}"
                ]
                - result[
                    f"new_ww__{metric}"
                ]
            )

            result[
                f"startup_cold_cold_investor__{metric}"
            ] = (
                result[
                    f"new_cc__{metric}"
                ]
                - result[
                    f"new_cw__{metric}"
                ]
            )

            result[
                f"investor_cold_cold_startup__{metric}"
            ] = (
                result[
                    f"new_cc__{metric}"
                ]
                - result[
                    f"new_wc__{metric}"
                ]
            )

            # Descriptive 2x2 difference-in-differences among NEW pairs:
            #
            # (CC - CW) - (WC - WW)
            #
            # Equivalent:
            # CC - CW - WC + WW
            #
            # This measures departure from additivity on the raw outcome scale.
            result[
                f"cold_start_interaction__{metric}"
            ] = (
                result[
                    f"new_cc__{metric}"
                ]
                - result[
                    f"new_cw__{metric}"
                ]
                - result[
                    f"new_wc__{metric}"
                ]
                + result[
                    f"new_ww__{metric}"
                ]
            )

        rows.append(result)

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================


def main() -> None:

    print("=" * 110)
    print(
        "PHASE 7.4 — NOVELTY × COLD-START "
        "INTERACTION DIAGNOSTIC V1"
    )
    print("=" * 110)

    print(
        "Scientific role:             "
        "STRATIFIED POST-HOC THESIS DIAGNOSTIC"
    )

    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Raw logits loaded:           NO")
    print("Inference executed:          NO")
    print("Test rescored:               NO")
    print("Model selection performed:   NO")

    # =========================================================================
    # 7.4.1 — Integrity gate
    # =========================================================================

    print_section(
        "7.4.1 — ANALYSIS INTEGRITY GATE"
    )

    for path in (
        SOURCE,
        PHASE_7_0_CONTRACT,
        PHASE_7_2_MANIFEST,
        PHASE_7_3_MANIFEST,
    ):

        require(
            path.exists(),
            f"Missing required artifact: {path}",
        )

        print(
            "FOUND  "
            f"{path.relative_to(REPO_ROOT)}"
        )

    source_hash = file_sha256(SOURCE)

    require(
        source_hash == EXPECTED_SOURCE_SHA256,
        "Frozen Phase-6 analysis bundle SHA256 drift.",
    )

    with PHASE_7_0_CONTRACT.open(
        "r",
        encoding="utf-8",
    ) as f:
        phase_7_0 = json.load(f)

    with PHASE_7_2_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        phase_7_2 = json.load(f)

    with PHASE_7_3_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        phase_7_3 = json.load(f)

    require(
        phase_7_0["status"] == "PASS",
        "Phase 7.0 contract is not PASS.",
    )

    require(
        phase_7_2["status"] == "COMPLETE",
        "Phase 7.2 is not COMPLETE.",
    )

    require(
        phase_7_3["status"] == "COMPLETE",
        "Phase 7.3 is not COMPLETE.",
    )

    print()
    print("Frozen source fingerprint:   PASS")
    print("Phase 7.0 prerequisite:       PASS")
    print("Phase 7.2 prerequisite:       PASS")
    print("Phase 7.3 prerequisite:       PASS")

    # =========================================================================
    # 7.4.2 — Construct diagnostic groups
    # =========================================================================

    print_section(
        "7.4.2 — CONSTRUCT NOVELTY × COLD-START DIAGNOSTIC GROUPS"
    )

    df = pd.read_parquet(SOURCE)

    require(
        len(df) == EXPECTED_ROWS,
        "Final-test row-count drift.",
    )

    require(
        df["interaction_id"].is_unique,
        "interaction_id is not unique.",
    )

    df["diagnostic_group"] = (
        build_diagnostic_group(df)
    )

    require(
        ~df["diagnostic_group"]
        .eq("INVALID")
        .any(),
        (
            "Encountered an invalid novelty/"
            "cold-start combination."
        ),
    )

    counts = (
        df["diagnostic_group"]
        .value_counts()
    )

    for group_key in GROUP_ORDER:

        observed = int(
            counts[group_key]
        )

        expected = (
            EXPECTED_COUNTS[group_key]
        )

        require(
            observed == expected,
            (
                f"Count drift for {group_key}: "
                f"expected {expected}, "
                f"observed {observed}."
            ),
        )

        print(
            f"{GROUP_LABELS[group_key]:<48} "
            f"{observed:>7,} "
            f"({observed / len(df):6.2%})"
        )

    require(
        sum(EXPECTED_COUNTS.values())
        == EXPECTED_ROWS,
        "Diagnostic groups do not cover complete test set.",
    )

    print()
    print("Diagnostic partition:         PASS")

    # =========================================================================
    # 7.4.3 — Performance table
    # =========================================================================

    print_section(
        "7.4.3 — PERFORMANCE BY DIAGNOSTIC GROUP"
    )

    metric_rows = []

    for group_key in GROUP_ORDER:

        subgroup = df.loc[
            df["diagnostic_group"]
            == group_key
        ]

        metrics = calculate_metrics(
            subgroup
        )

        metrics["group_key"] = group_key
        metrics["group_label"] = (
            GROUP_LABELS[group_key]
        )

        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(
        metric_rows
    )

    indexed = (
        metrics_df
        .set_index("group_key")
    )

    print(
        metrics_df[
            [
                "group_label",
                "n",
                "HR@1",
                "HR@5",
                "HR@10",
                "NDCG@10",
                "mean_rank",
                "median_rank",
                "P75_rank",
                "rank_gt_50_share",
            ]
        ].to_string(
            index=False,
            formatters={
                "HR@1":
                    lambda x: f"{x:.4f}",
                "HR@5":
                    lambda x: f"{x:.4f}",
                "HR@10":
                    lambda x: f"{x:.4f}",
                "NDCG@10":
                    lambda x: f"{x:.4f}",
                "mean_rank":
                    lambda x: f"{x:.2f}",
                "median_rank":
                    lambda x: f"{x:.1f}",
                "P75_rank":
                    lambda x: f"{x:.1f}",
                "rank_gt_50_share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    # =========================================================================
    # 7.4.4 — Key thesis contrasts
    # =========================================================================

    print_section(
        "7.4.4 — KEY THESIS CONTRASTS"
    )

    comparisons = [
        (
            "pair_novelty_warm_warm",
            "New warm/warm - Prior warm/warm",
            "new_ww",
            "prior_ww",
            "PRIMARY",
        ),
        (
            "startup_cold_warm_investor",
            "New warm-investor/cold-startup - New warm/warm",
            "new_wc",
            "new_ww",
            "PRIMARY",
        ),
        (
            "investor_cold_warm_startup",
            "New cold-investor/warm-startup - New warm/warm",
            "new_cw",
            "new_ww",
            "PRIMARY",
        ),
        (
            "dual_cold_vs_ww",
            "New dual-cold - New warm/warm",
            "new_cc",
            "new_ww",
            "PRIMARY",
        ),
        (
            "startup_cold_cold_investor",
            "New dual-cold - New cold-investor/warm-startup",
            "new_cc",
            "new_cw",
            "SECONDARY",
        ),
        (
            "investor_cold_cold_startup",
            "New dual-cold - New warm-investor/cold-startup",
            "new_cc",
            "new_wc",
            "SECONDARY",
        ),
    ]

    contrast_rows = []

    for (
        comparison_key,
        comparison_label,
        numerator_key,
        reference_key,
        priority,
    ) in comparisons:

        numerator = indexed.loc[
            numerator_key
        ]

        reference = indexed.loc[
            reference_key
        ]

        row = {
            "comparison_key":
                comparison_key,

            "comparison_label":
                comparison_label,

            "priority":
                priority,

            "numerator_group":
                numerator_key,

            "reference_group":
                reference_key,

            "delta_HR@1":
                float(
                    numerator["HR@1"]
                    - reference["HR@1"]
                ),

            "delta_HR@5":
                float(
                    numerator["HR@5"]
                    - reference["HR@5"]
                ),

            "delta_HR@10":
                float(
                    numerator["HR@10"]
                    - reference["HR@10"]
                ),

            "delta_NDCG@10":
                float(
                    numerator["NDCG@10"]
                    - reference["NDCG@10"]
                ),

            "delta_mean_rank":
                float(
                    numerator["mean_rank"]
                    - reference["mean_rank"]
                ),

            "delta_median_rank":
                float(
                    numerator["median_rank"]
                    - reference["median_rank"]
                ),

            "delta_rank_gt_50_share":
                float(
                    numerator[
                        "rank_gt_50_share"
                    ]
                    - reference[
                        "rank_gt_50_share"
                    ]
                ),
        }

        contrast_rows.append(row)

    contrasts_df = pd.DataFrame(
        contrast_rows
    )

    print(
        contrasts_df[
            [
                "priority",
                "comparison_label",
                "delta_HR@10",
                "delta_NDCG@10",
                "delta_mean_rank",
                "delta_median_rank",
                "delta_rank_gt_50_share",
            ]
        ].to_string(
            index=False,
            formatters={
                "delta_HR@10":
                    lambda x: f"{x:+.4f}",
                "delta_NDCG@10":
                    lambda x: f"{x:+.4f}",
                "delta_mean_rank":
                    lambda x: f"{x:+.2f}",
                "delta_median_rank":
                    lambda x: f"{x:+.2f}",
                "delta_rank_gt_50_share":
                    lambda x: f"{x:+.2%}",
            },
        )
    )

    # =========================================================================
    # 7.4.5 — Descriptive 2x2 cold-start interaction among NEW pairs
    # =========================================================================

    print_section(
        "7.4.5 — DESCRIPTIVE COLD-START INTERACTION AMONG NEW PAIRS"
    )

    interaction = {}

    for metric in (
        "HR@10",
        "NDCG@10",
        "mean_rank",
    ):

        value = float(
            indexed.loc[
                "new_cc",
                metric,
            ]
            - indexed.loc[
                "new_cw",
                metric,
            ]
            - indexed.loc[
                "new_wc",
                metric,
            ]
            + indexed.loc[
                "new_ww",
                metric,
            ]
        )

        interaction[
            metric
        ] = value

        print(
            f"{metric:<15} "
            f"{value:+.6f}"
        )

    print()
    print(
        "Definition:"
    )
    print(
        "  (ColdI/ColdS - ColdI/WarmS)"
    )
    print(
        "  - (WarmI/ColdS - WarmI/WarmS)"
    )
    print()
    print(
        "Interpretation: departure from additivity "
        "on the observed outcome scale only."
    )

    # =========================================================================
    # 7.4.6 — Failure concentration
    # =========================================================================

    print_section(
        "7.4.6 — WHERE ARE NEW-PAIR FAILURES CONCENTRATED?"
    )

    new_df = df.loc[
        df["new_to_investor_pair"]
    ].copy()

    require(
        len(new_df) == 16_446,
        "New-to-investor count drift.",
    )

    total_new_top10_failures = int(
        (
            new_df["positive_rank"]
            > 10
        ).sum()
    )

    total_new_severe_failures = int(
        (
            new_df["positive_rank"]
            > 50
        ).sum()
    )

    failure_rows = []

    for group_key in NEW_GROUPS:

        subgroup = new_df.loc[
            new_df["diagnostic_group"]
            == group_key
        ]

        top10_fail_n = int(
            (
                subgroup["positive_rank"]
                > 10
            ).sum()
        )

        severe50_n = int(
            (
                subgroup["positive_rank"]
                > 50
            ).sum()
        )

        failure_rows.append(
            {
                "group_key":
                    group_key,

                "group_label":
                    GROUP_LABELS[
                        group_key
                    ],

                "n":
                    len(subgroup),

                "top10_failure_n":
                    top10_fail_n,

                "top10_failure_rate":
                    float(
                        top10_fail_n
                        / len(subgroup)
                    ),

                "share_of_all_new_top10_failures":
                    float(
                        top10_fail_n
                        / total_new_top10_failures
                    ),

                "rank_gt_50_n":
                    severe50_n,

                "rank_gt_50_rate":
                    float(
                        severe50_n
                        / len(subgroup)
                    ),

                "share_of_all_new_rank_gt_50_failures":
                    float(
                        severe50_n
                        / total_new_severe_failures
                    ),
            }
        )

    failures_df = pd.DataFrame(
        failure_rows
    )

    startup_cold_severe_n = int(
        failures_df.loc[
            failures_df[
                "group_key"
            ].isin(
                [
                    "new_wc",
                    "new_cc",
                ]
            ),
            "rank_gt_50_n",
        ].sum()
    )

    startup_cold_top10_fail_n = int(
        failures_df.loc[
            failures_df[
                "group_key"
            ].isin(
                [
                    "new_wc",
                    "new_cc",
                ]
            ),
            "top10_failure_n",
        ].sum()
    )

    startup_cold_severe_share = (
        startup_cold_severe_n
        / total_new_severe_failures
    )

    startup_cold_top10_fail_share = (
        startup_cold_top10_fail_n
        / total_new_top10_failures
    )

    print(
        failures_df[
            [
                "group_label",
                "top10_failure_n",
                "top10_failure_rate",
                "share_of_all_new_top10_failures",
                "rank_gt_50_n",
                "rank_gt_50_rate",
                "share_of_all_new_rank_gt_50_failures",
            ]
        ].to_string(
            index=False,
            formatters={
                "top10_failure_rate":
                    lambda x: f"{x:.2%}",
                "share_of_all_new_top10_failures":
                    lambda x: f"{x:.2%}",
                "rank_gt_50_rate":
                    lambda x: f"{x:.2%}",
                "share_of_all_new_rank_gt_50_failures":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    print()
    print(
        f"All new-pair Top-10 failures:     "
        f"{total_new_top10_failures:,}"
    )

    print(
        f"All new-pair rank>50 failures:    "
        f"{total_new_severe_failures:,}"
    )

    print()
    print(
        "Cold-start startup contribution:"
    )

    print(
        f"  Top-10 failures:                "
        f"{startup_cold_top10_fail_n:,} "
        f"({startup_cold_top10_fail_share:.2%})"
    )

    print(
        f"  Rank>50 failures:               "
        f"{startup_cold_severe_n:,} "
        f"({startup_cold_severe_share:.2%})"
    )

    # =========================================================================
    # 7.4.7 — Rank-band decomposition
    # =========================================================================

    print_section(
        "7.4.7 — RANK-BAND DECOMPOSITION"
    )

    df["rank_band"] = pd.cut(
        df["positive_rank"],
        bins=[
            0,
            1,
            5,
            10,
            20,
            50,
            100,
        ],
        labels=RANK_LABELS,
        include_lowest=True,
        ordered=True,
    )

    rank_bands = (
        df.groupby(
            [
                "diagnostic_group",
                "rank_band",
            ],
            observed=False,
        )
        .size()
        .rename("n")
        .reset_index()
    )

    group_sizes = (
        df["diagnostic_group"]
        .value_counts()
    )

    rank_bands[
        "within_group_share"
    ] = (
        rank_bands["n"]
        / rank_bands[
            "diagnostic_group"
        ].map(
            group_sizes
        )
    )

    # =========================================================================
    # 7.4.8 — HR@k curves
    # =========================================================================

    print_section(
        "7.4.8 — CUMULATIVE HR@K CURVES"
    )

    cdf_rows = []

    for group_key in GROUP_ORDER:

        ranks = (
            df.loc[
                df["diagnostic_group"]
                == group_key,
                "positive_rank",
            ]
            .astype(int)
        )

        for k in range(1, 101):

            cdf_rows.append(
                {
                    "group_key":
                        group_key,

                    "group_label":
                        GROUP_LABELS[
                            group_key
                        ],

                    "k":
                        k,

                    "HR@k":
                        float(
                            (ranks <= k)
                            .mean()
                        ),
                }
            )

    cdf_df = pd.DataFrame(
        cdf_rows
    )

    for cutoff in (
        1,
        5,
        10,
        20,
        50,
    ):

        print()
        print(
            f"Cutoff k={cutoff}"
        )

        subset = cdf_df.loc[
            cdf_df["k"] == cutoff
        ]

        for group_key in GROUP_ORDER:

            value = float(
                subset.loc[
                    subset["group_key"]
                    == group_key,
                    "HR@k",
                ].iloc[0]
            )

            print(
                f"  {GROUP_LABELS[group_key]:<48} "
                f"{value:.2%}"
            )

    # =========================================================================
    # 7.4.9 — Investor-cluster bootstrap
    # =========================================================================

    print_section(
        "7.4.9 — INVESTOR-CLUSTER BOOTSTRAP"
    )

    print(
        f"Bootstrap replicates:         "
        f"{BOOTSTRAP_REPS:,}"
    )

    print(
        f"Bootstrap seed:               "
        f"{BOOTSTRAP_SEED}"
    )

    print(
        f"Investor clusters:            "
        f"{df['investor_id'].nunique():,}"
    )

    bootstrap_df = (
        investor_cluster_bootstrap(
            df=df,
            reps=BOOTSTRAP_REPS,
            seed=BOOTSTRAP_SEED,
        )
    )

    require(
        len(bootstrap_df)
        == BOOTSTRAP_REPS,
        (
            "Unexpected number of valid "
            "bootstrap replicates."
        ),
    )

    bootstrap_rows = []

    bootstrap_comparison_map = {
        "pair_novelty_warm_warm":
            "pair_novelty_ww",

        "startup_cold_warm_investor":
            "startup_cold_warm_investor",

        "investor_cold_warm_startup":
            "investor_cold_warm_startup",

        "dual_cold_vs_ww":
            "dual_cold_vs_ww",

        "startup_cold_cold_investor":
            "startup_cold_cold_investor",

        "investor_cold_cold_startup":
            "investor_cold_cold_startup",
    }

    contrasts_lookup = (
        contrasts_df
        .set_index(
            "comparison_key"
        )
    )

    for (
        comparison_key,
        bootstrap_prefix,
    ) in bootstrap_comparison_map.items():

        for metric in (
            "HR@10",
            "NDCG@10",
            "mean_rank",
        ):

            column = (
                f"{bootstrap_prefix}"
                f"__{metric}"
            )

            low, high = percentile_ci(
                bootstrap_df[column]
            )

            estimate_column = {
                "HR@10":
                    "delta_HR@10",
                "NDCG@10":
                    "delta_NDCG@10",
                "mean_rank":
                    "delta_mean_rank",
            }[metric]

            bootstrap_rows.append(
                {
                    "comparison_key":
                        comparison_key,

                    "comparison_label":
                        contrasts_lookup.loc[
                            comparison_key,
                            "comparison_label",
                        ],

                    "metric":
                        metric,

                    "estimate":
                        float(
                            contrasts_lookup.loc[
                                comparison_key,
                                estimate_column,
                            ]
                        ),

                    "ci95_low":
                        low,

                    "ci95_high":
                        high,

                    "bootstrap_unit":
                        "investor_id",

                    "repetitions":
                        BOOTSTRAP_REPS,
                }
            )

    # Interaction uncertainty.
    for metric in (
        "HR@10",
        "NDCG@10",
        "mean_rank",
    ):

        column = (
            f"cold_start_interaction"
            f"__{metric}"
        )

        low, high = percentile_ci(
            bootstrap_df[column]
        )

        bootstrap_rows.append(
            {
                "comparison_key":
                    "cold_start_interaction",

                "comparison_label":
                    (
                        "New-pair investor × startup "
                        "cold-start interaction"
                    ),

                "metric":
                    metric,

                "estimate":
                    interaction[metric],

                "ci95_low":
                    low,

                "ci95_high":
                    high,

                "bootstrap_unit":
                    "investor_id",

                "repetitions":
                    BOOTSTRAP_REPS,
            }
        )

    bootstrap_summary = pd.DataFrame(
        bootstrap_rows
    )

    primary_hr = (
        bootstrap_summary.loc[
            (
                bootstrap_summary["metric"]
                == "HR@10"
            )
            &
            (
                bootstrap_summary[
                    "comparison_key"
                ].isin(
                    [
                        "pair_novelty_warm_warm",
                        "startup_cold_warm_investor",
                        "investor_cold_warm_startup",
                        "dual_cold_vs_ww",
                        "cold_start_interaction",
                    ]
                )
            )
        ]
    )

    print(
        primary_hr[
            [
                "comparison_key",
                "estimate",
                "ci95_low",
                "ci95_high",
            ]
        ].to_string(
            index=False,
            formatters={
                "estimate":
                    lambda x: f"{x:+.6f}",
                "ci95_low":
                    lambda x: f"{x:+.6f}",
                "ci95_high":
                    lambda x: f"{x:+.6f}",
            },
        )
    )

    # =========================================================================
    # 7.4.10 — Evidence interpretation structure
    # =========================================================================

    print_section(
        "7.4.10 — THESIS-EVIDENCE INTERPRETATION"
    )

    pair_novelty_row = (
        contrasts_lookup.loc[
            "pair_novelty_warm_warm"
        ]
    )

    startup_cold_row = (
        contrasts_lookup.loc[
            "startup_cold_warm_investor"
        ]
    )

    investor_cold_row = (
        contrasts_lookup.loc[
            "investor_cold_warm_startup"
        ]
    )

    dual_cold_row = (
        contrasts_lookup.loc[
            "dual_cold_vs_ww"
        ]
    )

    print(
        "A) ENTITY-WARM PAIR NOVELTY"
    )

    print(
        "   New warm/warm - prior warm/warm:"
    )

    print(
        f"   ΔHR@10 = "
        f"{pair_novelty_row['delta_HR@10']:+.2%}"
    )

    print(
        f"   ΔNDCG@10 = "
        f"{pair_novelty_row['delta_NDCG@10']:+.4f}"
    )

    print(
        f"   Δmean rank = "
        f"{pair_novelty_row['delta_mean_rank']:+.2f}"
    )

    print()

    print(
        "B) COLD START WITHIN NEW-PAIR DISCOVERY"
    )

    print(
        f"   Startup cold, investor warm: "
        f"ΔHR@10 "
        f"{startup_cold_row['delta_HR@10']:+.2%}"
    )

    print(
        f"   Investor cold, startup warm: "
        f"ΔHR@10 "
        f"{investor_cold_row['delta_HR@10']:+.2%}"
    )

    print(
        f"   Dual cold:                  "
        f"ΔHR@10 "
        f"{dual_cold_row['delta_HR@10']:+.2%}"
    )

    print()

    print(
        "C) FAILURE CONCENTRATION"
    )

    print(
        f"   Cold-start startups account for "
        f"{startup_cold_top10_fail_share:.2%} "
        f"of new-pair Top-10 failures."
    )

    print(
        f"   Cold-start startups account for "
        f"{startup_cold_severe_share:.2%} "
        f"of new-pair rank>50 failures."
    )

    print()

    print(
        "INTERPRETATION BOUNDARY:"
    )

    print(
        "   These are stratified observational "
        "diagnostics of the frozen test."
    )

    print(
        "   They support identification of difficult "
        "evaluation regimes, not causal mechanisms."
    )

    # =========================================================================
    # 7.4.11 — Write artifacts
    # =========================================================================

    print_section(
        "7.4.11 — WRITE ANALYSIS ARTIFACTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_path = (
        OUT_DIR
        / "phase_7_4_diagnostic_group_metrics_V1.csv"
    )

    contrasts_path = (
        OUT_DIR
        / "phase_7_4_thesis_key_contrasts_V1.csv"
    )

    interaction_path = (
        OUT_DIR
        / "phase_7_4_cold_start_interaction_V1.json"
    )

    failures_path = (
        OUT_DIR
        / "phase_7_4_failure_concentration_V1.csv"
    )

    rank_bands_path = (
        OUT_DIR
        / "phase_7_4_rank_bands_V1.csv"
    )

    cdf_path = (
        OUT_DIR
        / "phase_7_4_rank_cdf_V1.csv"
    )

    bootstrap_reps_path = (
        OUT_DIR
        / "phase_7_4_investor_cluster_bootstrap_replicates_V1.csv"
    )

    bootstrap_summary_path = (
        OUT_DIR
        / "phase_7_4_investor_cluster_bootstrap_summary_V1.csv"
    )

    insights_path = (
        OUT_DIR
        / "phase_7_4_thesis_evidence_V1.json"
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    contrasts_df.to_csv(
        contrasts_path,
        index=False,
    )

    json_dump(
        interaction,
        interaction_path,
    )

    failures_df.to_csv(
        failures_path,
        index=False,
    )

    rank_bands.to_csv(
        rank_bands_path,
        index=False,
    )

    cdf_df.to_csv(
        cdf_path,
        index=False,
    )

    bootstrap_df.to_csv(
        bootstrap_reps_path,
        index=False,
    )

    bootstrap_summary.to_csv(
        bootstrap_summary_path,
        index=False,
    )

    thesis_evidence = {
        "phase":
            "7.4",

        "scientific_scope":
            (
                "Stratified post-hoc diagnostic of "
                "pair novelty and entity cold start."
            ),

        "diagnostic_groups":
            EXPECTED_COUNTS,

        "primary_questions": {
            "pair_novelty_among_warm_entities":
                (
                    "new_ww versus prior_ww"
                ),

            "cold_start_within_new_pair_discovery":
                (
                    "new_wc, new_cw and new_cc "
                    "versus new_ww"
                ),
        },

        "failure_concentration": {
            "new_pair_top10_failures":
                total_new_top10_failures,

            "new_pair_rank_gt_50_failures":
                total_new_severe_failures,

            "cold_startup_top10_failure_n":
                startup_cold_top10_fail_n,

            "cold_startup_top10_failure_share":
                startup_cold_top10_fail_share,

            "cold_startup_rank_gt_50_n":
                startup_cold_severe_n,

            "cold_startup_rank_gt_50_share":
                startup_cold_severe_share,
        },

        "cold_start_interaction":
            interaction,

        "uncertainty": {
            "method":
                (
                    "Percentile bootstrap with "
                    "investor_id as resampling unit."
                ),

            "repetitions":
                BOOTSTRAP_REPS,

            "seed":
                BOOTSTRAP_SEED,

            "limitation":
                (
                    "Startup clustering and multiway "
                    "dependence are deferred to "
                    "Phase 7 robustness analysis."
                ),
        },

        "evidence_boundary": {
            "descriptive":
                (
                    "Observed subgroup performance "
                    "and failure concentration."
                ),

            "statistical":
                (
                    "Investor-cluster bootstrap "
                    "intervals for pre-specified "
                    "contrasts."
                ),

            "interpretation":
                (
                    "Performance differences identify "
                    "difficult recommendation regimes."
                ),

            "not_established":
                (
                    "Causal mechanism or proof that a "
                    "specific inductive architecture "
                    "will improve performance."
                ),
        },
    }

    json_dump(
        thesis_evidence,
        insights_path,
    )

    # =========================================================================
    # 7.4.12 — Figure 1: HR@10 across diagnostic groups
    # =========================================================================

    print_section(
        "7.4.12 — GENERATE PRESENTATION-READY FIGURES"
    )

    figure_paths = []

    plot_metrics = (
        metrics_df
        .set_index("group_key")
        .loc[GROUP_ORDER]
    )

    display_short = [
        "Prior\nwarm/warm",
        "New\nwarm/warm",
        "New\nwarm investor\ncold startup",
        "New\ncold investor\nwarm startup",
        "New\ndual cold",
    ]

    values = (
        plot_metrics["HR@10"]
        .to_numpy()
        * 100.0
    )

    fig, ax = plt.subplots(
        figsize=(12.5, 7)
    )

    bars = ax.bar(
        np.arange(
            len(GROUP_ORDER)
        ),
        values,
    )

    ax.set_xticks(
        np.arange(
            len(GROUP_ORDER)
        ),
        display_short,
    )

    ax.set_ylabel(
        "HR@10 (%)"
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.set_title(
        "ITRS Top-10 retrieval across novelty and cold-start regimes"
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    for bar, value in zip(
        bars,
        values,
    ):

        ax.annotate(
            f"{value:.1f}%",
            (
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_4_fig1_"
                "HR10_novelty_cold_start_V1"
            ),
        )
    )

    # =========================================================================
    # Figure 2: HR@k curves
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    for group_key in GROUP_ORDER:

        subset = cdf_df.loc[
            cdf_df["group_key"]
            == group_key
        ]

        ax.plot(
            subset["k"],
            subset["HR@k"]
            * 100.0,
            linewidth=2.0,
            label=GROUP_LABELS[
                group_key
            ],
        )

    ax.axvline(
        10,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlim(
        1,
        100,
    )

    ax.set_ylim(
        0,
        101,
    )

    ax.set_xlabel(
        "Recommendation cutoff k"
    )

    ax.set_ylabel(
        "Cumulative hit rate (%)"
    )

    ax.set_title(
        "Positive-startup retrieval by novelty × cold-start regime"
    )

    ax.legend(
        fontsize=8,
    )

    ax.grid(
        alpha=0.20,
    )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_4_fig2_"
                "rank_cdf_novelty_cold_start_V1"
            ),
        )
    )

    # =========================================================================
    # Figure 3: primary HR@10 contrast forest plot
    # =========================================================================

    forest_keys = [
        "pair_novelty_warm_warm",
        "startup_cold_warm_investor",
        "investor_cold_warm_startup",
        "dual_cold_vs_ww",
    ]

    forest_labels = [
        "New vs prior\n(warm/warm only)",
        "Startup cold\n(new, investor warm)",
        "Investor cold\n(new, startup warm)",
        "Dual cold\n(new vs new warm/warm)",
    ]

    forest = (
        bootstrap_summary.loc[
            (
                bootstrap_summary["metric"]
                == "HR@10"
            )
            &
            (
                bootstrap_summary[
                    "comparison_key"
                ].isin(
                    forest_keys
                )
            )
        ]
        .set_index(
            "comparison_key"
        )
        .loc[
            forest_keys
        ]
    )

    estimates = (
        forest["estimate"]
        .to_numpy()
        * 100.0
    )

    lows = (
        forest["ci95_low"]
        .to_numpy()
        * 100.0
    )

    highs = (
        forest["ci95_high"]
        .to_numpy()
        * 100.0
    )

    y = np.arange(
        len(forest_keys)
    )

    fig, ax = plt.subplots(
        figsize=(10.5, 6.5)
    )

    ax.errorbar(
        estimates,
        y,
        xerr=[
            estimates - lows,
            highs - estimates,
        ],
        fmt="o",
        capsize=4,
    )

    ax.axvline(
        0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_yticks(
        y,
        forest_labels,
    )

    ax.invert_yaxis()

    ax.set_xlabel(
        "Difference in HR@10 (percentage points)"
    )

    ax.set_title(
        "Pre-specified thesis contrasts with investor-cluster 95% intervals"
    )

    ax.grid(
        axis="x",
        alpha=0.20,
    )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_4_fig3_"
                "HR10_primary_contrast_forest_V1"
            ),
        )
    )

    # =========================================================================
    # Figure 4: severe-failure contribution
    # =========================================================================

    failure_plot = (
        failures_df
        .set_index("group_key")
        .loc[
            NEW_GROUPS
        ]
    )

    severe_share = (
        failure_plot[
            "share_of_all_new_rank_gt_50_failures"
        ]
        .to_numpy()
        * 100.0
    )

    failure_labels = [
        "New\nwarm/warm",
        "New\nwarm investor\ncold startup",
        "New\ncold investor\nwarm startup",
        "New\ndual cold",
    ]

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    bars = ax.bar(
        np.arange(
            len(NEW_GROUPS)
        ),
        severe_share,
    )

    ax.set_xticks(
        np.arange(
            len(NEW_GROUPS)
        ),
        failure_labels,
    )

    ax.set_ylabel(
        "Share of all new-pair rank>50 failures (%)"
    )

    ax.set_title(
        "Which regimes generate the severe new-to-investor failure tail?"
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    for bar, value in zip(
        bars,
        severe_share,
    ):

        ax.annotate(
            f"{value:.1f}%",
            (
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
        )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_4_fig4_"
                "severe_failure_contribution_V1"
            ),
        )
    )

    # =========================================================================
    # 7.4.13 — Provenance manifest
    # =========================================================================

    manifest_path = (
        OUT_DIR
        / "phase_7_4_analysis_manifest_V1.json"
    )

    manifest = {
        "phase":
            "7.4",

        "schema_version":
            "ITRS_PHASE7_4_NOVELTY_COLD_START_INTERACTION_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            (
                "Stratified post-hoc thesis diagnostic "
                "of pair novelty and cold start."
            ),

        "source": {
            "path":
                str(
                    SOURCE.relative_to(
                        REPO_ROOT
                    )
                ),

            "sha256":
                source_hash,

            "rows":
                len(df),
        },

        "diagnostic_groups":
            GROUP_LABELS,

        "group_counts":
            EXPECTED_COUNTS,

        "primary_comparisons": [
            "pair_novelty_warm_warm",
            "startup_cold_warm_investor",
            "investor_cold_warm_startup",
            "dual_cold_vs_ww",
        ],

        "uncertainty_method": {
            "method":
                (
                    "One-way investor-cluster "
                    "percentile bootstrap"
                ),

            "repetitions":
                BOOTSTRAP_REPS,

            "seed":
                BOOTSTRAP_SEED,
        },

        "interpretation_boundary":
            (
                "Stratification reduces confounding "
                "between pair novelty and cold-start "
                "status but does not establish "
                "causal effects."
            ),

        "tables": [
            str(
                metrics_path.relative_to(
                    REPO_ROOT
                )
            ),

            str(
                contrasts_path.relative_to(
                    REPO_ROOT
                )
            ),

            str(
                failures_path.relative_to(
                    REPO_ROOT
                )
            ),

            str(
                rank_bands_path.relative_to(
                    REPO_ROOT
                )
            ),

            str(
                cdf_path.relative_to(
                    REPO_ROOT
                )
            ),

            str(
                bootstrap_summary_path.relative_to(
                    REPO_ROOT
                )
            ),
        ],

        "figures": [
            str(
                path.relative_to(
                    REPO_ROOT
                )
            )
            for path in figure_paths
        ],

        "prohibited_operations": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "raw_logits_loaded": False,
            "inference_executed": False,
            "test_rescored": False,
            "negative_resampling": False,
            "candidate_modification": False,
            "model_selection": False,
        },
    }

    json_dump(
        manifest,
        manifest_path,
    )

    # =========================================================================
    # Hash derived artifacts
    # =========================================================================

    artifact_paths = [
        metrics_path,
        contrasts_path,
        interaction_path,
        failures_path,
        rank_bands_path,
        cdf_path,
        bootstrap_reps_path,
        bootstrap_summary_path,
        insights_path,
        *figure_paths,
        manifest_path,
    ]

    hashes = {}

    for path in artifact_paths:

        relative = str(
            path.relative_to(
                REPO_ROOT
            )
        )

        hashes[relative] = (
            file_sha256(path)
        )

        print(
            f"WROTE  {relative}"
        )

    hashes_path = (
        OUT_DIR
        / "phase_7_4_derived_artifact_sha256_V1.json"
    )

    json_dump(
        hashes,
        hashes_path,
    )

    print(
        "WROTE  "
        f"{hashes_path.relative_to(REPO_ROOT)}"
    )

    # =========================================================================
    # Final status
    # =========================================================================

    print_section(
        "PHASE 7.4 V1 RESULT"
    )

    print("Frozen source binding:                 PASS")
    print("Diagnostic partition:                  PASS")
    print("Warm/warm novelty comparison:          COMPLETE")
    print("New-pair cold-start comparison:        COMPLETE")
    print("Failure-concentration analysis:        COMPLETE")
    print("Cold-start interaction diagnostic:     COMPLETE")
    print("Investor-cluster uncertainty:          COMPLETE")
    print("Presentation-ready figures:            COMPLETE")
    print("Thesis-evidence artifact:              COMPLETE")
    print("Provenance manifest:                   COMPLETE")

    print()
    print("Model inference executed:              NO")
    print("Final test rescored:                   NO")
    print("Final test modified:                   NO")

    print()
    print(
        "PHASE 7.4 NOVELTY × COLD-START "
        "INTERACTION STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()