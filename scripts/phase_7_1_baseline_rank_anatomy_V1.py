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
# Phase 7.1 — Baseline Rank Anatomy V1
#
# Scientific role:
#   Descriptive post-hoc analysis of the immutable Phase-6 one-shot final test.
#
# This script:
#   - does NOT load the model
#   - does NOT load checkpoints
#   - does NOT load raw logits
#   - does NOT perform inference
#   - does NOT rescore test cases
#   - does NOT modify candidate sets
#   - does NOT perform model selection
#
# It analyzes only the already-frozen positive ranks and metrics contained in:
#
#   analysis_ready_test_cases.parquet
# =============================================================================


REPO_ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    REPO_ROOT
    / "data/experimental/phase_6/full_training/100pct/final_test/"
    / "analysis_ready_test_cases.parquet"
)

SOURCE_MANIFEST = (
    REPO_ROOT
    / "data/experimental/phase_6/full_training/100pct/final_test/"
    / "analysis_ready_test_cases_manifest.json"
)

PHASE_7_0_CONTRACT = (
    REPO_ROOT
    / "data/experimental/phase_7/phase_7_0_analysis_contract/"
    / "phase_7_0_analysis_contract_V2.json"
)

OUT_DIR = (
    REPO_ROOT
    / "data/experimental/phase_7/phase_7_1_baseline_rank_anatomy"
)

FIG_DIR = OUT_DIR / "figures"

EXPECTED_SOURCE_SHA256 = (
    "d70f21bff0006e094c5d567d307c0664"
    "b41a11e7250a0e69aaec610e1810132d"
)

EXPECTED_ROWS = 20_264
EXPECTED_HITS_AT_10 = 11_072
EXPECTED_HR10 = 0.5463876825898144
EXPECTED_NDCG10 = 0.39870973012365235

FLOAT_TOL = 5e-13


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

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

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


def save_figure(
    fig,
    stem: str,
) -> list[Path]:

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
            "Author": "Phase 7 ITRS reproduction analysis",
        },
    )

    plt.close(fig)

    return [png, pdf]


# =============================================================================
# Main
# =============================================================================


def main() -> None:

    print("=" * 110)
    print(
        "PHASE 7.1 — BASELINE RANK ANATOMY V1"
    )
    print("=" * 110)

    print(
        "Scientific role:             "
        "DESCRIPTIVE POST-HOC ANALYSIS"
    )
    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Raw logits loaded:           NO")
    print("Inference executed:          NO")
    print("Test rescored:               NO")
    print("Model selection performed:   NO")

    # =========================================================================
    # 7.1.1 — Integrity gate
    # =========================================================================

    print_section(
        "7.1.1 — PHASE-7 ANALYSIS INTEGRITY GATE"
    )

    for path in (
        SOURCE,
        SOURCE_MANIFEST,
        PHASE_7_0_CONTRACT,
    ):

        require(
            path.exists(),
            f"Missing required source: {path}",
        )

        print(
            f"FOUND  {path.relative_to(REPO_ROOT)}"
        )

    source_hash = file_sha256(SOURCE)

    require(
        source_hash == EXPECTED_SOURCE_SHA256,
        "Frozen Phase-6.9d bundle SHA256 drift.",
    )

    with SOURCE_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as f:
        source_manifest = json.load(f)

    with PHASE_7_0_CONTRACT.open(
        "r",
        encoding="utf-8",
    ) as f:
        phase_7_0_contract = json.load(f)

    require(
        source_manifest["output"]["sha256"]
        == EXPECTED_SOURCE_SHA256,
        "Source-manifest fingerprint drift.",
    )

    require(
        phase_7_0_contract["status"] == "PASS",
        "Phase 7.0 contract is not PASS.",
    )

    require(
        phase_7_0_contract[
            "phase_7_rules"
        ]["test_rescoring_allowed"]
        is False,
        (
            "Phase 7.0 contract does not "
            "forbid test rescoring."
        ),
    )

    print()
    print("Frozen source fingerprint:   PASS")
    print("Phase 7.0 contract:           PASS")

    # =========================================================================
    # 7.1.2 — Load ranks
    # =========================================================================

    print_section(
        "7.1.2 — LOAD IMMUTABLE FINAL-TEST RANKS"
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

    require(
        df["positive_rank"].notna().all(),
        "positive_rank contains missing values.",
    )

    require(
        df["positive_rank"]
        .between(1, 100)
        .all(),
        "positive_rank outside 1..100.",
    )

    rank = df["positive_rank"].astype(int)

    print(f"Test cases:                   {len(df):,}")
    print(
        f"Observed rank range:          "
        f"{rank.min()}..{rank.max()}"
    )

    # =========================================================================
    # 7.1.3 — Metric reconstruction
    # =========================================================================

    print_section(
        "7.1.3 — BASELINE RANK METRICS"
    )

    hr1 = float(
        (rank <= 1).mean()
    )

    hr5 = float(
        (rank <= 5).mean()
    )

    hr10 = float(
        (rank <= 10).mean()
    )

    hits1 = int(
        (rank <= 1).sum()
    )

    hits5 = int(
        (rank <= 5).sum()
    )

    hits10 = int(
        (rank <= 10).sum()
    )

    ndcg10 = float(
        df["NDCG@10"].mean()
    )

    mean_rank = float(
        rank.mean()
    )

    median_rank = float(
        rank.median()
    )

    std_rank = float(
        rank.std(ddof=1)
    )

    require(
        hits10 == EXPECTED_HITS_AT_10,
        "Frozen Hits@10 drift.",
    )

    require(
        abs(
            hr10 - EXPECTED_HR10
        )
        <= FLOAT_TOL,
        "Frozen HR@10 drift.",
    )

    require(
        abs(
            ndcg10 - EXPECTED_NDCG10
        )
        <= FLOAT_TOL,
        "Frozen NDCG@10 drift.",
    )

    print(
        f"Hits@1:                       "
        f"{hits1:,}"
    )
    print(
        f"HR@1:                         "
        f"{hr1:.6f}  ({hr1:.2%})"
    )

    print()
    print(
        f"Hits@5:                       "
        f"{hits5:,}"
    )
    print(
        f"HR@5:                         "
        f"{hr5:.6f}  ({hr5:.2%})"
    )

    print()
    print(
        f"Hits@10:                      "
        f"{hits10:,}"
    )
    print(
        f"HR@10:                        "
        f"{hr10:.6f}  ({hr10:.2%})"
    )

    print()
    print(
        f"NDCG@10:                      "
        f"{ndcg10:.6f}"
    )

    print()
    print(
        f"Mean positive rank:           "
        f"{mean_rank:.6f}"
    )
    print(
        f"Median positive rank:         "
        f"{median_rank:.0f}"
    )
    print(
        f"Rank standard deviation:      "
        f"{std_rank:.6f}"
    )

    # =========================================================================
    # 7.1.4 — Rank quantiles
    # =========================================================================

    print_section(
        "7.1.4 — POSITIVE-RANK QUANTILES"
    )

    quantile_levels = [
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
    ]

    quantiles = {
        f"P{int(q * 100):02d}":
            float(rank.quantile(q))
        for q in quantile_levels
    }

    for label, value in quantiles.items():

        print(
            f"{label:<10} "
            f"{value:>10.2f}"
        )

    # =========================================================================
    # 7.1.5 — Long-tail diagnostics
    # =========================================================================

    print_section(
        "7.1.5 — LONG-TAIL FAILURE DIAGNOSTICS"
    )

    tail_thresholds = [
        10,
        20,
        50,
        75,
    ]

    tail_metrics = {}

    for threshold in tail_thresholds:

        count = int(
            (rank > threshold).sum()
        )

        share = float(
            (rank > threshold).mean()
        )

        key = f"rank_gt_{threshold}"

        tail_metrics[key] = {
            "n": count,
            "share": share,
        }

        print(
            f"Rank > {threshold:<3} "
            f"{count:>7,}   "
            f"({share:>7.2%})"
        )

    # Incremental Top-K gains.
    rank_1_only = int(
        (rank == 1).sum()
    )

    ranks_2_5 = int(
        rank.between(
            2,
            5,
        ).sum()
    )

    ranks_6_10 = int(
        rank.between(
            6,
            10,
        ).sum()
    )

    print()
    print("TOP-10 COMPOSITION")
    print("-" * 50)

    print(
        f"Rank 1:                       "
        f"{rank_1_only:>7,} "
        f"({rank_1_only / len(df):.2%})"
    )

    print(
        f"Ranks 2–5:                    "
        f"{ranks_2_5:>7,} "
        f"({ranks_2_5 / len(df):.2%})"
    )

    print(
        f"Ranks 6–10:                   "
        f"{ranks_6_10:>7,} "
        f"({ranks_6_10 / len(df):.2%})"
    )

    # =========================================================================
    # 7.1.6 — Exact rank-frequency table
    # =========================================================================

    print_section(
        "7.1.6 — EXACT RANK FREQUENCY"
    )

    rank_frequency = (
        rank.value_counts()
        .reindex(
            range(1, 101),
            fill_value=0,
        )
        .rename_axis("positive_rank")
        .reset_index(name="n")
    )

    rank_frequency["share"] = (
        rank_frequency["n"]
        / len(df)
    )

    rank_frequency[
        "cumulative_n"
    ] = rank_frequency["n"].cumsum()

    rank_frequency[
        "cumulative_share"
    ] = rank_frequency[
        "cumulative_n"
    ] / len(df)

    print(
        rank_frequency.head(15)
        .to_string(index=False)
    )

    print()
    print(
        "... exact ranks 1–100 will be "
        "written to CSV."
    )

    # =========================================================================
    # 7.1.7 — Top-K cumulative curve
    # =========================================================================

    print_section(
        "7.1.7 — CUMULATIVE TOP-K BEHAVIOR"
    )

    topk_rows = []

    for k in range(1, 101):

        hit_count = int(
            (rank <= k).sum()
        )

        hit_rate = float(
            hit_count / len(df)
        )

        topk_rows.append(
            {
                "k": k,
                "hits_at_k": hit_count,
                "HR@k": hit_rate,
            }
        )

    topk = pd.DataFrame(
        topk_rows
    )

    for k in (
        1,
        3,
        5,
        10,
        20,
        50,
        75,
        100,
    ):

        row = topk.loc[
            topk["k"] == k
        ].iloc[0]

        print(
            f"HR@{k:<3} "
            f"{row['HR@k']:.6f}  "
            f"({row['HR@k']:.2%})"
        )

    # =========================================================================
    # 7.1.8 — Rank-band decomposition
    # =========================================================================

    print_section(
        "7.1.8 — RANK-BAND DECOMPOSITION"
    )

    rank_band = pd.cut(
        rank,
        bins=[
            0,
            1,
            5,
            10,
            20,
            50,
            100,
        ],
        labels=[
            "Rank 1",
            "Ranks 2–5",
            "Ranks 6–10",
            "Ranks 11–20",
            "Ranks 21–50",
            "Ranks 51–100",
        ],
        include_lowest=True,
        right=True,
        ordered=True,
    )

    rank_bands = (
        rank_band
        .value_counts(sort=False)
        .rename_axis("rank_band")
        .reset_index(name="n")
    )

    rank_bands["share"] = (
        rank_bands["n"]
        / len(df)
    )

    print(
        rank_bands.to_string(
            index=False,
            formatters={
                "share":
                    lambda x: f"{x:.2%}",
            },
        )
    )

    # =========================================================================
    # 7.1.9 — Descriptive insights
    # =========================================================================

    print_section(
        "7.1.9 — DESCRIPTIVE INSIGHTS"
    )

    top1_share_of_top10 = (
        hits1 / hits10
        if hits10 > 0
        else np.nan
    )

    top5_share_of_top10 = (
        hits5 / hits10
        if hits10 > 0
        else np.nan
    )

    median_top10 = (
        median_rank <= 10
    )

    mean_median_gap = (
        mean_rank - median_rank
    )

    descriptive_insights = {

        "scope": (
            "Descriptive evidence only. "
            "No causal interpretation and "
            "no subgroup inference."
        ),

        "top_k": {
            "HR@1": hr1,
            "HR@5": hr5,
            "HR@10": hr10,
            (
                "top1_share_of_all_top10_hits"
            ): top1_share_of_top10,
            (
                "top5_share_of_all_top10_hits"
            ): top5_share_of_top10,
        },

        "distribution_shape": {
            "mean_rank": mean_rank,
            "median_rank": median_rank,
            "mean_minus_median": (
                mean_median_gap
            ),
            "median_is_within_top10": (
                bool(median_top10)
            ),
            "interpretive_note": (
                "If mean rank is substantially "
                "larger than median rank, the "
                "empirical distribution has a "
                "high-rank tail. This describes "
                "the observed distribution and "
                "does not identify its cause."
            ),
        },

        "tail": tail_metrics,

        "quantiles": quantiles,
    }

    print(
        f"Median case within Top-10:    "
        f"{median_top10}"
    )

    print(
        f"Mean - median rank gap:       "
        f"{mean_median_gap:.2f}"
    )

    print(
        f"Top-1 share of Top-10 hits:   "
        f"{top1_share_of_top10:.2%}"
    )

    print(
        f"Top-5 share of Top-10 hits:   "
        f"{top5_share_of_top10:.2%}"
    )

    print()
    print(
        "Interpretation boundary:"
    )

    print(
        "  These results describe HOW the "
        "frozen model ranks positives."
    )

    print(
        "  They do not yet identify WHICH "
        "subgroups generate the tail."
    )

    # =========================================================================
    # 7.1.10 — Write data artifacts
    # =========================================================================

    print_section(
        "7.1.10 — WRITE DERIVED ANALYSIS ARTIFACTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rank_frequency_path = (
        OUT_DIR
        / "phase_7_1_rank_frequency_V1.csv"
    )

    topk_path = (
        OUT_DIR
        / "phase_7_1_topk_curve_V1.csv"
    )

    rank_bands_path = (
        OUT_DIR
        / "phase_7_1_rank_bands_V1.csv"
    )

    baseline_metrics_path = (
        OUT_DIR
        / "phase_7_1_baseline_metrics_V1.json"
    )

    insights_path = (
        OUT_DIR
        / "phase_7_1_descriptive_insights_V1.json"
    )

    rank_frequency.to_csv(
        rank_frequency_path,
        index=False,
    )

    topk.to_csv(
        topk_path,
        index=False,
    )

    rank_bands.to_csv(
        rank_bands_path,
        index=False,
    )

    baseline_metrics = {

        "n": len(df),

        "hits": {
            "Hits@1": hits1,
            "Hits@5": hits5,
            "Hits@10": hits10,
        },

        "ranking_metrics": {
            "HR@1": hr1,
            "HR@5": hr5,
            "HR@10": hr10,
            "NDCG@10": ndcg10,
        },

        "rank_summary": {
            "mean": mean_rank,
            "median": median_rank,
            "std": std_rank,
            "min": int(rank.min()),
            "max": int(rank.max()),
            **quantiles,
        },

        "tail": tail_metrics,
    }

    json_dump(
        baseline_metrics,
        baseline_metrics_path,
    )

    json_dump(
        descriptive_insights,
        insights_path,
    )

    # =========================================================================
    # 7.1.11 — Figure 1: exact positive-rank distribution
    # =========================================================================

    print_section(
        "7.1.11 — GENERATE PRESENTATION-READY FIGURES"
    )

    figure_paths = []

    fig, ax = plt.subplots(
        figsize=(12, 6.5)
    )

    ax.bar(
        rank_frequency[
            "positive_rank"
        ],
        rank_frequency["n"],
        width=0.85,
    )

    ax.axvline(
        5,
        linestyle="--",
        linewidth=1.2,
        label="Top-5 cutoff",
    )

    ax.axvline(
        10,
        linestyle="--",
        linewidth=1.2,
        label="Top-10 cutoff",
    )

    ax.set_title(
        "Positive-rank distribution on the frozen ITRS final test"
    )

    ax.set_xlabel(
        "Positive startup rank among 100 candidates"
    )

    ax.set_ylabel(
        "Number of test cases"
    )

    ax.set_xlim(
        0.5,
        100.5,
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_1_fig1_"
                "positive_rank_distribution_V1"
            ),
        )
    )

    # =========================================================================
    # Figure 2: cumulative HR@k
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )

    ax.plot(
        topk["k"],
        topk["HR@k"] * 100.0,
        linewidth=2.0,
    )

    for cutoff in (
        1,
        5,
        10,
    ):

        value = float(
            topk.loc[
                topk["k"] == cutoff,
                "HR@k",
            ].iloc[0]
            * 100.0
        )

        ax.scatter(
            [cutoff],
            [value],
            s=45,
            zorder=3,
        )

        ax.annotate(
            f"HR@{cutoff} = {value:.1f}%",
            xy=(
                cutoff,
                value,
            ),
            xytext=(
                8,
                8,
            ),
            textcoords="offset points",
        )

    ax.axvline(
        10,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_title(
        "Cumulative retrieval of the positive startup"
    )

    ax.set_xlabel(
        "Recommendation cutoff k"
    )

    ax.set_ylabel(
        "Hit rate (%)"
    )

    ax.set_xlim(
        1,
        100,
    )

    ax.set_ylim(
        0,
        101,
    )

    ax.grid(
        alpha=0.20,
    )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_1_fig2_"
                "cumulative_HR_at_k_V1"
            ),
        )
    )

    # =========================================================================
    # Figure 3: rank-band composition
    # =========================================================================

    fig, ax = plt.subplots(
        figsize=(10.5, 6.5)
    )

    x = np.arange(
        len(rank_bands)
    )

    shares_pct = (
        rank_bands["share"]
        .to_numpy()
        * 100.0
    )

    bars = ax.bar(
        x,
        shares_pct,
    )

    ax.set_xticks(
        x,
        rank_bands[
            "rank_band"
        ].astype(str),
        rotation=20,
        ha="right",
    )

    ax.set_title(
        "Where the positive startup appears in the ranking"
    )

    ax.set_xlabel(
        "Positive-rank band"
    )

    ax.set_ylabel(
        "Share of test cases (%)"
    )

    ax.grid(
        axis="y",
        alpha=0.20,
    )

    for bar, value in zip(
        bars,
        shares_pct,
    ):

        ax.annotate(
            f"{value:.1f}%",
            xy=(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height(),
            ),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()

    figure_paths.extend(
        save_figure(
            fig,
            (
                "phase_7_1_fig3_"
                "rank_band_distribution_V1"
            ),
        )
    )

    # =========================================================================
    # 7.1.12 — Provenance manifest
    # =========================================================================

    manifest_path = (
        OUT_DIR
        / "phase_7_1_analysis_manifest_V1.json"
    )

    manifest = {

        "phase": "7.1",

        "schema_version":
            "ITRS_PHASE7_1_BASELINE_RANK_ANATOMY_V1",

        "status": "COMPLETE",

        "scientific_role": (
            "Descriptive post-hoc rank analysis "
            "of the immutable Phase-6 final test."
        ),

        "source": {
            "path": str(
                SOURCE.relative_to(
                    REPO_ROOT
                )
            ),
            "sha256": source_hash,
            "rows": len(df),
        },

        "phase_7_0_contract": str(
            PHASE_7_0_CONTRACT.relative_to(
                REPO_ROOT
            )
        ),

        "prohibited_operations": {
            "model_loaded": False,
            "checkpoint_loaded": False,
            "raw_logits_loaded": False,
            "inference_executed": False,
            "test_rescored": False,
            "negative_resampling": False,
            "model_selection": False,
        },

        "metrics_artifact": str(
            baseline_metrics_path.relative_to(
                REPO_ROOT
            )
        ),

        "insights_artifact": str(
            insights_path.relative_to(
                REPO_ROOT
            )
        ),

        "tables": [
            str(
                rank_frequency_path.relative_to(
                    REPO_ROOT
                )
            ),
            str(
                topk_path.relative_to(
                    REPO_ROOT
                )
            ),
            str(
                rank_bands_path.relative_to(
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

        "interpretation_boundary": (
            "Phase 7.1 describes aggregate rank "
            "behavior only. It does not attribute "
            "failures to novelty, cold start, "
            "history, or structural coverage."
        ),
    }

    json_dump(
        manifest,
        manifest_path,
    )

    # =========================================================================
    # Hash derived artifacts
    # =========================================================================

    artifact_paths = [
        rank_frequency_path,
        topk_path,
        rank_bands_path,
        baseline_metrics_path,
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

        hashes[relative] = file_sha256(
            path
        )

        print(
            f"WROTE  {relative}"
        )

    hashes_path = (
        OUT_DIR
        / "phase_7_1_derived_artifact_sha256_V1.json"
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
    # Final result
    # =========================================================================

    print_section(
        "PHASE 7.1 V1 RESULT"
    )

    print("Frozen source binding:            PASS")
    print("Aggregate metric reconstruction:  PASS")
    print("Rank quantiles:                   COMPLETE")
    print("Long-tail diagnostics:            COMPLETE")
    print("Rank-frequency table:             COMPLETE")
    print("Top-K cumulative curve:           COMPLETE")
    print("Rank-band decomposition:          COMPLETE")
    print("Presentation figures:             COMPLETE")
    print("Provenance manifest:              COMPLETE")

    print()
    print("Model inference executed:         NO")
    print("Final test rescored:              NO")
    print("Final test modified:              NO")

    print()
    print(
        "PHASE 7.1 BASELINE RANK ANATOMY STATUS: COMPLETE"
    )


if __name__ == "__main__":
    main()