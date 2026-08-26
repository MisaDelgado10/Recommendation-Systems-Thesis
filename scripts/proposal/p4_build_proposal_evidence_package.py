from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =============================================================================
# P4 — BUILD PROPOSAL-READY EVIDENCE PACKAGE
# =============================================================================
#
# NO TRAINING
# NO INFERENCE
# NO MODEL SELECTION
#
# Pure reporting/visualization over frozen P1-P3 outputs.
# =============================================================================


ROOT = Path(__file__).resolve().parents[2]

EVIDENCE = (
    ROOT
    / "data"
    / "experimental"
    / "proposal_evidence"
)

FIGURES = EVIDENCE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Inputs
# =============================================================================

SUBGROUPS = EVIDENCE / "07_subgroup_performance.csv"
HISTORY = EVIDENCE / "08_startup_history_performance.csv"
CANDIDATES = EVIDENCE / "12_discovery_candidate_statistics.csv"
DISCOVERY = EVIDENCE / "13_new_to_investor_discovery_metrics.csv"
FEATURES = EVIDENCE / "17_feature_coverage_by_startup_history.csv"


# =============================================================================
# Outputs
# =============================================================================

OUT_SUBGROUP_TABLE = (
    EVIDENCE
    / "19_proposal_ready_subgroup_performance.csv"
)

OUT_FEATURE_TABLE = (
    EVIDENCE
    / "20_proposal_ready_cold_start_features.csv"
)

OUT_CLAIMS = (
    EVIDENCE
    / "21_proposal_key_evidence_claims.csv"
)

OUT_SUMMARY = (
    EVIDENCE
    / "proposal_evidence_final_summary.json"
)


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def banner(text: str) -> None:
    print()
    print("=" * 112)
    print(text)
    print("=" * 112)


def get_row(
    df: pd.DataFrame,
    column: str,
    value: str,
) -> pd.Series:

    mask = (
        df[column]
        .astype(str)
        .eq(str(value))
    )

    require(
        int(mask.sum()) == 1,
        (
            f"Expected exactly one row where "
            f"{column}={value}; found {int(mask.sum())}."
        ),
    )

    return df.loc[mask].iloc[0]


def relative_drop(old: float, new: float) -> float:
    return (old - new) / old


def save_bar_chart(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    filename: str,
    ylim: tuple[float, float] | None = None,
) -> None:

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )

    bars = ax.bar(
        labels,
        values,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    for bar, value in zip(
        bars,
        values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    fig.savefig(
        FIGURES / filename,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def main() -> None:

    banner(
        "P4 — BUILD PROPOSAL-READY EVIDENCE PACKAGE"
    )

    for path in [
        SUBGROUPS,
        HISTORY,
        CANDIDATES,
        DISCOVERY,
        FEATURES,
    ]:
        require(
            path.exists(),
            f"Missing required P1-P3 output: {path}",
        )

        print(
            "FOUND ",
            path.relative_to(ROOT),
        )

    subgroup = pd.read_csv(SUBGROUPS)
    history = pd.read_csv(HISTORY)
    candidate = pd.read_csv(CANDIDATES)
    discovery = pd.read_csv(DISCOVERY)
    features = pd.read_csv(FEATURES)

    # =========================================================================
    # 1. Core diagnostic rows
    # =========================================================================

    repeat = get_row(
        subgroup,
        "group",
        "repeat_pair",
    )

    warm_warm = get_row(
        subgroup,
        "group",
        "novel_warm_warm",
    )

    cold_startup = get_row(
        subgroup,
        "group",
        "novel_cold_startup",
    )

    cold_investor = get_row(
        subgroup,
        "group",
        "novel_cold_investor",
    )

    both_cold = get_row(
        subgroup,
        "group",
        "novel_both_cold",
    )

    all_test = get_row(
        subgroup,
        "group",
        "all_test",
    )

    all_discovery = get_row(
        subgroup,
        "group",
        "all_new_to_investor",
    )

    require(
        int(all_test["n"]) == 20_264,
        "All-test count drift.",
    )

    require(
        int(all_discovery["n"]) == 16_446,
        "New-to-investor count drift.",
    )

    # =========================================================================
    # 2. Proposal-ready subgroup table
    # =========================================================================

    subgroup_order = [
        (
            "Repeat pair",
            repeat,
        ),
        (
            "Novel pair: warm investor + warm startup",
            warm_warm,
        ),
        (
            "Novel pair: warm investor + cold startup",
            cold_startup,
        ),
        (
            "Novel pair: cold investor + warm startup",
            cold_investor,
        ),
        (
            "Novel pair: both cold",
            both_cold,
        ),
    ]

    ready_rows = []

    for label, row in subgroup_order:

        ready_rows.append(
            {
                "Evaluation group":
                    label,

                "N":
                    int(row["n"]),

                "Share of test (%)":
                    float(
                        row["share_of_test"]
                        * 100
                    ),

                "HR@10":
                    float(row["HR@10"]),

                "NDCG@10":
                    float(row["NDCG@10"]),

                "Mean positive rank":
                    float(
                        row[
                            "mean_positive_rank"
                        ]
                    ),

                "Median positive rank":
                    float(
                        row[
                            "median_positive_rank"
                        ]
                    ),
            }
        )

    ready_subgroups = pd.DataFrame(
        ready_rows
    )

    ready_subgroups.to_csv(
        OUT_SUBGROUP_TABLE,
        index=False,
    )

    # =========================================================================
    # 3. Cold-start feature coverage
    # =========================================================================

    unique_features = features.loc[
        features[
            "level"
        ].astype(str).eq(
            "unique_startup"
        )
    ].copy()

    cold_feature_row = get_row(
        unique_features,
        "group",
        "0",
    )

    feature_table = pd.DataFrame(
        [
            {
                "Feature signal":
                    "Description / Doc2Vec",

                "Coverage among unique zero-history startups (%)":
                    float(
                        cold_feature_row[
                            "doc2vec_coverage"
                        ]
                        * 100
                    ),
            },
            {
                "Feature signal":
                    "Category features",

                "Coverage among unique zero-history startups (%)":
                    float(
                        cold_feature_row[
                            "category_coverage"
                        ]
                        * 100
                    ),
            },
            {
                "Feature signal":
                    "Incoming heterogeneous structure",

                "Coverage among unique zero-history startups (%)":
                    float(
                        cold_feature_row[
                            "incoming_structural_coverage"
                        ]
                        * 100
                    ),
            },
            {
                "Feature signal":
                    "Any currently model-usable side information",

                "Coverage among unique zero-history startups (%)":
                    float(
                        cold_feature_row[
                            "model_usable_side_info_coverage"
                        ]
                        * 100
                    ),
            },
        ]
    )

    feature_table.to_csv(
        OUT_FEATURE_TABLE,
        index=False,
    )

    # =========================================================================
    # 4. Quantified thesis claims
    # =========================================================================

    discovery_share = (
        float(all_discovery["n"])
        / float(all_test["n"])
    )

    repeat_to_novel_hr_drop = relative_drop(
        float(repeat["HR@10"]),
        float(warm_warm["HR@10"]),
    )

    repeat_to_novel_ndcg_drop = relative_drop(
        float(repeat["NDCG@10"]),
        float(warm_warm["NDCG@10"]),
    )

    novel_to_cold_hr_drop = relative_drop(
        float(warm_warm["HR@10"]),
        float(cold_startup["HR@10"]),
    )

    novel_to_cold_ndcg_drop = relative_drop(
        float(warm_warm["NDCG@10"]),
        float(cold_startup["NDCG@10"]),
    )

    history_zero = get_row(
        history,
        "startup_history_bin",
        "0",
    )

    history_high = get_row(
        history,
        "startup_history_bin",
        "10+",
    )

    discovery_candidate = get_row(
        candidate,
        "group",
        "all_new_to_investor",
    )

    claims = pd.DataFrame(
        [
            {
                "claim_id":
                    "C1",

                "thesis_role":
                    "Discovery-task prevalence",

                "measure":
                    "Share of test cases that are new-to-investor",

                "value":
                    discovery_share,

                "display_value":
                    f"{discovery_share:.2%}",

                "interpretation":
                    (
                        "New-to-investor relationships dominate "
                        "the frozen final test."
                    ),
            },
            {
                "claim_id":
                    "C2",

                "thesis_role":
                    "Novel-pair difficulty",

                "measure":
                    "Relative HR@10 drop: repeat pair -> novel warm-warm",

                "value":
                    repeat_to_novel_hr_drop,

                "display_value":
                    f"{repeat_to_novel_hr_drop:.1%}",

                "interpretation":
                    (
                        "Novel relationship discovery is harder "
                        "even when both entities have prior history."
                    ),
            },
            {
                "claim_id":
                    "C3",

                "thesis_role":
                    "Novel-pair difficulty",

                "measure":
                    "Relative NDCG@10 drop: repeat pair -> novel warm-warm",

                "value":
                    repeat_to_novel_ndcg_drop,

                "display_value":
                    f"{repeat_to_novel_ndcg_drop:.1%}",

                "interpretation":
                    (
                        "Novel-pair degradation is also present "
                        "in ranking position quality."
                    ),
            },
            {
                "claim_id":
                    "C4",

                "thesis_role":
                    "Cold-start difficulty",

                "measure":
                    "Relative HR@10 drop: novel warm-warm -> novel cold-startup",

                "value":
                    novel_to_cold_hr_drop,

                "display_value":
                    f"{novel_to_cold_hr_drop:.1%}",

                "interpretation":
                    (
                        "Startup cold-start introduces a large "
                        "additional degradation beyond novel-pair difficulty."
                    ),
            },
            {
                "claim_id":
                    "C5",

                "thesis_role":
                    "Cold-start difficulty",

                "measure":
                    "Relative NDCG@10 drop: novel warm-warm -> novel cold-startup",

                "value":
                    novel_to_cold_ndcg_drop,

                "display_value":
                    f"{novel_to_cold_ndcg_drop:.1%}",

                "interpretation":
                    (
                        "Cold-start degradation is especially strong "
                        "in top-rank quality."
                    ),
            },
            {
                "claim_id":
                    "C6",

                "thesis_role":
                    "History sensitivity",

                "measure":
                    "HR@10 at zero prior startup interactions",

                "value":
                    float(
                        history_zero["HR@10"]
                    ),

                "display_value":
                    f"{float(history_zero['HR@10']):.3f}",

                "interpretation":
                    (
                        "Performance is weakest for startups with "
                        "no prior investment history."
                    ),
            },
            {
                "claim_id":
                    "C7",

                "thesis_role":
                    "History sensitivity",

                "measure":
                    "HR@10 at 10+ prior startup interactions",

                "value":
                    float(
                        history_high["HR@10"]
                    ),

                "display_value":
                    f"{float(history_high['HR@10']):.3f}",

                "interpretation":
                    (
                        "Performance rises sharply for startups "
                        "with substantial investment history."
                    ),
            },
            {
                "claim_id":
                    "C8",

                "thesis_role":
                    "Inductive feasibility",

                "measure":
                    "Description coverage among unique cold-start test startups",

                "value":
                    float(
                        cold_feature_row[
                            "doc2vec_coverage"
                        ]
                    ),

                "display_value":
                    (
                        f"{float(cold_feature_row['doc2vec_coverage']):.2%}"
                    ),

                "interpretation":
                    (
                        "Zero investment history does not imply "
                        "absence of content information."
                    ),
            },
            {
                "claim_id":
                    "C9",

                "thesis_role":
                    "Inductive feasibility",

                "measure":
                    "Category coverage among unique cold-start test startups",

                "value":
                    float(
                        cold_feature_row[
                            "category_coverage"
                        ]
                    ),

                "display_value":
                    (
                        f"{float(cold_feature_row['category_coverage']):.2%}"
                    ),

                "interpretation":
                    (
                        "Most cold-start startups also retain "
                        "categorical side information."
                    ),
            },
            {
                "claim_id":
                    "C10",

                "thesis_role":
                    "Architecture constraint",

                "measure":
                    "Incoming structural coverage among unique cold-start startups",

                "value":
                    float(
                        cold_feature_row[
                            "incoming_structural_coverage"
                        ]
                    ),

                "display_value":
                    (
                        f"{float(cold_feature_row['incoming_structural_coverage']):.2%}"
                    ),

                "interpretation":
                    (
                        "Graph-neighborhood information is sparse for "
                        "cold-start startups, motivating content-first induction."
                    ),
            },
            {
                "claim_id":
                    "C11",

                "thesis_role":
                    "Discovery-scale realism",

                "measure":
                    "Median strict discovery candidate universe",

                "value":
                    float(
                        discovery_candidate[
                            "candidate_pool_median"
                        ]
                    ),

                "display_value":
                    (
                        f"{float(discovery_candidate['candidate_pool_median']):,.0f}"
                    ),

                "interpretation":
                    (
                        "The realistic startup discovery universe is "
                        "orders of magnitude larger than the sampled 100-candidate evaluation."
                    ),
            },
        ]
    )

    claims.to_csv(
        OUT_CLAIMS,
        index=False,
    )

    # =========================================================================
    # 5. Figure — subgroup HR@10
    # =========================================================================

    labels = [
        "Repeat",
        "Novel\nwarm-warm",
        "Novel\ncold startup",
        "Novel\ncold investor",
        "Novel\nboth cold",
    ]

    subgroup_rows = [
        repeat,
        warm_warm,
        cold_startup,
        cold_investor,
        both_cold,
    ]

    save_bar_chart(
        labels=labels,
        values=[
            float(row["HR@10"])
            for row in subgroup_rows
        ],
        title=(
            "ITRS HR@10 by New-to-Investor and Cold-Start Status"
        ),
        ylabel="HR@10",
        filename="01_itrs_hr10_by_subgroup.png",
        ylim=(0, 0.7),
    )

    # =========================================================================
    # 6. Figure — subgroup NDCG@10
    # =========================================================================

    save_bar_chart(
        labels=labels,
        values=[
            float(row["NDCG@10"])
            for row in subgroup_rows
        ],
        title=(
            "ITRS NDCG@10 by New-to-Investor and Cold-Start Status"
        ),
        ylabel="NDCG@10",
        filename="02_itrs_ndcg10_by_subgroup.png",
        ylim=(0, 0.4),
    )

    # =========================================================================
    # 7. Figure — performance vs startup history
    # =========================================================================

    history_order = [
        "0",
        "1",
        "2-4",
        "5-9",
        "10+",
    ]

    history_plot = (
        history
        .assign(
            startup_history_bin=lambda x:
                x[
                    "startup_history_bin"
                ].astype(str)
        )
        .set_index(
            "startup_history_bin"
        )
        .loc[
            history_order
        ]
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    ax.plot(
        history_order,
        history_plot["HR@10"],
        marker="o",
        label="HR@10",
    )

    ax.plot(
        history_order,
        history_plot["NDCG@10"],
        marker="o",
        label="NDCG@10",
    )

    ax.set_title(
        "ITRS Performance vs. Prior Startup Investment History"
    )

    ax.set_xlabel(
        "Number of pre-T60 startup investment events"
    )

    ax.set_ylabel(
        "Ranking metric"
    )

    ax.set_ylim(
        0,
        0.75,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES
        / "03_performance_vs_startup_history.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =========================================================================
    # 8. Figure — cold-start feature coverage
    # =========================================================================

    feature_labels = [
        "Description",
        "Category",
        "Incoming\nstructure",
        "Any usable\nside info",
    ]

    feature_values = [
        float(
            cold_feature_row[
                "doc2vec_coverage"
            ]
            * 100
        ),
        float(
            cold_feature_row[
                "category_coverage"
            ]
            * 100
        ),
        float(
            cold_feature_row[
                "incoming_structural_coverage"
            ]
            * 100
        ),
        float(
            cold_feature_row[
                "model_usable_side_info_coverage"
            ]
            * 100
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(8.5, 5.5)
    )

    bars = ax.bar(
        feature_labels,
        feature_values,
    )

    ax.set_title(
        "Side-Information Coverage for Zero-History Startups"
    )

    ax.set_ylabel(
        "Coverage among unique cold-start startups (%)"
    )

    ax.set_ylim(
        0,
        110,
    )

    for bar, value in zip(
        bars,
        feature_values,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    fig.savefig(
        FIGURES
        / "04_cold_start_feature_coverage.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # =========================================================================
    # 9. Final JSON summary
    # =========================================================================

    final_summary = {
        "schema_version":
            "PROPOSAL_EVIDENCE_FINAL_V1",

        "status":
            "P0_P1_P2_P3_P4_COMPLETE",

        "headline_evidence": {
            "test_cases":
                int(all_test["n"]),

            "new_to_investor_cases":
                int(all_discovery["n"]),

            "new_to_investor_share":
                discovery_share,

            "repeat_pair_HR10":
                float(
                    repeat["HR@10"]
                ),

            "novel_warm_warm_HR10":
                float(
                    warm_warm["HR@10"]
                ),

            "novel_cold_startup_HR10":
                float(
                    cold_startup["HR@10"]
                ),

            "repeat_to_novel_HR10_relative_drop":
                repeat_to_novel_hr_drop,

            "novel_to_cold_startup_HR10_relative_drop":
                novel_to_cold_hr_drop,

            "novel_to_cold_startup_NDCG10_relative_drop":
                novel_to_cold_ndcg_drop,

            "cold_start_unique_description_coverage":
                float(
                    cold_feature_row[
                        "doc2vec_coverage"
                    ]
                ),

            "cold_start_unique_category_coverage":
                float(
                    cold_feature_row[
                        "category_coverage"
                    ]
                ),

            "cold_start_unique_incoming_structure_coverage":
                float(
                    cold_feature_row[
                        "incoming_structural_coverage"
                    ]
                ),

            "cold_start_unique_any_usable_side_info":
                float(
                    cold_feature_row[
                        "model_usable_side_info_coverage"
                    ]
                ),

            "median_discovery_candidate_universe":
                float(
                    discovery_candidate[
                        "candidate_pool_median"
                    ]
                ),
        },

        "interpretation_guards": [
            (
                "P1 subgroup differences are diagnostic associations, "
                "not causal estimates."
            ),
            (
                "P2 reuses the frozen Phase-6 sampled candidate sets; "
                "no neural rescoring was performed."
            ),
            (
                "P3 demonstrates feature availability in frozen inputs; "
                "it does not independently establish historical "
                "availability of every static metadata field."
            ),
            (
                "The sampled 100-candidate evaluation is not equivalent "
                "to ranking the full ~311k discovery universe."
            ),
        ],

        "proposal_recommendation": (
            "Use a content-first inductive startup representation "
            "with optional heterogeneous structural enrichment, "
            "integrated with the existing temporal investor-preference "
            "model and evaluated explicitly on new-to-investor discovery."
        ),

        "outputs": {
            "subgroup_table":
                str(
                    OUT_SUBGROUP_TABLE.relative_to(
                        ROOT
                    )
                ),

            "cold_start_feature_table":
                str(
                    OUT_FEATURE_TABLE.relative_to(
                        ROOT
                    )
                ),

            "key_claims":
                str(
                    OUT_CLAIMS.relative_to(
                        ROOT
                    )
                ),

            "figures_directory":
                str(
                    FIGURES.relative_to(
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
            final_summary,
            handle,
            indent=2,
        )

    # =========================================================================
    # 10. Console
    # =========================================================================

    banner(
        "PROPOSAL-READY CORE PERFORMANCE TABLE"
    )

    print(
        ready_subgroups.to_string(
            index=False,
            formatters={
                "Share of test (%)":
                    lambda x:
                        f"{x:.2f}",

                "HR@10":
                    lambda x:
                        f"{x:.4f}",

                "NDCG@10":
                    lambda x:
                        f"{x:.4f}",

                "Mean positive rank":
                    lambda x:
                        f"{x:.2f}",

                "Median positive rank":
                    lambda x:
                        f"{x:.1f}",
            },
        )
    )

    banner(
        "PROPOSAL-READY KEY CLAIMS"
    )

    print(
        claims[
            [
                "claim_id",
                "thesis_role",
                "display_value",
                "measure",
            ]
        ].to_string(
            index=False
        )
    )

    banner(
        "P4 OUTPUTS"
    )

    for path in [
        OUT_SUBGROUP_TABLE,
        OUT_FEATURE_TABLE,
        OUT_CLAIMS,
        OUT_SUMMARY,
    ]:
        print(
            "WROTE ",
            path.relative_to(ROOT),
        )

    print()
    print("FIGURES")

    for path in sorted(
        FIGURES.glob("*.png")
    ):
        print(
            "WROTE ",
            path.relative_to(ROOT),
        )

    banner(
        "P4 COMPLETE — PROPOSAL EVIDENCE PACKAGE READY"
    )


if __name__ == "__main__":
    main()
