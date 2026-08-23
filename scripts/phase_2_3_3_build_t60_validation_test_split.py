from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 2.3.3 — BUILD PAPER-STYLE T60 VALIDATION / TEST SPLIT
# =============================================================================
#
# PAPER PROTOCOL
# --------------
#
# The ITRS paper uses:
#
#     - first t-1 temporal fragments as historical training data
#     - 10% of events in the final fragment Tt as validation
#     - remaining events in Tt as test
#
# The paper does NOT report a random seed.
#
# REPRODUCTION-SPECIFIC DECISION:
#
#     SPLIT_SEED = 42
#
# The split is EVENT-LEVEL and UNSTRATIFIED to remain as close as possible
# to the paper's stated random Tt split.
#
# =============================================================================


INPUT_PATH = Path(
    "data/experimental/phase_2/audits/"
    "t60_evaluation_interaction_flags.parquet"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_2/model_ready"
)

AUDIT_DIR = Path(
    "data/experimental/phase_2/audits"
)

SPLIT_OUTPUT = (
    OUTPUT_DIR
    / "t60_validation_test_split.parquet"
)

ASSIGNMENT_OUTPUT = (
    OUTPUT_DIR
    / "t60_split_assignments.csv"
)

SUMMARY_OUTPUT = (
    AUDIT_DIR
    / "t60_validation_test_split_summary.csv"
)

PAIR_OVERLAP_OUTPUT = (
    AUDIT_DIR
    / "t60_validation_test_pair_overlap.csv"
)

STATUS_OUTPUT = (
    AUDIT_DIR
    / "t60_validation_test_status_distribution.csv"
)


EXPECTED_T60_ROWS = 22_515

VALIDATION_FRACTION = 0.10

# Reproduction-specific choice.
# The ITRS paper does not report its random seed.
SPLIT_SEED = 42


def separator(char="=", width=120):
    print(char * width)


def pct(num, den):

    if den == 0:
        return np.nan

    return num / den * 100


def summarize_split(subset, split_name):

    return {
        "split": split_name,

        "interactions": len(subset),

        "unique_investors": (
            subset["investor_id"]
            .nunique()
        ),

        "unique_startups": (
            subset["startup_id"]
            .nunique()
        ),

        "unique_pairs": (
            subset[
                [
                    "investor_id",
                    "startup_id",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),

        "unique_funding_rounds": (
            subset["funding_round_id"]
            .nunique()
        ),

        "new_to_investor_pair_events": int(
            subset[
                "new_to_investor_pair"
            ]
            .sum()
        ),

        "new_to_investor_pair_event_share_pct": pct(
            subset[
                "new_to_investor_pair"
            ]
            .sum(),
            len(subset),
        ),

        "prior_pair_repeat_events": int(
            subset[
                "pair_seen_before_t60"
            ]
            .sum()
        ),

        "prior_pair_repeat_event_share_pct": pct(
            subset[
                "pair_seen_before_t60"
            ]
            .sum(),
            len(subset),
        ),

        "events_with_interaction_cold_investor": int(
            (
                ~subset[
                    "investor_seen_before_t60"
                ]
            )
            .sum()
        ),

        "interaction_cold_investor_event_share_pct": pct(
            (
                ~subset[
                    "investor_seen_before_t60"
                ]
            )
            .sum(),
            len(subset),
        ),

        "events_with_interaction_cold_startup": int(
            (
                ~subset[
                    "startup_seen_before_t60"
                ]
            )
            .sum()
        ),

        "interaction_cold_startup_event_share_pct": pct(
            (
                ~subset[
                    "startup_seen_before_t60"
                ]
            )
            .sum(),
            len(subset),
        ),
    }


def main():

    separator()
    print(
        "PHASE 2.3.3 — "
        "BUILD PAPER-STYLE T60 VALIDATION / TEST SPLIT"
    )
    separator()

    # -------------------------------------------------------------------------
    # 1. Load audited T60 interaction pool
    # -------------------------------------------------------------------------

    t60 = pd.read_parquet(
        INPUT_PATH
    )

    if len(t60) != EXPECTED_T60_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_T60_ROWS:,} T60 interactions; "
            f"found {len(t60):,}."
        )

    if t60["interaction_id"].duplicated().any():
        raise ValueError(
            "T60 contains duplicate interaction_id values."
        )

    print(
        f"\nT60 interactions:       "
        f"{len(t60):,}"
    )

    print(
        f"Validation fraction:    "
        f"{VALIDATION_FRACTION:.2%}"
    )

    print(
        f"Reproduction seed:      "
        f"{SPLIT_SEED}"
    )

    # -------------------------------------------------------------------------
    # 2. Establish stable row ordering before random sampling
    #
    # Sorting by canonical interaction_id means that the same seed produces
    # the same split even if parquet row ordering later changes.
    # -------------------------------------------------------------------------

    t60 = (
        t60.sort_values(
            "interaction_id",
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    # -------------------------------------------------------------------------
    # 3. Determine exact validation size
    # -------------------------------------------------------------------------

    validation_size = int(
        np.floor(
            len(t60)
            * VALIDATION_FRACTION
        )
    )

    test_size = (
        len(t60)
        - validation_size
    )

    print(
        f"\nValidation events:       "
        f"{validation_size:,}"
    )

    print(
        f"Test events:             "
        f"{test_size:,}"
    )

    # -------------------------------------------------------------------------
    # 4. Random EVENT-level split
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(
        SPLIT_SEED
    )

    validation_indices = rng.choice(
        len(t60),
        size=validation_size,
        replace=False,
    )

    is_validation = np.zeros(
        len(t60),
        dtype=bool,
    )

    is_validation[
        validation_indices
    ] = True

    t60["evaluation_split"] = np.where(
        is_validation,
        "validation",
        "test",
    )

    validation = (
        t60[
            t60["evaluation_split"]
            == "validation"
        ]
        .copy()
    )

    test = (
        t60[
            t60["evaluation_split"]
            == "test"
        ]
        .copy()
    )

    # -------------------------------------------------------------------------
    # 5. Verify exact split sizes
    # -------------------------------------------------------------------------

    if len(validation) != validation_size:
        raise ValueError(
            "Validation size mismatch."
        )

    if len(test) != test_size:
        raise ValueError(
            "Test size mismatch."
        )

    if (
        len(validation)
        + len(test)
        != EXPECTED_T60_ROWS
    ):
        raise ValueError(
            "Validation + test does not reconstruct T60."
        )

    # -------------------------------------------------------------------------
    # 6. Pair overlap between validation and test
    #
    # Because the paper uses event-level random splitting, the same
    # investor-startup pair can potentially appear in both.
    #
    # We do NOT prevent this. We only measure it.
    # -------------------------------------------------------------------------

    validation_pairs = (
        validation[
            [
                "investor_id",
                "startup_id",
            ]
        ]
        .drop_duplicates()
    )

    test_pairs = (
        test[
            [
                "investor_id",
                "startup_id",
            ]
        ]
        .drop_duplicates()
    )

    pair_overlap = (
        validation_pairs.merge(
            test_pairs,
            on=[
                "investor_id",
                "startup_id",
            ],
            how="inner",
        )
    )

    pair_overlap_count = len(
        pair_overlap
    )

    validation_events_on_overlap_pairs = (
        validation.merge(
            pair_overlap,
            on=[
                "investor_id",
                "startup_id",
            ],
            how="inner",
        )
        .shape[0]
    )

    test_events_on_overlap_pairs = (
        test.merge(
            pair_overlap,
            on=[
                "investor_id",
                "startup_id",
            ],
            how="inner",
        )
        .shape[0]
    )

    # -------------------------------------------------------------------------
    # 7. Funding-round overlap diagnostic
    #
    # Several investors can participate in the same funding round.
    # Event-level splitting can therefore place interactions from the same
    # funding round into both validation and test.
    #
    # This is diagnostic only; the paper's stated split is not altered.
    # -------------------------------------------------------------------------

    validation_rounds = set(
        validation[
            "funding_round_id"
        ]
        .unique()
    )

    test_rounds = set(
        test[
            "funding_round_id"
        ]
        .unique()
    )

    round_overlap = (
        validation_rounds
        & test_rounds
    )

    # -------------------------------------------------------------------------
    # 8. Split-level summaries
    # -------------------------------------------------------------------------

    summary = pd.DataFrame(
        [
            summarize_split(
                validation,
                "validation",
            ),
            summarize_split(
                test,
                "test",
            ),
        ]
    )

    # -------------------------------------------------------------------------
    # 9. Interaction-cold-start status distribution by split
    # -------------------------------------------------------------------------

    status_distribution = (
        t60.groupby(
            [
                "evaluation_split",
                "interaction_cold_start_status",
            ],
            observed=True,
        )
        .size()
        .rename(
            "interaction_count"
        )
        .reset_index()
    )

    split_totals = (
        status_distribution.groupby(
            "evaluation_split"
        )[
            "interaction_count"
        ]
        .transform("sum")
    )

    status_distribution[
        "interaction_share_within_split_pct"
    ] = (
        status_distribution[
            "interaction_count"
        ]
        / split_totals
        * 100
    )

    # -------------------------------------------------------------------------
    # 10. Pair-overlap output
    # -------------------------------------------------------------------------

    pair_overlap_output = (
        pair_overlap.copy()
    )

    if len(pair_overlap_output) > 0:

        validation_pair_counts = (
            validation.groupby(
                [
                    "investor_id",
                    "startup_id",
                ],
                observed=True,
            )
            .size()
            .rename(
                "validation_event_count"
            )
            .reset_index()
        )

        test_pair_counts = (
            test.groupby(
                [
                    "investor_id",
                    "startup_id",
                ],
                observed=True,
            )
            .size()
            .rename(
                "test_event_count"
            )
            .reset_index()
        )

        pair_overlap_output = (
            pair_overlap_output
            .merge(
                validation_pair_counts,
                on=[
                    "investor_id",
                    "startup_id",
                ],
                how="left",
            )
            .merge(
                test_pair_counts,
                on=[
                    "investor_id",
                    "startup_id",
                ],
                how="left",
            )
        )

    # -------------------------------------------------------------------------
    # 11. Save outputs
    # -------------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    t60.to_parquet(
        SPLIT_OUTPUT,
        index=False,
    )

    t60[
        [
            "interaction_id",
            "evaluation_split",
        ]
    ].to_csv(
        ASSIGNMENT_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    pair_overlap_output.to_csv(
        PAIR_OVERLAP_OUTPUT,
        index=False,
    )

    status_distribution.to_csv(
        STATUS_OUTPUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # 12. Print summary
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "VALIDATION / TEST SUMMARY"
    )
    separator("-")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # -------------------------------------------------------------------------
    # 13. Print cold-start composition
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "INTERACTION-COLD-START STATUS BY SPLIT"
    )
    separator("-")

    print(
        status_distribution.to_string(
            index=False,
            float_format=lambda x: f"{x:.2f}",
        )
    )

    # -------------------------------------------------------------------------
    # 14. Print overlap diagnostics
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "EVENT-LEVEL SPLIT OVERLAP DIAGNOSTICS"
    )
    separator("-")

    print(
        f"Unique investor-startup pairs "
        f"in both validation and test:        "
        f"{pair_overlap_count:,}"
    )

    print(
        f"Validation events on overlap pairs:  "
        f"{validation_events_on_overlap_pairs:,}"
    )

    print(
        f"Test events on overlap pairs:        "
        f"{test_events_on_overlap_pairs:,}"
    )

    print(
        f"Funding rounds represented in both:  "
        f"{len(round_overlap):,}"
    )

    # -------------------------------------------------------------------------
    # 15. Reproducibility metadata
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "REPRODUCTION METADATA"
    )
    separator("-")

    print(
        f"Split type:              "
        f"random event-level"
    )

    print(
        f"Stratified:              "
        f"No"
    )

    print(
        f"Validation fraction:     "
        f"{VALIDATION_FRACTION:.2%}"
    )

    print(
        f"Validation integer rule: "
        f"floor(N * validation_fraction)"
    )

    print(
        f"Random seed:             "
        f"{SPLIT_SEED}"
    )

    print(
        """
The random seed is reproduction-specific because the ITRS paper
does not report one.

Pair and funding-round overlap are NOT removed because doing so would
change the paper's stated event-level random split protocol.
"""
    )

    # -------------------------------------------------------------------------
    # 16. Final consistency checks
    # -------------------------------------------------------------------------

    separator("-")
    print(
        "GLOBAL CONSISTENCY CHECKS"
    )
    separator("-")

    print(
        f"T60 input events:        "
        f"{EXPECTED_T60_ROWS:,}"
    )

    print(
        f"Validation events:       "
        f"{len(validation):,}"
    )

    print(
        f"Test events:             "
        f"{len(test):,}"
    )

    print(
        f"Validation + test:       "
        f"{len(validation) + len(test):,}"
    )

    if t60["interaction_id"].nunique() != EXPECTED_T60_ROWS:
        raise ValueError(
            "Split output lost or duplicated interaction IDs."
        )

    separator()
    print(
        "PHASE 2.3.3 SPLIT COMPLETE"
    )
    separator()

    print(f"""
Outputs written to:

{SPLIT_OUTPUT}
{ASSIGNMENT_OUTPUT}
{SUMMARY_OUTPUT}
{PAIR_OVERLAP_OUTPUT}
{STATUS_OUTPUT}

No event was removed from T60.
No cold-start entity was removed.
No historical-repeat pair was removed.
No negative candidates have been sampled yet.

Next:
Phase 2.3.4 — Final temporal-split integrity audit and Phase-2 closure.
""")


if __name__ == "__main__":
    main()