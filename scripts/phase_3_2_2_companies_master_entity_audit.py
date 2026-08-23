from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PHASE 3.2.2 — COMPANIES MASTER-ENTITY HYPOTHESIS AUDIT
# =============================================================================
#
# HYPOTHESIS
# ----------
#
# The Crunchbase companies export may act as a master registry for
# organization-like entities, including investors and startups.
#
# This script tests that hypothesis using IDs only.
#
# It DOES NOT assume the hypothesis is true in advance.
#
# =============================================================================


RAW_DIR = Path("data/raw")

PHASE1_PATH = Path(
    "data/processed/interactions.parquet"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_3/audits"
)

FIGURE_DIR = Path(
    "data/experimental/phase_3/figures"
)


SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "companies_master_entity_coverage_summary.csv"
)

COMPANY_CHUNK_OUTPUT = (
    OUTPUT_DIR
    / "companies_chunk_global_id_audit.csv"
)

UNMATCHED_INVESTOR_OUTPUT = (
    OUTPUT_DIR
    / "investor_ids_not_in_companies.csv"
)

FIGURE_OUTPUT = (
    FIGURE_DIR
    / "companies_master_entity_id_coverage.png"
)


def separator(char="=", width=120):
    print(char * width)


def pct(numerator, denominator):

    if denominator == 0:
        return np.nan

    return (
        numerator
        / denominator
        * 100
    )


def normalized_string_set(series):

    return set(
        series
        .dropna()
        .astype("string")
        .str.strip()
        .loc[
            lambda s:
                s.ne("")
        ]
        .tolist()
    )


def main():

    separator()

    print(
        "PHASE 3.2.2 — "
        "COMPANIES MASTER-ENTITY HYPOTHESIS AUDIT"
    )

    separator()

    # =========================================================================
    # 1. Discover company chunks
    # =========================================================================

    company_files = sorted(
        RAW_DIR.glob(
            "companies*.csv"
        )
    )

    investor_files = sorted(
        RAW_DIR.glob(
            "CRUNCHBASE_investor*.csv"
        )
    )

    if not company_files:

        raise FileNotFoundError(
            "No companies*.csv files found."
        )

    if len(investor_files) != 1:

        raise ValueError(
            "Expected exactly one "
            "CRUNCHBASE_investor*.csv file. "
            f"Found {len(investor_files)}."
        )

    investor_path = (
        investor_files[0]
    )

    print(
        f"\nCompany chunks found: "
        f"{len(company_files)}"
    )

    print(
        f"Investor file:        "
        f"{investor_path.name}"
    )

    # =========================================================================
    # 2. Inspect actual schemas
    # =========================================================================

    company_header = (
        pd.read_csv(
            company_files[0],
            nrows=0,
        )
        .columns
        .tolist()
    )

    investor_header = (
        pd.read_csv(
            investor_path,
            nrows=0,
        )
        .columns
        .tolist()
    )

    separator("-")

    print(
        "COMPANIES SCHEMA"
    )

    separator("-")

    for column in company_header:
        print(
            f"  - {column}"
        )

    separator("-")

    print(
        "INVESTOR SCHEMA"
    )

    separator("-")

    for column in investor_header:
        print(
            f"  - {column}"
        )

    if "id" not in company_header:

        raise ValueError(
            "Companies files do not contain "
            "an exact 'id' column."
        )

    if "id" not in investor_header:

        raise ValueError(
            "Investor file does not contain "
            "an exact 'id' column."
        )

    # =========================================================================
    # 3. Load all company IDs
    # =========================================================================

    all_company_ids = []

    chunk_rows = []

    for path in company_files:

        chunk = pd.read_csv(
            path,
            usecols=["id"],
            dtype={
                "id": "string"
            },
        )

        ids = (
            chunk["id"]
            .dropna()
            .str.strip()
        )

        ids = ids[
            ids.ne("")
        ]

        unique_ids = (
            ids.nunique()
        )

        duplicate_inside_chunk = (
            len(ids)
            - unique_ids
        )

        chunk_rows.append(
            {
                "file_name": (
                    path.name
                ),

                "rows": (
                    len(chunk)
                ),

                "nonmissing_ids": (
                    len(ids)
                ),

                "unique_ids_inside_chunk": (
                    unique_ids
                ),

                "duplicate_ids_inside_chunk": (
                    duplicate_inside_chunk
                ),
            }
        )

        all_company_ids.append(
            ids
        )

    company_ids_series = (
        pd.concat(
            all_company_ids,
            ignore_index=True,
        )
    )

    company_total_rows = len(
        company_ids_series
    )

    company_unique_ids = (
        company_ids_series
        .nunique()
    )

    company_cross_file_duplicates = (
        company_total_rows
        - company_unique_ids
    )

    company_id_set = set(
        company_ids_series
        .unique()
    )

    # =========================================================================
    # 4. Investor ID coverage
    # =========================================================================

    investor_df = pd.read_csv(
        investor_path,
        usecols=["id"],
        dtype={
            "id": "string"
        },
    )

    investor_ids = (
        investor_df["id"]
        .dropna()
        .str.strip()
    )

    investor_ids = investor_ids[
        investor_ids.ne("")
    ]

    investor_unique_ids = set(
        investor_ids.unique()
    )

    investor_ids_in_companies = (
        investor_unique_ids
        & company_id_set
    )

    investor_ids_not_in_companies = (
        investor_unique_ids
        - company_id_set
    )

    # =========================================================================
    # 5. Phase-1 canonical ID coverage
    # =========================================================================

    phase1 = pd.read_parquet(
        PHASE1_PATH,
        columns=[
            "investor_id",
            "startup_id",
        ],
    )

    canonical_investor_ids = (
        normalized_string_set(
            phase1[
                "investor_id"
            ]
        )
    )

    canonical_startup_ids = (
        normalized_string_set(
            phase1[
                "startup_id"
            ]
        )
    )

    canonical_investors_in_companies = (
        canonical_investor_ids
        & company_id_set
    )

    canonical_startups_in_companies = (
        canonical_startup_ids
        & company_id_set
    )

    canonical_investors_in_investor_file = (
        canonical_investor_ids
        & investor_unique_ids
    )

    # =========================================================================
    # 6. Summary
    # =========================================================================

    summary_rows = [
        {
            "population": (
                "companies_all_rows"
            ),
            "unique_ids": (
                company_unique_ids
            ),
            "matched_to_companies": (
                company_unique_ids
            ),
            "coverage_pct": 100.0,
        },
        {
            "population": (
                "investor_file_ids"
            ),
            "unique_ids": (
                len(
                    investor_unique_ids
                )
            ),
            "matched_to_companies": (
                len(
                    investor_ids_in_companies
                )
            ),
            "coverage_pct": pct(
                len(
                    investor_ids_in_companies
                ),
                len(
                    investor_unique_ids
                ),
            ),
        },
        {
            "population": (
                "phase1_canonical_investor_ids"
            ),
            "unique_ids": (
                len(
                    canonical_investor_ids
                )
            ),
            "matched_to_companies": (
                len(
                    canonical_investors_in_companies
                )
            ),
            "coverage_pct": pct(
                len(
                    canonical_investors_in_companies
                ),
                len(
                    canonical_investor_ids
                ),
            ),
        },
        {
            "population": (
                "phase1_canonical_startup_ids"
            ),
            "unique_ids": (
                len(
                    canonical_startup_ids
                )
            ),
            "matched_to_companies": (
                len(
                    canonical_startups_in_companies
                )
            ),
            "coverage_pct": pct(
                len(
                    canonical_startups_in_companies
                ),
                len(
                    canonical_startup_ids
                ),
            ),
        },
        {
            "population": (
                "phase1_investor_ids_in_investor_file"
            ),
            "unique_ids": (
                len(
                    canonical_investor_ids
                )
            ),
            "matched_to_companies": (
                len(
                    canonical_investors_in_investor_file
                )
            ),
            "coverage_pct": pct(
                len(
                    canonical_investors_in_investor_file
                ),
                len(
                    canonical_investor_ids
                ),
            ),
        },
    ]

    summary = pd.DataFrame(
        summary_rows
    )

    chunk_summary = pd.DataFrame(
        chunk_rows
    )

    # =========================================================================
    # 7. Save unmatched investor IDs
    # =========================================================================

    unmatched_investors = pd.DataFrame(
        {
            "investor_id_not_in_companies":
                sorted(
                    investor_ids_not_in_companies
                )
        }
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    chunk_summary.to_csv(
        COMPANY_CHUNK_OUTPUT,
        index=False,
    )

    unmatched_investors.to_csv(
        UNMATCHED_INVESTOR_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 8. Coverage figure
    # =========================================================================

    plot_data = (
        summary[
            summary[
                "population"
            ]
            != "companies_all_rows"
        ]
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        plot_data[
            "population"
        ],
        plot_data[
            "coverage_pct"
        ],
    )

    ax.set_ylim(
        0,
        105,
    )

    ax.set_ylabel(
        "Coverage in reference population (%)"
    )

    ax.set_title(
        "Crunchbase Master-Entity ID Coverage"
    )

    ax.tick_params(
        axis="x",
        rotation=25,
    )

    fig.tight_layout()

    fig.savefig(
        FIGURE_OUTPUT,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    # =========================================================================
    # 9. Terminal report
    # =========================================================================

    separator("-")

    print(
        "GLOBAL COMPANIES-ID INTEGRITY"
    )

    separator("-")

    print(
        f"Company chunks:                    "
        f"{len(company_files):,}"
    )

    print(
        f"Company ID rows across chunks:     "
        f"{company_total_rows:,}"
    )

    print(
        f"Globally unique company IDs:       "
        f"{company_unique_ids:,}"
    )

    print(
        f"Cross-file duplicate company IDs:  "
        f"{company_cross_file_duplicates:,}"
    )

    separator("-")

    print(
        "MASTER-ENTITY COVERAGE"
    )

    separator("-")

    print(
        summary.to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}",
        )
    )

    separator("-")

    print(
        "INVESTOR -> COMPANIES COVERAGE"
    )

    separator("-")

    print(
        f"Unique investor-file IDs:          "
        f"{len(investor_unique_ids):,}"
    )

    print(
        f"Investor IDs found in companies:   "
        f"{len(investor_ids_in_companies):,}"
    )

    print(
        f"Investor IDs NOT in companies:     "
        f"{len(investor_ids_not_in_companies):,}"
    )

    print(
        f"Coverage:                           "
        f"{pct(len(investor_ids_in_companies), len(investor_unique_ids)):.4f}%"
    )

    separator("-")

    print(
        "PHASE-1 CANONICAL COVERAGE"
    )

    separator("-")

    print(
        f"Canonical investors:               "
        f"{len(canonical_investor_ids):,}"
    )

    print(
        f"Canonical investors in companies:  "
        f"{len(canonical_investors_in_companies):,}"
        f" "
        f"({pct(len(canonical_investors_in_companies), len(canonical_investor_ids)):.4f}%)"
    )

    print(
        f"Canonical investors in investor file:"
        f" {len(canonical_investors_in_investor_file):,}"
        f" "
        f"({pct(len(canonical_investors_in_investor_file), len(canonical_investor_ids)):.4f}%)"
    )

    print(
        f"\nCanonical startups:                "
        f"{len(canonical_startup_ids):,}"
    )

    print(
        f"Canonical startups in companies:   "
        f"{len(canonical_startups_in_companies):,}"
        f" "
        f"({pct(len(canonical_startups_in_companies), len(canonical_startup_ids)):.4f}%)"
    )

    separator()

    print(
        "PHASE 3.2.2 MASTER-ENTITY AUDIT COMPLETE"
    )

    separator()

    print(
        f"""
Outputs written to:

{SUMMARY_OUTPUT}
{COMPANY_CHUNK_OUTPUT}
{UNMATCHED_INVESTOR_OUTPUT}

Figure written to:

{FIGURE_OUTPUT}


INTERPRETATION RULE

Do NOT conclude that companies is the master organizational entity table
solely because coverage is high.

We will interpret:

- global companies-ID uniqueness,
- investor -> companies coverage,
- Phase-1 investor coverage,
- Phase-1 startup coverage,

together.

The next step will depend on these observed results.
"""
    )


if __name__ == "__main__":
    main()