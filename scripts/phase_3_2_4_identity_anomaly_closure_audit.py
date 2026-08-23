from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PHASE 3.2.4 — IDENTITY ANOMALY CLOSURE AUDIT
# =============================================================================
#
# PURPOSE
# -------
#
# Close the two small identity anomalies remaining after Phase 3.2.3:
#
# 1. Company IDs duplicated across export chunks with inconsistent
#    identifying attributes.
#
# 2. Canonical investment events where:
#
#        investor_id == startup_id
#
# No records are removed or corrected in this audit.
# =============================================================================


RAW_DIR = Path(
    "data/raw"
)

INTERACTIONS_PATH = Path(
    "data/processed/interactions.parquet"
)

DUPLICATE_DETAILS_PATH = Path(
    "data/experimental/phase_3/audits/"
    "companies_cross_chunk_duplicate_details.csv"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_3/audits"
)

FIGURE_DIR = Path(
    "data/experimental/phase_3/figures"
)


ANOMALOUS_COMPANY_OUTPUT = (
    OUTPUT_DIR
    / "companies_inconsistent_duplicate_records.csv"
)

SELF_EVENT_OUTPUT = (
    OUTPUT_DIR
    / "self_identity_investment_events.parquet"
)

SELF_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "self_identity_investment_summary.csv"
)

SELF_TYPE_OUTPUT = (
    OUTPUT_DIR
    / "self_identity_investment_type_distribution.csv"
)

SELF_RESOLUTION_OUTPUT = (
    OUTPUT_DIR
    / "self_identity_resolution_quality_distribution.csv"
)

SELF_ENTITY_OUTPUT = (
    OUTPUT_DIR
    / "self_identity_entity_summary.csv"
)

SELF_TYPE_FIGURE = (
    FIGURE_DIR
    / "self_identity_investment_type_distribution.png"
)


EXPECTED_SELF_EVENTS = 110


def separator(char="=", width=120):
    print(char * width)


def pct(num, den):

    if den == 0:
        return float("nan")

    return (
        num
        / den
        * 100
    )


def clean_string(series):

    return (
        series
        .astype("string")
        .str.strip()
    )


def main():

    separator()

    print(
        "PHASE 3.2.4 — "
        "IDENTITY ANOMALY CLOSURE AUDIT"
    )

    separator()

    # =========================================================================
    # 1. Locate inconsistent cross-chunk company IDs
    # =========================================================================

    duplicate_details = pd.read_csv(
        DUPLICATE_DETAILS_PATH,
        dtype={
            "company_id": "string"
        },
    )

    inconsistency_cols = [
        column
        for column in [
            "name_consistent",
            "link_consistent",
            "website_consistent",
            "company_type_consistent",
        ]
        if column
        in duplicate_details.columns
    ]

    inconsistent_mask = pd.Series(
        False,
        index=duplicate_details.index,
    )

    for column in inconsistency_cols:

        values = (
            duplicate_details[
                column
            ]
        )

        if values.dtype == object:

            inconsistent = (
                values
                .astype("string")
                .str.lower()
                .eq("false")
            )

        else:

            inconsistent = (
                values
                == False
            )

        inconsistent_mask = (
            inconsistent_mask
            |
            inconsistent
        )

    inconsistent_ids = set(
        duplicate_details.loc[
            inconsistent_mask,
            "company_id",
        ]
        .dropna()
        .astype("string")
        .tolist()
    )

    separator("-")

    print(
        "INCONSISTENT CROSS-CHUNK COMPANY IDs"
    )

    separator("-")

    print(
        f"IDs requiring raw-record inspection: "
        f"{len(inconsistent_ids):,}"
    )

    # =========================================================================
    # 2. Reload raw records for those IDs
    # =========================================================================

    raw_anomalies = []

    company_files = sorted(
        RAW_DIR.glob(
            "companies*.csv"
        )
    )

    for path in company_files:

        header = (
            pd.read_csv(
                path,
                nrows=0,
            )
            .columns
            .tolist()
        )

        usecols = [
            column
            for column in [
                "id",
                "name",
                "link",
                "website",
                "company_type",
                "description",
                "operating_status",
                "founded_on",
            ]
            if column in header
        ]

        chunk = pd.read_csv(
            path,
            usecols=usecols,
            dtype="string",
            low_memory=False,
        )

        chunk["id"] = (
            clean_string(
                chunk["id"]
            )
        )

        subset = (
            chunk[
                chunk["id"]
                .isin(
                    inconsistent_ids
                )
            ]
            .copy()
        )

        if len(subset) > 0:

            subset[
                "source_file"
            ] = path.name

            raw_anomalies.append(
                subset
            )

    if raw_anomalies:

        anomalous_company_records = (
            pd.concat(
                raw_anomalies,
                ignore_index=True,
            )
            .sort_values(
                [
                    "id",
                    "source_file",
                ]
            )
        )

    else:

        anomalous_company_records = (
            pd.DataFrame()
        )

    print(
        "\nRaw duplicate records:"
    )

    if len(
        anomalous_company_records
    ) > 0:

        print(
            anomalous_company_records
            .to_string(
                index=False
            )
        )

    else:

        print(
            "None."
        )

    # =========================================================================
    # 3. Load canonical interactions
    # =========================================================================

    interactions = pd.read_parquet(
        INTERACTIONS_PATH
    )

    interactions[
        "investor_id"
    ] = (
        clean_string(
            interactions[
                "investor_id"
            ]
        )
    )

    interactions[
        "startup_id"
    ] = (
        clean_string(
            interactions[
                "startup_id"
            ]
        )
    )

    self_events = (
        interactions[
            interactions[
                "investor_id"
            ]
            .eq(
                interactions[
                    "startup_id"
                ]
            )
        ]
        .copy()
    )

    if len(
        self_events
    ) != EXPECTED_SELF_EVENTS:

        raise ValueError(
            f"Expected {EXPECTED_SELF_EVENTS} "
            f"self-ID events from Phase 3.2.3, "
            f"found {len(self_events)}."
        )

    # =========================================================================
    # 4. Core self-event diagnostics
    # =========================================================================

    self_events[
        "names_exact_match"
    ] = (
        clean_string(
            self_events[
                "investor_name"
            ]
        )
        .str.lower()
        .eq(
            clean_string(
                self_events[
                    "startup_name"
                ]
            )
            .str.lower()
        )
    )

    unique_self_entities = (
        self_events[
            "investor_id"
        ]
        .nunique()
    )

    unique_self_rounds = (
        self_events[
            "funding_round_id"
        ]
        .nunique()
    )

    exact_name_matches = int(
        self_events[
            "names_exact_match"
        ]
        .sum()
    )

    # =========================================================================
    # 5. Investment-type distribution
    # =========================================================================

    investment_type_distribution = (
        self_events[
            "investment_type"
        ]
        .fillna(
            "<missing>"
        )
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "investment_type"
        )
        .reset_index(
            name="interaction_count"
        )
    )

    investment_type_distribution[
        "interaction_share_pct"
    ] = (
        investment_type_distribution[
            "interaction_count"
        ]
        / len(
            self_events
        )
        * 100
    )

    # =========================================================================
    # 6. Resolution / parse-quality diagnostics
    # =========================================================================

    resolution_group_cols = [
        column
        for column in [
            "startup_resolution_method",
            "parse_quality_status",
            "canonical_name_looks_malformed",
        ]
        if column
        in self_events.columns
    ]

    if resolution_group_cols:

        resolution_quality = (
            self_events.groupby(
                resolution_group_cols,
                dropna=False,
                observed=True,
            )
            .size()
            .rename(
                "interaction_count"
            )
            .reset_index()
        )

        resolution_quality[
            "interaction_share_pct"
        ] = (
            resolution_quality[
                "interaction_count"
            ]
            / len(
                self_events
            )
            * 100
        )

    else:

        resolution_quality = (
            pd.DataFrame()
        )

    # =========================================================================
    # 7. Entity-level summary
    # =========================================================================

    entity_summary = (
        self_events.groupby(
            [
                "investor_id",
                "investor_name",
                "startup_name",
            ],
            dropna=False,
            observed=True,
        )
        .agg(
            self_event_count=(
                "interaction_id",
                "size",
            ),

            unique_funding_rounds=(
                "funding_round_id",
                "nunique",
            ),

            first_self_event=(
                "announced_on",
                "min",
            ),

            last_self_event=(
                "announced_on",
                "max",
            ),

            investment_types=(
                "investment_type",
                lambda x:
                    " | ".join(
                        sorted(
                            set(
                                x.dropna()
                                .astype(str)
                            )
                        )
                    ),
            ),

            all_names_exact_match=(
                "names_exact_match",
                "all",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "self_event_count",
                "investor_id",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    # =========================================================================
    # 8. Main summary
    # =========================================================================

    summary = pd.DataFrame(
        [
            {
                "metric":
                    "self_identity_interaction_events",
                "value":
                    len(
                        self_events
                    ),
            },
            {
                "metric":
                    "unique_self_identity_entities",
                "value":
                    unique_self_entities,
            },
            {
                "metric":
                    "unique_self_identity_funding_rounds",
                "value":
                    unique_self_rounds,
            },
            {
                "metric":
                    "events_with_exact_investor_startup_name_match",
                "value":
                    exact_name_matches,
            },
            {
                "metric":
                    "exact_name_match_share_pct",
                "value":
                    pct(
                        exact_name_matches,
                        len(
                            self_events
                        ),
                    ),
            },
            {
                "metric":
                    "earliest_self_identity_event",
                "value":
                    str(
                        self_events[
                            "announced_on"
                        ]
                        .min()
                    ),
            },
            {
                "metric":
                    "latest_self_identity_event",
                "value":
                    str(
                        self_events[
                            "announced_on"
                        ]
                        .max()
                    ),
            },
        ]
    )

    # =========================================================================
    # 9. Save outputs
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    anomalous_company_records.to_csv(
        ANOMALOUS_COMPANY_OUTPUT,
        index=False,
    )

    self_events.to_parquet(
        SELF_EVENT_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SELF_SUMMARY_OUTPUT,
        index=False,
    )

    investment_type_distribution.to_csv(
        SELF_TYPE_OUTPUT,
        index=False,
    )

    resolution_quality.to_csv(
        SELF_RESOLUTION_OUTPUT,
        index=False,
    )

    entity_summary.to_csv(
        SELF_ENTITY_OUTPUT,
        index=False,
    )

    # =========================================================================
    # 10. Figure
    # =========================================================================

    if len(
        investment_type_distribution
    ) > 0:

        plot_data = (
            investment_type_distribution
            .sort_values(
                "interaction_count",
                ascending=True,
            )
        )

        fig, ax = plt.subplots(
            figsize=(
                9,
                max(
                    4.5,
                    len(plot_data)
                    * 0.45,
                ),
            )
        )

        ax.barh(
            plot_data[
                "investment_type"
            ],
            plot_data[
                "interaction_count"
            ],
        )

        ax.set_xlabel(
            "Self-ID investment events"
        )

        ax.set_title(
            "Investment Types of investor_id = startup_id Events"
        )

        fig.tight_layout()

        fig.savefig(
            SELF_TYPE_FIGURE,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    # =========================================================================
    # 11. Terminal report
    # =========================================================================

    separator("-")

    print(
        "SELF-IDENTITY INVESTMENT SUMMARY"
    )

    separator("-")

    print(
        f"Self-ID events:                    "
        f"{len(self_events):,}"
    )

    print(
        f"Unique underlying IDs:             "
        f"{unique_self_entities:,}"
    )

    print(
        f"Unique funding rounds:             "
        f"{unique_self_rounds:,}"
    )

    print(
        f"Exact investor/startup name match: "
        f"{exact_name_matches:,} "
        f"({pct(exact_name_matches, len(self_events)):.2f}%)"
    )

    print(
        f"Date range:                        "
        f"{self_events['announced_on'].min()} "
        f"-> "
        f"{self_events['announced_on'].max()}"
    )

    separator("-")

    print(
        "INVESTMENT-TYPE DISTRIBUTION"
    )

    separator("-")

    print(
        investment_type_distribution
        .to_string(
            index=False,
            float_format=lambda x:
                f"{x:.2f}",
        )
    )

    separator("-")

    print(
        "RESOLUTION / PARSE QUALITY"
    )

    separator("-")

    if len(
        resolution_quality
    ) > 0:

        print(
            resolution_quality
            .to_string(
                index=False,
                float_format=lambda x:
                    f"{x:.2f}",
            )
        )

    else:

        print(
            "No resolution-quality columns available."
        )

    separator("-")

    print(
        "TOP SELF-IDENTITY ENTITIES"
    )

    separator("-")

    print(
        entity_summary
        .head(30)
        .to_string(
            index=False
        )
    )

    separator()

    print(
        "PHASE 3.2.4 AUDIT COMPLETE"
    )

    separator()

    print(
        f"""
Outputs written to:

{ANOMALOUS_COMPANY_OUTPUT}
{SELF_EVENT_OUTPUT}
{SELF_SUMMARY_OUTPUT}
{SELF_TYPE_OUTPUT}
{SELF_RESOLUTION_OUTPUT}
{SELF_ENTITY_OUTPUT}

Figure written to:

{SELF_TYPE_FIGURE}


NO DATA WAS REMOVED OR MODIFIED.

Next:
Phase 3.2.5 — Relationship-Source Dataset Discovery
"""
    )


if __name__ == "__main__":
    main()