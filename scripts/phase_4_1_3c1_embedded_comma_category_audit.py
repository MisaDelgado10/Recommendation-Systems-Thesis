from pathlib import Path

import pandas as pd


# =============================================================================
# PHASE 4.1.3c.1 — EMBEDDED-COMMA CATEGORY AUDIT
# =============================================================================

RAW_DIR = Path("data/raw")
NODE_PATH = Path(
    "data/experimental/phase_3/graph/nodes.parquet"
)

OUT_DIR = Path(
    "data/experimental/phase_4/audits"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

INVESTOR_PATH = (
    RAW_DIR / "CRUNCHBASE_investor_20260531_333726.csv"
)

COMPANY_PATHS = sorted(
    RAW_DIR.glob("companies*.csv")
)

EXPECTED_INVESTORS = 165_975
EXPECTED_STARTUPS = 311_589

CHUNK_SIZE = 200_000

HVAC_FULL = (
    "Heating, Ventilation, and Air Conditioning (HVAC)"
)

HVAC_FRAGMENTS = {
    "Heating",
    "Ventilation",
    "and Air Conditioning (HVAC)",
}


# =============================================================================
# Helpers
# =============================================================================

def banner(title):
    print()
    print("=" * 115)
    print(title)
    print("=" * 115)


def present_mask(series):
    text = series.astype("string").str.strip()

    return (
        series.notna()
        & text.notna()
        & text.ne("")
        & ~text.str.lower().isin(
            ["nan", "none", "null"]
        )
    )


def audit_source(
    df,
    source,
):
    banner(
        f"{source.upper()} — EMBEDDED-COMMA AUDIT"
    )

    work = df.loc[
        present_mask(df["categories"]),
        ["id", "categories"],
    ].copy()

    work["id"] = work["id"].astype(str)

    raw = (
        work["categories"]
        .astype("string")
        .str.strip()
    )

    # -------------------------------------------------------------------------
    # 1. Comma + space
    #
    # Normal observed Crunchbase list delimiters generally look like:
    #
    #   Software,Health Care,FinTech
    #
    # Embedded punctuation may instead appear as:
    #
    #   Heating, Ventilation, and Air Conditioning (HVAC)
    #
    # We AUDIT this distinction here. We do not assume it yet.
    # -------------------------------------------------------------------------

    comma_space_mask = raw.str.contains(
        ", ",
        regex=False,
    )

    comma_space_rows = work.loc[
        comma_space_mask
    ].copy()

    print(
        f"Nonempty category rows:                 "
        f"{len(work):,}"
    )

    print(
        f"Rows containing literal ', ':          "
        f"{len(comma_space_rows):,}"
    )

    print(
        f"Unique raw values containing ', ':      "
        f"{comma_space_rows['categories'].nunique():,}"
    )


    # -------------------------------------------------------------------------
    # 2. Exact HVAC phrase
    # -------------------------------------------------------------------------

    hvac_mask = raw.str.contains(
        HVAC_FULL,
        regex=False,
    )

    hvac_rows = work.loc[
        hvac_mask
    ].copy()

    print()
    print(
        f"Rows containing full HVAC category:     "
        f"{len(hvac_rows):,}"
    )

    print(
        f"Unique entities containing full HVAC:   "
        f"{hvac_rows['id'].nunique():,}"
    )


    # -------------------------------------------------------------------------
    # 3. Comma-space rows NOT explained by HVAC
    # -------------------------------------------------------------------------

    unexplained_mask = (
        comma_space_mask
        & ~hvac_mask
    )

    unexplained = work.loc[
        unexplained_mask
    ].copy()

    print()
    print(
        f"Comma-space rows NOT explained by HVAC: "
        f"{len(unexplained):,}"
    )

    print(
        f"Unique unexplained raw values:           "
        f"{unexplained['categories'].nunique():,}"
    )


    # -------------------------------------------------------------------------
    # 4. Provisional fragment co-occurrence
    # -------------------------------------------------------------------------

    exploded = (
        work
        .assign(
            token=lambda x:
                x["categories"]
                .astype("string")
                .str.split(",")
        )
        .explode("token")
        .reset_index(drop=True)
    )

    exploded["token"] = (
        exploded["token"]
        .astype("string")
        .str.strip()
    )

    entity_token_sets = (
        exploded
        .groupby("id")["token"]
        .agg(set)
    )

    entity_has_artifact = (
        entity_token_sets
        .apply(
            lambda tokens:
                "and Air Conditioning (HVAC)"
                in tokens
        )
    )

    artifact_ids = set(
        entity_has_artifact[
            entity_has_artifact
        ].index
    )

    hvac_ids = set(
        hvac_rows["id"].astype(str)
    )

    print()
    print(
        f"Entities with provisional artifact token:"
        f" {len(artifact_ids):,}"
    )

    print(
        f"Artifact entities also containing exact "
        f"raw HVAC phrase: "
        f"{len(artifact_ids & hvac_ids):,}"
    )

    print(
        f"Artifact entities WITHOUT exact HVAC:    "
        f"{len(artifact_ids - hvac_ids):,}"
    )

    print(
        f"Exact HVAC entities missing artifact:    "
        f"{len(hvac_ids - artifact_ids):,}"
    )


    # -------------------------------------------------------------------------
    # 5. Do Heating / Ventilation appear independently?
    # -------------------------------------------------------------------------

    fragment_records = []

    for fragment in sorted(HVAC_FRAGMENTS):

        ids_with_fragment = set(
            exploded.loc[
                exploded["token"].eq(fragment),
                "id",
            ].astype(str)
        )

        ids_outside_hvac = (
            ids_with_fragment - hvac_ids
        )

        fragment_records.append(
            {
                "source": source,
                "fragment": fragment,
                "entities_with_fragment":
                    len(ids_with_fragment),
                "entities_with_fragment_and_hvac":
                    len(ids_with_fragment & hvac_ids),
                "entities_with_fragment_outside_hvac":
                    len(ids_outside_hvac),
            }
        )

    fragment_df = pd.DataFrame(
        fragment_records
    )

    print()
    print("HVAC PROVISIONAL FRAGMENT AUDIT")
    print("-" * 115)

    print(
        fragment_df.to_string(
            index=False
        )
    )


    # -------------------------------------------------------------------------
    # 6. Print unexplained comma-space examples
    # -------------------------------------------------------------------------

    if not unexplained.empty:

        print()
        print(
            "UNEXPLAINED COMMA-SPACE RAW VALUES "
            "(first 50 unique)"
        )
        print("-" * 115)

        examples = (
            unexplained["categories"]
            .drop_duplicates()
            .head(50)
        )

        for value in examples:
            print(value)


    # -------------------------------------------------------------------------
    # 7. Save diagnostics
    # -------------------------------------------------------------------------

    comma_space_rows = comma_space_rows.copy()
    comma_space_rows["source"] = source

    unexplained = unexplained.copy()
    unexplained["source"] = source

    return (
        {
            "source": source,
            "nonempty_category_rows": len(work),
            "rows_with_comma_space":
                len(comma_space_rows),
            "rows_with_exact_hvac":
                len(hvac_rows),
            "comma_space_rows_not_hvac":
                len(unexplained),
            "artifact_entities":
                len(artifact_ids),
            "artifact_without_exact_hvac":
                len(artifact_ids - hvac_ids),
            "hvac_without_artifact":
                len(hvac_ids - artifact_ids),
        },
        fragment_df,
        comma_space_rows,
        unexplained,
    )


# =============================================================================
# Load frozen universe
# =============================================================================

banner(
    "PHASE 4.1.3c.1 — "
    "EMBEDDED-COMMA CATEGORY AUDIT"
)

nodes = pd.read_parquet(
    NODE_PATH,
    columns=[
        "node_type",
        "raw_entity_id",
    ],
)

investor_ids = set(
    nodes.loc[
        nodes["node_type"].eq("investor"),
        "raw_entity_id",
    ].astype(str)
)

startup_ids = set(
    nodes.loc[
        nodes["node_type"].eq("startup"),
        "raw_entity_id",
    ].astype(str)
)

assert len(investor_ids) == EXPECTED_INVESTORS
assert len(startup_ids) == EXPECTED_STARTUPS

print(
    f"Frozen Investors: {len(investor_ids):,}"
)

print(
    f"Frozen Startups:  {len(startup_ids):,}"
)


# =============================================================================
# Investor categories
# =============================================================================

banner("LOADING INVESTORS")

investor_df = pd.read_csv(
    INVESTOR_PATH,
    usecols=[
        "id",
        "categories",
    ],
    dtype={
        "id": "string",
    },
    low_memory=False,
)

investor_df["id"] = (
    investor_df["id"].astype(str)
)

investor_df = investor_df.loc[
    investor_df["id"].isin(investor_ids)
].copy()

assert (
    investor_df["id"].nunique()
    == EXPECTED_INVESTORS
)


# =============================================================================
# Startup categories
# =============================================================================

banner("LOADING STARTUPS")

startup_parts = []

for i, path in enumerate(
    COMPANY_PATHS,
    start=1,
):

    print(
        f"[{i:02d}/{len(COMPANY_PATHS):02d}] "
        f"{path.name}"
    )

    for chunk in pd.read_csv(
        path,
        usecols=[
            "id",
            "categories",
        ],
        dtype={
            "id": "string",
        },
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):

        chunk["id"] = (
            chunk["id"].astype(str)
        )

        matched = chunk.loc[
            chunk["id"].isin(startup_ids)
        ].copy()

        if not matched.empty:
            startup_parts.append(
                matched
            )

startup_df = pd.concat(
    startup_parts,
    ignore_index=True,
)

# Consume Phase-4.1.3b duplicate-field consistency audit.
startup_df = (
    startup_df
    .drop_duplicates(
        subset=["id"],
        keep="first",
    )
    .copy()
)

assert (
    startup_df["id"].nunique()
    == EXPECTED_STARTUPS
)


# =============================================================================
# Run audits
# =============================================================================

results = []

fragment_frames = []
comma_space_frames = []
unexplained_frames = []

for source, df in [
    ("investor", investor_df),
    ("startup", startup_df),
]:

    (
        summary,
        fragments,
        comma_space,
        unexplained,
    ) = audit_source(
        df,
        source,
    )

    results.append(summary)
    fragment_frames.append(fragments)
    comma_space_frames.append(
        comma_space
    )
    unexplained_frames.append(
        unexplained
    )


# =============================================================================
# Consolidate and save
# =============================================================================

summary_df = pd.DataFrame(
    results
)

fragment_df = pd.concat(
    fragment_frames,
    ignore_index=True,
)

comma_space_df = pd.concat(
    comma_space_frames,
    ignore_index=True,
)

unexplained_df = pd.concat(
    unexplained_frames,
    ignore_index=True,
)


summary_path = (
    OUT_DIR
    / "description_category_embedded_comma_summary.csv"
)

fragment_path = (
    OUT_DIR
    / "description_category_hvac_fragment_audit.csv"
)

comma_space_path = (
    OUT_DIR
    / "description_category_comma_space_rows.csv"
)

unexplained_path = (
    OUT_DIR
    / "description_category_unexplained_comma_space_rows.csv"
)


summary_df.to_csv(
    summary_path,
    index=False,
)

fragment_df.to_csv(
    fragment_path,
    index=False,
)

comma_space_df.to_csv(
    comma_space_path,
    index=False,
)

unexplained_df.to_csv(
    unexplained_path,
    index=False,
)


# =============================================================================
# Final summary
# =============================================================================

banner(
    "PHASE 4.1.3c.1 SUMMARY"
)

print(
    summary_df.to_string(
        index=False
    )
)

print()
print("Outputs:")

for path in [
    summary_path,
    fragment_path,
    comma_space_path,
    unexplained_path,
]:
    print(f"  {path}")

print()
print(
    "Parser remains UNFROZEN until this "
    "targeted audit is reviewed."
)

print(
    "\nPHASE 4.1.3c.1 STATUS: COMPLETE"
)