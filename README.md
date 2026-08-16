## Data Reconstruction Status

### Phase 1 — Canonical Investment Interactions

**Status:** Complete  
**Final output:** `data/processed/interactions.parquet`

Phase 1 reconstructs high-confidence investor–organization investment events from the raw Crunchbase funding export.

The raw funding data does not directly provide a clean investor identifier for each investor mention, and funded organization names are not always unique. Therefore, Phase 1 implements a conservative entity-resolution pipeline before constructing model-ready investment events.

### Final Phase 1 Dataset

| Metric | Value |
|---|---:|
| Canonical interactions | 1,208,051 |
| Unique investors | 165,975 |
| Unique funded organizations | 311,589 |
| Unique funding rounds | 560,270 |
| Unique investor–organization pairs | 991,291 |
| Pairs with more than one investment event | 153,310 |
| Maximum events for one pair | 45 |
| Same-day repeated investor–organization groups | 2,570 |
| Exact duplicate interaction groups | 0 |
| Date range | 1945-01-01 to 2026-06-01 |

The canonical interaction identifier is:

`interaction_id = funding_round_id::investor_id`

One canonical interaction represents one uniquely resolved investor participating in one uniquely identified Crunchbase funding round for one resolved funded organization.

### Phase 1 Pipeline

```text
Raw Crunchbase funding data
        |
        v
Funding and date audit
        |
        v
Investor-field syntax audit
        |
        v
Canonical investor dictionary segmentation
        |
        v
Investor ID resolution
        |
        v
investor_mentions.parquet
        |
        v
Company exact-name resolution
        |
        v
Company identity + website audit
        |
        v
Website-assisted company disambiguation
        |
        v
startup_resolution_by_round.parquet
        |
        v
Join investor + startup identities
        |
        v
candidate_interactions.parquet
        |
        v
Duplicate / same-day event audit
        |
        v
data/processed/interactions.parquet