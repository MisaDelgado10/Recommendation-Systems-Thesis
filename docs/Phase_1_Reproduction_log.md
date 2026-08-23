```markdown
# Reproduction Log

## Phase 1 — Data Reconstruction and Canonical Investment Events

**Completed:** 2026-08-16  
**Status:** COMPLETE  
**Final dataset:** `data/processed/interactions.parquet`

### Objective

Transform the raw Crunchbase funding export into a high-confidence investor–organization interaction dataset suitable for reproducing the interaction component of ITRS.

The conceptual investment event is:

`e(i,j,k) = (investor_i, organization_j, time_k)`

The Phase 1 canonical event is defined as one uniquely resolved investor participating in one uniquely identified Crunchbase funding round for one resolved funded organization.

---

### Main Inputs

Funding data:

`data/raw/CRUNCHBASE_funding_20260531_797177.csv`

Investor master:

`data/raw/CRUNCHBASE_investor_20260531_333726.csv`

Company master:

`data/raw/companies*.csv`

---

### Raw Funding Audit

- Funding-round rows: 797,177
- Funding rounds with investor information: 602,867
- Funding rounds without investor information: 194,310
- Missing funding-round IDs: 0
- Missing announced dates: 0
- Missing organization names: 1
- Unparseable dates: 0
- Unique organization names: 393,206
- Earliest raw announced date: 1900-01-01
- Latest raw announced date: 2026-06-20

Important observation:

The funding filename contains `20260531`, but the actual data extends through `2026-06-20`.

Therefore, temporal cutoffs must be determined from the actual `announced_on` distribution rather than from the export filename.

---

### Investor Master Audit

- Investor rows: 333,726
- Unique investor IDs: 333,726
- Missing investor names: 1
- Canonical normalized investor names: 320,990
- Duplicate-name groups: 9,591
- Entities in duplicate-name groups: 22,326
- Canonical names containing commas: 1,491
- Canonical names containing semicolons: 3
- Canonical names containing pipes: 45

A naïve comma parser was rejected because canonical investor names can themselves contain commas.

Naïve comma splitting produced:

- Candidate tokens: 1,339,156
- Exact master matches: 1,332,255
- Unmatched fragments: 6,901

Frequent broken fragments included corporate suffixes such as `LLC`, `Inc.`, and `Ltd.`.

---

### Investor Parsing Method

Investor-string parsing was reformulated as exact dictionary segmentation against canonical Crunchbase investor names.

Row-level result:

- Funding rows with investor data: 602,867
- Exactly one valid segmentation: 602,850
- Multiple valid segmentations: 0
- No valid segmentation: 17

All 17 unresolved rows contained the same blocking source name:

`HALA Ventures`

HALA Ventures was preserved as an unmatched source investor rather than assigned an invented ID.

Partial recovery produced:

- 85 known investor mentions
- 17 unmatched HALA Ventures mentions

Malformed-looking canonical investor names were preserved rather than silently corrected.

Observed master anomalies:

- Names starting with comma: 0
- Names ending with comma: 92
- Names containing consecutive commas: 2
- Funding rows containing empty comma chunks: 103

---

### Investor Mention Dataset

Output:

`data/interim/investor_mentions.parquet`

Total normalized investor mentions:

1,335,550

Investor ID resolution:

| Status | Mentions |
|---|---:|
| resolved_unique | 1,221,062 |
| ambiguous_multiple_ids | 114,471 |
| unmatched_master | 17 |

Approximately 91.43% of investor mentions have a uniquely resolved investor ID.

No arbitrary investor ID is assigned when a canonical investor name maps to multiple Crunchbase entities.

---

### Company Identity Audit

Combined company identity master:

- Raw identity rows: 4,689,624
- Unique company IDs: 4,675,397
- Missing company IDs: 0
- Missing company names: 5
- Rows with website: 4,600,253
- Unique exact websites: 4,575,861
- Company IDs appearing more than once: 14,227
- Repeated IDs with conflicting names: 1
- Repeated IDs with conflicting websites: 0

The one conflicting-name ID appeared as `Loyalte` / `Loialte` but had the same company ID and exact website.

---

### Exact Company-Name Resolution

Unique funded organization names:

| Status | Count |
|---|---:|
| resolved_unique | 283,704 |
| ambiguous_multiple_ids | 32,306 |
| unmatched_master | 4,094 |

Investor-mention-level impact:

| Status | Mentions |
|---|---:|
| resolved_unique | 1,089,327 |
| ambiguous_multiple_ids | 239,132 |
| unmatched_master | 7,091 |

Exact company names alone resolved approximately 81.56% of investor mentions.

Generic names can correspond to many different companies. Examples observed in the company master included:

- Atlas: 70 IDs
- Spark: 69 IDs
- Synergy: 67 IDs
- Bloom: 64 IDs
- Aurora: 43 IDs
- Scout: 43 IDs

---

### Website Validation

Exact website mappings in the company identity master:

- Websites mapping to one company ID: 4,566,282
- Websites mapping to multiple company IDs: 9,579

Website evidence was validated on funding rounds where both the exact company name and exact website independently mapped to one company ID.

Validation result:

- Agreements: 495,869
- Conflicts: 0

Therefore, exact website equality was accepted as a strong secondary identity signal.

For ambiguous company names:

- Website selected an existing name candidate: 83,246 funding rounds
- Website selected an ID outside the name candidates: 4

Only the 83,246 consistent cases were accepted.

There were 38 unmatched-name funding rounds with a unique exact-website candidate. These were preserved as unresolved website-only candidates rather than automatically accepted.

---

### Final Startup / Organization Resolution

Funding-round level:

| Resolution method | Funding rounds |
|---|---:|
| resolved_by_name | 510,935 |
| resolved_by_name_and_website | 83,246 |
| unresolved_unmatched_name | 5,649 |
| unresolved_ambiguous_name | 2,995 |
| unresolved_website_only_candidate | 38 |
| unresolved_name_website_conflict | 4 |

Summary:

- Funding rounds with investor data: 602,867
- Resolved funding rounds: 594,181
- Unresolved funding rounds: 8,686
- Unique resolved organizations: 325,246
- Startup-resolved investor mentions: 1,321,416

Startup identity is resolved for approximately 98.94% of investor mentions.

---

### Interaction Eligibility

After joining investor and startup identities:

| Eligibility status | Mentions |
|---|---:|
| eligible | 1,208,051 |
| unresolved_investor | 113,365 |
| unresolved_startup | 13,011 |
| unresolved_investor_and_startup | 1,123 |

All eligible rows contain:

- funding_round_id
- investor_id
- startup_id
- announced_on

The final interaction retention rate is approximately 90.45% of normalized investor mentions.

---

### Duplicate Audit

Exact duplicate key:

`investor_id + startup_id + funding_round_id`

Exact duplicate groups:

0

Same investor + same startup + same date groups:

2,570

Rows belonging to those groups:

5,146

Extra rows beyond one per group:

2,576

Same-day groups with multiple investment types:

2,501

Same-day groups with one investment type:

69

Same-day groups containing repeated funding-round IDs:

0

Decision:

Same-day events with different funding-round IDs are preserved as separate source events.

---

### Final Canonical Dataset

Output:

`data/processed/interactions.parquet`

Final statistics:

| Metric | Value |
|---|---:|
| Canonical interactions | 1,208,051 |
| Unique investors | 165,975 |
| Unique funded organizations | 311,589 |
| Unique funding rounds | 560,270 |
| Unique investor–organization pairs | 991,291 |
| Pairs with more than one event | 153,310 |
| Maximum events for one pair | 45 |
| Earliest event | 1945-01-01 |
| Latest event | 2026-06-01 |

Canonical interaction ID:

`interaction_id = funding_round_id::investor_id`

---

### Important Decisions

1. Do not use naïve comma splitting for investor lists.
2. Use exact canonical dictionary segmentation.
3. Do not force IDs for duplicate investor names.
4. Preserve unmatched HALA Ventures mentions without inventing IDs.
5. Preserve malformed source names and attach quality flags.
6. Use conservative company-name normalization.
7. Use exact website matching only after independent validation.
8. Website-assisted company resolution is accepted only when the website-selected ID belongs to the exact-name candidate set.
9. Website-only company candidates are not automatically accepted.
10. Same-day events with different funding-round IDs are preserved.
11. Do not apply temporal filtering during entity reconstruction.
12. Do not apply VC-only funding-type filtering during the canonical-data phase.
13. Do not collapse follow-on investment events.
14. Do not use future-looking outcome information for entity resolution.

---

### Known Limitations

The final interaction table is a high-confidence subset and should not be described as all Crunchbase investments.

Investor ambiguity is now the main remaining identity-resolution bottleneck.

The project field `startup_id` refers technically to the funded Crunchbase organization and can include entities other than early-stage startups.

The canonical dataset still includes grants, debt financing, post-IPO rounds, secondary-market transactions, and other nontraditional venture-financing types.

Historical coverage is uneven.

The final clean date range does not imply that the entire 1945–2026 period should be used for model training.

---

### Research Insights

Entity resolution is a major dependency of real-world recommendation modeling.

Startup/company identity resolution is already very high (~98.94%), so further coverage improvements would obtain more value from investor disambiguation than additional startup-name matching.

Approximately 15.47% of unique investor–organization pairs contain repeated investment events, demonstrating that new-target discovery and follow-on investment are distinct behaviors present in the data.

The raw Crunchbase universe is much larger than the interaction universe. The experimental heterogeneous graph should therefore be defined by the task and temporal window rather than by automatically loading every Crunchbase entity.

---

### Next Phase

Phase 2 should begin with a temporal distribution audit.

Before selecting the ITRS temporal window:

- inspect interactions by year, month, and quarter;
- measure active investors by quarter;
- measure active organizations by quarter;
- inspect investor sequence lengths;
- inspect recent-period completeness;
- determine how repeated events inside one temporal segment map to implicit feedback;
- only then determine how to reproduce the paper's `t = 60` temporal segmentation.

Do not assume that `2026-06-01` is the correct endpoint merely because it is the latest canonical event.