# Phase 2 — Temporal Reconstruction ✅

**Completion date:** 2026-08-17  
**Project:** ITRS Crunchbase Reproduction  
**Status:** COMPLETE  
**Next phase:** Phase 3 — Heterogeneous Graph Reconstruction

## Objective

Transform the immutable Phase-1 canonical interaction table into a temporally valid, paper-grounded experimental split compatible with *A Trend-aware Investment Target Recommendation System with Heterogeneous Graph (ITRS)*.

Phase 2 intentionally separated temporal reconstruction from graph construction, negative sampling, investment-type filtering, and model training.

## Inputs

- `data/processed/interactions.parquet`
- ITRS paper: Chen et al. (2021), *A Trend-aware Investment Target Recommendation System with Heterogeneous Graph*
- Approximate data-collection reference: 2026-06-02
- Phase-2 audit outputs under `data/experimental/phase_2/audits/`

## Paper-grounded temporal design

The ITRS paper specifies:

- `t = 60` in the main experiment.
- `t = 60` corresponds to **3-month temporal segments**.
- The detailed history therefore spans **15 years**.
- Events older than the detailed horizon are assigned to **T0**.
- Event order inside one temporal segment is ignored.
- The final temporal fragment is used for validation/test.
- 10% of final-fragment events are used for validation and the remainder for test.
- The paper does **not** report a random seed or an explicit minimum investor-history threshold.

## Final temporal construction

```text
T0      <= 2011-03-31
T1-T60   2011-04-01 -> 2026-03-31
T60      2026-01-01 -> 2026-03-31
```

Detailed temporal design:

- Detailed segments: **60**
- Months per segment: **3**
- Detailed horizon: **15 years**
- Calendar alignment: **standard calendar quarters**
- Selected experimental endpoint: **2026-03-31**
- Reproduction-specific validation/test seed: **42**

## Final counts

| Layer | Interactions |
|---|---:|
| Phase-1 canonical | 1,208,051 |
| ITRS temporal experiment | 1,195,937 |
| T0 compressed prehistory | 100,173 |
| T1-T59 detailed historical training | 1,073,249 |
| Historical training pool T0-T59 | 1,173,422 |
| T60 evaluation pool | 22,515 |
| Validation | 2,251 |
| Test | 20,264 |
| Post-endpoint canonical tail | 12,114 |

## Important temporal-audit findings

- `announced_on` is the operational event-time field and is 100% parseable in the canonical interaction layer.
- The canonical range is `1945-01-01 -> 2026-06-01`.
- Early Crunchbase history is extremely sparse.
- The last zero-interaction month is `1993-07`; continuous nonzero monthly coverage begins in `1993-08`, but continuity alone was not treated as sufficient evidence for a cutoff.
- June 2026 is objectively calendar-incomplete.
- May 2026 is calendar-complete, but database/reporting maturity cannot be measured from the available snapshot.
- No `created_at`, `updated_at`, `published_at`, or ingestion timestamp is available to directly estimate Crunchbase reporting lag.
- Endpoint maturity buffers are therefore **hypothetical sensitivity scenarios**, not estimates of a true reporting delay.
- Investor activity is strongly heavy-tailed; the global median sequence length is 1.
- Moving the start date later did **not** improve median sequence length, so temporal-window selection and investor minimum-history eligibility remain separate decisions.
- Same-day ties are substantial; no within-day chronology is invented.

## T60 evaluation-pool findings

T60 contains:

- 22,515 interactions
- 11,884 investors
- 8,992 startups
- 22,327 unique investor-startup pairs

Interaction-cold start:

- Cold investors: 3,060 = 25.75% of T60 investors
- Cold startups: 4,715 = 52.44% of T60 startups
- Events with a cold investor: 15.73%
- Events with a cold startup: 40.17%

Pair novelty:

- New-to-investor unique pairs: 18,161 = 81.34%
- Previously seen unique pairs: 4,166 = 18.66%
- New-to-investor T60 events: 18,296 = 81.26%
- Events repeating a prior pair: 4,219 = 18.74%
- Pairs repeated within T60: 172 = 0.77%

`interaction-cold` means **no previous investment interaction**. It does not yet mean that the entity is absent from the future heterogeneous graph.

## Validation/test split

Paper-style event-level random split:

- Validation fraction: 10%
- Integer rule: `floor(N * 0.10)`
- Validation: 2,251
- Test: 20,264
- Stratified: No
- Random seed: 42

Seed 42 is reproduction-specific because the paper does not report its seed.

Population composition stayed close to full T60:

| Split | New-pair events | Cold-investor events | Cold-startup events |
|---|---:|---:|---:|
| Full T60 | 81.261% | 15.727% | 40.164% |
| Validation | 82.186% | 16.704% | 41.715% |
| Test | 81.159% | 15.619% | 39.992% |

## Leakage and dependence diagnostics

- Validation/test interaction-ID overlap: **0**
- Validation/test unique-pair overlap: **33**
- Validation/test funding-round overlap: **1,315**
- Historical T0-T59 / T60 funding-round overlap: **0**

The validation/test funding-round overlap is an evaluation-dependence limitation caused by event-level splitting of multi-investor funding rounds. Both subsets are held out, so it is not historical-training leakage.

**Critical Phase-3 rule:** no T60 held-out investment relationship may be inserted into the graph used for training.

## Final model-ready outputs

```text
data/experimental/phase_2/model_ready/
├── interactions_itrs_temporal_split.parquet
├── t60_holdout_pair_manifest.parquet
├── t60_validation_test_split.parquet
└── t60_split_assignments.csv
```

Selected temporal outputs:

```text
data/experimental/phase_2/temporal/
├── interactions_itrs_temporal.parquet
├── itrs_segment_metadata.csv
└── post_endpoint_interactions.parquet
```

Final integrity/audit outputs:

```text
data/experimental/phase_2/audits/
├── phase_2_final_temporal_integrity_checks.csv
├── phase_2_final_temporal_split_summary.csv
├── phase_2_validation_test_distribution_diagnostics.csv
└── phase_2_validation_test_overlap_diagnostics.csv
```

Additional Phase-2 audit outputs in the same directory document annual/monthly/quarterly coverage, investor-history distributions, censoring sensitivity, start-year tradeoffs, and ITRS-compatible window comparisons.

## Confirmed final-stage scripts

```text
scripts/phase_2_1_7_1_candidate_start_year_audit.py
scripts/phase_2_1_7_2_joint_window_sensitivity.py
scripts/phase_2_2_2_itrs_compatible_window_comparison.py
scripts/phase_2_3_1_build_itrs_temporal_interactions.py
scripts/phase_2_3_2_t60_evaluation_cold_start_audit.py
scripts/phase_2_3_3_build_t60_validation_test_split.py
scripts/phase_2_3_4_final_temporal_integrity_audit.py
```

Earlier Phase-2.1 audit scripts already present in the repository should remain versioned together with these final-stage scripts.

## Methodological decisions

1. Keep `data/processed/interactions.parquet` immutable.
2. Use `announced_on` as operational investment-event time.
3. Do not infer actual reporting lag from the available single snapshot.
4. Use the paper-grounded 15-year / 60 x 3-month temporal structure.
5. Preserve older history in T0 instead of deleting it.
6. Select `2026-03-31` as the experimental endpoint.
7. Use standard calendar-quarter-aligned detailed segments.
8. Keep interaction-cold investors and startups in T60.
9. Keep prior investor-startup repeat events for strict ITRS reproduction.
10. Use the paper-style 10% validation / 90% test event split.
11. Fix seed 42 only for reproducibility; it is not a paper parameter.
12. Preserve validation/test pair and funding-round overlap rather than changing the paper-style event split.
13. Do not perform negative sampling yet.
14. Do not filter investment types yet.
15. Do not impose a minimum investor-history threshold yet.

## Known limitations / unresolved issues

- `announced_on` is not an ingestion timestamp or necessarily the exact investment-closing date.
- Actual Crunchbase reporting lag is unidentifiable from the current snapshot.
- The ITRS paper has a minor T0/t indexing ambiguity; the reproduction follows the clear algorithmic intent.
- The paper reports 91,710 total events while its stated split counts sum to 91,711.
- Interaction-cold start has been audited, but graph/entity-level cold start remains unknown.
- Validation/test funding-round dependence is substantial because funding rounds may contain multiple investor events.
- Investor minimum-history eligibility remains unresolved.
- Training/evaluation negative sampling remains unresolved.
- Heterogeneous graph construction has not started.

## Phase-3 handoff constraints

Phase 3 must:

- Treat the Phase-2 temporal split as frozen.
- Use `t60_holdout_pair_manifest.parquet` to prevent graph-label leakage.
- Never include held-out T60 investment relations as training graph edges.
- Distinguish interaction-cold start from graph/entity-cold start.
- Reconstruct the ITRS graph schema from the paper first.
- Map only relationships that are genuinely supported by the available Crunchbase files.
- Never invent missing Tianyancha-equivalent relations.
- Defer negative sampling until the graph and candidate universe are audited.

## Next

**Phase 3 — Heterogeneous Graph Reconstruction**

Start with:

**Phase 3.1 — Exact ITRS Graph-Schema Reconstruction**
