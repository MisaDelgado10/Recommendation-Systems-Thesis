# Phase 7 Closure Report - ITRS Reproduction on Crunchbase

**Project:** Reproduction of *A Trend-aware Investment Target Recommendation System with Heterogeneous Graph (ITRS)*  
**Dataset adaptation:** Audited Crunchbase reconstruction  
**Phase:** 7 - Post-hoc Final-Test Analysis and Thesis Diagnostics  
**Status:** COMPLETE / CLOSED  
**Previous phase:** Phase 6 - CUDA Production Training, Final Model Selection, and One-Shot Test  
**Next phase:** Phase 8 - Research Design Freeze and Inductive Model Development  
**Closure date:** September 21, 2026

---

## Executive summary

Phase 7 performed a strictly post-hoc diagnostic analysis of the immutable Phase-6 one-shot final test.

It did not retrain the model, change the selected checkpoint, regenerate negatives, modify candidates, rescore the final test, or perform further model selection.

The analysis showed that aggregate ITRS performance masks a much sharper task-specific weakness:

- repeated investor-startup relationships are ranked extremely well;
- genuinely new-to-investor relationships are materially harder even when both entities have prior history;
- startup cold start is associated with the largest observed performance degradation;
- heterogeneous structural support is associated with better ranking specifically for cold-start startups;
- the strongest structural association is concentrated in founder-related information rather than acquisition-only structure;
- among already-connected cold-start startups, greater structural degree does not show a clear monotonic benefit;
- broader investor T0/T1-T59 temporal coverage does not show a clear performance advantage;
- severe rank-tail failures are concentrated in structurally isolated cold-start startups;
- the central adjusted findings remain stable under investor clustering, startup clustering, pair clustering, two-way investor/startup clustering, and pair-balanced weighting.

Frozen aggregate final test:

| Metric | Value |
|---|---:|
| HR@10 | **0.546387682590** |
| NDCG@10 | **0.398709730124** |
| Hits@10 | **11,072 / 20,264** |
| Mean positive rank | **23.034347** |
| Median positive rank | **7** |

The main thesis conclusion is not that ITRS is generally weak. Rather:

> ITRS aggregate future-event performance masks a distinct new-to-investor discovery failure mode, especially when candidate startups lack prior investment history. Founder-related heterogeneous side information is associated with partial recovery in this regime, motivating an inductive startup representation designed explicitly for cold-start discovery.

Phase 7 is diagnostic and observational. It does **not** establish causality and does **not** prove that a proposed inductive architecture will improve performance.

---

# 1. Frozen Phase-6 source and Phase-7 contract

Authoritative Phase-7 input:

```text
data/experimental/phase_6/full_training/100pct/final_test/
analysis_ready_test_cases.parquet
```

Frozen source:

```text
Rows:     20,264
Columns:  74
SHA256:   d70f21bff0006e094c5d567d307c0664b41a11e7250a0e69aaec610e1810132d
```

Phase-6 selected model and final test remained immutable throughout Phase 7.

Prohibited Phase-7 operations:

```text
Model retraining:               NO
Checkpoint reselection:         NO
Model loading for rescoring:    NO
Inference rerun:                NO
Negative regeneration:          NO
Candidate modification:         NO
Final-test rescoring:           NO
Test-based model selection:     NO
```

Phase-7 scientific question:

> Where does the reproduced ITRS model fail, and is its main weakness specifically new-to-investor startup discovery under cold-start conditions?

Primary evaluation endpoint for adjusted diagnostics:

```text
Hit@10 = 1 if positive_rank <= 10 else 0
```

NDCG@10 and rank statistics were retained as secondary diagnostics.

---

# 2. Phase-7 subphase record

| Subphase | Objective | Main result |
|---|---|---|
| 7.0 | Analysis contract and integrity audit | Frozen Phase-6 bundle and subgroup semantics verified; post-hoc-only contract established |
| 7.1 | Baseline rank anatomy | Aggregate performance is polarized: many rank-1 hits coexist with a substantial deep tail |
| 7.2 | Pair novelty | Prior pairs HR@10 97.85%; new pairs HR@10 44.61% |
| 7.3 | Cold-start regimes | Startup cold start produced the largest single-entity degradation |
| 7.4 | Novelty x cold-start interaction | New warm/warm HR@10 81.75%; warm-investor/cold-startup 16.91%; dual-cold 8.41% |
| 7.5A | Structural coverage composition | 72.21% of test cases have neither endpoint structurally connected |
| 7.5B | Structural performance | Structural startup support associated with higher cold-start ranking; founder signal strongest |
| 7.5C | Degree-performance | No clear monotonic benefit from additional degree among already-connected cold-start startups |
| 7.6A | Temporal-history composition | Usable temporal variation concentrated on investor side |
| 7.6B | Temporal-history performance | Broader investor T0/T1-T59 coverage showed no clear HR@10 advantage |
| 7.7 | Error archetypes and rank-tail anatomy | Structurally isolated cold-start startups dominate severe rank>50 failures |
| 7.8 | Adjusted Hit@10 diagnostics | Novelty, startup cold start, cold-start structural support, and founder signal survive mutual adjustment |
| 7.9 | Dependence and weighting robustness | Central estimates stable under two-way clustering and pair-balanced weighting |
| 7.10 | Thesis evidence synthesis and closure | Evidence package consolidated; Phase 7 formally CLOSED |

Failed analysis-script versions were preserved when applicable. The failures were analysis-assumption or display-layer failures and did not modify the frozen data or final test.

---

# 3. Baseline rank anatomy

Frozen final-test ranking anatomy:

```text
Hits@1:              5,431
HR@1:                0.268012
Hits@5:              9,400
HR@5:                0.463877
Hits@10:            11,072
HR@10:               0.546388
NDCG@10:             0.398710
Mean rank:          23.034347
Median rank:         7
```

Selected rank quantiles:

```text
P10:   1
P25:   1
P50:   7
P75:  38
P90:  74
P95:  86
P99:  97
```

Tail composition:

| Rank band | Share |
|---|---:|
| Rank 1 | 26.80% |
| Rank 2-5 | 19.59% |
| Rank 6-10 | 8.25% |
| Rank 11-20 | 9.32% |
| Rank 21-50 | 16.63% |
| Rank 51-100 | 19.41% |

The aggregate result therefore combines a large set of very easy cases with a substantial deep-error tail.

---

# 4. Pair novelty: repeated relationships vs new-to-investor discovery

Final-test composition:

```text
New-to-investor pairs:     16,446 / 20,264 = 81.16%
Prior relationships:        3,818 / 20,264 = 18.84%
```

Unadjusted performance:

| Pair type | N | HR@10 | NDCG@10 | Mean rank | Median rank | Rank > 50 |
|---|---:|---:|---:|---:|---:|---:|
| Prior relationship | 3,818 | 0.9785 | 0.8493 | 1.99 | 1 | 0.00% |
| New-to-investor | 16,446 | 0.4461 | 0.2941 | 27.92 | 15 | 23.91% |

Raw new-minus-prior differences:

```text
Delta HR@10:      -0.532457
Delta NDCG@10:    -0.555183
Delta mean rank: +25.934
Delta median:     +14
```

All 3,933 rank>50 failures were new-to-investor cases.

The raw novelty contrast is confounded by cold-start composition because 58.11% of new-pair cases contain at least one cold endpoint, while all prior-pair cases are warm/warm. Phase 7.4 therefore performed the required stratification.

---

# 5. Novelty x cold-start interaction

Five mutually exclusive diagnostic groups were frozen:

| Diagnostic group | N | HR@10 | NDCG@10 | Mean rank | Median rank | Rank > 50 |
|---|---:|---:|---:|---:|---:|---:|
| Prior pair - warm investor + warm startup | 3,818 | 0.9785 | 0.8493 | 1.99 | 1 | 0.00% |
| New pair - warm investor + warm startup | 6,889 | 0.8175 | 0.5829 | 7.22 | 2 | 1.93% |
| New pair - warm investor + cold startup | 6,392 | 0.1691 | 0.0802 | 44.21 | 41 | 41.74% |
| New pair - cold investor + warm startup | 1,453 | 0.3297 | 0.1717 | 26.17 | 18 | 16.66% |
| New pair - cold investor + cold startup | 1,712 | 0.0841 | 0.0345 | 51.91 | 53 | 51.99% |

Key unadjusted contrasts:

```text
New warm/warm - prior warm/warm:
Delta HR@10 = -0.160988

New warm-investor/cold-startup - new warm/warm:
Delta HR@10 = -0.648418

New cold-investor/warm-startup - new warm/warm:
Delta HR@10 = -0.487872

New dual-cold - new warm/warm:
Delta HR@10 = -0.733423
```

The important result is that a residual novelty gap remains even when both entities are warm, while startup cold start introduces a much larger additional degradation.

---

# 6. Structural coverage composition

Overall joint structural coverage:

| Structural state | N | Share |
|---|---:|---:|
| Both connected | 604 | 2.98% |
| Investor connected only | 1,621 | 8.00% |
| Startup connected only | 3,407 | 16.81% |
| Neither connected | 14,632 | 72.21% |

Cold-start structural support:

```text
Cold-start startup cases:             8,104
Structurally connected cold startups:   957 = 11.81%

Cold-start investor cases:            3,165
Structurally connected cold investors:    93 = 2.94%
```

Within the thesis-primary new-pair / warm-investor / cold-startup regime:

```text
Startup structurally connected:   809
Startup structurally isolated:  5,583
```

This cell size made startup-side structural diagnostics feasible. Cold-investor structural support was much rarer and remained a secondary/small-N analysis.

---

# 7. Structural availability and source-specific performance

Within new-to-investor, warm-investor, cold-startup cases:

| Startup structural state | N | HR@10 |
|---|---:|---:|
| Connected | 809 | 23.98% |
| Isolated | 5,583 | 15.89% |

Unadjusted difference:

```text
Delta HR@10:   +8.09 percentage points
Delta NDCG@10: +0.0339
Delta mean rank: -10.84 positions
```

The association persisted when investor connectivity was held fixed:

```text
Investor structurally isolated:
Delta HR@10 = +7.87 pp

Investor structurally connected:
Delta HR@10 = +8.24 pp
```

Structural source classes in the cold-start startup regime:

```text
Founder only:               706
Acquisition only:           100
Founder + acquisition:        3
Neither:                  5,583
```

Unadjusted HR@10:

```text
Founder only:       25.35%
Acquisition only:   14.00%
Neither:            15.89%
```

This suggested that the structural association is not a generic connectivity effect and is concentrated primarily in founder-related information.

---

# 8. Structural degree intensity

Phase 7.5C separated two questions:

1. whether any usable structural support exists; and
2. whether additional structural degree improves performance once support exists.

Among the 809 connected cold-start startups, positive startup degree ranged from 2 to 26.

Observed continuous associations:

```text
Spearman log-degree vs HR@10:   -0.0254
Spearman log-degree vs NDCG@10: -0.0269
Spearman log-degree vs rank:    +0.0187
```

Investor-cluster bootstrap for the Hit@10 slope per +1 log1p(degree):

```text
Estimate:  -0.027810
95% CI:    [-0.095410, +0.044729]
```

Interpretation:

> The Phase-7 evidence supports a distinction between structural availability and structural intensity. Having some usable side information is associated with better cold-start ranking, while simply accumulating more structural edges is not clearly beneficial.

---

# 9. Temporal-history diagnostics

Literal history classes were defined only from frozen indicators:

```text
cold             = not seen in T0 or T1-T59
t0_only          = seen in T0, not T1-T59
t1_t59_only      = seen in T1-T59, not T0
t0_and_t1_t59    = seen in both T0 and T1-T59
```

Inside the 6,889 new warm/warm cases, investor history provided the only adequately sized formal comparison:

```text
Investor T1-T59 only:     5,170
Investor T0 + T1-T59:     1,710
Investor T0 only:             9
```

Primary investor-history contrast:

```text
Delta HR@10: +1.22 pp
95% CI:      [-1.11, +3.51] pp
```

With startup history fixed to T1-T59 only:

```text
Delta HR@10: +1.35 pp
95% CI:      [-0.97, +3.63] pp
```

The intervals cross zero. The available evidence therefore does not show a clear ranking advantage from broader investor T0/T1-T59 coverage alone.

Startup history remained descriptive because only 90 new warm/warm startup cases were in the T0+T1-T59 class, below the pre-specified formal-analysis threshold of N=100.

---

# 10. Error archetypes and severe rank tail

Phase 7.7 used pre-established novelty, cold-start, and structural dimensions rather than outcome-tuned subgroup definitions.

The largest severe-error archetype was:

```text
New pair
+ warm investor
+ cold-start startup
+ structurally isolated startup
```

It represents 27.55% of the final test but contributes 62.32% of all rank>50 failures, corresponding to 2.26x exposure-relative enrichment.

Across the entire severe tail:

```text
Rank > 50 failures:                          3,933
Cold-start startup involved:                 90.47%
Cold + structurally isolated startup:        82.91%
Warm investor + cold isolated startup:       62.32%
```

Structural connectivity did not cleanly rescue dual-cold cases. The clearest structural association remained the warm-investor / cold-startup regime.

The realized-positive collision subgroup contained only 47 cases and did not explain the severe tail.

---

# 11. Adjusted Hit@10 diagnostics

Primary endpoint:

```text
Hit@10
```

Estimator:

```text
Linear probability model
+ investor-cluster robust CR1 covariance
```

Three pre-specified models were used.

## 11.1 Adjusted warm/warm pair novelty

Population:

```text
Warm investor + warm startup
N = 10,707
```

Adjusted new-minus-prior pair difference:

```text
Estimate:  -0.159887
95% CI:    [-0.171156, -0.148618]
```

The warm/warm novelty gap therefore remains after adjustment for structural coverage and literal T0 exposure indicators.

## 11.2 Startup cold start x structural availability

Population:

```text
New-to-investor + warm investor
N = 13,281
```

Adjusted estimands:

| Estimand | Estimate | 95% CI |
|---|---:|---:|
| Cold-start startup penalty when structurally isolated | -0.659734 | [-0.682087, -0.637382] |
| Cold-start startup penalty when structurally connected | -0.584473 | [-0.623297, -0.545649] |
| Startup structure among warm startups | +0.004948 | [-0.016689, +0.026585] |
| Startup structure among cold-start startups | +0.080209 | [+0.049015, +0.111403] |
| Cold-minus-warm structural interaction | +0.075262 | [+0.036764, +0.113759] |

The structural association is therefore concentrated specifically in the startup cold-start regime.

## 11.3 Structural source under startup cold start

Population:

```text
New-to-investor + warm investor + cold-start startup
N = 6,392
```

Adjusted source-specific associations:

| Structural source | Estimate | 95% CI |
|---|---:|---:|
| Founder-related support vs neither | +0.094053 | [+0.060252, +0.127853] |
| Acquisition-only support vs neither | -0.019796 | [-0.095402, +0.055810] |

All three linear-probability models produced fitted probabilities within [0,1].

These are observational adjusted associations, not causal estimates.

---

# 12. Dependence and weighting robustness

Phase 7.9 reused the exact Phase-7.8 model formulas without specification search.

Covariance sensitivity included:

```text
Investor-cluster CR1
Startup-cluster CR1
Investor-startup pair-cluster CR1
Two-way investor + startup CR1
```

Two-way covariance:

```text
V_two_way = V_investor + V_startup - V_pair
```

Selected two-way results:

| Estimand | Estimate | Two-way 95% CI |
|---|---:|---:|
| New vs prior - warm/warm | -0.159887 | [-0.173829, -0.145945] |
| Cold-start penalty - startup isolated | -0.659734 | [-0.685709, -0.633760] |
| Cold-start penalty - startup connected | -0.584473 | [-0.643188, -0.525758] |
| Startup structure - cold-start startups | +0.080209 | [+0.028575, +0.131844] |
| Cold-minus-warm structure interaction | +0.075262 | [+0.015293, +0.135230] |
| Founder signal vs neither | +0.094053 | [+0.036634, +0.151471] |
| Acquisition-only vs neither | -0.019796 | [-0.103888, +0.064296] |

Pair-balanced audit:

```text
Event rows:                         20,264
Unique investor-startup pairs:      20,111
Pairs with >1 final-test event:        139
Events belonging to repeated pairs:    292
Maximum events in one pair:              15
```

Pair-balanced point estimates were nearly identical to event-level estimates.

Examples:

```text
Warm/warm novelty:
Event:          -0.159887
Pair-balanced:  -0.160205

Cold-start startup structure:
Event:          +0.080209
Pair-balanced:  +0.081817

Founder signal:
Event:          +0.094053
Pair-balanced:  +0.095940
```

The main Phase-7 findings are therefore not artifacts of one clustering assumption or repeated-pair weighting.

---

# 13. Consolidated thesis evidence

| Evidence | Status | Main result |
|---|---|---|
| Aggregate final-test performance | Supported descriptive baseline | HR@10 = 0.546388 |
| New-to-investor novelty gap | Supported | Adjusted warm/warm delta HR@10 = -15.99 pp |
| Startup cold-start degradation | Supported | Adjusted isolated-startup penalty = -65.97 pp |
| Structural support under startup cold start | Supported | Adjusted delta HR@10 = +8.02 pp |
| Structural support among warm startups | No clear association | +0.49 pp; CI crosses zero |
| Founder-related structural support | Supported | +9.41 pp; robust interval excludes zero |
| Acquisition-only structural support | No clear association | -1.98 pp; CI crosses zero |
| Structural degree intensity | No clear association | Hit@10 slope CI crosses zero |
| Broader investor temporal coverage | No clear association | +1.22 pp; CI crosses zero |
| Severe-error concentration | Supported descriptive result | 82.91% of rank>50 failures involve cold + structurally isolated startup |
| Dependence / repeated-pair robustness | Supported | Central findings survive two-way clustering and pair-balanced weighting |

---

# 14. Interpretation boundary

Phase 7 supports the following empirical diagnosis:

1. New-to-investor discovery is a distinct and harder recommendation task than repeated relationship ranking.
2. The novelty gap persists even when both endpoints have prior observations.
3. Startup cold start is the dominant observed degradation.
4. Heterogeneous structural support is associated with partial improvement specifically when startup investment history is absent.
5. Founder-related structure carries the clearest structural association.
6. Greater degree among already-connected cold-start startups does not show a clear additional benefit.
7. Broader investor temporal coverage alone does not explain the new-pair gap.
8. Severe failures concentrate in structurally isolated cold-start startups.
9. The central conclusions survive adjusted and dependence-robust diagnostics.

Phase 7 does **not** establish:

```text
Causal effects:                          NO
Founder structure as proven mechanism:  NO
Inductive GNN superiority:              NO
Future model improvement:               NO
All heterogeneous relations useful:     NO
More graph degree always better:         NO
```

The frozen Phase-6 final test must not be reused as a model-selection set.

---

# 15. Thesis problem statement after Phase 7

A thesis-ready problem statement is:

> Although the reproduced ITRS achieves 54.64% HR@10 on the frozen one-shot final test, aggregate future-event performance masks substantial task heterogeneity. New-to-investor relationships remain harder than prior investor-startup relationships even when both endpoints have historical observations: the adjusted warm/warm new-pair difference is -15.99 percentage points under two-way investor/startup clustered uncertainty. Within new-pair cases, startup cold start is associated with a much larger degradation, while heterogeneous structural support is associated with approximately +8.02 percentage points higher Hit@10 among cold-start startups. The structural association is concentrated in founder-related support, whereas acquisition-only support, structural degree intensity, and broader investor temporal coverage do not show clear benefits. Most severe rank-tail failures involve cold-start startups that are structurally isolated.

This motivates evaluation and modeling explicitly designed for new-to-investor discovery under startup cold start.

---

# 16. Phase 8 handoff - research direction

Phase 8 should move from diagnosis to controlled model-development experiments.

Recommended research direction:

> Develop an inductive heterogeneous startup representation for new-to-investor discovery under startup cold start, prioritize founder-related side information, and retain the ITRS investor/trend branch initially so that the startup-representation change can be tested in isolation.

Recommended Phase-8 sequence:

## Phase 8.0 - Evaluation reset and confirmatory holdout design

Before architecture development, determine whether an additional untouched temporal holdout can be reserved from the Crunchbase timeline.

Reason:

Phase 7 used the Phase-6 final test diagnostically to identify the exact weaknesses the next model will target. The Phase-6 test remains valid as the reproduction diagnostic benchmark, but it should not be treated as the only pristine confirmatory set for a new architecture explicitly designed in response to those diagnostics.

Preferred options:

1. reserve a later untouched temporal period, if available; or
2. freeze a rolling-origin / backtesting protocol with windows not used for Phase-7 diagnosis.

## Phase 8.1 - Freeze research questions and hypotheses

Primary research question:

> Can an inductive heterogeneous startup representation improve new-to-investor ranking for cold-start startups without materially degrading new warm/warm discovery?

Secondary questions:

- Are founder relations the primary useful side-information source?
- Does acquisition structure add incremental value?
- Does the trend-aware investor component still add value after startup cold-start representation is improved?
- Does the proposed model help specifically where Phase 7 predicts it should help?

## Phase 8.2 - Validation-side diagnostic strata

Reconstruct the Phase-7 strata on validation data before model development:

```text
new warm/warm
new warm-investor / cold-startup
new cold-investor / warm-startup
new dual-cold
startup structural connected / isolated
founder / acquisition structural source
```

## Phase 8.3 - Baseline matrix

At minimum:

```text
B0: frozen reproduced ITRS baseline
B1: simple inductive startup feature baseline
B2: inductive heterogeneous startup encoder
B3: B2 + ITRS trend-aware investor representation
```

## Phase 8.4 - Controlled relation ablations

Required ablations:

```text
founder-only
acquisition-only
founder + acquisition
no heterogeneous side relations
trend-aware investor branch on/off
```

## Phase 8.5 - Validation-only model selection

Model selection should be frozen before training.

Recommended primary validation target:

```text
new-to-investor + warm-investor + cold-startup NDCG@10
```

Recommended secondary metrics / guardrails:

```text
HR@10 in the same primary stratum
new warm/warm NDCG@10 / HR@10
aggregate validation HR@10 / NDCG@10
```

## Phase 8.6 - Robustness and final freeze

Before confirmatory evaluation:

- freeze architecture;
- freeze hyperparameters;
- freeze evaluation strata;
- freeze checkpoint-selection rule;
- freeze ablation set;
- freeze statistical analysis plan.

Only then execute the new untouched confirmatory evaluation.

---

# 17. Phase-7 artifact family

Authoritative analysis root:

```text
data/experimental/phase_7/
```

Major completed analysis families:

```text
phase_7_0_analysis_contract/
phase_7_1_baseline_rank_anatomy/
phase_7_2_pair_novelty/
phase_7_3_cold_start/
phase_7_4_novelty_cold_start_interaction/
phase_7_5a_structural_coverage_composition/
phase_7_5b_structural_performance/
phase_7_5c_degree_performance/
phase_7_6a_temporal_history_composition/
phase_7_6b_temporal_history_performance/
phase_7_7_error_archetypes/
phase_7_8_adjusted_hit10/
phase_7_9_dependence_robustness/
phase_7_10_thesis_evidence_synthesis/
```

Final synthesis products include:

```text
phase_7_10_thesis_evidence_table_V1.csv
phase_7_10_research_task_alignment_V1.csv
phase_7_10_thesis_evidence_synthesis_V1.md
phase_7_10_supervisor_slide_summary_V1.md
phase_7_10_phase_closure_evidence_V1.json
phase_7_10_analysis_manifest_V1.json
phase_7_10_derived_artifact_sha256_V1.json
```

The repository's older root-level `phase_7_inputs/` and `phase_7_outputs/` layout predates this final post-hoc diagnostic series and should be retained as historical audit material rather than silently overwritten.

---

# 18. Closure decision

Phase 7 is COMPLETE / CLOSED.

Central frozen diagnostic conclusions:

```text
Aggregate final-test HR@10:                      0.546388
Adjusted warm/warm new-vs-prior gap:            -0.159887
Adjusted startup cold-start penalty, isolated:  -0.659734
Adjusted structure association, cold startups:  +0.080209
Adjusted founder signal:                         +0.094053
Cold isolated startup share of rank>50 failures: 82.91%
```

Final Phase-7 policy:

```text
Phase-6 selected checkpoint:  IMMUTABLE
Phase-6 final test:            DIAGNOSTICALLY CLOSED
Future model selection:       VALIDATION ONLY
New architecture claims:      REQUIRE NEW CONTROLLED EXPERIMENTS
```

**PHASE 7 STATUS: COMPLETE / CLOSED**

**Next project action:** begin Phase 8.0 by freezing the confirmatory evaluation design before implementing the inductive cold-start model.
