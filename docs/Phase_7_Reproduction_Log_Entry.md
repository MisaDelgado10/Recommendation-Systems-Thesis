# Reproduction Log Entry - Phase 7: Post-hoc Final-Test Analysis and Thesis Diagnostics

**Project:** Reproduction of *A Trend-aware Investment Target Recommendation System with Heterogeneous Graph (ITRS)* using audited Crunchbase data  
**Status:** COMPLETE / CLOSED  
**Previous phase:** Phase 6 - CUDA Production Training and Final Evaluation  
**Next phase:** Phase 8 - Research Design Freeze and Inductive Model Development  
**Phase role:** Diagnose the immutable one-shot final test without retraining, rescoring, or further model selection.

---

## Phase objective

Phase 7 analyzed the frozen Phase-6 final test to determine where the reproduced ITRS succeeds and fails, with special attention to the thesis target of new-to-investor startup discovery under cold start.

The phase was intentionally post-hoc. Its goal was not to improve metrics, but to answer:

1. How much aggregate performance is driven by repeated investor-startup relationships?
2. Does a new-to-investor gap persist when investor and startup are both historically observed?
3. Which cold-start regime is most difficult?
4. Is heterogeneous graph structure associated with better ranking under cold start?
5. Which relation types appear most informative?
6. Are findings robust to clustering, repeated entities, and repeated pairs?
7. Which empirical findings justify the next thesis experiments?

All questions were closed without changing the Phase-6 model or final test.

---

## Chronological subphase record

| Subphase | Objective | Main result / decision |
|---|---|---|
| 7.0 | Bind immutable Phase-6 analysis source and Phase-7 safeguards | Source SHA, schema, metrics, subgroup semantics PASS |
| 7.1 | Analyze rank distribution | Final test combines strong top ranks with substantial deep tail |
| 7.2 | Compare new-to-investor vs prior relationships | New pairs HR@10 44.61%; prior pairs 97.85% |
| 7.3 | Compare four cold-start regimes | Startup cold start produces larger degradation than investor cold start |
| 7.4 | Joint novelty x cold-start stratification | Warm/warm novelty gap persists; startup cold start compounds it sharply |
| 7.5A | Outcome-blind structural composition audit | 72.21% of events have neither endpoint structurally connected |
| 7.5B | Structural-performance diagnostic | Cold-start startup structure associated with +8.09 pp unadjusted HR@10 |
| 7.5C | Degree-performance diagnostic | No clear monotonic benefit from more degree once startup is connected |
| 7.6A | Outcome-blind temporal-history composition | Formal history variation adequate mainly on investor side |
| 7.6B | Temporal-history performance | Broader investor T0/T1-T59 coverage shows no clear advantage |
| 7.7 | Error archetypes | Cold + structurally isolated startups dominate severe rank tail |
| 7.8 | Adjusted Hit@10 | Central novelty/cold-start/founder findings survive mutual adjustment |
| 7.9 | Dependence/weighting robustness | Central findings survive two-way clustering and pair-balanced weighting |
| 7.10 | Evidence synthesis | Thesis evidence package frozen; Phase 7 closed |

---

## Frozen final-test baseline

```text
Cases:                20,264
Candidates / case:    100
HR@10:                0.546387682590
NDCG@10:              0.398709730124
Hits@10:              11,072
Mean positive rank:   23.034347
Median positive rank: 7
```

Source bundle:

```text
data/experimental/phase_6/full_training/100pct/final_test/
analysis_ready_test_cases.parquet

SHA256:
d70f21bff0006e094c5d567d307c0664b41a11e7250a0e69aaec610e1810132d
```

---

## Main diagnostic results

### Pair novelty

```text
Prior-pair HR@10:       97.85%
New-pair HR@10:         44.61%

Prior warm/warm HR@10: 97.85%
New warm/warm HR@10:   81.75%
```

The novelty gap therefore persists even after cold endpoints are removed.

### Cold-start regimes

```text
New warm/warm:                  81.75% HR@10
New warm investor/cold startup: 16.91% HR@10
New cold investor/warm startup: 32.97% HR@10
New dual cold:                   8.41% HR@10
```

Startup cold start is the strongest single-entity degradation.

### Structural support

Within new warm-investor / cold-startup cases:

```text
Connected startup HR@10: 23.98%
Isolated startup HR@10:  15.89%
Unadjusted delta:        +8.09 pp
```

Founder-only structure showed the clearest source-specific signal.

### Degree intensity

Among already-connected cold-start startups:

```text
Spearman log-degree vs HR@10: -0.0254
Hit@10 slope 95% CI:          [-0.095410, +0.044729]
```

No clear evidence supports a simple "more edges is better" interpretation.

### Temporal history

Within new warm/warm cases:

```text
Investor T0+T1-T59 minus T1-T59-only HR@10:
+1.22 pp
95% CI: [-1.11, +3.51] pp
```

No clear advantage from broader investor temporal coverage alone.

### Error tail

```text
Rank>50 failures:                    3,933
Cold-start startup involved:         90.47%
Cold + structurally isolated startup:82.91%
Warm-I + cold isolated startup:      62.32%
```

---

## Adjusted analysis

Primary endpoint:

```text
Hit@10
```

Estimator:

```text
Linear probability model
+ investor-cluster robust CR1 covariance
```

Adjusted Phase-7.8 findings:

```text
Warm/warm new vs prior:                -15.99 pp
Cold-start penalty, startup isolated:  -65.97 pp
Cold-start penalty, startup connected: -58.45 pp
Startup structure among warm startups:  +0.49 pp
Startup structure among cold startups:  +8.02 pp
Cold-minus-warm structural interaction: +7.53 pp
Founder signal vs neither:              +9.41 pp
Acquisition-only vs neither:             -1.98 pp
```

The warm-start structural and acquisition-only intervals cross zero. The new-pair, cold-start, cold-start structural, interaction, and founder intervals exclude zero under the Phase-7.8 investor-cluster specification.

---

## Dependence robustness

Phase 7.9 reused the exact Phase-7.8 formulas and evaluated:

```text
investor-cluster CR1
startup-cluster CR1
pair-cluster CR1
two-way investor + startup CR1
pair-balanced weighting
```

Selected two-way intervals:

```text
Warm/warm novelty:
-15.99 pp, 95% CI [-17.38, -14.59]

Cold-start startup structure:
+8.02 pp, 95% CI [+2.86, +13.18]

Founder signal:
+9.41 pp, 95% CI [+3.66, +15.15]
```

Pair-balanced point estimates were nearly unchanged.

---

## Phase-7 thesis conclusion

Phase 7 supports the following diagnosis:

> The reproduced ITRS is not simply weak on future-event prediction. Its aggregate performance masks a distinct weakness in genuinely new-to-investor startup discovery, especially when the candidate startup has no prior investment history. Heterogeneous side information is associated with partial improvement in that cold-start regime, with founder-related structure showing the clearest signal.

The phase does not establish causality or prove an inductive architecture.

---

## Phase-8 handoff

Phase 8 should test the diagnosis with controlled model-development experiments.

Primary direction:

```text
Inductive heterogeneous startup representation
+ founder-side information
+ new-to-investor evaluation
+ startup cold-start primary stratum
```

The Phase-6 final test remains closed for model selection.

Before architecture work, Phase 8.0 should freeze a confirmatory evaluation protocol, preferably using an additional untouched temporal holdout or a pre-frozen rolling-origin design.

---

## Phase 7 closure

Phase 7 is COMPLETE / CLOSED.

No retraining, checkpoint reselection, inference rerun, final-test rescoring, negative resampling, or test-based model selection occurred.

Authoritative comprehensive report:

```text
docs/Phase_7_Closure_Report_ITRS_Crunchbase.md
```
