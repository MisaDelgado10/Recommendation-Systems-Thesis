# Phase 7 Repository Updates

**Status:** COMPLETE / CLOSED  
**Purpose:** Repository-oriented record of the post-hoc final-test diagnostics, thesis evidence synthesis, and Phase-8 handoff.

---

## Documentation convention

Phase 7 follows the documentation pattern already used in the repository:

```text
docs/Phase_7_Closure_Report_ITRS_Crunchbase.md
docs/Phase_7_Closure_Report_ITRS_Crunchbase.docx
docs/Phase_7_Reproduction_Log_Entry.md
docs/Phase_7_Repo_Updates.md
```

The closure report is the comprehensive scientific record. The reproduction log is the chronological phase summary. This file records repository-oriented additions and boundaries.

---

## Authoritative Phase-7 analysis root

```text
data/experimental/phase_7/
```

Major analysis families:

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

---

## Phase-7 scripts from the final diagnostic series

The final diagnostic workflow includes the later script family:

```text
scripts/phase_7_5a_structural_coverage_composition_audit_V2.py
scripts/phase_7_5b_structural_coverage_performance_diagnostic_V1.py
scripts/phase_7_5c_degree_performance_diagnostic_V1.py
scripts/phase_7_6a_temporal_history_support_composition_audit_V1.py
scripts/phase_7_6b_temporal_history_performance_diagnostic_V1.py
scripts/phase_7_7_error_archetypes_and_rank_tail_V1.py
scripts/phase_7_8_adjusted_hit10_statistical_diagnostics_V1.py
scripts/phase_7_9_dependence_and_weighting_robustness_V1.py
scripts/phase_7_10_thesis_evidence_synthesis_and_closure_V1.py
```

Earlier Phase-7.0 through Phase-7.4 scripts/manifests from the current working tree should be committed together with the final analysis package. Failed analysis-script versions should remain preserved when they form part of the audit trail.

---

## Final Phase-7 synthesis products

```text
data/experimental/phase_7/phase_7_10_thesis_evidence_synthesis/
```

Key outputs:

```text
phase_7_10_thesis_evidence_table_V1.csv
phase_7_10_research_task_alignment_V1.csv
phase_7_10_thesis_evidence_synthesis_V1.md
phase_7_10_supervisor_slide_summary_V1.md
phase_7_10_phase_closure_evidence_V1.json
phase_7_10_analysis_manifest_V1.json
phase_7_10_derived_artifact_sha256_V1.json
```

Presentation figures:

```text
figures/phase_7_10_fig1_central_adjusted_evidence_V1.png
figures/phase_7_10_fig1_central_adjusted_evidence_V1.pdf
figures/phase_7_10_fig2_thesis_failure_funnel_V1.png
figures/phase_7_10_fig2_thesis_failure_funnel_V1.pdf
```

---

## Frozen Phase-7 headline evidence

```text
Overall final-test HR@10:                      0.546388
Prior-pair HR@10:                              0.9785
New-pair HR@10:                                0.4461
New warm/warm HR@10:                           0.8175
New warm-investor/cold-startup HR@10:          0.1691
New dual-cold HR@10:                           0.0841
Adjusted warm/warm new-vs-prior difference:   -0.159887
Adjusted cold-start structure association:     +0.080209
Adjusted founder signal:                       +0.094053
Cold isolated startup share of rank>50 tail:    82.91%
```

---

## Robustness products

Phase 7.9 tracks:

```text
phase_7_9_event_covariance_sensitivity_V1.csv
phase_7_9_pair_balanced_covariance_sensitivity_V1.csv
phase_7_9_pair_collapse_invariance_audit_V1.csv
phase_7_9_event_vs_pair_balanced_estimands_V1.csv
phase_7_9_covariance_diagnostics_V1.csv
phase_7_9_robustness_synthesis_V1.csv
phase_7_9_robustness_summary_V1.json
phase_7_9_analysis_manifest_V1.json
phase_7_9_derived_artifact_sha256_V1.json
```

---

## Repository-layout note

The current remote `master` also contains an older Phase-7 root layout:

```text
phase_7_inputs/
phase_7_outputs/
```

and an earlier script family including:

```text
scripts/phase_7_3a_t60_subgroup_performance_audit.py
scripts/phase_7_3b_conditional_discovery_difficulty_audit.py
scripts/phase_7_3c_discovery_specific_cold_start_audit.py
scripts/phase_7_4_training_budget_scaling_analysis.py
```

These should not be silently deleted or rewritten. They are historical repository material from an earlier Phase-7 analysis layout.

The final post-hoc diagnostic series documented here uses:

```text
data/experimental/phase_7/
```

as its authoritative analysis root.

Before the Phase-7 closure commit, verify that the intended final scripts and lightweight derived artifacts from the current working tree are added without erasing the earlier audit material.

---

## Ordinary-Git boundary

The Phase-7 documentation and lightweight analysis tables/manifests should be tracked in ordinary Git.

Large or redundant binary artifacts may remain subject to the existing repository size policy.

The Phase-6 immutable analysis bundle remains the scientific source:

```text
data/experimental/phase_6/full_training/100pct/final_test/
analysis_ready_test_cases.parquet
```

SHA256:

```text
d70f21bff0006e094c5d567d307c0664b41a11e7250a0e69aaec610e1810132d
```

---

## Phase-8 handoff

Phase 7 closes diagnostic analysis of the Phase-6 final test.

Recommended next repository family:

```text
data/experimental/phase_8/
scripts/phase_8_*.py
docs/Phase_8_*.md
```

Phase 8 should begin by freezing the new research/evaluation contract before model development.

The Phase-6 final test must not be used for future checkpoint or architecture selection.
