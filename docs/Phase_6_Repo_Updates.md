# Phase 6 Repository Updates

**Status:** COMPLETE / FROZEN  
**Purpose:** Repository-oriented record of CUDA qualification, production training, one-shot final evaluation, and Phase-7 handoff.

---

## Phase-6 scripts now tracked

### CUDA qualification and throughput

```text
scripts/phase_6_1_cuda_d_both_sparse_qualification.py
scripts/phase_6_1_cuda_mac_reference_qualification.py
scripts/phase_6_1b_cuda_strict_fp32_qualification.py
scripts/phase_6_2_packed_sparse_cuda_strict_fp32_qualification.py
scripts/phase_6_3_packed_sparse_cuda_throughput_benchmark.py
```

### Reduced-training pilots

```text
scripts/phase_6_4_freeze_1pct_training_subset.py
scripts/phase_6_5a_build_1pct_filtered_epoch_streams.py
scripts/phase_6_5b_1pct_bounded_trainer_integration.py
scripts/phase_6_5c_1pct_20epoch_pilot.py
scripts/phase_6_6a_freeze_5pct_training_subset.py
scripts/phase_6_6b_build_5pct_filtered_epoch_streams.py
scripts/phase_6_6c_5pct_bounded_trainer_integration.py
scripts/phase_6_6d_5pct_20epoch_pilot.py
scripts/phase_6_7a_freeze_10pct_training_subset.py
scripts/phase_6_7b_build_10pct_filtered_epoch_streams.py
scripts/phase_6_7c_10pct_bounded_trainer_integration.py
scripts/phase_6_7d_10pct_20epoch_pilot.py
```

### Production/final-test closure

```text
scripts/phase_6_9a_freeze_100pct_20epoch_streams_V1.py
scripts/phase_6_9a_finalize_100pct_20epoch_stream_freeze_V2.py
scripts/phase_6_9b_100pct_20epoch_production_training_V1.py
scripts/phase_6_9c_final_test_authorization_and_one_shot_evaluation_V1.py
scripts/phase_6_9c_final_test_authorization_and_one_shot_evaluation_V2.py
scripts/phase_6_9d_build_analysis_ready_test_bundle_V1.py
```

The V1 final-test evaluator is intentionally preserved as audit trail. It failed during read-only preflight because it omitted the Phase-6 portable initialization bridge; it scored zero test cases. V2 is authoritative.

---

## Final tracked data products

```text
data/experimental/phase_6/contracts/phase_6_9c_final_test_authorization_V2.json
data/experimental/phase_6/contracts/phase_6_9c_final_test_result_V2.json

data/experimental/phase_6/full_training/100pct/final_test/
final_test_case_metrics.parquet
final_test_raw_logits.npy
final_test_summary.json
analysis_ready_test_cases.parquet
analysis_ready_test_cases_manifest.json
```

---

## Frozen metrics

```text
Selected display epoch: 19

Validation HR@10:    0.536206130609
Validation NDCG@10:  0.389611543437

Test HR@10:          0.546387682590
Test NDCG@10:        0.398709730124
Hits@10:             11,072 / 20,264
Mean positive rank:  23.034347
Median positive rank: 7
```

---

## Key fingerprints

```text
Selected best checkpoint:
682e517744fab52367702290673c8c5b187f965caf4e075a7c2d29e79459ce98

Final case metrics:
e09895011115c570bd028f8fa3713df2cfdf8bc8020c3b5ea51ffd879f6f7d7a

Final raw logits:
e7cfdb289bcdb0bb7a19af14914c7b745abaaa0b97af64b3535e58fb45870361

Final summary:
3f6b3123fff803ce23da02567343de5ac0e665f3c27060209ee869b670892840

Final result contract:
9b393d4bf2c862685a008901d00332203f870bd6032874836c8389dca9576b71

Analysis-ready bundle:
d70f21bff0006e094c5d567d307c0664b41a11e7250a0e69aaec610e1810132d
```

---

## Ordinary-Git boundary

The repository contains the scripts and lightweight final-analysis artifacts required for audit and Phase-7 analysis.

Large runtime files are not necessarily tracked in ordinary Git, including large checkpoints, full 20-epoch binary training streams, tar archives, RunPod logs, and environment caches.

Therefore:

> `git pull` is sufficient for the planned post-hoc Phase-7 analysis, but not necessarily for recreating the full CUDA production workspace byte-for-byte.

---

## Phase-7 handoff

Primary local-analysis artifact:

```text
data/experimental/phase_6/full_training/100pct/final_test/
analysis_ready_test_cases.parquet
```

It contains 20,264 final-test cases and 74 columns with ranking metrics plus temporal, cold-start, pair-history, and structural metadata.
