# Reproduction Log Entry - Phase 6: CUDA Production Training and Final Evaluation

**Project:** Reproduction of *A Trend-aware Investment Target Recommendation System with Heterogeneous Graph (ITRS)* using audited Crunchbase data  
**Status:** COMPLETE / FROZEN  
**Previous phase:** Phase 5 - Training and Evaluation Reconstruction  
**Next phase:** Phase 7 - Post-hoc Final-Test Analysis and Thesis Diagnostics  
**Production environment:** NVIDIA GeForce RTX 4090, PyTorch 2.7.0+cu118, CUDA 11.8

---

## Phase objective

Phase 6 executed the experiment frozen in Phase 5:

1. qualify the accelerated CUDA implementation under the predeclared numerical-equivalence policy;
2. establish practical production throughput;
3. execute the full 100% 20-epoch schedule with validation-only checkpoint selection;
4. evaluate the frozen selected checkpoint exactly once on the untouched test set.

All four objectives were completed.

---

## Chronological record

| Subphase | Objective | Main result |
|---|---|---|
| 6.1 / 6.1b | CUDA portability and strict FP32 qualification | RTX 4090 path reproduced frozen numerical anchors |
| 6.2 | Packed + sparse CUDA qualification | 22/22 numerical-policy checks PASS |
| 6.3 | Production-like throughput benchmark | ~3.18 s/batch mean |
| 6.4-6.7 | 1%, 5%, 10% reduced-supervision pilots | End-to-end CUDA training/validation path exercised |
| 6.9a | Freeze full 20-epoch streams | 20/20 logical hashes and 40/40 binary stream files PASS |
| 6.9b | Full 100% production training | 20 epochs, 209,620 optimizer steps, 20 validations |
| 6.9c | One-shot final test | 20,264 cases scored exactly once |
| 6.9d | Analysis-ready bundle | 20,264 rows x 74 columns, no rescoring |

---

## CUDA qualification

Strict FP32 settings:

```text
TF32 matmul:               disabled
TF32 cuDNN:                disabled
float32 matmul precision:  highest
cuDNN benchmark:           disabled
cuDNN deterministic:       enabled
deterministic algorithms:  enabled
```

The production path combined packed embedding dispatch with `D_BOTH_SPARSE`.

Phase 6.2 passed 22/22 frozen numerical-policy checks.

---

## Throughput

Qualified RTX 4090 benchmark:

```text
Mean end-to-end:  3.176 s/batch
Median:           3.183 s/batch
p95:              3.418 s/batch
Mean GPU:         3.134 s/batch
```

Restart checkpoints were written every 100 batches and at the end-of-epoch/pre-validation boundary.

---

## Reduced-supervision pilots

Observed best validation results:

| Training budget | HR@10 | NDCG@10 |
|---:|---:|---:|
| 1% | 0.1630 | 0.0802 |
| 5% | 0.2674 | 0.1413 |
| 10% | 0.3510 | 0.1887 |

These were pilot/scaling diagnostics, not final-test results.

---

## Full-data stream freeze

Phase 6.9a froze all 20 production epoch streams before training:

```text
Epoch logical hashes: 20/20 PASS
Binary stream files:  40/40 PASS
Optimizer steps:      0
Validation scored:    0
Test accessed:        NO
```

Registry SHA256:

```text
e6970082a06b3b097e0858e4d0761240c99c3a255b3227339cea6243821e2eb1
```

---

## Full 100% production training

```text
Epochs:                    20
Positive events / epoch:   1,073,249
Negatives / positive:      4
Examples / epoch:          5,366,245
Batches / epoch:           10,481
Total optimizer steps:     209,620
Validation cases / epoch:  2,251
```

Best checkpoint:

```text
Display epoch:          19
Epoch index:            18
Validation HR@10:       0.536206130609
Validation NDCG@10:     0.389611543437
```

Best-checkpoint SHA256:

```text
682e517744fab52367702290673c8c5b187f965caf4e075a7c2d29e79459ce98
```

Training closed with `test_accessed=false` and `test_scored=false`.

---

## One-shot final test

Phase 6.9c separated:

```text
--preflight-only
--authorize
--run-once
```

Authorization was persisted before test metrics were observed.

Final test:

```text
Cases:                20,264
Candidates / case:    100
HR@10:                0.546387682590
NDCG@10:              0.398709730124
Hits@10:              11,072
Mean positive rank:   23.034347
Median positive rank: 7
Evaluation time:      71.47 s
```

Integrity:

```text
Model parameters changed:     NO
Optimizer instantiated:       NO
Backward executed:            NO
Checkpoint reselection:       NO
Final-test passes executed:   1
```

Final result contract SHA256:

```text
9b393d4bf2c862685a008901d00332203f870bd6032874836c8389dca9576b71
```

---

## Analysis-ready Phase-7 handoff

Phase 6.9d created:

```text
data/experimental/phase_6/full_training/100pct/final_test/
analysis_ready_test_cases.parquet
```

```text
Rows:     20,264
Columns:  74
SHA256:   d70f21bff0006e094c5d567d307c0664b41a11e7250a0e69aaec610e1810132d
```

Selected subgroup counts:

```text
New-to-investor:      16,446
Cold investors:        3,165
Cold startups:         8,104
Connected investors:   2,225
Connected startups:    4,011
```

The bundle is post-hoc only: no model loading, inference, optimizer, backward, or test rescoring.

---

## Phase 6 closure

Phase 6 is COMPLETE / FROZEN.

Final 100% reproduction test result:

```text
HR@10:    0.546387682590
NDCG@10:  0.398709730124
```

The final test must not be rerun for model selection. Phase 7 is limited to post-hoc diagnosis unless a separately defined new experiment is created.
