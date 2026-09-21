# Phase 6 Closure Report - ITRS Reproduction on Crunchbase

**Project:** Reproduction of *A Trend-aware Investment Target Recommendation System with Heterogeneous Graph (ITRS)*  
**Dataset adaptation:** Audited Crunchbase reconstruction  
**Phase:** 6 - CUDA Production Training, Final Model Selection, and One-Shot Test  
**Status:** COMPLETE / FROZEN  
**Initial Phase-6 closure publication:** `master` commit `e7df969b1cd6ea975ebaf689cf23777f49ca45e5`

---

## Executive summary

Phase 6 executed the production experiment specified and frozen in Phase 5.

It:

- qualified a strict FP32 CUDA runtime on RTX 4090;
- benchmarked practical throughput;
- exercised reduced-supervision pilots;
- froze all 20 full-data epoch streams before production training;
- completed 20 epochs and 209,620 optimizer steps;
- validated all 2,251 validation cases after every epoch;
- selected display Epoch 19 using validation only;
- preserved test isolation through selection;
- authorized the final test before observing test metrics;
- executed exactly one final test pass over 20,264 cases;
- froze final per-case metrics, raw logits, summary, result contract, and hashes;
- built an analysis-ready 20,264-row post-hoc dataset.

Final one-shot test:

| Metric | Value |
|---|---:|
| HR@10 | **0.546387682590** |
| NDCG@10 | **0.398709730124** |
| Hits@10 | **11,072 / 20,264** |
| Mean positive rank | **23.034347** |
| Median positive rank | **7** |

The central integrity claim is that the final test was not used for training, runtime design, early stopping, or checkpoint selection.

---

# 1. Frozen Phase-5 inputs

```text
Training positives / epoch:     1,073,249
Negatives / positive:           4
Examples / epoch:               5,366,245
Batch size:                     512
Batches / epoch:                10,481
Epochs:                         20
Optimizer steps:                209,620
Validation cases:               2,251
Test cases:                     20,264
Candidates / evaluation case:   100
```

Optimizer:

```text
Adam
lr = 0.001
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0
```

Checkpoint selection:

```text
1. maximum validation NDCG@10
2. maximum validation HR@10
3. earliest epoch
```

Ranking:

```text
raw logit descending
startup_local ascending tie-break
```

---

# 2. CUDA portability and qualification

The canonical initialization was made portable through a frozen initial-state artifact so Linux/CUDA construction reproduced the authoritative Phase-4 state before the existing hash gate ran.

```text
Initial model logical SHA256:
49e822ea7fad35c458f47e134c94c05eac099b68c5c468e2c71559c8c88998ab

Portable initial-state file SHA256:
a7451fa138e440dd6dd6e563759844982ec68126d9fc02febdc463e203ee6224
```

Production environment:

```text
Python:        3.11.14
PyTorch:       2.7.0+cu118
CUDA runtime:  11.8
GPU:           NVIDIA GeForce RTX 4090
```

Strict CUDA policy disabled TF32 and enabled deterministic algorithms.

The packed + sparse CUDA path passed 22/22 frozen numerical-policy checks.

---

# 3. Throughput and checkpointing

Production-like throughput:

```text
Mean end-to-end:  3.176 s / batch
Median:           3.183 s / batch
p95:              3.418 s / batch
Mean GPU time:    3.134 s / batch
```

Operational restart checkpoints were persisted every 100 batches and before validation at each epoch boundary.

---

# 4. Reduced-supervision pilots

Best observed validation metrics:

| Budget | HR@10 | NDCG@10 |
|---:|---:|---:|
| 1% | 0.1630 | 0.0802 |
| 5% | 0.2674 | 0.1413 |
| 10% | 0.3510 | 0.1887 |

The pilots were diagnostics; the scientific target remained the full 100% training run.

---

# 5. Full-data stream freeze

Phase 6.9a froze all 20 epoch-dependent negative matrices and example orders before training.

```text
Epoch logical hashes:    20/20 PASS
Binary stream files:     40/40 PASS
Anchor epochs 0/1/2/19:  PASS
Optimizer steps:         0
Validation:              0
Test:                    0
```

Stream registry SHA256:

```text
e6970082a06b3b097e0858e4d0761240c99c3a255b3227339cea6243821e2eb1
```

V1 generated the streams successfully but failed only in a pandas CSV fingerprint bookkeeping comparison. V2 independently verified the persisted files and finalized the freeze without regenerating training streams.

---

# 6. Full production training

Phase 6.9b executed:

```text
20 epochs
5,366,245 examples / epoch
10,481 batches / epoch
209,620 optimizer steps
20 full validation passes
```

The test remained disabled.

Selected checkpoint:

```text
Display epoch:              19
Epoch index:                18
Validation HR@10:           0.5362061306086184
Validation NDCG@10:         0.3896115434372846
```

Key hashes:

```text
best.pt
682e517744fab52367702290673c8c5b187f965caf4e075a7c2d29e79459ce98

latest.pt
024acf6b9c35fd0020476eb2f88369fa1d0fa6d9cf27d0cc123794aa54bb5fb2

epoch_metrics.csv
35f2781127a1a68e2ca88c3a46cdc95ec7039e5c1d5f71908de60894efe3a42d

Phase-6.9b contract
795672172d34e63e4a88dfc6033518135a50b9ff449625bc71c446f431337401
```

---

# 7. Final-test authorization and one-shot execution

The evaluator enforced three distinct modes:

```text
--preflight-only
--authorize
--run-once
```

Preflight scored zero test cases.

Authorization fixed:

```text
Selected display epoch:       19
Test cases:                    20,264
Candidates / case:            100
Authorized passes:            1
Test cases before auth:        0
Checkpoint reselection:        false
```

Authorization SHA256:

```text
90135d1617b6eaf753919e344f91c97b73a0b5a628f61b73948399dfc6f2f0ba
```

Final test:

```text
HR@10:                0.5463876825898144
NDCG@10:              0.39870973012365235
Hits@10:              11,072
Cases:                20,264
Mean positive rank:   23.034346624555862
Median positive rank: 7.0
Elapsed:              71.469 s
```

Integrity:

```text
passes_executed = 1
model_state_unchanged = true
rng_state_unchanged = true
gradients_created = false
optimizer_instantiated = false
backward_executed = false
checkpoint_reselection_performed = false
```

---

# 8. Final artifact fingerprints

```text
final_test_case_metrics.parquet
e09895011115c570bd028f8fa3713df2cfdf8bc8020c3b5ea51ffd879f6f7d7a

final_test_raw_logits.npy
e7cfdb289bcdb0bb7a19af14914c7b745abaaa0b97af64b3535e58fb45870361

final_test_summary.json
3f6b3123fff803ce23da02567343de5ac0e665f3c27060209ee869b670892840

phase_6_9c_final_test_result_V2.json
9b393d4bf2c862685a008901d00332203f870bd6032874836c8389dca9576b71
```

---

# 9. Analysis-ready Phase-7 handoff

Phase 6.9d created `analysis_ready_test_cases.parquet`:

```text
Rows:     20,264
Columns:  74
SHA256:   d70f21bff0006e094c5d567d307c0664b41a11e7250a0e69aaec610e1810132d
```

Selected counts:

```text
New-to-investor:      16,446
Cold investors:        3,165
Cold startups:         8,104
Connected investors:   2,225
Connected startups:    4,011
```

The bundle joins final metrics with temporal, pair-history, cold-start, and graph-coverage metadata. It does not rescore test.

---

# 10. Metric interpretation boundary

Each test event ranks one observed positive against 99 frozen negatives.

```text
HR@10 = 1 if positive_rank <= 10 else 0
NDCG@10 = 1/log2(rank+1) if rank <= 10 else 0
```

Therefore the final HR@10 is a sampled 100-candidate ranking result, not a full-catalog top-10 claim.

The positive label is an observed investment event, not future startup success.

---

# 11. Repository reproducibility boundary

The repository tracks the relevant Phase-6 scripts and lightweight final-analysis products.

Large transient/runtime artifacts, such as full checkpoints, full epoch binary streams, tar archives, and RunPod logs, are not guaranteed to be stored in ordinary Git.

Consequently:

> A local `git pull` is sufficient for post-hoc Phase-7 analysis using the committed final bundle, but not necessarily for reconstructing the entire production workspace byte-for-byte without the separately frozen large artifacts.

---

# 12. Closure decision

Phase 6 is CLOSED / FROZEN.

Final test:

```text
HR@10:    0.546387682590
NDCG@10:  0.398709730124
```

Any Phase-7 work must treat the selected checkpoint and final test as immutable.
