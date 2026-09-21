# Reproduction Log Entry - Phase 5: Training and Evaluation Reconstruction

**Project:** Reproduction of *A Trend-aware Investment Target Recommendation System with Heterogeneous Graph (ITRS)* using audited Crunchbase data  
**Status:** COMPLETE / FROZEN  
**Previous phase:** Phase 4 - Model Reconstruction  
**Next phase:** Phase 6 - CUDA Qualification and Production Training  
**Phase role:** Freeze and prove the complete training/evaluation protocol before production training.

---

## Phase objective

Phase 5 converted the frozen Phase-4 model into a fully specified, auditable training and evaluation experiment. The phase deliberately separated scientific semantics from runtime engineering. Its objective was not to maximize performance, but to prove that every decision required for training, validation, checkpoint selection, and eventual test evaluation had been fixed before production training.

The major questions were:

1. What is a valid training negative at historical period `h`?
2. How are the 100-candidate validation/test sets constructed?
3. What optimizer, batching, shuffling, checkpoint, and selection rules are used?
4. Can the complete model execute forward, backward, Adam updates, checkpoint/resume, and ranking deterministically?
5. Can a faster runtime be accepted without silently changing the experiment?

Phase 5 closed all five questions.

---

## Chronological subphase record

| Subphase | Objective | Main result / decision |
|---|---|---|
| 5.1.1 | Audit and freeze training-negative semantics | Four negatives per positive; exclude pairs positive at or before period `h`; future-positive pairs remain eligible before their first positive event |
| 5.1.2 | Audit and freeze T60 evaluation candidates | One positive + 99 frozen negatives per T60 event; 22,515 total cases; 2,251 validation; 20,264 test |
| 5.2 | Freeze optimizer and training control | Adam, lr=0.001, 20 epochs, batch 512, no early stopping, validation after every epoch |
| 5.3.1 | Reconstruct side-effect-free training runtime | Exact Phase-4 topology/forward reused; real-data forward/BCE/backward verified |
| 5.3.1l-5.3.1m | Freeze epoch-0 stream and first Adam step | Deterministic stream, gradients, model update, and optimizer-state fingerprints frozen |
| 5.3.2 | Prove checkpoint/resume | Resumed trajectory reproduced uninterrupted training trajectory |
| 5.3.3 | Freeze validation ranking semantics | Raw-logit ranking; deterministic tie-break; HR@10 and NDCG@10 frozen |
| 5.3.5 | Generalize stream generation to 20 epochs | Deterministic negative matrices and shuffled orders frozen for all epoch indices |
| 5.3.6-5.3.7 | Launch/integrity closure | Production launch contract and full reproduction-integrity gate closed |
| 5.4 | Runtime feasibility and acceleration audits | Exact CPU impractical; sparse-gradient path accepted under frozen numerical-equivalence policy; CUDA handoff justified |

---

## Frozen training-negative semantics

For a positive event `(o,b,h)` with investor `o`, startup `b`, and historical segment `h`, a startup is eligible as a negative when:

```text
startup_role_node(b)
AND
NOT EXISTS positive_event(o,b,s) WITH segment_number(s) <= h
```

Consequences:

- a startup that becomes positive only in a future period remains eligible before that future positive;
- a startup already positive for the investor at or before `h` is ineligible;
- no T60 event is used for training;
- four negatives are sampled for every positive;
- the four negatives are sampled without replacement within a positive;
- negatives regenerate independently for every epoch.

Training volume per epoch:

```text
Positive events:          1,073,249
Negatives / positive:     4
Negative examples:        4,292,996
Examples / epoch:         5,366,245
Batch size:               512
Batches / epoch:          10,481
Final batch size:         485
Epochs:                   20
Optimizer steps:          209,620
```

This negative-sampling runtime is a `PAPER_UNSPECIFIED_REPRODUCTION_CHOICE`.

---

## Frozen evaluation candidate semantics

Every T60 evaluation event is treated as one ranking case.

For each case:

```text
1 focal positive startup
+ 99 frozen sampled negative startups
-----------------------------------
100 candidates
```

Case counts:

```text
Total T60 evaluation cases:  22,515
Validation cases:             2,251
Test cases:                  20,264
```

The negative pool excludes the focal positive startup and startups that formed an observed positive investor-startup pair before T60. Other T60 positives are not removed merely because they are positive elsewhere in the same held-out period.

A collision audit found 50 realized slots/cases where a sampled negative was another T60 positive for the focal investor. These collisions were retained after freezing; no post-hoc resampling was allowed.

Key frozen fingerprints:

```text
Evaluation negative matrix:
7f98269d0291382dacfc783ffd66ad6d2c2f8775877d57dc0d83504e40d8716d

Evaluation case manifest:
44b4b7e1ec1b1978249318a080df02d0d9f617845263513594de64e71b969e0c

Evaluation generation manifest:
b3e43fa19deb57ce55b2499838055e53c520bc0395912ef0de7a340c55ac20b8
```

This candidate construction is protected as an `EVALUATION_INTEGRITY_GUARD`.

---

## Frozen optimizer and training control

```text
Optimizer:          Adam
Learning rate:      0.001
Betas:              (0.9, 0.999)
Epsilon:            1e-8
Weight decay:       0
Epochs:             20
Early stopping:     NO
Validation:         after every completed epoch
Checkpoint choice:  max validation NDCG@10
Secondary:          max validation HR@10
Final tie-break:    earliest epoch
Test influence:     FORBIDDEN
```

---

## Exact numerical anchors

A real T60 validation case produced:

```text
logit:       -0.2320666015
probability:  0.4422423244
BCE:          0.8158972263
```

All 32 trainable parameter tensors received finite, non-zero gradients.

Frozen epoch-0 fingerprints:

```text
Positive stream:
73b074a80675793b811fbdc8a0609883c857fb2a687a2e01c31865ade5b509d1

Negative matrix:
47015b147b1949562c0f6737a6f3a3f2d7cabd2d2202e4e57456d884a1e23fe6

Example order:
0156be3ee623ade1ae696557337bfb324e9011adb7df8be9648ecb0a426c134e

First batch:
8408432b944bcd0805af9c34ff1b2db3ea938e0649a75d381b7839b86cd280ea
```

First Adam step:

```text
BCE:                 0.7080879807
Mean logit:          0.0455230400
Logit SHA256:        35b89aaed29d51d2ebb7ba1cadf2dc4bb5e8f81cf3aa78bc216b3cc6fed13845
Gradient SHA256:     8c542430813d8ca91b8397409954ea92295a2b55bcc420661783fb865010845d
Post-step model:     42a521f11d8f24e4144d0215d6e1b34d5f8bf0c2d8848624e4f7c3130699035d
Optimizer state:     5ce2683c21f456b9d5d15eb876b049c5e6db1215db5a026630f093f7f9d49891
```

Checkpoint/resume reproduced the uninterrupted trajectory.

---

## Frozen ranking semantics

For every validation or test case, all 100 candidates receive a raw model logit.

```text
Primary key:    raw logit descending
Tie-break:      startup_local ascending
Positive rank:  1-based
```

Per-case metrics:

```text
HR@10 = 1 if positive_rank <= 10 else 0

NDCG@10 =
    1 / log2(positive_rank + 1), if positive_rank <= 10
    0,                            otherwise
```

Because there is exactly one relevant startup per case, `IDCG@10 = 1`.

The untrained validation baseline was:

```text
HR@10:          0.091514882275
NDCG@10:        0.040193099163
Hits@10:        206 / 2,251
Mean rank:      50.706353
Median rank:    50
```

---

## Runtime engineering conclusion

Representative local runtimes:

| Runtime | Status | Approx. sec/batch | Approx. 20-epoch runtime |
|---|---:|---:|---:|
| Canonical exact CPU | byte-exact reference | 25.249 | ~61.3 days |
| Dense MPS | numerically acceptable but slow | ~31.5 warm | ~76.4 days |
| Packed MPS | numerically acceptable but slow | ~15.3 warm | ~37.0 days |
| `D_BOTH_SPARSE` CPU | PASS | ~6.8 | ~16.5 days |
| Packed + sparse CPU | PASS | ~6.9 | ~16.6 days |

Phase 5 therefore closed with compute capacity, not unresolved scientific semantics, as the blocker.

---

## Phase 5 closure

At closure, the full training/evaluation specification was frozen and production execution was handed to Phase 6. No final trained test metric was produced in Phase 5.

Authoritative comprehensive report:

```text
docs/ITRS_Reproduction_Phase_5_Closure_and_Phase_6_Handoff.md
```
