# ITRS Reproduction - Phase 5 Closure and Phase 6 Handoff

**Project:** Reproduction of *A Trend-aware Investment Target Recommendation System with Heterogeneous Graph (ITRS)* using audited Crunchbase data  
**Document status:** Phase 5 COMPLETE / FROZEN  
**Phase boundary decision:** Production training, final model selection, test evaluation, paper comparison, and final experimental conclusions are moved to Phase 6.  
**Date of closure:** August 23, 2026

---

## Executive Summary

Phase 5 established a complete, auditable, and reproducible training and evaluation pipeline for the ITRS reproduction. The phase did **not** produce final trained-model recommendation metrics because the full 20-epoch experiment is computationally impractical on the currently available Mac hardware.

The work completed in Phase 5 includes:

- freezing the semantics of training negatives and evaluation negatives;
- freezing the 20-epoch training schedule, optimizer, batching, shuffling, checkpoint-selection logic, and test isolation rules;
- proving end-to-end forward, backward, and Adam updates against frozen numerical anchors;
- proving exact checkpoint/resume behavior;
- proving the complete epoch-0 training stream and generalized 20-epoch stream construction;
- freezing T60 validation ranking semantics and establishing an untrained validation baseline;
- closing a full reproduction-integrity gate before any production training;
- benchmarking the exact CPU path;
- profiling the runtime bottleneck;
- testing several acceleration strategies on CPU and Apple MPS;
- freezing a numerical-equivalence policy before evaluating non-byte-exact accelerated runtimes;
- identifying the fastest validated local runtime and demonstrating that local full training remains impractical.

The strongest validated local runtime is `D_BOTH_SPARSE` on CPU at approximately **6.8 seconds per batch**, compared with approximately **25.25 seconds per batch** for the canonical exact CPU path. Even after this approximately 3.7x acceleration, the projected runtime for the full 20-epoch experiment remains approximately **16.5 days**. Apple MPS was numerically equivalent but slower, and packing plus sparse CPU execution did not improve further.

Therefore, Phase 5 is closed with the following scientific and engineering conclusion:

> The ITRS reproduction pipeline is reconstructed, validated, and ready for production training. The remaining blocker is computational capacity rather than unresolved data, model, training, or evaluation semantics.

The final production experiment is intentionally deferred to **Phase 6**, where the frozen pipeline will be moved to CUDA-capable hardware, validated against the already-frozen numerical-equivalence policy, trained for 20 epochs, evaluated on validation after every epoch, and tested exactly once using the best validation checkpoint.

---

# 1. Phase 5 Scope and Closure Decision

## 1.1 Original role of Phase 5

Phase 5 was the transition from the frozen model reconstruction of Phase 4 to a reproducible training and evaluation experiment. Its purpose was to answer four questions:

1. What exactly constitutes a valid negative example during training?
2. What exactly constitutes a valid candidate set during T60 evaluation?
3. Can the reconstructed ITRS model execute deterministic forward, backward, optimization, checkpoint, and validation operations?
4. Can the resulting training procedure be executed in a practical amount of time on the available hardware?

The first three questions were fully resolved. The fourth was resolved negatively for the current local hardware.

## 1.2 Revised project boundary

As of the Phase 5 closure commit, the project adopts the following boundary:

### Phase 5 - COMPLETE / FROZEN

Phase 5 now contains:

- training-data semantics;
- negative sampling;
- evaluation candidate construction;
- optimizer and training control;
- deterministic stream generation;
- end-to-end training-path integrity;
- checkpoint/resume proof;
- validation semantics;
- runtime feasibility;
- acceleration audits;
- numerical-equivalence policy;
- local-compute feasibility conclusion.

### Phase 6 - OPEN / PENDING

Phase 6 will contain:

- CUDA runtime validation;
- full 20-epoch production training;
- epoch-level validation;
- best-checkpoint selection;
- one final test evaluation;
- final HR@10 and NDCG@10;
- comparison with the ITRS paper;
- additional diagnostic analyses;
- final reproduction conclusions.

This reclassification is a project-management decision. It does not alter any frozen training or evaluation semantics.

---

# 2. Frozen Inputs Entering Phase 5

Phase 5 treated the outputs of Phases 1 through 4 as fixed inputs.

## 2.1 Phase 1 - Canonical investment interactions

Canonical file:

`data/processed/interactions.parquet`

Key count:

- total canonical interaction rows: **1,208,051**

No Phase 5 procedure modifies this artifact.

## 2.2 Phase 2 - Temporal experiment split

Canonical file:

`data/experimental/phase_2/model_ready/interactions_itrs_temporal_split.parquet`

Frozen experiment structure:

- T0 through T60 experiment rows: **1,195,937**
- post-endpoint rows excluded: **12,114**
- T0-T59 training-history rows: **1,173,422**
- T60 rows: **22,515**
- T60 investors: **11,884**
- T60 startups: **8,992**
- unique T60 investor-startup pairs: **22,327**
- validation events: **2,251**
- test events: **20,264**

The T60 holdout is event-level, unstratified, and generated with seed 42. T60 is completely excluded from training.

## 2.3 Phase 3 - Structural graph

Canonical graph artifacts include:

- `node_index.parquet`
- `relation_index.csv`
- `edge_index.npy`
- `edge_type.npy`
- `edge_manifest.parquet`
- `graph_variant_masks.npz`
- `model_ready_graph_metadata.json`

Frozen graph statistics:

- total role nodes: **477,564**
- investors: **165,975**
- startups: **311,589**
- directed structural edges: **158,818**
- typed relation channels: **12**
- structural isolates retained: **402,807**

## 2.4 Phase 4 - Model reconstruction

The reconstructed ITRS model was frozen before Phase 5 training logic was introduced.

Core architecture:

- investor latent embedding dimension: 40
- startup latent embedding dimension: 40
- description representation `F_d`: 40
- trend representation `F_t`: 40
- structural representation `F_s`: 40
- pair representation: 280 dimensions
- scoring MLP: `280 -> 128 -> 64 -> 32 -> 16 -> 1`
- objective: raw logit with `BCEWithLogitsLoss`
- trainable parameters: **19,217,929**
- trainable tensors: **32**
- optimizer parameters are shared with the frozen model; no duplicated latent tables are introduced.

Trend reconstruction:

- T_h uses only T0 through T(h-1);
- T60 uses T0-T59;
- repeated investor-startup mentions within a historical period collapse to one pair;
- empty periods are represented by an 80-dimensional zero vector;
- attention query dimension: 80;
- attention item dimension: 80;
- GRU input dimension: 80;
- GRU hidden dimension: 40;
- GRU layers: 2.

Description reconstruction:

- Doc2Vec input dimension: 32;
- category multi-hot input dimension: 802;
- text branch: `Linear(32,20) -> ReLU`;
- category branch: `Linear(802,20) -> ReLU`;
- concatenated description output: 40.

Structural reconstruction:

- two RGCN layers;
- five bases;
- 40-dimensional hidden/output representations;
- relation-specific incoming mean normalization;
- separate root transform;
- ReLU after each layer;
- no dropout, residual connection, or normalization layer;
- isolates use the root-only path.

Frozen initialization:

- Kaiming normal;
- `a=0.0`;
- `mode="fan_in"`;
- `nonlinearity="relu"`;
- zero biases;
- global seed 42;
- PyTorch 2.7.0 reference runtime.

Initial model-state SHA256:

`49e822ea7fad35c458f47e134c94c05eac099b68c5c468e2c71559c8c88998ab`

---

# 3. Phase 5.1 - Training and Evaluation Example Semantics

## 3.1 Negative-space audit

Before freezing negative sampling, the available startup universe and observed-pair space were audited.

Key counts:

- startup role-node universe: **311,589**
- startups observed anywhere in T0-T60: **309,306**
- startups observed before T60: **304,591**
- startup role nodes never observed in T0-T60: **2,283**
- distinct investor-startup pairs in T0-T60: **981,535**
- positive target events in T1-T60: **1,095,764**

A stricter "prefix prior" interpretation was explicitly rejected because it would incorrectly exclude future-positive startups from being sampled as negatives before their first observed positive event. That interpretation would have removed approximately 45.14% of training positives and 40.16% of T60 cases from the candidate logic.

## 3.2 Frozen training-negative definition

For a positive event `(o,b,h)` with investor `o`, startup `b`, and segment `h`, a startup candidate is eligible as a negative if:

1. it is a valid startup-role node; and
2. there is **no** observed positive event for `(o,b)` in any segment `s <= h`.

Formally:

`eligible_negative(o,b,h) := startup_role_node(b) AND NOT EXISTS positive_event(o,b,s) WITH segment_number(s) <= h`

Important consequences:

- a startup that becomes positive only in a future period `s > h` is still an eligible negative at period `h`;
- a pair that is already positive at or before `h` is ineligible;
- no additional startup filter is applied;
- T60 is never used to create training examples.

Classification:

`PAPER_UNSPECIFIED_REPRODUCTION_CHOICE`

## 3.3 Frozen training-negative runtime

Training uses:

- negatives per positive: **K = 4**
- negatives regenerated independently every epoch;
- uniform sampling over the eligible startup set;
- sampling without replacement within each positive's four negatives;
- deterministic epoch-specific RNG derived from:
  - namespace: `ITRS_PHASE5_TRAIN_NEGATIVE`
  - base seed: 42
  - SHA256(namespace | base_seed | epoch)
  - first 8 bytes interpreted as little-endian unsigned integer
  - reduced modulo `2^63 - 1`
  - NumPy PCG64 generator.

Per epoch:

- positives: **1,073,249**
- negatives: **4,292,996**
- total examples: **5,366,245**

Epoch 0 audit:

- accepted negatives: **4,292,996**
- repaired/rejected raw draws: **1,759**
  - forbidden because positive at or before h: 1,743
  - duplicate within same positive: 16
- accepted future-positive negatives: **1,264**
- accepted never-positive negatives: **4,291,732**

Epoch-0 negative matrix shape:

`(1,073,249, 4)` as int32

Epoch-0 negative matrix SHA256:

`47015b147b1949562c0f6737a6f3a3f2d7cabd2d2202e4e57456d884a1e23fe6`

## 3.4 Frozen T60 evaluation candidates

Evaluation follows the paper-style sampled ranking protocol:

- one focal positive;
- 99 sampled negatives;
- 100 candidates per evaluation case;
- ranking metrics: HR@10 and NDCG@10.

T60 case counts:

- total: **22,515**
- validation: **2,251**
- test: **20,264**

Negative eligibility for a T60 case excludes:

- any startup with a positive pair for the focal investor before T60;
- the focal positive startup itself.

Other T60 positives are not removed from the sampled-negative pool because they are simultaneous held-out labels rather than prior knowledge.

Important audit result:

- realized collision slots/cases with another T60 positive: **50**
- no collision resampling was performed after freezing the candidate matrix.

Frozen candidate hashes:

- logical evaluation matrix SHA256:  
  `33c660d79640b690bfd97f65a0b7cd9bfdd2950ce21aedd1f1e6c773228da88c`
- physical negative matrix SHA256:  
  `7f98269d0291382dacfc783ffd66ad6d2c2f8775877d57dc0d83504e40d8716d`
- case-manifest SHA256:  
  `44b4b7e1ec1b1978249318a080df02d0d9f617845263513594de64e71b969e0c`

Classification:

`EVALUATION_INTEGRITY_GUARD`

---

# 4. Phase 5.2 - Training Control and Model Selection

## 4.1 Frozen optimizer

Optimizer:

- PyTorch 2.7.0 `Adam`
- learning rate: **0.001**
- betas: **(0.9, 0.999)**
- epsilon: **1e-8**
- weight decay: **0**
- no alternative optimizer behavior introduced.

## 4.2 Frozen training schedule

The paper specifies 20 training epochs. The reproduction therefore freezes:

- epochs: **20**
- positives per epoch: **1,073,249**
- total examples per epoch: **5,366,245**
- batch size: **512**
- full batches per epoch: **10,480**
- final batch size: **485**
- total batches per epoch: **10,481**
- total example presentations: **107,324,900**
- total optimizer steps: **209,620**
- no early stopping.

## 4.3 Deterministic epoch shuffling

All examples are shuffled once per epoch.

Seed namespace:

`ITRS_PHASE5_TRAIN_ORDER`

The order seed is deterministically derived independently for every epoch from base seed 42.

Epoch-0 order SHA256:

`0156be3ee623ade1ae696557337bfb324e9011adb7df8be9648ecb0a426c134e`

Epoch-0 first-batch SHA256:

`8408432b944bcd0805af9c34ff1b2db3ea938e0649a75d381b7839b86cd280ea`

## 4.4 Frozen validation and checkpoint selection

Validation is performed after every completed epoch.

Checkpoint selection rules:

1. primary criterion: highest validation NDCG@10;
2. secondary criterion: highest validation HR@10;
3. final tie-break: earliest epoch.

The test set is forbidden for:

- early stopping;
- hyperparameter choice;
- checkpoint selection;
- runtime optimization;
- numerical-equivalence decisions.

The test set will be evaluated exactly once after the best checkpoint has been selected.

Classification:

`EVALUATION_INTEGRITY_GUARD`

---

# 5. Phase 5.3 - End-to-End Training Integrity and Reproducibility

Phase 5.3 proved that the frozen data, model, optimizer, and evaluation semantics can operate together reproducibly before any production training is launched.

## 5.1 Side-effect-free model loading

The canonical Phase 4 topology and forward implementation were loaded through a sanitized, side-effect-free runtime path.

Canonical model-topology source:

`scripts/phase_4_7_1b_freeze_neural_initialization_seed_contract.py`

Canonical forward source:

`scripts/phase_4_6_2_end_to_end_itrs_forward_bce_audit.py`

The Phase 5 executor reuses the exact seven frozen forward methods rather than reconstructing an independent training model.

## 5.2 Real validation-case forward/backward proof

A real T60 validation case was executed end-to-end.

Observed values:

- logit: **-0.2320666015**
- probability: **0.4422423244**
- BCE: **0.8158972263**

All 32 trainable tensors received finite, non-zero gradients.

This established that the complete reconstructed architecture participates in backpropagation.

## 5.3 Exact epoch-0 stream proof

Frozen epoch-0 anchors:

- positive examples: **1,073,249**
- positive stream SHA256:  
  `73b074a80675793b811fbdc8a0609883c857fb2a687a2e01c31865ade5b509d1`
- negative matrix SHA256:  
  `47015b147b1949562c0f6737a6f3a3f2d7cabd2d2202e4e57456d884a1e23fe6`
- shuffled order SHA256:  
  `0156be3ee623ade1ae696557337bfb324e9011adb7df8be9648ecb0a426c134e`
- first batch SHA256:  
  `8408432b944bcd0805af9c34ff1b2db3ea938e0649a75d381b7839b86cd280ea`

## 5.4 Exact first-batch backward and Adam proof

Frozen batch-0 anchors:

- batch size: **512**
- positives: 105
- negatives: 407
- unique `(Investor,h)` keys: 510
- description nodes required: 36,759
- BCE: **0.7080879807**
- mean logit: **0.0455230400**

Hashes:

- logit SHA256:  
  `35b89aaed29d51d2ebb7ba1cadf2dc4bb5e8f81cf3aa78bc216b3cc6fed13845`
- gradient SHA256:  
  `8c542430813d8ca91b8397409954ea92295a2b55bcc420661783fb865010845d`
- post-step model SHA256:  
  `42a521f11d8f24e4144d0215d6e1b34d5f8bf0c2d8848624e4f7c3130699035d`
- optimizer-state SHA256:  
  `5ce2683c21f456b9d5d15eb876b049c5e6db1215db5a026630f093f7f9d49891`

All 32 gradients were finite and non-zero.

## 5.5 Checkpoint/resume proof

The runtime was checkpointed after batch 0 and restored before batch 1. The resumed trajectory reproduced the uninterrupted trajectory exactly.

Frozen batch-1 anchors:

- BCE: **0.6636360884**
- logit SHA256:  
  `cfbc4106103abf9478b8f04f0e0d909bed37659e5ee7e29257bce0a7dd4beb26`
- gradient SHA256:  
  `8c066fd5f8002e1edd0a282f4ac549a3903f590716b38ca060b0f01088594f22`
- post-two-step model SHA256:  
  `c41702cda99092a7fb63bb0a8227e658851b3ac4cbc373d90cdd6816eccdd196`
- optimizer-state SHA256:  
  `569a6691424ac32d0f252728750281cffd175a2b6b6c6ea1913f5f497200b00d`

This proof is critical for Phase 6 because full production training will require durable resumability.

## 5.6 Frozen validation ranking semantics

For every T60 validation case:

- the 100 candidates are scored;
- raw logits are sorted descending;
- startup-local index ascending is the deterministic tie-break;
- HR@10 is computed per event;
- NDCG@10 is computed per event;
- aggregate validation metrics are event means;
- non-finite logits are a hard failure.

The full structural representation `F_s` is computed once per validation pass.

## 5.7 Untrained validation baseline

Before production training, the exact frozen initial model was scored on all 2,251 validation cases.

Initial validation baseline:

- **HR@10 = 0.091514882275**
- hits at 10: 206
- **NDCG@10 = 0.040193099163**
- mean positive rank: 50.706353
- median positive rank: 50

This is an **untrained baseline**, not a final ITRS result.

## 5.8 Generalized 20-epoch stream proof

The training-stream generator was extended from epoch 0 to all 20 epochs while preserving the frozen negative and order-seed contracts.

Verified anchors include:

Epoch 1:

- negative SHA256:  
  `f7b415e0f305e049cc94c7e4261b683800085f8fa7f7b2df11ff3658dea9d850`
- order SHA256:  
  `2da43d28e540ed48cb557ca889190b5dfebc7c1207cbd3882df2fa14ca2a28d8`

Epoch 19:

- negative SHA256:  
  `06f9a11d8986ba9b7e0242fc414234789ccc7adba91cab6da68ebc21e44682b7`
- order SHA256:  
  `c6236cb081ddba7f72eb0d36199500ac6b7695646e21a824fb84b85fae769329`

All generalized stream invariants passed.

## 5.9 Full reproduction-integrity closure

Before any production training, a read-only integrity closure was executed.

Result:

- **71 / 71 critical checks PASS**
- production training performed: NO
- validation/test production evaluation performed: NO
- production launch status: ALLOWED

This was the formal authorization point for training. The later decision not to launch locally was due only to runtime feasibility.

---

# 6. Phase 5.4 - Runtime Feasibility and Acceleration Engineering

## 6.1 Exact canonical CPU benchmark

The exact frozen training path was benchmarked on:

- macOS 26.5.2
- Apple arm64
- Python 3.11.14
- PyTorch 2.7.0
- initial intra-op threads: 4

Results:

- batch 0: **27.746 s**
- batch 1: **26.101 s**
- mean: **26.923 s/batch**
- conservative: **27.746 s/batch**

Projected runtime:

- one epoch: approximately **3.27 days**
- 20 epochs: approximately **65.32 days**
- conservative 20-epoch projection: approximately **67.32 days**

All frozen batch anchors remained exact.

Conclusion:

> The canonical exact CPU implementation is scientifically valid but operationally impractical.

## 6.2 Lean exact CPU and thread tuning

Candidate thread counts:

- 4
- 6
- 8
- 10

All candidates were tested for exactness before timing.

Best exact configuration:

- **8 intra-op threads**

Exact two-step timing at 8 threads:

- batch 0: **26.072 s**
- batch 1: **24.427 s**
- mean: **25.249 s/batch**

Projected 20 epochs:

- approximately **61.26 days**

Conclusion:

> Proof hashing and thread configuration were not the main bottleneck. CPU thread tuning was effectively exhausted.

## 6.3 High-level exact bottleneck profile

At 8 threads, exact batch profiling showed:

Batch 0:

- backward: **25.270 s / 96.41%**
- unattributed preprocessing/checks: 0.593 s / 2.26%
- structural forward: 0.132 s / 0.50%
- trend GRU: 0.081 s / 0.31%
- trend attention: 0.068 s / 0.26%
- Adam: 0.043 s / 0.17%
- description: 0.019 s / 0.07%
- scoring: 0.003 s / 0.01%

Batch 1 showed the same pattern, with backward at 97.34%.

Aggregate:

- backward mean: **24.507 s / 96.87%**
- all forward branches combined: small relative share.

Conclusion:

> The primary runtime problem is autograd backward, not RGCN forward, trend forward, description encoding, or scoring.

## 6.4 Low-level autograd operator profile

A PyTorch profiler identified the original exact-CPU pathology.

Largest self-CPU contributors included:

- `aten::add_`: 11.873 s / 48.49%
- `aten::fill_`: 8.695 s / 35.51%
- `EmbeddingBackward0`: 7,085 evaluations

The frozen training implementation performs thousands of separate embedding lookups for historical trend periods and investor queries. Dense embedding backward repeatedly allocates and accumulates large mostly-zero gradients.

Conclusion:

> The exact implementation is mathematically correct but autograd-inefficient because the same large embedding tables are touched by thousands of independent backward nodes.

## 6.5 Packed embedding audit

Two packing strategies were tested.

The strongest packing candidate reduced the same logical embedding work to one/few large lookups.

Result:

- batch-0 time: approximately **7.54 s**
- forward logits: exact
- gradient/state hashes: not byte-exact

Conclusion:

> Packing dramatically improves runtime but changes floating-point gradient accumulation order. It cannot be accepted under a byte-exact trajectory requirement.

No inexact runtime was silently accepted.

## 6.6 Sparse embedding backward audit

Two sparse-gradient variants were tested:

- `C_STARTUP_SPARSE`
- `D_BOTH_SPARSE`

Batch-0 timing:

- `C_STARTUP_SPARSE`: **8.027 s**
- `D_BOTH_SPARSE`: **6.786 s**

Both variants preserved exact forward logits but changed byte-level gradient/state hashes.

The final gradients were dense/strided after combining with the structural branch, confirming that this optimization did not change Adam into a sparse optimizer or create sparse parameter states.

Conclusion:

> Sparse embedding backward removes most of the repeated dense-gradient allocation cost, but changes floating-point accumulation order.

## 6.7 Numerical-divergence characterization

Because byte-level hashes cannot distinguish negligible floating-point reordering from materially different learning behavior, canonical CPU and `D_BOTH_SPARSE` were compared statefully through two Adam updates.

Batch 0:

- loss difference: **0**
- logit max absolute difference: **0**
- gradient relative L2 error: **1.3088e-9**
- gradient cosine: **1.0**
- gradient sign agreement: **1.0**
- parameter relative L2 error: **3.3129e-10**
- parameter max absolute difference: **5.9605e-8**
- exact parameter-element fraction: **0.9999124**

Batch 1:

- loss difference: **0**
- logit max absolute difference: **2.3842e-7**
- gradient relative L2 error: **4.1155e-8**
- gradient cosine: **1.0**
- gradient sign agreement: **0.999999375**
- parameter relative L2 error: **1.1589e-9**
- parameter max absolute difference: **2.5332e-7**
- exact parameter-element fraction: **0.9990368**

Conclusion:

> The sparse runtime differs from the canonical CPU trajectory only by negligible float32 accumulation-order effects over the audited stateful steps.

## 6.8 Frozen numerical-equivalence policy

Before testing Apple MPS, the project froze:

`ITRS_PHASE5_NUMERICAL_EQUIVALENCE_V1`

Thresholds:

- loss absolute difference <= **1e-6**
- logit max absolute difference <= **1e-5**
- gradient relative L2 error <= **1e-5**
- gradient cosine similarity >= **0.999999**
- gradient sign agreement >= **0.99999**
- parameter relative L2 error <= **1e-7**
- parameter max absolute difference <= **1e-5**
- Adam `exp_avg` relative L2 error <= **1e-5**
- Adam `exp_avg` cosine similarity >= **0.999999**
- Adam `exp_avg_sq` relative L2 error <= **1e-5**
- Adam `exp_avg_sq` cosine similarity >= **0.999999**

The policy was deliberately frozen before observing device-acceleration results.

`D_BOTH_SPARSE` passed all 22 metric checks across the two stateful batches.

Classification:

`IMPLEMENTATION_EQUIVALENT_CHOICE`

## 6.9 Canonical-dense Apple MPS audit

Apple MPS availability:

- built: TRUE
- available: TRUE

The device-aware executor was first required to reproduce the frozen CPU batch-0 and batch-1 trajectory byte-exactly before MPS results were interpreted.

MPS numerical equivalence:

- PASS under the frozen policy.

Timing:

- batch 0: **47.719 s**
- batch 1 warm: **31.496 s**
- two-batch mean: **39.607 s**

Projected 20 epochs:

- approximately **96.09 days** using the two-batch mean;
- approximately **76.41 days** using warm batch-1 timing.

Conclusion:

> Apple MPS is numerically valid but slower than the canonical lean CPU path for the fragmented dense embedding/autograd workload.

## 6.10 Packed MPS audit

Packing was combined with MPS to reduce GPU kernel fragmentation.

Observed call reductions:

Batch 0:

- startup embedding calls: **6,663 -> 1**
- investor embedding calls: **511 -> 1**

Batch 1:

- startup embedding calls: **6,431 -> 1**
- investor embedding calls: **513 -> 1**

Numerical equivalence:

- PASS under the frozen policy.

Timing:

- batch 0: **16.717 s**
- batch 1 warm: **15.264 s**
- mean: **15.991 s**

Projected 20 epochs:

- mean: approximately **38.80 days**
- warm projection: approximately **37.03 days**

Conclusion:

> Packing improves MPS substantially, but packed MPS remains much slower than sparse CPU.

## 6.11 Sparse CPU residual bottleneck profile

The fastest validated local runtime, `D_BOTH_SPARSE`, was profiled again after sparse-gradient acceleration.

High-level profile:

- total batch time: **6.691 s**
- backward: **6.129 s / 91.60%**
- preprocessing/checks: 0.289 s / 4.32%
- structural forward: 0.079 s / 1.18%
- trend GRU: 0.078 s / 1.17%
- trend attention: 0.069 s / 1.03%
- Adam: 0.039 s / 0.59%
- description: 0.007 s / 0.11%
- scoring: 0.001 s / 0.01%

Operator profile:

- `aten::add`: 21.26% self CPU
- `aten::cat`: 18.20%
- `aten::fill_`: 16.85%
- `aten::add_`: 15.89%
- `EmbeddingBackward0`: still **7,085 evaluations**
- embedding-named self share: 5.19%
- `fill_ + add_` share: 32.74%

Conclusion:

> Sparse gradients remove the original full-table embedding-gradient pathology, but backward remains dominated by a highly fragmented autograd graph.

## 6.12 Packed + sparse CPU audit

A final combined local optimization was tested:

`PACKED_ALL_PLUS_D_BOTH_SPARSE_CPU`

Call reductions remained dramatic:

Batch 0:

- startup calls: **6,663 -> 1**
- investor calls: **511 -> 1**

Batch 1:

- startup calls: **6,431 -> 1**
- investor calls: **513 -> 1**

Numerical equivalence:

- PASS under the frozen policy.

Timing:

- batch 0: **6.365 s**
- batch 1: **7.345 s**
- mean: **6.855 s**
- speedup vs canonical lean CPU: **3.68x**
- speedup vs `D_BOTH_SPARSE`: **0.99x**

Projected runtime:

- one epoch: approximately **19.96 hours**
- 20 epochs, mean: approximately **16.63 days**
- 20 epochs, warm batch-1 projection: approximately **17.82 days**

Conclusion:

> Packing does not improve the already-sparse CPU runtime. Local optimization has reached a practical plateau.

---

# 7. Phase 5 Runtime Leaderboard

| Runtime | Numerical status | Approx. seconds/batch | Approx. 20-epoch runtime | Decision |
|---|---:|---:|---:|---|
| Canonical exact CPU | Byte-exact reference | 25.249 | 61.3 days | Valid but impractical |
| Canonical dense MPS, warm | Numerical-equivalence PASS | 31.496 | 76.4 days | Rejected |
| Packed MPS, warm | Numerical-equivalence PASS | 15.264 | 37.0 days | Rejected |
| `D_BOTH_SPARSE` CPU | Numerical-equivalence PASS | **6.786** | **~16.5 days** | Best validated local runtime |
| Packed + sparse CPU | Numerical-equivalence PASS | 6.855 mean | 16.6 days | No improvement |

## 7.1 Frozen runtime conclusion

The local optimization branch is considered exhausted.

Best validated local runtime:

`D_BOTH_SPARSE CPU`

Approximate local cost:

- **~6.8 s/batch**
- **~19.8 hours/epoch**
- **~16.5 days/20 epochs**

This is too slow for the immediate experimental deadline and is therefore not selected for production training on the current machine.

---

# 8. Results That Exist at Phase 5 Closure

Phase 5 does contain experimental results, but they must be interpreted correctly.

## 8.1 Available result: untrained validation baseline

The frozen initial model, before training, achieved:

- **Validation HR@10: 0.091514882275**
- **Validation NDCG@10: 0.040193099163**

These values establish the initial ranking baseline.

They are **not** the performance of the trained ITRS reproduction.

## 8.2 Available result: reproducibility and integrity

The following are demonstrated experimentally:

- deterministic training-stream generation;
- deterministic batch construction;
- exact canonical forward/backward anchors;
- exact Adam-update anchors;
- exact checkpoint/resume;
- deterministic evaluation candidates;
- frozen validation-ranking semantics;
- test-set isolation;
- numerical equivalence of accelerated CPU/MPS candidates under a predeclared policy.

## 8.3 Available result: runtime feasibility

The canonical and accelerated runtime measurements provide a clear compute-feasibility result:

> The current Mac hardware cannot complete the frozen 20-epoch reproduction within the immediate experimental deadline.

## 8.4 Results that do NOT yet exist

At Phase 5 closure, the project does **not** yet have:

- trained epoch-1 through epoch-20 losses;
- trained validation HR@10 per epoch;
- trained validation NDCG@10 per epoch;
- selected best epoch;
- final test HR@10;
- final test NDCG@10;
- paper-vs-reproduction performance comparison;
- trained cold-start or structural-coverage diagnostics.

These are Phase 6 deliverables.

---

# 9. Scientific Conclusions from Phase 5

## 9.1 The reproduction specification is operational

The project has moved beyond conceptual reconstruction. A concrete training example can be decoded, scored by the complete reconstructed model, differentiated across all trainable tensors, optimized by Adam, checkpointed, restored, and evaluated under deterministic ranking semantics.

## 9.2 Evaluation leakage controls are explicit

T60 is not used for training. Validation is used only for checkpoint selection. Test has not been used for model selection, runtime decisions, or numerical-equivalence decisions.

## 9.3 The paper leaves implementation choices that required explicit reproduction contracts

Several behaviors, especially negative-sampling runtime details, RNG derivation, tie-breaking, and some training-control details, required explicit reproduction choices because the paper does not fully specify them.

These choices have been isolated and labeled rather than silently assumed.

## 9.4 Exact reproducibility and numerical reproducibility were separated

The canonical CPU trajectory remains the byte-exact scientific reference.

Accelerated implementations may differ at the last bits because floating-point accumulation order changes. They are accepted only if they satisfy the predeclared `ITRS_PHASE5_NUMERICAL_EQUIVALENCE_V1` policy.

This distinction prevents performance engineering from silently altering the experiment.

## 9.5 Hardware, not unresolved semantics, is the remaining blocker

No unresolved issue in the following categories currently prevents production training:

- temporal split;
- graph topology;
- model architecture;
- negative sampling;
- batch construction;
- optimizer;
- checkpointing;
- validation semantics;
- model-selection rules;
- test isolation.

The remaining issue is compute throughput.

---

# 10. Dataset Adaptations and Reproduction Caveats Carried into Phase 6

These adaptations were established before Phase 5 and remain relevant when interpreting final Phase 6 results.

## 10.1 Source dataset difference

The original ITRS work uses Tianyancha-derived data, whereas this reproduction uses audited Crunchbase data.

Classification:

`DATASET_ADAPTATION`

## 10.2 Structural-relation adaptation

The reproduction uses 12 Crunchbase-derived typed structural relation channels, including an `ACQUIRED` / `ACQUIRED_BY` adaptation.

Classification:

`DATASET_ADAPTATION`

## 10.3 Static descriptions and categories

Description/category features are current static snapshots rather than historically versioned observations.

Classification:

`DATASET_ADAPTATION`

## 10.4 Founder-sharing relation

`SHARED_FOUNDER` is treated as a current, unversioned snapshot.

Classification:

`DATASET_ADAPTATION`

## 10.5 Structural sparsity

A large fraction of role nodes are structural isolates:

- isolates: **402,807 / 477,564**

Among T60 unique pairs:

- both investor and startup structurally connected: 653 / 2.9247%
- investor-only connected: 1,798
- startup-only connected: 3,763
- neither connected: 16,113 / 72.1682%

Structural coverage is a diagnostic variable. It is not used to filter T60 evaluation cases.

---

# 11. Key Phase 5 Artifacts

The following artifacts should be retained with the Phase 5 closure commit.

## 11.1 Training/evaluation contracts

Representative contract directory:

`data/experimental/phase_5/contracts/`

Important contracts include:

- production training launch configuration;
- production training launch authorization;
- Phase 5.4.2 lean exact CPU runtime contract;
- Phase 5.4.4 autograd operator profile contract;
- Phase 5.4.5 packed embedding acceleration contract;
- Phase 5.4.6 sparse embedding acceleration contract;
- Phase 5.4.7a numerical divergence characterization contract;
- `phase_5_4_numerical_equivalence_policy.json`;
- Phase 5.4.7b numerical-equivalence policy contract;
- Phase 5.4.8a MPS feasibility contract;
- Phase 5.4.8b packed MPS feasibility contract;
- Phase 5.4.9 sparse CPU residual bottleneck contract;
- Phase 5.4.10 packed+sparse CPU acceleration contract.

## 11.2 Audit outputs

Representative audit root:

`data/experimental/phase_5/audits/`

The audit tree contains:

- exact CPU timing;
- thread-equivalence tests;
- high-level component timing;
- PyTorch operator profiling;
- packed embedding equivalence tests;
- sparse embedding equivalence tests;
- numerical-divergence tables;
- MPS numerical policy checks;
- packed MPS call-plan and timing outputs;
- residual sparse CPU profiling;
- packed+sparse CPU timing and policy outputs;
- final-invariant CSV files;
- manifests.

## 11.3 Training implementation scripts

The Phase 5 script family includes:

- negative-sampling audits and freezes;
- production controller dry runs;
- checkpoint/resume roundtrip proof;
- full validation dry run;
- generalized 20-epoch stream proof;
- reproduction-integrity closure;
- exact CPU feasibility benchmark;
- runtime threading benchmark;
- bottleneck profiles;
- packing/sparse acceleration audits;
- numerical-equivalence policy;
- MPS audits;
- residual runtime profiles.

Failed script versions are intentionally preserved as part of the audit trail.

---

# 12. Phase 5 Frozen Decisions - Do Not Reopen in Phase 6

Unless a new contradiction is discovered in the frozen inputs, Phase 6 should not reopen the following:

1. T0-T59 are the only training-history periods.
2. T60 is fully held out from training.
3. Training negatives follow the frozen `eligible_negative(o,b,h)` rule.
4. Four training negatives are generated per positive.
5. Training negatives regenerate independently every epoch.
6. Training examples are shuffled once per epoch using frozen deterministic seeds.
7. Batch size is 512, with a final batch of 485.
8. Adam configuration is frozen.
9. Training lasts 20 epochs with no early stopping.
10. Validation is performed after every completed epoch.
11. Best checkpoint uses NDCG@10 primary, HR@10 secondary, earliest epoch final tie-break.
12. Test is evaluated exactly once after best-checkpoint selection.
13. T60 evaluation uses one positive plus 99 sampled negatives.
14. Raw logits are the ranking score.
15. Startup-local ascending index is the deterministic tie-break.
16. Canonical CPU remains the byte-exact reference.
17. Accelerated devices/runtimes are judged against `ITRS_PHASE5_NUMERICAL_EQUIVALENCE_V1`.
18. Local Mac optimization is closed unless a clearly new hardware/runtime capability becomes available.

---

# 13. Phase 6 - Production Training, Final Evaluation, and Conclusions

Phase 6 begins from the frozen Phase 5 repository state.

## Phase 6.1 - CUDA Runtime Qualification

### Goal

Move the frozen production training path to CUDA-capable NVIDIA hardware without changing scientific semantics.

### Required checks

A CUDA candidate must:

- load the exact frozen Phase 4 model state;
- use the exact frozen Phase 5 training stream;
- use the same Adam hyperparameters;
- use the same BCE objective;
- execute the same batch examples in the same order;
- preserve model and optimizer state semantics;
- pass `ITRS_PHASE5_NUMERICAL_EQUIVALENCE_V1` over a small stateful qualification run.

### Classification

`IMPLEMENTATION_EQUIVALENT_CHOICE`

### Stop condition

Once a CUDA runtime passes the frozen numerical policy and offers practical throughput, runtime engineering ends.

No new tolerance should be created specifically for CUDA.

---

## Phase 6.2 - Full 20-Epoch Production Training

### Goal

Execute the frozen training experiment.

For every epoch:

1. generate that epoch's deterministic training-negative matrix;
2. generate that epoch's deterministic shuffled order;
3. train all 10,481 batches;
4. compute the example-weighted mean training BCE;
5. write a durable checkpoint;
6. evaluate all 2,251 validation cases;
7. record validation HR@10;
8. record validation NDCG@10;
9. update best-checkpoint status using the frozen selection rule.

Expected total:

- **20 epochs**
- **209,620 optimizer steps**
- **107,324,900 training-example presentations**

The test split remains untouched during this stage.

---

## Phase 6.3 - Validation-Based Best-Checkpoint Selection

Frozen selection rule:

1. highest validation NDCG@10;
2. if tied, highest validation HR@10;
3. if still tied, earliest epoch.

Required final training summary:

| Epoch | Mean Train BCE | Val HR@10 | Val NDCG@10 | Best So Far |
|---:|---:|---:|---:|---|

The selected checkpoint must be recorded together with:

- epoch number;
- model-state hash;
- optimizer-state hash if retained;
- validation metrics;
- checkpoint path.

---

## Phase 6.4 - Final Test Evaluation

After all 20 epochs are complete:

1. load the selected best checkpoint;
2. verify its identity;
3. score all **20,264** frozen test cases;
4. compute final event-mean HR@10;
5. compute final event-mean NDCG@10;
6. do not perform further model selection.

Primary final reproduction outputs:

- **Test HR@10**
- **Test NDCG@10**

This is the first point at which the reproduction can claim final trained-model recommendation performance.

---

## Phase 6.5 - Reproduction Comparison and Diagnostics

Once final test metrics exist, Phase 6 should analyze:

### Paper comparison

- compare the reproduced HR@10 and NDCG@10 with the values reported by ITRS;
- distinguish direct performance differences from dataset adaptations;
- avoid claiming numerical replication when the underlying dataset differs.

### Learning evidence

Compare:

- untrained validation baseline;
- best trained validation performance;
- final held-out test performance.

### Optional post-hoc diagnostic slices

These should not be used for checkpoint selection:

- warm vs cold investors;
- warm vs cold startups;
- structural connectedness;
- structural isolates;
- new-to-investor vs previously observed investor-startup pair;
- history-length groups;
- possibly relation-coverage groups.

These diagnostics can help explain where ITRS succeeds or fails on Crunchbase.

---

## Phase 6.6 - Final Reproduction Conclusions

Phase 6 should end by answering:

1. Was the ITRS computational pipeline reproducible from the paper description plus explicit reproduction choices?
2. Does the reconstructed model learn above its untrained baseline on Crunchbase?
3. How close are final HR@10 and NDCG@10 to the original paper?
4. Which differences are plausibly attributable to dataset/source adaptations?
5. How does extreme structural sparsity affect the usefulness of the heterogeneous graph branch?
6. Which implementation details were materially underspecified by the paper?
7. What parts of the reproduction are exact, implementation-equivalent, or dataset-adapted?
8. What improvements or thesis directions are justified by the observed results?

---

# 14. Phase 6 Completion Criteria

Phase 6 is complete only when all of the following are true:

- [ ] CUDA or another practical production runtime passes the frozen numerical-equivalence policy.
- [ ] All 20 epochs are completed.
- [ ] Every epoch has a recorded mean training BCE.
- [ ] Every epoch has validation HR@10 and NDCG@10.
- [ ] The best checkpoint is selected only from validation metrics.
- [ ] The final test split is evaluated exactly once.
- [ ] Final test HR@10 is recorded.
- [ ] Final test NDCG@10 is recorded.
- [ ] Results are compared with the ITRS paper.
- [ ] Dataset adaptations and limitations are explicitly discussed.
- [ ] Final reproduction conclusions are written.
- [ ] No Phase 5 frozen decision is silently changed.

---

# 15. Supervisor-Facing Status Summary

At the Phase 5 closure point, the project can be summarized as follows:

| Work item | Status |
|---|---|
| Canonical Crunchbase interaction reconstruction | Complete |
| Temporal split | Complete / Frozen |
| Heterogeneous structural graph | Complete / Frozen |
| ITRS architecture reconstruction | Complete / Frozen |
| Training-negative semantics | Complete / Frozen |
| T60 evaluation candidates | Complete / Frozen |
| Adam/training schedule | Complete / Frozen |
| End-to-end forward/backward proof | Complete |
| Checkpoint/resume proof | Complete |
| 20-epoch deterministic stream generation | Complete |
| Validation ranking implementation | Complete |
| Initial untrained validation baseline | Complete |
| Full integrity closure | 71/71 checks PASS |
| CPU runtime feasibility | Complete |
| Apple MPS feasibility | Complete |
| Numerical-equivalence policy | Frozen |
| Local runtime optimization | Closed |
| Full 20-epoch production training | Pending - Phase 6 |
| Final test HR@10/NDCG@10 | Pending - Phase 6 |
| Paper-vs-reproduction comparison | Pending - Phase 6 |
| Final experimental conclusions | Pending - Phase 6 |

A concise supervisor statement is:

> The ITRS reproduction pipeline is fully reconstructed and validated through the production-training boundary. The current blocker is computational throughput: the best validated local runtime still projects to approximately 16.5 days for the frozen 20-epoch experiment. Final trained-model HR@10 and NDCG@10 therefore require Phase 6 execution on CUDA-capable hardware.

---

# 16. Final Phase 5 Status

**PHASE 5 STATUS: COMPLETE / FROZEN**

Completed:

- training and evaluation semantics;
- deterministic negative sampling and ordering;
- training controller specification;
- optimizer and checkpoint behavior;
- end-to-end differentiation;
- checkpoint/resume;
- validation semantics;
- complete stream generation;
- integrity closure;
- numerical-equivalence policy;
- local runtime feasibility and acceleration evaluation.

Not performed in Phase 5:

- production 20-epoch training;
- final best-checkpoint selection;
- final test evaluation;
- final trained HR@10 and NDCG@10;
- final ITRS paper comparison.

These are intentionally moved to **Phase 6**.

**Test set status at Phase 5 closure: NOT USED FOR PRODUCTION EVALUATION OR MODEL SELECTION.**

**Next project action:** qualify the frozen runtime on CUDA-capable hardware and launch Phase 6 production training.
