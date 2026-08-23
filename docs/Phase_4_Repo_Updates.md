# Phase 4 - Model Reconstruction ✅

**Completion date:** 2026-08-20  
**Project:** ITRS Crunchbase Reproduction  
**Status:** COMPLETE / FROZEN  
**Next phase:** Phase 5 - Training and Evaluation

## Objective

Reconstruct the ITRS neural architecture from the paper against the frozen Phase-2 temporal split and Phase-3 graph, explicitly classify paper-unspecified implementation choices, verify the complete computation and autograd graph, freeze canonical initialization/seed, and create an immutable Phase-5 handoff.

## Final model

| Component | Frozen result |
|---|---|
| Investor latent | `[165975,40]` |
| Startup latent | `[311589,40]` |
| Description | `40` |
| Trend | `40` |
| Structural | `40` |
| Investor scoring vector | `160` |
| Startup scoring vector | `120` |
| Pair input | `280` |
| Scoring MLP | `280→128→64→32→16→1` |
| Parameter tensors | `32` |
| Trainable parameters | `19,217,929` |

## Final parameter budget

| Component | Parameters |
|---|---:|
| Investor embeddings | 6,639,000 |
| Startup embeddings | 12,463,560 |
| Description encoder | 16,720 |
| Trend module | 32,480 |
| R-GCN | 19,320 |
| Scoring MLP | 46,849 |
| **Full model** | **19,217,929** |

## Frozen architecture rules

- Exactly two shared latent tables: `L_o`, `L_b`.
- Description: Doc2Vec32 + category802 -> two 20-D ReLU branches -> `F_d40`.
- Trend target `T_h` consumes `T0..T(h-1)` only.
- Empty historical period = `zero80`.
- Trend = bilinear attention + 2-layer GRU(80,40) + sigmoid projection.
- Preference propagation = full frozen 12-channel graph, 2-layer R-GCN, 5 bases, 40-D output.
- R-GCN aggregation = relation-specific incoming mean + root transform.
- Scoring feature order = `F_t, L_o, F_d,o, F_s,o, L_b, F_d,b, F_s,b`.
- Scorer = `280→128→64→32→16→1`, four ReLU hidden layers.
- Training output = raw logit; probability = sigmoid; loss interface = `BCEWithLogitsLoss`.
- Learned `F_d`, `F_t`, `F_s` are recomputed from current parameters; do not persist stale caches across updates.

## Static/runtime artifacts to preserve

```text
data/experimental/phase_4/doc2vec/vectors/doc2vec_vectors_all.npy
data/experimental/phase_4/doc2vec/vectors/doc2vec_vector_manifest.parquet
data/experimental/phase_4/description_labels/description_label_multihot.npz
data/experimental/phase_4/description_labels/description_label_vector_manifest.parquet
data/experimental/phase_4/trend_runtime/trend_period_ptr.npy
data/experimental/phase_4/trend_runtime/trend_startup_node_indices.npy
data/experimental/phase_4/trend_runtime/trend_period_startup_counts.npy
```

## Canonical initialization

```text
function        torch.nn.init.kaiming_normal_
distribution    normal
a               0.0
mode            fan_in
nonlinearity    relu
bias            zero
seed            42
canonical init  CPU dedicated torch.Generator
reference torch 2.7.0
```

Canonical state fingerprint:

```text
49e822ea7fad35c458f47e134c94c05eac099b68c5c468e2c71559c8c88998ab
```

## End-to-end verification

Phase 4 verified a real T60 validation positive through:

```text
static inputs -> F_d -> shared L_o/L_b -> F_t -> F_s -> pair280 -> scorer -> logit -> BCE -> backward
```

All expected parameter gradients exist, are finite, and are non-zero. No training took place.

## Final closure outputs

```text
data/experimental/phase_4/closure/phase_4_closure_manifest.json
data/experimental/phase_4/closure/phase_4_to_phase_5_handoff_contract.json
data/experimental/phase_4/closure/phase_4_final_model_decision_register.csv
data/experimental/phase_4/closure/phase_4_deferred_training_evaluation_decisions.csv
data/experimental/phase_4/closure/phase_4_known_limitations_and_adaptations.csv
data/experimental/phase_4/closure/phase_4_authoritative_artifact_registry.csv
data/experimental/phase_4/closure/phase_4_parameter_summary.csv
data/experimental/phase_4/closure/phase_4_final_contract_status_audit.csv
data/experimental/phase_4/closure/Phase_4_Reproduction_Log_Entry.md
data/experimental/phase_4/closure/phase_4_closure_artifact_hashes.csv
```

## Phase-5 handoff constraints

Phase 5 must treat as immutable:

1. Phase-2 temporal segmentation and split identities.
2. T60 holdout identities.
3. Phase-3 role-node identity and structural graph.
4. The 12 typed relation channels.
5. Phase-4 static description inputs.
6. Trend CSR memberships.
7. Description/trend/R-GCN/scoring architecture.
8. Shared `L_o/L_b` ownership.
9. Canonical Kaiming initialization and neural seed42.

Phase 5 may resolve only the decisions explicitly deferred:

- training negative:positive ratio
- training negative candidate eligibility
- training historical negative exclusion
- training epoch count
- early stopping
- weight decay
- evaluation candidate-generation runtime contract

## Git guidance

Version these compact reproducibility assets:

```text
scripts/phase_4_*.py
docs/reproduction_log.md
README.md
compact CSV/JSON contracts, decision registers, and hashes
```

Generally keep large feature/tensor/runtime artifacts outside Git:

```text
*.npy
*.npz
large *.parquet
large model/runtime arrays
```

## Suggested commit

```text
docs: close Phase 4 ITRS model reconstruction and freeze Phase-5 handoff
```
