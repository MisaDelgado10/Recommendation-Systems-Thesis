# Reproduction Log Entry - Phase 4: Model Reconstruction

**Completion date:** 2026-08-20  
**Status:** COMPLETE / FROZEN  
**Previous phase:** Phase 3 - Heterogeneous Graph Reconstruction  
**Next phase:** Phase 5 - Training and Evaluation

## Phase objective

Reconstruct the complete ITRS neural architecture against the frozen Phase-2 temporal experiment and Phase-3 heterogeneous graph, verify every module and the end-to-end BCE/autograd path, freeze a reproducible global initialization/seed contract, and produce an immutable Phase-5 handoff without performing training.

## Chronological subphase record

| Subphase | Objective | Main result / decision |
|---|---|---|
| 4.1 | Establish model reconstruction contract | Four ITRS modules reconstructed without reopening Phase 2 or Phase 3 |
| 4.2 | Description extraction | Doc2Vec32 + category802; shared 20+20 ReLU branches; final `F_d=40`; 16,720 parameters |
| 4.3 | Trend extraction | 80-D query/items; target `T_h` uses `T0..T(h-1)`; bilinear attention + 2-layer GRU; `F_t=40`; CSR runtime frozen |
| 4.4 | Preference propagation | Shared latent input; full frozen 12-channel graph; 2-layer R-GCN, 5 bases, ReLU; 19,320 parameters; full forward/autograd PASS |
| 4.5 | Recommendation scoring | Exact feature order; pair280; `280→128→64→32→16→1`; raw logits + `BCEWithLogitsLoss`; 46,849 parameters |
| 4.6 | Complete ITRS integration | Exactly two shared latent tables; 19,217,929 total parameters; real T60 validation forward/BCE/backward PASS |
| 4.7 | Integrity and initialization | All cross-contract checks PASS; Kaiming normal/fan_in/ReLU, zero biases, neural seed42; canonical state hash frozen |
| 4.8 | Closure and handoff | Final closure manifest, Phase-5 handoff, decision register, artifact registry, limitation register, hashes, and reproduction log written |

## Final model specification

```text
Role nodes:                 477,564
Investor latent table:      [165,975, 40]
Startup latent table:       [311,589, 40]
Description dimension:      40
Trend dimension:            40
Structural dimension:       40
Investor scoring vector:    160
Startup scoring vector:     120
Pair vector:                280
Scoring MLP:                280 -> 128 -> 64 -> 32 -> 16 -> 1
Trainable tensors:          32
Trainable parameters:       19,217,929
```

## Description contract

- Static Doc2Vec matrix: `(477564, 32)`, float32.
- All-zero Doc2Vec rows: 2,670.
- Shared category vocabulary: 802.
- Sparse category matrix: `(477564, 802)`, 1,230,068 nonzeros.
- All-zero category rows: 111,919.
- Text branch: `Linear(32,20,bias=True) -> ReLU`.
- Label branch: `Linear(802,20,bias=True) -> ReLU`.
- Concatenated description representation: `F_d=40`.
- 20/20 split is a paper-unspecified reproduction choice.

## Trend contract

- Historical T0-T59 events: 1,173,422.
- Collapsed historical memberships: 1,145,364.
- Active Investor-periods: 554,171.
- Empty Investor-periods: 9,404,329.
- T60 Investors with no prior history: 3,060 / 11,884 = 25.7489%.
- Investor query: `[L_o || F_d,o] = 80`.
- Startup item: `[L_b || F_d,b] = 80`.
- Same Investor-Startup within one period collapses to one attention item.
- Empty periods are explicit `zero80`.
- Target `T_h` consumes `T0..T(h-1)`; T60 consumes T0-T59.
- GRU: input80, hidden40, 2 layers.
- Output: `Linear(40,40,bias=False) -> Sigmoid`.
- Trend parameters: 32,480.

## Preference-propagation contract

- Initial structural matrix: `cat(L_o.weight, L_b.weight)` -> `[477564,40]`.
- Graph: 158,818 directed structural edges, 12 typed channels.
- R-GCN: 2 layers, 5 bases, 40 dimensions.
- Relation-specific incoming mean normalization.
- Separate root/self transform; no explicit self-loop relation.
- ReLU after each layer.
- No layer bias, dropout, normalization, or residual.
- Structural parameters: 19,320.
- Isolates remain and use root-only transformation.

## Scoring contract

```text
R_o = [F_t || L_o || F_d,o || F_s,o] = 160
R_b = [L_b || F_d,b || F_s,b]         = 120
Pair = [R_o || R_b]                    = 280
```

Frozen feature order:

```text
F_t, L_o, F_d,o, F_s,o, L_b, F_d,b, F_s,b
```

Scorer:

```text
280 -> 128 -> 64 -> 32 -> 16 -> 1
```

- ReLU after all four hidden layers.
- Training forward returns raw logits.
- Probability is `sigmoid(logit)`.
- Loss interface is `BCEWithLogitsLoss`.
- Hidden widths are an NCF-guided paper-unspecified reproduction choice.

## Shared latent ownership

Only two trainable latent tables exist:

```text
L_o: [165975,40]
L_b: [311589,40]
```

The same tables feed trend extraction, R-GCN preference propagation, and recommendation scoring. No separate structural/trend/scoring embeddings are permitted.

## End-to-end integration

A real frozen T60 validation positive was propagated through the complete reconstructed architecture. Verified:

- real Doc2Vec and category rows,
- real T0-T59 history,
- full Phase-3 graph,
- description forward,
- trend attention + GRU forward,
- full-graph R-GCN forward,
- 160-D Investor representation,
- 120-D Startup representation,
- 280-D pair representation,
- raw logit and sigmoid,
- `BCEWithLogitsLoss` equivalence,
- complete backward propagation.

Every expected trainable parameter family received finite non-zero gradients. No optimizer or optimizer step was used.

## Final initialization contract

Paper-specified family: **Kaiming**.

Frozen reproduction variant:

```text
torch.nn.init.kaiming_normal_(
    a=0.0,
    mode="fan_in",
    nonlinearity="relu"
)
```

- Biases: zero.
- Global neural seed: 42.
- Canonical initialization: CPU with dedicated `torch.Generator`.
- Reference PyTorch: 2.7.0.
- 21 Kaiming parameter tensors + 11 zero-bias tensors = all 32 trainable tensors.

Canonical initial-state SHA256:

```text
49e822ea7fad35c458f47e134c94c05eac099b68c5c468e2c71559c8c88998ab
```

Two independent seed-42 initializations were byte-identical; seed43 changed the state.

## Final parameter budget

| Component | Parameters |
|---|---:|
| Investor latent embeddings | 6,639,000 |
| Startup latent embeddings | 12,463,560 |
| Description encoder | 16,720 |
| Trend module | 32,480 |
| R-GCN | 19,320 |
| Scoring MLP | 46,849 |
| **FULL ITRS** | **19,217,929** |

## Known limitations / adaptations

1. Original ITRS 103-relation vocabulary is unavailable; Phase 4 consumes the frozen 12-channel Crunchbase adaptation.
2. Description features are current-snapshot rather than historically versioned.
3. `SHARED_FOUNDER` remains `current_snapshot_unversioned`.
4. `ACQUIRED/ACQUIRED_BY` is a Crunchbase-specific structural adaptation.
5. 402,807 / 477,564 role nodes are structurally isolated.
6. 72.1682% of T60 pairs have neither endpoint structurally connected.
7. Description branch widths are paper-unspecified.
8. Scoring hidden widths are paper-unspecified.
9. R-GCN activation is paper-unspecified; ReLU is frozen.
10. Exact Kaiming variant and neural seed are paper-unspecified; the reproduction choices are explicitly frozen.
11. No training/evaluation performance is claimed in Phase 4.

## Final authoritative closure outputs

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

## Decisions handed to Phase 5

- early stopping
- evaluation candidate-generation runtime contract
- training epoch count
- training historical negative exclusion
- training negative candidate eligibility
- training negative:positive ratio
- weight decay

Phase 5 must freeze the training/evaluation contract before the first optimizer step.

## Suggested commit

```text
docs: close Phase 4 ITRS model reconstruction and freeze Phase-5 handoff
```
