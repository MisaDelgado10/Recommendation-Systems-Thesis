# Phase 5 Repository Updates

**Status:** COMPLETE / FROZEN  
**Purpose:** Repository-oriented index of the Phase-5 training/evaluation reconstruction.

---

## What Phase 5 added

Phase 5 introduced the executable and audit infrastructure required to turn the Phase-4 model into a reproducible training experiment.

The repository now contains script families for training-negative semantics, T60 candidate construction, optimizer/training control, end-to-end numerical proofs, checkpoint/resume, validation ranking, generalized 20-epoch streams, integrity closure, runtime profiling, packed/sparse acceleration, numerical-equivalence policy, and MPS feasibility.

Failed/intermediate script versions are retained intentionally as audit trail.

---

## Key script groups

### Training-negative semantics

```text
scripts/phase_5_1_1a_negative_sampling_feasibility_temporal_collision_audit.py
scripts/phase_5_1_1b_freeze_training_negative_semantics.py
scripts/phase_5_1_1c_training_negative_ratio_sampling_runtime_audit.py
scripts/phase_5_1_1d_freeze_training_negative_sampling_runtime.py
```

### Evaluation candidates

```text
scripts/phase_5_1_2a_t60_evaluation_candidate_leakage_collision_audit.py
scripts/phase_5_1_2b_freeze_t60_evaluation_candidate_runtime.py
scripts/phase_5_1_2c_generate_t60_evaluation_candidates.py
```

### Training control

```text
scripts/phase_5_2_1_training_control_optimizer_audit.py
scripts/phase_5_2_2_freeze_training_control_optimizer_runtime.py
```

### Numerical/training proof

```text
scripts/phase_5_3_1k_canonical_composed_real_data_forward_bce_backward_preflight.py
scripts/phase_5_3_1l_1_freeze_epoch0_training_stream.py
scripts/phase_5_3_1l_2b_corrected_adam_epoch0_first_batch_preflight.py
scripts/phase_5_3_1m_first_adam_weight_update_proof.py
scripts/phase_5_3_2b_checkpoint_resume_roundtrip_proof.py
```

### Validation/evaluation semantics

```text
scripts/phase_5_3_3a_validation_ranking_metric_semantics_audit.py
scripts/phase_5_3_3b_canonical_real_validation_scoring_preflight.py
scripts/phase_5_3_3c_full_validation_split_runtime_dry_run.py
scripts/phase_5_3_3d_validation_checkpoint_selection_integration_proof.py
```

### Full-stream/launch closure

```text
scripts/phase_5_3_5_generalized_20_epoch_training_stream_generator_proof_V6.py
scripts/phase_5_3_6_production_training_launch_contract_freeze_V1.py
scripts/phase_5_3_7_full_reproduction_integrity_closure_audit_V2.py
```

### Runtime/acceleration

```text
scripts/phase_5_4_1_exact_cpu_training_path_feasibility_benchmark_V1.py
scripts/phase_5_4_2_lean_exact_cpu_runtime_threading_benchmark_V1.py
scripts/phase_5_4_3_exact_runtime_bottleneck_profile_V1.py
scripts/phase_5_4_4_exact_autograd_operator_bottleneck_profile_V1.py
scripts/phase_5_4_5_packed_embedding_equivalence_acceleration_audit_V1.py
scripts/phase_5_4_6_sparse_embedding_backward_equivalence_acceleration_audit_V1.py
scripts/phase_5_4_7a_canonical_vs_sparse_numerical_divergence_audit_V1.py
scripts/phase_5_4_7b_freeze_numerical_equivalence_policy_V1.py
scripts/phase_5_4_8a_mps_numerical_equivalence_feasibility_audit_V1.py
scripts/phase_5_4_8b_packed_mps_numerical_equivalence_feasibility_audit_V2.py
scripts/phase_5_4_9_sparse_cpu_residual_bottleneck_profile_V1.py
scripts/phase_5_4_10_packed_sparse_cpu_equivalence_runtime_audit_V1.py
```

---

## Core frozen values carried into Phase 6

```text
Training positives / epoch:  1,073,249
Negatives / positive:        4
Examples / epoch:            5,366,245
Batch size:                  512
Batches / epoch:             10,481
Epochs:                      20
Optimizer steps:             209,620

Validation cases:            2,251
Test cases:                  20,264
Candidates / case:           100
```

Checkpoint selection:

```text
1. maximum validation NDCG@10
2. maximum validation HR@10
3. earliest epoch
```

Important fingerprints:

```text
Initial model:
49e822ea7fad35c458f47e134c94c05eac099b68c5c468e2c71559c8c88998ab

Epoch-0 positive stream:
73b074a80675793b811fbdc8a0609883c857fb2a687a2e01c31865ade5b509d1

Evaluation negative matrix:
7f98269d0291382dacfc783ffd66ad6d2c2f8775877d57dc0d83504e40d8716d

Evaluation case manifest:
44b4b7e1ec1b1978249318a080df02d0d9f617845263513594de64e71b969e0c
```

Existing comprehensive documentation:

```text
docs/ITRS_Reproduction_Phase_5_Closure_and_Phase_6_Handoff.md
docs/ITRS_Reproduction_Phase_5_Closure_and_Phase_6_Handoff.docx
```
