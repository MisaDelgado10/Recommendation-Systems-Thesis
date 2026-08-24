#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import pandas as pd
import torch


# =============================================================================
# Sources
# =============================================================================

PHASE6_BASE_PATH = Path(
    "scripts/phase_6_1_cuda_mac_reference_qualification.py"
)

PACKED_RUNTIME_PATH = Path(
    "scripts/"
    "phase_5_4_8b_packed_mps_numerical_equivalence_feasibility_audit_V2.py"
)

POLICY_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_numerical_equivalence_policy.json"
)

OUT_DIR = Path(
    "data/experimental/phase_6/audits/"
    "phase_6_2_packed_sparse_cuda_strict_fp32"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_2_packed_sparse_cuda_strict_fp32_qualification.json"
)


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def banner(text):
    print("\n" + "=" * 112)
    print(text)
    print("=" * 112)


def load_module(path, name):
    require(path.exists(), f"Missing source: {path}")

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    require(
        spec is not None
        and spec.loader is not None,
        f"Cannot import {path}",
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def main():

    banner(
        "PHASE 6.2 — PACKED + D_BOTH_SPARSE CUDA "
        "STRICT-FP32 NUMERICAL QUALIFICATION"
    )

    print("Production training launched: NO")
    print("Validation cases scored:      0")
    print("Test cases scored:            0")

    # =========================================================================
    # Strict CUDA mode — identical to qualified Phase 6.1b
    # =========================================================================

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(True)

    require(
        torch.cuda.is_available(),
        "CUDA unavailable.",
    )

    require(
        torch.__version__.startswith("2.7.0"),
        f"PyTorch drift: {torch.__version__}",
    )

    cuda = torch.device("cuda:0")

    print()
    print("GPU:                      ", torch.cuda.get_device_name(0))
    print("PyTorch:                  ", torch.__version__)
    print("CUDA runtime:             ", torch.version.cuda)
    print("matmul TF32:              ", torch.backends.cuda.matmul.allow_tf32)
    print("cuDNN TF32:               ", torch.backends.cudnn.allow_tf32)
    print("float32 matmul precision: ", torch.get_float32_matmul_precision())
    print("deterministic algorithms: ", torch.are_deterministic_algorithms_enabled())

    # =========================================================================
    # Load already-proven Phase-6 portable-reference infrastructure
    # =========================================================================

    q = load_module(
        PHASE6_BASE_PATH,
        "_phase6_base",
    )

    packed = load_module(
        PACKED_RUNTIME_PATH,
        "_phase6_packed_runtime",
    )

    # Exact file fingerprints already frozen in Phase 6.1.
    require(
        q.INITIAL_STATE_PATH.exists(),
        "Missing portable initial-state artifact.",
    )

    require(
        q.MAC_REFERENCE_PATH.exists(),
        "Missing Mac numerical-reference artifact.",
    )

    require(
        q.file_sha256(q.INITIAL_STATE_PATH)
        == q.EXPECTED_INITIAL_FILE_SHA,
        "Portable initial-state file SHA mismatch.",
    )

    require(
        q.file_sha256(q.MAC_REFERENCE_PATH)
        == q.EXPECTED_REFERENCE_FILE_SHA,
        "Mac reference file SHA mismatch.",
    )

    initial_payload = torch.load(
        q.INITIAL_STATE_PATH,
        map_location="cpu",
        weights_only=False,
    )

    mac_reference = torch.load(
        q.MAC_REFERENCE_PATH,
        map_location="cpu",
        weights_only=False,
    )

    # =========================================================================
    # Frozen runtime
    # =========================================================================

    roundtrip = q.load_module(
        q.ROUNDTRIP_PATH,
        "_phase6_2_roundtrip",
    )

    preflight = (
        roundtrip.load_preflight_runtime()
    )

    q.install_portable_initialization_bridge(
        preflight,
        initial_payload["model_state_dict"],
    )

    (
        probe_model,
        probe_optimizer,
        probe_hash_fn,
        *_,
    ) = roundtrip.construct_fresh_training_state(
        preflight
    )

    require(
        probe_hash_fn(probe_model)
        == q.EXPECTED_INITIAL_MODEL_SHA,
        "Portable initial-state gate failed.",
    )

    require(
        len(probe_optimizer.state) == 0,
        "Fresh Adam state should be empty.",
    )

    del probe_model
    del probe_optimizer

    # =========================================================================
    # Frozen policy
    # =========================================================================

    policy = json.loads(
        POLICY_PATH.read_text(
            encoding="utf-8"
        )
    )

    require(
        policy["schema_version"]
        == q.EXPECTED_POLICY,
        "Numerical-policy version drift.",
    )

    require(
        policy["status"] == "FROZEN",
        "Numerical policy not frozen.",
    )

    thresholds = policy["thresholds"]

    # =========================================================================
    # Frozen stream
    # =========================================================================

    stream = roundtrip.load_epoch0_stream(
        preflight
    )

    shared_cpu = (
        roundtrip.load_shared_inputs(
            preflight
        )
    )

    batches = {
        0: roundtrip.decode_batch(
            stream,
            0,
        ),
        1: roundtrip.decode_batch(
            stream,
            1,
        ),
    }

    references = {
        0: q.load_reference_state(
            batch_index=0,
            roundtrip=roundtrip,
            preflight=preflight,
            mac_reference=mac_reference,
        ),
        1: q.load_reference_state(
            batch_index=1,
            roundtrip=roundtrip,
            preflight=preflight,
            mac_reference=mac_reference,
        ),
    }

    # =========================================================================
    # Corrected Phase-5 device executor
    # =========================================================================

    (
        lean_execute,
        set_executor_device,
        executor_metadata,
    ) = packed.build_corrected_lean_device_executor(
        roundtrip
    )

    require(
        executor_metadata[
            "require_statements_removed"
        ] == 25,
        "Unexpected executor guard count.",
    )

    require(
        executor_metadata[
            "torch_from_numpy_rewrites"
        ] == 10,
        "Unexpected device-rewrite count.",
    )

    # =========================================================================
    # Candidate
    # =========================================================================

    banner(
        "CONSTRUCT PACKED + D_BOTH_SPARSE CUDA CANDIDATE"
    )

    (
        model,
        old_optimizer,
        hash_fn,
        *_,
    ) = roundtrip.construct_fresh_training_state(
        preflight
    )

    require(
        hash_fn(model)
        == q.EXPECTED_INITIAL_MODEL_SHA,
        "Candidate initial state drift.",
    )

    del old_optimizer

    model = model.to(cuda)

    model.startup_embedding.sparse = True
    model.investor_embedding.sparse = True

    model.train()

    optimizer = (
        preflight.build_frozen_adam(
            model
        )
    )

    require(
        len(optimizer.state) == 0,
        "CUDA Adam must begin empty.",
    )

    require(
        hash_fn(model)
        == q.EXPECTED_INITIAL_MODEL_SHA,
        "CPU → CUDA changed initial parameter bytes.",
    )

    shared_cuda = dict(shared_cpu)

    shared_cuda["edge_index"] = (
        shared_cpu["edge_index"].to(cuda)
    )

    shared_cuda["edge_type"] = (
        shared_cpu["edge_type"].to(cuda)
    )

    capture = packed.LogitCapture(
        model
    )

    capture.install()

    set_executor_device("cuda:0")

    print("Initial state:               BYTE-EXACT")
    print("startup_embedding.sparse:    True")
    print("investor_embedding.sparse:   True")
    print("Embedding strategy:          PACKED")

    # =========================================================================
    # Stateful batch 0 -> batch 1
    # =========================================================================

    rows = []
    policy_rows_all = []
    plan_rows = []

    for batch_index in (0, 1):

        batch = batches[batch_index]

        reference = references[
            batch_index
        ]

        # ---------------------------------------------------------------------
        # Deterministic Phase-5 packed call plan
        # ---------------------------------------------------------------------

        plan = (
            packed.build_embedding_call_plan(
                batch,
                shared_cpu,
                num_history_periods=(
                    roundtrip.NUM_HISTORY_PERIODS
                ),
                num_investors=(
                    roundtrip.NUM_INVESTORS
                ),
            )
        )

        startup_calls = (
            list(
                plan[
                    "history_startup_calls"
                ]
            )
            + [
                plan[
                    "pair_startup_call"
                ]
            ]
        )

        investor_calls = (
            list(
                plan[
                    "trend_investor_calls"
                ]
            )
            + [
                plan[
                    "pair_investor_call"
                ]
            ]
        )

        plan_rows.append(
            {
                "batch_index": batch_index,
                "canonical_startup_calls": (
                    len(startup_calls)
                ),
                "packed_startup_calls": 1,
                "canonical_investor_calls": (
                    len(investor_calls)
                ),
                "packed_investor_calls": 1,
                "unique_trend_keys": (
                    plan["unique_key_count"]
                ),
                "packed_startup_indices": int(
                    sum(
                        len(x)
                        for x in startup_calls
                    )
                ),
                "packed_investor_indices": int(
                    sum(
                        len(x)
                        for x in investor_calls
                    )
                ),
            }
        )

        torch.cuda.synchronize()

        start = time.perf_counter()

        # Important: create packed graph nodes fresh for every batch.
        optimizer.zero_grad(
            set_to_none=True
        )

        startup_dispatch = (
            packed.DevicePackedEmbeddingDispatch(
                model.startup_embedding,
                startup_calls,
                device=cuda,
            )
        )

        investor_dispatch = (
            packed.DevicePackedEmbeddingDispatch(
                model.investor_embedding,
                investor_calls,
                device=cuda,
            )
        )

        startup_dispatch.install()
        investor_dispatch.install()

        try:

            candidate_result = (
                lean_execute(
                    model,
                    optimizer,
                    lambda _model: (
                        "LEAN_RUNTIME_HASH_SKIPPED"
                    ),
                    batch,
                    shared_cuda,
                )
            )

            startup_dispatch.assert_consumed()
            investor_dispatch.assert_consumed()

            torch.cuda.synchronize()

        finally:

            startup_dispatch.restore()
            investor_dispatch.restore()

        elapsed = (
            time.perf_counter()
            - start
        )

        candidate_logits = (
            capture.cpu_copy()
        )

        # ---------------------------------------------------------------------
        # Frozen policy against original Mac CPU reference
        # ---------------------------------------------------------------------

        logit = packed.tensor_stats(
            reference["logits"],
            candidate_logits,
        )

        gradient = (
            packed.compare_model_gradients(
                reference["model"],
                model,
            )
        )

        parameter = (
            packed.compare_model_parameters(
                reference["model"],
                model,
            )
        )

        adam_avg = (
            packed.compare_adam_group(
                reference["model"],
                reference["optimizer"],
                model,
                optimizer,
                "exp_avg",
            )
        )

        adam_sq = (
            packed.compare_adam_group(
                reference["model"],
                reference["optimizer"],
                model,
                optimizer,
                "exp_avg_sq",
            )
        )

        summary = {
            "batch_index": batch_index,
            "cuda_seconds": elapsed,
            "reference_loss": (
                reference["loss"]
            ),
            "cuda_loss": float(
                candidate_result["loss"]
            ),
            "loss_abs_diff": abs(
                float(
                    candidate_result["loss"]
                )
                - reference["loss"]
            ),
            "logit_max_abs_diff": (
                logit["max_abs_diff"]
            ),
            "gradient_relative_l2_error": (
                gradient[
                    "relative_l2_error"
                ]
            ),
            "gradient_cosine_similarity": (
                gradient[
                    "cosine_similarity"
                ]
            ),
            "gradient_sign_agreement": (
                gradient[
                    "sign_agreement"
                ]
            ),
            "parameter_relative_l2_error": (
                parameter[
                    "relative_l2_error"
                ]
            ),
            "parameter_max_abs_diff": (
                parameter[
                    "max_abs_diff"
                ]
            ),
            "adam_exp_avg_relative_l2_error": (
                adam_avg[
                    "relative_l2_error"
                ]
            ),
            "adam_exp_avg_cosine_similarity": (
                adam_avg[
                    "cosine_similarity"
                ]
            ),
            "adam_exp_avg_sq_relative_l2_error": (
                adam_sq[
                    "relative_l2_error"
                ]
            ),
            "adam_exp_avg_sq_cosine_similarity": (
                adam_sq[
                    "cosine_similarity"
                ]
            ),
        }

        policy_rows = packed.policy_rows(
            batch_index,
            summary,
            thresholds,
        )

        policy_rows_all.extend(
            policy_rows
        )

        rows.append(summary)

        passed = all(
            r["result"] == "PASS"
            for r in policy_rows
        )

        print()
        print(f"BATCH {batch_index}")
        print(
            f"  packed CUDA seconds:              "
            f"{elapsed:.3f}"
        )
        print(
            f"  startup embedding calls:          "
            f"{len(startup_calls)} -> 1"
        )
        print(
            f"  investor embedding calls:         "
            f"{len(investor_calls)} -> 1"
        )
        print(
            f"  loss abs diff:                    "
            f"{summary['loss_abs_diff']:.12e}"
        )
        print(
            f"  logit max abs diff:               "
            f"{summary['logit_max_abs_diff']:.12e}"
        )
        print(
            f"  gradient relative L2:             "
            f"{summary['gradient_relative_l2_error']:.12e}"
        )
        print(
            f"  gradient cosine:                  "
            f"{summary['gradient_cosine_similarity']:.12f}"
        )
        print(
            f"  gradient sign agreement:          "
            f"{summary['gradient_sign_agreement']:.12f}"
        )
        print(
            f"  parameter relative L2:            "
            f"{summary['parameter_relative_l2_error']:.12e}"
        )
        print(
            f"  parameter max abs diff:           "
            f"{summary['parameter_max_abs_diff']:.12e}"
        )
        print(
            f"  Adam exp_avg relative L2:         "
            f"{summary['adam_exp_avg_relative_l2_error']:.12e}"
        )
        print(
            f"  Adam exp_avg_sq relative L2:      "
            f"{summary['adam_exp_avg_sq_relative_l2_error']:.12e}"
        )
        print(
            f"  FROZEN POLICY:                    "
            f"{'PASS' if passed else 'FAIL'}"
        )

    # =========================================================================
    # Decision
    # =========================================================================

    passed_checks = sum(
        r["result"] == "PASS"
        for r in policy_rows_all
    )

    total_checks = len(
        policy_rows_all
    )

    qualified = bool(
        total_checks == 22
        and passed_checks == 22
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        rows
    ).to_csv(
        OUT_DIR
        / "packed_sparse_cuda_batch_summary.csv",
        index=False,
    )

    pd.DataFrame(
        plan_rows
    ).to_csv(
        OUT_DIR
        / "packed_embedding_call_plan.csv",
        index=False,
    )

    pd.DataFrame(
        policy_rows_all
    ).to_csv(
        OUT_DIR
        / "packed_sparse_cuda_policy_audit.csv",
        index=False,
    )

    contract = {
        "phase": "6.2",
        "status": (
            "QUALIFIED"
            if qualified
            else "FAILED"
        ),
        "candidate_runtime": (
            "PACKED_ALL_PLUS_D_BOTH_SPARSE_"
            "CUDA_STRICT_FP32"
        ),
        "reference_runtime": (
            "FROZEN_MAC_CANONICAL_CPU"
        ),
        "numerical_policy": (
            q.EXPECTED_POLICY
        ),
        "strict_fp32": True,
        "tf32_enabled": False,
        "deterministic_algorithms": True,
        "stateful_batches_tested": 2,
        "policy_checks_passed": (
            passed_checks
        ),
        "policy_checks_total": (
            total_checks
        ),
        "numerically_equivalent": (
            qualified
        ),
        "batch_seconds": {
            str(row["batch_index"]): (
                row["cuda_seconds"]
            )
            for row in rows
        },
        "production_training_launched": False,
        "validation_cases_scored": 0,
        "test_cases_scored": 0,
    }

    CONTRACT_PATH.write_text(
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    banner(
        "PHASE 6.2 RESULT"
    )

    print(
        f"Policy checks:       "
        f"{passed_checks}/{total_checks}"
    )

    print(
        "Production training: NO"
    )

    print(
        "Validation accessed: NO"
    )

    print(
        "Test accessed:       NO"
    )

    if qualified:

        print()
        print(
            "PACKED + D_BOTH_SPARSE CUDA STRICT FP32: "
            "NUMERICALLY EQUIVALENT"
        )
        print(
            "PHASE 6.2: PASS"
        )

    else:

        print()
        print(
            "PACKED + D_BOTH_SPARSE CUDA STRICT FP32: "
            "FAILED FROZEN POLICY"
        )
        print(
            "PHASE 6.2: FAIL"
        )

        print()
        print("FAILED CHECKS:")

        for row in policy_rows_all:
            if row["result"] != "PASS":
                print(
                    f"  batch={row['batch_index']} "
                    f"{row['metric']}: "
                    f"{row['actual']} "
                    f"{row['comparator']} "
                    f"{row['threshold']}"
                )

        raise AssertionError(
            "Packed+sparse CUDA candidate failed frozen policy."
        )


if __name__ == "__main__":
    main()
