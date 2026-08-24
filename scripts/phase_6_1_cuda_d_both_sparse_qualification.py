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


HELPER_PATH = Path(
    "scripts/phase_5_4_8a_mps_numerical_equivalence_feasibility_audit_V1.py"
)

OUT_DIR = Path(
    "data/experimental/phase_6/audits/phase_6_1_cuda_d_both_sparse"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_1_cuda_d_both_sparse_qualification.json"
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def banner(text):
    print("\n" + "=" * 110)
    print(text)
    print("=" * 110)


def main():

    banner(
        "PHASE 6.1 — RTX 4090 CUDA D_BOTH_SPARSE NUMERICAL QUALIFICATION"
    )

    print("Production training launched: NO")
    print("Validation scored:            NO")
    print("Test scored:                  NO")

    assert torch.__version__.startswith("2.7.0"), torch.__version__
    assert torch.cuda.is_available()

    print()
    print("PyTorch:", torch.__version__)
    print("CUDA runtime:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

    helper = load_module(
        HELPER_PATH,
        "_phase6_cuda_helper",
    )

    roundtrip = helper.load_module(
        helper.ROUNDTRIP_SOURCE_PATH,
        "_phase6_roundtrip",
    )

    policy = helper.load_json(
        helper.POLICY_PATH
    )

    assert (
        policy["schema_version"]
        == "ITRS_PHASE5_NUMERICAL_EQUIVALENCE_V1"
    )

    thresholds = policy["thresholds"]

    torch.set_num_threads(8)

    banner("LOAD FROZEN PHASE-5 RUNTIME")

    preflight = roundtrip.load_preflight_runtime()

    stream = roundtrip.load_epoch0_stream(
        preflight
    )

    shared_cpu = roundtrip.load_shared_inputs(
        preflight
    )

    batch0 = roundtrip.decode_batch(
        stream, 0
    )

    batch1 = roundtrip.decode_batch(
        stream, 1
    )

    (
        lean_execute,
        set_executor_device,
        executor_metadata,
    ) = helper.build_lean_device_executor(
        roundtrip
    )

    print(
        "Removed audit-only require calls:",
        executor_metadata["require_statements_removed"],
    )
    print(
        "Device-aware from_numpy rewrites:",
        executor_metadata["torch_from_numpy_rewrites"],
    )

    banner("CONSTRUCT IDENTICAL INITIAL MODEL STATES")

    (
        reference_model,
        reference_optimizer,
        reference_hash_fn,
        *_,
    ) = roundtrip.construct_fresh_training_state(
        preflight
    )

    (
        candidate_model,
        candidate_optimizer_cpu,
        candidate_hash_fn,
        *_,
    ) = roundtrip.construct_fresh_training_state(
        preflight
    )

    expected_initial = (
        "49e822ea7fad35c458f47e134c94c05e"
        "ac099b68c5c468e2c71559c8c88998ab"
    )

    reference_initial = reference_hash_fn(
        reference_model
    )

    candidate_initial = candidate_hash_fn(
        candidate_model
    )

    assert reference_initial == expected_initial
    assert candidate_initial == expected_initial

    print("CPU reference initial state: BYTE-EXACT")
    print("CUDA candidate pre-transfer: BYTE-EXACT")

    del candidate_optimizer_cpu

    cuda = torch.device("cuda:0")

    candidate_model = candidate_model.to(
        cuda
    )

    # Frozen Phase-5 accelerated implementation choice.
    candidate_model.startup_embedding.sparse = True
    candidate_model.investor_embedding.sparse = True

    candidate_model.train()

    candidate_optimizer = (
        preflight.build_frozen_adam(
            candidate_model
        )
    )

    assert len(candidate_optimizer.state) == 0

    shared_cuda = dict(shared_cpu)

    shared_cuda["edge_index"] = (
        shared_cpu["edge_index"].to(cuda)
    )

    shared_cuda["edge_type"] = (
        shared_cpu["edge_type"].to(cuda)
    )

    reference_capture = helper.LogitCapture(
        reference_model
    )
    candidate_capture = helper.LogitCapture(
        candidate_model
    )

    reference_capture.install()
    candidate_capture.install()

    anchors = [
        (
            0,
            batch0,
            helper.EXPECTED_BATCH0_LOSS,
            helper.EXPECTED_BATCH0_LOGIT_SHA256,
            helper.EXPECTED_BATCH0_GRADIENT_SHA256,
            helper.EXPECTED_BATCH0_POST_MODEL_SHA256,
            helper.EXPECTED_BATCH0_OPTIMIZER_SHA256,
        ),
        (
            1,
            batch1,
            helper.EXPECTED_BATCH1_LOSS,
            helper.EXPECTED_BATCH1_LOGIT_SHA256,
            helper.EXPECTED_BATCH1_GRADIENT_SHA256,
            helper.EXPECTED_BATCH1_POST_MODEL_SHA256,
            helper.EXPECTED_BATCH1_OPTIMIZER_SHA256,
        ),
    ]

    batch_rows = []
    policy_rows_all = []

    banner("STATEFUL CPU → CUDA COMPARISON")

    for (
        batch_index,
        batch,
        expected_loss,
        expected_logit_sha,
        expected_gradient_sha,
        expected_model_sha,
        expected_optimizer_sha,
    ) in anchors:

        # --------------------------------------------------
        # Canonical CPU reference
        # --------------------------------------------------

        set_executor_device("cpu")

        reference_start = time.perf_counter()

        reference_result = lean_execute(
            reference_model,
            reference_optimizer,
            lambda model: "SKIPPED",
            batch,
            shared_cpu,
        )

        reference_seconds = (
            time.perf_counter()
            - reference_start
        )

        reference_logits = (
            reference_capture.cpu_copy()
        )

        reference_gradient_sha = (
            roundtrip.gradient_logical_sha256(
                reference_model
            )
        )

        reference_model_sha = (
            reference_hash_fn(
                reference_model
            )
        )

        reference_optimizer_sha = (
            roundtrip.optimizer_state_logical_sha256(
                reference_model,
                reference_optimizer,
            )
        )

        reference_exact = (
            abs(
                float(reference_result["loss"])
                - float(expected_loss)
            )
            <= 5e-10
            and roundtrip.tensor_sha256(
                reference_logits
            )
            == expected_logit_sha
            and reference_gradient_sha
            == expected_gradient_sha
            and reference_model_sha
            == expected_model_sha
            and reference_optimizer_sha
            == expected_optimizer_sha
        )

        assert reference_exact, (
            f"CPU frozen anchor failed at batch {batch_index}"
        )

        # --------------------------------------------------
        # CUDA D_BOTH_SPARSE candidate
        # --------------------------------------------------

        set_executor_device("cuda:0")

        torch.cuda.synchronize()

        cuda_start = time.perf_counter()

        candidate_result = lean_execute(
            candidate_model,
            candidate_optimizer,
            lambda model: "SKIPPED",
            batch,
            shared_cuda,
        )

        torch.cuda.synchronize()

        cuda_seconds = (
            time.perf_counter()
            - cuda_start
        )

        candidate_logits = (
            candidate_capture.cpu_copy()
        )

        # --------------------------------------------------
        # Numerical comparisons
        # --------------------------------------------------

        logit = helper.tensor_stats(
            reference_logits,
            candidate_logits,
        )

        gradient = (
            helper.compare_model_gradients(
                reference_model,
                candidate_model,
            )
        )

        parameter = (
            helper.compare_model_parameters(
                reference_model,
                candidate_model,
            )
        )

        adam_avg = helper.compare_adam_group(
            reference_model,
            reference_optimizer,
            candidate_model,
            candidate_optimizer,
            "exp_avg",
        )

        adam_sq = helper.compare_adam_group(
            reference_model,
            reference_optimizer,
            candidate_model,
            candidate_optimizer,
            "exp_avg_sq",
        )

        loss_diff = abs(
            float(candidate_result["loss"])
            - float(reference_result["loss"])
        )

        summary = {
            "batch_index": batch_index,
            "cpu_seconds": reference_seconds,
            "cuda_seconds": cuda_seconds,
            "reference_loss": float(
                reference_result["loss"]
            ),
            "cuda_loss": float(
                candidate_result["loss"]
            ),
            "loss_abs_diff": loss_diff,
            "logit_max_abs_diff": (
                logit["max_abs_diff"]
            ),
            "gradient_relative_l2_error": (
                gradient["relative_l2_error"]
            ),
            "gradient_cosine_similarity": (
                gradient["cosine_similarity"]
            ),
            "gradient_sign_agreement": (
                gradient["sign_agreement"]
            ),
            "parameter_relative_l2_error": (
                parameter["relative_l2_error"]
            ),
            "parameter_max_abs_diff": (
                parameter["max_abs_diff"]
            ),
            "adam_exp_avg_relative_l2_error": (
                adam_avg["relative_l2_error"]
            ),
            "adam_exp_avg_cosine_similarity": (
                adam_avg["cosine_similarity"]
            ),
            "adam_exp_avg_sq_relative_l2_error": (
                adam_sq["relative_l2_error"]
            ),
            "adam_exp_avg_sq_cosine_similarity": (
                adam_sq["cosine_similarity"]
            ),
        }

        rows = helper.policy_rows(
            batch_index,
            summary,
            thresholds,
        )

        policy_rows_all.extend(rows)
        batch_rows.append(summary)

        passed = all(
            row["result"] == "PASS"
            for row in rows
        )

        print()
        print(f"BATCH {batch_index}")
        print(
            f"  CPU frozen reference: {reference_seconds:.3f}s"
        )
        print(
            f"  RTX 4090 CUDA:         {cuda_seconds:.3f}s"
        )
        print(
            f"  loss abs diff:         {loss_diff:.3e}"
        )
        print(
            "  gradient rel L2:       "
            f"{gradient['relative_l2_error']:.3e}"
        )
        print(
            "  parameter rel L2:      "
            f"{parameter['relative_l2_error']:.3e}"
        )
        print(
            "  policy:                "
            f"{'PASS' if passed else 'FAIL'}"
        )

    policy_pass = all(
        row["result"] == "PASS"
        for row in policy_rows_all
    )

    checks_passed = sum(
        row["result"] == "PASS"
        for row in policy_rows_all
    )

    total_checks = len(
        policy_rows_all
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
        batch_rows
    ).to_csv(
        OUT_DIR / "cuda_batch_summary.csv",
        index=False,
    )

    pd.DataFrame(
        policy_rows_all
    ).to_csv(
        OUT_DIR / "cuda_numerical_policy_audit.csv",
        index=False,
    )

    contract = {
        "phase": "6.1",
        "candidate_runtime": (
            "D_BOTH_SPARSE_CUDA"
        ),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "policy": (
            "ITRS_PHASE5_NUMERICAL_EQUIVALENCE_V1"
        ),
        "stateful_batches_tested": 2,
        "policy_checks_passed": checks_passed,
        "policy_checks_total": total_checks,
        "numerically_equivalent": policy_pass,
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

    banner("CUDA QUALIFICATION RESULT")

    print(
        f"Policy checks: {checks_passed}/{total_checks}"
    )

    print(
        "D_BOTH_SPARSE_CUDA: "
        + (
            "NUMERICALLY EQUIVALENT"
            if policy_pass
            else "FAILED"
        )
    )

    if not policy_pass:
        raise AssertionError(
            "CUDA candidate failed frozen numerical policy."
        )


if __name__ == "__main__":
    main()
