#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Configuration
# =============================================================================

TOTAL_BATCHES = 32
WARMUP_BATCHES = 4

BATCHES_PER_EPOCH = 10_481
TOTAL_EPOCHS = 20
TOTAL_OPTIMIZER_STEPS = 209_620

PHASE6_BASE_PATH = Path(
    "scripts/phase_6_1_cuda_mac_reference_qualification.py"
)

PACKED_RUNTIME_PATH = Path(
    "scripts/"
    "phase_5_4_8b_packed_mps_numerical_equivalence_feasibility_audit_V2.py"
)

PHASE6_2_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_2_packed_sparse_cuda_strict_fp32_qualification.json"
)

OUT_DIR = Path(
    "data/experimental/phase_6/audits/"
    "phase_6_3_packed_sparse_cuda_throughput"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_3_packed_sparse_cuda_throughput_benchmark.json"
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


def human_duration(seconds):
    seconds = float(seconds)

    if seconds < 60:
        return f"{seconds:.2f} sec"

    minutes = seconds / 60

    if minutes < 60:
        return f"{minutes:.2f} min"

    hours = minutes / 60

    if hours < 48:
        return f"{hours:.2f} h"

    return f"{hours / 24:.2f} days"


def main():

    banner(
        "PHASE 6.3 — QUALIFIED PACKED + SPARSE CUDA "
        "PRODUCTION-LIKE THROUGHPUT BENCHMARK"
    )

    print("Full epoch launched:          NO")
    print("Production training launched: NO")
    print("Validation accessed:          NO")
    print("Test accessed:                NO")
    print("Production checkpoint saved:  NO")
    print()
    print(f"Sequential batches:           {TOTAL_BATCHES}")
    print(f"Warm-up batches discarded:    {WARMUP_BATCHES}")
    print(
        f"Measured batches:             "
        f"{TOTAL_BATCHES - WARMUP_BATCHES}"
    )

    # =========================================================================
    # Strict FP32 CUDA configuration — exactly qualified configuration
    # =========================================================================

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.set_float32_matmul_precision(
        "highest"
    )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.use_deterministic_algorithms(
        True
    )

    require(
        torch.cuda.is_available(),
        "CUDA unavailable.",
    )

    require(
        torch.__version__.startswith("2.7.0"),
        f"PyTorch drift: {torch.__version__}",
    )

    cuda = torch.device("cuda:0")

    banner("CUDA RUNTIME")

    print(
        "GPU:                      ",
        torch.cuda.get_device_name(0),
    )
    print(
        "PyTorch:                  ",
        torch.__version__,
    )
    print(
        "CUDA runtime:             ",
        torch.version.cuda,
    )
    print(
        "matmul TF32:              ",
        torch.backends.cuda.matmul.allow_tf32,
    )
    print(
        "cuDNN TF32:               ",
        torch.backends.cudnn.allow_tf32,
    )
    print(
        "float32 matmul precision: ",
        torch.get_float32_matmul_precision(),
    )
    print(
        "deterministic algorithms: ",
        torch.are_deterministic_algorithms_enabled(),
    )

    # =========================================================================
    # Gate on Phase 6.2
    # =========================================================================

    require(
        PHASE6_2_CONTRACT_PATH.exists(),
        "Missing Phase-6.2 qualification contract.",
    )

    phase6_2 = json.loads(
        PHASE6_2_CONTRACT_PATH.read_text(
            encoding="utf-8"
        )
    )

    require(
        phase6_2["status"] == "QUALIFIED",
        "Phase 6.2 is not QUALIFIED.",
    )

    require(
        phase6_2[
            "numerically_equivalent"
        ] is True,
        "Phase 6.2 numerical equivalence not TRUE.",
    )

    require(
        int(
            phase6_2[
                "policy_checks_passed"
            ]
        ) == 22,
        "Phase 6.2 did not pass 22 checks.",
    )

    require(
        int(
            phase6_2[
                "policy_checks_total"
            ]
        ) == 22,
        "Phase 6.2 policy total is not 22.",
    )

    print()
    print(
        "Phase 6.2 qualification gate: PASS"
    )

    # =========================================================================
    # Load Phase-6 portable runtime infrastructure
    # =========================================================================

    q = load_module(
        PHASE6_BASE_PATH,
        "_phase6_3_base",
    )

    packed = load_module(
        PACKED_RUNTIME_PATH,
        "_phase6_3_packed",
    )

    require(
        q.file_sha256(
            q.INITIAL_STATE_PATH
        )
        == q.EXPECTED_INITIAL_FILE_SHA,
        "Portable initial state SHA drift.",
    )

    require(
        q.file_sha256(
            q.MAC_REFERENCE_PATH
        )
        == q.EXPECTED_REFERENCE_FILE_SHA,
        "Mac reference SHA drift.",
    )

    initial_payload = torch.load(
        q.INITIAL_STATE_PATH,
        map_location="cpu",
        weights_only=False,
    )

    roundtrip = q.load_module(
        q.ROUNDTRIP_PATH,
        "_phase6_3_roundtrip",
    )

    preflight = (
        roundtrip.load_preflight_runtime()
    )

    q.install_portable_initialization_bridge(
        preflight,
        initial_payload[
            "model_state_dict"
        ],
    )

    # =========================================================================
    # Frozen epoch-0 stream and shared inputs
    # =========================================================================

    banner(
        "LOAD FROZEN EPOCH-0 TRAINING STREAM"
    )

    stream = (
        roundtrip.load_epoch0_stream(
            preflight
        )
    )

    shared_cpu = (
        roundtrip.load_shared_inputs(
            preflight
        )
    )

    (
        lean_execute,
        set_executor_device,
        executor_metadata,
    ) = (
        packed
        .build_corrected_lean_device_executor(
            roundtrip
        )
    )

    require(
        executor_metadata[
            "require_statements_removed"
        ]
        == 25,
        "Executor guard-removal drift.",
    )

    require(
        executor_metadata[
            "torch_from_numpy_rewrites"
        ]
        == 10,
        "Executor device rewrite drift.",
    )

    print("Frozen stream:             PASS")
    print("Device executor:           PASS")

    # =========================================================================
    # Exact initial model -> CUDA
    # =========================================================================

    banner(
        "INITIALIZE QUALIFIED CUDA CANDIDATE"
    )

    (
        model,
        old_optimizer,
        hash_fn,
        *_,
    ) = (
        roundtrip
        .construct_fresh_training_state(
            preflight
        )
    )

    require(
        hash_fn(model)
        == q.EXPECTED_INITIAL_MODEL_SHA,
        "Fresh model does not match frozen initial state.",
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
        "Fresh CUDA Adam is not empty.",
    )

    require(
        hash_fn(model)
        == q.EXPECTED_INITIAL_MODEL_SHA,
        "CPU->CUDA parameter bytes changed.",
    )

    shared_cuda = dict(
        shared_cpu
    )

    shared_cuda[
        "edge_index"
    ] = shared_cpu[
        "edge_index"
    ].to(cuda)

    shared_cuda[
        "edge_type"
    ] = shared_cpu[
        "edge_type"
    ].to(cuda)

    set_executor_device(
        "cuda:0"
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(
        "Frozen initial state:      PASS"
    )
    print(
        "startup sparse:            True"
    )
    print(
        "investor sparse:           True"
    )
    print(
        "embedding dispatch:        PACKED"
    )

    # =========================================================================
    # Sequential production-like benchmark
    # =========================================================================

    banner(
        "EXECUTE 32 STATEFUL TRAINING BATCHES"
    )

    rows = []

    for batch_index in range(
        TOTAL_BATCHES
    ):

        # Start end-to-end timer BEFORE batch decoding and call-plan creation.
        e2e_start = time.perf_counter()

        batch = (
            roundtrip.decode_batch(
                stream,
                batch_index,
            )
        )

        plan_start = time.perf_counter()

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

        plan_seconds = (
            time.perf_counter()
            - plan_start
        )

        # Previous batch was synchronized at its end.
        torch.cuda.synchronize()

        gpu_start = time.perf_counter()

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

            result = lean_execute(
                model,
                optimizer,
                lambda _model: (
                    "LEAN_RUNTIME_HASH_SKIPPED"
                ),
                batch,
                shared_cuda,
            )

            startup_dispatch.assert_consumed()
            investor_dispatch.assert_consumed()

            torch.cuda.synchronize()

        finally:

            startup_dispatch.restore()
            investor_dispatch.restore()

        gpu_seconds = (
            time.perf_counter()
            - gpu_start
        )

        e2e_seconds = (
            time.perf_counter()
            - e2e_start
        )

        measured = (
            batch_index
            >= WARMUP_BATCHES
        )

        row = {
            "batch_index": batch_index,
            "measured": measured,
            "batch_size": len(batch),
            "loss": float(
                result["loss"]
            ),
            "plan_seconds": plan_seconds,
            "gpu_section_seconds": (
                gpu_seconds
            ),
            "end_to_end_seconds": (
                e2e_seconds
            ),
            "startup_embedding_calls": (
                len(startup_calls)
            ),
            "investor_embedding_calls": (
                len(investor_calls)
            ),
            "unique_trend_keys": (
                plan[
                    "unique_key_count"
                ]
            ),
        }

        rows.append(row)

        marker = (
            "MEASURE"
            if measured
            else "WARMUP"
        )

        print(
            f"batch {batch_index:02d} | "
            f"{marker:7s} | "
            f"e2e={e2e_seconds:7.3f}s | "
            f"gpu={gpu_seconds:7.3f}s | "
            f"plan={plan_seconds:6.3f}s | "
            f"startup_calls={len(startup_calls):4d} | "
            f"investor_calls={len(investor_calls):4d}"
        )

    # =========================================================================
    # Results
    # =========================================================================

    frame = pd.DataFrame(rows)

    measured = frame[
        frame["measured"]
    ].copy()

    times = (
        measured[
            "end_to_end_seconds"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    gpu_times = (
        measured[
            "gpu_section_seconds"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    mean_seconds = float(
        np.mean(times)
    )

    median_seconds = float(
        np.median(times)
    )

    p95_seconds = float(
        np.percentile(
            times,
            95,
        )
    )

    std_seconds = float(
        np.std(
            times,
            ddof=1,
        )
    )

    mean_gpu_seconds = float(
        np.mean(
            gpu_times
        )
    )

    epoch_seconds = (
        mean_seconds
        * BATCHES_PER_EPOCH
    )

    twenty_epoch_seconds = (
        mean_seconds
        * TOTAL_OPTIMIZER_STEPS
    )

    peak_allocated_gb = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    peak_reserved_gb = (
        torch.cuda.max_memory_reserved()
        / 1024**3
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        OUT_DIR
        / "batch_runtime_measurements.csv",
        index=False,
    )

    summary = {
        "phase": "6.3",
        "status": "BENCHMARK_COMPLETE",
        "runtime": (
            "PACKED_ALL_PLUS_"
            "D_BOTH_SPARSE_"
            "CUDA_STRICT_FP32"
        ),
        "gpu": (
            torch.cuda.get_device_name(0)
        ),
        "torch_version": (
            torch.__version__
        ),
        "cuda_runtime": (
            torch.version.cuda
        ),
        "total_batches_executed": (
            TOTAL_BATCHES
        ),
        "warmup_batches": (
            WARMUP_BATCHES
        ),
        "measured_batches": (
            len(measured)
        ),
        "mean_end_to_end_seconds_per_batch": (
            mean_seconds
        ),
        "median_end_to_end_seconds_per_batch": (
            median_seconds
        ),
        "p95_end_to_end_seconds_per_batch": (
            p95_seconds
        ),
        "std_end_to_end_seconds_per_batch": (
            std_seconds
        ),
        "mean_gpu_section_seconds_per_batch": (
            mean_gpu_seconds
        ),
        "projected_epoch_seconds_training_only": (
            epoch_seconds
        ),
        "projected_20_epoch_seconds_training_only": (
            twenty_epoch_seconds
        ),
        "peak_cuda_allocated_gb": (
            peak_allocated_gb
        ),
        "peak_cuda_reserved_gb": (
            peak_reserved_gb
        ),
        "production_training_launched": (
            False
        ),
        "validation_accessed": (
            False
        ),
        "test_accessed": (
            False
        ),
        "production_checkpoint_saved": (
            False
        ),
    }

    CONTRACT_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    banner(
        "PHASE 6.3 THROUGHPUT RESULT"
    )

    print(
        f"Measured batches:           "
        f"{len(measured)}"
    )

    print(
        f"Mean end-to-end / batch:    "
        f"{mean_seconds:.3f} s"
    )

    print(
        f"Median end-to-end / batch:  "
        f"{median_seconds:.3f} s"
    )

    print(
        f"P95 end-to-end / batch:     "
        f"{p95_seconds:.3f} s"
    )

    print(
        f"Std dev:                    "
        f"{std_seconds:.3f} s"
    )

    print(
        f"Mean GPU section / batch:   "
        f"{mean_gpu_seconds:.3f} s"
    )

    print(
        f"Projected epoch training:   "
        f"{human_duration(epoch_seconds)}"
    )

    print(
        f"Projected 20 epochs train:  "
        f"{human_duration(twenty_epoch_seconds)}"
    )

    print(
        f"Peak CUDA allocated:        "
        f"{peak_allocated_gb:.2f} GiB"
    )

    print(
        f"Peak CUDA reserved:         "
        f"{peak_reserved_gb:.2f} GiB"
    )

    print()
    print(
        "Validation overhead:        NOT INCLUDED"
    )

    print(
        "Checkpoint overhead:        NOT INCLUDED"
    )

    print(
        "Test accessed:              NO"
    )

    print(
        "Production training:        NO"
    )

    print()
    print(
        "PHASE 6.3: BENCHMARK COMPLETE"
    )


if __name__ == "__main__":
    main()
