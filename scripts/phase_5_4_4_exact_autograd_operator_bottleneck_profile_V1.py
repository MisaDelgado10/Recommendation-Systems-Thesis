#!/usr/bin/env python3
"""
Phase 5.4.4 — Exact Autograd Operator Bottleneck Profile

Purpose
-------
Identify the low-level PyTorch operator(s) responsible for the ~24.5 second
canonical backward pass observed in Phase 5.4.3.

This phase:
    - profiles ONE exact frozen epoch-0 batch (batch 0)
    - uses the exact Phase-5.4.2 CPU runtime (8 intra-op threads)
    - removes proof-only hashes from inside the profiler interval
    - recomputes all frozen numerical fingerprints after profiling
    - requires exact batch-0 numerical equivalence

No production training state is retained.
No checkpoint is written.
No validation or test is accessed.

The output ranks PyTorch operators by self CPU time and total CPU time.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import math
import platform
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile


# =============================================================================
# Paths
# =============================================================================

ROUNDTRIP_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_2b_checkpoint_resume_roundtrip_proof.py"
)

PHASE_5_4_3_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_3_exact_runtime_bottleneck_profile_contract.json"
)

PHASE_5_4_2_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_2_lean_exact_cpu_runtime_contract.json"
)

PHASE_5_4_AUTHORIZATION_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_production_training_launch_authorization.json"
)

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_4_4"
)

OPERATOR_PROFILE_PATH = (
    AUDIT_DIR
    / "autograd_operator_profile_by_self_cpu.csv"
)

OPERATOR_TOTAL_PATH = (
    AUDIT_DIR
    / "autograd_operator_profile_by_total_cpu.csv"
)

TOP_OPERATOR_PATH = (
    AUDIT_DIR
    / "autograd_top_operator_summary.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_4_4_final_invariants.csv"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_4_exact_autograd_operator_profile_contract.json"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_4_4_exact_autograd_operator_profile_manifest.json"
)


# =============================================================================
# Frozen anchors
# =============================================================================

EXPECTED_PYTORCH = "2.7.0"
EXPECTED_SELECTED_THREADS = 8

EXPECTED_INITIAL_MODEL_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_BATCH0_LOSS = 0.7080879807

EXPECTED_BATCH0_LOGIT_SHA256 = (
    "35b89aaed29d51d2ebb7ba1cadf2dc4b"
    "b5e8f81cf3aa78bc216b3cc6fed13845"
)

EXPECTED_BATCH0_GRADIENT_SHA256 = (
    "8c542430813d8ca91b8397409954ea92"
    "295a2b55bcc420661783fb865010845d"
)

EXPECTED_BATCH0_POST_MODEL_SHA256 = (
    "42a521f11d8f24e4144d0215d6e1b34d"
    "5f8bf0c2d8848624e4f7c3130699035d"
)

EXPECTED_BATCH0_OPTIMIZER_SHA256 = (
    "5ce2683c21f456b9d5d15eb876b049c5"
    "e6db1215db5a026630f093f7f9d49891"
)


# =============================================================================
# Helpers
# =============================================================================

def banner(text: str) -> None:
    print("\n" + "=" * 118)
    print(text)
    print("=" * 118)


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def load_json(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def load_module(
    path: Path,
    module_name: str,
):
    require(
        path.exists(),
        f"Missing source: {path}",
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        path,
    )

    require(
        spec is not None
        and spec.loader is not None,
        f"Could not import {path}.",
    )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def close_float(
    actual: float,
    expected: float,
    tolerance: float = 5e-10,
) -> bool:
    return (
        math.isfinite(
            float(actual)
        )
        and abs(
            float(actual)
            - float(expected)
        )
        <= tolerance
    )


def event_rows(
    profiler,
) -> pd.DataFrame:
    rows = []

    for event in profiler.key_averages():
        rows.append(
            {
                "operator": str(
                    event.key
                ),
                "count": int(
                    event.count
                ),
                "self_cpu_time_us": float(
                    event.self_cpu_time_total
                ),
                "cpu_time_total_us": float(
                    event.cpu_time_total
                ),
                "self_cpu_time_ms": float(
                    event.self_cpu_time_total
                    / 1000.0
                ),
                "cpu_time_total_ms": float(
                    event.cpu_time_total
                    / 1000.0
                ),
                "self_cpu_time_s": float(
                    event.self_cpu_time_total
                    / 1_000_000.0
                ),
                "cpu_time_total_s": float(
                    event.cpu_time_total
                    / 1_000_000.0
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.4.4 — "
        "EXACT AUTOGRAD OPERATOR BOTTLENECK PROFILE"
    )

    print(
        "Production training launched:         NO"
    )
    print(
        "Profiled production-equivalent batch: 1"
    )
    print(
        "Production checkpoint written:        NO"
    )
    print(
        "Validation cases scored:              0"
    )
    print(
        "Test cases scored:                    0"
    )

    # =========================================================================
    # Gate
    # =========================================================================

    banner(
        "PREREQUISITE GATE"
    )

    for path in (
        ROUNDTRIP_SOURCE_PATH,
        PHASE_5_4_2_CONTRACT_PATH,
        PHASE_5_4_3_CONTRACT_PATH,
        PHASE_5_4_AUTHORIZATION_PATH,
    ):
        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

        print(
            f"FOUND  {path}"
        )

    runtime_contract = load_json(
        PHASE_5_4_2_CONTRACT_PATH
    )

    profile_contract = load_json(
        PHASE_5_4_3_CONTRACT_PATH
    )

    authorization = load_json(
        PHASE_5_4_AUTHORIZATION_PATH
    )

    require(
        runtime_contract.get(
            "status"
        )
        == "COMPLETE",
        (
            "Phase-5.4.2 runtime contract is not COMPLETE."
        ),
    )

    require(
        int(
            runtime_contract.get(
                "selected_threads",
                -1,
            )
        )
        == EXPECTED_SELECTED_THREADS,
        (
            "Phase-5.4.2 selected thread count drift."
        ),
    )

    require(
        profile_contract.get(
            "status"
        )
        == "COMPLETE",
        (
            "Phase-5.4.3 profile contract is not COMPLETE."
        ),
    )

    require(
        profile_contract.get(
            "dominant_component"
        )
        == "backward",
        (
            "Phase-5.4.3 no longer identifies backward "
            "as the dominant component."
        ),
    )

    require(
        authorization.get(
            "training_allowed"
        )
        is True,
        (
            "Production training is not authorized."
        ),
    )

    print(
        "Phase-5.4.2 exact lean runtime:        PASS"
    )
    print(
        "Phase-5.4.3 dominant component:        backward"
    )
    print(
        "Phase-5.4 launch authorization:        ALLOWED"
    )

    # =========================================================================
    # Load frozen runtime
    # =========================================================================

    banner(
        "LOAD FROZEN RUNTIME"
    )

    roundtrip = load_module(
        ROUNDTRIP_SOURCE_PATH,
        "_itrs_phase5_4_4_roundtrip",
    )

    require(
        torch.__version__
        == EXPECTED_PYTORCH,
        (
            "Reference PyTorch version drift."
        ),
    )

    torch.set_num_threads(
        EXPECTED_SELECTED_THREADS
    )

    preflight = (
        roundtrip
        .load_preflight_runtime()
    )

    stream = (
        roundtrip
        .load_epoch0_stream(
            preflight
        )
    )

    shared = (
        roundtrip
        .load_shared_inputs(
            preflight
        )
    )

    batch0 = (
        roundtrip
        .decode_batch(
            stream,
            0,
        )
    )

    (
        model,
        optimizer,
        canonical_hash_fn,
        runtime_ast_sha,
        adapter_sha,
        removed_guard_sha,
    ) = (
        roundtrip
        .construct_fresh_training_state(
            preflight
        )
    )

    initial_sha = (
        canonical_hash_fn(
            model
        )
    )

    require(
        initial_sha
        == EXPECTED_INITIAL_MODEL_SHA256,
        (
            "Canonical initial model SHA drift."
        ),
    )

    print(
        f"Platform:                              "
        f"{platform.platform()}"
    )
    print(
        f"PyTorch:                               "
        f"{torch.__version__}"
    )
    print(
        f"Torch intra-op threads:                "
        f"{torch.get_num_threads()}"
    )
    print(
        "Canonical initial model:              PASS"
    )

    # =========================================================================
    # Remove proof-only hashes from profiler interval
    # =========================================================================

    true_gradient_hash_fn = (
        roundtrip
        .gradient_logical_sha256
    )

    true_optimizer_hash_fn = (
        roundtrip
        .optimizer_state_logical_sha256
    )

    roundtrip.gradient_logical_sha256 = (
        lambda model: "SKIPPED_INSIDE_PROFILER"
    )

    roundtrip.optimizer_state_logical_sha256 = (
        lambda model, optimizer: "SKIPPED_INSIDE_PROFILER"
    )

    dummy_model_hash_fn = (
        lambda model: "SKIPPED_INSIDE_PROFILER"
    )

    # =========================================================================
    # Profile one exact anchored batch
    # =========================================================================

    banner(
        "PROFILE EXACT FROZEN BATCH 0"
    )

    wall_start = time.perf_counter()

    with profile(
        activities=[
            ProfilerActivity.CPU,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
    ) as prof:

        result = (
            roundtrip
            .execute_training_batch(
                model,
                optimizer,
                dummy_model_hash_fn,
                batch0,
                shared,
            )
        )

    wall_seconds = (
        time.perf_counter()
        - wall_start
    )

    # Exact numerical proof outside profiler interval.
    gradient_sha = (
        true_gradient_hash_fn(
            model
        )
    )

    post_model_sha = (
        canonical_hash_fn(
            model
        )
    )

    optimizer_sha = (
        true_optimizer_hash_fn(
            model,
            optimizer,
        )
    )

    exact = bool(
        close_float(
            result[
                "loss"
            ],
            EXPECTED_BATCH0_LOSS,
        )
        and result[
            "logit_sha256"
        ]
        == EXPECTED_BATCH0_LOGIT_SHA256
        and gradient_sha
        == EXPECTED_BATCH0_GRADIENT_SHA256
        and post_model_sha
        == EXPECTED_BATCH0_POST_MODEL_SHA256
        and optimizer_sha
        == EXPECTED_BATCH0_OPTIMIZER_SHA256
    )

    require(
        exact,
        (
            "Profiler run did not preserve exact "
            "frozen batch-0 numerical trajectory."
        ),
    )

    print(
        f"Profiled batch BCE:                    "
        f"{result['loss']:.10f}"
    )
    print(
        f"Profiler wall time:                    "
        f"{wall_seconds:.3f} s"
    )
    print(
        "Frozen numerical anchors:             EXACT"
    )

    # =========================================================================
    # Extract operator profile
    # =========================================================================

    banner(
        "TOP AUTOGRAD / CPU OPERATORS"
    )

    operator_df = event_rows(
        prof
    )

    require(
        len(
            operator_df
        )
        > 0,
        (
            "PyTorch profiler returned no operator events."
        ),
    )

    by_self_df = (
        operator_df.sort_values(
            [
                "self_cpu_time_us",
                "operator",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    total_self_cpu_us = float(
        by_self_df[
            "self_cpu_time_us"
        ].sum()
    )

    by_self_df[
        "self_cpu_share_percent"
    ] = (
        100.0
        * by_self_df[
            "self_cpu_time_us"
        ]
        / total_self_cpu_us
    )

    by_self_df[
        "self_rank"
    ] = range(
        1,
        len(
            by_self_df
        )
        + 1,
    )

    by_total_df = (
        operator_df.sort_values(
            [
                "cpu_time_total_us",
                "operator",
            ],
            ascending=[
                False,
                True,
            ],
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    total_cpu_us = float(
        by_total_df[
            "cpu_time_total_us"
        ].sum()
    )

    by_total_df[
        "cpu_total_share_percent"
    ] = (
        100.0
        * by_total_df[
            "cpu_time_total_us"
        ]
        / total_cpu_us
    )

    by_total_df[
        "total_rank"
    ] = range(
        1,
        len(
            by_total_df
        )
        + 1,
    )

    display_columns = [
        "operator",
        "count",
        "self_cpu_time_s",
        "cpu_time_total_s",
        "self_cpu_share_percent",
    ]

    print(
        by_self_df[
            display_columns
        ]
        .head(
            30
        )
        .to_string(
            index=False,
            formatters={
                "self_cpu_time_s": (
                    lambda value: (
                        f"{value:.6f}"
                    )
                ),
                "cpu_time_total_s": (
                    lambda value: (
                        f"{value:.6f}"
                    )
                ),
                "self_cpu_share_percent": (
                    lambda value: (
                        f"{value:.2f}"
                    )
                ),
            },
        )
    )

    top_operator = str(
        by_self_df.iloc[
            0
        ][
            "operator"
        ]
    )

    top_operator_self_seconds = float(
        by_self_df.iloc[
            0
        ][
            "self_cpu_time_s"
        ]
    )

    top_operator_share = float(
        by_self_df.iloc[
            0
        ][
            "self_cpu_share_percent"
        ]
    )

    embedding_rows = by_self_df.loc[
        by_self_df[
            "operator"
        ].str.contains(
            "embedding",
            case=False,
            regex=False,
        )
    ].copy()

    index_rows = by_self_df.loc[
        by_self_df[
            "operator"
        ].str.contains(
            "index",
            case=False,
            regex=False,
        )
    ].copy()

    scatter_rows = by_self_df.loc[
        by_self_df[
            "operator"
        ].str.contains(
            "scatter",
            case=False,
            regex=False,
        )
    ].copy()

    embedding_self_seconds = float(
        embedding_rows[
            "self_cpu_time_s"
        ].sum()
    )

    embedding_share = float(
        embedding_rows[
            "self_cpu_share_percent"
        ].sum()
    )

    index_scatter_self_seconds = float(
        pd.concat(
            [
                index_rows,
                scatter_rows,
            ],
            ignore_index=True,
        )
        .drop_duplicates(
            subset=[
                "operator",
            ]
        )[
            "self_cpu_time_s"
        ]
        .sum()
    )

    print()
    print(
        f"Top operator:                          "
        f"{top_operator}"
    )
    print(
        f"Top operator self CPU:                 "
        f"{top_operator_self_seconds:.3f} s"
    )
    print(
        f"Top operator self CPU share:           "
        f"{top_operator_share:.2f}%"
    )
    print()
    print(
        f"All embedding-named ops self CPU:      "
        f"{embedding_self_seconds:.3f} s"
    )
    print(
        f"All embedding-named ops share:         "
        f"{embedding_share:.2f}%"
    )
    print(
        f"Index/scatter named ops self CPU:      "
        f"{index_scatter_self_seconds:.3f} s"
    )

    top_summary_df = pd.DataFrame(
        [
            {
                "top_operator": (
                    top_operator
                ),
                "top_operator_self_seconds": (
                    top_operator_self_seconds
                ),
                "top_operator_self_share_percent": (
                    top_operator_share
                ),
                "embedding_named_self_seconds": (
                    embedding_self_seconds
                ),
                "embedding_named_self_share_percent": (
                    embedding_share
                ),
                "index_scatter_named_self_seconds": (
                    index_scatter_self_seconds
                ),
                "profile_wall_seconds": (
                    wall_seconds
                ),
                "batch0_exact_frozen": (
                    True
                ),
            }
        ]
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.4.4 INVARIANTS"
    )

    checks = [
        (
            "production_training_not_launched",
            True,
        ),
        (
            "validation_cases_scored_zero",
            True,
        ),
        (
            "test_cases_scored_zero",
            True,
        ),
        (
            "batch0_exact_frozen",
            exact,
        ),
        (
            "operator_profile_nonempty",
            (
                len(
                    operator_df
                )
                > 0
            ),
        ),
        (
            "top_operator_identified",
            (
                len(
                    top_operator
                )
                > 0
            ),
        ),
        (
            "top_operator_positive_cpu_time",
            (
                math.isfinite(
                    top_operator_self_seconds
                )
                and top_operator_self_seconds > 0
            ),
        ),
    ]

    invariant_df = pd.DataFrame(
        [
            {
                "check": (
                    name
                ),
                "result": (
                    "PASS"
                    if passed
                    else "FAIL"
                ),
            }
            for (
                name,
                passed,
            ) in checks
        ]
    )

    require(
        bool(
            (
                invariant_df[
                    "result"
                ]
                == "PASS"
            ).all()
        ),
        (
            "At least one Phase-5.4.4 invariant failed."
        ),
    )

    print(
        invariant_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Write outputs
    # =========================================================================

    banner(
        "WRITE PHASE-5.4.4 OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    by_self_df.to_csv(
        OPERATOR_PROFILE_PATH,
        index=False,
    )

    by_total_df.to_csv(
        OPERATOR_TOTAL_PATH,
        index=False,
    )

    top_summary_df.to_csv(
        TOP_OPERATOR_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    contract = {
        "phase": (
            "5.4.4"
        ),
        "title": (
            "Exact Autograd Operator Bottleneck Profile"
        ),
        "status": (
            "COMPLETE"
        ),
        "classification": (
            "RUNTIME_PROFILING_ONLY"
        ),
        "selected_exact_cpu_threads": (
            EXPECTED_SELECTED_THREADS
        ),
        "profiled_batches": (
            1
        ),
        "batch0_exact_frozen": (
            True
        ),
        "profile_wall_seconds": (
            wall_seconds
        ),
        "top_operator": (
            top_operator
        ),
        "top_operator_self_seconds": (
            top_operator_self_seconds
        ),
        "top_operator_self_share_percent": (
            top_operator_share
        ),
        "embedding_named_self_seconds": (
            embedding_self_seconds
        ),
        "embedding_named_self_share_percent": (
            embedding_share
        ),
        "index_scatter_named_self_seconds": (
            index_scatter_self_seconds
        ),
        "production_training_launched": (
            False
        ),
        "production_checkpoint_written": (
            False
        ),
        "validation_cases_scored": (
            0
        ),
        "test_cases_scored": (
            0
        ),
        "next_phase": (
            "5.4.5_TARGETED_IMPLEMENTATION_EQUIVALENT_ACCELERATION"
        ),
    }

    CONTRACT_PATH.write_text(
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    manifest = {
        "phase": (
            "5.4.4"
        ),
        "status": (
            "EXACT_AUTOGRAD_OPERATOR_PROFILE_COMPLETE"
        ),
        "top_operator": (
            top_operator
        ),
        "top_operator_self_share_percent": (
            top_operator_share
        ),
        "embedding_named_self_share_percent": (
            embedding_share
        ),
        "production_training_steps": (
            0
        ),
        "validation_cases_scored": (
            0
        ),
        "test_cases_scored": (
            0
        ),
        "contract": str(
            CONTRACT_PATH
        ),
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    for path in (
        OPERATOR_PROFILE_PATH,
        OPERATOR_TOTAL_PATH,
        TOP_OPERATOR_PATH,
        FINAL_INVARIANT_PATH,
        CONTRACT_PATH,
        MANIFEST_PATH,
    ):
        print(
            f"WROTE  {path}"
        )

    # Restore helpers.
    roundtrip.gradient_logical_sha256 = (
        true_gradient_hash_fn
    )

    roundtrip.optimizer_state_logical_sha256 = (
        true_optimizer_hash_fn
    )

    del model
    del optimizer
    del batch0
    del stream
    del shared

    gc.collect()

    banner(
        "PHASE 5.4.4 FINAL STATUS"
    )

    print(
        f"Top self-CPU operator:                "
        f"{top_operator}"
    )
    print(
        f"Top operator self share:              "
        f"{top_operator_share:.2f}%"
    )
    print(
        f"Embedding-named self share:           "
        f"{embedding_share:.2f}%"
    )
    print()
    print(
        "Exact batch-0 numerical trajectory:   PRESERVED"
    )
    print(
        "Production training launched:         NO"
    )
    print(
        "Production checkpoint written:        NO"
    )
    print(
        "Validation cases scored:              0"
    )
    print(
        "Test cases scored:                    0"
    )

    banner(
        "PHASE 5.4.4 COMPLETE / "
        "LOW-LEVEL AUTOGRAD BOTTLENECK IDENTIFIED"
    )


if __name__ == "__main__":
    main()
