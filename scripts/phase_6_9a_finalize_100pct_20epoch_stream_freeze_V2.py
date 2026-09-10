#!/usr/bin/env python3
"""
Phase 6.9a — Finalize Existing 100% 20-Epoch Production Stream Freeze V2

Purpose
-------
Recover cleanly from the Phase-6.9a V1 post-generation bookkeeping failure.

V1 successfully:
    - copied the canonical 1,073,249-positive training order
    - generated all 20 full-data negative matrices
    - generated all 20 full-data example orders
    - persisted all 40 binary stream files
    - reloaded and verified every epoch immediately after persistence
    - reproduced the frozen Phase-5 anchors for epochs 0, 1 and 19
    - reproduced the independent Phase-6 epoch-2 benchmark anchor

V1 then failed only when it compared a DataFrame-derived CSV fingerprint before
and after pandas CSV round-trip parsing. The persisted CSV itself is intact.

This V2 finalizer DOES NOT regenerate or rewrite any training stream.
It independently verifies the already-persisted artifacts, binds the persisted
registry by its actual byte SHA256, and writes the final manifest + V2 contract.

Boundary
--------
No neural model is instantiated.
No Adam optimizer is instantiated.
No forward/backward occurs.
No optimizer.step() occurs.
No validation is executed.
No test case is accessed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


NUM_EPOCHS = 20
POSITIVE_EVENTS = 1_073_249
NEGATIVES_PER_POSITIVE = 4
NEGATIVES_PER_EPOCH = 4_292_996
EXAMPLES_PER_EPOCH = 5_366_245
BATCH_SIZE = 512
BATCHES_PER_EPOCH = 10_481
FINAL_BATCH_SIZE = 485

EXPECTED_REPOSITORY_HEAD = (
    "6c94a4e787d2bc7a27e9c1ebced3ddf41132d915"
)

EXPECTED_POSITIVE_SHA = (
    "73b074a80675793b811fbdc8a0609883"
    "c857fb2a687a2e01c31865ade5b509d1"
)

EXPECTED_REGISTRY_FILE_SHA256 = (
    "e6970082a06b3b097e0858e4d0761240"
    "c99c3a255b3227339cea6243821e2eb1"
)

EXPECTED_ANCHORS = {
    0: {
        "negative_sha256": (
            "47015b147b1949562c0f6737a6f3a3f2"
            "d7cabd2d2202e4e57456d884a1e23fe6"
        ),
        "order_sha256": (
            "0156be3ee623ade1ae696557337bfb324"
            "e9011adb7df8be9648ecb0a426c134e"
        ),
        "status": "PHASE5_EXACT_FROZEN_REGRESSION",
    },
    1: {
        "negative_sha256": (
            "f7b415e0f305e049cc94c7e4261b6838"
            "00085f8fa7f7b2df11ff3658dea9d850"
        ),
        "order_sha256": (
            "2da43d28e540ed48cb557ca889190b5df"
            "ebc7c1207cbd3882df2fa14ca2a28d8"
        ),
        "status": "PHASE5_DOUBLE_REGENERATION_EXACT",
    },
    2: {
        "negative_sha256": (
            "66e2a71c7df08f03d221002541c07611"
            "4a8e6820eae6b42e38988910912e6710"
        ),
        "order_sha256": (
            "9d5a5e91ad90160ce79b88a0995a8ef7"
            "6ac9cfcd6de00b4cbbfac96fde183474"
        ),
        "status": "PHASE6_PREGENERATION_BENCHMARK_REGRESSION",
    },
    19: {
        "negative_sha256": (
            "06f9a11d8986ba9b7e0242fc41423478"
            "9ccc7adba91cab6da68ebc21e44682b7"
        ),
        "order_sha256": (
            "c6236cb081ddba7f72eb0d36199500ac6"
            "b7695646e21a824fb84b85fae769329"
        ),
        "status": "PHASE5_DOUBLE_REGENERATION_EXACT",
    },
}

ROUNDTRIP_PATH = Path(
    "scripts/phase_5_3_2b_checkpoint_resume_roundtrip_proof.py"
)

GENERATOR_PATH = Path(
    "scripts/phase_5_3_5_generalized_20_epoch_training_stream_generator_proof_V6.py"
)

V1_SCRIPT_PATH = Path(
    "scripts/phase_6_9a_freeze_100pct_20epoch_streams_V1.py"
)

PHASE5_GENERATOR_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_5_generalized_training_stream_generator_contract.json"
)

PHASE5_LAUNCH_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_6_production_training_launch_contract.json"
)

V1_LOG_PATH = Path(
    "data/experimental/phase_6/"
    "phase_6_9a_freeze_100pct_20epoch_streams_V1.log"
)

OUTPUT_ROOT = Path(
    "data/experimental/phase_6/full_training/100pct"
)

POSITIVE_PATH = (
    OUTPUT_ROOT / "canonical_training_positive_event_order.parquet"
)

STREAM_DIR = (
    OUTPUT_ROOT / "epoch_streams"
)

REGISTRY_PATH = (
    STREAM_DIR / "full_epoch_stream_registry.csv"
)

MANIFEST_PATH = (
    STREAM_DIR / "full_epoch_stream_manifest_V2.json"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_9a_100pct_20epoch_stream_freeze_V2.json"
)


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def banner(text):
    print()
    print("=" * 118)
    print(text)
    print("=" * 118)


def git_head():
    return (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        )
        .strip()
    )


def load_module(path, name):
    require(path.exists(), f"Missing source: {path}")

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    require(
        spec is not None and spec.loader is not None,
        f"Could not import: {path}",
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def load_json(path):
    require(path.exists(), f"Missing JSON: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def atomic_json_write(payload, path):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = Path(str(path) + ".tmp")

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary,
        path,
    )


def epoch_negative_path(epoch):
    return (
        STREAM_DIR
        / f"epoch_{epoch:02d}_negative_startup_local.npy"
    )


def epoch_order_path(epoch):
    return (
        STREAM_DIR
        / f"epoch_{epoch:02d}_example_order.npy"
    )


def current_script_path():
    return Path(__file__).resolve()


def verify_existing_streams():
    repository_commit = git_head()

    require(
        repository_commit == EXPECTED_REPOSITORY_HEAD,
        (
            "Repository HEAD drift. Expected "
            f"{EXPECTED_REPOSITORY_HEAD}, got "
            f"{repository_commit}."
        ),
    )

    for path in (
        ROUNDTRIP_PATH,
        GENERATOR_PATH,
        V1_SCRIPT_PATH,
        PHASE5_GENERATOR_CONTRACT_PATH,
        PHASE5_LAUNCH_CONTRACT_PATH,
        POSITIVE_PATH,
        REGISTRY_PATH,
    ):
        require(
            path.exists(),
            f"Missing prerequisite/artifact: {path}",
        )

    generator_contract = load_json(
        PHASE5_GENERATOR_CONTRACT_PATH
    )

    launch_contract = load_json(
        PHASE5_LAUNCH_CONTRACT_PATH
    )

    require(
        generator_contract["status"] == "FROZEN",
        "Phase-5.3.5 generator contract is not FROZEN.",
    )

    require(
        launch_contract["status"] == "FROZEN",
        "Phase-5.3.6 launch contract is not FROZEN.",
    )

    require(
        bool(
            launch_contract["test_integrity"][
                "test_forbidden_during_training"
            ]
        ),
        "Frozen launch contract no longer forbids test during training.",
    )

    roundtrip = load_module(
        ROUNDTRIP_PATH,
        "_phase6_9a_v2_roundtrip",
    )

    generator = load_module(
        GENERATOR_PATH,
        "_phase6_9a_v2_generator",
    )

    preflight = roundtrip.load_preflight_runtime()

    banner("VERIFY SHARED CANONICAL POSITIVE STREAM")

    positive_order = pd.read_parquet(POSITIVE_PATH)

    require(
        len(positive_order) == POSITIVE_EVENTS,
        "Positive-event count drift.",
    )

    positive_sha = (
        preflight.positive_stream_logical_sha256(
            positive_order
        )
    )

    require(
        positive_sha == EXPECTED_POSITIVE_SHA,
        "Positive logical SHA drift.",
    )

    print(f"rows:                           {len(positive_order):,}")
    print(f"logical_sha256:                 {positive_sha}")
    print("positive_stream:                PASS")

    banner("VERIFY PERSISTED REGISTRY BY BYTE SHA256")

    registry_file_sha = file_sha256(
        REGISTRY_PATH
    )

    require(
        registry_file_sha == EXPECTED_REGISTRY_FILE_SHA256,
        (
            "Persisted registry byte SHA drift. "
            "This V2 intentionally binds the exact V1 registry file."
        ),
    )

    registry = pd.read_csv(REGISTRY_PATH)

    require(
        len(registry) == NUM_EPOCHS,
        "Registry row-count drift.",
    )

    require(
        len(registry.columns) == 25,
        "Registry column-count drift.",
    )

    require(
        not bool(
            registry.isna().any().any()
        ),
        "Registry contains nulls.",
    )

    require(
        registry["epoch_index"].tolist()
        == list(range(NUM_EPOCHS)),
        "Registry epoch order drift.",
    )

    require(
        (
            registry["positive_order_sha256"]
            == EXPECTED_POSITIVE_SHA
        ).all(),
        "Registry positive SHA column drift.",
    )

    require(
        (
            registry["negative_shape_rows"]
            == POSITIVE_EVENTS
        ).all(),
        "Registry negative row-count drift.",
    )

    require(
        (
            registry["negative_shape_cols"]
            == NEGATIVES_PER_POSITIVE
        ).all(),
        "Registry negative column-count drift.",
    )

    require(
        (
            registry["negative_dtype"]
            == "int32"
        ).all(),
        "Registry negative dtype drift.",
    )

    require(
        (
            registry["negative_dtype_str"]
            == "<i4"
        ).all(),
        "Registry negative dtype.str drift.",
    )

    require(
        (
            registry["order_length"]
            == EXAMPLES_PER_EPOCH
        ).all(),
        "Registry order length drift.",
    )

    require(
        (
            registry["order_dtype"]
            == "int64"
        ).all(),
        "Registry order dtype drift.",
    )

    require(
        (
            registry["generation_status"]
            == "FROZEN_PRETRAINING"
        ).all(),
        "Registry generation-status drift.",
    )

    require(
        (
            registry["repair_positions"]
            == (
                registry["forbidden_candidate_rejections"]
                + registry["duplicate_candidate_rejections"]
            )
        ).all(),
        "Registry rejection-accounting drift.",
    )

    require(
        (
            registry["accepted_future_positive"]
            + registry["accepted_never_positive"]
            == NEGATIVES_PER_EPOCH
        ).all(),
        "Registry accepted-negative accounting drift.",
    )

    print(f"rows:                           {len(registry)}")
    print(f"columns:                        {len(registry.columns)}")
    print(f"file_sha256:                    {registry_file_sha}")
    print("nulls:                          0")
    print("registry_semantics:             PASS")

    banner("VERIFY 20-EPOCH SEED BINDINGS")

    for epoch in range(NUM_EPOCHS):
        row = registry.iloc[epoch]

        expected_negative_seed = (
            generator.derive_epoch_seed(
                generator.NEGATIVE_NAMESPACE,
                epoch,
            )
        )

        expected_order_seed = (
            generator.derive_epoch_seed(
                generator.ORDER_NAMESPACE,
                epoch,
            )
        )

        require(
            int(row["negative_seed"])
            == int(expected_negative_seed),
            f"Epoch {epoch} negative seed drift.",
        )

        require(
            int(row["order_seed"])
            == int(expected_order_seed),
            f"Epoch {epoch} order seed drift.",
        )

    print("20 / 20 epoch seed pairs:        PASS")

    banner("VERIFY 40 PERSISTED BINARY STREAM FILES")

    total_stream_bytes = 0
    verified_rows = []

    for epoch in range(NUM_EPOCHS):
        row = registry.iloc[epoch]

        negative_path = Path(
            row["negative_file"]
        )

        order_path = Path(
            row["order_file"]
        )

        require(
            negative_path == epoch_negative_path(epoch),
            f"Epoch {epoch} negative path drift.",
        )

        require(
            order_path == epoch_order_path(epoch),
            f"Epoch {epoch} order path drift.",
        )

        require(
            negative_path.exists(),
            f"Missing epoch {epoch} negative matrix.",
        )

        require(
            order_path.exists(),
            f"Missing epoch {epoch} example order.",
        )

        negative = np.load(
            negative_path,
            mmap_mode="r",
        )

        order = np.load(
            order_path,
            mmap_mode="r",
        )

        require(
            negative.shape
            == (
                POSITIVE_EVENTS,
                NEGATIVES_PER_POSITIVE,
            ),
            f"Epoch {epoch} negative shape drift.",
        )

        require(
            negative.dtype == np.dtype(np.int32),
            f"Epoch {epoch} negative dtype drift.",
        )

        require(
            negative.dtype.str == "<i4",
            f"Epoch {epoch} negative dtype.str drift.",
        )

        require(
            order.shape == (EXAMPLES_PER_EPOCH,),
            f"Epoch {epoch} order shape drift.",
        )

        require(
            order.dtype == np.dtype(np.int64),
            f"Epoch {epoch} order dtype drift.",
        )

        negative_logical_sha = (
            preflight.array_logical_sha256(
                np.asarray(negative)
            )
        )

        order_logical_sha = (
            preflight.array_logical_sha256(
                np.asarray(order)
            )
        )

        require(
            negative_logical_sha
            == str(row["negative_sha256"]),
            f"Epoch {epoch} negative logical SHA drift.",
        )

        require(
            order_logical_sha
            == str(row["order_sha256"]),
            f"Epoch {epoch} order logical SHA drift.",
        )

        negative_file_sha = file_sha256(
            negative_path
        )

        order_file_sha = file_sha256(
            order_path
        )

        require(
            negative_file_sha
            == str(row["negative_file_sha256"]),
            f"Epoch {epoch} negative file SHA drift.",
        )

        require(
            order_file_sha
            == str(row["order_file_sha256"]),
            f"Epoch {epoch} order file SHA drift.",
        )

        anchor = EXPECTED_ANCHORS.get(epoch)

        if anchor is not None:
            require(
                negative_logical_sha
                == anchor["negative_sha256"],
                f"Epoch {epoch} negative anchor regression failed.",
            )

            require(
                order_logical_sha
                == anchor["order_sha256"],
                f"Epoch {epoch} order anchor regression failed.",
            )

            require(
                str(row["anchor_status"])
                == anchor["status"],
                f"Epoch {epoch} anchor-status drift.",
            )

        total_stream_bytes += negative_path.stat().st_size
        total_stream_bytes += order_path.stat().st_size

        verified_rows.append(
            {
                "epoch_index": epoch,
                "negative_logical_sha256": negative_logical_sha,
                "order_logical_sha256": order_logical_sha,
                "negative_file_sha256": negative_file_sha,
                "order_file_sha256": order_file_sha,
                "negative_bytes": int(
                    negative_path.stat().st_size
                ),
                "order_bytes": int(
                    order_path.stat().st_size
                ),
                "anchor_status": str(
                    row["anchor_status"]
                ),
            }
        )

        print(
            f"epoch={epoch:02d} "
            f"negative={negative_logical_sha[:12]}... "
            f"order={order_logical_sha[:12]}... "
            "PASS"
        )

    require(
        len(verified_rows) == NUM_EPOCHS,
        "Verified-row count drift.",
    )

    require(
        len(
            {
                row["negative_logical_sha256"]
                for row in verified_rows
            }
        )
        == NUM_EPOCHS,
        "Expected 20 distinct negative logical hashes.",
    )

    require(
        len(
            {
                row["order_logical_sha256"]
                for row in verified_rows
            }
        )
        == NUM_EPOCHS,
        "Expected 20 distinct order logical hashes.",
    )

    print()
    print("epochs_verified:                 20 / 20")
    print("binary_files_verified:           40 / 40")
    print(f"total_stream_bytes:              {total_stream_bytes:,}")
    print(
        f"total_stream_GiB:                "
        f"{total_stream_bytes / (1024 ** 3):.3f}"
    )

    verification = {
        "repository_commit": repository_commit,
        "positive_logical_sha256": positive_sha,
        "positive_file_sha256": file_sha256(
            POSITIVE_PATH
        ),
        "registry_file_sha256": registry_file_sha,
        "registry_rows": NUM_EPOCHS,
        "registry_columns": int(
            len(registry.columns)
        ),
        "total_stream_bytes": int(
            total_stream_bytes
        ),
        "verified_epochs": verified_rows,
    }

    return verification, registry


def finalize():
    require(
        not MANIFEST_PATH.exists(),
        (
            "V2 manifest already exists. "
            "Use --verify-only instead of overwriting."
        ),
    )

    require(
        not CONTRACT_PATH.exists(),
        (
            "V2 contract already exists. "
            "Use --verify-only instead of overwriting."
        ),
    )

    verification, registry = (
        verify_existing_streams()
    )

    banner("WRITE V2 MANIFEST + CONTRACT")

    file_entries = [
        {
            "role": "CANONICAL_POSITIVE_STREAM",
            "path": str(POSITIVE_PATH),
            "bytes": int(
                POSITIVE_PATH.stat().st_size
            ),
            "file_sha256": file_sha256(
                POSITIVE_PATH
            ),
        },
        {
            "role": "PERSISTED_STREAM_REGISTRY",
            "path": str(REGISTRY_PATH),
            "bytes": int(
                REGISTRY_PATH.stat().st_size
            ),
            "file_sha256": verification[
                "registry_file_sha256"
            ],
        },
    ]

    for epoch in range(NUM_EPOCHS):
        negative_path = epoch_negative_path(
            epoch
        )

        order_path = epoch_order_path(
            epoch
        )

        row = registry.iloc[epoch]

        file_entries.append(
            {
                "role": (
                    f"EPOCH_{epoch:02d}_NEGATIVE_MATRIX"
                ),
                "path": str(negative_path),
                "bytes": int(
                    negative_path.stat().st_size
                ),
                "file_sha256": str(
                    row["negative_file_sha256"]
                ),
                "logical_sha256": str(
                    row["negative_sha256"]
                ),
            }
        )

        file_entries.append(
            {
                "role": (
                    f"EPOCH_{epoch:02d}_EXAMPLE_ORDER"
                ),
                "path": str(order_path),
                "bytes": int(
                    order_path.stat().st_size
                ),
                "file_sha256": str(
                    row["order_file_sha256"]
                ),
                "logical_sha256": str(
                    row["order_sha256"]
                ),
            }
        )

    manifest = {
        "phase": "6.9a",
        "script_version": "V2",
        "status": "COMPLETE",
        "recovery_role": (
            "finalize already-generated V1 streams "
            "without regeneration"
        ),
        "repository_commit": verification[
            "repository_commit"
        ],
        "file_count": len(file_entries),
        "files": file_entries,
        "boundary": {
            "stream_files_regenerated_by_v2": 0,
            "model_instantiated": False,
            "optimizer_instantiated": False,
            "forward_backward": False,
            "optimizer_steps": 0,
            "validation_executed": False,
            "validation_cases_scored": 0,
            "test_accessed": False,
            "test_cases_scored": 0,
        },
    }

    atomic_json_write(
        manifest,
        MANIFEST_PATH,
    )

    contract = {
        "phase": "6.9a",
        "title": (
            "100% 20-Epoch Production Training Stream Freeze"
        ),
        "script_version": "V2",
        "status": "FROZEN_FOR_100PCT_PRODUCTION",
        "scientific_role": (
            "pre-training materialization/freeze of the "
            "already-frozen Phase-5.3.5 generalized "
            "epoch-indexed training stream"
        ),
        "recovery_from_v1": {
            "v1_generation_completed": True,
            "v1_binary_stream_files_written": 40,
            "v1_binary_stream_files_verified_after_write": 40,
            "v1_failure_stage": (
                "post-generation registry bookkeeping"
            ),
            "v1_failure_reason": (
                "fragile DataFrame -> CSV -> read_csv -> CSV "
                "fingerprint comparison changed float textual "
                "representation after pandas round-trip"
            ),
            "v2_stream_regeneration": False,
            "v2_registry_policy": (
                "bind exact persisted registry bytes by SHA256"
            ),
            "expected_persisted_registry_file_sha256": (
                EXPECTED_REGISTRY_FILE_SHA256
            ),
        },
        "repository_commit": verification[
            "repository_commit"
        ],
        "sources": {
            "roundtrip_runtime": str(ROUNDTRIP_PATH),
            "roundtrip_runtime_file_sha256": (
                file_sha256(ROUNDTRIP_PATH)
            ),
            "generalized_generator_v6": str(
                GENERATOR_PATH
            ),
            "generalized_generator_v6_file_sha256": (
                file_sha256(GENERATOR_PATH)
            ),
            "v1_generation_script": str(
                V1_SCRIPT_PATH
            ),
            "v1_generation_script_file_sha256": (
                file_sha256(V1_SCRIPT_PATH)
            ),
            "v1_log": (
                str(V1_LOG_PATH)
                if V1_LOG_PATH.exists()
                else None
            ),
            "v1_log_file_sha256": (
                file_sha256(V1_LOG_PATH)
                if V1_LOG_PATH.exists()
                else None
            ),
            "v2_finalizer_script": str(
                current_script_path()
            ),
            "v2_finalizer_script_file_sha256": (
                file_sha256(
                    current_script_path()
                )
            ),
            "phase5_generator_contract": str(
                PHASE5_GENERATOR_CONTRACT_PATH
            ),
            "phase5_generator_contract_file_sha256": (
                file_sha256(
                    PHASE5_GENERATOR_CONTRACT_PATH
                )
            ),
            "phase5_launch_contract": str(
                PHASE5_LAUNCH_CONTRACT_PATH
            ),
            "phase5_launch_contract_file_sha256": (
                file_sha256(
                    PHASE5_LAUNCH_CONTRACT_PATH
                )
            ),
        },
        "positive_stream": {
            "path": str(POSITIVE_PATH),
            "rows": POSITIVE_EVENTS,
            "logical_sha256": verification[
                "positive_logical_sha256"
            ],
            "file_sha256": verification[
                "positive_file_sha256"
            ],
            "shared_across_all_epochs": True,
        },
        "registry": {
            "path": str(REGISTRY_PATH),
            "rows": NUM_EPOCHS,
            "columns": int(
                verification["registry_columns"]
            ),
            "file_sha256": verification[
                "registry_file_sha256"
            ],
            "binding_policy": (
                "exact persisted file bytes"
            ),
        },
        "training_stream": {
            "epochs": NUM_EPOCHS,
            "positive_events_per_epoch": (
                POSITIVE_EVENTS
            ),
            "negatives_per_positive": (
                NEGATIVES_PER_POSITIVE
            ),
            "negative_examples_per_epoch": (
                NEGATIVES_PER_EPOCH
            ),
            "examples_per_epoch": (
                EXAMPLES_PER_EPOCH
            ),
            "batch_size": BATCH_SIZE,
            "batches_per_epoch": (
                BATCHES_PER_EPOCH
            ),
            "final_batch_size": FINAL_BATCH_SIZE,
            "negative_dtype": "int32",
            "negative_dtype_str": "<i4",
            "order_dtype": "int64",
            "binary_files": 40,
            "total_binary_stream_bytes": (
                verification[
                    "total_stream_bytes"
                ]
            ),
            "total_binary_stream_GiB": (
                verification[
                    "total_stream_bytes"
                ]
                / (1024 ** 3)
            ),
            "distinct_negative_logical_hashes": int(
                registry[
                    "negative_sha256"
                ].nunique()
            ),
            "distinct_order_logical_hashes": int(
                registry[
                    "order_sha256"
                ].nunique()
            ),
        },
        "regression_anchors": {
            str(epoch): payload
            for epoch, payload
            in EXPECTED_ANCHORS.items()
        },
        "verification": {
            "positive_stream": "PASS",
            "registry_byte_binding": "PASS",
            "registry_semantics": "PASS",
            "epoch_seed_bindings": "20/20 PASS",
            "binary_stream_files": "40/40 PASS",
            "epoch_logical_hashes": "20/20 PASS",
            "epoch_file_hashes": "40/40 PASS",
            "anchor_epochs_0_1_2_19": "PASS",
        },
        "boundary": {
            "model_instantiated": False,
            "optimizer_instantiated": False,
            "forward_backward": False,
            "optimizer_steps": 0,
            "validation_executed": False,
            "validation_cases_scored": 0,
            "test_accessed": False,
            "test_cases_scored": 0,
        },
        "artifacts": {
            "positive_stream": str(POSITIVE_PATH),
            "epoch_registry": str(REGISTRY_PATH),
            "manifest": str(MANIFEST_PATH),
            "contract": str(CONTRACT_PATH),
        },
        "next_phase": {
            "id": "6.9b",
            "title": (
                "100% 20-Epoch CUDA Production Trainer Assembly"
            ),
            "training_allowed_only_after": (
                "Phase-6.9a V2 finalizer and verify-only PASS"
            ),
        },
    }

    atomic_json_write(
        contract,
        CONTRACT_PATH,
    )

    require(
        MANIFEST_PATH.exists(),
        "V2 manifest write failed.",
    )

    require(
        CONTRACT_PATH.exists(),
        "V2 contract write failed.",
    )

    print(f"WROTE  {MANIFEST_PATH}")
    print(f"WROTE  {CONTRACT_PATH}")

    banner("PHASE 6.9a V2 FINAL STATUS")

    print("Existing V1 stream regeneration:       NO")
    print("Positive stream verified:              PASS")
    print("Persisted registry byte SHA:           PASS")
    print("Epoch seed bindings:                   20 / 20 PASS")
    print("Binary stream files:                   40 / 40 PASS")
    print("Epoch logical fingerprints:            20 / 20 PASS")
    print("Regression anchors 0/1/2/19:           PASS")
    print("Optimizer steps executed:              0")
    print("Validation cases scored:               0")
    print("Test cases accessed/scored:            0")
    print()
    print(
        "PHASE 6.9a V2: COMPLETE / "
        "100% 20-EPOCH STREAMS FROZEN FOR PRODUCTION"
    )


def verify_only():
    require(
        MANIFEST_PATH.exists(),
        (
            "Cannot --verify-only: "
            "V2 manifest missing."
        ),
    )

    require(
        CONTRACT_PATH.exists(),
        (
            "Cannot --verify-only: "
            "V2 contract missing."
        ),
    )

    verification, _ = verify_existing_streams()

    manifest = load_json(MANIFEST_PATH)
    contract = load_json(CONTRACT_PATH)

    require(
        manifest["status"] == "COMPLETE",
        "V2 manifest status drift.",
    )

    require(
        manifest["file_count"] == 42,
        "V2 manifest file-count drift.",
    )

    require(
        contract["status"]
        == "FROZEN_FOR_100PCT_PRODUCTION",
        "V2 contract status drift.",
    )

    require(
        contract["registry"]["file_sha256"]
        == verification[
            "registry_file_sha256"
        ],
        "V2 contract registry binding drift.",
    )

    require(
        contract["positive_stream"][
            "logical_sha256"
        ]
        == EXPECTED_POSITIVE_SHA,
        "V2 contract positive binding drift.",
    )

    require(
        contract["boundary"]["optimizer_steps"] == 0,
        "V2 contract optimizer boundary drift.",
    )

    require(
        not bool(
            contract["boundary"]["test_accessed"]
        ),
        "V2 contract test boundary drift.",
    )

    banner("PHASE 6.9a V2 VERIFY-ONLY RESULT")

    print("Manifest:                        PASS")
    print("Contract:                        PASS")
    print("Persisted registry:              PASS")
    print("Positive stream:                 PASS")
    print("20 epoch streams:                PASS")
    print("40 binary files:                 PASS")
    print("Optimizer steps executed:        0")
    print("Validation cases scored:         0")
    print("Test cases accessed/scored:      0")
    print()
    print("PHASE 6.9a V2 VERIFY-ONLY: PASS")


def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--finalize",
        action="store_true",
        help=(
            "Verify the already-generated V1 full streams "
            "and write the V2 manifest + contract. "
            "Does not regenerate any stream."
        ),
    )

    mode.add_argument(
        "--verify-only",
        action="store_true",
        help=(
            "Re-verify the completed V2 freeze without "
            "rewriting any artifact."
        ),
    )

    args = parser.parse_args()

    banner(
        "PHASE 6.9a — "
        "FINALIZE EXISTING 100% 20-EPOCH STREAM FREEZE V2"
    )

    print(
        f"Positive events / epoch:               {POSITIVE_EVENTS:,}"
    )
    print(
        f"Examples / epoch:                      {EXAMPLES_PER_EPOCH:,}"
    )
    print(
        f"Batches / epoch:                       {BATCHES_PER_EPOCH:,}"
    )
    print(
        f"Epochs:                                {NUM_EPOCHS}"
    )
    print(
        "Existing streams regenerated:          NO"
    )
    print(
        "Model instantiated:                    NO"
    )
    print(
        "Adam instantiated:                     NO"
    )
    print(
        "Optimizer steps:                       0"
    )
    print(
        "Validation:                            DISABLED"
    )
    print(
        "Test:                                  DISABLED"
    )

    if args.finalize:
        finalize()
        return

    verify_only()


if __name__ == "__main__":
    main()