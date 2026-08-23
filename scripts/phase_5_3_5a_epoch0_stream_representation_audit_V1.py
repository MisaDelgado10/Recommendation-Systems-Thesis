#!/usr/bin/env python3
"""
Phase 5.3.5a — Frozen Epoch-0 Stream Representation Audit

Purpose
-------
Diagnose the remaining Phase-5.3.5 epoch-0 logical-hash mismatch WITHOUT
regenerating 1,073,249 positive rows and WITHOUT changing any frozen scientific
semantics.

Observed before this audit
--------------------------
V4 established:
    - first 10,000 regenerated negative rows match the frozen matrix
    - the full sequential sampler completes without any row-level mismatch
    - rejection/acceptance diagnostics exactly reproduce Phase-5.3.1l.1:
          rejected RNG draws          1,759
          forbidden rejections        1,743
          duplicate rejections           16
          future-positive accepted    1,264
          never-positive accepted 4,291,732
    - nevertheless, generated logical SHA256 differs

Therefore the immediate hypothesis is REPRESENTATION mismatch rather than
VALUE mismatch, especially dtype.

This audit reads the authoritative frozen epoch-0 artifacts and computes the
same frozen logical SHA under several explicit dtype representations.

No negative sampling occurs.
No model/Adam/forward/backward occurs.
No validation/test occurs.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROUNDTRIP_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_2b_checkpoint_resume_roundtrip_proof.py"
)

EXPECTED_FROZEN_NEGATIVE_SHA256 = (
    "47015b147b1949562c0f6737a6f3a3f2"
    "d7cabd2d2202e4e57456d884a1e23fe6"
)

EXPECTED_V4_GENERATED_INT64_SHA256 = (
    "de40bb466ff979a382b23f13d6cff404"
    "57320dad833cbcd4c1d851c3cbe21d2d"
)

EXPECTED_FROZEN_ORDER_SHA256 = (
    "0156be3ee623ade1ae696557337bfb324"
    "e9011adb7df8be9648ecb0a426c134e"
)

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_5a"
)

AUDIT_CSV_PATH = (
    AUDIT_DIR
    / "epoch0_stream_representation_audit.csv"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_5a_epoch0_stream_representation_audit.json"
)


def banner(text: str) -> None:
    print("\n" + "=" * 118)
    print(text)
    print("=" * 118)


def require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def is_main_guard(node: ast.AST) -> bool:
    if not isinstance(node, ast.If):
        return False

    test = node.test

    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def load_guarded_module(path: Path, module_name: str):
    require(path.exists(), f"Missing source: {path}")

    tree = ast.parse(
        path.read_text(encoding="utf-8"),
        filename=str(path),
    )

    guards = [
        node
        for node in tree.body
        if is_main_guard(node)
    ]

    require(
        len(guards) == 1,
        f"Expected one __main__ guard in {path}.",
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

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def dtype_record(
    *,
    artifact: str,
    representation: str,
    array: np.ndarray,
    hash_fn,
) -> dict:
    value = np.asarray(array)

    return {
        "artifact": artifact,
        "representation": representation,
        "dtype": str(value.dtype),
        "dtype_str": value.dtype.str,
        "byteorder": value.dtype.byteorder,
        "itemsize": int(value.dtype.itemsize),
        "shape": str(tuple(value.shape)),
        "c_contiguous": bool(value.flags.c_contiguous),
        "f_contiguous": bool(value.flags.f_contiguous),
        "logical_sha256": hash_fn(value),
    }


def main() -> None:
    banner(
        "PHASE 5.3.5a — "
        "FROZEN EPOCH-0 STREAM REPRESENTATION AUDIT"
    )

    print("Negative sampling executed:            NO")
    print("Neural model instantiated:            NO")
    print("Optimizer instantiated:               NO")
    print("Validation/test accessed:             NO")

    banner("LOAD AUTHORITATIVE FROZEN EPOCH-0 STREAM")

    runtime_2b = load_guarded_module(
        ROUNDTRIP_SOURCE_PATH,
        "_itrs_phase5_3_5a_runtime2b",
    )

    preflight = runtime_2b.load_preflight_runtime()
    stream = runtime_2b.load_epoch0_stream(preflight)

    frozen_negative = np.asarray(
        stream["negative_matrix"]
    )

    frozen_order = np.asarray(
        stream["epoch_order"]
    )

    hash_fn = preflight.array_logical_sha256

    frozen_negative_sha = hash_fn(
        frozen_negative
    )

    frozen_order_sha = hash_fn(
        frozen_order
    )

    require(
        frozen_negative_sha
        == EXPECTED_FROZEN_NEGATIVE_SHA256,
        "Authoritative frozen negative hash drift.",
    )

    require(
        frozen_order_sha
        == EXPECTED_FROZEN_ORDER_SHA256,
        "Authoritative frozen order hash drift.",
    )

    print(
        f"Frozen negative shape:                "
        f"{tuple(frozen_negative.shape)}"
    )
    print(
        f"Frozen negative dtype:                "
        f"{frozen_negative.dtype}"
    )
    print(
        f"Frozen negative dtype.str:            "
        f"{frozen_negative.dtype.str}"
    )
    print(
        f"Frozen negative itemsize:             "
        f"{frozen_negative.dtype.itemsize}"
    )
    print(
        f"Frozen negative C contiguous:         "
        f"{frozen_negative.flags.c_contiguous}"
    )
    print(
        f"Frozen negative F contiguous:         "
        f"{frozen_negative.flags.f_contiguous}"
    )
    print()
    print("Frozen negative logical SHA256:")
    print(frozen_negative_sha)

    banner(
        "REPRESENTATION-HASH PROBES ON IDENTICAL FROZEN VALUES"
    )

    rows = []

    rows.append(
        dtype_record(
            artifact="negative_matrix",
            representation="native_frozen",
            array=frozen_negative,
            hash_fn=hash_fn,
        )
    )

    candidate_dtypes = [
        np.int32,
        np.int64,
        np.uint32,
        np.uint64,
    ]

    for dtype in candidate_dtypes:
        casted = np.ascontiguousarray(
            frozen_negative.astype(
                dtype,
                copy=False,
            )
        )

        rows.append(
            dtype_record(
                artifact="negative_matrix",
                representation=(
                    f"cast_{np.dtype(dtype).name}"
                ),
                array=casted,
                hash_fn=hash_fn,
            )
        )

    rows.append(
        dtype_record(
            artifact="epoch_order",
            representation="native_frozen",
            array=frozen_order,
            hash_fn=hash_fn,
        )
    )

    audit_df = pd.DataFrame(rows)

    audit_df[
        "matches_frozen_negative_hash"
    ] = (
        audit_df[
            "logical_sha256"
        ]
        == EXPECTED_FROZEN_NEGATIVE_SHA256
    )

    audit_df[
        "matches_v4_generated_hash"
    ] = (
        audit_df[
            "logical_sha256"
        ]
        == EXPECTED_V4_GENERATED_INT64_SHA256
    )

    audit_df[
        "matches_frozen_order_hash"
    ] = (
        audit_df[
            "logical_sha256"
        ]
        == EXPECTED_FROZEN_ORDER_SHA256
    )

    print(
        audit_df.to_string(
            index=False
        )
    )

    v4_representation_matches = (
        audit_df.loc[
            audit_df[
                "artifact"
            ]
            == "negative_matrix",
            "matches_v4_generated_hash",
        ]
    )

    representation_explanation_found = bool(
        v4_representation_matches.any()
    )

    matching_rows = audit_df.loc[
        audit_df[
            "matches_v4_generated_hash"
        ]
    ].copy()

    banner("DIAGNOSIS")

    if representation_explanation_found:
        require(
            len(matching_rows) >= 1,
            "Internal representation-match inconsistency.",
        )

        print(
            "V4 hash is reproduced by re-encoding the "
            "FROZEN VALUES as:"
        )

        for row in matching_rows.itertuples(
            index=False
        ):
            print(
                f"  {row.representation} / "
                f"dtype={row.dtype} / "
                f"dtype.str={row.dtype_str}"
            )

        print()
        print(
            "Conclusion: VALUE STREAM IS CONSISTENT; "
            "LOGICAL HASH MISMATCH IS REPRESENTATION-LEVEL."
        )
    else:
        print(
            "No tested dtype representation reproduced "
            "the V4 generated hash."
        )
        print(
            "Do NOT modify the generator yet. "
            "Next audit must compare full value equality "
            "and exact hash-function inputs."
        )

    # The native frozen representation must, by definition, reproduce
    # the frozen hash.
    native_row = audit_df.loc[
        (
            audit_df[
                "artifact"
            ]
            == "negative_matrix"
        )
        & (
            audit_df[
                "representation"
            ]
            == "native_frozen"
        )
    ].iloc[0]

    require(
        bool(
            native_row[
                "matches_frozen_negative_hash"
            ]
        ),
        "Native frozen representation did not reproduce frozen hash.",
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_df.to_csv(
        AUDIT_CSV_PATH,
        index=False,
    )

    contract = {
        "phase": "5.3.5a",
        "title": (
            "Frozen Epoch-0 Stream Representation Audit"
        ),
        "status": (
            "DIAGNOSIS_COMPLETE"
        ),
        "frozen_negative": {
            "shape": list(
                frozen_negative.shape
            ),
            "dtype": str(
                frozen_negative.dtype
            ),
            "dtype_str": (
                frozen_negative.dtype.str
            ),
            "itemsize": int(
                frozen_negative.dtype.itemsize
            ),
            "c_contiguous": bool(
                frozen_negative.flags.c_contiguous
            ),
            "f_contiguous": bool(
                frozen_negative.flags.f_contiguous
            ),
            "logical_sha256": (
                frozen_negative_sha
            ),
        },
        "v4_observed_generated_sha256": (
            EXPECTED_V4_GENERATED_INT64_SHA256
        ),
        "representation_explanation_found": (
            representation_explanation_found
        ),
        "matching_representations": (
            matching_rows[
                [
                    "representation",
                    "dtype",
                    "dtype_str",
                    "logical_sha256",
                ]
            ]
            .to_dict(
                orient="records"
            )
        ),
        "scientific_boundary": {
            "negative_sampling_executed": False,
            "model_instantiated": False,
            "optimizer_instantiated": False,
            "validation_accessed": False,
            "test_accessed": False,
            "frozen_decisions_changed": False,
        },
    }

    CONTRACT_PATH.write_text(
        json.dumps(
            contract,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    banner("WRITE OUTPUTS")

    print(f"WROTE  {AUDIT_CSV_PATH}")
    print(f"WROTE  {CONTRACT_PATH}")

    banner("PHASE 5.3.5a FINAL STATUS")

    print(
        f"Frozen negative dtype:                "
        f"{frozen_negative.dtype}"
    )
    print(
        f"Frozen negative logical SHA256:       "
        f"{frozen_negative_sha}"
    )
    print(
        f"V4 generated logical SHA256:          "
        f"{EXPECTED_V4_GENERATED_INT64_SHA256}"
    )
    print(
        "Representation explanation found:     "
        f"{'YES' if representation_explanation_found else 'NO'}"
    )
    print()
    print(
        "Scientific decisions changed:         NO"
    )
    print(
        "Training executed:                    NO"
    )
    print(
        "Test accessed:                        NO"
    )


if __name__ == "__main__":
    main()