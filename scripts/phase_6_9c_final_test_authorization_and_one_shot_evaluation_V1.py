#!/usr/bin/env python3
"""
Phase 6.9c — Final-Test Authorization + One-Shot Evaluation V1

Purpose
-------
Authorize and execute exactly one final test evaluation for the already-selected
100% production checkpoint from Phase 6.9b.

Scientific boundary
-------------------
- Training is complete: 20 / 20 epochs, 209,620 optimizer steps.
- Checkpoint selection is already frozen from validation only.
- Selected checkpoint: display Epoch 19 (zero-based epoch_index 18).
- Selected best.pt physical SHA256 is fixed below.
- Frozen Phase-5 evaluation candidates are reused exactly.
- Test cases: 20,264 event-level cases.
- Candidates / case: 1 focal positive + 99 frozen negatives.
- Ranking: raw logit descending, tie-break startup_local ascending.
- Metrics: HR@10 and NDCG@10, event-level arithmetic mean.
- No optimizer, backward, training, checkpoint selection, or model update occurs.
- --run-once refuses to execute after final-test outputs already exist.

Modes
-----
--preflight-only
    Read-only integrity gate. Scores ZERO test cases and writes NOTHING.

--authorize
    Re-runs the read-only gate and writes a frozen authorization contract.
    Scores ZERO test cases.

--run-once
    Requires the authorization contract, verifies all bindings again, then
    evaluates all 20,264 test cases exactly once and writes final artifacts.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PHASE = "6.9c"
EXPERIMENT = "FULL_100PCT_ITRS_FINAL_TEST"
SCHEMA_VERSION = "ITRS_PHASE6_9C_FINAL_TEST_V1"

EXPECTED_REPOSITORY_HEAD = (
    "6c94a4e787d2bc7a27e9c1ebced3ddf41132d915"
)

NUM_INVESTORS = 165_975
NUM_STARTUPS = 311_589
NUM_NODES = 477_564

VALIDATION_CASES = 2_251
TEST_CASES = 20_264
TOTAL_CASES = 22_515
NEGATIVES_PER_CASE = 99
CANDIDATES_PER_CASE = 100
K = 10
TEST_CHUNK_SIZE = 64

EXPECTED_BEST_DISPLAY_EPOCH = 19
EXPECTED_BEST_EPOCH_INDEX = 18
EXPECTED_BEST_VALIDATION_NDCG10 = 0.3896115434372846
EXPECTED_BEST_VALIDATION_HR10 = 0.5362061306086184

EXPECTED_BEST_CHECKPOINT_SHA256 = (
    "682e517744fab52367702290673c8c5b1"
    "87f965caf4e075a7c2d29e79459ce98"
)
EXPECTED_LATEST_CHECKPOINT_SHA256 = (
    "024acf6b9c35fd0020476eb2f88369fa"
    "1d0fa6d9cf27d0cc123794aa54bb5fb2"
)
EXPECTED_EPOCH_METRICS_SHA256 = (
    "35f2781127a1a68e2ca88c3a46cdc95e"
    "c7039e5c1d5f71908de60894efe3a42d"
)
EXPECTED_PHASE6_9B_CONTRACT_SHA256 = (
    "795672172d34e63e4a88dfc603351813"
    "5a50b9ff449625bc71c446f431337401"
)
EXPECTED_NEGATIVE_MATRIX_SHA256 = (
    "7f98269d0291382dacfc783ffd66ad6d2"
    "c2f8775877d57dc0d83504e40d8716d"
)
EXPECTED_CASE_MANIFEST_SHA256 = (
    "44b4b7e1ec1b1978249318a080df02d0"
    "d9f617845263513594de64e71b969e0c"
)
EXPECTED_PHASE5_EVAL_GENERATION_MANIFEST_SHA256 = (
    "b3e43fa19deb57ce55b2499838055e53"
    "c520bc0395912ef0de7a340c55ac20b8"
)

ROUNDTRIP_SOURCE_PATH = Path(
    "scripts/phase_5_3_2b_checkpoint_resume_roundtrip_proof.py"
)
RANKING_SOURCE_PATH = Path(
    "scripts/phase_5_3_3a_validation_ranking_metric_semantics_audit.py"
)
PREFLIGHT_SOURCE_PATH = Path(
    "scripts/phase_5_3_3b_canonical_real_validation_scoring_preflight.py"
)
FULL_VALIDATION_RUNTIME_PATH = Path(
    "scripts/phase_5_3_3c_full_validation_split_runtime_dry_run.py"
)

RANKING_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_3a_validation_ranking_metric_contract.json"
)
EVAL_CANDIDATE_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_1_2b_t60_evaluation_candidate_runtime_contract.json"
)
PHASE6_9B_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_9b_100pct_20epoch_production_training_V1.json"
)
BEST_CHECKPOINT_PATH = Path(
    "data/experimental/phase_6/full_training/100pct/"
    "run_20epoch/checkpoints/best.pt"
)
LATEST_CHECKPOINT_PATH = Path(
    "data/experimental/phase_6/full_training/100pct/"
    "run_20epoch/checkpoints/latest.pt"
)
EPOCH_METRICS_PATH = Path(
    "data/experimental/phase_6/full_training/100pct/"
    "run_20epoch/epoch_metrics.csv"
)
CASE_MANIFEST_PATH = Path(
    "data/experimental/phase_5/model_ready/evaluation/"
    "t60_evaluation_case_manifest.parquet"
)
NEGATIVE_MATRIX_PATH = Path(
    "data/experimental/phase_5/model_ready/evaluation/"
    "t60_evaluation_negative_node_indices.npy"
)
EVAL_GENERATION_MANIFEST_PATH = Path(
    "data/experimental/phase_5/model_ready/evaluation/"
    "phase_5_1_2c_generation_manifest.json"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_6/full_training/100pct/final_test"
)
AUTHORIZATION_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_9c_final_test_authorization_V1.json"
)
CASE_METRICS_PATH = OUTPUT_DIR / "final_test_case_metrics.parquet"
RAW_LOGITS_PATH = OUTPUT_DIR / "final_test_raw_logits.npy"
SUMMARY_PATH = OUTPUT_DIR / "final_test_summary.json"
RESULT_CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_9c_final_test_result_V1.json"
)


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def banner(text):
    print()
    print("=" * 118)
    print(text)
    print("=" * 118)


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def array_logical_sha256(array):
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def dataframe_logical_sha256(frame, columns):
    digest = hashlib.sha256()

    for column in columns:
        require(column in frame.columns, f"Missing dataframe hash column: {column}")
        digest.update(column.encode("utf-8"))
        digest.update(b"\0")
        series = frame[column]

        if pd.api.types.is_integer_dtype(series.dtype):
            value = np.ascontiguousarray(series.to_numpy(dtype=np.int64))
            digest.update(value.tobytes(order="C"))
        elif pd.api.types.is_float_dtype(series.dtype):
            value = np.ascontiguousarray(series.to_numpy(dtype=np.float64))
            digest.update(value.tobytes(order="C"))
        elif pd.api.types.is_bool_dtype(series.dtype):
            value = np.ascontiguousarray(series.to_numpy(dtype=np.bool_))
            digest.update(value.tobytes(order="C"))
        else:
            for item in series.astype(str).tolist():
                digest.update(item.encode("utf-8"))
                digest.update(b"\0")

        digest.update(b"\0")

    return digest.hexdigest()


def load_json(path):
    require(path.exists(), f"Missing JSON: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json_write(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_parquet_write(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def atomic_npy_write(array, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npy")
    np.save(temporary, array, allow_pickle=False)
    os.replace(temporary, path)


def git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def require_clean_tracked_tree():
    result = subprocess.run(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    require(
        result.stdout.strip() == "",
        "Tracked Git working tree is not clean.\n" + result.stdout,
    )


def load_module(path, name):
    require(path.exists(), f"Missing source: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    require(
        spec is not None and spec.loader is not None,
        f"Could not import {path}",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def current_script_sha256():
    return file_sha256(Path(__file__).resolve())


def clear_stale_temp_files():
    candidates = [
        CASE_METRICS_PATH.with_name(CASE_METRICS_PATH.name + ".tmp.parquet"),
        RAW_LOGITS_PATH.with_name(RAW_LOGITS_PATH.name + ".tmp.npy"),
        Path(str(SUMMARY_PATH) + ".tmp"),
        Path(str(RESULT_CONTRACT_PATH) + ".tmp"),
    ]

    for path in candidates:
        if path.exists():
            path.unlink()


def final_outputs_exist():
    return {
        str(path): path.exists()
        for path in (
            CASE_METRICS_PATH,
            RAW_LOGITS_PATH,
            SUMMARY_PATH,
            RESULT_CONTRACT_PATH,
        )
    }


def verify_phase6_9b_selection():
    require(
        file_sha256(BEST_CHECKPOINT_PATH) == EXPECTED_BEST_CHECKPOINT_SHA256,
        "Selected best.pt SHA256 drift.",
    )
    require(
        file_sha256(LATEST_CHECKPOINT_PATH) == EXPECTED_LATEST_CHECKPOINT_SHA256,
        "latest.pt SHA256 drift.",
    )
    require(
        file_sha256(EPOCH_METRICS_PATH) == EXPECTED_EPOCH_METRICS_SHA256,
        "epoch_metrics.csv SHA256 drift.",
    )
    require(
        file_sha256(PHASE6_9B_CONTRACT_PATH) == EXPECTED_PHASE6_9B_CONTRACT_SHA256,
        "Phase 6.9b result-contract SHA256 drift.",
    )

    contract = load_json(PHASE6_9B_CONTRACT_PATH)

    require(
        contract["status"] == "COMPLETE_100PCT_20EPOCH_PRODUCTION_VALIDATION_ONLY",
        "Phase 6.9b is not validation-only complete.",
    )
    require(
        int(contract["selection"]["best_display_epoch"]) == EXPECTED_BEST_DISPLAY_EPOCH,
        "Frozen best display epoch drift.",
    )
    require(
        int(contract["selection"]["best_epoch_index"]) == EXPECTED_BEST_EPOCH_INDEX,
        "Frozen best zero-based epoch drift.",
    )
    require(
        float(contract["selection"]["best_validation_NDCG@10"])
        == EXPECTED_BEST_VALIDATION_NDCG10,
        "Frozen best validation NDCG@10 drift.",
    )
    require(
        float(contract["selection"]["best_validation_HR@10"])
        == EXPECTED_BEST_VALIDATION_HR10,
        "Frozen best validation HR@10 drift.",
    )
    require(
        contract["test_boundary"]["test_accessed"] is False
        and contract["test_boundary"]["test_scored"] is False
        and contract["test_boundary"]["test_metrics_reported"] is False,
        "Phase 6.9b test boundary drift.",
    )

    metrics = pd.read_csv(EPOCH_METRICS_PATH)
    require(len(metrics) == 20, "Expected exactly 20 validation-history rows.")
    require(
        int(metrics.iloc[-1]["global_optimizer_step"]) == 209_620,
        "Final optimizer-step count drift.",
    )

    best_position = int(metrics["validation_NDCG@10"].idxmax())
    best_row = metrics.loc[best_position]

    require(
        int(best_row["display_epoch"]) == EXPECTED_BEST_DISPLAY_EPOCH,
        "epoch_metrics best epoch drift.",
    )
    require(
        float(best_row["validation_NDCG@10"]) == EXPECTED_BEST_VALIDATION_NDCG10,
        "epoch_metrics best NDCG drift.",
    )
    require(
        float(best_row["validation_HR@10"]) == EXPECTED_BEST_VALIDATION_HR10,
        "epoch_metrics best HR drift.",
    )

    checkpoint = torch.load(
        BEST_CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    require(
        checkpoint["schema_version"] == "ITRS_PHASE6_100PCT_PRODUCTION_V1",
        "Selected checkpoint schema drift.",
    )
    require(
        checkpoint["checkpoint_role"] == "BEST_VALIDATION",
        "Selected checkpoint role drift.",
    )

    state = checkpoint["controller_state"]

    require(
        int(state["best_validation_epoch"]) == EXPECTED_BEST_EPOCH_INDEX,
        "Checkpoint best-validation epoch drift.",
    )
    require(
        int(state["epoch_index"]) == EXPECTED_BEST_EPOCH_INDEX,
        "Checkpoint model epoch drift.",
    )
    require(
        float(state["best_validation_ndcg10"]) == EXPECTED_BEST_VALIDATION_NDCG10,
        "Checkpoint best validation NDCG drift.",
    )
    require(
        float(state["best_validation_hr10"]) == EXPECTED_BEST_VALIDATION_HR10,
        "Checkpoint best validation HR drift.",
    )
    require(
        state["test_accessed"] is False and state["test_scored"] is False,
        "Selected checkpoint already records test access.",
    )
    require(
        checkpoint["test_boundary"]["test_accessed"] is False
        and checkpoint["test_boundary"]["test_scored"] is False,
        "Selected checkpoint test boundary drift.",
    )

    return contract, metrics, checkpoint


def verify_frozen_evaluation_artifacts(*, preflight_runtime):
    require(
        file_sha256(CASE_MANIFEST_PATH) == EXPECTED_CASE_MANIFEST_SHA256,
        "Frozen evaluation case-manifest SHA256 drift.",
    )
    require(
        file_sha256(NEGATIVE_MATRIX_PATH) == EXPECTED_NEGATIVE_MATRIX_SHA256,
        "Frozen evaluation negative-matrix SHA256 drift.",
    )
    require(
        file_sha256(EVAL_GENERATION_MANIFEST_PATH)
        == EXPECTED_PHASE5_EVAL_GENERATION_MANIFEST_SHA256,
        "Phase-5 evaluation generation manifest SHA256 drift.",
    )

    candidate_contract = load_json(EVAL_CANDIDATE_CONTRACT_PATH)
    ranking_contract = load_json(RANKING_CONTRACT_PATH)

    require(
        candidate_contract["status"] == "FROZEN",
        "Evaluation candidate contract is not FROZEN.",
    )
    require(
        ranking_contract["status"] == "FROZEN",
        "Ranking contract is not FROZEN.",
    )
    require(
        int(candidate_contract["paper_specified"]["candidate_count_per_case"])
        == CANDIDATES_PER_CASE,
        "Candidate count contract drift.",
    )
    require(
        int(candidate_contract["paper_specified"]["positive_candidates_per_case"]) == 1,
        "Positive count contract drift.",
    )
    require(
        int(candidate_contract["paper_specified"]["random_negatives_per_case"])
        == NEGATIVES_PER_CASE,
        "Negative count contract drift.",
    )
    require(
        candidate_contract["sampling"]["generate_once"] is True
        and candidate_contract["sampling"]["regenerate_across_checkpoints"] is False
        and candidate_contract["sampling"]["reuse_for_every_test_pass"] is True,
        "Frozen evaluation sampling policy drift.",
    )
    require(
        ranking_contract["ranking"]["score"] == "raw model logit",
        "Ranking score drift.",
    )
    require(
        ranking_contract["ranking"]["primary_order"] == "logit descending",
        "Ranking primary order drift.",
    )
    require(
        ranking_contract["ranking"]["tie_break"] == "startup_local ascending",
        "Ranking tie-break drift.",
    )

    case_manifest = pd.read_parquet(CASE_MANIFEST_PATH)
    negative_raw = np.load(NEGATIVE_MATRIX_PATH, mmap_mode="r")

    require(len(case_manifest) == TOTAL_CASES, "Evaluation case-manifest row count drift.")
    require(
        negative_raw.shape == (TOTAL_CASES, NEGATIVES_PER_CASE),
        "Evaluation negative-matrix shape drift.",
    )

    resolved_cases, binding_metadata = (
        preflight_runtime.resolve_case_bindings(
            case_manifest,
            ranking_contract,
        )
    )

    require(
        len(resolved_cases) == TOTAL_CASES,
        "Resolved evaluation case count drift.",
    )
    require(
        resolved_cases["matrix_row_index"].tolist() == list(range(TOTAL_CASES)),
        "Evaluation case/matrix alignment drift.",
    )

    split_counts = resolved_cases["split"].value_counts()
    require(
        int(split_counts["validation"]) == VALIDATION_CASES,
        "Validation count drift.",
    )
    require(
        int(split_counts["test"]) == TEST_CASES,
        "Test count drift.",
    )

    test_cases = (
        resolved_cases.loc[
            resolved_cases["split"] == "test"
        ]
        .copy()
        .reset_index(drop=True)
    )

    require(
        test_cases["matrix_row_index"].to_numpy(dtype=np.int64).tolist()
        == list(range(VALIDATION_CASES, TOTAL_CASES)),
        "Test cases are not the frozen contiguous suffix.",
    )

    coordinate = preflight_runtime.infer_negative_matrix_coordinate(negative_raw)

    negative_local = preflight_runtime.normalize_negative_matrix_to_local(
        negative_raw,
        coordinate,
    )

    matrix_rows = test_cases["matrix_row_index"].to_numpy(dtype=np.int64)
    test_negative_local = negative_local[matrix_rows]
    positive_local = test_cases["positive_startup_local"].to_numpy(dtype=np.int64)

    test_candidates = np.concatenate(
        [positive_local[:, None], test_negative_local],
        axis=1,
    ).astype(np.int64, copy=False)

    require(
        test_candidates.shape == (TEST_CASES, CANDIDATES_PER_CASE),
        "Test candidate matrix shape drift.",
    )

    unique_counts = np.apply_along_axis(
        lambda row: len(np.unique(row)),
        axis=1,
        arr=test_candidates,
    )

    require(
        bool((unique_counts == CANDIDATES_PER_CASE).all()),
        "At least one test case has duplicate candidates.",
    )
    require(
        bool((test_candidates >= 0).all())
        and bool((test_candidates < NUM_STARTUPS).all()),
        "Test candidate startup-local domain drift.",
    )

    return {
        "candidate_contract": candidate_contract,
        "ranking_contract": ranking_contract,
        "case_manifest": case_manifest,
        "negative_raw": negative_raw,
        "negative_coordinate": coordinate,
        "resolved_cases": resolved_cases,
        "binding_metadata": binding_metadata,
        "test_cases": test_cases,
        "test_candidates": test_candidates,
        "test_candidate_logical_sha256": array_logical_sha256(test_candidates),
    }


def build_test_chunk_schedule():
    chunks = []
    start = 0
    chunk_index = 0

    while start < TEST_CASES:
        end = min(start + TEST_CHUNK_SIZE, TEST_CASES)
        chunks.append((chunk_index, start, end))
        start = end
        chunk_index += 1

    covered = []
    for _, start, end in chunks:
        covered.extend(range(start, end))

    require(
        covered == list(range(TEST_CASES)),
        "Test chunk schedule contains gap/overlap/reordering.",
    )

    return chunks


def load_runtime():
    preflight_runtime = load_module(
        PREFLIGHT_SOURCE_PATH,
        "_phase6_9c_preflight",
    )
    full_validation_runtime = load_module(
        FULL_VALIDATION_RUNTIME_PATH,
        "_phase6_9c_full_validation",
    )
    ranking_runtime = load_module(
        RANKING_SOURCE_PATH,
        "_phase6_9c_ranking",
    )
    runtime_2b = load_module(
        ROUNDTRIP_SOURCE_PATH,
        "_phase6_9c_roundtrip",
    )

    for symbol in (
        "construct_canonical_validation_model",
        "resolve_case_bindings",
        "infer_negative_matrix_coordinate",
        "normalize_negative_matrix_to_local",
        "score_validation_case",
        "rng_snapshot",
        "rng_equal",
    ):
        require(
            hasattr(preflight_runtime, symbol),
            f"Frozen preflight runtime missing symbol: {symbol}",
        )

    require(
        hasattr(full_validation_runtime, "compute_chunk_features"),
        "Frozen full-validation runtime missing compute_chunk_features.",
    )

    for symbol in (
        "rank_candidates",
        "metrics_from_positive_rank",
        "aggregate_event_level_metrics",
    ):
        require(
            hasattr(ranking_runtime, symbol),
            f"Frozen ranking runtime missing symbol: {symbol}",
        )

    return (
        preflight_runtime,
        full_validation_runtime,
        ranking_runtime,
        runtime_2b,
    )


def construct_selected_test_model(*, preflight_runtime, runtime_2b, checkpoint):
    (
        canonical_preflight,
        model,
        hash_fn,
        *_,
    ) = preflight_runtime.construct_canonical_validation_model(runtime_2b)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    selected_state_sha = hash_fn(model)

    require(
        all(parameter.grad is None for parameter in model.parameters()),
        "Selected final-test model unexpectedly has gradients.",
    )

    shared = runtime_2b.load_shared_inputs(canonical_preflight)

    return model, hash_fn, selected_state_sha, shared


def preflight_gate():
    banner("PHASE 6.9c — FINAL TEST READ-ONLY PREFLIGHT")

    require(
        torch.__version__.startswith("2.7.0"),
        f"Expected PyTorch 2.7.0, got {torch.__version__}",
    )

    repository_commit = git_head()

    require(
        repository_commit == EXPECTED_REPOSITORY_HEAD,
        (
            "Repository HEAD drift. "
            f"Expected {EXPECTED_REPOSITORY_HEAD}, got {repository_commit}."
        ),
    )

    require_clean_tracked_tree()

    for path in (
        ROUNDTRIP_SOURCE_PATH,
        RANKING_SOURCE_PATH,
        PREFLIGHT_SOURCE_PATH,
        FULL_VALIDATION_RUNTIME_PATH,
        RANKING_CONTRACT_PATH,
        EVAL_CANDIDATE_CONTRACT_PATH,
        PHASE6_9B_CONTRACT_PATH,
        BEST_CHECKPOINT_PATH,
        LATEST_CHECKPOINT_PATH,
        EPOCH_METRICS_PATH,
        CASE_MANIFEST_PATH,
        NEGATIVE_MATRIX_PATH,
        EVAL_GENERATION_MANIFEST_PATH,
    ):
        require(path.exists(), f"Missing prerequisite: {path}")

    (
        phase6_9b_contract,
        epoch_metrics,
        best_checkpoint,
    ) = verify_phase6_9b_selection()

    (
        preflight_runtime,
        full_validation_runtime,
        ranking_runtime,
        runtime_2b,
    ) = load_runtime()

    evaluation = verify_frozen_evaluation_artifacts(
        preflight_runtime=preflight_runtime,
    )

    (
        model,
        model_hash_fn,
        selected_state_sha,
        shared,
    ) = construct_selected_test_model(
        preflight_runtime=preflight_runtime,
        runtime_2b=runtime_2b,
        checkpoint=best_checkpoint,
    )

    require(
        int(model.investor_embedding.weight.shape[0]) == NUM_INVESTORS,
        "Investor embedding row-count drift.",
    )
    require(
        int(model.startup_embedding.weight.shape[0]) == NUM_STARTUPS,
        "Startup embedding row-count drift.",
    )

    chunks = build_test_chunk_schedule()

    bindings = {
        "repository_commit": repository_commit,
        "evaluator_script_sha256": current_script_sha256(),
        "phase6_9b_contract_sha256": EXPECTED_PHASE6_9B_CONTRACT_SHA256,
        "epoch_metrics_sha256": EXPECTED_EPOCH_METRICS_SHA256,
        "selected_best_checkpoint_sha256": EXPECTED_BEST_CHECKPOINT_SHA256,
        "latest_checkpoint_sha256": EXPECTED_LATEST_CHECKPOINT_SHA256,
        "evaluation_case_manifest_sha256": EXPECTED_CASE_MANIFEST_SHA256,
        "evaluation_negative_matrix_sha256": EXPECTED_NEGATIVE_MATRIX_SHA256,
        "evaluation_generation_manifest_sha256": (
            EXPECTED_PHASE5_EVAL_GENERATION_MANIFEST_SHA256
        ),
        "ranking_contract_sha256": file_sha256(RANKING_CONTRACT_PATH),
        "evaluation_candidate_contract_sha256": file_sha256(
            EVAL_CANDIDATE_CONTRACT_PATH
        ),
        "selected_model_state_logical_sha256": selected_state_sha,
        "test_candidate_matrix_logical_sha256": (
            evaluation["test_candidate_logical_sha256"]
        ),
    }

    print("Phase 6.9b validation-only completion: PASS")
    print("Frozen best checkpoint:                PASS")
    print(f"Best display epoch:                    {EXPECTED_BEST_DISPLAY_EPOCH}")
    print(
        f"Best validation NDCG@10:               "
        f"{EXPECTED_BEST_VALIDATION_NDCG10:.12f}"
    )
    print(
        f"Best validation HR@10:                 "
        f"{EXPECTED_BEST_VALIDATION_HR10:.12f}"
    )
    print("Frozen evaluation artifacts:           PASS")
    print(f"Evaluation cases:                      {TOTAL_CASES:,}")
    print(f"Validation cases:                      {VALIDATION_CASES:,}")
    print(f"Test cases bound:                      {TEST_CASES:,}")
    print(f"Candidates / test case:                {CANDIDATES_PER_CASE}")
    print(f"Test chunks:                           {len(chunks)}")
    print("Optimizer instantiated:                NO")
    print("Backward executed:                     NO")
    print("Test cases scored in preflight:        0")
    print("Checkpoint selection after test:       FORBIDDEN")

    return {
        "phase6_9b_contract": phase6_9b_contract,
        "epoch_metrics": epoch_metrics,
        "best_checkpoint": best_checkpoint,
        "preflight_runtime": preflight_runtime,
        "full_validation_runtime": full_validation_runtime,
        "ranking_runtime": ranking_runtime,
        "runtime_2b": runtime_2b,
        "evaluation": evaluation,
        "model": model,
        "model_hash_fn": model_hash_fn,
        "selected_state_sha": selected_state_sha,
        "shared": shared,
        "chunks": chunks,
        "bindings": bindings,
    }


def write_authorization(context):
    require(
        not AUTHORIZATION_PATH.exists(),
        "Authorization contract already exists. Refusing to overwrite.",
    )

    outputs = final_outputs_exist()

    require(
        not any(outputs.values()),
        f"Final-test output already exists before authorization: {outputs}",
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": "AUTHORIZED_FOR_ONE_SHOT_FINAL_TEST",
        "experiment": EXPERIMENT,
        "authorization_boundary": {
            "training_complete": True,
            "checkpoint_selection_complete": True,
            "selection_criterion": (
                "validation NDCG@10, then HR@10, then earliest epoch"
            ),
            "selected_best_display_epoch": EXPECTED_BEST_DISPLAY_EPOCH,
            "selected_best_epoch_index": EXPECTED_BEST_EPOCH_INDEX,
            "selected_best_validation_NDCG@10": EXPECTED_BEST_VALIDATION_NDCG10,
            "selected_best_validation_HR@10": EXPECTED_BEST_VALIDATION_HR10,
            "test_cases_scored_before_authorization": 0,
            "checkpoint_reselection_after_test": False,
        },
        "test_protocol": {
            "cases": TEST_CASES,
            "candidates_per_case": CANDIDATES_PER_CASE,
            "positive_per_case": 1,
            "negatives_per_case": NEGATIVES_PER_CASE,
            "metrics": ["HR@10", "NDCG@10"],
            "ranking": (
                "raw logit descending; startup_local ascending tie-break"
            ),
            "aggregation": (
                "arithmetic mean across event-level test cases"
            ),
            "passes_authorized": 1,
        },
        "bindings": context["bindings"],
        "output_namespace": {
            "case_metrics": str(CASE_METRICS_PATH),
            "raw_logits": str(RAW_LOGITS_PATH),
            "summary": str(SUMMARY_PATH),
            "result_contract": str(RESULT_CONTRACT_PATH),
        },
    }

    atomic_json_write(payload, AUTHORIZATION_PATH)
    return payload


def verify_authorization(context):
    require(
        AUTHORIZATION_PATH.exists(),
        "Missing final-test authorization contract. Run --authorize first.",
    )

    authorization = load_json(AUTHORIZATION_PATH)

    require(
        authorization["schema_version"] == SCHEMA_VERSION,
        "Authorization schema drift.",
    )
    require(
        authorization["status"] == "AUTHORIZED_FOR_ONE_SHOT_FINAL_TEST",
        "Final test is not authorized.",
    )
    require(
        authorization["bindings"] == context["bindings"],
        "Authorization bindings differ from the current frozen runtime.",
    )
    require(
        int(authorization["test_protocol"]["passes_authorized"]) == 1,
        "Authorization does not permit exactly one pass.",
    )

    return authorization


def execute_one_shot_test(context):
    outputs = final_outputs_exist()

    require(
        not any(outputs.values()),
        (
            "One-shot final test refuses to run because "
            f"final output already exists: {outputs}"
        ),
    )

    clear_stale_temp_files()
    verify_authorization(context)

    model = context["model"]
    hash_fn = context["model_hash_fn"]
    selected_state_sha = context["selected_state_sha"]
    shared = context["shared"]
    preflight_runtime = context["preflight_runtime"]
    full_validation_runtime = context["full_validation_runtime"]
    ranking_runtime = context["ranking_runtime"]
    test_cases = context["evaluation"]["test_cases"]
    test_candidates = context["evaluation"]["test_candidates"]
    chunks = context["chunks"]

    require(len(test_cases) == TEST_CASES, "Final test case count drift.")
    require(
        test_candidates.shape == (TEST_CASES, CANDIDATES_PER_CASE),
        "Final test candidate matrix shape drift.",
    )

    state_before = hash_fn(model)

    require(
        state_before == selected_state_sha,
        "Selected model state changed before final test.",
    )

    rng_before = preflight_runtime.rng_snapshot()

    metric_rows = []

    raw_logits = np.empty(
        (TEST_CASES, CANDIDATES_PER_CASE),
        dtype=np.float32,
    )

    banner("PHASE 6.9c — ONE-SHOT FINAL TEST")

    print(f"Selected display epoch:                {EXPECTED_BEST_DISPLAY_EPOCH}")
    print(f"Test cases:                            {TEST_CASES:,}")
    print(f"Candidates / case:                     {CANDIDATES_PER_CASE}")
    print(f"Chunks:                                {len(chunks)}")
    print("Optimizer instantiated:                NO")
    print("Backward:                              NO")
    print("Checkpoint reselection:                FORBIDDEN")

    start_time = time.perf_counter()

    with torch.no_grad():
        latent_all = torch.cat(
            [
                model.investor_embedding.weight,
                model.startup_embedding.weight,
            ],
            dim=0,
        )

        structural = model.preference_propagation(
            latent_all,
            shared["edge_index"],
            shared["edge_type"],
        )

        require(
            isinstance(structural, dict) and "F_s" in structural,
            "Final-test structural output invalid.",
        )

        F_s_all = structural["F_s"]

        for chunk_index, start, end in chunks:
            chunk_cases = (
                test_cases.iloc[start:end]
                .copy()
                .reset_index(drop=True)
            )

            chunk_candidates = test_candidates[start:end]

            features = full_validation_runtime.compute_chunk_features(
                preflight_runtime=preflight_runtime,
                model=model,
                chunk_cases=chunk_cases,
                chunk_candidate_matrix_local=chunk_candidates,
                shared=shared,
                F_s_all=F_s_all,
            )

            for local_position, row in chunk_cases.iterrows():
                global_test_position = start + int(local_position)

                investor_global = int(row["investor_global"])
                positive_local = int(row["positive_startup_local"])
                candidates_local = chunk_candidates[local_position]

                require(
                    int(candidates_local[0]) == positive_local,
                    (
                        "Focal positive is not candidate position 0 "
                        "in frozen final-test binding."
                    ),
                )

                logits = preflight_runtime.score_validation_case(
                    model,
                    investor_global,
                    candidates_local,
                    features,
                )

                logits_np = (
                    logits.detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )

                require(
                    bool(np.isfinite(logits_np).all()),
                    "Final-test logits contain NaN/inf.",
                )

                raw_logits[global_test_position] = logits_np

                positive_rank = ranking_runtime.rank_candidates(
                    logits_np,
                    candidates_local,
                    positive_local,
                )

                hr10, ndcg10 = ranking_runtime.metrics_from_positive_rank(
                    positive_rank,
                    k=K,
                )

                metric_rows.append(
                    {
                        "test_position": global_test_position,
                        "matrix_row_index": int(row["matrix_row_index"]),
                        "interaction_id": str(row["interaction_id"]),
                        "investor_global": investor_global,
                        "positive_startup_local": positive_local,
                        "positive_rank": int(positive_rank),
                        "HR@10": float(hr10),
                        "NDCG@10": float(ndcg10),
                        "chunk_index": chunk_index,
                    }
                )

            del features

            if (
                chunk_index == 0
                or (chunk_index + 1) % 25 == 0
                or end == TEST_CASES
            ):
                elapsed_now = time.perf_counter() - start_time
                print(
                    f"chunk={chunk_index + 1:03d}/{len(chunks)} "
                    f"cases={end:05d}/{TEST_CASES} "
                    f"elapsed={elapsed_now:.1f}s"
                )

        del F_s_all

    elapsed = time.perf_counter() - start_time

    metrics = pd.DataFrame(metric_rows)

    require(
        len(metrics) == TEST_CASES,
        "Final test did not score exactly 20,264 cases.",
    )
    require(
        metrics["test_position"].tolist() == list(range(TEST_CASES)),
        "Final-test metric rows reordered.",
    )
    require(
        metrics["matrix_row_index"].tolist()
        == list(range(VALIDATION_CASES, TOTAL_CASES)),
        "Final-test matrix-row binding drift.",
    )
    require(
        bool(
            metrics["positive_rank"]
            .between(1, CANDIDATES_PER_CASE)
            .all()
        ),
        "Final-test positive rank outside 1..100.",
    )

    hr10, ndcg10 = ranking_runtime.aggregate_event_level_metrics(metrics)

    state_after = hash_fn(model)
    rng_after = preflight_runtime.rng_snapshot()

    require(
        state_after == state_before,
        "Final test changed model parameters.",
    )
    require(
        preflight_runtime.rng_equal(rng_before, rng_after),
        "Final test changed Python/NumPy/Torch CPU RNG state.",
    )
    require(
        all(parameter.grad is None for parameter in model.parameters()),
        "Final test created gradients.",
    )

    hit_count = int(metrics["HR@10"].sum())
    mean_rank = float(metrics["positive_rank"].mean())
    median_rank = float(metrics["positive_rank"].median())

    case_metrics_logical_sha = dataframe_logical_sha256(
        metrics,
        columns=[
            "test_position",
            "matrix_row_index",
            "interaction_id",
            "investor_global",
            "positive_startup_local",
            "positive_rank",
            "HR@10",
            "NDCG@10",
            "chunk_index",
        ],
    )

    raw_logits_logical_sha = array_logical_sha256(raw_logits)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": "ONE_SHOT_FINAL_TEST_COMPLETE",
        "experiment": EXPERIMENT,
        "selected_checkpoint": {
            "path": str(BEST_CHECKPOINT_PATH),
            "physical_sha256": EXPECTED_BEST_CHECKPOINT_SHA256,
            "display_epoch": EXPECTED_BEST_DISPLAY_EPOCH,
            "epoch_index": EXPECTED_BEST_EPOCH_INDEX,
            "validation_NDCG@10": EXPECTED_BEST_VALIDATION_NDCG10,
            "validation_HR@10": EXPECTED_BEST_VALIDATION_HR10,
        },
        "test": {
            "cases": TEST_CASES,
            "candidates_per_case": CANDIDATES_PER_CASE,
            "HR@10": float(hr10),
            "NDCG@10": float(ndcg10),
            "hit_count_at_10": hit_count,
            "mean_positive_rank": mean_rank,
            "median_positive_rank": median_rank,
            "elapsed_seconds": elapsed,
        },
        "integrity": {
            "model_state_before_sha256": state_before,
            "model_state_after_sha256": state_after,
            "model_state_unchanged": True,
            "rng_state_unchanged": True,
            "gradients_created": False,
            "optimizer_instantiated": False,
            "backward_executed": False,
            "checkpoint_reselection_performed": False,
            "test_passes_executed": 1,
            "case_metrics_logical_sha256": case_metrics_logical_sha,
            "raw_logits_logical_sha256": raw_logits_logical_sha,
        },
        "bindings": context["bindings"],
        "authorization_contract": {
            "path": str(AUTHORIZATION_PATH),
            "physical_sha256": file_sha256(AUTHORIZATION_PATH),
        },
    }

    atomic_parquet_write(metrics, CASE_METRICS_PATH)
    atomic_npy_write(raw_logits, RAW_LOGITS_PATH)

    summary["artifacts"] = {
        "case_metrics": {
            "path": str(CASE_METRICS_PATH),
            "physical_sha256": file_sha256(CASE_METRICS_PATH),
        },
        "raw_logits": {
            "path": str(RAW_LOGITS_PATH),
            "physical_sha256": file_sha256(RAW_LOGITS_PATH),
        },
    }

    atomic_json_write(summary, SUMMARY_PATH)

    result_contract = {
        "schema_version": SCHEMA_VERSION,
        "phase": PHASE,
        "status": "COMPLETE_FINAL_TEST_FROZEN",
        "experiment": EXPERIMENT,
        "selection_frozen_before_test": True,
        "selected_best_checkpoint_sha256": EXPECTED_BEST_CHECKPOINT_SHA256,
        "selected_best_display_epoch": EXPECTED_BEST_DISPLAY_EPOCH,
        "test_metrics": {
            "HR@10": float(hr10),
            "NDCG@10": float(ndcg10),
            "cases": TEST_CASES,
            "candidates_per_case": CANDIDATES_PER_CASE,
        },
        "test_integrity": {
            "passes_executed": 1,
            "model_state_unchanged": True,
            "rng_state_unchanged": True,
            "gradients_created": False,
            "optimizer_instantiated": False,
            "backward_executed": False,
            "checkpoint_reselection_performed": False,
        },
        "bindings": context["bindings"],
        "artifacts": {
            "authorization": {
                "path": str(AUTHORIZATION_PATH),
                "sha256": file_sha256(AUTHORIZATION_PATH),
            },
            "summary": {
                "path": str(SUMMARY_PATH),
                "sha256": file_sha256(SUMMARY_PATH),
            },
            "case_metrics": {
                "path": str(CASE_METRICS_PATH),
                "sha256": file_sha256(CASE_METRICS_PATH),
            },
            "raw_logits": {
                "path": str(RAW_LOGITS_PATH),
                "sha256": file_sha256(RAW_LOGITS_PATH),
            },
        },
    }

    atomic_json_write(result_contract, RESULT_CONTRACT_PATH)

    return (
        hr10,
        ndcg10,
        hit_count,
        mean_rank,
        median_rank,
        elapsed,
    )


def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(required=True)

    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--authorize", action="store_true")
    mode.add_argument("--run-once", action="store_true")

    args = parser.parse_args()

    context = preflight_gate()

    if args.preflight_only:
        banner("PHASE 6.9c PREFLIGHT RESULT")

        outputs = final_outputs_exist()

        print("Selection frozen before test:         PASS")
        print("Selected best checkpoint SHA:         PASS")
        print("Frozen test case binding:             PASS")
        print("Frozen 99-negative lists:             PASS")
        print("Ranking/metric semantics:             PASS")
        print("Test cases scored:                    0")
        print(f"Authorization exists:                 {AUTHORIZATION_PATH.exists()}")
        print(f"Any final-test output exists:         {any(outputs.values())}")
        print()
        print("PHASE 6.9c PREFLIGHT: PASS")
        return

    if args.authorize:
        write_authorization(context)

        banner("PHASE 6.9c AUTHORIZATION RESULT")

        print("Selection frozen before test:         PASS")
        print(f"Selected display epoch:               {EXPECTED_BEST_DISPLAY_EPOCH}")
        print("Final-test passes authorized:         1")
        print(f"Test cases authorized:                {TEST_CASES:,}")
        print("Test cases scored during authorization: 0")
        print(f"WROTE  {AUTHORIZATION_PATH}")
        print(f"authorization_sha256:                 {file_sha256(AUTHORIZATION_PATH)}")
        print()
        print("PHASE 6.9c: AUTHORIZED FOR ONE-SHOT FINAL TEST")
        return

    (
        hr10,
        ndcg10,
        hit_count,
        mean_rank,
        median_rank,
        elapsed,
    ) = execute_one_shot_test(context)

    banner("PHASE 6.9c FINAL STATUS")

    print(f"Selected best display epoch:          {EXPECTED_BEST_DISPLAY_EPOCH}")
    print(f"Test cases scored:                    {TEST_CASES:,} / {TEST_CASES:,}")
    print(f"Test HR@10:                           {hr10:.12f}")
    print(f"Test NDCG@10:                         {ndcg10:.12f}")
    print(f"Hits @10:                             {hit_count:,} / {TEST_CASES:,}")
    print(f"Mean positive rank:                   {mean_rank:.6f}")
    print(f"Median positive rank:                 {median_rank:.6f}")
    print(f"Evaluation seconds:                   {elapsed:.2f}")
    print("Model parameters changed:             NO")
    print("Checkpoint reselection performed:     NO")
    print("Final-test passes executed:            1")
    print()
    print(f"WROTE  {CASE_METRICS_PATH}")
    print(f"WROTE  {RAW_LOGITS_PATH}")
    print(f"WROTE  {SUMMARY_PATH}")
    print(f"WROTE  {RESULT_CONTRACT_PATH}")
    print()
    print("PHASE 6.9c: COMPLETE / ONE-SHOT FINAL TEST FROZEN")


if __name__ == "__main__":
    main()
