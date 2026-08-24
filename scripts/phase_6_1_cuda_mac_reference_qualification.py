#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import torch


# =============================================================================
# Frozen paths
# =============================================================================

INITIAL_STATE_PATH = Path(
    "phase6_frozen_initial_model_state.pt"
)

MAC_REFERENCE_PATH = Path(
    "phase6_mac_reference_batches_0_1.pt"
)

HELPER_PATH = Path(
    "scripts/phase_5_4_8a_mps_numerical_equivalence_feasibility_audit_V1.py"
)

ROUNDTRIP_PATH = Path(
    "scripts/phase_5_3_2b_checkpoint_resume_roundtrip_proof.py"
)

POLICY_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_4_numerical_equivalence_policy.json"
)

CHECKPOINT_PATHS = {
    0: Path(
        "data/experimental/phase_5/audits/phase_5_3_2c/"
        "dry_run_latest_after_batch0.pt"
    ),
    1: Path(
        "data/experimental/phase_5/audits/phase_5_3_2c/"
        "dry_run_latest_after_batch1.pt"
    ),
}

OUT_DIR = Path(
    "data/experimental/phase_6/audits/"
    "phase_6_1_cuda_mac_reference"
)

CONTRACT_PATH = Path(
    "data/experimental/phase_6/contracts/"
    "phase_6_1_cuda_mac_reference_qualification.json"
)


# =============================================================================
# Frozen anchors
# =============================================================================

EXPECTED_COMMIT = (
    "6c94a4e787d2bc7a27e9c1ebced3ddf41132d915"
)

EXPECTED_INITIAL_FILE_SHA = (
    "a7451fa138e440dd6dd6e563759844982ec68126d9fc02febdc463e203ee6224"
)

EXPECTED_REFERENCE_FILE_SHA = (
    "e146e217b8fa3e7de53a30921ec6baa265f5d6c33de7bc1de10d0d01c5d2b6dd"
)

EXPECTED_INITIAL_MODEL_SHA = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED = {
    0: {
        "loss": 0.7080879807,
        "logit": (
            "35b89aaed29d51d2ebb7ba1cadf2dc4b"
            "b5e8f81cf3aa78bc216b3cc6fed13845"
        ),
        "gradient": (
            "8c542430813d8ca91b8397409954ea92"
            "295a2b55bcc420661783fb865010845d"
        ),
        "model": (
            "42a521f11d8f24e4144d0215d6e1b34d"
            "5f8bf0c2d8848624e4f7c3130699035d"
        ),
        "optimizer": (
            "5ce2683c21f456b9d5d15eb876b049c5"
            "e6db1215db5a026630f093f7f9d49891"
        ),
        "global_step": 1,
        "next_batch": 1,
    },
    1: {
        "loss": 0.6636360884,
        "logit": (
            "cfbc4106103abf9478b8f04f0e0d909b"
            "ed37659e5ee7e29257bce0a7dd4beb26"
        ),
        "gradient": (
            "8c066fd5f8002e1edd0a282f4ac549a3"
            "903f590716b38ca060b0f01088594f22"
        ),
        "model": (
            "c41702cda99092a7fb63bb0a8227e658"
            "851b3ac4cbc373d90cdd6816eccdd196"
        ),
        "optimizer": (
            "569a6691424ac32d0f252728750281cffd"
            "175a2b6b6c6ea1913f5f497200b00d"
        ),
        "global_step": 2,
        "next_batch": 2,
    },
}

EXPECTED_POLICY = "ITRS_PHASE5_NUMERICAL_EQUIVALENCE_V1"


# =============================================================================
# Helpers
# =============================================================================

def banner(text: str) -> None:
    print("\n" + "=" * 112)
    print(text)
    print("=" * 112)


def require(condition, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def load_module(path: Path, name: str):
    require(
        path.exists(),
        f"Missing Python source: {path}",
    )

    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

    require(
        spec is not None
        and spec.loader is not None,
        f"Could not import {path}",
    )

    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


def install_portable_initialization_bridge(
    preflight,
    frozen_state_dict: dict,
):
    """
    Preserve the original Phase-5 compose_canonical_model implementation.

    Only the canonical builder's freshly generated Linux parameter bytes are
    replaced with the authoritative Mac state_dict BEFORE the existing
    Phase-5 initial-state hash gate runs.

    Therefore the original Phase-5 gate is satisfied, not bypassed.
    """

    original_compose = (
        preflight.compose_canonical_model
    )

    def portable_compose(
        canonical_runtime,
        methods,
        adapter_callable,
    ):
        original_builder = (
            canonical_runtime.build_canonical_model
        )

        def portable_builder(*args, **kwargs):
            model = original_builder(
                *args,
                **kwargs,
            )

            model.load_state_dict(
                frozen_state_dict,
                strict=True,
            )

            return model

        canonical_runtime.build_canonical_model = (
            portable_builder
        )

        try:
            model, hash_fn = original_compose(
                canonical_runtime,
                methods,
                adapter_callable,
            )
        finally:
            canonical_runtime.build_canonical_model = (
                original_builder
            )

        require(
            hash_fn(model)
            == EXPECTED_INITIAL_MODEL_SHA,
            (
                "Portable initialization bridge did not "
                "reproduce frozen Mac state."
            ),
        )

        return model, hash_fn

    preflight.compose_canonical_model = (
        portable_compose
    )


def load_reference_state(
    *,
    batch_index: int,
    roundtrip,
    preflight,
    mac_reference: dict,
):
    """
    Reconstruct architecture using exact portable initial state, then load the
    audited Mac production checkpoint after the requested batch.

    Mac canonical gradients are attached to the reference model so the frozen
    numerical policy can compare them directly against CUDA.
    """

    checkpoint_path = (
        CHECKPOINT_PATHS[batch_index]
    )

    require(
        checkpoint_path.exists(),
        f"Missing Mac checkpoint: {checkpoint_path}",
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    expected = EXPECTED[batch_index]

    require(
        int(
            checkpoint["global_optimizer_step"]
        )
        == expected["global_step"],
        (
            f"Checkpoint global step drift "
            f"for batch {batch_index}."
        ),
    )

    require(
        int(
            checkpoint["next_batch_index"]
        )
        == expected["next_batch"],
        (
            f"Checkpoint next_batch_index drift "
            f"for batch {batch_index}."
        ),
    )

    (
        model,
        optimizer,
        hash_fn,
        *_,
    ) = roundtrip.construct_fresh_training_state(
        preflight
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    batch_reference = (
        mac_reference["batches"][batch_index]
    )

    named_parameters = dict(
        model.named_parameters()
    )

    reference_gradients = (
        batch_reference["gradients"]
    )

    require(
        set(reference_gradients.keys())
        == set(named_parameters.keys()),
        (
            f"Mac gradient namespace drift "
            f"at batch {batch_index}."
        ),
    )

    for name, parameter in (
        named_parameters.items()
    ):
        gradient = (
            reference_gradients[name]
            .detach()
            .cpu()
            .clone()
        )

        require(
            tuple(gradient.shape)
            == tuple(parameter.shape),
            (
                f"Gradient shape mismatch "
                f"for {name}."
            ),
        )

        parameter.grad = gradient

    logits = (
        batch_reference["logits"]
        .detach()
        .cpu()
        .clone()
    )

    # -------------------------------------------------------------
    # Exact Mac reference fingerprint gates
    # -------------------------------------------------------------

    require(
        abs(
            float(batch_reference["loss"])
            - float(expected["loss"])
        )
        <= 5e-10,
        (
            f"Mac loss fingerprint drift "
            f"at batch {batch_index}."
        ),
    )

    require(
        roundtrip.tensor_sha256(logits)
        == expected["logit"],
        (
            f"Mac logit fingerprint drift "
            f"at batch {batch_index}."
        ),
    )

    require(
        roundtrip.gradient_logical_sha256(
            model
        )
        == expected["gradient"],
        (
            f"Mac gradient fingerprint drift "
            f"at batch {batch_index}."
        ),
    )

    require(
        hash_fn(model)
        == expected["model"],
        (
            f"Mac checkpoint model fingerprint drift "
            f"at batch {batch_index}."
        ),
    )

    require(
        roundtrip.optimizer_state_logical_sha256(
            model,
            optimizer,
        )
        == expected["optimizer"],
        (
            f"Mac checkpoint Adam fingerprint drift "
            f"at batch {batch_index}."
        ),
    )

    return {
        "model": model,
        "optimizer": optimizer,
        "logits": logits,
        "loss": float(
            batch_reference["loss"]
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 6.1 — MAC CANONICAL → RTX 4090 CUDA "
        "D_BOTH_SPARSE NUMERICAL QUALIFICATION"
    )

    print("Production training launched: NO")
    print("Validation cases scored:      0")
    print("Test cases scored:            0")

    # =========================================================================
    # Repository + artifact identity
    # =========================================================================

    banner(
        "FROZEN REPOSITORY + PORTABLE ARTIFACT GATE"
    )

    commit = (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        )
        .strip()
    )

    require(
        commit == EXPECTED_COMMIT,
        (
            f"Repository commit drift:\n"
            f"{commit}\n!=\n{EXPECTED_COMMIT}"
        ),
    )

    require(
        INITIAL_STATE_PATH.exists(),
        f"Missing {INITIAL_STATE_PATH}",
    )

    require(
        MAC_REFERENCE_PATH.exists(),
        f"Missing {MAC_REFERENCE_PATH}",
    )

    initial_file_sha = file_sha256(
        INITIAL_STATE_PATH
    )

    reference_file_sha = file_sha256(
        MAC_REFERENCE_PATH
    )

    require(
        initial_file_sha
        == EXPECTED_INITIAL_FILE_SHA,
        "Frozen initial-state file SHA mismatch.",
    )

    require(
        reference_file_sha
        == EXPECTED_REFERENCE_FILE_SHA,
        "Mac reference file SHA mismatch.",
    )

    print("Git commit:              PASS")
    print("Initial state file SHA:  PASS")
    print("Mac reference file SHA:  PASS")

    # =========================================================================
    # Environment
    # =========================================================================

    banner("CUDA ENVIRONMENT")

    require(
        torch.__version__.startswith(
            "2.7.0"
        ),
        f"PyTorch drift: {torch.__version__}",
    )

    require(
        torch.cuda.is_available(),
        "CUDA is unavailable.",
    )

    print("PyTorch:      ", torch.__version__)
    print("CUDA runtime: ", torch.version.cuda)
    print("GPU:          ", torch.cuda.get_device_name(0))

    # =========================================================================
    # Load portable Mac anchors
    # =========================================================================

    initial_payload = torch.load(
        INITIAL_STATE_PATH,
        map_location="cpu",
        weights_only=False,
    )

    mac_reference = torch.load(
        MAC_REFERENCE_PATH,
        map_location="cpu",
        weights_only=False,
    )

    require(
        initial_payload[
            "logical_model_sha256"
        ]
        == EXPECTED_INITIAL_MODEL_SHA,
        "Initial-state logical SHA drift.",
    )

    require(
        int(
            initial_payload[
                "parameter_count"
            ]
        )
        == 19_217_929,
        "Initial parameter-count drift.",
    )

    require(
        int(
            initial_payload[
                "parameter_tensors"
            ]
        )
        == 32,
        "Initial tensor-count drift.",
    )

    require(
        mac_reference[
            "schema_version"
        ]
        == "ITRS_PHASE6_MAC_REFERENCE_V1",
        "Mac reference schema drift.",
    )

    require(
        mac_reference[
            "initial_model_sha256"
        ]
        == EXPECTED_INITIAL_MODEL_SHA,
        "Mac reference initial SHA drift.",
    )

    require(
        set(
            mac_reference["batches"].keys()
        )
        == {0, 1},
        "Mac reference does not contain exactly batches 0 and 1.",
    )

    print("Portable initial model: PASS")
    print("Mac batch references:   PASS")

    # =========================================================================
    # Load exact frozen Phase-5 runtime
    # =========================================================================

    banner(
        "LOAD FROZEN PHASE-5 NUMERICAL RUNTIME"
    )

    helper = load_module(
        HELPER_PATH,
        "_phase6_helper",
    )

    roundtrip = load_module(
        ROUNDTRIP_PATH,
        "_phase6_roundtrip",
    )

    preflight = (
        roundtrip.load_preflight_runtime()
    )

    install_portable_initialization_bridge(
        preflight,
        initial_payload[
            "model_state_dict"
        ],
    )

    policy = json.loads(
        POLICY_PATH.read_text(
            encoding="utf-8"
        )
    )

    require(
        policy["status"] == "FROZEN",
        "Numerical policy is not frozen.",
    )

    require(
        policy["schema_version"]
        == EXPECTED_POLICY,
        "Numerical policy version drift.",
    )

    thresholds = policy["thresholds"]

    torch.set_num_threads(8)

    # =========================================================================
    # Frozen stream + shared inputs
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

    # =========================================================================
    # Portable-constructor proof BEFORE touching CUDA training
    # =========================================================================

    banner(
        "PORTABLE CONSTRUCTOR + MAC REFERENCE CHECKPOINT PREFLIGHT"
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
        == EXPECTED_INITIAL_MODEL_SHA,
        "Portable fresh model hash failed.",
    )

    require(
        len(probe_optimizer.state) == 0,
        "Portable fresh Adam is not empty.",
    )

    del probe_model
    del probe_optimizer

    references = {
        0: load_reference_state(
            batch_index=0,
            roundtrip=roundtrip,
            preflight=preflight,
            mac_reference=mac_reference,
        ),
        1: load_reference_state(
            batch_index=1,
            roundtrip=roundtrip,
            preflight=preflight,
            mac_reference=mac_reference,
        ),
    }

    print(
        "Portable initial-state gate:      PASS"
    )
    print(
        "Mac batch-0 checkpoint/logits:    PASS"
    )
    print(
        "Mac batch-1 checkpoint/logits:    PASS"
    )
    print(
        "Mac gradients + Adam fingerprints: PASS"
    )

    # =========================================================================
    # Derive lean device-aware executor
    # =========================================================================

    (
        lean_execute,
        set_executor_device,
        executor_metadata,
    ) = helper.build_lean_device_executor(
        roundtrip
    )

    require(
        executor_metadata[
            "require_statements_removed"
        ]
        == 25,
        "Unexpected lean-executor guard count.",
    )

    require(
        executor_metadata[
            "torch_from_numpy_rewrites"
        ]
        == 10,
        "Unexpected from_numpy rewrite count.",
    )

    print(
        "Lean executor provenance:         PASS"
    )

    # =========================================================================
    # Construct CUDA candidate from exact Mac state
    # =========================================================================

    banner(
        "CONSTRUCT RTX 4090 D_BOTH_SPARSE CANDIDATE"
    )

    (
        candidate_model,
        candidate_optimizer_cpu,
        candidate_hash_fn,
        *_,
    ) = roundtrip.construct_fresh_training_state(
        preflight
    )

    require(
        candidate_hash_fn(
            candidate_model
        )
        == EXPECTED_INITIAL_MODEL_SHA,
        "CUDA candidate pre-transfer SHA drift.",
    )

    del candidate_optimizer_cpu

    cuda = torch.device("cuda:0")

    candidate_model = (
        candidate_model.to(cuda)
    )

    # Phase-5 qualified accelerated implementation choice.
    candidate_model.startup_embedding.sparse = True
    candidate_model.investor_embedding.sparse = True

    candidate_model.train()

    candidate_optimizer = (
        preflight.build_frozen_adam(
            candidate_model
        )
    )

    require(
        len(candidate_optimizer.state) == 0,
        "Fresh CUDA Adam state is not empty.",
    )

    require(
        candidate_hash_fn(
            candidate_model
        )
        == EXPECTED_INITIAL_MODEL_SHA,
        (
            "Exact initial parameter bytes changed "
            "during CPU → CUDA transfer."
        ),
    )

    shared_cuda = dict(shared_cpu)

    shared_cuda["edge_index"] = (
        shared_cpu["edge_index"].to(
            cuda
        )
    )

    shared_cuda["edge_type"] = (
        shared_cpu["edge_type"].to(
            cuda
        )
    )

    capture = helper.LogitCapture(
        candidate_model
    )
    capture.install()

    print("Initial Mac parameter state: BYTE-EXACT")
    print("CUDA candidate initial state: BYTE-EXACT")
    print("startup_embedding.sparse:     True")
    print("investor_embedding.sparse:    True")
    print("Fresh CUDA Adam:               EMPTY")

    # =========================================================================
    # Stateful CUDA batch 0 → batch 1
    # =========================================================================

    banner(
        "STATEFUL MAC CANONICAL → CUDA NUMERICAL COMPARISON"
    )

    batch_rows = []
    policy_rows_all = []

    set_executor_device("cuda:0")

    for batch_index in (0, 1):

        batch = batches[batch_index]
        reference = references[batch_index]

        torch.cuda.synchronize()

        start = time.perf_counter()

        candidate_result = lean_execute(
            candidate_model,
            candidate_optimizer,
            lambda model: (
                "LEAN_RUNTIME_HASH_SKIPPED"
            ),
            batch,
            shared_cuda,
        )

        torch.cuda.synchronize()

        cuda_seconds = (
            time.perf_counter()
            - start
        )

        candidate_logits = (
            capture.cpu_copy()
        )

        # ---------------------------------------------------------------------
        # Frozen-policy numerical comparisons against ORIGINAL MAC tensors.
        # ---------------------------------------------------------------------

        logit = helper.tensor_stats(
            reference["logits"],
            candidate_logits,
        )

        gradient = (
            helper.compare_model_gradients(
                reference["model"],
                candidate_model,
            )
        )

        parameter = (
            helper.compare_model_parameters(
                reference["model"],
                candidate_model,
            )
        )

        adam_avg = (
            helper.compare_adam_group(
                reference["model"],
                reference["optimizer"],
                candidate_model,
                candidate_optimizer,
                "exp_avg",
            )
        )

        adam_sq = (
            helper.compare_adam_group(
                reference["model"],
                reference["optimizer"],
                candidate_model,
                candidate_optimizer,
                "exp_avg_sq",
            )
        )

        loss_abs_diff = abs(
            float(
                candidate_result["loss"]
            )
            - reference["loss"]
        )

        summary = {
            "batch_index": batch_index,
            "cuda_seconds": cuda_seconds,
            "reference_loss": (
                reference["loss"]
            ),
            "cuda_loss": float(
                candidate_result["loss"]
            ),
            "loss_abs_diff": (
                loss_abs_diff
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

        policy_rows = helper.policy_rows(
            batch_index,
            summary,
            thresholds,
        )

        policy_rows_all.extend(
            policy_rows
        )

        batch_rows.append(
            summary
        )

        passed = all(
            row["result"] == "PASS"
            for row in policy_rows
        )

        print()
        print(
            f"BATCH {batch_index}"
        )
        print(
            f"  RTX 4090 CUDA seconds:          "
            f"{cuda_seconds:.3f}"
        )
        print(
            f"  loss abs diff:                  "
            f"{loss_abs_diff:.12e}"
        )
        print(
            f"  logit max abs diff:             "
            f"{logit['max_abs_diff']:.12e}"
        )
        print(
            f"  gradient relative L2:           "
            f"{gradient['relative_l2_error']:.12e}"
        )
        print(
            f"  gradient cosine:                "
            f"{gradient['cosine_similarity']:.12f}"
        )
        print(
            f"  gradient sign agreement:        "
            f"{gradient['sign_agreement']:.12f}"
        )
        print(
            f"  parameter relative L2:          "
            f"{parameter['relative_l2_error']:.12e}"
        )
        print(
            f"  parameter max abs diff:         "
            f"{parameter['max_abs_diff']:.12e}"
        )
        print(
            f"  Adam exp_avg relative L2:       "
            f"{adam_avg['relative_l2_error']:.12e}"
        )
        print(
            f"  Adam exp_avg_sq relative L2:    "
            f"{adam_sq['relative_l2_error']:.12e}"
        )
        print(
            f"  FROZEN POLICY:                  "
            f"{'PASS' if passed else 'FAIL'}"
        )

    # =========================================================================
    # Final decision
    # =========================================================================

    passed_checks = sum(
        row["result"] == "PASS"
        for row in policy_rows_all
    )

    total_checks = len(
        policy_rows_all
    )

    qualification_pass = bool(
        passed_checks == total_checks
        and total_checks == 22
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
        OUT_DIR
        / "cuda_mac_reference_batch_summary.csv",
        index=False,
    )

    pd.DataFrame(
        policy_rows_all
    ).to_csv(
        OUT_DIR
        / "cuda_mac_reference_policy_audit.csv",
        index=False,
    )

    contract = {
        "phase": "6.1",
        "status": (
            "QUALIFIED"
            if qualification_pass
            else "FAILED"
        ),
        "candidate_runtime": (
            "D_BOTH_SPARSE_CUDA"
        ),
        "reference_runtime": (
            "FROZEN_MAC_CANONICAL_CPU"
        ),
        "repository_commit": (
            EXPECTED_COMMIT
        ),
        "portable_initial_state_file_sha256": (
            EXPECTED_INITIAL_FILE_SHA
        ),
        "mac_reference_file_sha256": (
            EXPECTED_REFERENCE_FILE_SHA
        ),
        "canonical_initial_model_sha256": (
            EXPECTED_INITIAL_MODEL_SHA
        ),
        "numerical_policy": (
            EXPECTED_POLICY
        ),
        "device": (
            torch.cuda.get_device_name(0)
        ),
        "torch_version": (
            torch.__version__
        ),
        "cuda_runtime": (
            torch.version.cuda
        ),
        "stateful_batches_tested": 2,
        "policy_checks_passed": (
            passed_checks
        ),
        "policy_checks_total": (
            total_checks
        ),
        "numerically_equivalent": (
            qualification_pass
        ),
        "production_training_launched": (
            False
        ),
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
        "PHASE 6.1 CUDA QUALIFICATION RESULT"
    )

    print(
        f"Frozen policy:       {EXPECTED_POLICY}"
    )
    print(
        f"Policy checks:       "
        f"{passed_checks}/{total_checks}"
    )
    print(
        "Stateful batches:    2"
    )
    print(
        "Validation accessed: NO"
    )
    print(
        "Test accessed:       NO"
    )
    print(
        "Production training: NO"
    )
    print()

    if qualification_pass:
        print(
            "D_BOTH_SPARSE_CUDA: NUMERICALLY EQUIVALENT"
        )
        print(
            "PHASE 6.1: PASS"
        )
    else:
        print(
            "D_BOTH_SPARSE_CUDA: FAILED FROZEN POLICY"
        )
        print(
            "PHASE 6.1: FAIL"
        )

        failed = [
            row
            for row in policy_rows_all
            if row["result"] != "PASS"
        ]

        print()
        print("FAILED CHECKS:")

        for row in failed:
            print(
                f"  batch={row['batch_index']} "
                f"{row['metric']}: "
                f"{row['actual']} "
                f"{row['comparator']} "
                f"{row['threshold']}"
            )

        raise AssertionError(
            "CUDA candidate failed frozen numerical-equivalence policy."
        )


if __name__ == "__main__":
    main()
