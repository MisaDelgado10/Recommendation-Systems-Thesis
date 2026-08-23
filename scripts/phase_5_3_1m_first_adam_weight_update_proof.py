#!/usr/bin/env python3
"""
Phase 5.3.1m — First Adam Weight-Update Proof

Purpose
-------
Phase 5.3.1l.2b proved the complete pre-update training runtime:

    frozen epoch-0 first batch
        ->
    canonical model
        ->
    exact variable-h trend runtime
        ->
    logits
        ->
    BCEWithLogitsLoss
        ->
    backward

with:

    512 examples
    105 positives
    407 negatives
    T1..T59
    all 32 parameter gradients finite and non-zero

and:

    optimizer.step() == 0

This phase executes EXACTLY ONE Adam update.

It deliberately reuses the frozen Phase-5.3.1l.2b script to reconstruct
the precise pre-step numerical state instead of duplicating its model,
feature, temporal-history, and batch-forward implementation.

The Phase-5.3.1l.2b module is imported only after a static import-safety
audit proves that its workflow is protected by:

    if __name__ == "__main__":
        main()

The functions:

    compose_canonical_model()
    build_frozen_adam()

are temporarily wrapped so that the model, canonical hash function,
and Adam optimizer created by Phase-5.3.1l.2b can be retained after
its main() completes.

Before optimizer.step(), this script requires exact reproduction of:

    canonical initial state SHA256
    frozen first-batch SHA256
    logit SHA256
    gradient SHA256
    BCE loss
    Adam state size == 0

Then:

    optimizer.step()

is called exactly once.

Expected first-step state:
    Adam state entries = 32
    every Adam step counter = 1
    every exp_avg finite
    every exp_avg_sq finite
    parameter topology unchanged
    buffers unchanged
    parameters finite
    parameter values changed
    canonical state SHA256 changed
    RNG states unchanged

No second optimizer.step() is permitted.

No checkpoint is written.
No second batch is executed.
No full epoch is executed.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Frozen Phase-5.3.1l.2b implementation
# =============================================================================

PREFLIGHT_SOURCE_PATH = Path(
    "scripts/"
    "phase_5_3_1l_2b_corrected_adam_epoch0_first_batch_preflight.py"
)

PREFLIGHT_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1l_2b_corrected_adam_first_batch_preflight_contract.json"
)

PREFLIGHT_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1l_2b/"
    "phase_5_3_1l_2b_adam_batch_preflight_manifest.json"
)


# =============================================================================
# Frozen pre-step numerical fingerprints
# =============================================================================

EXPECTED_INITIAL_STATE_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_FIRST_BATCH_SHA256 = (
    "8408432b944bcd0805af9c34ff1b2db3"
    "ea938e0649a75d381b7839b86cd280ea"
)

EXPECTED_LOGIT_SHA256 = (
    "35b89aaed29d51d2ebb7ba1cadf2dc4b"
    "b5e8f81cf3aa78bc216b3cc6fed13845"
)

EXPECTED_GRADIENT_SHA256 = (
    "8c542430813d8ca91b8397409954ea92"
    "295a2b55bcc420661783fb865010845d"
)

EXPECTED_BATCH_LOSS = 0.7080879807

EXPECTED_PARAMETER_TENSORS = 32
EXPECTED_PARAMETER_COUNT = 19_217_929

REFERENCE_TORCH_VERSION_PREFIX = "2.7.0"


# =============================================================================
# Frozen Adam
# =============================================================================

ADAM_LR = 0.001
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.0
ADAM_AMSGRAD = False
ADAM_FOREACH = False
ADAM_FUSED = False
ADAM_MAXIMIZE = False
ADAM_CAPTURABLE = False
ADAM_DIFFERENTIABLE = False


# =============================================================================
# Outputs
# =============================================================================

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1m"
)

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

IMPORT_SAFETY_PATH = (
    AUDIT_DIR
    / "phase_5_3_1l_2b_import_safety_audit.csv"
)

PRE_STEP_PATH = (
    AUDIT_DIR
    / "first_adam_update_pre_step_integrity.csv"
)

PARAMETER_UPDATE_PATH = (
    AUDIT_DIR
    / "first_adam_parameter_update_audit.csv"
)

ADAM_STATE_PATH = (
    AUDIT_DIR
    / "first_adam_optimizer_state_audit.csv"
)

STATE_TRANSITION_PATH = (
    AUDIT_DIR
    / "first_adam_state_transition_audit.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_1m_final_invariants.csv"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_1m_first_adam_update_manifest.json"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1m_first_adam_weight_update_contract.json"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1m_first_adam_update_decision_register.csv"
)


# =============================================================================
# Helpers
# =============================================================================

def banner(text: str) -> None:

    print(
        "\n"
        + "=" * 118
    )

    print(text)

    print(
        "=" * 118
    )


def require(
    condition: bool,
    message: str,
) -> None:

    if not bool(condition):

        raise AssertionError(message)


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(handle)


def file_sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        while True:

            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(block)

    return digest.hexdigest()


def tensor_sha256(
    tensor: torch.Tensor,
) -> str:

    value = (
        tensor
        .detach()
        .cpu()
        .contiguous()
    )

    digest = hashlib.sha256()

    digest.update(
        str(
            value.dtype
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        str(
            tuple(
                value.shape
            )
        ).encode(
            "utf-8"
        )
    )

    digest.update(
        value.numpy().tobytes(
            order="C"
        )
    )

    return digest.hexdigest()


def parameter_hashes(
    model: torch.nn.Module,
) -> dict[str, str]:

    return {
        name: tensor_sha256(parameter)
        for (
            name,
            parameter,
        ) in model.named_parameters()
    }


def buffer_hashes(
    model: torch.nn.Module,
) -> dict[str, str]:

    return {
        name: tensor_sha256(buffer)
        for (
            name,
            buffer,
        ) in model.named_buffers()
    }


def parameter_spec(
    model: torch.nn.Module,
) -> list[tuple]:

    return [
        (
            name,
            tuple(parameter.shape),
            str(parameter.dtype),
            bool(parameter.requires_grad),
        )
        for (
            name,
            parameter,
        ) in model.named_parameters()
    ]


def buffer_spec(
    model: torch.nn.Module,
) -> list[tuple]:

    return [
        (
            name,
            tuple(buffer.shape),
            str(buffer.dtype),
        )
        for (
            name,
            buffer,
        ) in model.named_buffers()
    ]


def module_spec(
    model: torch.nn.Module,
) -> list[tuple]:

    return [
        (
            name,
            module.__class__.__module__,
            module.__class__.__name__,
        )
        for (
            name,
            module,
        ) in model.named_modules()
    ]


def numpy_rng_state_equal(
    left,
    right,
) -> bool:

    return (
        left[0] == right[0]
        and np.array_equal(
            left[1],
            right[1],
        )
        and left[2:] == right[2:]
    )


def gradient_logical_sha256(
    model: torch.nn.Module,
) -> str:

    digest = hashlib.sha256()

    for (
        name,
        parameter,
    ) in model.named_parameters():

        require(
            parameter.grad is not None,
            (
                "Cannot hash missing "
                f"gradient: {name}"
            ),
        )

        digest.update(
            name.encode(
                "utf-8"
            )
        )

        digest.update(b"\0")

        digest.update(
            tensor_sha256(
                parameter.grad
            ).encode(
                "ascii"
            )
        )

        digest.update(b"\0")

    return digest.hexdigest()


def optimizer_state_logical_sha256(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> str:

    digest = hashlib.sha256()

    for (
        name,
        parameter,
    ) in model.named_parameters():

        require(
            parameter in optimizer.state,
            (
                "Missing Adam state for "
                f"{name}."
            ),
        )

        state = optimizer.state[
            parameter
        ]

        digest.update(
            name.encode(
                "utf-8"
            )
        )

        digest.update(b"\0")

        for key in (
            "step",
            "exp_avg",
            "exp_avg_sq",
        ):

            require(
                key in state,
                (
                    f"Adam state for {name} "
                    f"missing {key}."
                ),
            )

            digest.update(
                key.encode(
                    "utf-8"
                )
            )

            digest.update(b"\0")

            value = state[
                key
            ]

            require(
                isinstance(
                    value,
                    torch.Tensor,
                ),
                (
                    f"Adam state {name}.{key} "
                    "is not Tensor."
                ),
            )

            digest.update(
                tensor_sha256(
                    value
                ).encode(
                    "ascii"
                )
            )

            digest.update(b"\0")

    group = optimizer.param_groups[
        0
    ]

    frozen_group_fields = (
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "foreach",
        "maximize",
        "capturable",
        "differentiable",
        "fused",
    )

    for key in frozen_group_fields:

        digest.update(
            key.encode(
                "utf-8"
            )
        )

        digest.update(b"\0")

        digest.update(
            repr(
                group.get(key)
            ).encode(
                "utf-8"
            )
        )

        digest.update(b"\0")

    return digest.hexdigest()


# =============================================================================
# Static import-safety audit for Phase-5.3.1l.2b
# =============================================================================

def is_main_guard(
    node: ast.AST,
) -> bool:

    if not isinstance(
        node,
        ast.If,
    ):

        return False

    test = node.test

    if not isinstance(
        test,
        ast.Compare,
    ):

        return False

    if not (
        isinstance(
            test.left,
            ast.Name,
        )
        and test.left.id == "__name__"
    ):

        return False

    if len(test.ops) != 1:

        return False

    if not isinstance(
        test.ops[0],
        ast.Eq,
    ):

        return False

    if len(test.comparators) != 1:

        return False

    comparator = test.comparators[
        0
    ]

    return (
        isinstance(
            comparator,
            ast.Constant,
        )
        and comparator.value == "__main__"
    )


def direct_call_name(
    call: ast.Call,
) -> str:

    current = call.func

    parts = []

    while isinstance(
        current,
        ast.Attribute,
    ):

        parts.append(
            current.attr
        )

        current = current.value

    if isinstance(
        current,
        ast.Name,
    ):

        parts.append(
            current.id
        )

    return ".".join(
        reversed(parts)
    )


def audit_import_safety(
    source_path: Path,
):

    source = source_path.read_text(
        encoding="utf-8"
    )

    tree = ast.parse(
        source,
        filename=str(
            source_path
        ),
    )

    main_guards = [
        node
        for node in tree.body
        if is_main_guard(node)
    ]

    require(
        len(main_guards) == 1,
        (
            "Expected exactly one "
            "__main__ guard in 5.3.1l.2b."
        ),
    )

    unguarded_main_calls = 0

    unguarded_step_calls = 0

    unguarded_write_calls = 0

    allowed_top_level_types = (
        ast.Expr,
        ast.Import,
        ast.ImportFrom,
        ast.Assign,
        ast.AnnAssign,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.If,
    )

    unexpected_top_level = []

    write_suffixes = {
        "write_text",
        "write_bytes",
        "to_csv",
        "to_parquet",
        "save",
        "savez",
        "mkdir",
        "unlink",
        "rename",
        "replace",
    }

    for node in tree.body:

        if not isinstance(
            node,
            allowed_top_level_types,
        ):

            unexpected_top_level.append(
                type(node).__name__
            )

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):

            continue

        if is_main_guard(node):

            continue

        for candidate in ast.walk(node):

            if not isinstance(
                candidate,
                ast.Call,
            ):

                continue

            name = direct_call_name(
                candidate
            )

            if name == "main":

                unguarded_main_calls += 1

            if (
                name == "optimizer.step"
                or name.endswith(
                    ".optimizer.step"
                )
                or name.endswith(
                    ".step"
                )
                and "optimizer" in name
            ):

                unguarded_step_calls += 1

            final_component = (
                name.split(".")[-1]
                if name
                else ""
            )

            if final_component in (
                write_suffixes
            ):

                unguarded_write_calls += 1

    # Separately audit EVERY actual optimizer.step() call anywhere in source.
    all_optimizer_step_calls = []

    for candidate in ast.walk(tree):

        if not isinstance(
            candidate,
            ast.Call,
        ):

            continue

        name = direct_call_name(
            candidate
        )

        if (
            name == "optimizer.step"
            or name.endswith(
                ".optimizer.step"
            )
        ):

            all_optimizer_step_calls.append(
                int(
                    candidate.lineno
                )
            )

    require(
        len(
            unexpected_top_level
        )
        == 0,
        (
            "Unexpected top-level statement "
            "in 5.3.1l.2b: "
            f"{unexpected_top_level}"
        ),
    )

    require(
        unguarded_main_calls == 0,
        (
            "Unguarded main() call detected."
        ),
    )

    require(
        unguarded_step_calls == 0,
        (
            "Unguarded optimizer.step() "
            "detected."
        ),
    )

    require(
        unguarded_write_calls == 0,
        (
            "Import-time write-like call "
            "detected."
        ),
    )

    require(
        len(
            all_optimizer_step_calls
        )
        == 0,
        (
            "Phase-5.3.1l.2b source itself "
            "contains optimizer.step()."
        ),
    )

    audit_df = pd.DataFrame(
        [
            {
                "check": (
                    "main_guard_count"
                ),
                "actual": (
                    len(
                        main_guards
                    )
                ),
                "expected": (
                    1
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "unguarded_main_calls"
                ),
                "actual": (
                    unguarded_main_calls
                ),
                "expected": (
                    0
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "unguarded_optimizer_step_calls"
                ),
                "actual": (
                    unguarded_step_calls
                ),
                "expected": (
                    0
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "optimizer_step_calls_anywhere"
                ),
                "actual": (
                    len(
                        all_optimizer_step_calls
                    )
                ),
                "expected": (
                    0
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "unguarded_write_like_calls"
                ),
                "actual": (
                    unguarded_write_calls
                ),
                "expected": (
                    0
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "unexpected_top_level_statements"
                ),
                "actual": (
                    len(
                        unexpected_top_level
                    )
                ),
                "expected": (
                    0
                ),
                "status": (
                    "PASS"
                ),
            },
        ]
    )

    return (
        tree,
        audit_df,
    )


# =============================================================================
# Safe module loader
# =============================================================================

def load_preflight_module(
    source_path: Path,
):

    module_name = (
        "_itrs_phase5_3_1m_captured_preflight"
    )

    spec = importlib.util.spec_from_file_location(
        module_name,
        source_path,
    )

    require(
        spec is not None,
        (
            "Could not create import spec "
            "for 5.3.1l.2b."
        ),
    )

    require(
        spec.loader is not None,
        (
            "5.3.1l.2b import spec "
            "has no loader."
        ),
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


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.1m — "
        "FIRST ADAM WEIGHT-UPDATE PROOF"
    )

    print(
        "Frozen first batch:                   YES"
    )

    print(
        "Pre-step runtime reproduced:          YES"
    )

    print(
        "Adam instantiated:                    YES"
    )

    print(
        "Forward/BCE/backward reproduced:      YES"
    )

    print(
        "optimizer.step() allowed:             EXACTLY ONCE"
    )

    print(
        "Full epoch executed:                  NO"
    )

    print(
        "Checkpoint written:                   NO"
    )

    # =========================================================================
    # Authoritative prerequisites
    # =========================================================================

    banner(
        "AUTHORITATIVE PRE-STEP CONTRACT RECHECK"
    )

    for path in (
        PREFLIGHT_SOURCE_PATH,
        PREFLIGHT_CONTRACT_PATH,
        PREFLIGHT_MANIFEST_PATH,
    ):

        require(
            path.exists(),
            (
                "Missing authoritative input: "
                f"{path}"
            ),
        )

        print(
            f"FOUND  {path}"
        )

    preflight_contract = load_json(
        PREFLIGHT_CONTRACT_PATH
    )

    preflight_manifest = load_json(
        PREFLIGHT_MANIFEST_PATH
    )

    require(
        preflight_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.1l.2b contract "
            "is not frozen."
        ),
    )

    require(
        preflight_manifest[
            "status"
        ]
        == (
            "CORRECTED_ADAM_EPOCH0_FIRST_BATCH_"
            "FORWARD_BACKWARD_PREFLIGHT_PASSED"
        ),
        (
            "Unexpected 5.3.1l.2b "
            "manifest status."
        ),
    )

    require(
        preflight_manifest[
            "first_batch_sha256"
        ]
        == EXPECTED_FIRST_BATCH_SHA256,
        (
            "Frozen first-batch "
            "fingerprint changed."
        ),
    )

    require(
        preflight_manifest[
            "logit_sha256"
        ]
        == EXPECTED_LOGIT_SHA256,
        (
            "Frozen logit fingerprint "
            "changed."
        ),
    )

    require(
        preflight_manifest[
            "gradient_sha256"
        ]
        == EXPECTED_GRADIENT_SHA256,
        (
            "Frozen gradient fingerprint "
            "changed."
        ),
    )

    require(
        preflight_manifest[
            "canonical_state_sha256_after_backward"
        ]
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Frozen pre-step parameter state "
            "changed."
        ),
    )

    require(
        preflight_manifest[
            "Adam_instantiated"
        ]
        is True,
        (
            "5.3.1l.2b did not instantiate Adam."
        ),
    )

    require(
        int(
            preflight_manifest[
                "Adam_state_entries"
            ]
        )
        == 0,
        (
            "Adam state existed before "
            "first update."
        ),
    )

    require(
        int(
            preflight_manifest[
                "optimizer_steps"
            ]
        )
        == 0,
        (
            "Optimizer step occurred before "
            "Phase-5.3.1m."
        ),
    )

    preflight_source_sha = file_sha256(
        PREFLIGHT_SOURCE_PATH
    )

    print(
        "Phase-5.3.1l.2b:                     FROZEN / PASS"
    )

    print(
        "Frozen pre-step canonical SHA256:"
    )

    print(
        EXPECTED_INITIAL_STATE_SHA256
    )

    print()

    print(
        "Frozen logit SHA256:"
    )

    print(
        EXPECTED_LOGIT_SHA256
    )

    print()

    print(
        "Frozen gradient SHA256:"
    )

    print(
        EXPECTED_GRADIENT_SHA256
    )

    print()

    print(
        "Phase-5.3.1l.2b source SHA256:"
    )

    print(
        preflight_source_sha
    )

    # =========================================================================
    # Reference runtime
    # =========================================================================

    banner(
        "REFERENCE RUNTIME"
    )

    print(
        f"PyTorch:                              "
        f"{torch.__version__}"
    )

    print(
        f"NumPy:                                "
        f"{np.__version__}"
    )

    require(
        torch.__version__.startswith(
            REFERENCE_TORCH_VERSION_PREFIX
        ),
        (
            "Reference runtime requires "
            "PyTorch 2.7.0."
        ),
    )

    # =========================================================================
    # Static import-safety audit
    # =========================================================================

    banner(
        "PHASE-5.3.1l.2b IMPORT-SAFETY AUDIT"
    )

    (
        _,
        import_safety_df,
    ) = audit_import_safety(
        PREFLIGHT_SOURCE_PATH
    )

    require(
        bool(
            (
                import_safety_df[
                    "status"
                ]
                == "PASS"
            ).all()
        ),
        (
            "5.3.1l.2b import-safety "
            "audit failed."
        ),
    )

    print(
        import_safety_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Load guarded preflight module
    # =========================================================================

    banner(
        "LOAD FROZEN PRE-STEP RUNTIME"
    )

    preflight = load_preflight_module(
        PREFLIGHT_SOURCE_PATH
    )

    require(
        hasattr(
            preflight,
            "main",
        ),
        (
            "Imported preflight module "
            "has no main()."
        ),
    )

    require(
        hasattr(
            preflight,
            "compose_canonical_model",
        ),
        (
            "Imported preflight module "
            "has no compose_canonical_model()."
        ),
    )

    require(
        hasattr(
            preflight,
            "build_frozen_adam",
        ),
        (
            "Imported preflight module "
            "has no build_frozen_adam()."
        ),
    )

    require(
        hasattr(
            preflight,
            "gradient_logical_sha256",
        ),
        (
            "Imported preflight module "
            "has no gradient hash helper."
        ),
    )

    # =========================================================================
    # Capture model / Adam produced by exact frozen preflight
    # =========================================================================

    capture = {}

    original_compose = (
        preflight
        .compose_canonical_model
    )

    original_build_adam = (
        preflight
        .build_frozen_adam
    )

    def captured_compose(
        *args,
        **kwargs,
    ):

        (
            model,
            hash_fn,
        ) = original_compose(
            *args,
            **kwargs,
        )

        capture[
            "model"
        ] = model

        capture[
            "hash_fn"
        ] = hash_fn

        return (
            model,
            hash_fn,
        )

    def captured_build_adam(
        *args,
        **kwargs,
    ):

        optimizer = original_build_adam(
            *args,
            **kwargs,
        )

        capture[
            "optimizer"
        ] = optimizer

        return optimizer

    preflight.compose_canonical_model = (
        captured_compose
    )

    preflight.build_frozen_adam = (
        captured_build_adam
    )

    # =========================================================================
    # Re-execute exact frozen pre-step runtime
    # =========================================================================

    banner(
        "REPRODUCE EXACT PHASE-5.3.1l.2b PRE-STEP STATE"
    )

    preflight.main()

    require(
        "model"
        in capture,
        (
            "Failed to capture canonical model "
            "from 5.3.1l.2b."
        ),
    )

    require(
        "optimizer"
        in capture,
        (
            "Failed to capture Adam optimizer "
            "from 5.3.1l.2b."
        ),
    )

    require(
        "hash_fn"
        in capture,
        (
            "Failed to capture canonical "
            "state-hash function."
        ),
    )

    model = capture[
        "model"
    ]

    optimizer = capture[
        "optimizer"
    ]

    canonical_hash_fn = capture[
        "hash_fn"
    ]

    # Re-read freshly regenerated manifest.
    reproduced_manifest = load_json(
        PREFLIGHT_MANIFEST_PATH
    )

    require(
        reproduced_manifest[
            "logit_sha256"
        ]
        == EXPECTED_LOGIT_SHA256,
        (
            "Re-executed preflight logit "
            "fingerprint differs."
        ),
    )

    require(
        reproduced_manifest[
            "gradient_sha256"
        ]
        == EXPECTED_GRADIENT_SHA256,
        (
            "Re-executed preflight gradient "
            "fingerprint differs."
        ),
    )

    reproduced_loss = float(
        reproduced_manifest[
            "batch_loss"
        ]
    )

    require(
        abs(
            reproduced_loss
            - EXPECTED_BATCH_LOSS
        )
        <= 1e-10,
        (
            "Re-executed first-batch loss "
            "differs from frozen value."
        ),
    )

    # =========================================================================
    # Exact pre-step state validation
    # =========================================================================

    banner(
        "FIRST ADAM UPDATE — PRE-STEP STATE GATE"
    )

    pre_step_hash = canonical_hash_fn(
        model
    )

    require(
        pre_step_hash
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Captured model is not at "
            "canonical pre-step state."
        ),
    )

    trainable_parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    require(
        len(
            trainable_parameters
        )
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Trainable tensor count changed."
        ),
    )

    require(
        sum(
            int(
                parameter.numel()
            )
            for parameter
            in trainable_parameters
        )
        == EXPECTED_PARAMETER_COUNT,
        (
            "Trainable parameter count changed."
        ),
    )

    gradient_sha_before_step = (
        gradient_logical_sha256(
            model
        )
    )

    require(
        gradient_sha_before_step
        == EXPECTED_GRADIENT_SHA256,
        (
            "Captured gradients do not match "
            "frozen pre-step gradient fingerprint."
        ),
    )

    require(
        len(
            optimizer.state
        )
        == 0,
        (
            "Adam state must be empty "
            "immediately before first step."
        ),
    )

    require(
        len(
            optimizer.param_groups
        )
        == 1,
        (
            "Expected one Adam "
            "parameter group."
        ),
    )

    group = optimizer.param_groups[
        0
    ]

    require(
        float(
            group[
                "lr"
            ]
        )
        == ADAM_LR,
        (
            "Adam lr drift."
        ),
    )

    require(
        tuple(
            group[
                "betas"
            ]
        )
        == ADAM_BETAS,
        (
            "Adam betas drift."
        ),
    )

    require(
        float(
            group[
                "eps"
            ]
        )
        == ADAM_EPS,
        (
            "Adam eps drift."
        ),
    )

    require(
        float(
            group[
                "weight_decay"
            ]
        )
        == ADAM_WEIGHT_DECAY,
        (
            "Adam weight decay drift."
        ),
    )

    require(
        bool(
            group[
                "amsgrad"
            ]
        )
        == ADAM_AMSGRAD,
        (
            "Adam AMSGrad drift."
        ),
    )

    require(
        group.get(
            "foreach"
        )
        is ADAM_FOREACH,
        (
            "Adam foreach drift."
        ),
    )

    require(
        group.get(
            "fused"
        )
        is ADAM_FUSED,
        (
            "Adam fused drift."
        ),
    )

    require(
        bool(
            group.get(
                "maximize"
            )
        )
        == ADAM_MAXIMIZE,
        (
            "Adam maximize drift."
        ),
    )

    require(
        bool(
            group.get(
                "capturable"
            )
        )
        == ADAM_CAPTURABLE,
        (
            "Adam capturable drift."
        ),
    )

    require(
        bool(
            group.get(
                "differentiable"
            )
        )
        == ADAM_DIFFERENTIABLE,
        (
            "Adam differentiable drift."
        ),
    )

    optimizer_parameters = group[
        "params"
    ]

    require(
        len(
            optimizer_parameters
        )
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Adam parameter tensor count changed."
        ),
    )

    require(
        all(
            left is right
            for (
                left,
                right,
            ) in zip(
                optimizer_parameters,
                trainable_parameters,
            )
        ),
        (
            "Adam parameter identity/order "
            "does not match canonical model."
        ),
    )

    # Complete structural snapshot.
    parameter_spec_before = (
        parameter_spec(
            model
        )
    )

    buffer_spec_before = (
        buffer_spec(
            model
        )
    )

    module_spec_before = (
        module_spec(
            model
        )
    )

    buffer_hashes_before = (
        buffer_hashes(
            model
        )
    )

    parameter_hashes_before = (
        parameter_hashes(
            model
        )
    )

    # Numerical snapshots for update audit.
    parameter_values_before = {
        name: (
            parameter
            .detach()
            .clone()
        )
        for (
            name,
            parameter,
        ) in model.named_parameters()
    }

    gradient_values_before = {
        name: (
            parameter
            .grad
            .detach()
            .clone()
        )
        for (
            name,
            parameter,
        ) in model.named_parameters()
    }

    pre_step_df = pd.DataFrame(
        [
            {
                "check": (
                    "canonical_state_sha256"
                ),
                "actual": (
                    pre_step_hash
                ),
                "expected": (
                    EXPECTED_INITIAL_STATE_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "logit_sha256"
                ),
                "actual": (
                    reproduced_manifest[
                        "logit_sha256"
                    ]
                ),
                "expected": (
                    EXPECTED_LOGIT_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "gradient_sha256"
                ),
                "actual": (
                    gradient_sha_before_step
                ),
                "expected": (
                    EXPECTED_GRADIENT_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "batch_loss"
                ),
                "actual": (
                    reproduced_loss
                ),
                "expected": (
                    EXPECTED_BATCH_LOSS
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "Adam_state_entries"
                ),
                "actual": (
                    len(
                        optimizer.state
                    )
                ),
                "expected": (
                    0
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "parameter_tensors"
                ),
                "actual": (
                    len(
                        trainable_parameters
                    )
                ),
                "expected": (
                    EXPECTED_PARAMETER_TENSORS
                ),
                "status": (
                    "PASS"
                ),
            },
        ]
    )

    print(
        pre_step_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # FIRST AND ONLY optimizer.step()
    # =========================================================================

    banner(
        "EXECUTE FIRST ADAM WEIGHT UPDATE"
    )

    optimizer_step_count = 0

    python_rng_before_step = (
        random.getstate()
    )

    numpy_rng_before_step = (
        np.random.get_state()
    )

    torch_rng_before_step = (
        torch.get_rng_state().clone()
    )

    # =====================================================================
    # THE FIRST ACTUAL PARAMETER UPDATE OF THE ENTIRE REPRODUCTION
    # =====================================================================

    optimizer.step()

    optimizer_step_count += 1

    # =====================================================================
    # NO SECOND optimizer.step() IS PERMITTED BELOW THIS LINE
    # =====================================================================

    require(
        optimizer_step_count == 1,
        (
            "First Adam proof must execute "
            "exactly one optimizer.step()."
        ),
    )

    python_rng_after_step = (
        random.getstate()
    )

    numpy_rng_after_step = (
        np.random.get_state()
    )

    torch_rng_after_step = (
        torch.get_rng_state().clone()
    )

    # =========================================================================
    # Post-step parameter-state audit
    # =========================================================================

    banner(
        "POST-FIRST-STEP PARAMETER TRANSITION"
    )

    post_step_hash = canonical_hash_fn(
        model
    )

    require(
        post_step_hash
        != EXPECTED_INITIAL_STATE_SHA256,
        (
            "Parameter-state hash did not change "
            "after first optimizer.step()."
        ),
    )

    parameter_spec_after = (
        parameter_spec(
            model
        )
    )

    buffer_spec_after = (
        buffer_spec(
            model
        )
    )

    module_spec_after = (
        module_spec(
            model
        )
    )

    buffer_hashes_after = (
        buffer_hashes(
            model
        )
    )

    parameter_hashes_after = (
        parameter_hashes(
            model
        )
    )

    require(
        parameter_spec_after
        == parameter_spec_before,
        (
            "Parameter topology changed "
            "during Adam step."
        ),
    )

    require(
        buffer_spec_after
        == buffer_spec_before,
        (
            "Buffer topology changed "
            "during Adam step."
        ),
    )

    require(
        module_spec_after
        == module_spec_before,
        (
            "Module topology changed "
            "during Adam step."
        ),
    )

    require(
        buffer_hashes_after
        == buffer_hashes_before,
        (
            "Non-parameter buffers changed "
            "during optimizer.step()."
        ),
    )

    require(
        bool(
            all(
                torch.isfinite(
                    parameter
                ).all()
                for parameter
                in model.parameters()
            )
        ),
        (
            "Non-finite model parameter "
            "after first Adam step."
        ),
    )

    # Gradients should remain exactly the same until zero_grad().
    gradient_sha_after_step = (
        gradient_logical_sha256(
            model
        )
    )

    require(
        gradient_sha_after_step
        == EXPECTED_GRADIENT_SHA256,
        (
            "optimizer.step() unexpectedly "
            "modified gradients."
        ),
    )

    # Adam should consume no RNG.
    require(
        python_rng_before_step
        == python_rng_after_step,
        (
            "optimizer.step() changed "
            "Python RNG state."
        ),
    )

    require(
        numpy_rng_state_equal(
            numpy_rng_before_step,
            numpy_rng_after_step,
        ),
        (
            "optimizer.step() changed "
            "NumPy RNG state."
        ),
    )

    require(
        torch.equal(
            torch_rng_before_step,
            torch_rng_after_step,
        ),
        (
            "optimizer.step() changed "
            "torch RNG state."
        ),
    )

    # =========================================================================
    # Per-parameter update diagnostics
    # =========================================================================

    banner(
        "PER-PARAMETER FIRST-STEP UPDATE AUDIT"
    )

    update_rows = []

    changed_tensor_count = 0

    for (
        name,
        parameter,
    ) in model.named_parameters():

        before = parameter_values_before[
            name
        ]

        after = parameter.detach()

        gradient = gradient_values_before[
            name
        ]

        changed = not torch.equal(
            before,
            after,
        )

        if changed:
            changed_tensor_count += 1

        delta = (
            after
            - before
        )

        delta_abs = (
            delta
            .abs()
        )

        delta_abs_sum = float(
            delta_abs.sum()
        )

        delta_abs_max = float(
            delta_abs.max()
        )

        parameter_finite = bool(
            torch.isfinite(
                after
            ).all()
        )

        gradient_finite = bool(
            torch.isfinite(
                gradient
            ).all()
        )

        # Adam's first update should move opposite the gradient.
        update_dot_gradient = float(
            (
                delta.double()
                * gradient.double()
            ).sum()
        )

        opposite_gradient_direction = (
            update_dot_gradient
            < 0.0
        )

        update_rows.append(
            {
                "parameter": (
                    name
                ),

                "shape": (
                    str(
                        tuple(
                            parameter.shape
                        )
                    )
                ),

                "parameter_changed": (
                    changed
                ),

                "parameter_finite": (
                    parameter_finite
                ),

                "gradient_finite": (
                    gradient_finite
                ),

                "delta_abs_sum": (
                    delta_abs_sum
                ),

                "delta_abs_max": (
                    delta_abs_max
                ),

                "delta_dot_gradient": (
                    update_dot_gradient
                ),

                "opposite_gradient_direction": (
                    opposite_gradient_direction
                ),

                "pre_sha256": (
                    parameter_hashes_before[
                        name
                    ]
                ),

                "post_sha256": (
                    parameter_hashes_after[
                        name
                    ]
                ),

                "status": (
                    "PASS"
                    if (
                        changed
                        and parameter_finite
                        and gradient_finite
                        and delta_abs_sum > 0.0
                        and opposite_gradient_direction
                    )
                    else "FAIL"
                ),
            }
        )

    update_df = pd.DataFrame(
        update_rows
    )

    require(
        changed_tensor_count
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Expected all 32 trainable tensors "
            "to change on first Adam step."
        ),
    )

    require(
        bool(
            (
                update_df[
                    "status"
                ]
                == "PASS"
            ).all()
        ),
        (
            "At least one parameter tensor failed "
            "the first-step update audit."
        ),
    )

    print(
        update_df[
            [
                "parameter",
                "parameter_changed",
                "delta_abs_sum",
                "delta_abs_max",
                "delta_dot_gradient",
                "status",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # Adam first-state initialization audit
    # =========================================================================

    banner(
        "ADAM FIRST-STEP STATE INITIALIZATION"
    )

    require(
        len(
            optimizer.state
        )
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Adam should have exactly 32 "
            "state entries after first step."
        ),
    )

    beta1 = ADAM_BETAS[
        0
    ]

    beta2 = ADAM_BETAS[
        1
    ]

    adam_rows = []

    for (
        name,
        parameter,
    ) in model.named_parameters():

        require(
            parameter in optimizer.state,
            (
                "Missing Adam state for "
                f"{name}."
            ),
        )

        state = optimizer.state[
            parameter
        ]

        expected_keys = {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }

        actual_keys = set(
            state.keys()
        )

        require(
            actual_keys
            == expected_keys,
            (
                f"Unexpected Adam state keys "
                f"for {name}: {actual_keys}"
            ),
        )

        step = state[
            "step"
        ]

        exp_avg = state[
            "exp_avg"
        ]

        exp_avg_sq = state[
            "exp_avg_sq"
        ]

        require(
            isinstance(
                step,
                torch.Tensor,
            ),
            (
                f"{name}: Adam step "
                "is not Tensor."
            ),
        )

        require(
            float(
                step.item()
            )
            == 1.0,
            (
                f"{name}: Adam step "
                "counter != 1."
            ),
        )

        require(
            exp_avg.shape
            == parameter.shape,
            (
                f"{name}: exp_avg "
                "shape mismatch."
            ),
        )

        require(
            exp_avg_sq.shape
            == parameter.shape,
            (
                f"{name}: exp_avg_sq "
                "shape mismatch."
            ),
        )

        require(
            bool(
                torch.isfinite(
                    exp_avg
                ).all()
            ),
            (
                f"{name}: exp_avg "
                "contains non-finite values."
            ),
        )

        require(
            bool(
                torch.isfinite(
                    exp_avg_sq
                ).all()
            ),
            (
                f"{name}: exp_avg_sq "
                "contains non-finite values."
            ),
        )

        gradient = gradient_values_before[
            name
        ]

        expected_exp_avg = (
            gradient
            * (
                1.0
                - beta1
            )
        )

        expected_exp_avg_sq = (
            gradient.square()
            * (
                1.0
                - beta2
            )
        )

        exp_avg_matches = torch.allclose(
            exp_avg,
            expected_exp_avg,
            rtol=1e-6,
            atol=1e-9,
        )

        exp_avg_sq_matches = torch.allclose(
            exp_avg_sq,
            expected_exp_avg_sq,
            rtol=1e-6,
            atol=1e-12,
        )

        require(
            exp_avg_matches,
            (
                f"{name}: first Adam exp_avg "
                "does not match (1-beta1)*gradient."
            ),
        )

        require(
            exp_avg_sq_matches,
            (
                f"{name}: first Adam exp_avg_sq "
                "does not match (1-beta2)*gradient^2."
            ),
        )

        adam_rows.append(
            {
                "parameter": (
                    name
                ),

                "state_keys": (
                    ";".join(
                        sorted(
                            actual_keys
                        )
                    )
                ),

                "step": (
                    float(
                        step.item()
                    )
                ),

                "exp_avg_shape": (
                    str(
                        tuple(
                            exp_avg.shape
                        )
                    )
                ),

                "exp_avg_sq_shape": (
                    str(
                        tuple(
                            exp_avg_sq.shape
                        )
                    )
                ),

                "exp_avg_finite": (
                    True
                ),

                "exp_avg_sq_finite": (
                    True
                ),

                "exp_avg_matches_first_step_formula": (
                    exp_avg_matches
                ),

                "exp_avg_sq_matches_first_step_formula": (
                    exp_avg_sq_matches
                ),

                "status": (
                    "PASS"
                ),
            }
        )

    adam_state_df = pd.DataFrame(
        adam_rows
    )

    optimizer_state_sha = (
        optimizer_state_logical_sha256(
            model,
            optimizer,
        )
    )

    require(
        bool(
            (
                adam_state_df[
                    "status"
                ]
                == "PASS"
            ).all()
        ),
        (
            "Adam first-state audit failed."
        ),
    )

    print(
        "Adam state entries:                   "
        f"{len(optimizer.state)} / "
        f"{EXPECTED_PARAMETER_TENSORS}"
    )

    print(
        "Adam step counters:                   "
        "1 / all parameters"
    )

    print(
        "exp_avg first-step formula:           PASS"
    )

    print(
        "exp_avg_sq first-step formula:        PASS"
    )

    print()

    print(
        "Optimizer-state logical SHA256:"
    )

    print(
        optimizer_state_sha
    )

    # =========================================================================
    # State transition
    # =========================================================================

    banner(
        "FIRST TRAINING STATE TRANSITION"
    )

    transition_df = pd.DataFrame(
        [
            {
                "check": (
                    "pre_step_parameter_state"
                ),
                "value": (
                    pre_step_hash
                ),
                "expected": (
                    EXPECTED_INITIAL_STATE_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "post_step_parameter_state"
                ),
                "value": (
                    post_step_hash
                ),
                "expected": (
                    "DIFFERENT_FROM_PRE_STEP"
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "parameter_state_changed"
                ),
                "value": (
                    str(
                        post_step_hash
                        != pre_step_hash
                    )
                ),
                "expected": (
                    "True"
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "parameter_tensors_changed"
                ),
                "value": (
                    str(
                        changed_tensor_count
                    )
                ),
                "expected": (
                    str(
                        EXPECTED_PARAMETER_TENSORS
                    )
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "Adam_state_entries"
                ),
                "value": (
                    str(
                        len(
                            optimizer.state
                        )
                    )
                ),
                "expected": (
                    str(
                        EXPECTED_PARAMETER_TENSORS
                    )
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "optimizer_step_count"
                ),
                "value": (
                    str(
                        optimizer_step_count
                    )
                ),
                "expected": (
                    "1"
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "gradient_sha256_after_step"
                ),
                "value": (
                    gradient_sha_after_step
                ),
                "expected": (
                    EXPECTED_GRADIENT_SHA256
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "optimizer_step_RNG_neutral"
                ),
                "value": (
                    "True"
                ),
                "expected": (
                    "True"
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "parameter_topology_unchanged"
                ),
                "value": (
                    str(
                        parameter_spec_after
                        == parameter_spec_before
                    )
                ),
                "expected": (
                    "True"
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "module_topology_unchanged"
                ),
                "value": (
                    str(
                        module_spec_after
                        == module_spec_before
                    )
                ),
                "expected": (
                    "True"
                ),
                "status": (
                    "PASS"
                ),
            },

            {
                "check": (
                    "buffers_unchanged"
                ),
                "value": (
                    str(
                        buffer_hashes_after
                        == buffer_hashes_before
                    )
                ),
                "expected": (
                    "True"
                ),
                "status": (
                    "PASS"
                ),
            },
        ]
    )

    print(
        transition_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Canonical pre-step SHA256:"
    )

    print(
        pre_step_hash
    )

    print()

    print(
        "Canonical POST-FIRST-STEP SHA256:"
    )

    print(
        post_step_hash
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.1m INVARIANTS"
    )

    checks = [
        (
            "phase_5_3_1l_2b_contract_frozen",
            (
                preflight_contract[
                    "status"
                ]
                == "FROZEN"
            ),
        ),

        (
            "preflight_source_import_safe",
            bool(
                (
                    import_safety_df[
                        "status"
                    ]
                    == "PASS"
                ).all()
            ),
        ),

        (
            "preflight_source_contains_zero_optimizer_steps",
            (
                int(
                    import_safety_df.loc[
                        import_safety_df[
                            "check"
                        ]
                        == (
                            "optimizer_step_calls_anywhere"
                        ),
                        "actual",
                    ].iloc[0]
                )
                == 0
            ),
        ),

        (
            "first_batch_fingerprint_exact",
            (
                reproduced_manifest[
                    "first_batch_sha256"
                ]
                == EXPECTED_FIRST_BATCH_SHA256
            ),
        ),

        (
            "pre_step_logit_fingerprint_exact",
            (
                reproduced_manifest[
                    "logit_sha256"
                ]
                == EXPECTED_LOGIT_SHA256
            ),
        ),

        (
            "pre_step_gradient_fingerprint_exact",
            (
                gradient_sha_before_step
                == EXPECTED_GRADIENT_SHA256
            ),
        ),

        (
            "pre_step_loss_exact",
            (
                abs(
                    reproduced_loss
                    - EXPECTED_BATCH_LOSS
                )
                <= 1e-10
            ),
        ),

        (
            "pre_step_canonical_state_exact",
            (
                pre_step_hash
                == EXPECTED_INITIAL_STATE_SHA256
            ),
        ),

        (
            "Adam_state_empty_before_first_step",
            (
                len(
                    preflight_manifest.get(
                        "Adam_state_entries",
                        0,
                    )
                )
                == 0
                if isinstance(
                    preflight_manifest.get(
                        "Adam_state_entries",
                        0,
                    ),
                    list,
                )
                else int(
                    preflight_manifest[
                        "Adam_state_entries"
                    ]
                )
                == 0
            ),
        ),

        (
            "exactly_one_optimizer_step_executed",
            (
                optimizer_step_count
                == 1
            ),
        ),

        (
            "parameter_state_hash_changed",
            (
                post_step_hash
                != pre_step_hash
            ),
        ),

        (
            "all_32_parameter_tensors_changed",
            (
                changed_tensor_count
                == EXPECTED_PARAMETER_TENSORS
            ),
        ),

        (
            "all_parameter_updates_finite",
            bool(
                update_df[
                    "parameter_finite"
                ].all()
            ),
        ),

        (
            "all_parameter_updates_opposite_gradient",
            bool(
                update_df[
                    "opposite_gradient_direction"
                ].all()
            ),
        ),

        (
            "parameter_topology_unchanged",
            (
                parameter_spec_after
                == parameter_spec_before
            ),
        ),

        (
            "buffer_topology_unchanged",
            (
                buffer_spec_after
                == buffer_spec_before
            ),
        ),

        (
            "module_topology_unchanged",
            (
                module_spec_after
                == module_spec_before
            ),
        ),

        (
            "buffer_values_unchanged",
            (
                buffer_hashes_after
                == buffer_hashes_before
            ),
        ),

        (
            "Adam_state_entries_exactly_32",
            (
                len(
                    optimizer.state
                )
                == EXPECTED_PARAMETER_TENSORS
            ),
        ),

        (
            "all_Adam_step_counters_equal_one",
            bool(
                (
                    adam_state_df[
                        "step"
                    ]
                    == 1.0
                ).all()
            ),
        ),

        (
            "all_exp_avg_finite",
            bool(
                adam_state_df[
                    "exp_avg_finite"
                ].all()
            ),
        ),

        (
            "all_exp_avg_sq_finite",
            bool(
                adam_state_df[
                    "exp_avg_sq_finite"
                ].all()
            ),
        ),

        (
            "first_step_exp_avg_formula_verified",
            bool(
                adam_state_df[
                    "exp_avg_matches_first_step_formula"
                ].all()
            ),
        ),

        (
            "first_step_exp_avg_sq_formula_verified",
            bool(
                adam_state_df[
                    "exp_avg_sq_matches_first_step_formula"
                ].all()
            ),
        ),

        (
            "gradients_unchanged_by_optimizer_step",
            (
                gradient_sha_after_step
                == EXPECTED_GRADIENT_SHA256
            ),
        ),

        (
            "optimizer_step_rng_neutral",
            (
                python_rng_before_step
                == python_rng_after_step
                and numpy_rng_state_equal(
                    numpy_rng_before_step,
                    numpy_rng_after_step,
                )
                and torch.equal(
                    torch_rng_before_step,
                    torch_rng_after_step,
                )
            ),
        ),

        (
            "no_checkpoint_written",
            True,
        ),

        (
            "no_second_batch_executed",
            True,
        ),

        (
            "no_full_epoch_executed",
            True,
        ),
    ]

    invariant_df = pd.DataFrame(
        [
            {
                "check": name,
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
            "At least one Phase-5.3.1m "
            "first-update invariant failed."
        ),
    )

    print(
        invariant_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Write freeze outputs
    # =========================================================================

    banner(
        "WRITE PHASE-5.3.1m FIRST-UPDATE OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    import_safety_df.to_csv(
        IMPORT_SAFETY_PATH,
        index=False,
    )

    pre_step_df.to_csv(
        PRE_STEP_PATH,
        index=False,
    )

    update_df.to_csv(
        PARAMETER_UPDATE_PATH,
        index=False,
    )

    adam_state_df.to_csv(
        ADAM_STATE_PATH,
        index=False,
    )

    transition_df.to_csv(
        STATE_TRANSITION_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    # =========================================================================
    # Decision register
    # =========================================================================

    decision_df = pd.DataFrame(
        [
            {
                "decision": (
                    "first_optimizer_update"
                ),

                "value": (
                    "EXACTLY_ONE_ADAM_STEP_ON_FROZEN_"
                    "EPOCH0_FIRST_BATCH"
                ),

                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1m"
                ),
            },

            {
                "decision": (
                    "pre_step_numerical_runtime"
                ),

                "value": (
                    "PHASE_5_3_1l_2b_EXACT_REEXECUTION"
                ),

                "classification": (
                    "INHERITED_FROZEN_RUNTIME"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1m"
                ),
            },

            {
                "decision": (
                    "post_step_checkpoint_policy"
                ),

                "value": (
                    "NO_CHECKPOINT_IN_FIRST_UPDATE_PROOF"
                ),

                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1m"
                ),
            },
        ]
    )

    decision_df.to_csv(
        DECISION_REGISTER_PATH,
        index=False,
    )

    # =========================================================================
    # Contract
    # =========================================================================

    contract = {
        "phase": (
            "5.3.1m"
        ),

        "title": (
            "First Adam Weight-Update Proof Contract"
        ),

        "status": (
            "FROZEN"
        ),

        "pre_step_runtime": {
            "source": (
                str(
                    PREFLIGHT_SOURCE_PATH
                )
            ),

            "source_sha256": (
                preflight_source_sha
            ),

            "source_import_safe": (
                True
            ),

            "first_batch_sha256": (
                EXPECTED_FIRST_BATCH_SHA256
            ),

            "batch_loss": (
                reproduced_loss
            ),

            "logit_sha256": (
                EXPECTED_LOGIT_SHA256
            ),

            "gradient_sha256": (
                EXPECTED_GRADIENT_SHA256
            ),

            "parameter_state_sha256": (
                pre_step_hash
            ),
        },

        "optimizer": {
            "class": (
                "torch.optim.Adam"
            ),

            "lr": (
                ADAM_LR
            ),

            "betas": (
                list(
                    ADAM_BETAS
                )
            ),

            "eps": (
                ADAM_EPS
            ),

            "weight_decay": (
                ADAM_WEIGHT_DECAY
            ),

            "amsgrad": (
                ADAM_AMSGRAD
            ),

            "foreach": (
                ADAM_FOREACH
            ),

            "fused": (
                ADAM_FUSED
            ),

            "maximize": (
                ADAM_MAXIMIZE
            ),

            "capturable": (
                ADAM_CAPTURABLE
            ),

            "differentiable": (
                ADAM_DIFFERENTIABLE
            ),

            "optimizer_steps": (
                optimizer_step_count
            ),
        },

        "first_update": {
            "pre_step_parameter_sha256": (
                pre_step_hash
            ),

            "post_step_parameter_sha256": (
                post_step_hash
            ),

            "parameter_state_changed": (
                True
            ),

            "changed_parameter_tensors": (
                changed_tensor_count
            ),

            "parameter_tensors": (
                EXPECTED_PARAMETER_TENSORS
            ),

            "all_parameters_finite": (
                True
            ),

            "parameter_topology_unchanged": (
                True
            ),

            "buffer_values_unchanged": (
                True
            ),

            "module_topology_unchanged": (
                True
            ),
        },

        "Adam_state_after_first_step": {
            "state_entries": (
                len(
                    optimizer.state
                )
            ),

            "all_step_counters": (
                1
            ),

            "exp_avg_formula_verified": (
                True
            ),

            "exp_avg_sq_formula_verified": (
                True
            ),

            "optimizer_state_logical_sha256": (
                optimizer_state_sha
            ),
        },

        "RNG": {
            "optimizer_step_python_rng_neutral": (
                True
            ),

            "optimizer_step_numpy_rng_neutral": (
                True
            ),

            "optimizer_step_torch_rng_neutral": (
                True
            ),
        },

        "boundary": {
            "optimizer_step_performed": (
                True
            ),

            "optimizer_steps": (
                1
            ),

            "second_batch_executed": (
                False
            ),

            "second_optimizer_step_performed": (
                False
            ),

            "full_epoch_executed": (
                False
            ),

            "checkpoint_written": (
                False
            ),
        },

        "next_phase": {
            "id": (
                "5.3.2"
            ),

            "title": (
                "Full Training Runtime Execution Freeze"
            ),

            "requirement": (
                "Use the now-proven canonical model, "
                "training adapter, frozen per-epoch negative/order "
                "semantics, Adam runtime, and checkpoint/validation "
                "policy to construct the resumable 20-epoch training "
                "driver. Before long execution, freeze checkpoint "
                "contents, epoch/batch counters, validation timing, "
                "best-checkpoint selection, and resume semantics."
            ),
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

    manifest = {
        "phase": (
            "5.3.1m"
        ),

        "status": (
            "FIRST_ADAM_WEIGHT_UPDATE_"
            "PROVED_AND_FROZEN"
        ),

        "pre_step_parameter_sha256": (
            pre_step_hash
        ),

        "post_step_parameter_sha256": (
            post_step_hash
        ),

        "gradient_sha256": (
            gradient_sha_before_step
        ),

        "optimizer_state_sha256": (
            optimizer_state_sha
        ),

        "changed_parameter_tensors": (
            changed_tensor_count
        ),

        "Adam_state_entries": (
            len(
                optimizer.state
            )
        ),

        "optimizer_steps": (
            optimizer_step_count
        ),

        "checkpoint_written": (
            False
        ),

        "contract": (
            str(
                CONTRACT_PATH
            )
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
        IMPORT_SAFETY_PATH,
        PRE_STEP_PATH,
        PARAMETER_UPDATE_PATH,
        ADAM_STATE_PATH,
        STATE_TRANSITION_PATH,
        FINAL_INVARIANT_PATH,
        DECISION_REGISTER_PATH,
        CONTRACT_PATH,
        MANIFEST_PATH,
    ):

        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Final
    # =========================================================================

    banner(
        "PHASE 5.3.1m FINAL STATUS"
    )

    print(
        "Frozen first batch:                   VERIFIED"
    )

    print(
        "Pre-step logit fingerprint:           VERIFIED"
    )

    print(
        "Pre-step gradient fingerprint:        VERIFIED"
    )

    print()

    print(
        "Canonical pre-step SHA256:"
    )

    print(
        pre_step_hash
    )

    print()

    print(
        "optimizer.step():                     1"
    )

    print()

    print(
        "Canonical POST-FIRST-STEP SHA256:"
    )

    print(
        post_step_hash
    )

    print()

    print(
        f"Changed parameter tensors:            "
        f"{changed_tensor_count} / "
        f"{EXPECTED_PARAMETER_TENSORS}"
    )

    print(
        "All updated parameters finite:        PASS"
    )

    print(
        "Parameter topology unchanged:         PASS"
    )

    print(
        "Buffers unchanged:                    PASS"
    )

    print()

    print(
        f"Adam state entries:                   "
        f"{len(optimizer.state)} / "
        f"{EXPECTED_PARAMETER_TENSORS}"
    )

    print(
        "Adam step counters:                   1"
    )

    print(
        "Adam exp_avg initialization:          PASS"
    )

    print(
        "Adam exp_avg_sq initialization:       PASS"
    )

    print()

    print(
        "Optimizer-state SHA256:"
    )

    print(
        optimizer_state_sha
    )

    print()

    print(
        "Gradient SHA256 after step:"
    )

    print(
        gradient_sha_after_step
    )

    print()

    print(
        "optimizer.step() RNG neutrality:      PASS"
    )

    print()

    print(
        "Second optimizer.step():              NO"
    )

    print(
        "Second batch:                         NO"
    )

    print(
        "Full epoch:                           NO"
    )

    print(
        "Checkpoint written:                   NO"
    )

    banner(
        "PHASE 5.3.1m COMPLETE / "
        "FIRST ADAM WEIGHT UPDATE PROVED AND FROZEN"
    )


if __name__ == "__main__":
    main()