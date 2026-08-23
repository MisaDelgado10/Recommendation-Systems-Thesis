"""
Phase 5.3.1d — Canonical Model Instantiation and Initial-State Hash Gate
CORRECTED RERUN

SCRIPT VERSION
--------------
2026-08-21-r1-builder-seed-aware

Purpose
-------
Phase 5.3.1c froze the exact canonical numerical implementation:

Source:
    scripts/phase_4_7_1b_freeze_neural_initialization_seed_contract.py

Source SHA256:
    c55f3ea1646cec7fdc8ef69f2310d98f
    5ee95fab77f0c48392f4a9f76612761c

Model class:
    ITRSModel

Canonical internal model construction:
    ITRSModel()

Canonical builder:
    build_canonical_model(seed)

Frozen global neural seed:
    42

Canonical state-hash helper:
    model_parameter_state_sha256(model)

Canonical initial-state SHA256:
    49e822ea7fad35c458f47e134c94c05e
    ac099b68c5c468e2c71559c8c88998ab

IMPORTANT CORRECTION
--------------------
The previous Phase-5.3.1d script incorrectly assumed that the
canonical helper build_canonical_model() required zero arguments.

Phase-4 source provenance shows that:

    ITRSModel.__init__(self)

takes no runtime arguments, BUT:

    build_canonical_model(seed)

requires the initialization seed.

Therefore the authoritative Phase-5 runtime call is:

    build_canonical_model(seed=42)

This does NOT alter any frozen Phase-4 or Phase-5 decision.
It realizes the already-frozen global neural seed = 42 through the
actual canonical Phase-4 helper API.

HARD NUMERICAL GATE
-------------------
The model MUST reproduce exactly:

    trainable parameters = 19,217,929
    parameter tensors     = 32

and:

    model_parameter_state_sha256(model)
    ==
    49e822ea7fad35c458f47e134c94c05e
    ac099b68c5c468e2c71559c8c88998ab

If any check fails, execution stops BEFORE Adam is instantiated.

THIS SCRIPT DOES:
- import the frozen canonical Phase-4 implementation;
- verify runtime entrypoint signatures;
- call build_canonical_model(seed=42);
- instantiate exactly one canonical ITRS model;
- execute the frozen Phase-4 initialization;
- verify parameter/tensor counts;
- verify CPU placement;
- verify parameter finiteness;
- compute the canonical state SHA256;
- verify state hashing is repeatable;
- verify no parameter mutation occurs during the audit;
- write Phase-5.3.1d audit metadata.

THIS SCRIPT DOES NOT:
- instantiate Adam;
- instantiate the Phase-5 training-negative RNG;
- instantiate the Phase-5 training-order RNG;
- generate training negatives;
- generate training order;
- perform a training forward pass;
- perform backward propagation;
- call optimizer.step();
- save a neural checkpoint.

No training occurs here.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path

import pandas as pd
import torch


# =============================================================================
# Script version
# =============================================================================

SCRIPT_VERSION = (
    "2026-08-21-r1-builder-seed-aware"
)


# =============================================================================
# Frozen Phase-4 references
# =============================================================================

CANONICAL_SOURCE_PATH = Path(
    "scripts/"
    "phase_4_7_1b_freeze_neural_initialization_seed_contract.py"
)

CANONICAL_SOURCE_SHA256 = (
    "c55f3ea1646cec7fdc8ef69f2310d98f"
    "5ee95fab77f0c48392f4a9f76612761c"
)

CANONICAL_MODEL_CLASS = (
    "ITRSModel"
)

CANONICAL_BUILDER = (
    "build_canonical_model"
)

CANONICAL_STATE_HASH_FUNCTION = (
    "model_parameter_state_sha256"
)

CANONICAL_COUNT_FUNCTION = (
    "count_parameters"
)

CANONICAL_INITIAL_STATE_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_GLOBAL_NEURAL_SEED = 42

EXPECTED_TRAINABLE_PARAMETERS = 19_217_929
EXPECTED_PARAMETER_TENSORS = 32

REFERENCE_TORCH_VERSION_PREFIX = (
    "2.7.0"
)


# =============================================================================
# Frozen Phase-5 contracts
# =============================================================================

ENTRYPOINT_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_3_1c_canonical_numerical_entrypoint_contract.json"
)

TRAINING_CONTROL_CONTRACT_PATH = Path(
    "data/experimental/phase_5/contracts/"
    "phase_5_2_2_training_control_optimizer_runtime_contract.json"
)

PHASE_5_3_1B_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1b/"
    "phase_5_3_1b_provenance_manifest.json"
)

PHASE_4_INITIALIZATION_CONTRACT_PATH = Path(
    "data/experimental/phase_4/"
    "initialization_contract/"
    "phase_4_7_1b_neural_initialization_contract.json"
)

PHASE_4_INITIALIZATION_HASH_PATH = Path(
    "data/experimental/phase_4/"
    "initialization_contract/"
    "phase_4_7_1b_initialization_state_hash.json"
)


# =============================================================================
# Outputs
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1d"
)

PARAMETER_INVENTORY_PATH = (
    OUT_DIR
    / "canonical_model_parameter_inventory.csv"
)

MODEL_INTEGRITY_AUDIT_PATH = (
    OUT_DIR
    / "canonical_model_instantiation_integrity_audit.csv"
)

RUNTIME_FUNCTION_AUDIT_PATH = (
    OUT_DIR
    / "canonical_runtime_function_signature_audit.csv"
)

INITIAL_STATE_HASH_AUDIT_PATH = (
    OUT_DIR
    / "canonical_initial_state_hash_audit.csv"
)

BUILDER_RUNTIME_AUDIT_PATH = (
    OUT_DIR
    / "canonical_builder_runtime_audit.csv"
)

MANIFEST_PATH = (
    OUT_DIR
    / "phase_5_3_1d_model_instantiation_manifest.json"
)


# =============================================================================
# Generic helpers
# =============================================================================

def banner(
    text: str,
) -> None:

    print(
        "\n"
        + "=" * 118
    )

    print(
        text
    )

    print(
        "=" * 118
    )


def require(
    condition: bool,
    message: str,
) -> None:

    if not bool(
        condition
    ):

        raise AssertionError(
            message
        )


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:

            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(
            handle
        )


def callable_non_variadic_parameters(
    callable_object,
) -> list[
    inspect.Parameter
]:
    """
    Return every declared parameter except *args/**kwargs.
    """

    signature = inspect.signature(
        callable_object
    )

    return [
        parameter
        for parameter
        in signature.parameters.values()
        if parameter.kind
        not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    ]


def required_callable_parameters(
    callable_object,
) -> list[
    inspect.Parameter
]:
    """
    Return required non-variadic parameters.
    """

    return [
        parameter
        for parameter
        in callable_non_variadic_parameters(
            callable_object
        )
        if parameter.default
        is inspect._empty
    ]


def tensor_is_finite(
    tensor: torch.Tensor,
) -> bool:

    if not (
        tensor.is_floating_point()
        or tensor.is_complex()
    ):

        return True

    return bool(
        torch.isfinite(
            tensor
        ).all().item()
    )


# =============================================================================
# Canonical source import
# =============================================================================

def import_canonical_module(
    source_path: Path,
):
    """
    Import the exact frozen Phase-4 source file.

    The module is assigned a Phase-5-specific internal module name.

    Import does not invoke Phase-4 main() because the canonical script
    has the standard __name__ == "__main__" guard.
    """

    module_name = (
        "_itrs_phase4_canonical_phase5_3_1d"
    )

    specification = (
        importlib.util.spec_from_file_location(
            module_name,
            source_path,
        )
    )

    require(
        specification
        is not None,
        (
            "Could not create import specification "
            "for canonical Phase-4 source"
        ),
    )

    require(
        specification.loader
        is not None,
        (
            "Canonical Phase-4 import specification "
            "has no loader"
        ),
    )

    module = (
        importlib.util.module_from_spec(
            specification
        )
    )

    sys.modules[
        module_name
    ] = (
        module
    )

    specification.loader.exec_module(
        module
    )

    return (
        module,
        module_name,
    )


# =============================================================================
# Signature checks
# =============================================================================

def audit_model_class_signature(
    model_class,
) -> dict:

    signature = (
        inspect.signature(
            model_class
        )
    )

    required = (
        required_callable_parameters(
            model_class
        )
    )

    require(
        len(
            required
        )
        == 0,
        (
            "Canonical ITRSModel unexpectedly requires runtime "
            "constructor arguments.\n"
            f"Signature: {signature}"
        ),
    )

    return {
        "name": (
            CANONICAL_MODEL_CLASS
        ),
        "kind": (
            "class"
        ),
        "signature": (
            str(
                signature
            )
        ),
        "required_parameter_count": (
            len(
                required
            )
        ),
        "required_parameter_names": (
            ""
        ),
        "status": (
            "PASS"
        ),
    }


def audit_builder_signature(
    builder,
) -> tuple[
    dict,
    inspect.Signature,
]:
    """
    Frozen Phase-4 builder API requirement:

        build_canonical_model(seed)

    Exactly one required runtime parameter named 'seed' is expected.
    """

    signature = (
        inspect.signature(
            builder
        )
    )

    declared = (
        callable_non_variadic_parameters(
            builder
        )
    )

    required = (
        required_callable_parameters(
            builder
        )
    )

    require(
        len(
            required
        )
        == 1,
        (
            "Canonical build_canonical_model() must have exactly "
            "one required runtime argument.\n"
            f"Expected: (seed)\n"
            f"Actual:   {signature}"
        ),
    )

    seed_parameter = (
        required[
            0
        ]
    )

    require(
        seed_parameter.name
        == "seed",
        (
            "Canonical build_canonical_model() required parameter "
            "is not named 'seed'.\n"
            f"Signature: {signature}"
        ),
    )

    require(
        seed_parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ),
        (
            "Unsupported canonical builder seed-parameter kind.\n"
            f"Kind: {seed_parameter.kind}"
        ),
    )

    return (
        {
            "name": (
                CANONICAL_BUILDER
            ),
            "kind": (
                "function"
            ),
            "signature": (
                str(
                    signature
                )
            ),
            "declared_parameter_count": (
                len(
                    declared
                )
            ),
            "required_parameter_count": (
                len(
                    required
                )
            ),
            "required_parameter_names": (
                ";".join(
                    parameter.name
                    for parameter
                    in required
                )
            ),
            "frozen_phase5_call": (
                "build_canonical_model(seed=42)"
            ),
            "status": (
                "PASS"
            ),
        },
        signature,
    )


def audit_state_hash_signature(
    function,
) -> dict:

    signature = (
        inspect.signature(
            function
        )
    )

    required = (
        required_callable_parameters(
            function
        )
    )

    require(
        len(
            required
        )
        == 1,
        (
            "Canonical model_parameter_state_sha256() "
            "must have exactly one required runtime argument.\n"
            f"Signature: {signature}"
        ),
    )

    return {
        "name": (
            CANONICAL_STATE_HASH_FUNCTION
        ),
        "kind": (
            "function"
        ),
        "signature": (
            str(
                signature
            )
        ),
        "required_parameter_count": (
            len(
                required
            )
        ),
        "required_parameter_names": (
            ";".join(
                parameter.name
                for parameter
                in required
            )
        ),
        "status": (
            "PASS"
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.1d — "
        "CANONICAL MODEL INSTANTIATION AND INITIAL-STATE HASH GATE "
        "(CORRECTED RERUN)"
    )

    print(
        f"Script version:                       "
        f"{SCRIPT_VERSION}"
    )

    print()

    print(
        "Canonical module imported:            NO -> import allowed"
    )

    print(
        "Canonical model instantiated:         NO -> instantiation allowed"
    )

    print(
        "Canonical initialization RNG:         allowed only inside Phase-4 builder"
    )

    print(
        "Training-negative RNG instantiated:  NO"
    )

    print(
        "Training-order RNG instantiated:     NO"
    )

    print(
        "Optimizer instantiated:               NO"
    )

    print(
        "Forward pass performed:               NO"
    )

    print(
        "Backward pass performed:              NO"
    )

    print(
        "Optimizer steps:                      0"
    )

    # =========================================================================
    # Authoritative input existence
    # =========================================================================

    banner(
        "AUTHORITATIVE INPUT RECHECK"
    )

    required_paths = (
        CANONICAL_SOURCE_PATH,
        ENTRYPOINT_CONTRACT_PATH,
        TRAINING_CONTROL_CONTRACT_PATH,
        PHASE_5_3_1B_MANIFEST_PATH,
        PHASE_4_INITIALIZATION_CONTRACT_PATH,
        PHASE_4_INITIALIZATION_HASH_PATH,
    )

    for path in (
        required_paths
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

    # =========================================================================
    # Canonical source integrity
    # =========================================================================

    banner(
        "CANONICAL SOURCE INTEGRITY GATE"
    )

    actual_source_sha256 = (
        sha256_file(
            CANONICAL_SOURCE_PATH
        )
    )

    require(
        actual_source_sha256
        == CANONICAL_SOURCE_SHA256,
        (
            "Canonical Phase-4 implementation source changed.\n"
            f"Expected: {CANONICAL_SOURCE_SHA256}\n"
            f"Actual:   {actual_source_sha256}"
        ),
    )

    print(
        f"Canonical source:                    "
        f"{CANONICAL_SOURCE_PATH}"
    )

    print(
        f"Expected SHA256:                     "
        f"{CANONICAL_SOURCE_SHA256}"
    )

    print(
        f"Actual SHA256:                       "
        f"{actual_source_sha256}"
    )

    print(
        "Source integrity:                    PASS"
    )

    # =========================================================================
    # Recheck Phase-5.3.1c contract
    # =========================================================================

    banner(
        "PHASE 5.3.1c ENTRYPOINT-CONTRACT RECHECK"
    )

    entrypoint_contract = load_json(
        ENTRYPOINT_CONTRACT_PATH
    )

    require(
        entrypoint_contract[
            "phase"
        ]
        == "5.3.1c",
        (
            "Unexpected canonical entrypoint contract phase"
        ),
    )

    require(
        entrypoint_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.3.1c canonical entrypoint "
            "contract is not frozen"
        ),
    )

    require(
        Path(
            entrypoint_contract[
                "canonical_source"
            ][
                "path"
            ]
        )
        == CANONICAL_SOURCE_PATH,
        (
            "Frozen canonical source path drift"
        ),
    )

    require(
        entrypoint_contract[
            "canonical_source"
        ][
            "sha256"
        ]
        == CANONICAL_SOURCE_SHA256,
        (
            "Frozen canonical source SHA256 drift"
        ),
    )

    require(
        entrypoint_contract[
            "canonical_model"
        ][
            "class_name"
        ]
        == CANONICAL_MODEL_CLASS,
        (
            "Frozen canonical model-class drift"
        ),
    )

    require(
        int(
            entrypoint_contract[
                "canonical_model"
            ][
                "expected_trainable_parameters"
            ]
        )
        == EXPECTED_TRAINABLE_PARAMETERS,
        (
            "Frozen trainable-parameter count drift"
        ),
    )

    require(
        int(
            entrypoint_contract[
                "canonical_model"
            ][
                "expected_parameter_tensors"
            ]
        )
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Frozen parameter-tensor count drift"
        ),
    )

    require(
        entrypoint_contract[
            "numerical_oracle"
        ][
            "canonical_initial_state_sha256"
        ]
        == CANONICAL_INITIAL_STATE_SHA256,
        (
            "Frozen canonical initial-state hash drift"
        ),
    )

    canonical_construction_expression = (
        entrypoint_contract[
            "canonical_construction_call"
        ][
            "source"
        ]
    )

    require(
        canonical_construction_expression.strip()
        == "ITRSModel()",
        (
            "Frozen internal model-construction expression drift.\n"
            f"Expected: ITRSModel()\n"
            f"Actual:   {canonical_construction_expression}"
        ),
    )

    print(
        "5.3.1c status:                       FROZEN  PASS"
    )

    print(
        "Canonical internal constructor:      ITRSModel()  PASS"
    )

    print(
        f"Expected trainable parameters:       "
        f"{EXPECTED_TRAINABLE_PARAMETERS:,}"
    )

    print(
        f"Expected parameter tensors:          "
        f"{EXPECTED_PARAMETER_TENSORS}"
    )

    print(
        "Initial-state numerical oracle:      PASS"
    )

    # =========================================================================
    # Recheck Phase-4 initialization contract
    # =========================================================================

    banner(
        "PHASE-4 INITIALIZATION CONTRACT RECHECK"
    )

    phase4_init_contract = load_json(
        PHASE_4_INITIALIZATION_CONTRACT_PATH
    )

    phase4_init_hash = load_json(
        PHASE_4_INITIALIZATION_HASH_PATH
    )

    combined_phase4_init_text = (
        json.dumps(
            phase4_init_contract,
            sort_keys=True,
        )
        + "\n"
        + json.dumps(
            phase4_init_hash,
            sort_keys=True,
        )
    )

    require(
        str(
            EXPECTED_GLOBAL_NEURAL_SEED
        )
        in combined_phase4_init_text,
        (
            "Frozen Phase-4 initialization artifacts do not "
            "contain expected neural seed 42"
        ),
    )

    require(
        CANONICAL_INITIAL_STATE_SHA256
        in combined_phase4_init_text,
        (
            "Frozen canonical initial-state SHA256 missing "
            "from Phase-4 initialization artifacts"
        ),
    )

    print(
        f"Frozen neural seed:                  "
        f"{EXPECTED_GLOBAL_NEURAL_SEED}  PASS"
    )

    print(
        "Frozen initial-state SHA256:         PASS"
    )

    # =========================================================================
    # Training-control guard
    # =========================================================================

    banner(
        "PHASE 5.2.2 TRAINING-CONTROL GUARD"
    )

    training_contract = load_json(
        TRAINING_CONTROL_CONTRACT_PATH
    )

    require(
        training_contract[
            "status"
        ]
        == "FROZEN",
        (
            "Phase-5.2.2 training-control contract "
            "is not frozen"
        ),
    )

    require(
        training_contract[
            "optimizer_instantiated"
        ]
        is False,
        (
            "Phase-5.2.2 unexpectedly records "
            "optimizer instantiation"
        ),
    )

    require(
        training_contract[
            "training_performed"
        ]
        is False,
        (
            "Phase-5.2.2 unexpectedly records training"
        ),
    )

    print(
        "Training-control contract:           FROZEN  PASS"
    )

    print(
        "Prior optimizer instantiation:       NO      PASS"
    )

    print(
        "Prior training:                      NO      PASS"
    )

    # =========================================================================
    # PyTorch runtime
    # =========================================================================

    banner(
        "PYTORCH RUNTIME GUARD"
    )

    print(
        f"Runtime torch version:               "
        f"{torch.__version__}"
    )

    require(
        torch.__version__.startswith(
            REFERENCE_TORCH_VERSION_PREFIX
        ),
        (
            "Frozen Phase-4 initialization reference expects "
            f"PyTorch {REFERENCE_TORCH_VERSION_PREFIX}*. "
            f"Current runtime: {torch.__version__}"
        ),
    )

    print(
        "Reference PyTorch runtime:           PASS"
    )

    # =========================================================================
    # Import canonical source
    # =========================================================================

    banner(
        "CANONICAL MODULE IMPORT"
    )

    (
        canonical_module,
        imported_module_name,
    ) = import_canonical_module(
        CANONICAL_SOURCE_PATH
    )

    require(
        Path(
            canonical_module.__file__
        ).resolve()
        == CANONICAL_SOURCE_PATH.resolve(),
        (
            "Imported module path differs from frozen "
            "canonical source"
        ),
    )

    print(
        f"Imported module name:                "
        f"{imported_module_name}"
    )

    print(
        f"Imported module path:                "
        f"{canonical_module.__file__}"
    )

    print(
        "Canonical module imported:           YES"
    )

    # =========================================================================
    # Resolve canonical runtime API
    # =========================================================================

    banner(
        "CANONICAL RUNTIME ENTRYPOINT RESOLUTION"
    )

    for symbol in (
        CANONICAL_MODEL_CLASS,
        CANONICAL_BUILDER,
        CANONICAL_STATE_HASH_FUNCTION,
    ):

        require(
            hasattr(
                canonical_module,
                symbol,
            ),
            (
                "Canonical module does not expose "
                f"required symbol: {symbol}"
            ),
        )

    model_class = getattr(
        canonical_module,
        CANONICAL_MODEL_CLASS,
    )

    builder = getattr(
        canonical_module,
        CANONICAL_BUILDER,
    )

    state_hash_function = getattr(
        canonical_module,
        CANONICAL_STATE_HASH_FUNCTION,
    )

    count_function = (
        getattr(
            canonical_module,
            CANONICAL_COUNT_FUNCTION,
        )
        if hasattr(
            canonical_module,
            CANONICAL_COUNT_FUNCTION,
        )
        else None
    )

    model_signature_row = (
        audit_model_class_signature(
            model_class
        )
    )

    (
        builder_signature_row,
        builder_signature,
    ) = audit_builder_signature(
        builder
    )

    state_hash_signature_row = (
        audit_state_hash_signature(
            state_hash_function
        )
    )

    runtime_function_rows = [
        model_signature_row,
        builder_signature_row,
        state_hash_signature_row,
    ]

    if (
        count_function
        is not None
    ):

        count_signature = inspect.signature(
            count_function
        )

        count_required = (
            required_callable_parameters(
                count_function
            )
        )

        runtime_function_rows.append(
            {
                "name": (
                    CANONICAL_COUNT_FUNCTION
                ),
                "kind": (
                    "function"
                ),
                "signature": (
                    str(
                        count_signature
                    )
                ),
                "required_parameter_count": (
                    len(
                        count_required
                    )
                ),
                "required_parameter_names": (
                    ";".join(
                        parameter.name
                        for parameter
                        in count_required
                    )
                ),
                "status": (
                    "INFORMATIONAL"
                ),
            }
        )

    runtime_function_df = pd.DataFrame(
        runtime_function_rows
    )

    print(
        runtime_function_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Canonical model constructor API:     ()"
    )

    print(
        "Canonical builder API:               (seed)"
    )

    print(
        "Frozen Phase-5 builder call:          "
        "build_canonical_model(seed=42)"
    )

    # =========================================================================
    # Record corrected builder semantics
    # =========================================================================

    banner(
        "CANONICAL BUILDER SEED CONTRACT"
    )

    builder_parameters = (
        list(
            builder_signature.parameters.values()
        )
    )

    seed_parameter = (
        [
            parameter
            for parameter
            in builder_parameters
            if parameter.name
            == "seed"
        ]
    )

    require(
        len(
            seed_parameter
        )
        == 1,
        (
            "Could not uniquely resolve canonical builder "
            "parameter named seed"
        ),
    )

    seed_parameter = (
        seed_parameter[
            0
        ]
    )

    seed_can_be_keyword = (
        seed_parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    )

    # If the parameter were positional-only we would still be able to
    # realize the frozen seed, but the runtime call would be builder(42).
    # Current observed signature '(seed)' is expected to support keyword use.

    require(
        seed_can_be_keyword,
        (
            "Canonical builder seed parameter is positional-only; "
            "expected keyword-compatible signature (seed)"
        ),
    )

    builder_runtime_df = pd.DataFrame(
        [
            {
                "builder": (
                    CANONICAL_BUILDER
                ),
                "runtime_signature": (
                    str(
                        builder_signature
                    )
                ),
                "required_parameter_count": (
                    1
                ),
                "required_parameter": (
                    "seed"
                ),
                "frozen_seed": (
                    EXPECTED_GLOBAL_NEURAL_SEED
                ),
                "canonical_phase5_call": (
                    "build_canonical_model(seed=42)"
                ),
                "classification": (
                    "INHERITED_PHASE4_IMPLEMENTATION_RUNTIME"
                ),
                "reopens_frozen_decision": (
                    False
                ),
                "status": (
                    "PASS"
                ),
            }
        ]
    )

    print(
        builder_runtime_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # FIRST SUCCESSFUL MODEL INSTANTIATION ATTEMPT
    # =========================================================================

    banner(
        "CANONICAL ITRS MODEL INSTANTIATION"
    )

    print(
        "Calling authoritative Phase-4 helper:"
    )

    print()

    print(
        "    build_canonical_model(seed=42)"
    )

    print()

    # -------------------------------------------------------------------------
    # IMPORTANT:
    #
    # We intentionally DO NOT construct ITRSModel ourselves here.
    #
    # The canonical Phase-4 builder owns:
    # - model creation;
    # - CPU Generator creation;
    # - seed realization;
    # - canonical Kaiming initialization.
    #
    # This prevents a second independent implementation of Phase-4
    # initialization semantics inside Phase 5.
    # -------------------------------------------------------------------------

    model = builder(
        seed=EXPECTED_GLOBAL_NEURAL_SEED
    )

    require(
        isinstance(
            model,
            torch.nn.Module,
        ),
        (
            "build_canonical_model(seed=42) did not return "
            "a torch.nn.Module"
        ),
    )

    require(
        isinstance(
            model,
            model_class,
        ),
        (
            "Canonical builder returned unexpected model type.\n"
            f"Expected: {model_class}\n"
            f"Actual:   {type(model)}"
        ),
    )

    require(
        model.__class__.__name__
        == CANONICAL_MODEL_CLASS,
        (
            "Canonical runtime model class-name drift"
        ),
    )

    print(
        f"Runtime model class:                 "
        f"{model.__class__.__name__}"
    )

    print(
        "Model instantiated:                  YES"
    )

    print(
        "Instantiation route:                 "
        "build_canonical_model(seed=42)"
    )

    # =========================================================================
    # Parameter inventory
    # =========================================================================

    banner(
        "CANONICAL PARAMETER INVENTORY"
    )

    parameter_rows = []

    total_parameter_numel = 0
    trainable_parameter_numel = 0

    parameter_tensor_count = 0
    trainable_parameter_tensor_count = 0

    all_parameters_cpu = True
    all_parameters_finite = True

    for (
        parameter_index,
        (
            parameter_name,
            parameter,
        ),
    ) in enumerate(
        model.named_parameters()
    ):

        parameter_tensor_count += 1

        numel = int(
            parameter.numel()
        )

        total_parameter_numel += (
            numel
        )

        if (
            parameter.requires_grad
        ):

            trainable_parameter_tensor_count += 1

            trainable_parameter_numel += (
                numel
            )

        is_cpu = (
            parameter.device.type
            == "cpu"
        )

        is_finite = (
            tensor_is_finite(
                parameter.detach()
            )
        )

        all_parameters_cpu = (
            all_parameters_cpu
            and is_cpu
        )

        all_parameters_finite = (
            all_parameters_finite
            and is_finite
        )

        parameter_rows.append(
            {
                "parameter_index": (
                    parameter_index
                ),
                "parameter_name": (
                    parameter_name
                ),
                "shape": (
                    str(
                        tuple(
                            parameter.shape
                        )
                    )
                ),
                "numel": (
                    numel
                ),
                "requires_grad": (
                    bool(
                        parameter.requires_grad
                    )
                ),
                "dtype": (
                    str(
                        parameter.dtype
                    )
                ),
                "device": (
                    str(
                        parameter.device
                    )
                ),
                "is_cpu": (
                    is_cpu
                ),
                "is_finite": (
                    is_finite
                ),
            }
        )

    parameter_df = pd.DataFrame(
        parameter_rows
    )

    require(
        parameter_tensor_count
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Canonical parameter-tensor count mismatch.\n"
            f"Expected: {EXPECTED_PARAMETER_TENSORS}\n"
            f"Actual:   {parameter_tensor_count}"
        ),
    )

    require(
        trainable_parameter_tensor_count
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Not all canonical parameter tensors are trainable.\n"
            f"Expected trainable tensors: {EXPECTED_PARAMETER_TENSORS}\n"
            f"Actual:                     {trainable_parameter_tensor_count}"
        ),
    )

    require(
        trainable_parameter_numel
        == EXPECTED_TRAINABLE_PARAMETERS,
        (
            "Canonical trainable parameter count mismatch.\n"
            f"Expected: {EXPECTED_TRAINABLE_PARAMETERS:,}\n"
            f"Actual:   {trainable_parameter_numel:,}"
        ),
    )

    require(
        all_parameters_cpu,
        (
            "At least one canonical neural parameter "
            "is not on CPU"
        ),
    )

    require(
        all_parameters_finite,
        (
            "At least one canonical neural parameter "
            "contains NaN or Inf"
        ),
    )

    print(
        f"Parameter tensors:                  "
        f"{parameter_tensor_count}  PASS"
    )

    print(
        f"Trainable parameter tensors:        "
        f"{trainable_parameter_tensor_count}  PASS"
    )

    print(
        f"Trainable parameters:               "
        f"{trainable_parameter_numel:,}  PASS"
    )

    print(
        f"All parameters CPU:                 "
        f"{'YES' if all_parameters_cpu else 'NO'}  PASS"
    )

    print(
        f"All parameters finite:              "
        f"{'YES' if all_parameters_finite else 'NO'}  PASS"
    )

    # =========================================================================
    # Optional Phase-4 count helper
    # =========================================================================

    banner(
        "PHASE-4 PARAMETER-COUNT HELPER CROSS-CHECK"
    )

    helper_parameter_count = None
    helper_parameter_count_status = (
        "NOT_AVAILABLE"
    )

    if (
        count_function
        is not None
    ):

        count_required = (
            required_callable_parameters(
                count_function
            )
        )

        if (
            len(
                count_required
            )
            == 1
        ):

            try:

                helper_result = (
                    count_function(
                        model
                    )
                )

                if isinstance(
                    helper_result,
                    int,
                ):

                    helper_parameter_count = int(
                        helper_result
                    )

                    require(
                        helper_parameter_count
                        == EXPECTED_TRAINABLE_PARAMETERS,
                        (
                            "Phase-4 count_parameters(model) "
                            "disagrees with frozen parameter budget.\n"
                            f"Expected: {EXPECTED_TRAINABLE_PARAMETERS:,}\n"
                            f"Actual:   {helper_parameter_count:,}"
                        ),
                    )

                    helper_parameter_count_status = (
                        "PASS"
                    )

                else:

                    helper_parameter_count_status = (
                        "NON_INTEGER_RESULT_DIAGNOSTIC_ONLY"
                    )

            except TypeError:

                helper_parameter_count_status = (
                    "INCOMPATIBLE_HELPER_API_DIAGNOSTIC_ONLY"
                )

    print(
        f"count_parameters() cross-check:      "
        f"{helper_parameter_count_status}"
    )

    if (
        helper_parameter_count
        is not None
    ):

        print(
            f"Phase-4 helper result:               "
            f"{helper_parameter_count:,}"
        )

    # =========================================================================
    # Canonical state hash
    # =========================================================================

    banner(
        "CANONICAL INITIAL-STATE SHA256 GATE"
    )

    state_hash_1 = (
        state_hash_function(
            model
        )
    )

    require(
        isinstance(
            state_hash_1,
            str,
        ),
        (
            "model_parameter_state_sha256(model) "
            "did not return a string"
        ),
    )

    state_hash_2 = (
        state_hash_function(
            model
        )
    )

    require(
        isinstance(
            state_hash_2,
            str,
        ),
        (
            "Second state-hash call did not return a string"
        ),
    )

    require(
        state_hash_1
        == state_hash_2,
        (
            "Canonical state hashing is not repeatable.\n"
            f"First:  {state_hash_1}\n"
            f"Second: {state_hash_2}"
        ),
    )

    require(
        state_hash_1
        == CANONICAL_INITIAL_STATE_SHA256,
        (
            "\n"
            "CANONICAL INITIAL-STATE HASH MISMATCH\n"
            "=====================================\n"
            "\n"
            "Expected:\n"
            f"{CANONICAL_INITIAL_STATE_SHA256}\n"
            "\n"
            "Actual:\n"
            f"{state_hash_1}\n"
            "\n"
            "STOP.\n"
            "Adam MUST NOT be instantiated and training MUST NOT begin."
        ),
    )

    print(
        "Expected canonical state SHA256:"
    )

    print(
        CANONICAL_INITIAL_STATE_SHA256
    )

    print()

    print(
        "Actual canonical state SHA256:"
    )

    print(
        state_hash_1
    )

    print()

    print(
        "Repeated hash identical:            YES  PASS"
    )

    print(
        "Canonical state exact match:         YES  PASS"
    )

    # =========================================================================
    # Zero-update mutation guard
    # =========================================================================

    banner(
        "ZERO-UPDATE PARAMETER MUTATION GUARD"
    )

    # No forward pass, backward pass, optimizer, or mutation operation
    # has occurred. Rehash the model to prove inspection itself did not
    # alter its parameters.

    state_hash_after_integrity_audit = (
        state_hash_function(
            model
        )
    )

    require(
        state_hash_after_integrity_audit
        == CANONICAL_INITIAL_STATE_SHA256,
        (
            "Canonical parameter state changed during "
            "Phase-5.3.1d integrity inspection"
        ),
    )

    print(
        "Training forward passes:             0"
    )

    print(
        "Backward passes:                     0"
    )

    print(
        "Optimizer instantiated:              NO"
    )

    print(
        "Optimizer steps:                     0"
    )

    print(
        "Parameter state unchanged:           YES  PASS"
    )

    # =========================================================================
    # Final hard invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.1d NUMERICAL MODEL-GATE INVARIANTS"
    )

    checks = [
        (
            "canonical_source_sha256_exact",
            actual_source_sha256
            == CANONICAL_SOURCE_SHA256,
        ),

        (
            "phase_5_3_1c_contract_frozen",
            entrypoint_contract[
                "status"
            ]
            == "FROZEN",
        ),

        (
            "reference_torch_version",
            torch.__version__.startswith(
                REFERENCE_TORCH_VERSION_PREFIX
            ),
        ),

        (
            "ITRSModel_requires_zero_runtime_constructor_args",
            len(
                required_callable_parameters(
                    model_class
                )
            )
            == 0,
        ),

        (
            "builder_requires_exactly_one_seed_argument",
            (
                len(
                    required_callable_parameters(
                        builder
                    )
                )
                == 1
                and required_callable_parameters(
                    builder
                )[
                    0
                ].name
                == "seed"
            ),
        ),

        (
            "builder_called_with_frozen_seed_42",
            EXPECTED_GLOBAL_NEURAL_SEED
            == 42,
        ),

        (
            "canonical_model_instantiated",
            isinstance(
                model,
                model_class,
            ),
        ),

        (
            "parameter_tensor_count_32",
            parameter_tensor_count
            == EXPECTED_PARAMETER_TENSORS,
        ),

        (
            "trainable_parameter_tensor_count_32",
            trainable_parameter_tensor_count
            == EXPECTED_PARAMETER_TENSORS,
        ),

        (
            "trainable_parameter_count_19217929",
            trainable_parameter_numel
            == EXPECTED_TRAINABLE_PARAMETERS,
        ),

        (
            "all_parameters_cpu",
            all_parameters_cpu,
        ),

        (
            "all_parameters_finite",
            all_parameters_finite,
        ),

        (
            "state_hash_repeatable",
            state_hash_1
            == state_hash_2,
        ),

        (
            "canonical_initial_state_hash_exact",
            state_hash_1
            == CANONICAL_INITIAL_STATE_SHA256,
        ),

        (
            "state_unchanged_after_integrity_audit",
            state_hash_after_integrity_audit
            == CANONICAL_INITIAL_STATE_SHA256,
        ),

        (
            "training_negative_rng_not_instantiated",
            True,
        ),

        (
            "training_order_rng_not_instantiated",
            True,
        ),

        (
            "optimizer_not_instantiated",
            True,
        ),

        (
            "training_forward_pass_count_zero",
            True,
        ),

        (
            "backward_pass_count_zero",
            True,
        ),

        (
            "optimizer_step_count_zero",
            True,
        ),
    ]

    integrity_df = pd.DataFrame(
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
        (
            integrity_df[
                "result"
            ]
            == "PASS"
        ).all(),
        (
            "One or more corrected Phase-5.3.1d "
            "model-gate invariants failed"
        ),
    )

    print(
        integrity_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Write outputs
    # =========================================================================

    banner(
        "WRITE PHASE-5.3.1d AUDIT OUTPUTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    parameter_df.to_csv(
        PARAMETER_INVENTORY_PATH,
        index=False,
    )

    runtime_function_df.to_csv(
        RUNTIME_FUNCTION_AUDIT_PATH,
        index=False,
    )

    builder_runtime_df.to_csv(
        BUILDER_RUNTIME_AUDIT_PATH,
        index=False,
    )

    integrity_df.to_csv(
        MODEL_INTEGRITY_AUDIT_PATH,
        index=False,
    )

    state_hash_df = pd.DataFrame(
        [
            {
                "expected_sha256": (
                    CANONICAL_INITIAL_STATE_SHA256
                ),
                "actual_sha256_first": (
                    state_hash_1
                ),
                "actual_sha256_second": (
                    state_hash_2
                ),
                "actual_sha256_after_integrity_audit": (
                    state_hash_after_integrity_audit
                ),
                "repeatable": (
                    state_hash_1
                    == state_hash_2
                ),
                "exact_match": (
                    state_hash_1
                    == CANONICAL_INITIAL_STATE_SHA256
                ),
                "state_unchanged": (
                    state_hash_after_integrity_audit
                    == state_hash_1
                ),
            }
        ]
    )

    state_hash_df.to_csv(
        INITIAL_STATE_HASH_AUDIT_PATH,
        index=False,
    )

    manifest = {
        "phase": (
            "5.3.1d"
        ),

        "script_version": (
            SCRIPT_VERSION
        ),

        "title": (
            "Canonical Model Instantiation "
            "and Initial-State Hash Gate"
        ),

        "status": (
            "NUMERICAL_MODEL_INSTANTIATION_PASSED_"
            "OPTIMIZER_NOT_CREATED"
        ),

        "correction_from_failed_attempt": {
            "previous_wrong_assumption": (
                "build_canonical_model required zero arguments"
            ),
            "observed_runtime_signature": (
                str(
                    builder_signature
                )
            ),
            "correct_canonical_call": (
                "build_canonical_model(seed=42)"
            ),
            "frozen_seed": (
                EXPECTED_GLOBAL_NEURAL_SEED
            ),
            "reopened_frozen_decisions": False,
        },

        "canonical_source": {
            "path": (
                str(
                    CANONICAL_SOURCE_PATH
                )
            ),
            "sha256": (
                actual_source_sha256
            ),
        },

        "runtime": {
            "python": (
                sys.version
            ),
            "torch": (
                torch.__version__
            ),
            "device": (
                "cpu"
            ),
        },

        "entrypoint_runtime": {
            "model_class": (
                CANONICAL_MODEL_CLASS
            ),
            "model_class_signature": (
                str(
                    inspect.signature(
                        model_class
                    )
                )
            ),
            "canonical_internal_construction": (
                "ITRSModel()"
            ),
            "builder": (
                CANONICAL_BUILDER
            ),
            "builder_signature": (
                str(
                    builder_signature
                )
            ),
            "builder_seed_argument": (
                EXPECTED_GLOBAL_NEURAL_SEED
            ),
            "phase5_runtime_call": (
                "build_canonical_model(seed=42)"
            ),
        },

        "model": {
            "class": (
                model.__class__.__name__
            ),
            "construction_route": (
                "build_canonical_model(seed=42)"
            ),
            "parameter_tensor_count": (
                parameter_tensor_count
            ),
            "trainable_parameter_tensor_count": (
                trainable_parameter_tensor_count
            ),
            "total_parameter_numel": (
                total_parameter_numel
            ),
            "trainable_parameter_numel": (
                trainable_parameter_numel
            ),
            "all_parameters_cpu": (
                all_parameters_cpu
            ),
            "all_parameters_finite": (
                all_parameters_finite
            ),
            "training_mode_after_construction": (
                bool(
                    model.training
                )
            ),
        },

        "initial_state": {
            "expected_sha256": (
                CANONICAL_INITIAL_STATE_SHA256
            ),
            "actual_sha256": (
                state_hash_1
            ),
            "second_sha256": (
                state_hash_2
            ),
            "after_integrity_audit_sha256": (
                state_hash_after_integrity_audit
            ),
            "exact_match": (
                state_hash_1
                == CANONICAL_INITIAL_STATE_SHA256
            ),
            "repeatable": (
                state_hash_1
                == state_hash_2
            ),
            "state_unchanged": (
                state_hash_after_integrity_audit
                == state_hash_1
            ),
        },

        "numerical_boundary": {
            "canonical_module_imported": True,
            "model_instantiated": True,

            "canonical_initialization_executed": True,
            "canonical_initialization_seed": (
                EXPECTED_GLOBAL_NEURAL_SEED
            ),

            "training_negative_rng_instantiated": False,
            "training_order_rng_instantiated": False,

            "optimizer_instantiated": False,

            "training_forward_pass_performed": False,
            "backward_pass_performed": False,

            "optimizer_steps": 0,

            "model_checkpoint_saved": False,
        },

        "next_phase_gate": {
            "adam_may_be_instantiated": True,

            "condition": (
                "canonical initial-state hash exactly matched"
            ),

            "optimizer_step_still_forbidden": True,

            "next_required_checks": [
                (
                    "Reconstruct the canonical model again "
                    "with build_canonical_model(seed=42)."
                ),
                (
                    "Verify canonical initial-state hash before Adam creation."
                ),
                (
                    "Instantiate exact frozen Adam runtime."
                ),
                (
                    "Generate deterministic epoch-0 training negatives "
                    "under the frozen Phase-5.1.1d contract."
                ),
                (
                    "Generate deterministic epoch-0 training order "
                    "under the frozen Phase-5.2.2 contract."
                ),
                (
                    "Construct one canonical training mini-batch."
                ),
                (
                    "Run forward BCEWithLogitsLoss numerical preflight."
                ),
                (
                    "Run backward gradient-finiteness preflight."
                ),
                (
                    "Verify canonical parameter SHA256 remains unchanged "
                    "because optimizer.step() is still forbidden."
                ),
            ],
        },

        "outputs": [
            str(
                PARAMETER_INVENTORY_PATH
            ),
            str(
                MODEL_INTEGRITY_AUDIT_PATH
            ),
            str(
                RUNTIME_FUNCTION_AUDIT_PATH
            ),
            str(
                BUILDER_RUNTIME_AUDIT_PATH
            ),
            str(
                INITIAL_STATE_HASH_AUDIT_PATH
            ),
        ],
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
        PARAMETER_INVENTORY_PATH,
        MODEL_INTEGRITY_AUDIT_PATH,
        RUNTIME_FUNCTION_AUDIT_PATH,
        BUILDER_RUNTIME_AUDIT_PATH,
        INITIAL_STATE_HASH_AUDIT_PATH,
        MANIFEST_PATH,
    ):

        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Final status
    # =========================================================================

    banner(
        "PHASE 5.3.1d FINAL STATUS"
    )

    print(
        f"Script version:                     "
        f"{SCRIPT_VERSION}"
    )

    print()

    print(
        "Canonical module imported:          YES"
    )

    print(
        "Canonical model instantiated:       YES"
    )

    print(
        "Internal model construction:         ITRSModel()"
    )

    print(
        "Canonical builder:                   build_canonical_model(seed)"
    )

    print(
        "Canonical builder call:              build_canonical_model(seed=42)"
    )

    print()

    print(
        f"Parameter tensors:                  "
        f"{parameter_tensor_count}"
    )

    print(
        f"Trainable parameters:               "
        f"{trainable_parameter_numel:,}"
    )

    print()

    print(
        "Canonical initial-state SHA256:"
    )

    print(
        state_hash_1
    )

    print()

    print(
        "Canonical state exact match:         PASS"
    )

    print(
        "Repeated state hash stable:          PASS"
    )

    print(
        "Parameter state unchanged:           PASS"
    )

    print()

    print(
        "Training-negative RNG instantiated: NO"
    )

    print(
        "Training-order RNG instantiated:    NO"
    )

    print(
        "Optimizer instantiated:             NO"
    )

    print(
        "Training forward pass:              NO"
    )

    print(
        "Backward pass:                      NO"
    )

    print(
        "Optimizer steps:                    0"
    )

    print()

    print(
        "Adam instantiation gate:            OPEN"
    )

    print(
        "optimizer.step() gate:              CLOSED"
    )

    banner(
        "PHASE 5.3.1d COMPLETE / "
        "EXACT CANONICAL INITIAL STATE REPRODUCED"
    )

    # Explicitly release the model before process exit.
    # No checkpoint is saved.
    del model

    gc.collect()


if __name__ == "__main__":
    main()