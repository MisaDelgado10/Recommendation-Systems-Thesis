"""
Phase 5.3.1f — Side-Effect-Free Canonical Runtime Loader Proof and Freeze

Purpose
-------
Phase 5.3.1e proved that the canonical Phase-4.7.1b implementation
cannot safely be imported as a normal Python module during training:

- zero __main__ guards;
- 254 unprotected top-level calls;
- 8 direct top-level write-like calls.

However, the exact required numerical definitions occur before the
Phase-4 audit workflow and have already been source-fingerprinted.

This phase freezes a SIDE-EFFECT-FREE runtime loading mechanism.

Frozen canonical source
-----------------------
scripts/phase_4_7_1b_freeze_neural_initialization_seed_contract.py

SHA256
------
c55f3ea1646cec7fdc8ef69f2310d98f
5ee95fab77f0c48392f4a9f76612761c

Loading policy
--------------
1. Parse the exact frozen source with ast.parse().
2. Detect the first top-level Phase-4.7.1b audit banner.
3. Consider only top-level nodes BEFORE that boundary.
4. Retain only:
   - Import
   - ImportFrom
   - Assign
   - AnnAssign
   - FunctionDef
   - AsyncFunctionDef
   - ClassDef
5. Do NOT retain executable Expr statements such as OUT_DIR.mkdir(...).
6. Compile those ORIGINAL AST nodes into an in-memory runtime module.
7. Never execute the Phase-4 audit workflow.
8. Use the original:
       build_canonical_model(seed=42)
       model_parameter_state_sha256(model)
9. Require the exact canonical initial-state SHA256.
10. Require no mutation of Phase-4.7.1b frozen artifacts.

THIS SCRIPT MAY:
- execute imports and static configuration assignments;
- define the original Phase-4 functions/classes;
- instantiate one canonical model using seed 42.

THIS SCRIPT DOES NOT:
- normally import the Phase-4 script;
- execute its Phase-4.7.1b audit workflow;
- rewrite Phase-4 source;
- instantiate Adam;
- instantiate Phase-5 training-negative RNG;
- instantiate Phase-5 training-order RNG;
- run training;
- perform forward/backward propagation;
- call optimizer.step();
- save model weights.
"""

from __future__ import annotations

import ast
import copy
import gc
import hashlib
import json
import random
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Frozen source provenance
# =============================================================================

CANONICAL_SOURCE_PATH = Path(
    "scripts/"
    "phase_4_7_1b_freeze_neural_initialization_seed_contract.py"
)

CANONICAL_SOURCE_SHA256 = (
    "c55f3ea1646cec7fdc8ef69f2310d98f"
    "5ee95fab77f0c48392f4a9f76612761c"
)

EXPECTED_INITIAL_STATE_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_PARAMETER_COUNT = 19_217_929
EXPECTED_PARAMETER_TENSORS = 32
EXPECTED_SEED = 42

REFERENCE_TORCH_VERSION_PREFIX = "2.7.0"


# =============================================================================
# Required runtime symbols
# =============================================================================

REQUIRED_CLASSES = {
    "DescriptionEncoder",
    "TrendExtractor",
    "BasisRGCNLayer",
    "PreferencePropagation",
    "ScoringMLP",
    "ITRSModel",
}

REQUIRED_FUNCTIONS = {
    "count_parameters",
    "tensor_sha256",
    "model_parameter_state_sha256",
    "apply_canonical_initialization",
    "build_canonical_model",
}

REQUIRED_RUNTIME_SYMBOLS = (
    REQUIRED_CLASSES
    | REQUIRED_FUNCTIONS
)


# =============================================================================
# Phase-5 prerequisite
# =============================================================================

PHASE_5_3_1E_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1e/"
    "phase_5_3_1e_import_side_effect_manifest.json"
)

PHASE_5_3_1D_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1d/"
    "phase_5_3_1d_model_instantiation_manifest.json"
)


# =============================================================================
# Phase-4 artifacts that the old normal import rewrites
# =============================================================================

PHASE4_INIT_DIR = Path(
    "data/experimental/phase_4/"
    "initialization_contract"
)

PHASE4_WRITE_RISK_ARTIFACTS = (
    PHASE4_INIT_DIR
    / "phase_4_7_1b_initialization_decision_audit.csv",

    PHASE4_INIT_DIR
    / "phase_4_7_1b_parameter_initialization_audit.csv",

    PHASE4_INIT_DIR
    / "phase_4_7_1b_seed_reproducibility_audit.csv",

    PHASE4_INIT_DIR
    / "phase_4_7_1b_initialization_state_hash.json",

    PHASE4_INIT_DIR
    / "phase_4_7_1b_neural_initialization_contract.json",

    PHASE4_INIT_DIR
    / "phase_4_7_1b_artifact_hashes.csv",
)


# =============================================================================
# Outputs
# =============================================================================

AUDIT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1f"
)

CONTRACT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

RETAINED_STATEMENT_PATH = (
    AUDIT_DIR
    / "side_effect_free_loader_retained_statements.csv"
)

EXCLUDED_STATEMENT_PATH = (
    AUDIT_DIR
    / "side_effect_free_loader_excluded_statements.csv"
)

RETAINED_ASSIGNMENT_CALL_PATH = (
    AUDIT_DIR
    / "side_effect_free_loader_assignment_calls.csv"
)

RUNTIME_SYMBOL_PATH = (
    AUDIT_DIR
    / "side_effect_free_runtime_symbol_audit.csv"
)

PHASE4_ARTIFACT_IMMUTABILITY_PATH = (
    AUDIT_DIR
    / "phase4_artifact_immutability_audit.csv"
)

MODEL_PROOF_PATH = (
    AUDIT_DIR
    / "side_effect_free_loader_model_proof.csv"
)

FINAL_INVARIANT_PATH = (
    AUDIT_DIR
    / "phase_5_3_1f_final_invariants.csv"
)

MANIFEST_PATH = (
    AUDIT_DIR
    / "phase_5_3_1f_runtime_loader_manifest.json"
)

CONTRACT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1f_side_effect_free_runtime_loading_contract.json"
)

DECISION_REGISTER_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1f_runtime_loading_decision_register.csv"
)

FREEZE_AUDIT_PATH = (
    CONTRACT_DIR
    / "phase_5_3_1f_runtime_loading_freeze_audit.csv"
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

        raise AssertionError(
            message
        )


def sha256_file(
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


def sha256_text(
    text: str,
) -> str:

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(handle)


def dotted_name(
    node: ast.AST,
) -> str:

    if isinstance(
        node,
        ast.Name,
    ):

        return node.id

    if isinstance(
        node,
        ast.Attribute,
    ):

        left = dotted_name(
            node.value
        )

        if left:

            return (
                f"{left}.{node.attr}"
            )

        return node.attr

    return ""


def node_source(
    source: str,
    node: ast.AST,
) -> str:

    value = ast.get_source_segment(
        source,
        node,
    )

    if value is not None:

        return value

    return ast.unparse(node)


def compact_source(
    text: str,
    limit: int = 400,
) -> str:

    text = " ".join(
        text.split()
    )

    if len(text) <= limit:

        return text

    return (
        text[:limit]
        + " ..."
    )


def ast_sha256(
    node: ast.AST,
) -> str:

    payload = ast.dump(
        node,
        annotate_fields=True,
        include_attributes=False,
    )

    return sha256_text(
        payload
    )


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
# Workflow-boundary detection
# =============================================================================

def is_phase4_workflow_banner(
    node: ast.AST,
) -> bool:
    """
    Detect the top-level statement:

        banner(
            "PHASE 4.7.1b — "
            "FREEZE GLOBAL NEURAL INITIALIZATION "
            "AND SEED CONTRACT"
        )

    This is the beginning of the executable audit workflow.
    """

    if not isinstance(
        node,
        ast.Expr,
    ):

        return False

    if not isinstance(
        node.value,
        ast.Call,
    ):

        return False

    call = node.value

    if dotted_name(
        call.func
    ) != "banner":

        return False

    source = ast.unparse(
        call
    ).lower()

    return (
        "phase 4.7.1b"
        in source
        and "freeze global neural initialization"
        in source
    )


def find_workflow_boundary(
    tree: ast.Module,
) -> tuple[
    int,
    ast.AST,
]:

    matches = [
        (
            index,
            node,
        )
        for index, node
        in enumerate(
            tree.body
        )
        if is_phase4_workflow_banner(
            node
        )
    ]

    require(
        len(matches)
        == 1,
        (
            "Expected exactly one canonical Phase-4.7.1b "
            "workflow banner, found "
            f"{len(matches)}"
        ),
    )

    return matches[0]


# =============================================================================
# Retention policy
# =============================================================================

RETAINED_NODE_TYPES = (
    ast.Import,
    ast.ImportFrom,
    ast.Assign,
    ast.AnnAssign,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


def retain_prefix_node(
    node: ast.AST,
) -> bool:

    return isinstance(
        node,
        RETAINED_NODE_TYPES,
    )


def calls_inside_assignment(
    node: ast.AST,
) -> list[str]:

    if not isinstance(
        node,
        (
            ast.Assign,
            ast.AnnAssign,
        ),
    ):

        return []

    calls = []

    for candidate in ast.walk(node):

        if isinstance(
            candidate,
            ast.Call,
        ):

            calls.append(
                dotted_name(
                    candidate.func
                )
            )

    return calls


# These MUST NOT occur inside a retained top-level assignment.
# Function/class bodies are allowed to contain them because they are
# not executed by module construction.
PROHIBITED_RETAINED_ASSIGNMENT_CALL_FRAGMENTS = (
    "read_csv",
    "read_parquet",
    "load_json",
    "json.load",
    "json.dump",
    "open",
    "to_csv",
    "to_parquet",
    "mkdir",
    "build_canonical_model",
    "apply_canonical_initialization",
    "torch.Generator",
    "manual_seed",
)


def assignment_call_is_prohibited(
    call_name: str,
) -> bool:

    lower = (
        call_name.lower()
    )

    return any(
        fragment.lower()
        in lower
        for fragment
        in PROHIBITED_RETAINED_ASSIGNMENT_CALL_FRAGMENTS
    )


# =============================================================================
# Runtime symbol definitions
# =============================================================================

def top_level_symbol_nodes(
    tree: ast.Module,
) -> dict[
    str,
    ast.AST,
]:

    result = {}

    for node in tree.body:

        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):

            result[
                node.name
            ] = node

    return result


# =============================================================================
# Phase-4 artifact hash snapshot
# =============================================================================

def snapshot_phase4_artifacts() -> dict[
    str,
    str,
]:

    result = {}

    for path in (
        PHASE4_WRITE_RISK_ARTIFACTS
    ):

        require(
            path.exists(),
            (
                "Expected frozen Phase-4 artifact missing: "
                f"{path}"
            ),
        )

        result[
            str(path)
        ] = sha256_file(
            path
        )

    return result


# =============================================================================
# RNG snapshots
# =============================================================================

def numpy_legacy_rng_state_equal(
    a,
    b,
) -> bool:

    return (
        a[0]
        == b[0]
        and np.array_equal(
            a[1],
            b[1],
        )
        and a[2:]
        == b[2:]
    )


# =============================================================================
# Side-effect-free loader
# =============================================================================

def build_sanitized_runtime_module(
    source: str,
    tree: ast.Module,
    boundary_index: int,
):
    """
    Compile ORIGINAL AST nodes from the source prefix.

    No source rewriting of neural classes/functions occurs.

    Only allowed top-level declaration/configuration node types are retained.
    """

    prefix_nodes = (
        tree.body[
            :boundary_index
        ]
    )

    retained = [
        copy.deepcopy(node)
        for node
        in prefix_nodes
        if retain_prefix_node(
            node
        )
    ]

    sanitized_tree = ast.Module(
        body=retained,
        type_ignores=copy.deepcopy(
            tree.type_ignores
        ),
    )

    ast.fix_missing_locations(
        sanitized_tree
    )

    module_name = (
        "_itrs_phase4_canonical_runtime_phase5"
    )

    runtime_module = types.ModuleType(
        module_name
    )

    runtime_module.__file__ = str(
        CANONICAL_SOURCE_PATH.resolve()
    )

    runtime_module.__package__ = None

    sys.modules[
        module_name
    ] = runtime_module

    compiled = compile(
        sanitized_tree,
        filename=str(
            CANONICAL_SOURCE_PATH
        ),
        mode="exec",
    )

    exec(
        compiled,
        runtime_module.__dict__,
    )

    return (
        runtime_module,
        sanitized_tree,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.1f — "
        "SIDE-EFFECT-FREE CANONICAL RUNTIME LOADER PROOF AND FREEZE"
    )

    print(
        "Normal Phase-4 module import:         FORBIDDEN"
    )

    print(
        "Sanitized AST runtime execution:      allowed in this proof"
    )

    print(
        "Model instantiation:                  allowed for hash proof"
    )

    print(
        "Optimizer instantiated:               NO"
    )

    print(
        "Training-negative RNG instantiated:  NO"
    )

    print(
        "Training-order RNG instantiated:     NO"
    )

    print(
        "Training forward pass:                NO"
    )

    print(
        "Backward pass:                        NO"
    )

    print(
        "Optimizer steps:                      0"
    )

    # =========================================================================
    # Prior-phase recheck
    # =========================================================================

    banner(
        "AUTHORITATIVE PRIOR-PHASE RECHECK"
    )

    for path in (
        CANONICAL_SOURCE_PATH,
        PHASE_5_3_1D_MANIFEST_PATH,
        PHASE_5_3_1E_MANIFEST_PATH,
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

    source_sha = sha256_file(
        CANONICAL_SOURCE_PATH
    )

    require(
        source_sha
        == CANONICAL_SOURCE_SHA256,
        (
            "Canonical Phase-4 source changed.\n"
            f"Expected: {CANONICAL_SOURCE_SHA256}\n"
            f"Actual:   {source_sha}"
        ),
    )

    phase_5_3_1d = load_json(
        PHASE_5_3_1D_MANIFEST_PATH
    )

    phase_5_3_1e = load_json(
        PHASE_5_3_1E_MANIFEST_PATH
    )

    require(
        phase_5_3_1d[
            "status"
        ]
        == (
            "NUMERICAL_MODEL_INSTANTIATION_PASSED_"
            "OPTIMIZER_NOT_CREATED"
        ),
        (
            "Phase-5.3.1d numerical model gate not passed"
        ),
    )

    require(
        phase_5_3_1e[
            "status"
        ]
        == (
            "AUDIT_COMPLETE_"
            "RUNTIME_LOADING_POLICY_NOT_YET_FROZEN"
        ),
        (
            "Unexpected Phase-5.3.1e status"
        ),
    )

    require(
        phase_5_3_1e[
            "import_analysis"
        ][
            "main_guard_count"
        ]
        == 0,
        (
            "Phase-5.3.1e main-guard evidence changed"
        ),
    )

    require(
        phase_5_3_1e[
            "import_analysis"
        ][
            "import_side_effect_risk"
        ]
        is True,
        (
            "Phase-5.3.1e does not report import-side-effect risk"
        ),
    )

    require(
        phase_5_3_1e[
            "phase_5_3_1d"
        ][
            "optimizer_instantiated"
        ]
        is False,
        (
            "Optimizer was already instantiated"
        ),
    )

    require(
        int(
            phase_5_3_1e[
                "phase_5_3_1d"
            ][
                "optimizer_steps"
            ]
        )
        == 0,
        (
            "Optimizer step already occurred"
        ),
    )

    print(
        "Canonical source SHA256:              PASS"
    )

    print(
        "Phase-5.3.1d numerical gate:          PASS"
    )

    print(
        "Phase-5.3.1e import-risk audit:       PASS"
    )

    print(
        "Optimizer steps:                      0  PASS"
    )

    # =========================================================================
    # Runtime environment
    # =========================================================================

    banner(
        "REFERENCE RUNTIME"
    )

    print(
        f"PyTorch runtime:                     "
        f"{torch.__version__}"
    )

    require(
        torch.__version__.startswith(
            REFERENCE_TORCH_VERSION_PREFIX
        ),
        (
            "Canonical numerical fingerprint requires "
            f"PyTorch {REFERENCE_TORCH_VERSION_PREFIX}*. "
            f"Current: {torch.__version__}"
        ),
    )

    print(
        "Reference PyTorch runtime:             PASS"
    )

    # =========================================================================
    # Parse exact source
    # =========================================================================

    banner(
        "CANONICAL SOURCE AST BOUNDARY"
    )

    source = (
        CANONICAL_SOURCE_PATH.read_text(
            encoding="utf-8"
        )
    )

    tree = ast.parse(
        source,
        filename=str(
            CANONICAL_SOURCE_PATH
        ),
    )

    (
        boundary_index,
        boundary_node,
    ) = find_workflow_boundary(
        tree
    )

    boundary_line = int(
        boundary_node.lineno
    )

    print(
        f"Top-level statements total:          "
        f"{len(tree.body)}"
    )

    print(
        f"Workflow boundary statement index:   "
        f"{boundary_index}"
    )

    print(
        f"Workflow boundary source line:        "
        f"{boundary_line}"
    )

    print(
        "Boundary statement:"
    )

    print(
        compact_source(
            node_source(
                source,
                boundary_node,
            )
        )
    )

    require(
        boundary_line
        == 1272,
        (
            "Canonical Phase-4 workflow boundary changed.\n"
            f"Expected source line: 1272\n"
            f"Actual:               {boundary_line}"
        ),
    )

    # =========================================================================
    # Build retention audit
    # =========================================================================

    banner(
        "SIDE-EFFECT-FREE PREFIX RETENTION AUDIT"
    )

    retained_rows = []
    excluded_rows = []
    assignment_call_rows = []

    for (
        statement_index,
        node,
    ) in enumerate(
        tree.body[
            :boundary_index
        ]
    ):

        retain = retain_prefix_node(
            node
        )

        row = {
            "statement_index": (
                statement_index
            ),
            "line_number": (
                getattr(
                    node,
                    "lineno",
                    None,
                )
            ),
            "statement_type": (
                type(node).__name__
            ),
            "ast_sha256": (
                ast_sha256(
                    node
                )
            ),
            "source": (
                compact_source(
                    node_source(
                        source,
                        node,
                    )
                )
            ),
        }

        if retain:

            retained_rows.append(
                row
            )

        else:

            excluded_rows.append(
                row
            )

        if (
            retain
            and isinstance(
                node,
                (
                    ast.Assign,
                    ast.AnnAssign,
                ),
            )
        ):

            calls = calls_inside_assignment(
                node
            )

            for call_name in (
                calls
            ):

                assignment_call_rows.append(
                    {
                        "statement_index": (
                            statement_index
                        ),
                        "line_number": (
                            getattr(
                                node,
                                "lineno",
                                None,
                            )
                        ),
                        "call_name": (
                            call_name
                        ),
                        "prohibited": (
                            assignment_call_is_prohibited(
                                call_name
                            )
                        ),
                    }
                )

    retained_df = pd.DataFrame(
        retained_rows
    )

    excluded_df = pd.DataFrame(
        excluded_rows
    )

    assignment_call_df = pd.DataFrame(
        assignment_call_rows
    )

    if (
        assignment_call_df.empty
    ):

        prohibited_assignment_calls = (
            pd.DataFrame()
        )

    else:

        prohibited_assignment_calls = (
            assignment_call_df.loc[
                assignment_call_df[
                    "prohibited"
                ]
            ]
        )

    require(
        prohibited_assignment_calls.empty,
        (
            "A retained top-level configuration assignment "
            "contains a prohibited executable call.\n\n"
            f"{prohibited_assignment_calls}"
        ),
    )

    print(
        f"Prefix statements:                   "
        f"{boundary_index}"
    )

    print(
        f"Retained declaration/config nodes:   "
        f"{len(retained_df)}"
    )

    print(
        f"Excluded prefix executable nodes:     "
        f"{len(excluded_df)}"
    )

    print(
        f"Prohibited calls in retained assigns: "
        f"{len(prohibited_assignment_calls)}"
    )

    if not (
        excluded_df.empty
    ):

        print()

        print(
            excluded_df[
                [
                    "statement_index",
                    "line_number",
                    "statement_type",
                    "source",
                ]
            ].to_string(
                index=False
            )
        )

    # Critical expected exclusion:
    # OUT_DIR.mkdir(...)
    require(
        any(
            "OUT_DIR.mkdir"
            in value
            for value
            in excluded_df[
                "source"
            ].astype(str)
        ),
        (
            "Expected OUT_DIR.mkdir(...) import-time side effect "
            "was not found among excluded prefix nodes"
        ),
    )

    print()

    print(
        "OUT_DIR.mkdir(...) excluded:          PASS"
    )

    # =========================================================================
    # Runtime symbol integrity
    # =========================================================================

    banner(
        "CANONICAL NUMERICAL SYMBOL INTEGRITY"
    )

    symbol_nodes = (
        top_level_symbol_nodes(
            tree
        )
    )

    runtime_symbol_rows = []

    for symbol in sorted(
        REQUIRED_RUNTIME_SYMBOLS
    ):

        node = (
            symbol_nodes.get(
                symbol
            )
        )

        runtime_symbol_rows.append(
            {
                "symbol": (
                    symbol
                ),
                "found_in_source": (
                    node
                    is not None
                ),
                "before_workflow_boundary": (
                    (
                        node
                        is not None
                    )
                    and int(
                        node.lineno
                    )
                    < boundary_line
                ),
                "line_number": (
                    int(
                        node.lineno
                    )
                    if node
                    is not None
                    else None
                ),
                "ast_sha256": (
                    ast_sha256(
                        node
                    )
                    if node
                    is not None
                    else None
                ),
            }
        )

    runtime_symbol_df = pd.DataFrame(
        runtime_symbol_rows
    )

    require(
        runtime_symbol_df[
            "found_in_source"
        ].all(),
        (
            "At least one required runtime symbol "
            "is missing from canonical source"
        ),
    )

    require(
        runtime_symbol_df[
            "before_workflow_boundary"
        ].all(),
        (
            "At least one required runtime symbol occurs "
            "after the Phase-4 workflow boundary"
        ),
    )

    print(
        runtime_symbol_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Snapshot Phase-4 artifacts BEFORE loader execution
    # =========================================================================

    banner(
        "PHASE-4 ARTIFACT IMMUTABILITY — BEFORE"
    )

    phase4_before = (
        snapshot_phase4_artifacts()
    )

    for (
        path,
        digest,
    ) in phase4_before.items():

        print(
            f"{path}"
        )

        print(
            f"  {digest}"
        )

    # =========================================================================
    # RNG state BEFORE sanitized module execution
    # =========================================================================

    python_rng_before = (
        random.getstate()
    )

    numpy_rng_before = (
        np.random.get_state()
    )

    torch_rng_before = (
        torch.get_rng_state().clone()
    )

    # =========================================================================
    # Execute sanitized canonical runtime
    # =========================================================================

    banner(
        "EXECUTE SIDE-EFFECT-FREE CANONICAL RUNTIME MODULE"
    )

    (
        runtime_module,
        sanitized_tree,
    ) = build_sanitized_runtime_module(
        source,
        tree,
        boundary_index,
    )

    sanitized_ast_hash = (
        sha256_text(
            ast.dump(
                sanitized_tree,
                annotate_fields=True,
                include_attributes=False,
            )
        )
    )

    print(
        "Sanitized runtime module created:     YES"
    )

    print(
        "Normal Phase-4 module import used:    NO"
    )

    print(
        "Sanitized runtime AST SHA256:"
    )

    print(
        sanitized_ast_hash
    )

    # =========================================================================
    # RNG check after MODULE EXEC only
    # =========================================================================

    python_rng_after_exec = (
        random.getstate()
    )

    numpy_rng_after_exec = (
        np.random.get_state()
    )

    torch_rng_after_exec = (
        torch.get_rng_state().clone()
    )

    require(
        python_rng_before
        == python_rng_after_exec,
        (
            "Sanitized module execution changed Python random state"
        ),
    )

    require(
        numpy_legacy_rng_state_equal(
            numpy_rng_before,
            numpy_rng_after_exec,
        ),
        (
            "Sanitized module execution changed NumPy global RNG state"
        ),
    )

    require(
        torch.equal(
            torch_rng_before,
            torch_rng_after_exec,
        ),
        (
            "Sanitized module execution changed global torch RNG state"
        ),
    )

    print(
        "Python RNG unchanged by loader:       PASS"
    )

    print(
        "NumPy global RNG unchanged:           PASS"
    )

    print(
        "Torch global RNG unchanged:           PASS"
    )

    # =========================================================================
    # Runtime symbol resolution
    # =========================================================================

    banner(
        "SIDE-EFFECT-FREE RUNTIME SYMBOL RESOLUTION"
    )

    for symbol in sorted(
        REQUIRED_RUNTIME_SYMBOLS
    ):

        require(
            hasattr(
                runtime_module,
                symbol,
            ),
            (
                "Sanitized runtime module does not expose "
                f"required symbol: {symbol}"
            ),
        )

        print(
            f"{symbol:<40} PASS"
        )

    model_class = getattr(
        runtime_module,
        "ITRSModel",
    )

    builder = getattr(
        runtime_module,
        "build_canonical_model",
    )

    state_hash_function = getattr(
        runtime_module,
        "model_parameter_state_sha256",
    )

    count_function = getattr(
        runtime_module,
        "count_parameters",
    )

    # =========================================================================
    # Canonical model proof
    # =========================================================================

    banner(
        "SIDE-EFFECT-FREE LOADER NUMERICAL PROOF"
    )

    print(
        "Calling:"
    )

    print()

    print(
        "    build_canonical_model(seed=42)"
    )

    model = builder(
        seed=EXPECTED_SEED
    )

    require(
        isinstance(
            model,
            torch.nn.Module,
        ),
        (
            "Canonical builder did not return torch.nn.Module"
        ),
    )

    require(
        isinstance(
            model,
            model_class,
        ),
        (
            "Canonical builder returned wrong model class"
        ),
    )

    parameter_tensor_count = sum(
        1
        for _ in model.parameters()
    )

    trainable_parameter_count = sum(
        int(
            parameter.numel()
        )
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

    all_cpu = all(
        parameter.device.type
        == "cpu"
        for parameter
        in model.parameters()
    )

    all_finite = all(
        tensor_is_finite(
            parameter.detach()
        )
        for parameter
        in model.parameters()
    )

    helper_parameter_count = (
        count_function(
            model
        )
    )

    require(
        parameter_tensor_count
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Sanitized-loader parameter-tensor count mismatch.\n"
            f"Expected: {EXPECTED_PARAMETER_TENSORS}\n"
            f"Actual:   {parameter_tensor_count}"
        ),
    )

    require(
        trainable_parameter_count
        == EXPECTED_PARAMETER_COUNT,
        (
            "Sanitized-loader trainable parameter count mismatch.\n"
            f"Expected: {EXPECTED_PARAMETER_COUNT:,}\n"
            f"Actual:   {trainable_parameter_count:,}"
        ),
    )

    require(
        helper_parameter_count
        == EXPECTED_PARAMETER_COUNT,
        (
            "Phase-4 count_parameters() disagrees "
            "under sanitized runtime loader"
        ),
    )

    require(
        all_cpu,
        (
            "At least one model parameter is not on CPU"
        ),
    )

    require(
        all_finite,
        (
            "At least one model parameter contains NaN/Inf"
        ),
    )

    state_hash_1 = (
        state_hash_function(
            model
        )
    )

    state_hash_2 = (
        state_hash_function(
            model
        )
    )

    require(
        state_hash_1
        == state_hash_2,
        (
            "State hash is not repeatable under "
            "side-effect-free loader"
        ),
    )

    require(
        state_hash_1
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "\n"
            "SIDE-EFFECT-FREE RUNTIME DOES NOT REPRODUCE "
            "THE CANONICAL MODEL STATE\n"
            "\n"
            f"Expected:\n{EXPECTED_INITIAL_STATE_SHA256}\n"
            "\n"
            f"Actual:\n{state_hash_1}\n"
            "\n"
            "STOP. Runtime loading policy must NOT be frozen."
        ),
    )

    print(
        f"Parameter tensors:                   "
        f"{parameter_tensor_count}  PASS"
    )

    print(
        f"Trainable parameters:                "
        f"{trainable_parameter_count:,}  PASS"
    )

    print(
        "All parameters CPU:                  PASS"
    )

    print(
        "All parameters finite:               PASS"
    )

    print()

    print(
        "Canonical state SHA256:"
    )

    print(
        state_hash_1
    )

    print()

    print(
        "Exact canonical state match:         PASS"
    )

    # =========================================================================
    # Phase-4 artifact immutability AFTER loader + model proof
    # =========================================================================

    banner(
        "PHASE-4 ARTIFACT IMMUTABILITY — AFTER"
    )

    phase4_after = (
        snapshot_phase4_artifacts()
    )

    artifact_rows = []

    for path in (
        phase4_before
    ):

        before_hash = (
            phase4_before[
                path
            ]
        )

        after_hash = (
            phase4_after[
                path
            ]
        )

        same = (
            before_hash
            == after_hash
        )

        artifact_rows.append(
            {
                "path": (
                    path
                ),
                "before_sha256": (
                    before_hash
                ),
                "after_sha256": (
                    after_hash
                ),
                "unchanged": (
                    same
                ),
            }
        )

    artifact_df = pd.DataFrame(
        artifact_rows
    )

    require(
        artifact_df[
            "unchanged"
        ].all(),
        (
            "At least one frozen Phase-4.7.1b artifact "
            "was modified by the sanitized runtime loader"
        ),
    )

    print(
        artifact_df.to_string(
            index=False
        )
    )

    print()

    print(
        "Frozen Phase-4 artifacts unchanged:  PASS"
    )

    # =========================================================================
    # Model state unchanged after proof
    # =========================================================================

    final_model_hash = (
        state_hash_function(
            model
        )
    )

    require(
        final_model_hash
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Canonical model state changed during loader proof"
        ),
    )

    # =========================================================================
    # Proof summary
    # =========================================================================

    model_proof_df = pd.DataFrame(
        [
            {
                "loader": (
                    "sanitized_prefix_ast"
                ),
                "canonical_source_sha256": (
                    source_sha
                ),
                "workflow_boundary_index": (
                    boundary_index
                ),
                "workflow_boundary_line": (
                    boundary_line
                ),
                "sanitized_ast_sha256": (
                    sanitized_ast_hash
                ),
                "seed": (
                    EXPECTED_SEED
                ),
                "parameter_tensors": (
                    parameter_tensor_count
                ),
                "trainable_parameters": (
                    trainable_parameter_count
                ),
                "expected_state_sha256": (
                    EXPECTED_INITIAL_STATE_SHA256
                ),
                "actual_state_sha256": (
                    state_hash_1
                ),
                "exact_state_match": (
                    state_hash_1
                    == EXPECTED_INITIAL_STATE_SHA256
                ),
                "phase4_artifacts_unchanged": (
                    bool(
                        artifact_df[
                            "unchanged"
                        ].all()
                    )
                ),
            }
        ]
    )

    # =========================================================================
    # Freeze decision
    # =========================================================================

    banner(
        "FREEZE SIDE-EFFECT-FREE RUNTIME LOADING POLICY"
    )

    decision_df = pd.DataFrame(
        [
            {
                "decision": (
                    "normal_import_of_phase4_7_1b_source"
                ),
                "value": (
                    "FORBIDDEN"
                ),
                "classification": (
                    "EVALUATION_INTEGRITY_GUARD"
                ),
                "reason": (
                    "Canonical source has no __main__ guard, "
                    "254 unprotected calls, and import-time writes."
                ),
                "status": (
                    "FROZEN_PHASE_5_3_1f"
                ),
            },

            {
                "decision": (
                    "phase5_runtime_loading_mechanism"
                ),
                "value": (
                    "SANITIZED_PREFIX_AST"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "reason": (
                    "Executes original frozen AST declarations/config "
                    "only; excludes Phase-4 audit workflow."
                ),
                "status": (
                    "FROZEN_PHASE_5_3_1f"
                ),
            },

            {
                "decision": (
                    "workflow_boundary"
                ),
                "value": (
                    f"statement_index={boundary_index};"
                    f"line={boundary_line}"
                ),
                "classification": (
                    "IMPLEMENTATION_INTEGRITY_FINGERPRINT"
                ),
                "reason": (
                    "First top-level Phase-4.7.1b audit banner."
                ),
                "status": (
                    "FROZEN_PHASE_5_3_1f"
                ),
            },

            {
                "decision": (
                    "retained_top_level_node_types"
                ),
                "value": (
                    "Import;ImportFrom;Assign;AnnAssign;"
                    "FunctionDef;AsyncFunctionDef;ClassDef"
                ),
                "classification": (
                    "IMPLEMENTATION_EQUIVALENT_CHOICE"
                ),
                "reason": (
                    "Preserves imports, constants/configuration, "
                    "and exact numerical definitions."
                ),
                "status": (
                    "FROZEN_PHASE_5_3_1f"
                ),
            },

            {
                "decision": (
                    "canonical_builder_call"
                ),
                "value": (
                    "build_canonical_model(seed=42)"
                ),
                "classification": (
                    "INHERITED_PHASE4_IMPLEMENTATION_RUNTIME"
                ),
                "reason": (
                    "Exact frozen global neural seed."
                ),
                "status": (
                    "FROZEN_PHASE_5_3_1f"
                ),
            },

            {
                "decision": (
                    "sanitized_runtime_ast_sha256"
                ),
                "value": (
                    sanitized_ast_hash
                ),
                "classification": (
                    "IMPLEMENTATION_INTEGRITY_FINGERPRINT"
                ),
                "reason": (
                    "Exact AST fingerprint of frozen runtime loader payload."
                ),
                "status": (
                    "FROZEN_PHASE_5_3_1f"
                ),
            },

            {
                "decision": (
                    "required_numerical_oracle"
                ),
                "value": (
                    EXPECTED_INITIAL_STATE_SHA256
                ),
                "classification": (
                    "INHERITED_FROZEN_PHASE4_NUMERICAL_ORACLE"
                ),
                "reason": (
                    "Loader is accepted only if exact model state matches."
                ),
                "status": (
                    "FROZEN_PHASE_5_3_1f"
                ),
            },
        ]
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.1f FREEZE INVARIANTS"
    )

    checks = [
        (
            "canonical_source_sha256_exact",
            source_sha
            == CANONICAL_SOURCE_SHA256,
        ),

        (
            "phase_5_3_1d_passed",
            phase_5_3_1d[
                "status"
            ]
            == (
                "NUMERICAL_MODEL_INSTANTIATION_PASSED_"
                "OPTIMIZER_NOT_CREATED"
            ),
        ),

        (
            "phase_5_3_1e_import_risk_confirmed",
            phase_5_3_1e[
                "import_analysis"
            ][
                "import_side_effect_risk"
            ]
            is True,
        ),

        (
            "workflow_boundary_unique",
            boundary_line
            == 1272,
        ),

        (
            "all_runtime_symbols_before_boundary",
            bool(
                runtime_symbol_df[
                    "before_workflow_boundary"
                ].all()
            ),
        ),

        (
            "no_prohibited_retained_assignment_calls",
            prohibited_assignment_calls.empty,
        ),

        (
            "OUT_DIR_mkdir_excluded",
            any(
                "OUT_DIR.mkdir"
                in value
                for value
                in excluded_df[
                    "source"
                ].astype(str)
            ),
        ),

        (
            "sanitized_loader_does_not_change_python_rng",
            python_rng_before
            == python_rng_after_exec,
        ),

        (
            "sanitized_loader_does_not_change_numpy_global_rng",
            numpy_legacy_rng_state_equal(
                numpy_rng_before,
                numpy_rng_after_exec,
            ),
        ),

        (
            "sanitized_loader_does_not_change_torch_global_rng",
            torch.equal(
                torch_rng_before,
                torch_rng_after_exec,
            ),
        ),

        (
            "canonical_model_instantiated_from_sanitized_runtime",
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
            "parameter_count_19217929",
            trainable_parameter_count
            == EXPECTED_PARAMETER_COUNT,
        ),

        (
            "canonical_initial_state_hash_exact",
            state_hash_1
            == EXPECTED_INITIAL_STATE_SHA256,
        ),

        (
            "state_hash_repeatable",
            state_hash_1
            == state_hash_2,
        ),

        (
            "model_state_unchanged_after_proof",
            final_model_hash
            == EXPECTED_INITIAL_STATE_SHA256,
        ),

        (
            "phase4_frozen_artifacts_unchanged",
            bool(
                artifact_df[
                    "unchanged"
                ].all()
            ),
        ),

        (
            "optimizer_not_instantiated",
            True,
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
            "forward_pass_count_zero",
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

    freeze_df = pd.DataFrame(
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
            freeze_df[
                "result"
            ]
            == "PASS"
        ).all(),
        (
            "One or more Phase-5.3.1f runtime-loader "
            "freeze invariants failed"
        ),
    )

    print(
        freeze_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Write outputs
    # =========================================================================

    banner(
        "WRITE PHASE-5.3.1f OUTPUTS"
    )

    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    retained_df.to_csv(
        RETAINED_STATEMENT_PATH,
        index=False,
    )

    excluded_df.to_csv(
        EXCLUDED_STATEMENT_PATH,
        index=False,
    )

    assignment_call_df.to_csv(
        RETAINED_ASSIGNMENT_CALL_PATH,
        index=False,
    )

    runtime_symbol_df.to_csv(
        RUNTIME_SYMBOL_PATH,
        index=False,
    )

    artifact_df.to_csv(
        PHASE4_ARTIFACT_IMMUTABILITY_PATH,
        index=False,
    )

    model_proof_df.to_csv(
        MODEL_PROOF_PATH,
        index=False,
    )

    freeze_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    decision_df.to_csv(
        DECISION_REGISTER_PATH,
        index=False,
    )

    freeze_df.to_csv(
        FREEZE_AUDIT_PATH,
        index=False,
    )

    contract = {
        "phase": (
            "5.3.1f"
        ),

        "title": (
            "Side-Effect-Free Canonical Runtime Loading Contract"
        ),

        "status": (
            "FROZEN"
        ),

        "canonical_source": {
            "path": (
                str(
                    CANONICAL_SOURCE_PATH
                )
            ),
            "sha256": (
                source_sha
            ),
            "normal_python_import_allowed": (
                False
            ),
        },

        "loading_policy": {
            "mechanism": (
                "SANITIZED_PREFIX_AST"
            ),

            "classification": (
                "IMPLEMENTATION_EQUIVALENT_CHOICE"
            ),

            "workflow_boundary": {
                "statement_index": (
                    boundary_index
                ),
                "source_line": (
                    boundary_line
                ),
                "boundary_source": (
                    compact_source(
                        node_source(
                            source,
                            boundary_node,
                        )
                    )
                ),
            },

            "retained_node_types": [
                "Import",
                "ImportFrom",
                "Assign",
                "AnnAssign",
                "FunctionDef",
                "AsyncFunctionDef",
                "ClassDef",
            ],

            "retained_statement_count": (
                len(
                    retained_df
                )
            ),

            "excluded_prefix_statement_count": (
                len(
                    excluded_df
                )
            ),

            "sanitized_ast_sha256": (
                sanitized_ast_hash
            ),

            "execute_original_ast_nodes_only": (
                True
            ),

            "phase4_audit_workflow_executed": (
                False
            ),
        },

        "canonical_runtime": {
            "builder": (
                "build_canonical_model"
            ),
            "builder_call": (
                "build_canonical_model(seed=42)"
            ),
            "seed": (
                EXPECTED_SEED
            ),
            "state_hash_function": (
                "model_parameter_state_sha256"
            ),
            "count_function": (
                "count_parameters"
            ),
        },

        "proof": {
            "parameter_tensor_count": (
                parameter_tensor_count
            ),

            "trainable_parameter_count": (
                trainable_parameter_count
            ),

            "expected_initial_state_sha256": (
                EXPECTED_INITIAL_STATE_SHA256
            ),

            "actual_initial_state_sha256": (
                state_hash_1
            ),

            "exact_state_match": (
                True
            ),

            "state_hash_repeatable": (
                True
            ),

            "phase4_artifacts_unchanged": (
                True
            ),

            "module_exec_python_rng_unchanged": (
                True
            ),

            "module_exec_numpy_rng_unchanged": (
                True
            ),

            "module_exec_torch_rng_unchanged": (
                True
            ),
        },

        "training_boundary": {
            "optimizer_instantiated": (
                False
            ),

            "training_negative_rng_instantiated": (
                False
            ),

            "training_order_rng_instantiated": (
                False
            ),

            "forward_pass_performed": (
                False
            ),

            "backward_pass_performed": (
                False
            ),

            "optimizer_steps": (
                0
            ),
        },

        "next_phase_gate": {
            "runtime_loader_frozen": (
                True
            ),

            "adam_may_be_instantiated": (
                True
            ),

            "optimizer_step_still_forbidden": (
                True
            ),

            "next_phase": (
                "5.3.1g"
            ),

            "next_phase_title": (
                "Adam and Epoch-0 Training Batch "
                "Forward/Backward Preflight"
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
            "5.3.1f"
        ),

        "status": (
            "SIDE_EFFECT_FREE_RUNTIME_LOADER_PROVED_AND_FROZEN"
        ),

        "canonical_source_sha256": (
            source_sha
        ),

        "sanitized_ast_sha256": (
            sanitized_ast_hash
        ),

        "canonical_state_sha256": (
            state_hash_1
        ),

        "phase4_artifacts_unchanged": (
            bool(
                artifact_df[
                    "unchanged"
                ].all()
            )
        ),

        "optimizer_instantiated": (
            False
        ),

        "optimizer_steps": (
            0
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
        RETAINED_STATEMENT_PATH,
        EXCLUDED_STATEMENT_PATH,
        RETAINED_ASSIGNMENT_CALL_PATH,
        RUNTIME_SYMBOL_PATH,
        PHASE4_ARTIFACT_IMMUTABILITY_PATH,
        MODEL_PROOF_PATH,
        FINAL_INVARIANT_PATH,
        MANIFEST_PATH,
        CONTRACT_PATH,
        DECISION_REGISTER_PATH,
        FREEZE_AUDIT_PATH,
    ):

        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Final
    # =========================================================================

    banner(
        "PHASE 5.3.1f FINAL STATUS"
    )

    print(
        "Normal Phase-4 Python import:         FORBIDDEN"
    )

    print(
        "Runtime loading mechanism:            SANITIZED_PREFIX_AST"
    )

    print(
        f"Workflow boundary:                    "
        f"index={boundary_index}, line={boundary_line}"
    )

    print(
        f"Sanitized runtime AST SHA256:"
    )

    print(
        sanitized_ast_hash
    )

    print()

    print(
        "Canonical builder call:               "
        "build_canonical_model(seed=42)"
    )

    print(
        f"Parameter tensors:                    "
        f"{parameter_tensor_count}"
    )

    print(
        f"Trainable parameters:                 "
        f"{trainable_parameter_count:,}"
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
        "Exact canonical state match:          PASS"
    )

    print(
        "Phase-4 artifacts unchanged:          PASS"
    )

    print(
        "Runtime loader policy:                FROZEN"
    )

    print()

    print(
        "Optimizer instantiated:               NO"
    )

    print(
        "Training-negative RNG instantiated:  NO"
    )

    print(
        "Training-order RNG instantiated:     NO"
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

    print()

    print(
        "Adam instantiation gate:              OPEN"
    )

    print(
        "optimizer.step() gate:                CLOSED"
    )

    banner(
        "PHASE 5.3.1f COMPLETE / "
        "SIDE-EFFECT-FREE CANONICAL RUNTIME LOADER FROZEN"
    )

    del model

    gc.collect()


if __name__ == "__main__":
    main()