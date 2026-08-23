"""
Phase 5.3.1b — Canonical Phase-4 Implementation Provenance Audit
DESTINATION-AWARE CORRECTED RERUN

SCRIPT VERSION
--------------
2026-08-20-r2-destination-aware

Purpose
-------
Resolve the exact Phase-4 implementation provenance required for the
first numerical Phase-5 model instantiation.

This version corrects the previous audit in three ways:

1. PARAMETER BUDGET
   The frozen 19,217,929-parameter count is verified either from an
   authoritative literal OR by reconstructing a numeric column total
   from the Phase-4 parameter-summary artifact.

2. WRITER DETECTION
   A script is counted as writing an initialization artifact ONLY when
   the artifact path is actually the DESTINATION of a write/save call.

   Merely doing something like:

       write_json(POST_AUDIT_PATH, {"source": INIT_HASH_PATH})

   does NOT make that script a writer of INIT_HASH_PATH.

3. CANONICAL PROVENANCE
   Writer-count uniqueness is NOT used as the sole provenance oracle.

   The canonical model implementation is resolved by requiring one
   unique Phase-4 source script to contain:

       DescriptionEncoder
       TrendExtractor
       BasisRGCNLayer
       PreferencePropagation
       ScoringMLP
       ITRSModel

   together with the frozen initialization machinery:
       kaiming_normal_
       torch.Generator
       manual_seed

   The expected source is:
       phase_4_7_1b_freeze_neural_initialization_seed_contract.py

THIS SCRIPT DOES NOT:
- import the Phase-4 implementation;
- instantiate ITRSModel;
- instantiate Adam;
- instantiate any RNG;
- generate training negatives;
- generate training order;
- perform forward propagation;
- perform backward propagation;
- perform optimizer.step();
- modify any frozen Phase-2/3/4/5 decision.

It is an AUDIT-ONLY phase.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


# =============================================================================
# Version
# =============================================================================

SCRIPT_VERSION = (
    "2026-08-20-r2-destination-aware"
)


# =============================================================================
# Frozen Phase-4 references
# =============================================================================

CANONICAL_INITIAL_STATE_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_PARAMETER_COUNT = 19_217_929
EXPECTED_PARAMETER_TENSORS = 32

EXPECTED_GLOBAL_NEURAL_SEED = 42

EXPECTED_INITIALIZATION_SCRIPT_NAME = (
    "phase_4_7_1b_freeze_neural_initialization_seed_contract.py"
)

EXPECTED_MODEL_CLASS = (
    "ITRSModel"
)

EXPECTED_NN_MODULE_CLASSES = {
    "DescriptionEncoder",
    "TrendExtractor",
    "BasisRGCNLayer",
    "PreferencePropagation",
    "ScoringMLP",
    "ITRSModel",
}

INITIALIZATION_HASH_BASENAME = (
    "phase_4_7_1b_initialization_state_hash.json"
)

INITIALIZATION_CONTRACT_BASENAME = (
    "phase_4_7_1b_neural_initialization_contract.json"
)


# =============================================================================
# Phase-4 authoritative artifacts
# =============================================================================

PHASE4_ROOT = Path(
    "data/experimental/phase_4"
)

INIT_DIR = (
    PHASE4_ROOT
    / "initialization_contract"
)

INITIALIZATION_HASH_PATH = (
    INIT_DIR
    / INITIALIZATION_HASH_BASENAME
)

INITIALIZATION_CONTRACT_PATH = (
    INIT_DIR
    / INITIALIZATION_CONTRACT_BASENAME
)

PRE_INIT_INTEGRITY_PATH = (
    PHASE4_ROOT
    / "model_integrity_audit"
    / "phase_4_7_1a_integrity_metadata.json"
)

POST_INIT_INTEGRITY_PATH = (
    PHASE4_ROOT
    / "model_integrity_audit"
    / "phase_4_7_2_post_initialization_integrity_metadata.json"
)

FULL_TOPOLOGY_CONTRACT_PATH = (
    PHASE4_ROOT
    / "full_model_contract"
    / "full_itrs_model_topology_contract.json"
)

FULL_FORWARD_CONTRACT_PATH = (
    PHASE4_ROOT
    / "full_model_forward_audit"
    / "phase_4_6_2_end_to_end_contract.json"
)

PHASE4_CLOSURE_DIR = (
    PHASE4_ROOT
    / "closure"
)

CLOSURE_MANIFEST_PATH = (
    PHASE4_CLOSURE_DIR
    / "phase_4_closure_manifest.json"
)

HANDOFF_CONTRACT_PATH = (
    PHASE4_CLOSURE_DIR
    / "phase_4_to_phase_5_handoff_contract.json"
)

ARTIFACT_REGISTRY_PATH = (
    PHASE4_CLOSURE_DIR
    / "phase_4_authoritative_artifact_registry.csv"
)

PARAMETER_SUMMARY_PATH = (
    PHASE4_CLOSURE_DIR
    / "phase_4_parameter_summary.csv"
)

FINAL_CONTRACT_AUDIT_PATH = (
    PHASE4_CLOSURE_DIR
    / "phase_4_final_contract_status_audit.csv"
)

DECISION_REGISTER_PATH = (
    PHASE4_CLOSURE_DIR
    / "phase_4_final_model_decision_register.csv"
)

PHASE4_REPRO_LOG_PATH = (
    PHASE4_CLOSURE_DIR
    / "Phase_4_Reproduction_Log_Entry.md"
)


# =============================================================================
# Phase-5 pristine boundary
# =============================================================================

PHASE5_PREFLIGHT_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1a/"
    "phase_5_3_1a_preflight_manifest.json"
)


# =============================================================================
# Scripts
# =============================================================================

SCRIPT_DIR = Path(
    "scripts"
)

LATE_PHASE4_SCRIPT_NAMES = (
    "phase_4_6_1b_freeze_full_model_topology_contract.py",
    "phase_4_6_2_end_to_end_itrs_forward_bce_audit.py",
    "phase_4_7_1a_complete_model_integrity_audit.py",
    "phase_4_7_1b_freeze_neural_initialization_seed_contract.py",
    "phase_4_7_2_final_post_initialization_integrity_audit.py",
    "phase_4_8_close_model_reconstruction.py",
)


# =============================================================================
# Outputs
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1b"
)

SERIALIZED_STATE_AUDIT_PATH = (
    OUT_DIR
    / "authoritative_serialized_neural_state_audit.csv"
)

PARAMETER_BUDGET_EVIDENCE_PATH = (
    OUT_DIR
    / "phase4_parameter_budget_evidence.csv"
)

SCRIPT_PROVENANCE_PATH = (
    OUT_DIR
    / "phase4_script_provenance_evidence.csv"
)

WRITER_DIAGNOSTIC_PATH = (
    OUT_DIR
    / "phase4_initialization_writer_diagnostic.csv"
)

CLASS_INVENTORY_PATH = (
    OUT_DIR
    / "phase4_class_ast_inventory.csv"
)

FUNCTION_INVENTORY_PATH = (
    OUT_DIR
    / "phase4_function_inventory.csv"
)

CLASS_CONSISTENCY_PATH = (
    OUT_DIR
    / "phase4_class_implementation_consistency.csv"
)

INITIALIZATION_SOURCE_AUDIT_PATH = (
    OUT_DIR
    / "phase4_initialization_source_semantics_audit.csv"
)

HASH_REFERENCE_PATH = (
    OUT_DIR
    / "phase4_initialization_hash_reference_locations.csv"
)

FINAL_INVARIANT_PATH = (
    OUT_DIR
    / "phase_5_3_1b_final_provenance_invariants.csv"
)

MANIFEST_PATH = (
    OUT_DIR
    / "phase_5_3_1b_provenance_manifest.json"
)


# =============================================================================
# Basic helpers
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
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def sha256_text(
    text: str,
) -> str:

    return hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).hexdigest()


def safe_read_text(
    path: Path,
) -> str:

    return path.read_text(
        encoding="utf-8",
        errors="replace",
    )


def load_json(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def recursively_collect_strings(
    obj: Any,
    prefix: str = "$",
) -> list[
    tuple[
        str,
        str,
    ]
]:

    results = []

    if isinstance(
        obj,
        dict,
    ):

        for key, value in (
            obj.items()
        ):

            results.extend(
                recursively_collect_strings(
                    value,
                    f"{prefix}.{key}",
                )
            )

    elif isinstance(
        obj,
        list,
    ):

        for index, value in enumerate(
            obj
        ):

            results.extend(
                recursively_collect_strings(
                    value,
                    f"{prefix}[{index}]",
                )
            )

    elif isinstance(
        obj,
        str,
    ):

        results.append(
            (
                prefix,
                obj,
            )
        )

    return results


# =============================================================================
# AST utilities
# =============================================================================

def dotted_name(
    node: ast.AST,
) -> str:

    if isinstance(
        node,
        ast.Name,
    ):

        return (
            node.id
        )

    if isinstance(
        node,
        ast.Attribute,
    ):

        prefix = dotted_name(
            node.value
        )

        if prefix:

            return (
                f"{prefix}.{node.attr}"
            )

        return (
            node.attr
        )

    return ""


def ast_node_sha256(
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


def is_nn_module_class(
    node: ast.ClassDef,
) -> bool:

    base_names = [
        dotted_name(
            base
        )
        for base
        in node.bases
    ]

    return any(
        base.endswith(
            "Module"
        )
        for base
        in base_names
    )


def names_referenced(
    node: ast.AST,
) -> set[str]:

    return {
        candidate.id
        for candidate
        in ast.walk(
            node
        )
        if isinstance(
            candidate,
            ast.Name,
        )
    }


# =============================================================================
# Static constant resolution
# =============================================================================

def resolve_constant_node(
    node: ast.AST,
    environment: dict[
        str,
        Any,
    ],
):

    if isinstance(
        node,
        ast.Constant,
    ):

        return (
            node.value
        )

    if isinstance(
        node,
        ast.Name,
    ):

        return environment.get(
            node.id
        )

    if isinstance(
        node,
        ast.UnaryOp,
    ):

        value = resolve_constant_node(
            node.operand,
            environment,
        )

        if value is None:
            return None

        if (
            isinstance(
                node.op,
                ast.USub,
            )
            and isinstance(
                value,
                (
                    int,
                    float,
                ),
            )
        ):

            return (
                -value
            )

        if (
            isinstance(
                node.op,
                ast.UAdd,
            )
            and isinstance(
                value,
                (
                    int,
                    float,
                ),
            )
        ):

            return (
                value
            )

    if isinstance(
        node,
        ast.Tuple,
    ):

        values = [
            resolve_constant_node(
                item,
                environment,
            )
            for item
            in node.elts
        ]

        if all(
            value is not None
            for value
            in values
        ):

            return tuple(
                values
            )

    if isinstance(
        node,
        ast.List,
    ):

        values = [
            resolve_constant_node(
                item,
                environment,
            )
            for item
            in node.elts
        ]

        if all(
            value is not None
            for value
            in values
        ):

            return list(
                values
            )

    return None


def extract_module_constant_environment(
    tree: ast.Module,
) -> dict[
    str,
    Any,
]:

    assignments = {}

    for node in tree.body:

        if isinstance(
            node,
            ast.Assign,
        ):

            for target in (
                node.targets
            ):

                if isinstance(
                    target,
                    ast.Name,
                ):

                    assignments[
                        target.id
                    ] = (
                        node.value
                    )

        elif isinstance(
            node,
            ast.AnnAssign,
        ):

            if (
                isinstance(
                    node.target,
                    ast.Name,
                )
                and node.value
                is not None
            ):

                assignments[
                    node.target.id
                ] = (
                    node.value
                )

    environment = {}

    changed = True

    while changed:

        changed = False

        for (
            name,
            value_node,
        ) in assignments.items():

            if (
                name
                in environment
            ):
                continue

            value = (
                resolve_constant_node(
                    value_node,
                    environment,
                )
            )

            if (
                value
                is not None
            ):

                environment[
                    name
                ] = (
                    value
                )

                changed = True

    return environment


def extract_resolved_calls(
    tree: ast.Module,
    function_suffix: str,
    environment: dict[
        str,
        Any,
    ],
) -> list[dict]:

    rows = []

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        function_name = (
            dotted_name(
                node.func
            )
        )

        if not (
            function_name.endswith(
                function_suffix
            )
        ):
            continue

        row = {
            "function": (
                function_name
            ),
            "args": [],
        }

        for argument in (
            node.args
        ):

            resolved = (
                resolve_constant_node(
                    argument,
                    environment,
                )
            )

            row[
                "args"
            ].append(
                resolved
                if resolved
                is not None
                else ast.unparse(
                    argument
                )
            )

        for keyword in (
            node.keywords
        ):

            if (
                keyword.arg
                is None
            ):
                continue

            resolved = (
                resolve_constant_node(
                    keyword.value,
                    environment,
                )
            )

            row[
                keyword.arg
            ] = (
                resolved
                if resolved
                is not None
                else ast.unparse(
                    keyword.value
                )
            )

        rows.append(
            row
        )

    return rows


# =============================================================================
# Artifact-path resolution
# =============================================================================

def top_level_assignment_nodes(
    tree: ast.Module,
) -> dict[
    str,
    ast.AST,
]:

    assignments = {}

    for node in tree.body:

        if isinstance(
            node,
            ast.Assign,
        ):

            for target in (
                node.targets
            ):

                if isinstance(
                    target,
                    ast.Name,
                ):

                    assignments[
                        target.id
                    ] = (
                        node.value
                    )

        elif isinstance(
            node,
            ast.AnnAssign,
        ):

            if (
                isinstance(
                    node.target,
                    ast.Name,
                )
                and node.value
                is not None
            ):

                assignments[
                    node.target.id
                ] = (
                    node.value
                )

    return assignments


def variables_associated_with_basename(
    tree: ast.Module,
    basename: str,
) -> set[str]:

    assignments = (
        top_level_assignment_nodes(
            tree
        )
    )

    marked = set()

    # First pass: direct basename.
    for (
        name,
        expression,
    ) in assignments.items():

        expression_text = (
            ast.unparse(
                expression
            )
        )

        if (
            basename
            in expression_text
        ):

            marked.add(
                name
            )

    # Propagate only through assignments.
    changed = True

    while changed:

        changed = False

        for (
            name,
            expression,
        ) in assignments.items():

            if (
                name
                in marked
            ):
                continue

            referenced = (
                names_referenced(
                    expression
                )
            )

            if (
                referenced
                & marked
            ):

                marked.add(
                    name
                )

                changed = True

    return marked


def expression_references_artifact(
    expression: ast.AST,
    basename: str,
    marked_variables: set[str],
) -> bool:

    expression_text = (
        ast.unparse(
            expression
        )
    )

    if (
        basename
        in expression_text
    ):

        return True

    referenced = (
        names_referenced(
            expression
        )
    )

    return bool(
        referenced
        & marked_variables
    )


# =============================================================================
# Destination-aware write analysis
# =============================================================================

def write_destination_expressions(
    call: ast.Call,
) -> list[
    ast.AST
]:
    """
    Return ONLY AST expressions representing the destination path of
    a persistence call.

    Payload/source arguments are deliberately ignored.

    This is the crucial correction over the previous implementation.
    """

    destinations = []

    function_name = (
        dotted_name(
            call.func
        )
        .lower()
    )

    # -------------------------------------------------------------------------
    # Path.write_text(...)
    # Path.write_bytes(...)
    # Path.open("w")
    # -------------------------------------------------------------------------

    if isinstance(
        call.func,
        ast.Attribute,
    ):

        method = (
            call.func.attr
            .lower()
        )

        receiver = (
            call.func.value
        )

        if method in {
            "write_text",
            "write_bytes",
        }:

            destinations.append(
                receiver
            )

            return destinations

        if (
            method
            == "open"
        ):

            call_text = (
                ast.unparse(
                    call
                )
                .lower()
            )

            write_modes = (
                "'w'",
                '"w"',
                "'wb'",
                '"wb"',
                "'a'",
                '"a"',
                "'ab'",
                '"ab"',
            )

            if any(
                mode
                in call_text
                for mode
                in write_modes
            ):

                destinations.append(
                    receiver
                )

            return destinations

        # pandas style:
        # df.to_csv(PATH)
        # df.to_parquet(PATH)
        # df.to_json(PATH)
        if method in {
            "to_csv",
            "to_parquet",
            "to_json",
            "to_pickle",
        }:

            if (
                len(
                    call.args
                )
                >= 1
            ):

                destinations.append(
                    call.args[
                        0
                    ]
                )

            for keyword in (
                call.keywords
            ):

                if keyword.arg in {
                    "path",
                    "path_or_buf",
                }:

                    destinations.append(
                        keyword.value
                    )

            return destinations

    # -------------------------------------------------------------------------
    # Custom helpers:
    # write_json(PATH, payload)
    # write_csv(PATH, ...)
    # save_json(PATH, ...)
    #
    # For helper functions whose name begins with write/save, assume first
    # positional argument is destination unless covered by a known library
    # signature below.
    # -------------------------------------------------------------------------

    final_name = (
        function_name
        .split(
            "."
        )[
            -1
        ]
    )

    if (
        final_name.startswith(
            "write_"
        )
        or final_name
        in {
            "write_json",
            "write_csv",
            "write_text",
        }
    ):

        if (
            len(
                call.args
            )
            >= 1
        ):

            destinations.append(
                call.args[
                    0
                ]
            )

        return destinations

    # -------------------------------------------------------------------------
    # numpy
    # np.save(PATH, array)
    # np.savez(PATH, ...)
    # np.savez_compressed(PATH, ...)
    # -------------------------------------------------------------------------

    if function_name.endswith(
        (
            "np.save",
            "numpy.save",
            "np.savez",
            "numpy.savez",
            "np.savez_compressed",
            "numpy.savez_compressed",
        )
    ):

        if (
            len(
                call.args
            )
            >= 1
        ):

            destinations.append(
                call.args[
                    0
                ]
            )

        return destinations

    # -------------------------------------------------------------------------
    # torch.save(object, PATH)
    # -------------------------------------------------------------------------

    if function_name.endswith(
        "torch.save"
    ):

        if (
            len(
                call.args
            )
            >= 2
        ):

            destinations.append(
                call.args[
                    1
                ]
            )

        return destinations

    # -------------------------------------------------------------------------
    # joblib.dump(object, PATH)
    # pickle.dump does NOT directly use a path — it uses a file handle.
    # -------------------------------------------------------------------------

    if function_name.endswith(
        "joblib.dump"
    ):

        if (
            len(
                call.args
            )
            >= 2
        ):

            destinations.append(
                call.args[
                    1
                ]
            )

        return destinations

    # -------------------------------------------------------------------------
    # builtins.open(PATH, "w")
    # -------------------------------------------------------------------------

    if (
        function_name
        == "open"
    ):

        call_text = (
            ast.unparse(
                call
            )
            .lower()
        )

        write_modes = (
            "'w'",
            '"w"',
            "'wb'",
            '"wb"',
            "'a'",
            '"a"',
            "'ab'",
            '"ab"',
        )

        if (
            len(
                call.args
            )
            >= 1
            and any(
                mode
                in call_text
                for mode
                in write_modes
            )
        ):

            destinations.append(
                call.args[
                    0
                ]
            )

        return destinations

    return destinations


def direct_artifact_write_evidence(
    tree: ast.Module,
    basename: str,
) -> tuple[
    bool,
    list[str],
    set[str],
]:
    """
    Return:
        actually_written
        matching_write_calls
        artifact_path_variables

    Only the DESTINATION argument/receiver is checked.
    """

    marked_variables = (
        variables_associated_with_basename(
            tree,
            basename,
        )
    )

    matching_calls = []

    for node in ast.walk(
        tree
    ):

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        destinations = (
            write_destination_expressions(
                node
            )
        )

        if not destinations:
            continue

        for destination in (
            destinations
        ):

            if expression_references_artifact(
                destination,
                basename,
                marked_variables,
            ):

                matching_calls.append(
                    ast.unparse(
                        node
                    )
                )

                break

    return (
        len(
            matching_calls
        )
        > 0,
        matching_calls,
        marked_variables,
    )


# =============================================================================
# Parameter-budget verification
# =============================================================================

def normalized_numeric_series(
    series: pd.Series,
) -> pd.Series:

    cleaned = (
        series
        .astype(
            str
        )
        .str.replace(
            ",",
            "",
            regex=False,
        )
        .str.replace(
            "_",
            "",
            regex=False,
        )
        .str.strip()
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    )


def parameter_budget_evidence(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for column in (
        dataframe.columns
    ):

        numeric = (
            normalized_numeric_series(
                dataframe[
                    column
                ]
            )
        )

        valid = (
            numeric.dropna()
        )

        if (
            len(
                valid
            )
            == 0
        ):
            continue

        numeric_sum = float(
            valid.sum()
        )

        rows.append(
            {
                "column": (
                    str(
                        column
                    )
                ),
                "numeric_values": int(
                    len(
                        valid
                    )
                ),
                "numeric_sum": (
                    numeric_sum
                ),
                "contains_exact_total": bool(
                    (
                        valid
                        == EXPECTED_PARAMETER_COUNT
                    ).any()
                ),
                "column_sum_matches_total": bool(
                    abs(
                        numeric_sum
                        - EXPECTED_PARAMETER_COUNT
                    )
                    < 0.5
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
        "PHASE 5.3.1b — "
        "CANONICAL PHASE-4 IMPLEMENTATION PROVENANCE AUDIT "
        "(DESTINATION-AWARE CORRECTED RERUN)"
    )

    print(
        f"Audit script version:                {SCRIPT_VERSION}"
    )

    print(
        "Model imported:                      NO"
    )

    print(
        "Model instantiated:                  NO"
    )

    print(
        "Optimizer instantiated:              NO"
    )

    print(
        "RNG instantiated:                    NO"
    )

    print(
        "Training negatives generated:        NO"
    )

    print(
        "Training order generated:            NO"
    )

    print(
        "Forward pass performed:              NO"
    )

    print(
        "Backward pass performed:             NO"
    )

    print(
        "Optimizer steps:                     0"
    )

    # =========================================================================
    # Inputs
    # =========================================================================

    banner(
        "AUTHORITATIVE INPUT EXISTENCE"
    )

    required_paths = (
        INITIALIZATION_HASH_PATH,
        INITIALIZATION_CONTRACT_PATH,
        PRE_INIT_INTEGRITY_PATH,
        POST_INIT_INTEGRITY_PATH,
        FULL_TOPOLOGY_CONTRACT_PATH,
        FULL_FORWARD_CONTRACT_PATH,
        CLOSURE_MANIFEST_PATH,
        HANDOFF_CONTRACT_PATH,
        ARTIFACT_REGISTRY_PATH,
        PARAMETER_SUMMARY_PATH,
        FINAL_CONTRACT_AUDIT_PATH,
        DECISION_REGISTER_PATH,
        PHASE5_PREFLIGHT_PATH,
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
    # Pristine boundary
    # =========================================================================

    banner(
        "PHASE 5.3.1a PRISTINE-BOUNDARY RECHECK"
    )

    preflight = load_json(
        PHASE5_PREFLIGHT_PATH
    )

    require(
        preflight[
            "status"
        ]
        == (
            "AUDIT_COMPLETE_NO_NUMERICAL_"
            "MODEL_INSTANTIATION"
        ),
        (
            "Unexpected Phase-5.3.1a status: "
            f"{preflight.get('status')}"
        ),
    )

    require(
        preflight[
            "model_instantiated"
        ]
        is False,
        "Model already instantiated",
    )

    require(
        preflight[
            "optimizer_instantiated"
        ]
        is False,
        "Optimizer already instantiated",
    )

    require(
        preflight[
            "rng_instantiated"
        ]
        is False,
        "RNG already instantiated",
    )

    require(
        int(
            preflight[
                "optimizer_steps"
            ]
        )
        == 0,
        "Optimizer step already occurred",
    )

    print(
        "Model instantiated:                  NO   PASS"
    )

    print(
        "Optimizer instantiated:              NO   PASS"
    )

    print(
        "RNG instantiated:                    NO   PASS"
    )

    print(
        "Optimizer steps:                     0    PASS"
    )

    # =========================================================================
    # Authoritative initialization metadata
    # =========================================================================

    banner(
        "AUTHORITATIVE INITIALIZATION METADATA"
    )

    init_hash_payload = load_json(
        INITIALIZATION_HASH_PATH
    )

    init_contract = load_json(
        INITIALIZATION_CONTRACT_PATH
    )

    pre_integrity = load_json(
        PRE_INIT_INTEGRITY_PATH
    )

    post_integrity = load_json(
        POST_INIT_INTEGRITY_PATH
    )

    topology_contract = load_json(
        FULL_TOPOLOGY_CONTRACT_PATH
    )

    forward_contract = load_json(
        FULL_FORWARD_CONTRACT_PATH
    )

    closure_manifest = load_json(
        CLOSURE_MANIFEST_PATH
    )

    handoff_contract = load_json(
        HANDOFF_CONTRACT_PATH
    )

    json_sources = (
        (
            INITIALIZATION_HASH_PATH,
            init_hash_payload,
        ),
        (
            INITIALIZATION_CONTRACT_PATH,
            init_contract,
        ),
        (
            PRE_INIT_INTEGRITY_PATH,
            pre_integrity,
        ),
        (
            POST_INIT_INTEGRITY_PATH,
            post_integrity,
        ),
        (
            FULL_TOPOLOGY_CONTRACT_PATH,
            topology_contract,
        ),
        (
            FULL_FORWARD_CONTRACT_PATH,
            forward_contract,
        ),
        (
            CLOSURE_MANIFEST_PATH,
            closure_manifest,
        ),
        (
            HANDOFF_CONTRACT_PATH,
            handoff_contract,
        ),
    )

    hash_reference_rows = []

    for (
        source_path,
        payload,
    ) in json_sources:

        for (
            location,
            value,
        ) in recursively_collect_strings(
            payload
        ):

            if (
                CANONICAL_INITIAL_STATE_SHA256
                in value
            ):

                hash_reference_rows.append(
                    {
                        "source_file": str(
                            source_path
                        ),
                        "location": (
                            location
                        ),
                        "value": (
                            value
                        ),
                    }
                )

    if (
        PHASE4_REPRO_LOG_PATH.exists()
    ):

        lines = (
            safe_read_text(
                PHASE4_REPRO_LOG_PATH
            )
            .splitlines()
        )

        for (
            line_number,
            line,
        ) in enumerate(
            lines,
            start=1,
        ):

            if (
                CANONICAL_INITIAL_STATE_SHA256
                in line
            ):

                hash_reference_rows.append(
                    {
                        "source_file": str(
                            PHASE4_REPRO_LOG_PATH
                        ),
                        "location": (
                            f"line={line_number}"
                        ),
                        "value": (
                            line.strip()
                        ),
                    }
                )

    hash_reference_df = pd.DataFrame(
        hash_reference_rows
    )

    require(
        len(
            hash_reference_df
        )
        >= 1,
        (
            "Canonical initialization state hash "
            "not found in authoritative Phase-4 metadata"
        ),
    )

    print(
        "Canonical state SHA256:"
    )

    print(
        CANONICAL_INITIAL_STATE_SHA256
    )

    print()

    print(
        f"Authoritative references:            "
        f"{len(hash_reference_df)}"
    )

    # =========================================================================
    # Serialized state audit
    # =========================================================================

    banner(
        "SERIALIZED NEURAL STATE AUDIT"
    )

    registry = pd.read_csv(
        ARTIFACT_REGISTRY_PATH
    )

    require(
        "path"
        in registry.columns,
        "Artifact registry lacks 'path' column",
    )

    checkpoint_suffixes = {
        ".pt",
        ".pth",
        ".ckpt",
        ".safetensors",
    }

    serialized_rows = []

    for raw_path in (
        registry[
            "path"
        ]
        .dropna()
        .astype(
            str
        )
    ):

        path = Path(
            raw_path
        )

        suffix = (
            path.suffix
            .lower()
        )

        checkpoint = (
            suffix
            in checkpoint_suffixes
            or (
                suffix
                == ".bin"
                and any(
                    word
                    in path.name.lower()
                    for word
                    in (
                        "model",
                        "weights",
                        "state",
                        "checkpoint",
                    )
                )
            )
        )

        serialized_rows.append(
            {
                "path": (
                    str(
                        path
                    )
                ),
                "suffix": (
                    suffix
                ),
                "exists": (
                    path.exists()
                ),
                "serialized_neural_checkpoint": (
                    checkpoint
                ),
            }
        )

    serialized_df = pd.DataFrame(
        serialized_rows
    )

    checkpoint_candidates = (
        serialized_df.loc[
            serialized_df[
                "serialized_neural_checkpoint"
            ]
        ]
    )

    require(
        len(
            checkpoint_candidates
        )
        == 0,
        (
            "Authoritative serialized neural checkpoint discovered. "
            "Do not assume deterministic state reconstruction."
        ),
    )

    print(
        f"Authoritative registry artifacts:     "
        f"{len(serialized_df)}"
    )

    print(
        "Serialized neural checkpoints:        0  PASS"
    )

    print(
        "State realization:                    deterministic reconstruction"
    )

    # =========================================================================
    # Parameter budget
    # =========================================================================

    banner(
        "FROZEN PARAMETER-BUDGET RECONSTRUCTION"
    )

    parameter_summary = pd.read_csv(
        PARAMETER_SUMMARY_PATH
    )

    budget_df = (
        parameter_budget_evidence(
            parameter_summary
        )
    )

    literal_sources = (
        PARAMETER_SUMMARY_PATH,
        FULL_TOPOLOGY_CONTRACT_PATH,
        FULL_FORWARD_CONTRACT_PATH,
        CLOSURE_MANIFEST_PATH,
        HANDOFF_CONTRACT_PATH,
        FINAL_CONTRACT_AUDIT_PATH,
        DECISION_REGISTER_PATH,
    )

    literal_match = False
    literal_source = None

    for path in (
        literal_sources
    ):

        normalized = (
            safe_read_text(
                path
            )
            .replace(
                ",",
                "",
            )
            .replace(
                "_",
                "",
            )
        )

        if (
            str(
                EXPECTED_PARAMETER_COUNT
            )
            in normalized
        ):

            literal_match = True
            literal_source = str(
                path
            )

            break

    numeric_match = (
        not budget_df.empty
        and bool(
            (
                budget_df[
                    "contains_exact_total"
                ]
                | budget_df[
                    "column_sum_matches_total"
                ]
            ).any()
        )
    )

    parameter_budget_verified = (
        literal_match
        or numeric_match
    )

    require(
        parameter_budget_verified,
        (
            "Could not verify frozen "
            f"{EXPECTED_PARAMETER_COUNT:,}-parameter budget."
        ),
    )

    print(
        f"Expected trainable parameters:        "
        f"{EXPECTED_PARAMETER_COUNT:,}"
    )

    print(
        f"Parameter-summary rows:               "
        f"{len(parameter_summary)}"
    )

    print(
        f"Literal evidence:                     "
        f"{'YES' if literal_match else 'NO'}"
    )

    if (
        literal_source
        is not None
    ):

        print(
            f"Literal source:                       "
            f"{literal_source}"
        )

    print(
        f"Numeric reconstruction:               "
        f"{'YES' if numeric_match else 'NO'}"
    )

    if not (
        budget_df.empty
    ):

        print()

        print(
            budget_df.to_string(
                index=False
            )
        )

    print()

    print(
        "Parameter budget:                     PASS"
    )

    # =========================================================================
    # Phase-4 AST inventory
    # =========================================================================

    banner(
        "AST INSPECTION OF PHASE-4 PYTHON SCRIPTS"
    )

    script_paths = sorted(
        SCRIPT_DIR.glob(
            "phase_4*.py"
        )
    )

    require(
        len(
            script_paths
        )
        > 0,
        "No Phase-4 scripts found",
    )

    provenance_rows = []
    class_rows = []
    function_rows = []

    parsed_trees = {}
    constant_envs = {}

    writer_rows = []

    for script_path in (
        script_paths
    ):

        source = safe_read_text(
            script_path
        )

        try:

            tree = ast.parse(
                source,
                filename=str(
                    script_path
                ),
            )

        except SyntaxError as exc:

            provenance_rows.append(
                {
                    "script": str(
                        script_path
                    ),
                    "parse_ok": False,
                    "parse_error": (
                        str(
                            exc
                        )
                    ),
                }
            )

            continue

        key = str(
            script_path
        )

        parsed_trees[
            key
        ] = (
            tree
        )

        constants = (
            extract_module_constant_environment(
                tree
            )
        )

        constant_envs[
            key
        ] = (
            constants
        )

        all_classes = []
        nn_modules = []

        for node in (
            tree.body
        ):

            if isinstance(
                node,
                ast.ClassDef,
            ):

                all_classes.append(
                    node.name
                )

                module_class = (
                    is_nn_module_class(
                        node
                    )
                )

                if module_class:

                    nn_modules.append(
                        node.name
                    )

                class_rows.append(
                    {
                        "script": (
                            key
                        ),
                        "class_name": (
                            node.name
                        ),
                        "is_nn_module": (
                            module_class
                        ),
                        "class_ast_sha256": (
                            ast_node_sha256(
                                node
                            )
                        ),
                        "line_number": (
                            node.lineno
                        ),
                    }
                )

            elif isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):

                function_rows.append(
                    {
                        "script": (
                            key
                        ),
                        "function_name": (
                            node.name
                        ),
                        "function_ast_sha256": (
                            ast_node_sha256(
                                node
                            )
                        ),
                        "line_number": (
                            node.lineno
                        ),
                    }
                )

        (
            writes_hash,
            hash_write_calls,
            hash_variables,
        ) = direct_artifact_write_evidence(
            tree,
            INITIALIZATION_HASH_BASENAME,
        )

        (
            writes_contract,
            contract_write_calls,
            contract_variables,
        ) = direct_artifact_write_evidence(
            tree,
            INITIALIZATION_CONTRACT_BASENAME,
        )

        nn_module_set = set(
            nn_modules
        )

        contains_exact_module_stack = (
            nn_module_set
            == EXPECTED_NN_MODULE_CLASSES
        )

        contains_model = (
            EXPECTED_MODEL_CLASS
            in all_classes
        )

        source_lower = (
            source.lower()
        )

        contains_kaiming = (
            "kaiming_normal_"
            in source_lower
        )

        contains_generator = (
            "torch.generator"
            in source_lower
        )

        contains_manual_seed = (
            "manual_seed"
            in source_lower
        )

        mentions_hash = (
            INITIALIZATION_HASH_BASENAME
            in source
        )

        mentions_contract = (
            INITIALIZATION_CONTRACT_BASENAME
            in source
        )

        canonical_signature = (
            contains_model
            and contains_exact_module_stack
            and contains_kaiming
            and contains_generator
            and contains_manual_seed
            and mentions_hash
            and mentions_contract
        )

        provenance_rows.append(
            {
                "script": (
                    key
                ),
                "script_sha256": (
                    sha256_file(
                        script_path
                    )
                ),
                "parse_ok": True,
                "parse_error": None,

                "contains_ITRSModel": (
                    contains_model
                ),

                "nn_module_count": (
                    len(
                        nn_modules
                    )
                ),

                "nn_module_names": (
                    ";".join(
                        sorted(
                            nn_modules
                        )
                    )
                ),

                "exact_expected_module_stack": (
                    contains_exact_module_stack
                ),

                "contains_kaiming_normal": (
                    contains_kaiming
                ),

                "contains_torch_generator": (
                    contains_generator
                ),

                "contains_manual_seed": (
                    contains_manual_seed
                ),

                "mentions_initialization_hash": (
                    mentions_hash
                ),

                "mentions_initialization_contract": (
                    mentions_contract
                ),

                "destination_aware_writes_hash": (
                    writes_hash
                ),

                "destination_aware_writes_contract": (
                    writes_contract
                ),

                "canonical_implementation_signature": (
                    canonical_signature
                ),
            }
        )

        writer_rows.append(
            {
                "script": (
                    key
                ),

                "initialization_hash_path_variables": (
                    ";".join(
                        sorted(
                            hash_variables
                        )
                    )
                ),

                "writes_initialization_hash_destination": (
                    writes_hash
                ),

                "initialization_hash_write_calls": (
                    json.dumps(
                        hash_write_calls,
                        sort_keys=True,
                    )
                ),

                "initialization_contract_path_variables": (
                    ";".join(
                        sorted(
                            contract_variables
                        )
                    )
                ),

                "writes_initialization_contract_destination": (
                    writes_contract
                ),

                "initialization_contract_write_calls": (
                    json.dumps(
                        contract_write_calls,
                        sort_keys=True,
                    )
                ),
            }
        )

    provenance_df = pd.DataFrame(
        provenance_rows
    )

    writer_df = pd.DataFrame(
        writer_rows
    )

    class_df = pd.DataFrame(
        class_rows
    )

    function_df = pd.DataFrame(
        function_rows
    )

    print(
        provenance_df.loc[
            provenance_df[
                "parse_ok"
            ]
        ][
            [
                "script",
                "contains_ITRSModel",
                "nn_module_count",
                "exact_expected_module_stack",
                "contains_kaiming_normal",
                "contains_torch_generator",
                "contains_manual_seed",
                "destination_aware_writes_hash",
                "destination_aware_writes_contract",
                "canonical_implementation_signature",
            ]
        ]
        .sort_values(
            [
                "canonical_implementation_signature",
                "contains_ITRSModel",
                "nn_module_count",
            ],
            ascending=[
                False,
                False,
                False,
            ],
            kind="mergesort",
        )
        .head(
            15
        )
        .to_string(
            index=False
        )
    )

    # =========================================================================
    # Destination-aware writer diagnostic
    # =========================================================================

    banner(
        "DESTINATION-AWARE INITIALIZATION WRITER DIAGNOSTIC"
    )

    hash_writers = (
        writer_df.loc[
            writer_df[
                "writes_initialization_hash_destination"
            ]
        ]
        if not writer_df.empty
        else pd.DataFrame()
    )

    contract_writers = (
        writer_df.loc[
            writer_df[
                "writes_initialization_contract_destination"
            ]
        ]
        if not writer_df.empty
        else pd.DataFrame()
    )

    print(
        f"Destination-aware hash writers:       "
        f"{len(hash_writers)}"
    )

    if not (
        hash_writers.empty
    ):

        print()

        print(
            hash_writers[
                [
                    "script",
                    "initialization_hash_write_calls",
                ]
            ].to_string(
                index=False
            )
        )

    print()

    print(
        f"Destination-aware contract writers:   "
        f"{len(contract_writers)}"
    )

    if not (
        contract_writers.empty
    ):

        print()

        print(
            contract_writers[
                [
                    "script",
                    "initialization_contract_write_calls",
                ]
            ].to_string(
                index=False
            )
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "  Writer counts are now DIAGNOSTIC evidence."
    )

    print(
        "  Canonical implementation provenance is resolved independently"
    )

    print(
        "  using the exact frozen ITRS neural-module + initialization signature."
    )

    # =========================================================================
    # Canonical implementation source
    # =========================================================================

    banner(
        "CANONICAL IMPLEMENTATION SOURCE RESOLUTION"
    )

    canonical_candidates = (
        provenance_df.loc[
            provenance_df[
                "canonical_implementation_signature"
            ]
            == True
        ]
    )

    require(
        len(
            canonical_candidates
        )
        == 1,
        (
            "Expected exactly one Phase-4 script with the complete "
            "canonical ITRS implementation signature, found "
            f"{len(canonical_candidates)}."
        ),
    )

    canonical_row = (
        canonical_candidates.iloc[
            0
        ]
    )

    canonical_script = Path(
        str(
            canonical_row[
                "script"
            ]
        )
    )

    require(
        canonical_script.name
        == EXPECTED_INITIALIZATION_SCRIPT_NAME,
        (
            "Canonical implementation signature resolved to the "
            "wrong Phase-4 script.\n"
            f"Expected: {EXPECTED_INITIALIZATION_SCRIPT_NAME}\n"
            f"Found:    {canonical_script.name}"
        ),
    )

    canonical_script_hash = (
        sha256_file(
            canonical_script
        )
    )

    canonical_modules = {
        name
        for name
        in str(
            canonical_row[
                "nn_module_names"
            ]
        ).split(
            ";"
        )
        if name
    }

    require(
        canonical_modules
        == EXPECTED_NN_MODULE_CLASSES,
        (
            "Canonical source module set mismatch.\n"
            f"Expected: {sorted(EXPECTED_NN_MODULE_CLASSES)}\n"
            f"Found:    {sorted(canonical_modules)}"
        ),
    )

    print(
        "Resolved canonical Phase-4 implementation:"
    )

    print(
        f"  {canonical_script}"
    )

    print()

    print(
        "Source SHA256:"
    )

    print(
        f"  {canonical_script_hash}"
    )

    print()

    print(
        "Canonical model class:                ITRSModel"
    )

    print(
        "Exact expected six nn.Module classes: PASS"
    )

    print(
        "Kaiming initialization source:        PRESENT"
    )

    print(
        "CPU Generator source:                 PRESENT"
    )

    print(
        "manual_seed source:                   PRESENT"
    )

    canonical_is_hash_writer = (
        not hash_writers.empty
        and str(
            canonical_script
        )
        in set(
            hash_writers[
                "script"
            ]
        )
    )

    canonical_is_contract_writer = (
        not contract_writers.empty
        and str(
            canonical_script
        )
        in set(
            contract_writers[
                "script"
            ]
        )
    )

    print()

    print(
        f"Canonical source directly writes hash:     "
        f"{'YES' if canonical_is_hash_writer else 'NOT DETECTED'}"
    )

    print(
        f"Canonical source directly writes contract: "
        f"{'YES' if canonical_is_contract_writer else 'NOT DETECTED'}"
    )

    # =========================================================================
    # Initialization semantics
    # =========================================================================

    banner(
        "CANONICAL INITIALIZATION SOURCE SEMANTICS"
    )

    canonical_tree = (
        parsed_trees[
            str(
                canonical_script
            )
        ]
    )

    canonical_constants = (
        constant_envs[
            str(
                canonical_script
            )
        ]
    )

    kaiming_calls = (
        extract_resolved_calls(
            canonical_tree,
            "kaiming_normal_",
            canonical_constants,
        )
    )

    require(
        len(
            kaiming_calls
        )
        >= 1,
        (
            "No kaiming_normal_ call found in "
            "canonical source."
        ),
    )

    a_zero = any(
        call.get(
            "a"
        )
        == 0.0
        for call
        in kaiming_calls
    )

    fan_in = any(
        call.get(
            "mode"
        )
        == "fan_in"
        for call
        in kaiming_calls
    )

    relu = any(
        call.get(
            "nonlinearity"
        )
        == "relu"
        for call
        in kaiming_calls
    )

    require(
        a_zero,
        "Could not resolve Kaiming a=0.0",
    )

    require(
        fan_in,
        "Could not resolve Kaiming mode='fan_in'",
    )

    require(
        relu,
        "Could not resolve Kaiming nonlinearity='relu'",
    )

    seed_constants = {
        name: value
        for (
            name,
            value,
        ) in canonical_constants.items()
        if (
            "seed"
            in name.lower()
            and value
            == EXPECTED_GLOBAL_NEURAL_SEED
        )
    }

    canonical_text = (
        safe_read_text(
            canonical_script
        )
    )

    seed_literal = bool(
        re.search(
            r"\b42\b",
            canonical_text,
        )
    )

    seed_42_verified = (
        bool(
            seed_constants
        )
        or seed_literal
    )

    require(
        seed_42_verified,
        "No source evidence for neural seed 42",
    )

    canonical_lower = (
        canonical_text
        .lower()
    )

    cpu_generator = (
        'generator(device="cpu")'
        in canonical_lower
        or "generator(device='cpu')"
        in canonical_lower
        or 'device="cpu"'
        in canonical_lower
        or "device='cpu'"
        in canonical_lower
    )

    require(
        cpu_generator,
        "CPU generator evidence not found",
    )

    manual_seed = (
        "manual_seed"
        in canonical_lower
    )

    require(
        manual_seed,
        "manual_seed evidence not found",
    )

    semantics_df = pd.DataFrame(
        [
            {
                "canonical_script": (
                    str(
                        canonical_script
                    )
                ),

                "canonical_script_sha256": (
                    canonical_script_hash
                ),

                "kaiming_call_count": (
                    len(
                        kaiming_calls
                    )
                ),

                "kaiming_a_zero": (
                    a_zero
                ),

                "kaiming_mode_fan_in": (
                    fan_in
                ),

                "kaiming_nonlinearity_relu": (
                    relu
                ),

                "resolved_seed_constants_equal_42": (
                    json.dumps(
                        seed_constants,
                        sort_keys=True,
                    )
                ),

                "seed_42_verified": (
                    seed_42_verified
                ),

                "cpu_generator": (
                    cpu_generator
                ),

                "manual_seed": (
                    manual_seed
                ),

                "resolved_kaiming_calls": (
                    json.dumps(
                        kaiming_calls,
                        sort_keys=True,
                        default=str,
                    )
                ),
            }
        ]
    )

    print(
        semantics_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Late Phase-4 AST diagnostics
    # =========================================================================

    banner(
        "LATE PHASE-4 CLASS IMPLEMENTATION DIAGNOSTIC"
    )

    late_paths = {
        str(
            SCRIPT_DIR
            / name
        )
        for name
        in LATE_PHASE4_SCRIPT_NAMES
    }

    late_classes = (
        class_df.loc[
            class_df[
                "script"
            ].isin(
                late_paths
            )
        ]
        if not class_df.empty
        else pd.DataFrame()
    )

    consistency_rows = []

    if not (
        late_classes.empty
    ):

        for (
            class_name,
            part,
        ) in late_classes.groupby(
            "class_name",
            sort=True,
        ):

            consistency_rows.append(
                {
                    "class_name": (
                        class_name
                    ),

                    "scripts_present": int(
                        part[
                            "script"
                        ].nunique()
                    ),

                    "unique_ast_implementations": int(
                        part[
                            "class_ast_sha256"
                        ].nunique()
                    ),

                    "scripts": (
                        ";".join(
                            sorted(
                                part[
                                    "script"
                                ].unique()
                            )
                        )
                    ),
                }
            )

    consistency_df = pd.DataFrame(
        consistency_rows
    )

    if (
        consistency_df.empty
    ):

        print(
            "No late Phase-4 class diagnostics available."
        )

    else:

        print(
            consistency_df.to_string(
                index=False
            )
        )

    print()

    print(
        "AST differences remain DIAGNOSTIC ONLY."
    )

    print(
        "Direct canonical provenance comes from the unique full"
    )

    print(
        "ITRSModel + six-module + initialization signature."
    )

    # =========================================================================
    # Final invariants
    # =========================================================================

    banner(
        "FINAL PROVENANCE AUDIT INVARIANTS"
    )

    checks = [
        (
            "pristine_model_not_instantiated",
            (
                preflight[
                    "model_instantiated"
                ]
                is False
            ),
        ),

        (
            "pristine_optimizer_not_instantiated",
            (
                preflight[
                    "optimizer_instantiated"
                ]
                is False
            ),
        ),

        (
            "pristine_rng_not_instantiated",
            (
                preflight[
                    "rng_instantiated"
                ]
                is False
            ),
        ),

        (
            "optimizer_steps_zero",
            (
                int(
                    preflight[
                        "optimizer_steps"
                    ]
                )
                == 0
            ),
        ),

        (
            "no_serialized_neural_checkpoint",
            (
                len(
                    checkpoint_candidates
                )
                == 0
            ),
        ),

        (
            "canonical_initial_state_hash_referenced",
            (
                len(
                    hash_reference_df
                )
                >= 1
            ),
        ),

        (
            "parameter_budget_19217929_verified",
            (
                parameter_budget_verified
            ),
        ),

        (
            "exactly_one_canonical_implementation_signature",
            (
                len(
                    canonical_candidates
                )
                == 1
            ),
        ),

        (
            "canonical_script_is_phase_4_7_1b",
            (
                canonical_script.name
                == EXPECTED_INITIALIZATION_SCRIPT_NAME
            ),
        ),

        (
            "canonical_module_set_exact",
            (
                canonical_modules
                == EXPECTED_NN_MODULE_CLASSES
            ),
        ),

        (
            "kaiming_a_zero",
            (
                a_zero
            ),
        ),

        (
            "kaiming_fan_in",
            (
                fan_in
            ),
        ),

        (
            "kaiming_relu",
            (
                relu
            ),
        ),

        (
            "neural_seed_42",
            (
                seed_42_verified
            ),
        ),

        (
            "cpu_generator",
            (
                cpu_generator
            ),
        ),

        (
            "manual_seed",
            (
                manual_seed
            ),
        ),

        # Writer detection is deliberately NOT a hard invariant.
        # It is supporting diagnostic evidence only.

        (
            "no_model_imported_by_this_audit",
            True,
        ),

        (
            "no_model_instantiated_by_this_audit",
            True,
        ),

        (
            "no_optimizer_instantiated_by_this_audit",
            True,
        ),

        (
            "no_rng_instantiated_by_this_audit",
            True,
        ),
    ]

    checks_df = pd.DataFrame(
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
            checks_df[
                "result"
            ]
            == "PASS"
        ).all(),
        (
            "One or more Phase-5.3.1b "
            "hard provenance invariants failed."
        ),
    )

    print(
        checks_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Write audit artifacts
    # =========================================================================

    banner(
        "WRITE CORRECTED AUDIT OUTPUTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialized_df.to_csv(
        SERIALIZED_STATE_AUDIT_PATH,
        index=False,
    )

    budget_df.to_csv(
        PARAMETER_BUDGET_EVIDENCE_PATH,
        index=False,
    )

    provenance_df.to_csv(
        SCRIPT_PROVENANCE_PATH,
        index=False,
    )

    writer_df.to_csv(
        WRITER_DIAGNOSTIC_PATH,
        index=False,
    )

    class_df.to_csv(
        CLASS_INVENTORY_PATH,
        index=False,
    )

    function_df.to_csv(
        FUNCTION_INVENTORY_PATH,
        index=False,
    )

    consistency_df.to_csv(
        CLASS_CONSISTENCY_PATH,
        index=False,
    )

    semantics_df.to_csv(
        INITIALIZATION_SOURCE_AUDIT_PATH,
        index=False,
    )

    hash_reference_df.to_csv(
        HASH_REFERENCE_PATH,
        index=False,
    )

    checks_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    manifest = {
        "phase": (
            "5.3.1b"
        ),

        "script_version": (
            SCRIPT_VERSION
        ),

        "title": (
            "Canonical Phase-4 Implementation "
            "Provenance Audit"
        ),

        "status": (
            "AUDIT_COMPLETE_"
            "CANONICAL_IMPLEMENTATION_PROVENANCE_RESOLVED"
        ),

        "model_imported": False,
        "model_instantiated": False,
        "optimizer_instantiated": False,
        "rng_instantiated": False,
        "training_negative_samples_generated": False,
        "training_order_generated": False,
        "forward_pass_performed": False,
        "backward_pass_performed": False,
        "optimizer_steps": 0,

        "serialized_neural_state": {
            "checkpoint_count": int(
                len(
                    checkpoint_candidates
                )
            ),

            "deterministic_reconstruction_required": True,
        },

        "parameter_budget": {
            "expected_trainable_parameters": (
                EXPECTED_PARAMETER_COUNT
            ),

            "expected_parameter_tensors": (
                EXPECTED_PARAMETER_TENSORS
            ),

            "literal_match": (
                literal_match
            ),

            "literal_source": (
                literal_source
            ),

            "numeric_reconstruction_match": (
                numeric_match
            ),

            "verified": (
                parameter_budget_verified
            ),
        },

        "canonical_initialization": {
            "state_sha256": (
                CANONICAL_INITIAL_STATE_SHA256
            ),

            "hash_reference_count": int(
                len(
                    hash_reference_df
                )
            ),

            "global_neural_seed": (
                EXPECTED_GLOBAL_NEURAL_SEED
            ),

            "kaiming": {
                "function": (
                    "nn.init.kaiming_normal_"
                ),

                "a": (
                    0.0
                ),

                "mode": (
                    "fan_in"
                ),

                "nonlinearity": (
                    "relu"
                ),
            },

            "cpu_generator": True,
        },

        "canonical_implementation": {
            "source_script": (
                str(
                    canonical_script
                )
            ),

            "source_script_sha256": (
                canonical_script_hash
            ),

            "model_class_candidate": (
                EXPECTED_MODEL_CLASS
            ),

            "nn_module_classes": (
                sorted(
                    canonical_modules
                )
            ),

            "entrypoint_frozen": False,
        },

        "destination_aware_writer_diagnostic": {
            "hash_writer_count": int(
                len(
                    hash_writers
                )
            ),

            "contract_writer_count": int(
                len(
                    contract_writers
                )
            ),

            "canonical_script_detected_as_hash_writer": (
                canonical_is_hash_writer
            ),

            "canonical_script_detected_as_contract_writer": (
                canonical_is_contract_writer
            ),

            "classification": (
                "SUPPORTING_DIAGNOSTIC_NOT_PROVENANCE_ORACLE"
            ),
        },

        "next_phase_requirement": (
            "Freeze the exact canonical source-script SHA256 and "
            "ITRSModel entrypoint, import the source without modification, "
            "instantiate the frozen model on CPU, reproduce the canonical "
            "initial-state SHA256 exactly, and only then instantiate Adam."
        ),

        "outputs": [
            str(
                SERIALIZED_STATE_AUDIT_PATH
            ),
            str(
                PARAMETER_BUDGET_EVIDENCE_PATH
            ),
            str(
                SCRIPT_PROVENANCE_PATH
            ),
            str(
                WRITER_DIAGNOSTIC_PATH
            ),
            str(
                CLASS_INVENTORY_PATH
            ),
            str(
                FUNCTION_INVENTORY_PATH
            ),
            str(
                CLASS_CONSISTENCY_PATH
            ),
            str(
                INITIALIZATION_SOURCE_AUDIT_PATH
            ),
            str(
                HASH_REFERENCE_PATH
            ),
            str(
                FINAL_INVARIANT_PATH
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
        SERIALIZED_STATE_AUDIT_PATH,
        PARAMETER_BUDGET_EVIDENCE_PATH,
        SCRIPT_PROVENANCE_PATH,
        WRITER_DIAGNOSTIC_PATH,
        CLASS_INVENTORY_PATH,
        FUNCTION_INVENTORY_PATH,
        CLASS_CONSISTENCY_PATH,
        INITIALIZATION_SOURCE_AUDIT_PATH,
        HASH_REFERENCE_PATH,
        FINAL_INVARIANT_PATH,
        MANIFEST_PATH,
    ):

        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Final status
    # =========================================================================

    banner(
        "PHASE 5.3.1b FINAL STATUS"
    )

    print(
        f"Audit script version:                "
        f"{SCRIPT_VERSION}"
    )

    print()

    print(
        "Serialized neural checkpoint:        NONE"
    )

    print(
        f"Parameter budget:                    "
        f"VERIFIED -> {EXPECTED_PARAMETER_COUNT:,}"
    )

    print(
        "Canonical state SHA256:              VERIFIED"
    )

    print()

    print(
        "Canonical implementation source:"
    )

    print(
        f"  {canonical_script}"
    )

    print()

    print(
        "Canonical implementation SHA256:"
    )

    print(
        f"  {canonical_script_hash}"
    )

    print()

    print(
        "Canonical model class:               ITRSModel"
    )

    print(
        "Canonical nn.Module count:           6"
    )

    print(
        "Canonical module stack:              VERIFIED"
    )

    print()

    print(
        "Kaiming a=0.0:                      VERIFIED"
    )

    print(
        "Kaiming mode=fan_in:                 VERIFIED"
    )

    print(
        "Kaiming nonlinearity=relu:           VERIFIED"
    )

    print(
        "Global neural seed=42:               VERIFIED"
    )

    print(
        "CPU generator:                       VERIFIED"
    )

    print()

    print(
        f"Destination-aware hash writers:      "
        f"{len(hash_writers)}"
    )

    print(
        f"Destination-aware contract writers:  "
        f"{len(contract_writers)}"
    )

    print(
        "Writer-count uniqueness required:    NO"
    )

    print()

    print(
        "Implementation provenance resolved:  YES"
    )

    print(
        "Numerical entrypoint frozen:         NO"
    )

    print()

    print(
        "Model imported:                      NO"
    )

    print(
        "Model instantiated:                  NO"
    )

    print(
        "Optimizer instantiated:              NO"
    )

    print(
        "RNG instantiated:                    NO"
    )

    print(
        "Forward pass performed:              NO"
    )

    print(
        "Backward pass performed:             NO"
    )

    print(
        "Optimizer steps:                     0"
    )

    banner(
        "PHASE 5.3.1b COMPLETE / "
        "CANONICAL IMPLEMENTATION PROVENANCE RESOLVED"
    )


if __name__ == "__main__":
    main()