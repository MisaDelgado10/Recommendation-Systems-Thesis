"""
Phase 5.3.1e — Canonical Import Side-Effect and Runtime Extraction Audit

STATIC AUDIT ONLY.

Background
----------
Phase 5.3.1d reproduced the exact frozen Phase-4 model state:

    32 parameter tensors
    19,217,929 trainable parameters
    SHA256:
    49e822ea7fad35c458f47e134c94c05e
    ac099b68c5c468e2c71559c8c88998ab

However, importing the canonical Phase-4 source caused its complete
Phase-4.7.1b audit workflow to execute.

Before the training runtime is allowed to import this implementation,
we must determine WHY import has side effects and which exact source
definitions are needed for numerical model construction.

THIS SCRIPT DOES NOT:
- import the canonical Phase-4 Python module;
- execute canonical Phase-4 source;
- instantiate ITRSModel;
- instantiate any torch.Generator;
- instantiate Adam;
- generate negatives;
- run forward/backward;
- perform optimizer.step();
- modify Phase-4 artifacts.

It performs static AST inspection only.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pandas as pd


# =============================================================================
# Frozen provenance
# =============================================================================

CANONICAL_SOURCE_PATH = Path(
    "scripts/"
    "phase_4_7_1b_freeze_neural_initialization_seed_contract.py"
)

CANONICAL_SOURCE_SHA256 = (
    "c55f3ea1646cec7fdc8ef69f2310d98f"
    "5ee95fab77f0c48392f4a9f76612761c"
)

PHASE_5_3_1D_MANIFEST_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1d/"
    "phase_5_3_1d_model_instantiation_manifest.json"
)

EXPECTED_INITIAL_STATE_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_PARAMETER_COUNT = 19_217_929
EXPECTED_PARAMETER_TENSORS = 32


# =============================================================================
# Canonical numerical symbols
# =============================================================================

REQUIRED_RUNTIME_SYMBOLS = {
    # Neural classes
    "DescriptionEncoder",
    "TrendExtractor",
    "BasisRGCNLayer",
    "PreferencePropagation",
    "ScoringMLP",
    "ITRSModel",

    # Numerical construction / initialization
    "apply_canonical_initialization",
    "build_canonical_model",

    # Integrity helpers required before training
    "model_parameter_state_sha256",
    "tensor_sha256",
    "count_parameters",
}


# =============================================================================
# Outputs
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1e"
)

TOP_LEVEL_AUDIT_PATH = (
    OUT_DIR
    / "canonical_top_level_statement_audit.csv"
)

TOP_LEVEL_CALL_PATH = (
    OUT_DIR
    / "canonical_top_level_call_audit.csv"
)

RUNTIME_SYMBOL_PATH = (
    OUT_DIR
    / "canonical_runtime_symbol_inventory.csv"
)

MAIN_GUARD_PATH = (
    OUT_DIR
    / "canonical_main_guard_audit.csv"
)

WRITE_RISK_PATH = (
    OUT_DIR
    / "canonical_import_write_risk_audit.csv"
)

FINAL_INVARIANT_PATH = (
    OUT_DIR
    / "phase_5_3_1e_final_invariants.csv"
)

MANIFEST_PATH = (
    OUT_DIR
    / "phase_5_3_1e_import_side_effect_manifest.json"
)


# =============================================================================
# Helpers
# =============================================================================

def banner(text: str) -> None:

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


def ast_sha256(
    node: ast.AST,
) -> str:

    payload = ast.dump(
        node,
        annotate_fields=True,
        include_attributes=False,
    )

    return hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).hexdigest()


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

        prefix = dotted_name(
            node.value
        )

        if prefix:

            return (
                f"{prefix}.{node.attr}"
            )

        return node.attr

    return ""


def source_segment(
    source: str,
    node: ast.AST,
) -> str:

    segment = ast.get_source_segment(
        source,
        node,
    )

    if segment is not None:

        return segment

    return ast.unparse(
        node
    )


def compact_source(
    text: str,
    limit: int = 500,
) -> str:

    normalized = (
        " ".join(
            text.split()
        )
    )

    if len(
        normalized
    ) <= limit:

        return normalized

    return (
        normalized[
            :limit
        ]
        + " ..."
    )


def statement_kind(
    node: ast.AST,
) -> str:

    return type(
        node
    ).__name__


def is_definition_only_statement(
    node: ast.AST,
) -> bool:
    """
    Statements normally safe at import because they define symbols
    rather than execute workflow.

    Assignments are handled separately because RHS expressions may
    themselves contain function calls.
    """

    return isinstance(
        node,
        (
            ast.Import,
            ast.ImportFrom,
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef,
        ),
    )


def node_contains_call(
    node: ast.AST,
) -> bool:

    return any(
        isinstance(
            candidate,
            ast.Call,
        )
        for candidate
        in ast.walk(
            node
        )
    )


def find_direct_calls(
    node: ast.AST,
) -> list[str]:

    calls = []

    for candidate in ast.walk(
        node
    ):

        if not isinstance(
            candidate,
            ast.Call,
        ):
            continue

        calls.append(
            dotted_name(
                candidate.func
            )
        )

    return calls


def is_name_main_comparison(
    node: ast.AST,
) -> bool:
    """
    Detect:

        __name__ == "__main__"
        "__main__" == __name__
    """

    if not isinstance(
        node,
        ast.Compare,
    ):

        return False

    if len(
        node.ops
    ) != 1:

        return False

    if not isinstance(
        node.ops[
            0
        ],
        ast.Eq,
    ):

        return False

    if len(
        node.comparators
    ) != 1:

        return False

    left = node.left

    right = (
        node.comparators[
            0
        ]
    )

    def is_name_node(
        candidate: ast.AST,
    ) -> bool:

        return (
            isinstance(
                candidate,
                ast.Name,
            )
            and candidate.id
            == "__name__"
        )

    def is_main_literal(
        candidate: ast.AST,
    ) -> bool:

        return (
            isinstance(
                candidate,
                ast.Constant,
            )
            and candidate.value
            == "__main__"
        )

    return (
        (
            is_name_node(
                left
            )
            and is_main_literal(
                right
            )
        )
        or
        (
            is_main_literal(
                left
            )
            and is_name_node(
                right
            )
        )
    )


def top_level_main_guards(
    tree: ast.Module,
) -> list[
    ast.If
]:

    return [
        node
        for node
        in tree.body
        if (
            isinstance(
                node,
                ast.If,
            )
            and is_name_main_comparison(
                node.test
            )
        )
    ]


def function_definitions(
    tree: ast.Module,
) -> dict[
    str,
    ast.FunctionDef,
]:

    return {
        node.name: node
        for node
        in tree.body
        if isinstance(
            node,
            ast.FunctionDef,
        )
    }


def class_definitions(
    tree: ast.Module,
) -> dict[
    str,
    ast.ClassDef,
]:

    return {
        node.name: node
        for node
        in tree.body
        if isinstance(
            node,
            ast.ClassDef,
        )
    }


def names_loaded_by_node(
    node: ast.AST,
) -> set[str]:

    names = set()

    for candidate in ast.walk(
        node
    ):

        if (
            isinstance(
                candidate,
                ast.Name,
            )
            and isinstance(
                candidate.ctx,
                ast.Load,
            )
        ):

            names.add(
                candidate.id
            )

    return names


def write_like_call(
    call: ast.Call,
) -> bool:

    name = (
        dotted_name(
            call.func
        )
        .lower()
    )

    final = (
        name.split(
            "."
        )[
            -1
        ]
    )

    if final in {
        "write_text",
        "write_bytes",
        "to_csv",
        "to_parquet",
        "to_json",
        "save",
        "savez",
        "savez_compressed",
        "dump",
        "write_json",
        "write_csv",
    }:

        return True

    if (
        final.startswith(
            "write_"
        )
        or final.startswith(
            "save_"
        )
    ):

        return True

    if final == "open":

        text = (
            ast.unparse(
                call
            )
            .lower()
        )

        return any(
            mode
            in text
            for mode
            in (
                "'w'",
                '"w"',
                "'wb'",
                '"wb"',
                "'a'",
                '"a"',
            )
        )

    return False


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.1e — "
        "CANONICAL IMPORT SIDE-EFFECT AND RUNTIME EXTRACTION AUDIT"
    )

    print(
        "Canonical module imported:            NO"
    )

    print(
        "Model instantiated:                   NO"
    )

    print(
        "RNG instantiated:                     NO"
    )

    print(
        "Optimizer instantiated:               NO"
    )

    print(
        "Training performed:                   NO"
    )

    print(
        "Optimizer steps:                      0"
    )

    # =========================================================================
    # Input integrity
    # =========================================================================

    banner(
        "AUTHORITATIVE INPUT RECHECK"
    )

    require(
        CANONICAL_SOURCE_PATH.exists(),
        (
            "Missing canonical Phase-4 source"
        ),
    )

    require(
        PHASE_5_3_1D_MANIFEST_PATH.exists(),
        (
            "Missing Phase-5.3.1d manifest"
        ),
    )

    actual_source_hash = (
        sha256_file(
            CANONICAL_SOURCE_PATH
        )
    )

    require(
        actual_source_hash
        == CANONICAL_SOURCE_SHA256,
        (
            "Canonical Phase-4 source changed.\n"
            f"Expected: {CANONICAL_SOURCE_SHA256}\n"
            f"Actual:   {actual_source_hash}"
        ),
    )

    phase_5_3_1d = load_json(
        PHASE_5_3_1D_MANIFEST_PATH
    )

    require(
        phase_5_3_1d[
            "phase"
        ]
        == "5.3.1d",
        (
            "Unexpected prior phase manifest"
        ),
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
            "Phase-5.3.1d numerical gate "
            "did not pass"
        ),
    )

    require(
        phase_5_3_1d[
            "initial_state"
        ][
            "actual_sha256"
        ]
        == EXPECTED_INITIAL_STATE_SHA256,
        (
            "Phase-5.3.1d canonical state hash drift"
        ),
    )

    require(
        int(
            phase_5_3_1d[
                "model"
            ][
                "parameter_tensor_count"
            ]
        )
        == EXPECTED_PARAMETER_TENSORS,
        (
            "Prior parameter-tensor count drift"
        ),
    )

    require(
        int(
            phase_5_3_1d[
                "model"
            ][
                "trainable_parameter_numel"
            ]
        )
        == EXPECTED_PARAMETER_COUNT,
        (
            "Prior trainable-parameter count drift"
        ),
    )

    require(
        phase_5_3_1d[
            "numerical_boundary"
        ][
            "optimizer_instantiated"
        ]
        is False,
        (
            "Optimizer unexpectedly instantiated "
            "before Phase-5.3.1e"
        ),
    )

    require(
        int(
            phase_5_3_1d[
                "numerical_boundary"
            ][
                "optimizer_steps"
            ]
        )
        == 0,
        (
            "Optimizer step occurred before "
            "Phase-5.3.1e"
        ),
    )

    print(
        "Canonical source SHA256:              PASS"
    )

    print(
        "Phase-5.3.1d state hash:              PASS"
    )

    print(
        "Parameter budget:                     PASS"
    )

    print(
        "Optimizer steps:                      0  PASS"
    )

    # =========================================================================
    # Parse source
    # =========================================================================

    banner(
        "STATIC AST PARSE"
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

    print(
        "AST parse:                            PASS"
    )

    # =========================================================================
    # Main guard
    # =========================================================================

    banner(
        "__main__ GUARD AUDIT"
    )

    guards = (
        top_level_main_guards(
            tree
        )
    )

    main_guard_rows = []

    for guard in (
        guards
    ):

        main_guard_rows.append(
            {
                "line_number": (
                    guard.lineno
                ),

                "source": (
                    compact_source(
                        source_segment(
                            source,
                            guard,
                        )
                    )
                ),

                "body_statement_count": (
                    len(
                        guard.body
                    )
                ),

                "body_calls": (
                    ";".join(
                        call
                        for statement
                        in guard.body
                        for call
                        in find_direct_calls(
                            statement
                        )
                    )
                ),
            }
        )

    main_guard_df = pd.DataFrame(
        main_guard_rows
    )

    print(
        f"Top-level __main__ guards:            "
        f"{len(guards)}"
    )

    if not (
        main_guard_df.empty
    ):

        print()

        print(
            main_guard_df.to_string(
                index=False
            )
        )

    # =========================================================================
    # Top-level statements
    # =========================================================================

    banner(
        "TOP-LEVEL STATEMENT AUDIT"
    )

    statement_rows = []

    executable_top_level_indices = []

    for (
        index,
        node,
    ) in enumerate(
        tree.body
    ):

        kind = (
            statement_kind(
                node
            )
        )

        direct_definition = (
            is_definition_only_statement(
                node
            )
        )

        contains_call = (
            node_contains_call(
                node
            )
        )

        main_guard = (
            isinstance(
                node,
                ast.If,
            )
            and is_name_main_comparison(
                node.test
            )
        )

        # Assignments with no calls are generally configuration/static
        # definitions. Assignments with calls may execute work on import.
        assignment = isinstance(
            node,
            (
                ast.Assign,
                ast.AnnAssign,
            ),
        )

        safe_static_assignment = (
            assignment
            and not contains_call
        )

        import_safe_definition = (
            direct_definition
            or safe_static_assignment
            or main_guard
        )

        import_execution_risk = (
            not import_safe_definition
        )

        if import_execution_risk:

            executable_top_level_indices.append(
                index
            )

        statement_rows.append(
            {
                "statement_index": (
                    index
                ),

                "line_number": (
                    getattr(
                        node,
                        "lineno",
                        None,
                    )
                ),

                "statement_type": (
                    kind
                ),

                "definition_statement": (
                    direct_definition
                ),

                "assignment": (
                    assignment
                ),

                "contains_call": (
                    contains_call
                ),

                "main_guard": (
                    main_guard
                ),

                "import_safe_definition": (
                    import_safe_definition
                ),

                "import_execution_risk": (
                    import_execution_risk
                ),

                "calls": (
                    ";".join(
                        find_direct_calls(
                            node
                        )
                    )
                ),

                "source": (
                    compact_source(
                        source_segment(
                            source,
                            node,
                        )
                    )
                ),
            }
        )

    top_level_df = pd.DataFrame(
        statement_rows
    )

    risky_top_level_df = (
        top_level_df.loc[
            top_level_df[
                "import_execution_risk"
            ]
        ]
    )

    print(
        f"Top-level statements:                 "
        f"{len(top_level_df)}"
    )

    print(
        f"Import-execution-risk statements:     "
        f"{len(risky_top_level_df)}"
    )

    if not (
        risky_top_level_df.empty
    ):

        print()

        print(
            risky_top_level_df[
                [
                    "statement_index",
                    "line_number",
                    "statement_type",
                    "calls",
                    "source",
                ]
            ].to_string(
                index=False
            )
        )

    # =========================================================================
    # Direct top-level calls
    # =========================================================================

    banner(
        "TOP-LEVEL CALL AUDIT"
    )

    call_rows = []

    for (
        statement_index,
        node,
    ) in enumerate(
        tree.body
    ):

        # Ignore function/class bodies here. They are not executed merely
        # because the definition exists.
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
            ),
        ):

            continue

        # Ignore protected __main__ block because it should not execute
        # on import.
        if (
            isinstance(
                node,
                ast.If,
            )
            and is_name_main_comparison(
                node.test
            )
        ):

            continue

        for candidate in ast.walk(
            node
        ):

            if not isinstance(
                candidate,
                ast.Call,
            ):

                continue

            call_rows.append(
                {
                    "statement_index": (
                        statement_index
                    ),

                    "line_number": (
                        candidate.lineno
                    ),

                    "call_name": (
                        dotted_name(
                            candidate.func
                        )
                    ),

                    "write_like": (
                        write_like_call(
                            candidate
                        )
                    ),

                    "call_source": (
                        compact_source(
                            source_segment(
                                source,
                                candidate,
                            )
                        )
                    ),
                }
            )

    top_level_call_df = pd.DataFrame(
        call_rows
    )

    print(
        f"Unprotected top-level calls:          "
        f"{len(top_level_call_df)}"
    )

    if not (
        top_level_call_df.empty
    ):

        print()

        print(
            top_level_call_df.to_string(
                index=False
            )
        )

    # =========================================================================
    # Runtime symbol inventory
    # =========================================================================

    banner(
        "CANONICAL NUMERICAL RUNTIME SYMBOL INVENTORY"
    )

    functions = (
        function_definitions(
            tree
        )
    )

    classes = (
        class_definitions(
            tree
        )
    )

    runtime_rows = []

    for symbol in sorted(
        REQUIRED_RUNTIME_SYMBOLS
    ):

        if symbol in classes:

            node = (
                classes[
                    symbol
                ]
            )

            kind = (
                "class"
            )

        elif symbol in functions:

            node = (
                functions[
                    symbol
                ]
            )

            kind = (
                "function"
            )

        else:

            node = None
            kind = None

        runtime_rows.append(
            {
                "symbol": (
                    symbol
                ),

                "found": (
                    node
                    is not None
                ),

                "kind": (
                    kind
                ),

                "line_number": (
                    node.lineno
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

                "loaded_names": (
                    ";".join(
                        sorted(
                            names_loaded_by_node(
                                node
                            )
                        )
                    )
                    if node
                    is not None
                    else None
                ),
            }
        )

    runtime_df = pd.DataFrame(
        runtime_rows
    )

    require(
        runtime_df[
            "found"
        ].all(),
        (
            "At least one required canonical numerical "
            "runtime symbol is absent"
        ),
    )

    print(
        runtime_df[
            [
                "symbol",
                "kind",
                "line_number",
                "ast_sha256",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # Import write-risk
    # =========================================================================

    banner(
        "IMPORT-TIME WRITE-RISK AUDIT"
    )

    if (
        top_level_call_df.empty
    ):

        write_risk_df = pd.DataFrame(
            columns=[
                "statement_index",
                "line_number",
                "call_name",
                "write_like",
                "call_source",
            ]
        )

    else:

        write_risk_df = (
            top_level_call_df.loc[
                top_level_call_df[
                    "write_like"
                ]
            ].copy()
        )

    print(
        f"Direct write-like top-level calls:    "
        f"{len(write_risk_df)}"
    )

    if not (
        write_risk_df.empty
    ):

        print()

        print(
            write_risk_df.to_string(
                index=False
            )
        )

    # =========================================================================
    # Interpretation
    # =========================================================================

    banner(
        "IMPORT-SIDE-EFFECT INTERPRETATION"
    )

    unprotected_calls = (
        len(
            top_level_call_df
        )
    )

    import_side_effect_risk = (
        unprotected_calls
        > 0
    )

    if import_side_effect_risk:

        print(
            "Canonical Phase-4 file is NOT safe to use as a normal"
        )

        print(
            "training-library import without an explicit runtime-loading"
        )

        print(
            "policy."
        )

    else:

        print(
            "No unprotected top-level calls were found."
        )

        print(
            "The prior observed replay must therefore be explained by"
        )

        print(
            "another import-time mechanism before proceeding."
        )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "  This does NOT alter canonical neural implementation provenance."
    )

    print(
        "  It only determines how that frozen source may safely be loaded"
    )

    print(
        "  during Phase-5 training."
    )

    # =========================================================================
    # Hard invariants
    # =========================================================================

    banner(
        "FINAL PHASE-5.3.1e INVARIANTS"
    )

    checks = [
        (
            "canonical_source_hash_exact",
            (
                actual_source_hash
                == CANONICAL_SOURCE_SHA256
            ),
        ),

        (
            "phase_5_3_1d_passed",
            (
                phase_5_3_1d[
                    "status"
                ]
                == (
                    "NUMERICAL_MODEL_INSTANTIATION_PASSED_"
                    "OPTIMIZER_NOT_CREATED"
                )
            ),
        ),

        (
            "canonical_state_hash_exact",
            (
                phase_5_3_1d[
                    "initial_state"
                ][
                    "actual_sha256"
                ]
                == EXPECTED_INITIAL_STATE_SHA256
            ),
        ),

        (
            "optimizer_still_not_instantiated",
            (
                phase_5_3_1d[
                    "numerical_boundary"
                ][
                    "optimizer_instantiated"
                ]
                is False
            ),
        ),

        (
            "optimizer_steps_zero",
            (
                int(
                    phase_5_3_1d[
                        "numerical_boundary"
                    ][
                        "optimizer_steps"
                    ]
                )
                == 0
            ),
        ),

        (
            "all_required_runtime_symbols_found",
            (
                bool(
                    runtime_df[
                        "found"
                    ].all()
                )
            ),
        ),

        (
            "no_module_import_by_this_audit",
            True,
        ),

        (
            "no_model_instantiation_by_this_audit",
            True,
        ),

        (
            "no_rng_instantiation_by_this_audit",
            True,
        ),

        (
            "no_optimizer_instantiation_by_this_audit",
            True,
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
        (
            invariant_df[
                "result"
            ]
            == "PASS"
        ).all(),
        (
            "At least one Phase-5.3.1e "
            "import-side-effect audit invariant failed"
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
        "WRITE PHASE-5.3.1e AUDIT OUTPUTS"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    top_level_df.to_csv(
        TOP_LEVEL_AUDIT_PATH,
        index=False,
    )

    top_level_call_df.to_csv(
        TOP_LEVEL_CALL_PATH,
        index=False,
    )

    runtime_df.to_csv(
        RUNTIME_SYMBOL_PATH,
        index=False,
    )

    main_guard_df.to_csv(
        MAIN_GUARD_PATH,
        index=False,
    )

    write_risk_df.to_csv(
        WRITE_RISK_PATH,
        index=False,
    )

    invariant_df.to_csv(
        FINAL_INVARIANT_PATH,
        index=False,
    )

    manifest = {
        "phase": (
            "5.3.1e"
        ),

        "title": (
            "Canonical Import Side-Effect "
            "and Runtime Extraction Audit"
        ),

        "status": (
            "AUDIT_COMPLETE_RUNTIME_LOADING_POLICY_NOT_YET_FROZEN"
        ),

        "canonical_source": {
            "path": (
                str(
                    CANONICAL_SOURCE_PATH
                )
            ),

            "sha256": (
                actual_source_hash
            ),
        },

        "phase_5_3_1d": {
            "canonical_state_sha256": (
                phase_5_3_1d[
                    "initial_state"
                ][
                    "actual_sha256"
                ]
            ),

            "parameter_count": (
                phase_5_3_1d[
                    "model"
                ][
                    "trainable_parameter_numel"
                ]
            ),

            "parameter_tensor_count": (
                phase_5_3_1d[
                    "model"
                ][
                    "parameter_tensor_count"
                ]
            ),

            "optimizer_instantiated": False,

            "optimizer_steps": 0,
        },

        "import_analysis": {
            "top_level_statement_count": (
                len(
                    top_level_df
                )
            ),

            "main_guard_count": (
                len(
                    guards
                )
            ),

            "unprotected_top_level_call_count": (
                len(
                    top_level_call_df
                )
            ),

            "direct_top_level_write_like_call_count": (
                len(
                    write_risk_df
                )
            ),

            "import_side_effect_risk": (
                import_side_effect_risk
            ),
        },

        "runtime_symbols": {
            "required": (
                sorted(
                    REQUIRED_RUNTIME_SYMBOLS
                )
            ),

            "all_found": (
                bool(
                    runtime_df[
                        "found"
                    ].all()
                )
            ),
        },

        "numerical_actions": {
            "canonical_module_imported": False,
            "model_instantiated": False,
            "rng_instantiated": False,
            "optimizer_instantiated": False,
            "training_performed": False,
            "optimizer_steps": 0,
        },

        "decision_status": {
            "runtime_loading_policy_frozen": False,

            "reason": (
                "Canonical source import replayed Phase-4 audit workflow; "
                "static side-effect structure must be reviewed before "
                "the training loader is frozen."
            ),
        },

        "next_phase_requirement": (
            "Freeze a side-effect-free Phase-5 runtime loading mechanism "
            "that preserves the exact AST/source definitions required for "
            "ITRSModel, canonical initialization, and state hashing without "
            "re-executing the Phase-4 audit workflow."
        ),

        "outputs": [
            str(
                TOP_LEVEL_AUDIT_PATH
            ),
            str(
                TOP_LEVEL_CALL_PATH
            ),
            str(
                RUNTIME_SYMBOL_PATH
            ),
            str(
                MAIN_GUARD_PATH
            ),
            str(
                WRITE_RISK_PATH
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
        TOP_LEVEL_AUDIT_PATH,
        TOP_LEVEL_CALL_PATH,
        RUNTIME_SYMBOL_PATH,
        MAIN_GUARD_PATH,
        WRITE_RISK_PATH,
        FINAL_INVARIANT_PATH,
        MANIFEST_PATH,
    ):

        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Final
    # =========================================================================

    banner(
        "PHASE 5.3.1e FINAL STATUS"
    )

    print(
        "Canonical Phase-4 implementation:     UNCHANGED"
    )

    print(
        "Canonical state oracle:               UNCHANGED"
    )

    print(
        "Phase-5.3.1d numerical gate:          PASS"
    )

    print()

    print(
        f"Top-level __main__ guards:            "
        f"{len(guards)}"
    )

    print(
        f"Unprotected top-level calls:          "
        f"{len(top_level_call_df)}"
    )

    print(
        f"Direct top-level write-like calls:    "
        f"{len(write_risk_df)}"
    )

    print()

    print(
        f"Import-side-effect risk:              "
        f"{'YES' if import_side_effect_risk else 'NO'}"
    )

    print(
        "Runtime loading policy frozen:        NO"
    )

    print()

    print(
        "Canonical module imported by audit:   NO"
    )

    print(
        "Model instantiated by audit:          NO"
    )

    print(
        "Optimizer instantiated:               NO"
    )

    print(
        "Optimizer steps:                      0"
    )

    banner(
        "PHASE 5.3.1e COMPLETE / "
        "IMPORT SIDE-EFFECT STRUCTURE AUDITED"
    )


if __name__ == "__main__":
    main()