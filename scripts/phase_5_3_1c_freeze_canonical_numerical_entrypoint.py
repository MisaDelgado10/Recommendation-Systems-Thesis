"""
Phase 5.3.1c — Freeze Canonical Numerical Entrypoint Contract

SOURCE-LEVEL FREEZE ONLY.

Purpose
-------
Phase 5.3.1b resolved the authoritative Phase-4 implementation:

    scripts/phase_4_7_1b_freeze_neural_initialization_seed_contract.py

with source SHA256:

    c55f3ea1646cec7fdc8ef69f2310d98f5ee95fab77f0c48392f4a9f76612761c

and canonical model class:

    ITRSModel

Before importing or instantiating that model, this phase freezes the
EXACT numerical entrypoint provenance:

1. canonical source path + SHA256;
2. canonical ITRSModel class AST hash;
3. exact ITRSModel.__init__ source signature;
4. exact Phase-4 ITRSModel construction call;
5. source context containing that construction call;
6. initialization helper-function provenance;
7. state-hash helper-function provenance.

THIS SCRIPT DOES NOT:
- import the canonical Phase-4 Python module;
- execute Phase-4 source code;
- instantiate ITRSModel;
- instantiate an RNG;
- instantiate Adam;
- generate negatives;
- generate batch order;
- perform forward/backward propagation;
- perform optimizer.step().

The resulting contract will be used by Phase 5.3.1d, where numerical
instantiation is finally allowed.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pandas as pd


# =============================================================================
# Frozen provenance from Phase 5.3.1b
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

CANONICAL_INITIAL_STATE_SHA256 = (
    "49e822ea7fad35c458f47e134c94c05e"
    "ac099b68c5c468e2c71559c8c88998ab"
)

EXPECTED_PARAMETER_COUNT = 19_217_929
EXPECTED_PARAMETER_TENSORS = 32

EXPECTED_NN_MODULE_CLASSES = {
    "DescriptionEncoder",
    "TrendExtractor",
    "BasisRGCNLayer",
    "PreferencePropagation",
    "ScoringMLP",
    "ITRSModel",
}


# =============================================================================
# Authoritative prior audit
# =============================================================================

PHASE_5_3_1B_MANIFEST = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1b/"
    "phase_5_3_1b_provenance_manifest.json"
)

PHASE_4_INIT_CONTRACT = Path(
    "data/experimental/phase_4/"
    "initialization_contract/"
    "phase_4_7_1b_neural_initialization_contract.json"
)

PHASE_4_INIT_HASH = Path(
    "data/experimental/phase_4/"
    "initialization_contract/"
    "phase_4_7_1b_initialization_state_hash.json"
)


# =============================================================================
# Outputs
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_5/contracts"
)

CONTRACT_PATH = (
    OUT_DIR
    / "phase_5_3_1c_canonical_numerical_entrypoint_contract.json"
)

DECISION_REGISTER_PATH = (
    OUT_DIR
    / "phase_5_3_1c_canonical_numerical_entrypoint_decision_register.csv"
)

FREEZE_AUDIT_PATH = (
    OUT_DIR
    / "phase_5_3_1c_canonical_numerical_entrypoint_freeze_audit.csv"
)

SOURCE_DETAIL_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1c/"
    "canonical_entrypoint_source_detail.csv"
)

FUNCTION_DETAIL_PATH = Path(
    "data/experimental/phase_5/audits/"
    "phase_5_3_1c/"
    "canonical_support_function_inventory.csv"
)


# =============================================================================
# Helpers
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

        left = dotted_name(
            node.value
        )

        if left:

            return (
                f"{left}.{node.attr}"
            )

        return (
            node.attr
        )

    return ""


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


def node_source(
    source: str,
    node: ast.AST,
) -> str:

    text = ast.get_source_segment(
        source,
        node,
    )

    if text is not None:

        return (
            text
        )

    return ast.unparse(
        node
    )


def function_contains_call(
    function_node: ast.AST,
    suffix: str,
) -> bool:

    for node in ast.walk(
        function_node
    ):

        if not isinstance(
            node,
            ast.Call,
        ):
            continue

        name = dotted_name(
            node.func
        )

        if name.endswith(
            suffix
        ):

            return True

    return False


def function_contains_name_fragment(
    function_node: ast.AST,
    fragment: str,
) -> bool:

    fragment = (
        fragment.lower()
    )

    for node in ast.walk(
        function_node
    ):

        if isinstance(
            node,
            ast.Name,
        ):

            if (
                fragment
                in node.id.lower()
            ):

                return True

        elif isinstance(
            node,
            ast.Attribute,
        ):

            if (
                fragment
                in node.attr.lower()
            ):

                return True

        elif (
            isinstance(
                node,
                ast.Constant,
            )
            and isinstance(
                node.value,
                str,
            )
        ):

            if (
                fragment
                in node.value.lower()
            ):

                return True

    return False


def serialize_argument(
    argument: ast.arg,
) -> dict:

    return {
        "name": (
            argument.arg
        ),

        "annotation": (
            ast.unparse(
                argument.annotation
            )
            if argument.annotation
            is not None
            else None
        ),
    }


def extract_function_signature(
    node: ast.FunctionDef,
) -> dict:
    """
    Static source signature.

    No module import occurs.
    """

    positional = (
        list(
            node.args.posonlyargs
        )
        + list(
            node.args.args
        )
    )

    defaults = (
        list(
            node.args.defaults
        )
    )

    default_offset = (
        len(
            positional
        )
        - len(
            defaults
        )
    )

    positional_rows = []

    for index, argument in enumerate(
        positional
    ):

        if (
            index
            >= default_offset
        ):

            default_node = (
                defaults[
                    index
                    - default_offset
                ]
            )

            default_text = (
                ast.unparse(
                    default_node
                )
            )

        else:

            default_text = None

        row = (
            serialize_argument(
                argument
            )
        )

        row[
            "default"
        ] = (
            default_text
        )

        row[
            "positional_only"
        ] = (
            index
            < len(
                node.args.posonlyargs
            )
        )

        positional_rows.append(
            row
        )

    keyword_only_rows = []

    for (
        argument,
        default,
    ) in zip(
        node.args.kwonlyargs,
        node.args.kw_defaults,
    ):

        row = (
            serialize_argument(
                argument
            )
        )

        row[
            "default"
        ] = (
            ast.unparse(
                default
            )
            if default is not None
            else None
        )

        keyword_only_rows.append(
            row
        )

    return {
        "positional": (
            positional_rows
        ),

        "vararg": (
            node.args.vararg.arg
            if node.args.vararg
            is not None
            else None
        ),

        "keyword_only": (
            keyword_only_rows
        ),

        "kwarg": (
            node.args.kwarg.arg
            if node.args.kwarg
            is not None
            else None
        ),

        "return_annotation": (
            ast.unparse(
                node.returns
            )
            if node.returns
            is not None
            else None
        ),
    }


# =============================================================================
# Model-construction call visitor
# =============================================================================

class ModelConstructionVisitor(
    ast.NodeVisitor
):
    """
    Find ITRSModel(...) calls and record the containing function.
    """

    def __init__(
        self,
    ) -> None:

        super().__init__()

        self.function_stack = []

        self.class_stack = []

        self.calls = []

    def visit_FunctionDef(
        self,
        node: ast.FunctionDef,
    ) -> None:

        self.function_stack.append(
            node.name
        )

        self.generic_visit(
            node
        )

        self.function_stack.pop()

    def visit_AsyncFunctionDef(
        self,
        node: ast.AsyncFunctionDef,
    ) -> None:

        self.function_stack.append(
            node.name
        )

        self.generic_visit(
            node
        )

        self.function_stack.pop()

    def visit_ClassDef(
        self,
        node: ast.ClassDef,
    ) -> None:

        self.class_stack.append(
            node.name
        )

        self.generic_visit(
            node
        )

        self.class_stack.pop()

    def visit_Call(
        self,
        node: ast.Call,
    ) -> None:

        name = dotted_name(
            node.func
        )

        if (
            name
            == CANONICAL_MODEL_CLASS
            or name.endswith(
                f".{CANONICAL_MODEL_CLASS}"
            )
        ):

            self.calls.append(
                {
                    "node": (
                        node
                    ),

                    "function_stack": (
                        list(
                            self.function_stack
                        )
                    ),

                    "class_stack": (
                        list(
                            self.class_stack
                        )
                    ),

                    "line_number": (
                        node.lineno
                    ),
                }
            )

        self.generic_visit(
            node
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    banner(
        "PHASE 5.3.1c — "
        "FREEZE CANONICAL NUMERICAL ENTRYPOINT CONTRACT"
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
        "Forward pass performed:               NO"
    )

    print(
        "Backward pass performed:              NO"
    )

    print(
        "Optimizer steps:                      0"
    )

    # =========================================================================
    # Inputs
    # =========================================================================

    banner(
        "AUTHORITATIVE INPUT RECHECK"
    )

    for path in (
        CANONICAL_SOURCE_PATH,
        PHASE_5_3_1B_MANIFEST,
        PHASE_4_INIT_CONTRACT,
        PHASE_4_INIT_HASH,
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

    actual_source_hash = (
        sha256_file(
            CANONICAL_SOURCE_PATH
        )
    )

    require(
        actual_source_hash
        == CANONICAL_SOURCE_SHA256,
        (
            "Canonical Phase-4 implementation source changed.\n"
            f"Expected: {CANONICAL_SOURCE_SHA256}\n"
            f"Actual:   {actual_source_hash}"
        ),
    )

    print()

    print(
        "Canonical source SHA256:              PASS"
    )

    # =========================================================================
    # Phase-5.3.1b recheck
    # =========================================================================

    banner(
        "PHASE 5.3.1b PROVENANCE RECHECK"
    )

    provenance = load_json(
        PHASE_5_3_1B_MANIFEST
    )

    require(
        provenance[
            "phase"
        ]
        == "5.3.1b",
        "Unexpected provenance phase",
    )

    require(
        provenance[
            "status"
        ]
        == (
            "AUDIT_COMPLETE_"
            "CANONICAL_IMPLEMENTATION_PROVENANCE_RESOLVED"
        ),
        (
            "Phase-5.3.1b provenance is not resolved"
        ),
    )

    require(
        provenance[
            "model_imported"
        ]
        is False,
        "5.3.1b unexpectedly imported model",
    )

    require(
        provenance[
            "model_instantiated"
        ]
        is False,
        "5.3.1b unexpectedly instantiated model",
    )

    require(
        provenance[
            "optimizer_instantiated"
        ]
        is False,
        "5.3.1b unexpectedly instantiated optimizer",
    )

    require(
        provenance[
            "rng_instantiated"
        ]
        is False,
        "5.3.1b unexpectedly instantiated RNG",
    )

    require(
        int(
            provenance[
                "optimizer_steps"
            ]
        )
        == 0,
        "5.3.1b optimizer-step count is not zero",
    )

    provenance_impl = (
        provenance[
            "canonical_implementation"
        ]
    )

    require(
        Path(
            provenance_impl[
                "source_script"
            ]
        )
        == CANONICAL_SOURCE_PATH,
        (
            "5.3.1b canonical source path drift"
        ),
    )

    require(
        provenance_impl[
            "source_script_sha256"
        ]
        == CANONICAL_SOURCE_SHA256,
        (
            "5.3.1b canonical source hash drift"
        ),
    )

    require(
        provenance_impl[
            "model_class_candidate"
        ]
        == CANONICAL_MODEL_CLASS,
        (
            "5.3.1b model-class drift"
        ),
    )

    print(
        "Provenance status:                    PASS"
    )

    print(
        "Canonical source:                     PASS"
    )

    print(
        "Canonical model class:                ITRSModel"
    )

    print(
        "Pristine numerical boundary:          PASS"
    )

    # =========================================================================
    # Parse canonical source
    # =========================================================================

    banner(
        "STATIC PARSE OF CANONICAL SOURCE"
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
        "Python AST parse:                     PASS"
    )

    # =========================================================================
    # Resolve canonical class
    # =========================================================================

    banner(
        "CANONICAL ITRSModel CLASS RESOLUTION"
    )

    model_classes = [
        node
        for node
        in tree.body
        if (
            isinstance(
                node,
                ast.ClassDef,
            )
            and node.name
            == CANONICAL_MODEL_CLASS
        )
    ]

    require(
        len(
            model_classes
        )
        == 1,
        (
            "Expected exactly one top-level ITRSModel class, "
            f"found {len(model_classes)}"
        ),
    )

    model_class = (
        model_classes[
            0
        ]
    )

    model_class_hash = (
        ast_sha256(
            model_class
        )
    )

    module_classes = {
        node.name
        for node
        in tree.body
        if (
            isinstance(
                node,
                ast.ClassDef,
            )
            and any(
                dotted_name(
                    base
                ).endswith(
                    "Module"
                )
                for base
                in node.bases
            )
        )
    }

    require(
        module_classes
        == EXPECTED_NN_MODULE_CLASSES,
        (
            "Canonical nn.Module set changed.\n"
            f"Expected: {sorted(EXPECTED_NN_MODULE_CLASSES)}\n"
            f"Found:    {sorted(module_classes)}"
        ),
    )

    print(
        "ITRSModel definitions:                1  PASS"
    )

    print(
        "Canonical six-module stack:           PASS"
    )

    print(
        "ITRSModel class AST SHA256:"
    )

    print(
        model_class_hash
    )

    # =========================================================================
    # Resolve __init__
    # =========================================================================

    banner(
        "ITRSModel CONSTRUCTOR SIGNATURE"
    )

    init_methods = [
        node
        for node
        in model_class.body
        if (
            isinstance(
                node,
                ast.FunctionDef,
            )
            and node.name
            == "__init__"
        )
    ]

    require(
        len(
            init_methods
        )
        == 1,
        (
            "Expected exactly one ITRSModel.__init__, "
            f"found {len(init_methods)}"
        ),
    )

    init_method = (
        init_methods[
            0
        ]
    )

    constructor_signature = (
        extract_function_signature(
            init_method
        )
    )

    constructor_source = (
        node_source(
            source,
            init_method,
        )
    )

    constructor_ast_hash = (
        ast_sha256(
            init_method
        )
    )

    print(
        constructor_source
    )

    print()

    print(
        "Constructor AST SHA256:"
    )

    print(
        constructor_ast_hash
    )

    # =========================================================================
    # Resolve actual Phase-4 ITRSModel(...) construction
    # =========================================================================

    banner(
        "PHASE-4 CANONICAL ITRSModel CONSTRUCTION CALL"
    )

    visitor = (
        ModelConstructionVisitor()
    )

    visitor.visit(
        tree
    )

    calls = (
        visitor.calls
    )

    require(
        len(
            calls
        )
        >= 1,
        (
            "No ITRSModel(...) construction call found "
            "in canonical source"
        ),
    )

    # Preferred resolution:
    # 1. exactly one construction call total;
    # 2. otherwise exactly one inside main().
    if (
        len(
            calls
        )
        == 1
    ):

        selected_call = (
            calls[
                0
            ]
        )

        selection_rule = (
            "unique_call_in_source"
        )

    else:

        main_calls = [
            candidate
            for candidate
            in calls
            if (
                "main"
                in candidate[
                    "function_stack"
                ]
            )
        ]

        require(
            len(
                main_calls
            )
            == 1,
            (
                "Multiple ITRSModel construction calls found "
                "and no unique main()-scoped call could be resolved.\n"
                f"Total calls: {len(calls)}\n"
                f"main() calls: {len(main_calls)}"
            ),
        )

        selected_call = (
            main_calls[
                0
            ]
        )

        selection_rule = (
            "unique_main_scoped_call"
        )

    call_node = (
        selected_call[
            "node"
        ]
    )

    call_source = (
        node_source(
            source,
            call_node,
        )
    )

    call_ast_hash = (
        ast_sha256(
            call_node
        )
    )

    positional_arguments = [
        ast.unparse(
            argument
        )
        for argument
        in call_node.args
    ]

    keyword_arguments = {
        keyword.arg: (
            ast.unparse(
                keyword.value
            )
        )
        for keyword
        in call_node.keywords
        if keyword.arg
        is not None
    }

    print(
        f"Construction calls found:             "
        f"{len(calls)}"
    )

    print(
        f"Selection rule:                       "
        f"{selection_rule}"
    )

    print(
        f"Containing function stack:            "
        f"{selected_call['function_stack']}"
    )

    print(
        f"Source line:                          "
        f"{selected_call['line_number']}"
    )

    print()

    print(
        "Canonical construction expression:"
    )

    print(
        call_source
    )

    print()

    print(
        "Construction-call AST SHA256:"
    )

    print(
        call_ast_hash
    )

    # =========================================================================
    # Inventory top-level support functions
    # =========================================================================

    banner(
        "CANONICAL SUPPORT-FUNCTION INVENTORY"
    )

    function_rows = []

    initialization_functions = []

    hash_functions = []

    construction_functions = []

    for node in tree.body:

        if not isinstance(
            node,
            ast.FunctionDef,
        ):
            continue

        contains_kaiming = (
            function_contains_call(
                node,
                "kaiming_normal_",
            )
        )

        contains_manual_seed = (
            function_contains_call(
                node,
                "manual_seed",
            )
        )

        contains_sha256 = (
            function_contains_call(
                node,
                "sha256",
            )
            or function_contains_name_fragment(
                node,
                "sha256",
            )
        )

        contains_state_dict = (
            function_contains_name_fragment(
                node,
                "state_dict",
            )
        )

        contains_parameter = (
            function_contains_name_fragment(
                node,
                "parameter",
            )
        )

        contains_model_constructor = False

        for child in ast.walk(
            node
        ):

            if not isinstance(
                child,
                ast.Call,
            ):
                continue

            name = dotted_name(
                child.func
            )

            if (
                name
                == CANONICAL_MODEL_CLASS
                or name.endswith(
                    f".{CANONICAL_MODEL_CLASS}"
                )
            ):

                contains_model_constructor = (
                    True
                )

                break

        initialization_candidate = (
            contains_kaiming
            or contains_manual_seed
        )

        hash_candidate = (
            contains_sha256
            and (
                contains_state_dict
                or contains_parameter
            )
        )

        if (
            initialization_candidate
        ):

            initialization_functions.append(
                node.name
            )

        if (
            hash_candidate
        ):

            hash_functions.append(
                node.name
            )

        if (
            contains_model_constructor
        ):

            construction_functions.append(
                node.name
            )

        function_rows.append(
            {
                "function_name": (
                    node.name
                ),

                "line_number": (
                    node.lineno
                ),

                "function_ast_sha256": (
                    ast_sha256(
                        node
                    )
                ),

                "contains_kaiming_normal": (
                    contains_kaiming
                ),

                "contains_manual_seed": (
                    contains_manual_seed
                ),

                "contains_sha256": (
                    contains_sha256
                ),

                "contains_state_dict": (
                    contains_state_dict
                ),

                "contains_parameter_reference": (
                    contains_parameter
                ),

                "contains_ITRSModel_constructor": (
                    contains_model_constructor
                ),

                "initialization_candidate": (
                    initialization_candidate
                ),

                "state_hash_candidate": (
                    hash_candidate
                ),
            }
        )

    function_df = pd.DataFrame(
        function_rows
    )

    print(
        function_df.to_string(
            index=False
        )
    )

    print()

    print(
        f"Initialization-support function(s):   "
        f"{initialization_functions}"
    )

    print(
        f"State-hash-support function(s):       "
        f"{hash_functions}"
    )

    print(
        f"Model-construction function(s):       "
        f"{construction_functions}"
    )

    # At least the initialization machinery MUST resolve.
    require(
        len(
            initialization_functions
        )
        >= 1,
        (
            "Could not locate the source function(s) "
            "containing canonical initialization logic"
        ),
    )

    # Hash may technically be implemented inline in main().
    # Therefore failure to isolate a standalone helper is diagnostic,
    # not fatal. The whole source SHA remains authoritative.

    # =========================================================================
    # Freeze decision register
    # =========================================================================

    banner(
        "FREEZE CANONICAL NUMERICAL ENTRYPOINT"
    )

    decisions = pd.DataFrame(
        [
            {
                "decision": (
                    "canonical_source"
                ),

                "value": (
                    str(
                        CANONICAL_SOURCE_PATH
                    )
                ),

                "classification": (
                    "INHERITED_PHASE4_IMPLEMENTATION_PROVENANCE"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1c"
                ),
            },

            {
                "decision": (
                    "canonical_source_sha256"
                ),

                "value": (
                    CANONICAL_SOURCE_SHA256
                ),

                "classification": (
                    "INHERITED_PHASE4_IMPLEMENTATION_PROVENANCE"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1c"
                ),
            },

            {
                "decision": (
                    "canonical_model_class"
                ),

                "value": (
                    CANONICAL_MODEL_CLASS
                ),

                "classification": (
                    "INHERITED_PHASE4_IMPLEMENTATION_PROVENANCE"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1c"
                ),
            },

            {
                "decision": (
                    "canonical_model_class_ast_sha256"
                ),

                "value": (
                    model_class_hash
                ),

                "classification": (
                    "IMPLEMENTATION_INTEGRITY_FINGERPRINT"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1c"
                ),
            },

            {
                "decision": (
                    "canonical_constructor_ast_sha256"
                ),

                "value": (
                    constructor_ast_hash
                ),

                "classification": (
                    "IMPLEMENTATION_INTEGRITY_FINGERPRINT"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1c"
                ),
            },

            {
                "decision": (
                    "canonical_construction_call_ast_sha256"
                ),

                "value": (
                    call_ast_hash
                ),

                "classification": (
                    "IMPLEMENTATION_INTEGRITY_FINGERPRINT"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1c"
                ),
            },

            {
                "decision": (
                    "canonical_initial_state_sha256"
                ),

                "value": (
                    CANONICAL_INITIAL_STATE_SHA256
                ),

                "classification": (
                    "INHERITED_FROZEN_PHASE4_NUMERICAL_ORACLE"
                ),

                "status": (
                    "FROZEN_PHASE_5_3_1c"
                ),
            },
        ]
    )

    # =========================================================================
    # Hard freeze invariants
    # =========================================================================

    checks = [
        (
            "canonical_source_hash_exact",
            (
                actual_source_hash
                == CANONICAL_SOURCE_SHA256
            ),
        ),

        (
            "phase_5_3_1b_provenance_resolved",
            (
                provenance[
                    "status"
                ]
                == (
                    "AUDIT_COMPLETE_"
                    "CANONICAL_IMPLEMENTATION_PROVENANCE_RESOLVED"
                )
            ),
        ),

        (
            "exactly_one_ITRSModel_definition",
            (
                len(
                    model_classes
                )
                == 1
            ),
        ),

        (
            "exact_six_module_stack",
            (
                module_classes
                == EXPECTED_NN_MODULE_CLASSES
            ),
        ),

        (
            "exactly_one_ITRSModel_init",
            (
                len(
                    init_methods
                )
                == 1
            ),
        ),

        (
            "canonical_construction_call_resolved",
            (
                selected_call
                is not None
            ),
        ),

        (
            "initialization_source_resolved",
            (
                len(
                    initialization_functions
                )
                >= 1
            ),
        ),

        (
            "no_module_import",
            True,
        ),

        (
            "no_model_instantiation",
            True,
        ),

        (
            "no_rng_instantiation",
            True,
        ),

        (
            "no_optimizer_instantiation",
            True,
        ),

        (
            "optimizer_steps_zero",
            True,
        ),
    ]

    freeze_audit_df = pd.DataFrame(
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
            freeze_audit_df[
                "result"
            ]
            == "PASS"
        ).all(),
        (
            "At least one canonical numerical-entrypoint "
            "freeze invariant failed"
        ),
    )

    print(
        freeze_audit_df.to_string(
            index=False
        )
    )

    # =========================================================================
    # Write contract
    # =========================================================================

    banner(
        "WRITE FROZEN PHASE-5.3.1c CONTRACT"
    )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SOURCE_DETAIL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    source_detail_df = pd.DataFrame(
        [
            {
                "source_path": (
                    str(
                        CANONICAL_SOURCE_PATH
                    )
                ),

                "source_sha256": (
                    CANONICAL_SOURCE_SHA256
                ),

                "model_class": (
                    CANONICAL_MODEL_CLASS
                ),

                "model_class_ast_sha256": (
                    model_class_hash
                ),

                "constructor_ast_sha256": (
                    constructor_ast_hash
                ),

                "constructor_source": (
                    constructor_source
                ),

                "construction_call_line": (
                    selected_call[
                        "line_number"
                    ]
                ),

                "construction_call_selection_rule": (
                    selection_rule
                ),

                "construction_call_ast_sha256": (
                    call_ast_hash
                ),

                "construction_call_source": (
                    call_source
                ),
            }
        ]
    )

    source_detail_df.to_csv(
        SOURCE_DETAIL_PATH,
        index=False,
    )

    function_df.to_csv(
        FUNCTION_DETAIL_PATH,
        index=False,
    )

    decisions.to_csv(
        DECISION_REGISTER_PATH,
        index=False,
    )

    freeze_audit_df.to_csv(
        FREEZE_AUDIT_PATH,
        index=False,
    )

    contract = {
        "phase": (
            "5.3.1c"
        ),

        "title": (
            "Freeze Canonical Numerical Entrypoint Contract"
        ),

        "status": (
            "FROZEN"
        ),

        "numerical_actions": {
            "canonical_module_imported": False,
            "model_instantiated": False,
            "rng_instantiated": False,
            "optimizer_instantiated": False,
            "forward_pass_performed": False,
            "backward_pass_performed": False,
            "optimizer_steps": 0,
        },

        "canonical_source": {
            "path": (
                str(
                    CANONICAL_SOURCE_PATH
                )
            ),

            "sha256": (
                CANONICAL_SOURCE_SHA256
            ),
        },

        "canonical_model": {
            "class_name": (
                CANONICAL_MODEL_CLASS
            ),

            "class_ast_sha256": (
                model_class_hash
            ),

            "expected_nn_module_classes": (
                sorted(
                    EXPECTED_NN_MODULE_CLASSES
                )
            ),

            "expected_trainable_parameters": (
                EXPECTED_PARAMETER_COUNT
            ),

            "expected_parameter_tensors": (
                EXPECTED_PARAMETER_TENSORS
            ),
        },

        "constructor": {
            "signature": (
                constructor_signature
            ),

            "source": (
                constructor_source
            ),

            "ast_sha256": (
                constructor_ast_hash
            ),
        },

        "canonical_construction_call": {
            "selection_rule": (
                selection_rule
            ),

            "line_number": (
                selected_call[
                    "line_number"
                ]
            ),

            "function_stack": (
                selected_call[
                    "function_stack"
                ]
            ),

            "class_stack": (
                selected_call[
                    "class_stack"
                ]
            ),

            "source": (
                call_source
            ),

            "ast_sha256": (
                call_ast_hash
            ),

            "positional_argument_expressions": (
                positional_arguments
            ),

            "keyword_argument_expressions": (
                keyword_arguments
            ),
        },

        "initialization_support": {
            "functions": (
                initialization_functions
            ),
        },

        "state_hash_support": {
            "functions": (
                hash_functions
            ),

            "note": (
                "A standalone state-hash helper is not mandatory; "
                "the canonical source SHA and Phase-4 numerical oracle "
                "remain authoritative if hash computation is inline."
            ),
        },

        "model_construction_support": {
            "functions": (
                construction_functions
            ),
        },

        "numerical_oracle": {
            "canonical_initial_state_sha256": (
                CANONICAL_INITIAL_STATE_SHA256
            ),

            "rule": (
                "Phase 5.3.1d MUST reproduce this exact hash "
                "before Adam may be instantiated."
            ),
        },

        "phase_5_3_1d_gate": {
            "source_sha256_must_match": True,
            "class_ast_sha256_must_match": True,
            "constructor_ast_sha256_must_match": True,
            "construction_call_ast_sha256_must_match": True,
            "parameter_count_must_equal": (
                EXPECTED_PARAMETER_COUNT
            ),
            "parameter_tensor_count_must_equal": (
                EXPECTED_PARAMETER_TENSORS
            ),
            "initial_state_sha256_must_equal": (
                CANONICAL_INITIAL_STATE_SHA256
            ),
            "optimizer_may_be_instantiated_only_after_all_model_checks_pass": (
                True
            ),
            "optimizer_step_forbidden": (
                True
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

    for path in (
        CONTRACT_PATH,
        DECISION_REGISTER_PATH,
        FREEZE_AUDIT_PATH,
        SOURCE_DETAIL_PATH,
        FUNCTION_DETAIL_PATH,
    ):

        print(
            f"WROTE  {path}"
        )

    # =========================================================================
    # Final
    # =========================================================================

    banner(
        "PHASE 5.3.1c FINAL STATUS"
    )

    print(
        "Canonical source:                     FROZEN"
    )

    print(
        "Canonical source SHA256:              FROZEN"
    )

    print(
        "Canonical class:                      FROZEN -> ITRSModel"
    )

    print(
        "ITRSModel class AST:                  FROZEN"
    )

    print(
        "ITRSModel constructor:                FROZEN"
    )

    print(
        "Canonical construction call:          FROZEN"
    )

    print(
        "Initialization support provenance:    FROZEN"
    )

    print(
        "Canonical initial-state oracle:       FROZEN"
    )

    print()

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
        "Optimizer steps:                      0"
    )

    banner(
        "PHASE 5.3.1c COMPLETE / "
        "CANONICAL NUMERICAL ENTRYPOINT FULLY FROZEN"
    )


if __name__ == "__main__":
    main()