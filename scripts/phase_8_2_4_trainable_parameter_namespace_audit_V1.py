from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import sys

import torch


# =============================================================================
# PATHS
# =============================================================================

MODEL_MODULE_PATH = Path(
    "scripts/"
    "phase_8_2_3_candidate_model_modules_V1.py"
)

PREPROCESS_ROOT = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_1_feature_materialization/"
    "phase_8_2_1d_preprocessing_V2"
)

OUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_4_deterministic_initialization"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_PATH = (
    OUT_DIR
    / "phase_8_2_4_trainable_parameter_namespace_V1.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "phase_8_2_4_trainable_parameter_namespace_summary_V1.json"
)

HASH_PATH = (
    OUT_DIR
    / "phase_8_2_4_trainable_parameter_namespace_sha256_V1.json"
)


# =============================================================================
# FROZEN RANDOMNESS CONTRACT
# =============================================================================

BASE_SEED = 42

NAMESPACE = (
    "ITRS_PHASE8_NEURAL_INIT"
)

PAYLOAD_TEMPLATE = (
    "{namespace}|base_seed=42|scope={scope}|"
    "epoch=NA|component={component}|"
    "parameter={parameter}|shape={shape}|"
    "dtype={dtype}|replicate=NA|attempt=0|dimension=NA"
)


# =============================================================================
# EXECUTION SCOPES
# =============================================================================

SCOPE_CONFIG = {
    "DEV_T57": {
        "origin": "t57",
        "families": (
            "B1a",
            "B1b",
            "M1",
            "M2",
        ),
    },

    "DEV_T58": {
        "origin": "t58",
        "families": (
            "B1a",
            "B1b",
            "M1",
            "M2",
        ),
    },

    "DEV_T59": {
        "origin": "t59",
        "families": (
            "B1a",
            "B1b",
            "M1",
            "M2",
        ),
    },

    "T60_FINAL_IMPROVED": {
        "origin": "t60",
        "families": (
            "B1a",
            "B1b",
            "M1",
            "M2",
        ),
    },

    "APRIL_FINAL_M2": {
        "origin": "april",
        "families": (
            "M2",
        ),
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def require(
    condition,
    message,
):

    if not condition:
        raise RuntimeError(
            message
        )


def sha256_file(
    path,
):

    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


def shape_string(
    parameter,
):

    require(
        parameter.ndim >= 1,
        "Trainable scalar parameter is unsupported by frozen initialization.",
    )

    return "x".join(
        str(
            int(dim)
        )
        for dim in parameter.shape
    )


def dtype_string(
    parameter,
):

    value = str(
        parameter.dtype
    )

    require(
        value.startswith(
            "torch."
        ),
        (
            "Unexpected torch dtype serialization: "
            f"{value}"
        ),
    )

    return value[
        len("torch.") :
    ]


def derive_seed(
    *,
    scope,
    component,
    parameter_name,
    shape,
    dtype,
):

    payload = PAYLOAD_TEMPLATE.format(
        namespace=NAMESPACE,
        scope=scope,
        component=component,
        parameter=parameter_name,
        shape=shape,
        dtype=dtype,
    )

    digest = hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).digest()

    seed_u64 = int.from_bytes(
        digest[
            :8
        ],
        byteorder="big",
        signed=False,
    )

    return (
        payload,
        seed_u64,
    )


def resolve_semantic_component(
    model,
    parameter_name,
):

    parts = (
        parameter_name
        .split(".")
    )

    # Remove leaf parameter name.
    module_parts = parts[
        :-1
    ]

    # Deepest / nearest owning ancestor first.
    for depth in range(
        len(module_parts),
        -1,
        -1,
    ):

        module_path = ".".join(
            module_parts[
                :depth
            ]
        )

        if module_path:

            owner = model.get_submodule(
                module_path
            )

        else:

            owner = model

        component = getattr(
            owner,
            "semantic_component_id",
            None,
        )

        if component is not None:

            require(
                isinstance(
                    component,
                    str,
                )
                and bool(
                    component.strip()
                ),
                (
                    "Invalid semantic_component_id "
                    f"for parameter {parameter_name}."
                ),
            )

            return component

    raise RuntimeError(
        "No semantic_component_id resolved for "
        f"parameter: {parameter_name}"
    )


def initialization_rule(
    parameter_name,
    parameter,
):

    leaf = (
        parameter_name
        .split(".")[
            -1
        ]
    )

    if leaf.startswith(
        "bias"
    ) or leaf.endswith(
        "bias"
    ):

        return (
            "ZEROS"
        )

    require(
        parameter.ndim >= 2,
        (
            "Frozen non-bias rule is Kaiming normal, "
            "but parameter is not rank >= 2: "
            f"{parameter_name} {tuple(parameter.shape)}"
        ),
    )

    return (
        "KAIMING_NORMAL_A0_FAN_IN_RELU"
    )


def load_model_module():

    module_name = (
        "phase8_candidate_modules_for_namespace_audit"
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            module_name,
            MODEL_MODULE_PATH,
        )
    )

    require(
        spec is not None
        and spec.loader is not None,
        "Could not create model-module import spec.",
    )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    sys.modules[
        module_name
    ] = module

    spec.loader.exec_module(
        module
    )

    return module


def load_origin_dimensions(
    origin,
):

    vocab_path = (
        PREPROCESS_ROOT
        / f"phase_8_2_1d_{origin}_vocabularies_V2.json"
    )

    require(
        vocab_path.exists(),
        (
            "Missing origin vocabulary artifact: "
            f"{vocab_path}"
        ),
    )

    vocab = json.loads(
        vocab_path.read_text(
            encoding="utf-8"
        )
    )

    category_dim = len(
        vocab[
            "categories"
        ]
    )

    base_structured_dim = (
        len(
            vocab[
                "category_groups"
            ]
        )
        + len(
            vocab[
                "country"
            ]
        )
        + len(
            vocab[
                "region_location_groups"
            ]
        )
        + len(
            vocab[
                "company_type"
            ]
        )
        + 2
    )

    extended_structured_dim = (
        base_structured_dim
        + 1
        + len(
            vocab[
                "prior_sector"
            ]
        )
    )

    return {
        "vocab_path":
            str(
                vocab_path
            ),

        "category_dim":
            category_dim,

        "base_structured_dim":
            base_structured_dim,

        "extended_structured_dim":
            extended_structured_dim,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 120
    )

    print(
        "PHASE 8.2.4 — TRAINABLE PARAMETER NAMESPACE AUDIT V1"
    )

    print(
        "=" * 120
    )

    require(
        MODEL_MODULE_PATH.exists(),
        (
            "Missing Phase-8.2.3 model module: "
            f"{MODEL_MODULE_PATH}"
        ),
    )

    module = load_model_module()

    investor_label_dim = 802

    rows = []

    scope_dimension_summary = {}

    model_parameter_counts = {}

    # -------------------------------------------------------------------------
    # Enumerate exact model/scope parameter namespaces on META.
    # -------------------------------------------------------------------------

    for scope, config in SCOPE_CONFIG.items():

        origin = config[
            "origin"
        ]

        dims = load_origin_dimensions(
            origin
        )

        scope_dimension_summary[
            scope
        ] = {
            "origin":
                origin,

            **dims,
        }

        for family in config[
            "families"
        ]:

            if family == "B1a":

                structured_dim = None

            elif family == "B1b":

                structured_dim = dims[
                    "base_structured_dim"
                ]

            elif family in (
                "M1",
                "M2",
            ):

                structured_dim = dims[
                    "extended_structured_dim"
                ]

            else:

                raise RuntimeError(
                    f"Unknown family: {family}"
                )

            with torch.device(
                "meta"
            ):

                model = (
                    module
                    .Phase8CandidateModel(
                        model_family=family,
                        investor_label_input_dim=(
                            investor_label_dim
                        ),
                        startup_category_input_dim=(
                            dims[
                                "category_dim"
                            ]
                        ),
                        startup_structured_input_dim=(
                            structured_dim
                        ),
                    )
                )

            parameter_counter = 0

            for (
                parameter_name,
                parameter,
            ) in model.named_parameters():

                require(
                    parameter.requires_grad,
                    (
                        "Unexpected frozen parameter encountered: "
                        f"{family}/{parameter_name}"
                    ),
                )

                require(
                    parameter.device.type
                    == "meta",
                    (
                        "Namespace audit allocated a real parameter: "
                        f"{family}/{parameter_name}"
                    ),
                )

                component = (
                    resolve_semantic_component(
                        model,
                        parameter_name,
                    )
                )

                shape = shape_string(
                    parameter
                )

                dtype = dtype_string(
                    parameter
                )

                rule = initialization_rule(
                    parameter_name,
                    parameter,
                )

                (
                    payload,
                    seed_u64,
                ) = derive_seed(
                    scope=scope,
                    component=component,
                    parameter_name=parameter_name,
                    shape=shape,
                    dtype=dtype,
                )

                rows.append(
                    {
                        "model":
                            family,

                        "scope":
                            scope,

                        "origin":
                            origin,

                        "semantic_component_id":
                            component,

                        "parameter_name":
                            parameter_name,

                        "shape":
                            shape,

                        "dtype":
                            dtype,

                        "numel":
                            int(
                                parameter.numel()
                            ),

                        "derived_seed_u64":
                            seed_u64,

                        "initialization_rule":
                            rule,

                        "payload":
                            payload,
                    }
                )

                parameter_counter += 1

            model_parameter_counts[
                f"{scope}|{family}"
            ] = parameter_counter

    # -------------------------------------------------------------------------
    # Uniqueness: every trainable tensor exactly once per model/scope.
    # -------------------------------------------------------------------------

    row_keys = [
        (
            row[
                "model"
            ],
            row[
                "scope"
            ],
            row[
                "parameter_name"
            ],
        )
        for row in rows
    ]

    require(
        len(
            row_keys
        )
        == len(
            set(
                row_keys
            )
        ),
        (
            "Duplicate model/scope/parameter rows detected."
        ),
    )

    # Expected topology count:
    # four full candidate scopes:
    #   29 + 33 + 31 + 41 = 134 each
    # plus April M2 = 41
    expected_total_rows = (
        4
        * (
            29
            + 33
            + 31
            + 41
        )
        + 41
    )

    require(
        len(
            rows
        )
        == expected_total_rows,
        (
            "Unexpected total namespace row count: "
            f"{len(rows)} != {expected_total_rows}"
        ),
    )

    # -------------------------------------------------------------------------
    # Cross-family shared-tensor seed invariance within the same scope.
    # -------------------------------------------------------------------------

    by_scope_signature = {}

    for row in rows:

        signature = (
            row[
                "scope"
            ],
            row[
                "semantic_component_id"
            ],
            row[
                "parameter_name"
            ],
            row[
                "shape"
            ],
            row[
                "dtype"
            ],
        )

        by_scope_signature.setdefault(
            signature,
            []
        ).append(
            row
        )

    shared_groups = 0

    for signature, group in by_scope_signature.items():

        if len(
            group
        ) <= 1:
            continue

        shared_groups += 1

        seeds = {
            row[
                "derived_seed_u64"
            ]
            for row in group
        }

        payloads = {
            row[
                "payload"
            ]
            for row in group
        }

        require(
            len(
                seeds
            )
            == 1,
            (
                "Shared semantic tensor received "
                "different seeds: "
                f"{signature}"
            ),
        )

        require(
            len(
                payloads
            )
            == 1,
            (
                "Shared semantic tensor received "
                "different payloads: "
                f"{signature}"
            ),
        )

    # -------------------------------------------------------------------------
    # Same parameter signature across different scopes must use different seeds.
    # -------------------------------------------------------------------------

    cross_scope = {}

    for row in rows:

        signature = (
            row[
                "semantic_component_id"
            ],
            row[
                "parameter_name"
            ],
            row[
                "shape"
            ],
            row[
                "dtype"
            ],
        )

        cross_scope.setdefault(
            signature,
            {}
        ).setdefault(
            row[
                "scope"
            ],
            set(),
        ).add(
            row[
                "derived_seed_u64"
            ]
        )

    compared_cross_scope_groups = 0

    for signature, scope_map in cross_scope.items():

        if len(
            scope_map
        ) <= 1:
            continue

        compared_cross_scope_groups += 1

        one_seed_per_scope = {}

        for scope, seeds in scope_map.items():

            require(
                len(
                    seeds
                )
                == 1,
                (
                    "Same semantic tensor has more than "
                    "one seed inside a scope: "
                    f"{signature}, {scope}"
                ),
            )

            one_seed_per_scope[
                scope
            ] = next(
                iter(
                    seeds
                )
            )

        require(
            len(
                set(
                    one_seed_per_scope.values()
                )
            )
            == len(
                one_seed_per_scope
            ),
            (
                "Same semantic tensor reused a seed "
                "across different scopes: "
                f"{signature}"
            ),
        )

    # -------------------------------------------------------------------------
    # Rule sanity.
    # -------------------------------------------------------------------------

    bias_rows = [
        row
        for row in rows
        if row[
            "initialization_rule"
        ]
        == "ZEROS"
    ]

    matrix_rows = [
        row
        for row in rows
        if row[
            "initialization_rule"
        ]
        == "KAIMING_NORMAL_A0_FAN_IN_RELU"
    ]

    require(
        len(
            bias_rows
        )
        + len(
            matrix_rows
        )
        == len(
            rows
        ),
        (
            "Unclassified initialization rule exists."
        ),
    )

    # -------------------------------------------------------------------------
    # CSV.
    # -------------------------------------------------------------------------

    fieldnames = [
        "model",
        "scope",
        "origin",
        "semantic_component_id",
        "parameter_name",
        "shape",
        "dtype",
        "numel",
        "derived_seed_u64",
        "initialization_rule",
        "payload",
    ]

    with CSV_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # -------------------------------------------------------------------------
    # Summary.
    # -------------------------------------------------------------------------

    summary = {
        "schema_version":
            "PHASE_8_2_4_TRAINABLE_PARAMETER_NAMESPACE_AUDIT_V1",

        "status":
            "MATERIALIZED_AND_AUDITED",

        "namespace":
            NAMESPACE,

        "base_seed":
            BASE_SEED,

        "parameter_name_rule":
            "FULL_MODEL_NAMED_PARAMETERS_KEY",

        "semantic_component_rule":
            "NEAREST_SEMANTIC_MODULE_ANCESTOR",

        "shape_serialization":
            "LOWERCASE_X_JOINED_DIMENSIONS",

        "dtype_serialization":
            "TORCH_DTYPE_WITHOUT_TORCH_PREFIX",

        "total_namespace_rows":
            len(
                rows
            ),

        "expected_namespace_rows":
            expected_total_rows,

        "bias_rows":
            len(
                bias_rows
            ),

        "matrix_rows":
            len(
                matrix_rows
            ),

        "shared_within_scope_groups_audited":
            shared_groups,

        "cross_scope_groups_audited":
            compared_cross_scope_groups,

        "scope_dimensions":
            scope_dimension_summary,

        "model_parameter_tensor_counts":
            model_parameter_counts,

        "execution_state": {
            "model_instantiation_device":
                "meta",

            "real_parameter_values_allocated":
                False,

            "canonical_initialization_applied":
                False,

            "forward_executed":
                False,

            "optimizer_created":
                False,

            "training_performed":
                False,

            "performance_inspected":
                False,

            "T60_scored":
                False,

            "April_analyzed":
                False,
        },

        "next_step":
            (
                "Implement and audit canonical CPU "
                "parameter initialization."
            ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    hash_registry = {
        str(
            MODEL_MODULE_PATH
        ):
            sha256_file(
                MODEL_MODULE_PATH
            ),

        str(
            CSV_PATH
        ):
            sha256_file(
                CSV_PATH
            ),

        str(
            SUMMARY_PATH
        ):
            sha256_file(
                SUMMARY_PATH
            ),
    }

    HASH_PATH.write_text(
        json.dumps(
            hash_registry,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Compact terminal report only.
    # -------------------------------------------------------------------------

    print(
        "Namespace rows:                         "
        f"{len(rows):,}"
    )

    print(
        "Bias tensors (ZERO):                    "
        f"{len(bias_rows):,}"
    )

    print(
        "Matrix tensors (KAIMING):               "
        f"{len(matrix_rows):,}"
    )

    print(
        "Shared within-scope groups audited:      "
        f"{shared_groups:,}"
    )

    print(
        "Cross-scope groups audited:              "
        f"{compared_cross_scope_groups:,}"
    )

    print()

    for scope, dims in scope_dimension_summary.items():

        print(
            f"{scope:20s} | "
            f"categories={dims['category_dim']:4d} | "
            f"base={dims['base_structured_dim']:4d} | "
            f"extended={dims['extended_structured_dim']:4d}"
        )

    print()
    print(
        f"CSV:     {CSV_PATH}"
    )

    print(
        f"Summary: {SUMMARY_PATH}"
    )

    print(
        f"Hashes:  {HASH_PATH}"
    )

    print()
    print(
        "Real parameters allocated:              NO"
    )

    print(
        "Initialization applied:                 NO"
    )

    print(
        "Forward executed:                       NO"
    )

    print(
        "Optimizer created:                      NO"
    )

    print(
        "Training performed:                     NO"
    )

    print(
        "Performance inspected:                  NO"
    )

    print(
        "T60 scored:                             NO"
    )

    print(
        "April analyzed:                         NO"
    )

    print(
        "=" * 120
    )

    print(
        "PHASE 8.2.4 — TRAINABLE PARAMETER NAMESPACE AUDIT V1 READY"
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":

    main()
