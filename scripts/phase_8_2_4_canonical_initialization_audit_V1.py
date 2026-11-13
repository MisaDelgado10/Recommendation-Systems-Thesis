from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import sys
import gc

import torch
import torch.nn as nn


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

NAMESPACE_CSV = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_4_deterministic_initialization/"
    "phase_8_2_4_trainable_parameter_namespace_V1.csv"
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

HASH_TABLE_PATH = (
    OUT_DIR
    / "phase_8_2_4_initialized_parameter_hashes_V1.csv"
)

SUMMARY_PATH = (
    OUT_DIR
    / "phase_8_2_4_canonical_initialization_summary_V1.json"
)

ARTIFACT_HASH_PATH = (
    OUT_DIR
    / "phase_8_2_4_canonical_initialization_artifact_sha256_V1.json"
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


def tensor_sha256(
    tensor,
):

    require(
        tensor.device.type
        == "cpu",
        "Tensor hashing requires CPU tensor.",
    )

    data = (
        tensor
        .detach()
        .contiguous()
        .numpy()
        .tobytes()
    )

    return hashlib.sha256(
        data
    ).hexdigest()


def shape_string(
    tensor,
):

    return "x".join(
        str(
            int(dim)
        )
        for dim in tensor.shape
    )


def dtype_string(
    tensor,
):

    value = str(
        tensor.dtype
    )

    require(
        value.startswith(
            "torch."
        ),
        (
            "Unexpected dtype representation: "
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

    module_parts = parts[
        :-1
    ]

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
                    "Invalid semantic component for "
                    f"{parameter_name}"
                ),
            )

            return component

    raise RuntimeError(
        "No semantic component resolved for "
        f"{parameter_name}"
    )


def load_model_module():

    module_name = (
        "phase8_candidate_modules_for_init_audit"
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
        "Could not create model import spec.",
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
            "Missing vocabulary artifact: "
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

    base_dim = (
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

    extended_dim = (
        base_dim
        + 1
        + len(
            vocab[
                "prior_sector"
            ]
        )
    )

    return {
        "category_dim":
            category_dim,

        "base_dim":
            base_dim,

        "extended_dim":
            extended_dim,
    }


def instantiate_model(
    module,
    family,
    dimensions,
):

    if family == "B1a":

        structured_dim = None

    elif family == "B1b":

        structured_dim = (
            dimensions[
                "base_dim"
            ]
        )

    elif family in (
        "M1",
        "M2",
    ):

        structured_dim = (
            dimensions[
                "extended_dim"
            ]
        )

    else:

        raise RuntimeError(
            f"Unknown family: {family}"
        )

    return module.Phase8CandidateModel(
        model_family=family,
        investor_label_input_dim=802,
        startup_category_input_dim=(
            dimensions[
                "category_dim"
            ]
        ),
        startup_structured_input_dim=(
            structured_dim
        ),
    ).cpu()


def load_namespace_table():

    require(
        NAMESPACE_CSV.exists(),
        (
            "Missing namespace table: "
            f"{NAMESPACE_CSV}"
        ),
    )

    rows = []

    with NAMESPACE_CSV.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:
            rows.append(
                row
            )

    require(
        len(rows)
        == 577,
        (
            "Frozen namespace row count changed: "
            f"{len(rows)}"
        ),
    )

    return rows


def build_namespace_lookup(
    rows,
):

    lookup = {}

    for row in rows:

        key = (
            row[
                "scope"
            ],
            row[
                "model"
            ],
            row[
                "parameter_name"
            ],
        )

        require(
            key not in lookup,
            (
                "Duplicate namespace row: "
                f"{key}"
            ),
        )

        lookup[
            key
        ] = row

    return lookup


# =============================================================================
# CANONICAL INITIALIZATION
# =============================================================================

def initialize_model(
    *,
    model,
    family,
    scope,
    namespace_lookup,
    reverse_order,
):

    named_parameters = list(
        model.named_parameters()
    )

    if reverse_order:

        named_parameters = list(
            reversed(
                named_parameters
            )
        )

    seen = set()

    with torch.no_grad():

        for (
            parameter_name,
            parameter,
        ) in named_parameters:

            key = (
                scope,
                family,
                parameter_name,
            )

            require(
                key in namespace_lookup,
                (
                    "Parameter missing from frozen namespace: "
                    f"{key}"
                ),
            )

            row = namespace_lookup[
                key
            ]

            require(
                parameter.device.type
                == "cpu",
                (
                    "Canonical initialization must occur on CPU: "
                    f"{key}"
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

            require(
                component
                == row[
                    "semantic_component_id"
                ],
                (
                    "Semantic component drift: "
                    f"{key}"
                ),
            )

            require(
                shape
                == row[
                    "shape"
                ],
                (
                    "Shape drift: "
                    f"{key}: "
                    f"{shape} != {row['shape']}"
                ),
            )

            require(
                dtype
                == row[
                    "dtype"
                ],
                (
                    "dtype drift: "
                    f"{key}: "
                    f"{dtype} != {row['dtype']}"
                ),
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

            require(
                payload
                == row[
                    "payload"
                ],
                (
                    "Seed payload drift: "
                    f"{key}"
                ),
            )

            require(
                seed_u64
                == int(
                    row[
                        "derived_seed_u64"
                    ]
                ),
                (
                    "Derived seed drift: "
                    f"{key}"
                ),
            )

            generator = torch.Generator(
                device="cpu"
            )

            generator.manual_seed(
                seed_u64
            )

            rule = row[
                "initialization_rule"
            ]

            if rule == "ZEROS":

                nn.init.zeros_(
                    parameter
                )

            elif (
                rule
                == "KAIMING_NORMAL_A0_FAN_IN_RELU"
            ):

                require(
                    parameter.ndim >= 2,
                    (
                        "Kaiming target is not matrix-like: "
                        f"{key}"
                    ),
                )

                nn.init.kaiming_normal_(
                    parameter,
                    a=0,
                    mode="fan_in",
                    nonlinearity="relu",
                    generator=generator,
                )

            else:

                raise RuntimeError(
                    "Unknown initialization rule: "
                    f"{rule}"
                )

            require(
                bool(
                    torch.isfinite(
                        parameter
                    ).all()
                ),
                (
                    "Non-finite initialized parameter: "
                    f"{key}"
                ),
            )

            if rule == "ZEROS":

                require(
                    bool(
                        torch.count_nonzero(
                            parameter
                        )
                        == 0
                    ),
                    (
                        "Bias failed ZERO initialization: "
                        f"{key}"
                    ),
                )

            else:

                require(
                    bool(
                        torch.count_nonzero(
                            parameter
                        )
                        > 0
                    ),
                    (
                        "Matrix unexpectedly all-zero: "
                        f"{key}"
                    ),
                )

            seen.add(
                parameter_name
            )

    expected_names = {
        name
        for name, _
        in model.named_parameters()
    }

    require(
        seen
        == expected_names,
        (
            "Canonical initializer did not cover "
            "every trainable tensor exactly once."
        ),
    )


def parameter_hashes(
    model,
):

    return {
        name:
            tensor_sha256(
                parameter
            )

        for name, parameter
        in model.named_parameters()
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(
        "=" * 120
    )

    print(
        "PHASE 8.2.4 — CANONICAL INITIALIZATION AUDIT V1"
    )

    print(
        "=" * 120
    )

    require(
        MODEL_MODULE_PATH.exists(),
        "Candidate model module missing.",
    )

    namespace_rows = (
        load_namespace_table()
    )

    namespace_lookup = (
        build_namespace_lookup(
            namespace_rows
        )
    )

    module = load_model_module()

    output_rows = []

    shared_hash_registry = {}

    tested_models = 0

    tested_parameter_instances = 0

    # =========================================================================
    # ALL MODEL/SCOPE COMBINATIONS
    # =========================================================================

    for scope, config in SCOPE_CONFIG.items():

        origin = config[
            "origin"
        ]

        dimensions = (
            load_origin_dimensions(
                origin
            )
        )

        for family in config[
            "families"
        ]:

            # -----------------------------------------------------------------
            # Instance A:
            # arbitrary global RNG state + normal traversal.
            # -----------------------------------------------------------------

            torch.manual_seed(
                123456789
            )

            model_a = instantiate_model(
                module,
                family,
                dimensions,
            )

            initialize_model(
                model=model_a,
                family=family,
                scope=scope,
                namespace_lookup=namespace_lookup,
                reverse_order=False,
            )

            hashes_a = parameter_hashes(
                model_a
            )

            # -----------------------------------------------------------------
            # Instance B:
            # different global RNG state + reverse traversal.
            # -----------------------------------------------------------------

            torch.manual_seed(
                987654321
            )

            model_b = instantiate_model(
                module,
                family,
                dimensions,
            )

            initialize_model(
                model=model_b,
                family=family,
                scope=scope,
                namespace_lookup=namespace_lookup,
                reverse_order=True,
            )

            hashes_b = parameter_hashes(
                model_b
            )

            require(
                hashes_a.keys()
                == hashes_b.keys(),
                (
                    "Repeated model parameter namespace changed: "
                    f"{scope}/{family}"
                ),
            )

            # -----------------------------------------------------------------
            # Bitwise reproducibility and order independence.
            # -----------------------------------------------------------------

            for parameter_name in hashes_a:

                require(
                    hashes_a[
                        parameter_name
                    ]
                    == hashes_b[
                        parameter_name
                    ],
                    (
                        "Initialization depends on global RNG "
                        "or traversal order: "
                        f"{scope}/{family}/{parameter_name}"
                    ),
                )

                row = namespace_lookup[
                    (
                        scope,
                        family,
                        parameter_name,
                    )
                ]

                actual_hash = hashes_a[
                    parameter_name
                ]

                signature = (
                    scope,
                    row[
                        "semantic_component_id"
                    ],
                    parameter_name,
                    row[
                        "shape"
                    ],
                    row[
                        "dtype"
                    ],
                )

                if signature in shared_hash_registry:

                    require(
                        shared_hash_registry[
                            signature
                        ]
                        == actual_hash,
                        (
                            "Shared tensor rule violated: "
                            f"{signature}"
                        ),
                    )

                else:

                    shared_hash_registry[
                        signature
                    ] = actual_hash

                output_rows.append(
                    {
                        "model":
                            family,

                        "scope":
                            scope,

                        "origin":
                            origin,

                        "semantic_component_id":
                            row[
                                "semantic_component_id"
                            ],

                        "parameter_name":
                            parameter_name,

                        "shape":
                            row[
                                "shape"
                            ],

                        "dtype":
                            row[
                                "dtype"
                            ],

                        "derived_seed_u64":
                            row[
                                "derived_seed_u64"
                            ],

                        "initialization_rule":
                            row[
                                "initialization_rule"
                            ],

                        "initialized_tensor_sha256":
                            actual_hash,

                        "normal_vs_reverse_order_match":
                            True,

                        "different_global_rng_state_match":
                            True,
                    }
                )

                tested_parameter_instances += 1

            tested_models += 1

            del model_a
            del model_b

            gc.collect()

    # =========================================================================
    # GLOBAL AUDITS
    # =========================================================================

    require(
        tested_models
        == 17,
        (
            "Unexpected model/scope combinations: "
            f"{tested_models}"
        ),
    )

    require(
        tested_parameter_instances
        == 577,
        (
            "Unexpected initialized parameter instances: "
            f"{tested_parameter_instances}"
        ),
    )

    require(
        len(
            output_rows
        )
        == 577,
        (
            "Initialization hash table row count mismatch."
        ),
    )

    # -------------------------------------------------------------------------
    # Same complete semantic signature within scope => same tensor hash.
    # -------------------------------------------------------------------------

    signature_hashes = {}

    for row in output_rows:

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

        signature_hashes.setdefault(
            signature,
            set(),
        ).add(
            row[
                "initialized_tensor_sha256"
            ]
        )

    shared_signature_groups = 0

    for signature, hashes in signature_hashes.items():

        require(
            len(
                hashes
            )
            == 1,
            (
                "Identical semantic signature produced "
                "different initialized values: "
                f"{signature}"
            ),
        )

        occurrences = sum(
            1
            for row in output_rows
            if (
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
            == signature
        )

        if occurrences > 1:
            shared_signature_groups += 1

    # =========================================================================
    # WRITE HASH TABLE
    # =========================================================================

    fieldnames = [
        "model",
        "scope",
        "origin",
        "semantic_component_id",
        "parameter_name",
        "shape",
        "dtype",
        "derived_seed_u64",
        "initialization_rule",
        "initialized_tensor_sha256",
        "normal_vs_reverse_order_match",
        "different_global_rng_state_match",
    ]

    with HASH_TABLE_PATH.open(
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
            output_rows
        )

    # =========================================================================
    # SUMMARY
    # =========================================================================

    summary = {
        "schema_version":
            "PHASE_8_2_4_CANONICAL_INITIALIZATION_AUDIT_V1",

        "status":
            "INITIALIZED_AND_REPRODUCIBILITY_AUDITED",

        "canonical_device":
            "CPU",

        "namespace":
            NAMESPACE,

        "base_seed":
            BASE_SEED,

        "model_scope_combinations_tested":
            tested_models,

        "parameter_instances_tested":
            tested_parameter_instances,

        "expected_parameter_instances":
            577,

        "global_rng_states_tested": [
            123456789,
            987654321,
        ],

        "parameter_orders_tested": [
            "NORMAL_NAMED_PARAMETER_ORDER",
            "REVERSED_NAMED_PARAMETER_ORDER",
        ],

        "all_normal_reverse_hashes_match":
            True,

        "all_different_global_rng_hashes_match":
            True,

        "shared_semantic_signature_groups_audited":
            shared_signature_groups,

        "shared_tensor_rule":
            "PASS",

        "bias_rule":
            "ZEROS",

        "matrix_rule":
            (
                "torch.nn.init.kaiming_normal_"
                "(a=0, mode='fan_in', nonlinearity='relu')"
            ),

        "dedicated_CPU_generator_per_matrix":
            True,

        "seed_rederived_from_payload":
            True,

        "namespace_table_revalidated":
            True,

        "execution_state": {
            "real_parameter_values_allocated":
                True,

            "canonical_phase8_initialization_applied":
                True,

            "models_retained_after_audit":
                False,

            "checkpoint_written":
                False,

            "forward_executed":
                False,

            "graph_inference_executed":
                False,

            "optimizer_created":
                False,

            "backward_executed":
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
                "Phase 8.2.5 — forward/insertion audit "
                "using canonical initialization."
            ),

        "training_allowed_now":
            False,
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

    artifacts = {
        str(
            MODEL_MODULE_PATH
        ):
            sha256_file(
                MODEL_MODULE_PATH
            ),

        str(
            NAMESPACE_CSV
        ):
            sha256_file(
                NAMESPACE_CSV
            ),

        str(
            HASH_TABLE_PATH
        ):
            sha256_file(
                HASH_TABLE_PATH
            ),

        str(
            SUMMARY_PATH
        ):
            sha256_file(
                SUMMARY_PATH
            ),
    }

    ARTIFACT_HASH_PATH.write_text(
        json.dumps(
            artifacts,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    # =========================================================================
    # COMPACT REPORT
    # =========================================================================

    print(
        f"Model/scope combinations:               {tested_models}"
    )

    print(
        f"Parameter instances:                    {tested_parameter_instances}"
    )

    print(
        "Canonical device:                       CPU"
    )

    print(
        "Seed payload rederivation:              PASS"
    )

    print(
        "Namespace table revalidation:           PASS"
    )

    print(
        "Bias initialization:                    ZERO"
    )

    print(
        "Matrix initialization:                  KAIMING_NORMAL"
    )

    print(
        "Different global RNG states:             MATCH"
    )

    print(
        "Normal vs reversed parameter traversal:  MATCH"
    )

    print(
        "Shared semantic tensor rule:             PASS"
    )

    print(
        f"Shared signature groups audited:         {shared_signature_groups}"
    )

    print()
    print(
        f"Hashes:  {HASH_TABLE_PATH}"
    )

    print(
        f"Summary: {SUMMARY_PATH}"
    )

    print(
        f"Registry:{ARTIFACT_HASH_PATH}"
    )

    print()
    print(
        "Real parameters allocated:               YES — audit only"
    )

    print(
        "Canonical initialization applied:        YES"
    )

    print(
        "Checkpoint written:                      NO"
    )

    print(
        "Forward executed:                        NO"
    )

    print(
        "Optimizer created:                       NO"
    )

    print(
        "Backward executed:                       NO"
    )

    print(
        "Training performed:                      NO"
    )

    print(
        "Performance inspected:                   NO"
    )

    print(
        "T60 scored:                              NO"
    )

    print(
        "April analyzed:                          NO"
    )

    print(
        "Training allowed now:                    NO"
    )

    print(
        "=" * 120
    )

    print(
        "PHASE 8.2.4 — CANONICAL INITIALIZATION AUDIT V1 PASSED"
    )

    print(
        "=" * 120
    )


if __name__ == "__main__":

    main()
