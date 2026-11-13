from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.2.0 — IMPLEMENTATION / MATERIALIZATION PREFLIGHT V1
# =============================================================================
#
# Scientific role:
#
#   FRAMEWORK-DEVELOPMENT ENTRY GATE
#
# PURPOSE
# -------
#
# Establish the implementation dependency graph AFTER Phase 8.1 method design
# has been frozen and BEFORE any Phase-8 feature tensor, graph tensor, model,
# optimizer, or training run is created.
#
# This phase:
#
#   - verifies authoritative frozen contracts;
#   - verifies reusable Phase-4/5 source modules exist;
#   - freezes the implementation work-package order;
#   - freezes artifact ownership for each work package;
#   - prevents training before implementation audits pass.
#
# This phase DOES NOT:
#
#   - materialize features;
#   - materialize graphs;
#   - instantiate a model;
#   - instantiate an optimizer;
#   - generate training negatives;
#   - execute inference;
#   - train;
#   - inspect T60 performance;
#   - inspect April.
#
# =============================================================================


PHASE8_BASE = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation"
)

EXECUTION_CONTRACT = (
    PHASE8_BASE
    / "phase_8_1_12_execution_contract"
    / "phase_8_1_12_execution_contract_V3.json"
)

ARCHITECTURE_CONTRACT = (
    PHASE8_BASE
    / "phase_8_1_11_exact_neural_architecture"
    / "phase_8_1_11_architecture_contract_V1.json"
)

BRANCH_CONTRACT = (
    PHASE8_BASE
    / "phase_8_1_11a_model_family_branch_semantics"
    / "phase_8_1_11a_branch_semantics_amendment_V1.json"
)

INDUCTIVE_CONSISTENCY = (
    PHASE8_BASE
    / "phase_8_1_10a_inductive_consistency_closure"
    / "phase_8_1_10a_consistency_amendment_V1.json"
)

GRAPH_SCHEMA = (
    PHASE8_BASE
    / "phase_8_1_9_heterogeneous_graph_schema"
    / "phase_8_1_9_graph_schema_V1.json"
)

INPUT_CONTRACT = (
    PHASE8_BASE
    / "phase_8_1_7_input_contract_amendment"
    / "phase_8_1_7_contract_amendment_V2.json"
)


# =============================================================================
# Existing reusable implementation
# =============================================================================

SOURCE_FILES = [
    Path("src/models/itrs.py"),
    Path("src/models/rgcn.py"),
    Path("src/models/ncf.py"),
    Path("src/features/description.py"),
    Path("src/features/trend.py"),
    Path("src/evaluation/ranking.py"),
    Path("src/evaluation/negatives.py"),
    Path("scripts/train_itrs.py"),
]


# =============================================================================
# Existing baseline contracts
# =============================================================================

BASELINE_CONTRACTS = [
    Path(
        "data/experimental/phase_4/"
        "initialization_contract/"
        "phase_4_7_1b_neural_initialization_contract.json"
    ),
    Path(
        "data/experimental/phase_5/contracts/"
        "phase_5_1_1d_training_negative_sampling_runtime_contract.json"
    ),
    Path(
        "data/experimental/phase_5/contracts/"
        "phase_5_2_2_training_control_optimizer_runtime_contract.json"
    ),
]


OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_0_preflight"
)

WORK_PACKAGE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_0_work_package_plan_V1.csv"
)

SOURCE_INVENTORY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_0_reusable_source_inventory_V1.csv"
)

CONTRACT_INVENTORY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_0_contract_inventory_V1.csv"
)

PREFLIGHT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_0_preflight_contract_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_0_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_2_0_artifact_sha256_V1.json"
)


def separator(char="=", width=118):
    print(char * width)


def require(condition, message):

    if not condition:
        raise ValueError(message)


def read_json(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def write_json(path, obj):

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            obj,
            f,
            indent=2,
            sort_keys=True,
            default=str,
        )


def sha256_file(path, chunk_size=1024 * 1024):

    digest = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def main():

    separator()

    print(
        "PHASE 8.2.0 — "
        "IMPLEMENTATION / MATERIALIZATION PREFLIGHT V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "FRAMEWORK-DEVELOPMENT ENTRY GATE"
    )

    print(
        "Method design reopened:      NO"
    )

    print(
        "Features materialized:       NO"
    )

    print(
        "Graph materialized:          NO"
    )

    print(
        "Model instantiated:          NO"
    )

    print(
        "Training performed:          NO"
    )

    print(
        "T60 scored:                  NO"
    )

    print(
        "April analyzed:              NO"
    )

    # =========================================================================
    # 1. Phase-8 scientific contract gate
    # =========================================================================

    separator("-")

    print(
        "8.2.0.1 — AUTHORITATIVE METHOD-DESIGN CONTRACTS"
    )

    separator("-")

    phase8_contracts = [
        ("8.1.7 V2", INPUT_CONTRACT),
        ("8.1.9 V1", GRAPH_SCHEMA),
        ("8.1.10A V1", INDUCTIVE_CONSISTENCY),
        ("8.1.11 V1", ARCHITECTURE_CONTRACT),
        ("8.1.11A V1", BRANCH_CONTRACT),
        ("8.1.12 V3", EXECUTION_CONTRACT),
    ]

    contract_rows = []

    for phase, path in phase8_contracts:

        require(
            path.exists(),
            f"Missing authoritative contract: {path}",
        )

        payload = read_json(
            path
        )

        require(
            payload.get("status")
            ==
            "FROZEN",
            f"{phase} is not FROZEN.",
        )

        contract_rows.append(
            {
                "phase":
                    phase,

                "path":
                    str(path),

                "sha256":
                    sha256_file(path),

                "status":
                    payload.get("status"),
            }
        )

        print(
            f"{phase:<15} FROZEN  {path}"
        )

    execution = read_json(
        EXECUTION_CONTRACT
    )

    require(
        execution.get(
            "method_design_closed"
        )
        is True,
        "Method design is not closed.",
    )

    require(
        execution.get(
            "performance_observed_before_method_freeze"
        )
        is False,
        (
            "Performance was unexpectedly observed "
            "before method freeze."
        ),
    )

    require(
        execution.get(
            "training_performed"
        )
        is False,
        "Training unexpectedly occurred.",
    )

    require(
        execution.get(
            "T60_scored"
        )
        is False,
        "T60 unexpectedly scored.",
    )

    require(
        execution.get(
            "April_analyzed"
        )
        is False,
        "April unexpectedly analyzed.",
    )

    print()

    print(
        "METHOD-DESIGN FREEZE: PASS"
    )

    print(
        "T60 UNTOUCHED:        PASS"
    )

    print(
        "APRIL UNTOUCHED:      PASS"
    )

    # =========================================================================
    # 2. Reusable implementation inventory
    # =========================================================================

    separator("-")

    print(
        "8.2.0.2 — REUSABLE BASELINE SOURCE INVENTORY"
    )

    separator("-")

    source_rows = []

    for path in SOURCE_FILES:

        exists = path.exists()

        source_rows.append(
            {
                "path":
                    str(path),

                "exists":
                    exists,

                "sha256":
                    (
                        sha256_file(path)
                        if exists
                        else None
                    ),

                "action":
                    (
                        "REUSE_OR_ADAPT"
                        if exists
                        else
                        "MISSING_BLOCKER"
                    ),
            }
        )

        print(
            f"{'FOUND' if exists else 'MISSING':<8} "
            f"{path}"
        )

    source_df = pd.DataFrame(
        source_rows
    )

    require(
        source_df[
            "exists"
        ].all(),
        (
            "At least one reusable baseline source file "
            "is missing."
        ),
    )

    # =========================================================================
    # 3. Baseline execution contracts
    # =========================================================================

    separator("-")

    print(
        "8.2.0.3 — REUSABLE BASELINE CONTRACTS"
    )

    separator("-")

    for path in BASELINE_CONTRACTS:

        require(
            path.exists(),
            f"Missing baseline contract: {path}",
        )

        print(
            f"FOUND {path}"
        )

    # =========================================================================
    # 4. Freeze implementation work packages
    # =========================================================================

    separator("-")

    print(
        "8.2.0.4 — IMPLEMENTATION WORK-PACKAGE PLAN"
    )

    separator("-")

    work_rows = [
        {
            "order":
                1,

            "phase":
                "8.2.1",

            "name":
                "Origin-specific inductive feature materialization",

            "produces":
                (
                    "training-only vocabularies/scalers; "
                    "Startup content inputs; structured Startup "
                    "inputs; founder-history inputs"
                ),

            "training_allowed_after":
                False,

            "performance_allowed":
                False,

            "scientific_gate":
                (
                    "Verify temporal rules, training-only fitting, "
                    "identity freedom and feature coverage."
                ),
        },

        {
            "order":
                2,

            "phase":
                "8.2.2",

            "name":
                "Strict inductive heterogeneous graph materialization",

            "produces":
                (
                    "origin-specific node index; typed edge index; "
                    "fit-time graph mask; inference-time sink graph"
                ),

            "training_allowed_after":
                False,

            "performance_allowed":
                False,

            "scientific_gate":
                (
                    "Verify no held-out primary Startup information "
                    "enters parameter-fit message passing."
                ),
        },

        {
            "order":
                3,

            "phase":
                "8.2.3",

            "name":
                "Phase-8 model module implementation",

            "produces":
                (
                    "B1a/B1b/M1/M2 modules implementing frozen "
                    "40/40/40 Startup representation and "
                    "280-D scorer"
                ),

            "training_allowed_after":
                False,

            "performance_allowed":
                False,

            "scientific_gate":
                (
                    "Parameter namespace and exact architecture "
                    "must match 8.1.11/8.1.11A."
                ),
        },

        {
            "order":
                4,

            "phase":
                "8.2.4",

            "name":
                "Initialization namespace and reproducibility audit",

            "produces":
                (
                    "parameter registry; exact initialization "
                    "coverage; same-seed reproducibility proof"
                ),

            "training_allowed_after":
                False,

            "performance_allowed":
                False,

            "scientific_gate":
                (
                    "Every trainable tensor initialized exactly once; "
                    "no Startup/Person identity parameter."
                ),
        },

        {
            "order":
                5,

            "phase":
                "8.2.5",

            "name":
                "Forward / inductive insertion audit",

            "produces":
                (
                    "shape audit; finite outputs; held-out-node "
                    "insertion proof; sink-only message proof"
                ),

            "training_allowed_after":
                False,

            "performance_allowed":
                False,

            "scientific_gate":
                (
                    "Demonstrate unseen Startup can be encoded after "
                    "fit without changing model parameters."
                ),
        },

        {
            "order":
                6,

            "phase":
                "8.2.6",

            "name":
                "Training-stream integration dry run",

            "produces":
                (
                    "deterministic fold/epoch stream fingerprints; "
                    "shared-stream proof across B1a/B1b/M1/M2"
                ),

            "training_allowed_after":
                False,

            "performance_allowed":
                False,

            "scientific_gate":
                (
                    "No real optimization; prove negative/order RNG "
                    "and candidate alignment."
                ),
        },

        {
            "order":
                7,

            "phase":
                "8.2.7",

            "name":
                "One-step autograd / optimizer preflight",

            "produces":
                (
                    "temporary one-batch forward/backward/update "
                    "proof with no retained trained checkpoint"
                ),

            "training_allowed_after":
                False,

            "performance_allowed":
                False,

            "scientific_gate":
                (
                    "Verify gradients reach every intended trainable "
                    "branch and forbidden parameters do not exist."
                ),
        },

        {
            "order":
                8,

            "phase":
                "8.2.8",

            "name":
                "Framework integration closure",

            "produces":
                (
                    "final implementation hashes; runtime contract; "
                    "training-launch authorization"
                ),

            "training_allowed_after":
                True,

            "performance_allowed":
                False,

            "scientific_gate":
                (
                    "Only after PASS may comparative training begin."
                ),
        },
    ]

    work_df = pd.DataFrame(
        work_rows
    )

    print(
        work_df[
            [
                "order",
                "phase",
                "name",
                "training_allowed_after",
            ]
        ].to_string(
            index=False
        )
    )

    require(
        work_df[
            "order"
        ].tolist()
        ==
        list(
            range(
                1,
                len(work_df) + 1,
            )
        ),
        "Work-package order is not contiguous.",
    )

    require(
        work_df.iloc[:-1][
            "training_allowed_after"
        ].eq(
            False
        ).all(),
        (
            "Training became allowed before framework "
            "integration closure."
        ),
    )

    require(
        bool(
            work_df.iloc[-1][
                "training_allowed_after"
            ]
        )
        is True,
        (
            "Final framework closure does not authorize "
            "training."
        ),
    )

    require(
        work_df[
            "performance_allowed"
        ].eq(
            False
        ).all(),
        (
            "Performance inspection must remain forbidden "
            "through framework development."
        ),
    )

    # =========================================================================
    # 5. Artifact ownership rules
    # =========================================================================

    separator("-")

    print(
        "8.2.0.5 — ARTIFACT OWNERSHIP RULES"
    )

    separator("-")

    ownership = {
        "raw_data_mutation":
            False,

        "phase3_baseline_graph_mutation":
            False,

        "phase4_baseline_model_mutation":
            False,

        "phase6_baseline_result_mutation":
            False,

        "phase8_1_contract_mutation":
            False,

        "new_phase8_artifacts_root":
            (
                "data/experimental/phase_8/"
                "phase_8_2_framework_development"
            ),

        "new_phase8_source_namespace":
            "src/phase8",

        "existing_src_modules":
            (
                "reference/reuse where scientifically identical; "
                "do not silently modify baseline semantics"
            ),

        "failed_script_policy":
            (
                "preserve failed version; create incremented version"
            ),

        "performance_during_framework_development":
            "FORBIDDEN",
    }

    for key, value in ownership.items():

        print(
            f"{key:<42} {value}"
        )

    # =========================================================================
    # 6. Implementation dependency DAG
    # =========================================================================

    separator("-")

    print(
        "8.2.0.6 — IMPLEMENTATION DEPENDENCY DAG"
    )

    separator("-")

    dag = [
        "8.1 frozen contracts",
        "        |",
        "        v",
        "8.2.1 feature materialization",
        "        |",
        "        v",
        "8.2.2 graph materialization",
        "        |",
        "        v",
        "8.2.3 model modules",
        "        |",
        "        v",
        "8.2.4 initialization audit",
        "        |",
        "        v",
        "8.2.5 forward / inductive insertion audit",
        "        |",
        "        v",
        "8.2.6 training-stream integration",
        "        |",
        "        v",
        "8.2.7 one-step autograd preflight",
        "        |",
        "        v",
        "8.2.8 integration closure",
        "        |",
        "        v",
        "COMPARATIVE EXPERIMENTS",
    ]

    for line in dag:
        print(line)

    # =========================================================================
    # 7. Persist preflight
    # =========================================================================

    separator("-")

    print(
        "8.2.0.7 — PERSIST FRAMEWORK-DEVELOPMENT PREFLIGHT"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    contract_df = pd.DataFrame(
        contract_rows
    )

    contract_df.to_csv(
        CONTRACT_INVENTORY_OUTPUT,
        index=False,
    )

    source_df.to_csv(
        SOURCE_INVENTORY_OUTPUT,
        index=False,
    )

    work_df.to_csv(
        WORK_PACKAGE_OUTPUT,
        index=False,
    )

    preflight = {
        "schema_version":
            "PHASE_8_2_0_IMPLEMENTATION_PREFLIGHT_V1",

        "status":
            "FROZEN",

        "scientific_role":
            "FRAMEWORK_DEVELOPMENT_ENTRY_GATE",

        "authoritative_phase8_execution_contract":
            str(EXECUTION_CONTRACT),

        "authoritative_phase8_execution_contract_sha256":
            sha256_file(
                EXECUTION_CONTRACT
            ),

        "method_design_closed":
            True,

        "method_design_reopened":
            False,

        "work_packages":
            work_df.to_dict(
                orient="records"
            ),

        "artifact_ownership":
            ownership,

        "feature_materialization_started":
            False,

        "graph_materialization_started":
            False,

        "model_instantiated":
            False,

        "optimizer_instantiated":
            False,

        "training_performed":
            False,

        "performance_inspected":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,

        "training_allowed_now":
            False,

        "next_phase":
            (
                "Phase 8.2.1 — origin-specific inductive "
                "feature materialization."
            ),
    }

    write_json(
        PREFLIGHT_OUTPUT,
        preflight,
    )

    manifest = {
        "phase":
            "8.2.0",

        "status":
            "FROZEN",

        "method_design_closed":
            True,

        "framework_development_started":
            True,

        "outputs": [
            str(CONTRACT_INVENTORY_OUTPUT),
            str(SOURCE_INVENTORY_OUTPUT),
            str(WORK_PACKAGE_OUTPUT),
            str(PREFLIGHT_OUTPUT),
        ],

        "training_allowed_now":
            False,

        "training_performed":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    outputs = [
        CONTRACT_INVENTORY_OUTPUT,
        SOURCE_INVENTORY_OUTPUT,
        WORK_PACKAGE_OUTPUT,
        PREFLIGHT_OUTPUT,
        MANIFEST_OUTPUT,
    ]

    write_json(
        HASH_OUTPUT,
        {
            str(path):
                sha256_file(path)
            for path in outputs
        },
    )

    separator()

    print(
        "PHASE 8.2.0 FRAMEWORK-DEVELOPMENT PREFLIGHT FROZEN"
    )

    separator()

    print()

    print(
        "METHOD DESIGN:                     CLOSED"
    )

    print(
        "FRAMEWORK DEVELOPMENT:             STARTED"
    )

    print(
        "AUTHORITATIVE EXECUTION CONTRACT:  8.1.12 V3"
    )

    print(
        "IMPLEMENTATION WORK PACKAGES:      "
        f"{len(work_df)}"
    )

    print(
        "FEATURES MATERIALIZED:             NO"
    )

    print(
        "GRAPH MATERIALIZED:                NO"
    )

    print(
        "MODEL INSTANTIATED:                NO"
    )

    print(
        "TRAINING ALLOWED NOW:              NO"
    )

    print(
        "T60 SCORED:                        NO"
    )

    print(
        "APRIL ANALYZED:                    NO"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 8.2.1 — origin-specific inductive "
        "feature materialization."
    )


if __name__ == "__main__":
    main()
