from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.2.0 — IMPLEMENTATION / MATERIALIZATION PREFLIGHT V2
# =============================================================================
#
# Scientific role:
#
#   CORRECTED_FRAMEWORK_DEVELOPMENT_REENTRY_GATE
#
# V1 was entered prematurely against Phase 8.1.12 V3 and before the
# semantic repairs were complete.
#
# V2 does NOT erase that history. It:
#
#   - preserves V1 as an invalid/superseded entry gate;
#   - adopts Phase 8.1.12 V5 as the authoritative execution contract;
#   - adopts the repaired graph / consistency / branch contracts;
#   - records already-materialized 8.2.1A/B/C artifacts;
#   - explicitly keeps training blocked;
#   - sets Phase 8.2.1D as the next executable implementation step.
#
# No preprocessing is fit here.
# No graph is materialized here.
# No model is instantiated.
# No optimization occurs.
# =============================================================================


ROOT = Path(
    "data/experimental/phase_8"
)

METHOD = (
    ROOT
    / "phase_8_1_inductive_startup_representation"
)

FRAMEWORK = (
    ROOT
    / "phase_8_2_framework_development"
)

OUT_DIR = (
    FRAMEWORK
    / "phase_8_2_0_preflight"
)

FEATURE_DIR = (
    FRAMEWORK
    / "phase_8_2_1_feature_materialization"
)


# -----------------------------------------------------------------------------
# Historical V1 entry gate
# -----------------------------------------------------------------------------

V1_CONTRACT = (
    OUT_DIR
    / "phase_8_2_0_preflight_contract_V1.json"
)

V1_MANIFEST = (
    OUT_DIR
    / "phase_8_2_0_manifest_V1.json"
)


# -----------------------------------------------------------------------------
# Authoritative corrected method contracts
# -----------------------------------------------------------------------------

EXEC_V5 = (
    METHOD
    / "phase_8_1_12_execution_contract"
    / "phase_8_1_12_execution_contract_V5.json"
)

GRAPH_V2 = (
    METHOD
    / "phase_8_1_9_heterogeneous_graph_schema"
    / "phase_8_1_9_graph_schema_V2.json"
)

CONSISTENCY_V2 = (
    METHOD
    / "phase_8_1_10a_inductive_consistency_closure"
    / "phase_8_1_10a_consistency_amendment_V2.json"
)

ARCHITECTURE_V1 = (
    METHOD
    / "phase_8_1_11_exact_neural_architecture"
    / "phase_8_1_11_architecture_contract_V1.json"
)

BRANCH_V2 = (
    METHOD
    / "phase_8_1_11a_model_family_branch_semantics"
    / "phase_8_1_11a_branch_semantics_amendment_V2.json"
)

COLD_V1 = (
    METHOD
    / "phase_8_1_12a_outcome_independent_cold_start_masking"
    / "phase_8_1_12a_outcome_independent_masking_contract_V1.json"
)

FOUNDER_V1 = (
    METHOD
    / "phase_8_1_12b_origin_referenced_founder_history"
    / "phase_8_1_12b_origin_referenced_founder_history_contract_V1.json"
)


# -----------------------------------------------------------------------------
# Existing Phase-8.2.1 materialization state
# -----------------------------------------------------------------------------

STARTUP_SOURCE_V1 = (
    FEATURE_DIR
    / "phase_8_2_1a_canonical_startup_source_V1.parquet"
)

FOUNDER_MANIFEST_V2 = (
    FEATURE_DIR
    / "phase_8_2_1b_manifest_V2.json"
)

FOUNDER_ORIGIN_TABLE_V2 = (
    FEATURE_DIR
    / "phase_8_2_1b_origin_table_V2.csv"
)

COLD_MANIFEST_V2 = (
    FEATURE_DIR
    / "phase_8_2_1c_manifest_V2.json"
)

COLD_MASKS_V2 = (
    FEATURE_DIR
    / "phase_8_2_1c_origin_startup_masks_V2.parquet"
)

PREPROCESS_POP_V2 = (
    FEATURE_DIR
    / "phase_8_2_1c_preprocessing_fit_population_V2.parquet"
)

ORIGIN_SUMMARY_V2 = (
    FEATURE_DIR
    / "phase_8_2_1c_origin_population_summary_V2.csv"
)


# -----------------------------------------------------------------------------
# V2 outputs
# -----------------------------------------------------------------------------

CONTRACT_V2 = (
    OUT_DIR
    / "phase_8_2_0_preflight_contract_V2.json"
)

MANIFEST_V2 = (
    OUT_DIR
    / "phase_8_2_0_manifest_V2.json"
)

CONTRACT_INVENTORY_V2 = (
    OUT_DIR
    / "phase_8_2_0_contract_inventory_V2.csv"
)

SOURCE_INVENTORY_V2 = (
    OUT_DIR
    / "phase_8_2_0_reusable_source_inventory_V2.csv"
)

WORK_PLAN_V2 = (
    OUT_DIR
    / "phase_8_2_0_work_package_plan_V2.csv"
)

HASHES_V2 = (
    OUT_DIR
    / "phase_8_2_0_artifact_sha256_V2.json"
)


def sep(char="=", width=120):
    print(char * width)


def require(condition, message):

    if not bool(condition):
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
            ensure_ascii=False,
            allow_nan=False,
        )


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def main():

    sep()

    print(
        "PHASE 8.2.0 — "
        "IMPLEMENTATION / MATERIALIZATION PREFLIGHT V2"
    )

    sep()

    print()

    print(
        "Scientific role:             "
        "CORRECTED_FRAMEWORK_DEVELOPMENT_REENTRY_GATE"
    )

    print("Method design reopened:      NO")
    print("Feature discovery reopened:  NO")
    print("Preprocessing fit:           NO")
    print("Graph materialized:          NO")
    print("Model instantiated:          NO")
    print("Optimizer instantiated:      NO")
    print("Training performed:          NO")
    print("Performance inspected:       NO")
    print("T60 scored:                  NO")
    print("April analyzed:              NO")

    # =========================================================================
    # 1. Prerequisites
    # =========================================================================

    sep("-")

    print(
        "8.2.0-V2.1 — PREREQUISITE GATES"
    )

    sep("-")

    required = [
        V1_CONTRACT,
        V1_MANIFEST,
        EXEC_V5,
        GRAPH_V2,
        CONSISTENCY_V2,
        ARCHITECTURE_V1,
        BRANCH_V2,
        COLD_V1,
        FOUNDER_V1,
        STARTUP_SOURCE_V1,
        FOUNDER_MANIFEST_V2,
        FOUNDER_ORIGIN_TABLE_V2,
        COLD_MANIFEST_V2,
        COLD_MASKS_V2,
        PREPROCESS_POP_V2,
        ORIGIN_SUMMARY_V2,
    ]

    for path in required:

        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

        print(
            f"FOUND {path}"
        )

    v1 = read_json(
        V1_CONTRACT
    )

    v1_manifest = read_json(
        V1_MANIFEST
    )

    execution = read_json(
        EXEC_V5
    )

    graph = read_json(
        GRAPH_V2
    )

    consistency = read_json(
        CONSISTENCY_V2
    )

    architecture = read_json(
        ARCHITECTURE_V1
    )

    branch = read_json(
        BRANCH_V2
    )

    cold = read_json(
        COLD_V1
    )

    founder = read_json(
        FOUNDER_V1
    )

    # =========================================================================
    # 2. Prove V1 is no longer a valid entry gate
    # =========================================================================

    sep("-")

    print(
        "8.2.0-V2.2 — V1 ENTRY-GATE SUPERSESSION AUDIT"
    )

    sep("-")

    require(
        v1.get(
            "authoritative_phase8_execution_contract",
            "",
        ).endswith(
            "phase_8_1_12_execution_contract_V3.json"
        ),
        (
            "Expected V1 dependency on the superseded "
            "8.1.12 V3 contract was not found."
        ),
    )

    v1_blob = json.dumps(
        v1,
        sort_keys=True,
    )

    require(
        "held-out primary Startup"
        in
        v1_blob,
        (
            "Expected stale held-out-primary wording "
            "was not found in V1."
        ),
    )

    require(
        v1.get(
            "feature_materialization_started"
        )
        is False,
        (
            "Expected historical V1 materialization state "
            "was not found."
        ),
    )

    print(
        "V1 points to superseded V3:              YES"
    )

    print(
        "V1 contains stale held-out wording:      YES"
    )

    print(
        "V1 predates actual 8.2.1 materialization:YES"
    )

    print()

    print(
        "V1 status for scientific use: "
        "INVALID_ENTRY_GATE / FROZEN_SUPERSEDED"
    )

    # =========================================================================
    # 3. Verify corrected authoritative method chain
    # =========================================================================

    sep("-")

    print(
        "8.2.0-V2.3 — AUTHORITATIVE METHOD CHAIN"
    )

    sep("-")

    for name, obj in [
        ("8.1.12 V5", execution),
        ("8.1.9 V2", graph),
        ("8.1.10A V2", consistency),
        ("8.1.11 V1", architecture),
        ("8.1.11A V2", branch),
        ("8.1.12A V1", cold),
        ("8.1.12B V1", founder),
    ]:

        require(
            obj.get("status")
            ==
            "FROZEN",
            f"{name} is not FROZEN.",
        )

        print(
            f"{name:<18} FROZEN"
        )

    require(
        execution.get(
            "schema_version"
        )
        ==
        "PHASE_8_1_12_EXECUTION_CONTRACT_V5",
        "V5 schema mismatch.",
    )

    require(
        execution.get(
            "method_design_closed"
        )
        is True,
        "V5 method design is not closed.",
    )

    require(
        execution[
            "implementation_gate"
        ][
            "training_allowed_immediately"
        ]
        is False,
        "V5 unexpectedly authorizes immediate training.",
    )

    execution_blob = json.dumps(
        execution,
        sort_keys=True,
    )

    for marker in [
        "ALL_COLD_t_STARTUPS",
        "phase_8_1_9_graph_schema_V2.json",
        "phase_8_1_10a_consistency_amendment_V2.json",
        "phase_8_1_11a_branch_semantics_amendment_V2.json",
        "ITRS_PHASE8_TRAIN_NEGATIVE",
        "ITRS_PHASE8_TRAIN_ORDER",
        "ITRS_PHASE8_NEURAL_INIT",
        "TWO_WAY_PIGEONHOLE_CLUSTER_BOOTSTRAP",
    ]:

        require(
            marker
            in
            execution_blob,
            (
                "Authoritative V5 marker missing: "
                f"{marker}"
            ),
        )

    print()

    print(
        "V5 semantic-chain audit:                PASS"
    )

    print(
        "V5 method-design closure:               PASS"
    )

    print(
        "Immediate training remains blocked:     PASS"
    )

    # =========================================================================
    # 4. Record current materialization state honestly
    # =========================================================================

    sep("-")

    print(
        "8.2.0-V2.4 — CORRECTED REENTRY MATERIALIZATION STATE"
    )

    sep("-")

    materialization_state = [
        {
            "component":
                "8.2.1A canonical Startup source",

            "artifact":
                str(
                    STARTUP_SOURCE_V1
                ),

            "status":
                "REUSABLE_SUBSTRATE",

            "authoritative_for_next_steps":
                True,

            "learned_parameters":
                False,

            "note":
                (
                    "Prematurely materialized under the old entry gate, "
                    "but mechanically reusable and revalidated by the "
                    "corrected downstream semantic contracts."
                ),
        },
        {
            "component":
                "8.2.1B founder identity + origin history",

            "artifact":
                str(
                    FOUNDER_MANIFEST_V2
                ),

            "status":
                "PASS_CORRECTED_V2",

            "authoritative_for_next_steps":
                True,

            "learned_parameters":
                False,

            "note":
                (
                    "V2 uses prediction-origin H(s,t); "
                    "V1 predictive history is superseded."
                ),
        },
        {
            "component":
                "8.2.1C cold/warm + fit populations",

            "artifact":
                str(
                    COLD_MANIFEST_V2
                ),

            "status":
                "PASS_CORRECTED_V2",

            "authoritative_for_next_steps":
                True,

            "learned_parameters":
                False,

            "note":
                (
                    "Reconstructs frozen origin eligibility and "
                    "outcome-independent COLD_t populations."
                ),
        },
    ]

    for row in materialization_state:

        print(
            f"{row['component']:<46} "
            f"{row['status']}"
        )

    print()

    print(
        "Learned preprocessing already fit:         NO"
    )

    print(
        "Graph tensors already materialized:        NO"
    )

    print(
        "Phase-8 model already instantiated:        NO"
    )

    print(
        "Phase-8 candidate performance inspected:   NO"
    )

    # =========================================================================
    # 5. Corrected work-package plan
    # =========================================================================

    sep("-")

    print(
        "8.2.0-V2.5 — CORRECTED WORK-PACKAGE PLAN"
    )

    sep("-")

    work_packages = [
        {
            "order":
                1,

            "phase":
                "8.2.1D",

            "name":
                (
                    "Origin-specific training-only preprocessing "
                    "fit + feature transformation"
                ),

            "produces":
                (
                    "Per-origin TF-IDF/SVD, categorical vocabularies, "
                    "numeric scalers and transformed Startup feature "
                    "artifacts."
                ),

            "scientific_gate":
                (
                    "Fit exclusively on the authoritative 8.2.1C V2 "
                    "preprocessing-fit population; COLD_t raw values, "
                    "evaluation candidates and future membership may "
                    "not influence fit."
                ),

            "performance_allowed":
                False,

            "training_allowed_after":
                False,
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
                    "Origin-specific typed graph tensors, fit-time "
                    "ALL_COLD_t masks and sink-only inference graphs."
                ),

            "scientific_gate":
                (
                    "During fit, ALL COLD_t Startup features and "
                    "founder/acquisition incident edges are hidden. "
                    "At inference, only incoming origin-eligible "
                    "messages may be restored."
                ),

            "performance_allowed":
                False,

            "training_allowed_after":
                False,
        },
        {
            "order":
                3,

            "phase":
                "8.2.3",

            "name":
                "Phase-8 candidate model implementation",

            "produces":
                (
                    "B1a/B1b/M1/M2 modules implementing the frozen "
                    "280-D pair representation and scorer."
                ),

            "scientific_gate":
                (
                    "Implementation must match 8.1.11 V1 architecture "
                    "plus authoritative 8.1.11A V2 branch semantics."
                ),

            "performance_allowed":
                False,

            "training_allowed_after":
                False,
        },
        {
            "order":
                4,

            "phase":
                "8.2.4",

            "name":
                "Initialization namespace + reproducibility audit",

            "produces":
                (
                    "Complete trainable-parameter registry, per-tensor "
                    "derived seeds and same-semantics initialization proof."
                ),

            "scientific_gate":
                (
                    "Every trainable tensor occurs exactly once under "
                    "the V5 semantic parameter-key initialization rule; "
                    "no Startup/Person identity parameter exists."
                ),

            "performance_allowed":
                False,

            "training_allowed_after":
                False,
        },
        {
            "order":
                5,

            "phase":
                "8.2.5",

            "name":
                "Forward + inductive insertion audit",

            "produces":
                (
                    "Shape/finite-output audit and explicit proof of "
                    "sink-only COLD_t insertion with frozen parameters."
                ),

            "scientific_gate":
                (
                    "Restoring a COLD_t Startup may change that Startup's "
                    "representation but must not alter fitted Investor, "
                    "Person or other Startup representations."
                ),

            "performance_allowed":
                False,

            "training_allowed_after":
                False,
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
                    "Deterministic fold/epoch negative and ordering "
                    "fingerprints shared across B1a/B1b/M1/M2."
                ),

            "scientific_gate":
                (
                    "Use V5 ITRS_PHASE8_TRAIN_NEGATIVE and "
                    "ITRS_PHASE8_TRAIN_ORDER namespaces; no optimization "
                    "or performance inspection."
                ),

            "performance_allowed":
                False,

            "training_allowed_after":
                False,
        },
        {
            "order":
                7,

            "phase":
                "8.2.7",

            "name":
                "One-step autograd / optimizer implementation preflight",

            "produces":
                (
                    "Temporary one-batch forward/backward/update audit "
                    "with model state discarded and no trained checkpoint."
                ),

            "scientific_gate":
                (
                    "Verify intended gradients and optimizer plumbing "
                    "without evaluating ranking performance or retaining "
                    "the updated model state."
                ),

            "performance_allowed":
                False,

            "training_allowed_after":
                False,
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
                    "Final implementation hashes, runtime contract and "
                    "explicit comparative-training launch authorization."
                ),

            "scientific_gate":
                (
                    "Only an audited PASS under 8.1.12 V5 may authorize "
                    "comparative B1a/B1b/M1/M2 development training."
                ),

            "performance_allowed":
                False,

            "training_allowed_after":
                True,
        },
    ]

    for wp in work_packages:

        print(
            f"{wp['order']}. {wp['phase']:<8} "
            f"{wp['name']}"
        )

    # =========================================================================
    # 6. Contract inventory
    # =========================================================================

    contract_rows = []

    contract_sources = [
        (
            "FINAL_EXECUTION_CONTRACT",
            EXEC_V5,
            "AUTHORITATIVE",
        ),
        (
            "GRAPH_SCHEMA",
            GRAPH_V2,
            "AUTHORITATIVE",
        ),
        (
            "INDUCTIVE_CONSISTENCY",
            CONSISTENCY_V2,
            "AUTHORITATIVE",
        ),
        (
            "EXACT_ARCHITECTURE",
            ARCHITECTURE_V1,
            "AUTHORITATIVE",
        ),
        (
            "BRANCH_SEMANTICS",
            BRANCH_V2,
            "AUTHORITATIVE",
        ),
        (
            "COLD_MASK_SEMANTICS",
            COLD_V1,
            "AUTHORITATIVE",
        ),
        (
            "FOUNDER_HISTORY_SEMANTICS",
            FOUNDER_V1,
            "AUTHORITATIVE",
        ),
        (
            "OLD_PREFLIGHT",
            V1_CONTRACT,
            "FROZEN_SUPERSEDED_INVALID_ENTRY_GATE",
        ),
    ]

    for role, path, status in contract_sources:

        contract_rows.append(
            {
                "role":
                    role,

                "path":
                    str(path),

                "status":
                    status,

                "sha256":
                    sha256_file(
                        path
                    ),
            }
        )

    contract_df = pd.DataFrame(
        contract_rows
    )

    # =========================================================================
    # 7. Reusable source inventory
    # =========================================================================

    source_rows = []

    reusable_sources = [
        (
            "canonical_startup_source",
            STARTUP_SOURCE_V1,
            "REUSABLE_SUBSTRATE",
        ),
        (
            "founder_origin_history",
            FOUNDER_ORIGIN_TABLE_V2,
            "AUTHORITATIVE_V2",
        ),
        (
            "origin_cold_masks",
            COLD_MASKS_V2,
            "AUTHORITATIVE_V2",
        ),
        (
            "preprocessing_fit_population",
            PREPROCESS_POP_V2,
            "AUTHORITATIVE_V2",
        ),
        (
            "origin_population_summary",
            ORIGIN_SUMMARY_V2,
            "AUTHORITATIVE_V2",
        ),
    ]

    for role, path, status in reusable_sources:

        source_rows.append(
            {
                "role":
                    role,

                "path":
                    str(path),

                "status":
                    status,

                "sha256":
                    sha256_file(
                        path
                    ),
            }
        )

    source_df = pd.DataFrame(
        source_rows
    )

    work_df = pd.DataFrame(
        work_packages
    )

    # =========================================================================
    # 8. Build V2 preflight contract
    # =========================================================================

    sep("-")

    print(
        "8.2.0-V2.6 — BUILD CORRECTED ENTRY CONTRACT"
    )

    sep("-")

    contract = {
        "schema_version":
            "PHASE_8_2_0_IMPLEMENTATION_PREFLIGHT_V2",

        "status":
            "FROZEN",

        "scientific_role":
            "CORRECTED_FRAMEWORK_DEVELOPMENT_REENTRY_GATE",

        "supersession":
            {
                "supersedes":
                    str(
                        V1_CONTRACT
                    ),

                "V1_file_modified":
                    False,

                "V1_artifacts_preserved":
                    True,

                "V1_scientific_status":
                    "INVALID_ENTRY_GATE_FROZEN_SUPERSEDED",

                "reason":
                    (
                        "V1 adopted the superseded 8.1.12 V3 "
                        "execution contract and predates the "
                        "subsequent semantic repairs and corrected "
                        "8.2.1 materializations."
                    ),

                "performance_based_correction":
                    False,

                "phase8_candidate_performance_seen":
                    False,
            },

        "authoritative_phase8_execution_contract":
            str(
                EXEC_V5
            ),

        "authoritative_phase8_execution_contract_sha256":
            sha256_file(
                EXEC_V5
            ),

        "authoritative_contract_chain":
            {
                "graph_schema":
                    str(
                        GRAPH_V2
                    ),

                "inductive_consistency":
                    str(
                        CONSISTENCY_V2
                    ),

                "architecture":
                    str(
                        ARCHITECTURE_V1
                    ),

                "branch_semantics":
                    str(
                        BRANCH_V2
                    ),

                "cold_mask":
                    str(
                        COLD_V1
                    ),

                "founder_history":
                    str(
                        FOUNDER_V1
                    ),
            },

        "method_design_closed":
            True,

        "method_design_reopened":
            False,

        "feature_discovery_closed":
            True,

        "feature_discovery_reopened":
            False,

        "framework_development_started":
            True,

        "corrected_reentry_gate":
            True,

        "feature_materialization_started":
            True,

        "learned_preprocessing_started":
            False,

        "graph_materialization_started":
            False,

        "model_instantiated":
            False,

        "optimizer_instantiated":
            False,

        "performance_inspected":
            False,

        "training_performed":
            False,

        "training_allowed_now":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,

        "materialization_state_at_reentry":
            materialization_state,

        "artifact_ownership":
            {
                "raw_data_mutation":
                    False,

                "phase3_baseline_graph_mutation":
                    False,

                "phase4_baseline_model_mutation":
                    False,

                "phase6_baseline_result_mutation":
                    False,

                "superseded_contract_mutation":
                    False,

                "failed_script_policy":
                    (
                        "Preserve failed/superseded versions; "
                        "create incremented versions."
                    ),

                "new_phase8_artifacts_root":
                    str(
                        FRAMEWORK
                    ),

                "new_phase8_source_namespace":
                    "src/phase8",

                "performance_during_framework_development":
                    "FORBIDDEN",
            },

        "work_packages":
            work_packages,

        "next_phase":
            (
                "Phase 8.2.1D — origin-specific "
                "training-only preprocessing fit + "
                "feature transformation."
            ),

        "training_launch_condition":
            (
                "Only Phase 8.2.8 PASS may authorize "
                "comparative candidate-model training."
            ),
    }

    # =========================================================================
    # 9. Final integrity audit BEFORE persistence
    # =========================================================================

    sep("-")

    print(
        "8.2.0-V2.7 — FINAL REENTRY-GATE INTEGRITY AUDIT"
    )

    sep("-")

    serialized = json.dumps(
        contract,
        sort_keys=True,
        allow_nan=False,
    )

    forbidden = [
        "phase_8_1_12_execution_contract_V3.json",
        "held-out primary Startup",
        "HELDOUT_PRIMARY",
        "training_allowed_now\": true",
    ]

    for marker in forbidden:

        require(
            marker
            not in
            serialized,
            (
                "Superseded entry-gate semantic marker "
                f"survived into V2: {marker}"
            ),
        )

    required_markers = [
        "phase_8_1_12_execution_contract_V5.json",
        "phase_8_1_9_graph_schema_V2.json",
        "phase_8_1_10a_consistency_amendment_V2.json",
        "phase_8_1_11a_branch_semantics_amendment_V2.json",
        "ALL COLD_t",
        "8.2.1D",
        "8.2.8",
    ]

    for marker in required_markers:

        require(
            marker
            in
            serialized,
            (
                "Required corrected preflight marker "
                f"missing: {marker}"
            ),
        )

    require(
        contract[
            "method_design_closed"
        ]
        is True,
        "Method design is not closed.",
    )

    require(
        contract[
            "learned_preprocessing_started"
        ]
        is False,
        (
            "Learned preprocessing was incorrectly "
            "marked as already started."
        ),
    )

    require(
        contract[
            "training_allowed_now"
        ]
        is False,
        "Training was incorrectly authorized.",
    )

    require(
        contract[
            "performance_inspected"
        ]
        is False,
        (
            "Phase-8 candidate performance was "
            "incorrectly marked as inspected."
        ),
    )

    print(
        "V5 execution contract adopted:          PASS"
    )

    print(
        "Corrected semantic chain adopted:       PASS"
    )

    print(
        "Existing 8.2.1A/B/C state disclosed:    PASS"
    )

    print(
        "Stale held-out-primary wording absent:  PASS"
    )

    print(
        "Next step = 8.2.1D:                     PASS"
    )

    print(
        "Training remains blocked:               PASS"
    )

    # =========================================================================
    # 10. Persist
    # =========================================================================

    sep("-")

    print(
        "8.2.0-V2.8 — PERSIST CORRECTED PREFLIGHT"
    )

    sep("-")

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    contract_df.to_csv(
        CONTRACT_INVENTORY_V2,
        index=False,
    )

    source_df.to_csv(
        SOURCE_INVENTORY_V2,
        index=False,
    )

    work_df.to_csv(
        WORK_PLAN_V2,
        index=False,
    )

    write_json(
        CONTRACT_V2,
        contract,
    )

    manifest = {
        "phase":
            "8.2.0",

        "version":
            "V2",

        "status":
            "FROZEN",

        "scientific_role":
            "CORRECTED_FRAMEWORK_DEVELOPMENT_REENTRY_GATE",

        "V1_status":
            "INVALID_ENTRY_GATE_FROZEN_SUPERSEDED",

        "V1_artifacts_preserved":
            True,

        "authoritative_execution_contract":
            str(
                EXEC_V5
            ),

        "method_design_closed":
            True,

        "framework_development_started":
            True,

        "feature_materialization_started":
            True,

        "learned_preprocessing_started":
            False,

        "graph_materialization_started":
            False,

        "model_instantiated":
            False,

        "training_allowed_now":
            False,

        "training_performed":
            False,

        "performance_inspected":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,

        "next_phase":
            "8.2.1D",

        "outputs": [
            str(
                CONTRACT_INVENTORY_V2
            ),
            str(
                SOURCE_INVENTORY_V2
            ),
            str(
                WORK_PLAN_V2
            ),
            str(
                CONTRACT_V2
            ),
        ],
    }

    write_json(
        MANIFEST_V2,
        manifest,
    )

    outputs = [
        CONTRACT_INVENTORY_V2,
        SOURCE_INVENTORY_V2,
        WORK_PLAN_V2,
        CONTRACT_V2,
        MANIFEST_V2,
    ]

    write_json(
        HASHES_V2,
        {
            str(path):
                sha256_file(
                    path
                )
            for path
            in outputs
        },
    )

    sep()

    print(
        "PHASE 8.2.0 V2 — CORRECTED FRAMEWORK ENTRY GATE FROZEN"
    )

    sep()

    print()

    print(
        "8.2.0 V1:                     INVALID_ENTRY_GATE / FROZEN_SUPERSEDED"
    )

    print(
        "8.2.0 V2:                     FROZEN / AUTHORITATIVE"
    )

    print()

    print(
        "AUTHORITATIVE METHOD CONTRACT: 8.1.12 V5"
    )

    print(
        "METHOD DESIGN CLOSED:           YES"
    )

    print(
        "FEATURE MATERIALIZATION STARTED:YES"
    )

    print(
        "LEARNED PREPROCESSING STARTED:  NO"
    )

    print(
        "GRAPH MATERIALIZED:             NO"
    )

    print(
        "MODEL INSTANTIATED:             NO"
    )

    print(
        "TRAINING AUTHORIZED NOW:        NO"
    )

    print(
        "TRAINING PERFORMED:             NO"
    )

    print(
        "T60 SCORED:                     NO"
    )

    print(
        "APRIL ANALYZED:                 NO"
    )

    print()

    print("NEXT:")
    print(
        "Phase 8.2.1D — origin-specific training-only "
        "preprocessing fit + feature transformation."
    )


if __name__ == "__main__":
    main()
