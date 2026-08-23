from pathlib import Path
import hashlib
import json
import sys

import pandas as pd


# =============================================================================
# PHASE 4.8 — PHASE-4 CLOSURE AND HANDOFF
#
# PURPOSE
# -------
# Produce the final authoritative closure package for:
#
#   PHASE 4 — MODEL RECONSTRUCTION
#
# No new modeling decision is introduced here.
#
# This script packages the already-frozen reconstruction into:
#
#   1. final Phase-4 closure manifest,
#   2. Phase-5 handoff contract,
#   3. final model decision register,
#   4. authoritative artifact registry,
#   5. parameter summary,
#   6. deferred training/evaluation decision register,
#   7. known limitations / reproduction adaptations,
#   8. repository reproduction-log entry,
#   9. closure artifact hashes.
#
#
# THIS SCRIPT DOES NOT:
#
#   - train,
#   - instantiate neural parameters,
#   - run forward,
#   - run backward,
#   - create an optimizer,
#   - generate negatives,
#   - choose negative sampling,
#   - choose epochs,
#   - choose early stopping,
#   - choose weight decay,
#   - change Phase 2,
#   - change Phase 3,
#   - change any Phase-4 contract.
#
#
# PHASE-4 CLOSURE PRINCIPLE
# -------------------------
# Phase 4 reconstructs the MODEL.
#
# Training and evaluation policy remains a later concern.
# =============================================================================


# =============================================================================
# ROOTS
# =============================================================================

PHASE_2_ROOT = Path(
    "data/experimental/phase_2"
)

PHASE_3_ROOT = Path(
    "data/experimental/phase_3"
)

PHASE_4_ROOT = Path(
    "data/experimental/phase_4"
)


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

OUT_DIR = (
    PHASE_4_ROOT
    / "closure"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FROZEN PHASE-2 HANDOFF INPUT
# =============================================================================

TEMPORAL_SPLIT_PATH = (
    PHASE_2_ROOT
    / "model_ready"
    / "interactions_itrs_temporal_split.parquet"
)

T60_HOLDOUT_MANIFEST_PATH = (
    PHASE_2_ROOT
    / "model_ready"
    / "t60_holdout_pair_manifest.parquet"
)


# =============================================================================
# FROZEN PHASE-3 HANDOFF INPUTS
# =============================================================================

NODE_INDEX_PATH = (
    PHASE_3_ROOT
    / "model_ready"
    / "node_index.parquet"
)

RELATION_INDEX_PATH = (
    PHASE_3_ROOT
    / "model_ready"
    / "relation_index.csv"
)

EDGE_INDEX_PATH = (
    PHASE_3_ROOT
    / "model_ready"
    / "edge_index.npy"
)

EDGE_TYPE_PATH = (
    PHASE_3_ROOT
    / "model_ready"
    / "edge_type.npy"
)

GRAPH_VARIANT_MASKS_PATH = (
    PHASE_3_ROOT
    / "model_ready"
    / "graph_variant_masks.npz"
)


# =============================================================================
# PHASE-4 STATIC DESCRIPTION INPUTS
# =============================================================================

DOC2VEC_ALL_PATH = (
    PHASE_4_ROOT
    / "doc2vec"
    / "vectors"
    / "doc2vec_vectors_all.npy"
)

DOC2VEC_MANIFEST_PATH = (
    PHASE_4_ROOT
    / "doc2vec"
    / "vectors"
    / "doc2vec_vector_manifest.parquet"
)

LABEL_MATRIX_PATH = (
    PHASE_4_ROOT
    / "description_labels"
    / "description_label_multihot.npz"
)

LABEL_MANIFEST_PATH = (
    PHASE_4_ROOT
    / "description_labels"
    / "description_label_vector_manifest.parquet"
)


# =============================================================================
# PHASE-4 TREND RUNTIME
# =============================================================================

TREND_PERIOD_PTR_PATH = (
    PHASE_4_ROOT
    / "trend_runtime"
    / "trend_period_ptr.npy"
)

TREND_STARTUP_INDICES_PATH = (
    PHASE_4_ROOT
    / "trend_runtime"
    / "trend_startup_node_indices.npy"
)

TREND_PERIOD_COUNTS_PATH = (
    PHASE_4_ROOT
    / "trend_runtime"
    / "trend_period_startup_counts.npy"
)


# =============================================================================
# PHASE-4 MODEL CONTRACTS
# =============================================================================

DOC2VEC_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "doc2vec_contract"
    / "doc2vec_contract.json"
)

DESCRIPTION_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "description_contract"
    / "description_contract.json"
)

DESCRIPTION_NEURAL_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "description_neural_contract"
    / "description_neural_contract.json"
)

TREND_HISTORY_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "trend_contract"
    / "trend_history_semantics_contract.json"
)

TREND_NEURAL_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "trend_neural_contract"
    / "trend_neural_contract.json"
)

TREND_RUNTIME_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "trend_runtime"
    / "trend_runtime_contract.json"
)

RGCN_NEURAL_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "rgcn_neural_contract"
    / "rgcn_neural_contract.json"
)

RGCN_INTEGRATION_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "rgcn_integration"
    / "rgcn_integration_contract.json"
)

PHASE_4_4_CLOSURE_PATH = (
    PHASE_4_ROOT
    / "rgcn_integration"
    / "phase_4_4_closure_manifest.json"
)

SCORING_INPUT_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "scoring_contract"
    / "scoring_input_contract.json"
)

SCORING_NEURAL_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "scoring_neural_contract"
    / "scoring_neural_contract.json"
)

SCORING_FORWARD_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "scoring_module"
    / "scoring_forward_contract.json"
)

PHASE_4_5_CLOSURE_PATH = (
    PHASE_4_ROOT
    / "scoring_module"
    / "phase_4_5_closure_manifest.json"
)

STATIC_INPUT_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "full_model_contract"
    / "full_model_static_input_contract.json"
)

FULL_MODEL_TOPOLOGY_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "full_model_contract"
    / "full_itrs_model_topology_contract.json"
)

PARAMETER_NAMESPACE_PATH = (
    PHASE_4_ROOT
    / "full_model_contract"
    / "full_model_parameter_namespace.csv"
)

COMPONENT_PARAMETER_AUDIT_PATH = (
    PHASE_4_ROOT
    / "full_model_contract"
    / "full_model_component_parameter_audit.csv"
)


# =============================================================================
# PHASE-4.6
# =============================================================================

PHASE_4_6_2_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "full_model_forward_audit"
    / "phase_4_6_2_end_to_end_contract.json"
)

PHASE_4_6_CLOSURE_PATH = (
    PHASE_4_ROOT
    / "full_model_forward_audit"
    / "phase_4_6_closure_manifest.json"
)


# =============================================================================
# PHASE-4.7 INITIALIZATION
# =============================================================================

INITIALIZATION_CONTRACT_PATH = (
    PHASE_4_ROOT
    / "initialization_contract"
    / "phase_4_7_1b_neural_initialization_contract.json"
)

INITIALIZATION_STATE_HASH_PATH = (
    PHASE_4_ROOT
    / "initialization_contract"
    / "phase_4_7_1b_initialization_state_hash.json"
)


# =============================================================================
# PHASE-4.7 INTEGRITY
# =============================================================================

PHASE_4_7_1A_METADATA_PATH = (
    PHASE_4_ROOT
    / "model_integrity_audit"
    / "phase_4_7_1a_integrity_metadata.json"
)

PHASE_4_7_2_METADATA_PATH = (
    PHASE_4_ROOT
    / "model_integrity_audit"
    / "phase_4_7_2_post_initialization_integrity_metadata.json"
)

PHASE_4_7_CLOSURE_PATH = (
    PHASE_4_ROOT
    / "model_integrity_audit"
    / "phase_4_7_closure_manifest.json"
)


# =============================================================================
# FROZEN POPULATION
# =============================================================================

NUM_INVESTORS = 165_975
NUM_STARTUPS = 311_589
NUM_NODES = 477_564

NUM_RELATIONS = 12
NUM_STRUCTURAL_EDGES = 158_818


# =============================================================================
# FROZEN PARAMETER TOTALS
# =============================================================================

FULL_PARAMETER_TOTAL = 19_217_929

PARAMETER_TENSOR_COUNT = 32


# =============================================================================
# EXPECTED COMPONENT PARAMETER COUNTS
# =============================================================================

EXPECTED_PARAMETER_COMPONENTS = {

    "Investor latent embeddings":
        6_639_000,

    "Startup latent embeddings":
        12_463_560,

    "Description encoder":
        16_720,

    "Trend module":
        32_480,

    "R-GCN":
        19_320,

    "Scoring MLP":
        46_849,
}


# =============================================================================
# FROZEN DEFERRED DECISIONS
# =============================================================================

DEFERRED_DECISIONS = {

    "training negative:positive ratio",

    "training negative candidate eligibility",

    "training historical negative exclusion",

    "training epoch count",

    "early stopping",

    "weight decay",

    "evaluation candidate-generation runtime contract",
}


# =============================================================================
# HELPERS
# =============================================================================

def banner(title):

    print()
    print("=" * 120)
    print(title)
    print("=" * 120)


def require(
    condition,
    message,
):

    if not condition:

        raise AssertionError(
            message
        )


def load_json(path):

    require(
        path.exists(),
        f"Missing required JSON: {path}",
    )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def sha256_file(path):

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        for chunk in iter(
            lambda:
                f.read(
                    1024 * 1024
                ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


# =============================================================================
# START
# =============================================================================

banner(
    "PHASE 4.8 — "
    "PHASE-4 MODEL RECONSTRUCTION CLOSURE AND HANDOFF"
)


# =============================================================================
# 1. ENVIRONMENT
# =============================================================================

banner(
    "ENVIRONMENT"
)


print(
    f"Python: "
    f"{sys.version.splitlines()[0]}"
)


# =============================================================================
# 2. REQUIRED FILE EXISTENCE
# =============================================================================

banner(
    "REQUIRED CLOSURE INPUT EXISTENCE"
)


required_paths = [

    TEMPORAL_SPLIT_PATH,
    T60_HOLDOUT_MANIFEST_PATH,

    NODE_INDEX_PATH,
    RELATION_INDEX_PATH,
    EDGE_INDEX_PATH,
    EDGE_TYPE_PATH,
    GRAPH_VARIANT_MASKS_PATH,

    DOC2VEC_ALL_PATH,
    DOC2VEC_MANIFEST_PATH,
    LABEL_MATRIX_PATH,
    LABEL_MANIFEST_PATH,

    TREND_PERIOD_PTR_PATH,
    TREND_STARTUP_INDICES_PATH,
    TREND_PERIOD_COUNTS_PATH,

    DOC2VEC_CONTRACT_PATH,
    DESCRIPTION_CONTRACT_PATH,
    DESCRIPTION_NEURAL_CONTRACT_PATH,

    TREND_HISTORY_CONTRACT_PATH,
    TREND_NEURAL_CONTRACT_PATH,
    TREND_RUNTIME_CONTRACT_PATH,

    RGCN_NEURAL_CONTRACT_PATH,
    RGCN_INTEGRATION_CONTRACT_PATH,
    PHASE_4_4_CLOSURE_PATH,

    SCORING_INPUT_CONTRACT_PATH,
    SCORING_NEURAL_CONTRACT_PATH,
    SCORING_FORWARD_CONTRACT_PATH,
    PHASE_4_5_CLOSURE_PATH,

    STATIC_INPUT_CONTRACT_PATH,
    FULL_MODEL_TOPOLOGY_CONTRACT_PATH,
    PARAMETER_NAMESPACE_PATH,
    COMPONENT_PARAMETER_AUDIT_PATH,

    PHASE_4_6_2_CONTRACT_PATH,
    PHASE_4_6_CLOSURE_PATH,

    INITIALIZATION_CONTRACT_PATH,
    INITIALIZATION_STATE_HASH_PATH,

    PHASE_4_7_1A_METADATA_PATH,
    PHASE_4_7_2_METADATA_PATH,
    PHASE_4_7_CLOSURE_PATH,
]


for path in required_paths:

    exists = path.exists()

    print(
        f"{str(path):<110} "
        f"{'FOUND' if exists else 'MISSING'}"
    )

    require(
        exists,
        f"Missing Phase-4 closure input: {path}",
    )


# =============================================================================
# 3. VERIFY ALL MAJOR CONTRACT STATUSES
# =============================================================================

banner(
    "FINAL CONTRACT STATUS AUDIT"
)


status_expectations = [

    (
        "Doc2Vec contract",
        DOC2VEC_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Description input contract",
        DESCRIPTION_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Description neural contract",
        DESCRIPTION_NEURAL_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Trend history contract",
        TREND_HISTORY_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Trend neural contract",
        TREND_NEURAL_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Trend runtime contract",
        TREND_RUNTIME_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "R-GCN neural contract",
        RGCN_NEURAL_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "R-GCN integration contract",
        RGCN_INTEGRATION_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Phase 4.4 closure",
        PHASE_4_4_CLOSURE_PATH,
        "COMPLETE",
    ),

    (
        "Scoring input contract",
        SCORING_INPUT_CONTRACT_PATH,
        "FROZEN_INPUT_CONTRACT",
    ),

    (
        "Scoring neural contract",
        SCORING_NEURAL_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Scoring forward contract",
        SCORING_FORWARD_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Phase 4.5 closure",
        PHASE_4_5_CLOSURE_PATH,
        "COMPLETE",
    ),

    (
        "Static input contract",
        STATIC_INPUT_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Full topology contract",
        FULL_MODEL_TOPOLOGY_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Phase 4.6 integration contract",
        PHASE_4_6_2_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Phase 4.6 closure",
        PHASE_4_6_CLOSURE_PATH,
        "COMPLETE",
    ),

    (
        "Initialization contract",
        INITIALIZATION_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Phase 4.7.1a integrity",
        PHASE_4_7_1A_METADATA_PATH,
        "COMPLETE_AUDIT_ONLY",
    ),

    (
        "Phase 4.7.2 integrity",
        PHASE_4_7_2_METADATA_PATH,
        "COMPLETE_AUDIT_ONLY",
    ),

    (
        "Phase 4.7 closure",
        PHASE_4_7_CLOSURE_PATH,
        "COMPLETE",
    ),
]


status_records = []


for (
    name,
    path,
    expected_status,
) in status_expectations:

    payload = load_json(
        path
    )

    actual_status = payload.get(
        "status"
    )

    exact = (
        actual_status
        == expected_status
    )

    print(
        f"{name:<34} "
        f"{str(actual_status):<24} "
        f"{'PASS' if exact else 'FAIL'}"
    )

    require(
        exact,
        f"Final closure status mismatch: {name}",
    )

    status_records.append(
        {

            "component":
                name,

            "path":
                str(
                    path
                ),

            "expected_status":
                expected_status,

            "actual_status":
                actual_status,

            "status":
                "PASS",
        }
    )


# =============================================================================
# 4. LOAD AUTHORITATIVE FINAL CONTRACTS
# =============================================================================

banner(
    "AUTHORITATIVE FINAL CONTRACTS"
)


static_contract = load_json(
    STATIC_INPUT_CONTRACT_PATH
)

topology_contract = load_json(
    FULL_MODEL_TOPOLOGY_CONTRACT_PATH
)

integration_contract = load_json(
    PHASE_4_6_2_CONTRACT_PATH
)

initialization_contract = load_json(
    INITIALIZATION_CONTRACT_PATH
)

initialization_state = load_json(
    INITIALIZATION_STATE_HASH_PATH
)

phase_4_7_metadata = load_json(
    PHASE_4_7_2_METADATA_PATH
)

phase_4_7_closure = load_json(
    PHASE_4_7_CLOSURE_PATH
)


# =============================================================================
# 5. FINAL MODEL DIMENSION / PARAMETER CONTRACT
# =============================================================================

banner(
    "FINAL MODEL ARCHITECTURE CONTRACT"
)


require(
    topology_contract[
        "parameter_budget"
    ][
        "total"
    ]
    == FULL_PARAMETER_TOTAL,
    "Final model parameter total changed.",
)


require(
    topology_contract[
        "latent_embeddings"
    ][
        "number_of_tables"
    ]
    == 2,
    "Final latent table count changed.",
)


require(
    topology_contract[
        "latent_embeddings"
    ][
        "investor"
    ][
        "shape"
    ]
    == [
        NUM_INVESTORS,
        40,
    ],
    "Investor latent-table shape changed.",
)


require(
    topology_contract[
        "latent_embeddings"
    ][
        "startup"
    ][
        "shape"
    ]
    == [
        NUM_STARTUPS,
        40,
    ],
    "Startup latent-table shape changed.",
)


require(
    topology_contract[
        "description"
    ][
        "output_dim"
    ]
    == 40,
    "Description dimension changed.",
)


require(
    topology_contract[
        "trend"
    ][
        "output_dim"
    ]
    == 40,
    "Trend dimension changed.",
)


require(
    topology_contract[
        "preference_propagation"
    ][
        "output_dim"
    ]
    == 40,
    "Structural dimension changed.",
)


require(
    topology_contract[
        "scoring"
    ][
        "pair_dimension"
    ]
    == 280,
    "Scoring pair dimension changed.",
)


require(
    topology_contract[
        "scoring"
    ][
        "architecture"
    ]
    == [
        280,
        128,
        64,
        32,
        16,
        1,
    ],
    "Scoring architecture changed.",
)


print(
    f"Investor latent:       "
    f"[{NUM_INVESTORS}, 40]"
)

print(
    f"Startup latent:        "
    f"[{NUM_STARTUPS}, 40]"
)

print(
    "Description dimension: 40"
)

print(
    "Trend dimension:       40"
)

print(
    "Structural dimension:  40"
)

print(
    "Pair dimension:        280"
)

print(
    "Scorer:                "
    "280 -> 128 -> 64 -> 32 -> 16 -> 1"
)

print(
    f"Trainable parameters:   "
    f"{FULL_PARAMETER_TOTAL:,}"
)


# =============================================================================
# 6. PARAMETER NAMESPACE / COMPONENT TOTAL FINAL CHECK
# =============================================================================

banner(
    "FINAL PARAMETER BUDGET CHECK"
)


parameter_namespace = pd.read_csv(
    PARAMETER_NAMESPACE_PATH
)

component_parameters = pd.read_csv(
    COMPONENT_PARAMETER_AUDIT_PATH
)


require(
    {
        "parameter",
        "numel",
    }
    .issubset(
        parameter_namespace.columns
    ),
    "Parameter namespace schema changed.",
)


require(
    len(
        parameter_namespace
    )
    == PARAMETER_TENSOR_COUNT,
    "Parameter tensor count changed.",
)


namespace_total = int(
    parameter_namespace[
        "numel"
    ].sum()
)


require(
    namespace_total
    == FULL_PARAMETER_TOTAL,
    "Parameter namespace total changed.",
)


parameter_summary_records = []


for (
    component,
    expected,
) in EXPECTED_PARAMETER_COMPONENTS.items():

    rows = component_parameters[
        component_parameters[
            "component"
        ]
        == component
    ]

    require(
        len(
            rows
        )
        == 1,
        (
            "Could not uniquely resolve component "
            f"parameter count: {component}"
        ),
    )

    actual = int(
        rows.iloc[
            0
        ][
            "actual_parameters"
        ]
    )

    require(
        actual
        == expected,
        (
            "Component parameter count changed: "
            f"{component}"
        ),
    )

    print(
        f"{component:<34} "
        f"{actual:>12,} PASS"
    )

    parameter_summary_records.append(
        {

            "component":
                component,

            "parameters":
                actual,

            "status":
                "FROZEN",
        }
    )


print(
    "-" * 55
)

print(
    f"{'FULL ITRS':<34} "
    f"{namespace_total:>12,} PASS"
)


# =============================================================================
# 7. FINAL INITIALIZATION CONTRACT
# =============================================================================

banner(
    "FINAL INITIALIZATION CONTRACT"
)


require(
    initialization_contract[
        "global_neural_seed"
    ][
        "value"
    ]
    == 42,
    "Global neural seed changed.",
)


require(
    initialization_contract[
        "kaiming"
    ][
        "function"
    ]
    == "torch.nn.init.kaiming_normal_",
    "Kaiming initialization function changed.",
)


require(
    initialization_contract[
        "kaiming"
    ][
        "distribution"
    ]
    == "normal",
    "Kaiming distribution changed.",
)


require(
    initialization_contract[
        "kaiming"
    ][
        "mode"
    ]
    == "fan_in",
    "Kaiming mode changed.",
)


require(
    initialization_contract[
        "kaiming"
    ][
        "nonlinearity"
    ]
    == "relu",
    "Kaiming nonlinearity changed.",
)


require(
    float(
        initialization_contract[
            "kaiming"
        ][
            "a"
        ]
    )
    == 0.0,
    "Kaiming a changed.",
)


require(
    initialization_contract[
        "bias"
    ][
        "value"
    ]
    == 0.0,
    "Bias initialization changed.",
)


canonical_state_sha256 = (
    initialization_state[
        "canonical_state_sha256"
    ]
)


require(
    canonical_state_sha256
    ==
    initialization_contract[
        "reproducibility"
    ][
        "canonical_state_sha256"
    ],
    "Canonical initialization fingerprint changed.",
)


print(
    "Function:       "
    "torch.nn.init.kaiming_normal_"
)

print(
    "Distribution:   normal"
)

print(
    "a:              0.0"
)

print(
    "mode:           fan_in"
)

print(
    "nonlinearity:   relu"
)

print(
    "bias:           zero"
)

print(
    "global seed:    42"
)

print(
    "canonical init: CPU"
)

print(
    "reference torch:2.7.0"
)

print()
print(
    "Canonical state SHA256:"
)

print(
    f"  {canonical_state_sha256}"
)


# =============================================================================
# 8. TRAINING BOUNDARY FINAL CHECK
# =============================================================================

banner(
    "PHASE-4 TRAINING BOUNDARY"
)


require(
    integration_contract[
        "training"
    ][
        "performed"
    ]
    is False,
    "Phase 4.6 unexpectedly records training.",
)


require(
    initialization_contract[
        "training"
    ][
        "performed"
    ]
    is False,
    "Phase 4.7 unexpectedly records training.",
)


require(
    phase_4_7_metadata[
        "training_performed"
    ]
    is False,
    "Phase 4.7.2 unexpectedly records training.",
)


print(
    "Training performed:       NO"
)

print(
    "Optimizer created:        NO"
)

print(
    "optimizer.step executed:  NO"
)

print(
    "Training epochs executed: NO"
)

print(
    "Negative sampling frozen: NO"
)

print(
    "Learned model persisted:  NO"
)


# =============================================================================
# 9. DEFERRED DECISION FINAL CHECK
# =============================================================================

banner(
    "DEFERRED TRAINING / EVALUATION DECISIONS"
)


actual_deferred = set(
    phase_4_7_closure[
        "deferred_to_training_evaluation"
    ]
)


require(
    actual_deferred
    == DEFERRED_DECISIONS,
    "Final deferred-decision set changed.",
)


deferred_records = []


for decision in sorted(
    DEFERRED_DECISIONS
):

    print(
        f"  - {decision}"
    )

    deferred_records.append(
        {

            "decision":
                decision,

            "status":
                "DEFERRED",

            "owner_phase":
                "Phase 5 — Training and Evaluation",

            "phase_4_changed":
                False,
        }
    )


require(
    len(
        phase_4_7_closure[
            "model_reconstruction_blockers_remaining"
        ]
    )
    == 0,
    "Phase-4 reconstruction blocker remains.",
)


print()
print(
    "Phase-4 model-reconstruction blockers:"
)

print(
    "  NONE"
)


# =============================================================================
# 10. FINAL DECISION REGISTER
# =============================================================================

banner(
    "BUILDING FINAL MODEL DECISION REGISTER"
)


decision_records = [

    # -------------------------------------------------------------------------
    # DESCRIPTION
    # -------------------------------------------------------------------------

    {
        "component":
            "Description",

        "decision":
            "Static textual embedding dimension",

        "value":
            "32",

        "classification":
            "PAPER_SPECIFIED",

        "frozen_phase":
            "4.2",
    },

    {
        "component":
            "Description",

        "decision":
            "Text neural branch",

        "value":
            "Linear(32,20,bias=True) -> ReLU",

        "classification":
            "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE",

        "frozen_phase":
            "4.2",
    },

    {
        "component":
            "Description",

        "decision":
            "Category input dimension",

        "value":
            "802",

        "classification":
            "CRUNCHBASE_SCHEMA_ADAPTATION_FROZEN",

        "frozen_phase":
            "4.2",
    },

    {
        "component":
            "Description",

        "decision":
            "Category neural branch",

        "value":
            "Linear(802,20,bias=True) -> ReLU",

        "classification":
            "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE",

        "frozen_phase":
            "4.2",
    },

    {
        "component":
            "Description",

        "decision":
            "Final description representation",

        "value":
            "20 text + 20 label = 40",

        "classification":
            "PAPER_DIMENSION_PRESERVED_REPRODUCTION_SPLIT",

        "frozen_phase":
            "4.2",
    },

    {
        "component":
            "Description",

        "decision":
            "Missing static description feature",

        "value":
            "zero vector / all-zero category vector",

        "classification":
            "FROZEN_REPRODUCTION_CONTRACT",

        "frozen_phase":
            "4.2",
    },


    # -------------------------------------------------------------------------
    # TREND
    # -------------------------------------------------------------------------

    {
        "component":
            "Trend",

        "decision":
            "Investor trend query",

        "value":
            "[L_o(40) || F_d,o(40)] = 80",

        "classification":
            "PAPER_SPECIFIED",

        "frozen_phase":
            "4.3",
    },

    {
        "component":
            "Trend",

        "decision":
            "Historical Startup item",

        "value":
            "[L_b(40) || F_d,b(40)] = 80",

        "classification":
            "PAPER_SPECIFIED",

        "frozen_phase":
            "4.3",
    },

    {
        "component":
            "Trend",

        "decision":
            "Target-period temporal semantics",

        "value":
            "T_h consumes history T0..T(h-1)",

        "classification":
            "FROZEN_PAPER_OPERATIONAL_MAPPING",

        "frozen_phase":
            "4.3",
    },

    {
        "component":
            "Trend",

        "decision":
            "Repeated Investor-Startup membership within period",

        "value":
            "collapse to one attention item",

        "classification":
            "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE",

        "frozen_phase":
            "4.3",
    },

    {
        "component":
            "Trend",

        "decision":
            "Empty historical period",

        "value":
            "explicit zero80",

        "classification":
            "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE",

        "frozen_phase":
            "4.3",
    },

    {
        "component":
            "Trend",

        "decision":
            "Trend attention",

        "value":
            "bilinear attention W[80,80] + softmax",

        "classification":
            "PAPER_SPECIFIED",

        "frozen_phase":
            "4.3",
    },

    {
        "component":
            "Trend",

        "decision":
            "Trend recurrent model",

        "value":
            "GRU input80 hidden40 layers2",

        "classification":
            "PAPER_SPECIFIED",

        "frozen_phase":
            "4.3",
    },

    {
        "component":
            "Trend",

        "decision":
            "Trend output",

        "value":
            "Linear(40,40,bias=False) -> Sigmoid",

        "classification":
            "PAPER_SPECIFIED_RECONSTRUCTION",

        "frozen_phase":
            "4.3",
    },


    # -------------------------------------------------------------------------
    # PREFERENCE PROPAGATION
    # -------------------------------------------------------------------------

    {
        "component":
            "Preference propagation",

        "decision":
            "Initial structural representation",

        "value":
            "cat(L_o.weight,L_b.weight) [477564,40]",

        "classification":
            "PAPER_SPECIFIED_WITH_FROZEN_ROLE_IDENTITY",

        "frozen_phase":
            "4.4",
    },

    {
        "component":
            "Preference propagation",

        "decision":
            "Structural relation channels",

        "value":
            "12 typed source-relation-target channels",

        "classification":
            "CRUNCHBASE_GRAPH_ADAPTATION_FROZEN",

        "frozen_phase":
            "4.4",
    },

    {
        "component":
            "Preference propagation",

        "decision":
            "R-GCN topology",

        "value":
            "2 layers, 5 bases, 40 -> 40 -> 40",

        "classification":
            "PAPER_SPECIFIED",

        "frozen_phase":
            "4.4",
    },

    {
        "component":
            "Preference propagation",

        "decision":
            "Relation aggregation",

        "value":
            "relation-specific incoming mean + separate root transform",

        "classification":
            "PAPER_SPECIFIED_RECONSTRUCTION",

        "frozen_phase":
            "4.4",
    },

    {
        "component":
            "Preference propagation",

        "decision":
            "R-GCN activation",

        "value":
            "ReLU after each layer",

        "classification":
            "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE",

        "frozen_phase":
            "4.4",
    },

    {
        "component":
            "Preference propagation",

        "decision":
            "R-GCN extras",

        "value":
            "no layer bias; dropout0; no BN/LN/residual",

        "classification":
            "PAPER_UNSPECIFIED_NO_ADDED_COMPONENT",

        "frozen_phase":
            "4.4",
    },


    # -------------------------------------------------------------------------
    # SCORING
    # -------------------------------------------------------------------------

    {
        "component":
            "Scoring",

        "decision":
            "Investor scorer representation",

        "value":
            "[F_t || L_o || F_d,o || F_s,o] = 160",

        "classification":
            "PAPER_SPECIFIED",

        "frozen_phase":
            "4.5",
    },

    {
        "component":
            "Scoring",

        "decision":
            "Startup scorer representation",

        "value":
            "[L_b || F_d,b || F_s,b] = 120",

        "classification":
            "PAPER_SPECIFIED",

        "frozen_phase":
            "4.5",
    },

    {
        "component":
            "Scoring",

        "decision":
            "Pair feature order",

        "value":
            "F_t,L_o,F_d,o,F_s,o,L_b,F_d,b,F_s,b",

        "classification":
            "PAPER_SPECIFIED",

        "frozen_phase":
            "4.5",
    },

    {
        "component":
            "Scoring",

        "decision":
            "Scoring MLP",

        "value":
            "280 -> 128 -> 64 -> 32 -> 16 -> 1",

        "classification":
            "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE_GUIDED_BY_NCF",

        "frozen_phase":
            "4.5",
    },

    {
        "component":
            "Scoring",

        "decision":
            "Scoring activation",

        "value":
            "ReLU after four hidden layers",

        "classification":
            "PAPER_SPECIFIED",

        "frozen_phase":
            "4.5",
    },

    {
        "component":
            "Scoring",

        "decision":
            "Output / loss implementation",

        "value":
            "raw logit + BCEWithLogitsLoss; sigmoid for probability",

        "classification":
            "MATHEMATICALLY_EQUIVALENT_NUMERICALLY_STABLE_IMPLEMENTATION",

        "frozen_phase":
            "4.5",
    },

    {
        "component":
            "Scoring",

        "decision":
            "Scorer extras",

        "value":
            "dropout0; no normalization; no residual",

        "classification":
            "PAPER_UNSPECIFIED_NO_ADDED_COMPONENT",

        "frozen_phase":
            "4.5",
    },


    # -------------------------------------------------------------------------
    # SHARED LATENT IDENTITY
    # -------------------------------------------------------------------------

    {
        "component":
            "Shared latent representation",

        "decision":
            "Number of latent embedding tables",

        "value":
            "2: Investor L_o and Startup L_b",

        "classification":
            "PAPER_SPECIFIED_RECONSTRUCTION",

        "frozen_phase":
            "4.6",
    },

    {
        "component":
            "Shared latent representation",

        "decision":
            "Latent reuse",

        "value":
            "same L_o/L_b used by trend, R-GCN and scorer",

        "classification":
            "FROZEN_MODEL_IDENTITY_CONTRACT",

        "frozen_phase":
            "4.6",
    },


    # -------------------------------------------------------------------------
    # INITIALIZATION
    # -------------------------------------------------------------------------

    {
        "component":
            "Initialization",

        "decision":
            "Initialization family",

        "value":
            "Kaiming",

        "classification":
            "PAPER_SPECIFIED",

        "frozen_phase":
            "4.7",
    },

    {
        "component":
            "Initialization",

        "decision":
            "Exact Kaiming variant",

        "value":
            "kaiming_normal_, a=0, fan_in, relu",

        "classification":
            "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE",

        "frozen_phase":
            "4.7",
    },

    {
        "component":
            "Initialization",

        "decision":
            "Bias initialization",

        "value":
            "zero",

        "classification":
            "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE",

        "frozen_phase":
            "4.7",
    },

    {
        "component":
            "Initialization",

        "decision":
            "Global neural seed",

        "value":
            "42",

        "classification":
            "PAPER_UNSPECIFIED_REPRODUCTION_CHOICE",

        "frozen_phase":
            "4.7",
    },

    {
        "component":
            "Initialization",

        "decision":
            "Canonical initialization environment",

        "value":
            "CPU, dedicated torch.Generator, PyTorch 2.7.0",

        "classification":
            "REPRODUCTION_ENVIRONMENT_LOCK",

        "frozen_phase":
            "4.7",
    },
]


decision_df = pd.DataFrame(
    decision_records
)


decision_path = (
    OUT_DIR
    / "phase_4_final_model_decision_register.csv"
)


decision_df.to_csv(
    decision_path,
    index=False,
)


print(
    f"Final model decisions registered: "
    f"{len(decision_df)}"
)


# =============================================================================
# 11. KNOWN LIMITATIONS / ADAPTATIONS
# =============================================================================

banner(
    "BUILDING KNOWN LIMITATIONS / ADAPTATIONS REGISTER"
)


limitation_records = [

    {
        "item":
            "Original ITRS relation vocabulary",

        "detail":
            (
                "Original exact 103-relation vocabulary "
                "is not recoverable from the paper/data."
            ),

        "reproduction_handling":
            (
                "Use frozen 12 typed Crunchbase "
                "source-relation-target channels."
            ),
    },

    {
        "item":
            "Historical investment structural edges",

        "detail":
            (
                "Historical investment events are not "
                "used as Phase-3 structural R-GCN edges."
            ),

        "reproduction_handling":
            (
                "Investment history enters through the "
                "trend module and supervised examples."
            ),
    },

    {
        "item":
            "Description temporal provenance",

        "detail":
            (
                "Crunchbase text/category description "
                "features are current-snapshot unversioned."
            ),

        "reproduction_handling":
            (
                "Frozen as a documented dataset limitation; "
                "not silently treated as historical snapshots."
            ),
    },

    {
        "item":
            "Shared-founder temporal provenance",

        "detail":
            (
                "Founder-based structural relation uses "
                "current-snapshot unversioned provenance."
            ),

        "reproduction_handling":
            (
                "Frozen and explicitly documented in "
                "Phase 3 / Phase 4 handoff."
            ),
    },

    {
        "item":
            "Structural sparsity",

        "detail":
            (
                "Only 74,757 of 477,564 role nodes are "
                "structurally connected."
            ),

        "reproduction_handling":
            (
                "Isolates remain in the model and receive "
                "R-GCN root/self transformation only."
            ),
    },

    {
        "item":
            "T60 structural coverage",

        "detail":
            (
                "72.1682% of frozen T60 holdout pairs have "
                "neither endpoint structurally connected."
            ),

        "reproduction_handling":
            (
                "Retained; no structural-coverage eligibility "
                "filter is introduced."
            ),
    },

    {
        "item":
            "T60 Investor cold start",

        "detail":
            (
                "25.7489% of T60 Investors have no prior "
                "investment history."
            ),

        "reproduction_handling":
            (
                "Retained; no minimum-history threshold. "
                "No-history trend uses the frozen zero-period "
                "sequence semantics."
            ),
    },

    {
        "item":
            "Scoring hidden widths",

        "detail":
            (
                "ITRS reports four hidden scoring layers "
                "but not their widths."
            ),

        "reproduction_handling":
            (
                "Freeze [128,64,32,16], classified as "
                "NCF-guided reproduction choice."
            ),
    },

    {
        "item":
            "R-GCN sigma activation",

        "detail":
            (
                "ITRS Eq. 9 uses sigma but does not explicitly "
                "define the activation."
            ),

        "reproduction_handling":
            (
                "Freeze ReLU and classify as "
                "paper-unspecified reproduction choice."
            ),
    },

    {
        "item":
            "Training/evaluation policy",

        "detail":
            (
                "Phase 4 reconstructed and audited the model "
                "but deliberately did not train or evaluate it."
            ),

        "reproduction_handling":
            (
                "Seven training/evaluation decisions are "
                "handed to Phase 5 unchanged."
            ),
    },
]


limitation_df = pd.DataFrame(
    limitation_records
)


limitation_path = (
    OUT_DIR
    / "phase_4_known_limitations_and_adaptations.csv"
)


limitation_df.to_csv(
    limitation_path,
    index=False,
)


# =============================================================================
# 12. AUTHORITATIVE ARTIFACT REGISTRY
# =============================================================================

banner(
    "BUILDING AUTHORITATIVE ARTIFACT REGISTRY"
)


artifact_specs = [

    # -------------------------------------------------------------------------
    # Frozen Phase-2 inputs
    # -------------------------------------------------------------------------

    (
        "Phase 2",
        "Temporal interaction split",
        TEMPORAL_SPLIT_PATH,
        "immutable upstream temporal examples",
    ),

    (
        "Phase 2",
        "T60 holdout pair manifest",
        T60_HOLDOUT_MANIFEST_PATH,
        "immutable upstream T60 holdout identities",
    ),


    # -------------------------------------------------------------------------
    # Frozen Phase-3 graph
    # -------------------------------------------------------------------------

    (
        "Phase 3",
        "Node index",
        NODE_INDEX_PATH,
        "global role-node index",
    ),

    (
        "Phase 3",
        "Relation index",
        RELATION_INDEX_PATH,
        "12 typed structural relation channels",
    ),

    (
        "Phase 3",
        "Edge index",
        EDGE_INDEX_PATH,
        "frozen directed structural graph",
    ),

    (
        "Phase 3",
        "Edge type",
        EDGE_TYPE_PATH,
        "typed structural edge IDs",
    ),

    (
        "Phase 3",
        "Graph variant masks",
        GRAPH_VARIANT_MASKS_PATH,
        "core/founder/acquisition graph masks",
    ),


    # -------------------------------------------------------------------------
    # Static Phase-4 feature inputs
    # -------------------------------------------------------------------------

    (
        "Phase 4",
        "Combined Doc2Vec vectors",
        DOC2VEC_ALL_PATH,
        "static 32-D description text input",
    ),

    (
        "Phase 4",
        "Doc2Vec manifest",
        DOC2VEC_MANIFEST_PATH,
        "Doc2Vec row identity / provenance",
    ),

    (
        "Phase 4",
        "Category label matrix",
        LABEL_MATRIX_PATH,
        "static sparse 802-D category input",
    ),

    (
        "Phase 4",
        "Category label manifest",
        LABEL_MANIFEST_PATH,
        "category row identity / provenance",
    ),


    # -------------------------------------------------------------------------
    # Trend runtime
    # -------------------------------------------------------------------------

    (
        "Phase 4",
        "Trend period pointer",
        TREND_PERIOD_PTR_PATH,
        "CSR Investor-period pointer",
    ),

    (
        "Phase 4",
        "Trend Startup indices",
        TREND_STARTUP_INDICES_PATH,
        "historical Startup membership identities",
    ),

    (
        "Phase 4",
        "Trend period counts",
        TREND_PERIOD_COUNTS_PATH,
        "historical membership counts by Investor-period",
    ),


    # -------------------------------------------------------------------------
    # Final contracts
    # -------------------------------------------------------------------------

    (
        "Phase 4",
        "Static input contract",
        STATIC_INPUT_CONTRACT_PATH,
        "authoritative Phase-4 static-input contract",
    ),

    (
        "Phase 4",
        "Full topology contract",
        FULL_MODEL_TOPOLOGY_CONTRACT_PATH,
        "authoritative frozen model architecture",
    ),

    (
        "Phase 4",
        "End-to-end integration contract",
        PHASE_4_6_2_CONTRACT_PATH,
        "verified numerical forward/BCE/backward contract",
    ),

    (
        "Phase 4",
        "Initialization contract",
        INITIALIZATION_CONTRACT_PATH,
        "authoritative initialization and neural-seed contract",
    ),

    (
        "Phase 4",
        "Initialization state fingerprint",
        INITIALIZATION_STATE_HASH_PATH,
        "canonical seed-42 initial-state SHA256",
    ),

    (
        "Phase 4",
        "Phase 4.7 closure",
        PHASE_4_7_CLOSURE_PATH,
        "final model-integrity closure",
    ),
]


artifact_records = []


for (
    source_phase,
    artifact,
    path,
    purpose,
) in artifact_specs:

    require(
        path.exists(),
        f"Authoritative artifact missing: {path}",
    )

    file_hash = sha256_file(
        path
    )

    file_bytes = int(
        path.stat().st_size
    )

    print(
        f"{artifact:<42} "
        f"{file_bytes:>12,} bytes "
        f"PASS"
    )

    artifact_records.append(
        {

            "source_phase":
                source_phase,

            "artifact":
                artifact,

            "path":
                str(
                    path
                ),

            "purpose":
                purpose,

            "sha256":
                file_hash,

            "bytes":
                file_bytes,

            "status":
                "AUTHORITATIVE",
        }
    )


artifact_df = pd.DataFrame(
    artifact_records
)


artifact_registry_path = (
    OUT_DIR
    / "phase_4_authoritative_artifact_registry.csv"
)


artifact_df.to_csv(
    artifact_registry_path,
    index=False,
)


# =============================================================================
# 13. SAVE PARAMETER SUMMARY
# =============================================================================

parameter_summary_df = pd.DataFrame(
    parameter_summary_records
)


parameter_summary_path = (
    OUT_DIR
    / "phase_4_parameter_summary.csv"
)


parameter_summary_df.to_csv(
    parameter_summary_path,
    index=False,
)


# =============================================================================
# 14. SAVE DEFERRED DECISION REGISTER
# =============================================================================

deferred_df = pd.DataFrame(
    deferred_records
)


deferred_path = (
    OUT_DIR
    / "phase_4_deferred_training_evaluation_decisions.csv"
)


deferred_df.to_csv(
    deferred_path,
    index=False,
)


# =============================================================================
# 15. FINAL PHASE-5 HANDOFF CONTRACT
# =============================================================================

banner(
    "FREEZING PHASE-5 HANDOFF CONTRACT"
)


handoff_contract = {

    "source_phase":
        "Phase 4 — Model Reconstruction",

    "status":
        "FROZEN_HANDOFF",

    "target_phase":
        "Phase 5 — Training and Evaluation",

    "model":
        {

            "name":
                "ITRS Crunchbase reproduction",

            "trainable_parameters":
                FULL_PARAMETER_TOTAL,

            "parameter_tensors":
                PARAMETER_TENSOR_COUNT,

            "role_nodes":
                NUM_NODES,

            "investors":
                NUM_INVESTORS,

            "startups":
                NUM_STARTUPS,

            "latent_dimension":
                40,

            "description_dimension":
                40,

            "trend_dimension":
                40,

            "structural_dimension":
                40,

            "investor_scoring_dimension":
                160,

            "startup_scoring_dimension":
                120,

            "pair_dimension":
                280,

            "scoring_architecture":
                [
                    280,
                    128,
                    64,
                    32,
                    16,
                    1,
                ],
        },

    "shared_latent_tables":
        {

            "investor":
                {

                    "symbol":
                        "L_o",

                    "shape":
                        [
                            NUM_INVESTORS,
                            40,
                        ],
                },

            "startup":
                {

                    "symbol":
                        "L_b",

                    "shape":
                        [
                            NUM_STARTUPS,
                            40,
                        ],
                },

            "shared_by":
                [
                    "trend",
                    "preference_propagation",
                    "scoring",
                ],

            "additional_branch_specific_latent_tables_allowed":
                False,
        },

    "description":
        {

            "doc2vec_input_dimension":
                32,

            "category_input_dimension":
                802,

            "text_branch":
                "Linear(32,20,bias=True)->ReLU",

            "label_branch":
                "Linear(802,20,bias=True)->ReLU",

            "output_dimension":
                40,
        },

    "trend":
        {

            "query":
                "[L_o || F_d,o]",

            "query_dimension":
                80,

            "historical_item":
                "[L_b || F_d,b]",

            "historical_item_dimension":
                80,

            "target_history_semantics":
                "T_h consumes T0..T(h-1)",

            "T60_history":
                "T0..T59",

            "empty_period":
                "zero80",

            "gru_layers":
                2,

            "gru_hidden_dimension":
                40,

            "trend_dimension":
                40,

            "candidate_affects_history":
                False,
        },

    "preference_propagation":
        {

            "graph_nodes":
                NUM_NODES,

            "directed_typed_edges":
                NUM_STRUCTURAL_EDGES,

            "relation_channels":
                NUM_RELATIONS,

            "layers":
                2,

            "bases":
                5,

            "dimension":
                40,

            "aggregation":
                (
                    "relation-specific incoming mean "
                    "+ separate root transform"
                ),

            "activation":
                "ReLU",

            "full_graph_forward":
                True,
        },

    "scoring":
        {

            "pair_order":
                [
                    "F_t",
                    "L_o",
                    "F_d,o",
                    "F_s,o",
                    "L_b",
                    "F_d,b",
                    "F_s,b",
                ],

            "architecture":
                [
                    280,
                    128,
                    64,
                    32,
                    16,
                    1,
                ],

            "hidden_activation":
                "ReLU",

            "training_output":
                "raw_logit",

            "probability":
                "sigmoid(logit)",

            "loss":
                "BCEWithLogitsLoss",
        },

    "initialization":
        {

            "family":
                "Kaiming",

            "function":
                "torch.nn.init.kaiming_normal_",

            "distribution":
                "normal",

            "a":
                0.0,

            "mode":
                "fan_in",

            "nonlinearity":
                "relu",

            "bias":
                "zero",

            "global_neural_seed":
                42,

            "canonical_initialization_device":
                "cpu",

            "reference_pytorch":
                "2.7.0",

            "canonical_state_sha256":
                canonical_state_sha256,
        },

    "phase_5_must_treat_as_immutable":
        [

            "Phase-2 temporal segmentation",

            "Phase-2 train/validation/test identities",

            "T60 holdout identities",

            "Phase-3 role-node identity",

            "Phase-3 structural graph",

            "12 typed relation channels",

            "Doc2Vec static vectors",

            "category static matrix",

            "trend CSR memberships",

            "description architecture",

            "trend architecture and temporal semantics",

            "R-GCN architecture",

            "scoring architecture and feature order",

            "shared L_o/L_b ownership",

            "Kaiming initialization contract",

            "global neural seed 42",
        ],

    "phase_5_is_allowed_to_resolve":
        sorted(
            DEFERRED_DECISIONS
        ),

    "runtime_reproducibility_note":
        {

            "backend_deterministic_algorithms":
                "NOT_FROZEN_IN_PHASE_4",

            "training_device":
                "NOT_FROZEN_IN_PHASE_4",

            "instruction":
                (
                    "Resolve runtime determinism separately "
                    "from the frozen canonical initial state."
                ),
        },

    "learned_feature_cache":
        {

            "F_d_persisted":
                False,

            "F_t_persisted":
                False,

            "F_s_persisted":
                False,

            "instruction":
                (
                    "Recompute from current trainable "
                    "parameters during optimization."
                ),
        },

    "negative_sampling":
        {

            "status":
                "DEFERRED_TO_PHASE_5",

            "phase_2_deferral_preserved":
                True,
        },

    "phase_4_training_performed":
        False,

    "phase_4_evaluation_performed":
        False,

    "phase_4_model_reconstruction_blockers_remaining":
        [],

    "next_action":
        (
            "Freeze the Phase-5 training/evaluation "
            "data-generation and optimization contract "
            "before training."
        ),
}


handoff_path = (
    OUT_DIR
    / "phase_4_to_phase_5_handoff_contract.json"
)


with open(
    handoff_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        handoff_contract,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 16. FINAL PHASE-4 CLOSURE MANIFEST
# =============================================================================

banner(
    "FREEZING FINAL PHASE-4 CLOSURE MANIFEST"
)


phase_4_closure = {

    "phase":
        "4",

    "name":
        "Model Reconstruction",

    "status":
        "COMPLETE_FROZEN",

    "subphases":
        {

            "4.1":
                "MODEL_CONTRACT_ESTABLISHED",

            "4.2":
                "DESCRIPTION_EXTRACTION_FROZEN",

            "4.3":
                "TREND_EXTRACTION_FROZEN",

            "4.4":
                "PREFERENCE_PROPAGATION_FROZEN",

            "4.5":
                "RECOMMENDATION_SCORING_FROZEN",

            "4.6":
                "END_TO_END_FORWARD_BCE_INTEGRATION_COMPLETE",

            "4.7":
                "MODEL_INTEGRITY_AND_INITIALIZATION_FROZEN",

            "4.8":
                "CLOSURE_AND_HANDOFF_COMPLETE",
        },

    "verified":
        {

            "static_feature_alignment":
                True,

            "shared_latent_parameter_identity":
                True,

            "description_forward":
                True,

            "trend_forward":
                True,

            "full_graph_rgcn_forward":
                True,

            "scoring_forward":
                True,

            "BCE_equivalence":
                True,

            "end_to_end_backward":
                True,

            "all_parameter_families_receive_gradient":
                True,

            "parameter_budget":
                True,

            "initialization_namespace":
                True,

            "same_seed_initialization_reproducibility":
                True,

            "different_seed_sensitivity":
                True,

            "global_cross_contract_integrity":
                True,

            "historical_contract_provenance_preserved":
                True,
        },

    "model":
        {

            "trainable_parameters":
                FULL_PARAMETER_TOTAL,

            "parameter_tensors":
                PARAMETER_TENSOR_COUNT,

            "canonical_initial_state_sha256":
                canonical_state_sha256,
        },

    "model_reconstruction_blockers_remaining":
        [],

    "deferred_to_phase_5":
        sorted(
            DEFERRED_DECISIONS
        ),

    "training_performed":
        False,

    "evaluation_performed":
        False,

    "phase_2_reopened":
        False,

    "phase_3_reopened":
        False,

    "final_handoff":
        str(
            handoff_path
        ),

    "next_phase":
        {

            "phase":
                "5",

            "name":
                "Training and Evaluation",

            "first_requirement":
                (
                    "Freeze training/evaluation "
                    "contract before optimization."
                ),
        },
}


phase_4_closure_path = (
    OUT_DIR
    / "phase_4_closure_manifest.json"
)


with open(
    phase_4_closure_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        phase_4_closure,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# 17. SAVE CONTRACT STATUS AUDIT
# =============================================================================

status_df = pd.DataFrame(
    status_records
)


status_path = (
    OUT_DIR
    / "phase_4_final_contract_status_audit.csv"
)


status_df.to_csv(
    status_path,
    index=False,
)


# =============================================================================
# 18. REPRODUCTION LOG ENTRY
# =============================================================================

banner(
    "WRITING PHASE-4 REPRODUCTION LOG ENTRY"
)


reproduction_log = f"""# Phase 4 — Model Reconstruction

Status: **COMPLETE / FROZEN**

## Final model

The reconstructed ITRS model contains **{FULL_PARAMETER_TOTAL:,} trainable parameters**
across **{PARAMETER_TENSOR_COUNT} trainable tensors**.

### Shared latent tables

- Investor latent table `L_o`: [{NUM_INVESTORS}, 40]
- Startup latent table `L_b`: [{NUM_STARTUPS}, 40]
- The same two latent tables are shared across trend extraction,
  preference propagation, and scoring.

### Description extraction

- Static Doc2Vec input: 32 dimensions
- Static Crunchbase category input: 802 dimensions
- Text projection: 32 -> 20 -> ReLU
- Category projection: 802 -> 20 -> ReLU
- Final description representation: 40 dimensions

### Trend extraction

- Investor query: [L_o || F_d,o] = 80
- Historical Startup item: [L_b || F_d,b] = 80
- Target T_h consumes T0..T(h-1)
- T60 therefore consumes T0..T59
- Bilinear attention
- GRU input = 80
- GRU hidden = 40
- GRU layers = 2
- Final trend representation = 40

### Preference propagation

- Full graph nodes: {NUM_NODES:,}
- Directed structural edges: {NUM_STRUCTURAL_EDGES:,}
- Typed relation channels: {NUM_RELATIONS}
- R-GCN layers: 2
- Bases per layer: 5
- Structural dimension: 40
- Relation-specific incoming mean normalization
- Separate neural root/self transform
- ReLU after each layer

### Recommendation scoring

- Investor representation: 160 dimensions
- Startup representation: 120 dimensions
- Pair representation: 280 dimensions
- Scoring MLP: 280 -> 128 -> 64 -> 32 -> 16 -> 1
- Hidden activation: ReLU
- Training output: raw logit
- Probability: sigmoid(logit)
- Objective: BCEWithLogitsLoss

### Initialization

- Paper family: Kaiming
- Frozen reproduction variant:
  `torch.nn.init.kaiming_normal_(a=0, mode="fan_in", nonlinearity="relu")`
- Bias initialization: zero
- Global neural seed: 42
- Canonical initialization device: CPU
- Reference PyTorch: 2.7.0
- Canonical initial-state SHA256:
  `{canonical_state_sha256}`

## Verification completed

Phase 4 verified:

- static feature identity alignment,
- description forward,
- trend attention and GRU forward,
- full-graph R-GCN forward,
- recommendation scoring forward,
- BCE implementation,
- complete backward propagation,
- shared latent gradient paths,
- exact parameter budget,
- exact initialization namespace,
- same-seed byte reproducibility,
- different-seed sensitivity,
- final cross-contract integrity.

No optimizer step or training epoch was executed during Phase 4.

## Decisions intentionally deferred to Phase 5

"""

for decision in sorted(
    DEFERRED_DECISIONS
):

    reproduction_log += (
        f"- {decision}\n"
    )


reproduction_log += """

## Phase-5 handoff rule

Phase 5 may resolve only training/evaluation/runtime decisions that were
explicitly deferred. It must consume the frozen Phase-2 split, Phase-3 graph,
Phase-4 architecture, static inputs, trend CSR, parameter ownership, and
initialization contract without modification.
"""


reproduction_log_path = (
    OUT_DIR
    / "Phase_4_Reproduction_Log_Entry.md"
)


with open(
    reproduction_log_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        reproduction_log
    )


# =============================================================================
# 19. FINAL OUTPUT HASH MANIFEST
# =============================================================================

banner(
    "HASHING PHASE-4 CLOSURE OUTPUTS"
)


closure_outputs = [

    phase_4_closure_path,

    handoff_path,

    decision_path,

    deferred_path,

    limitation_path,

    artifact_registry_path,

    parameter_summary_path,

    status_path,

    reproduction_log_path,
]


closure_hash_records = []


for path in closure_outputs:

    closure_hash_records.append(
        {

            "artifact":
                path.stem,

            "path":
                str(
                    path
                ),

            "sha256":
                sha256_file(
                    path
                ),

            "bytes":
                int(
                    path.stat().st_size
                ),
        }
    )


closure_hash_df = pd.DataFrame(
    closure_hash_records
)


closure_hash_path = (
    OUT_DIR
    / "phase_4_closure_artifact_hashes.csv"
)


closure_hash_df.to_csv(
    closure_hash_path,
    index=False,
)


# =============================================================================
# 20. FINAL CLOSURE RECHECK
# =============================================================================

banner(
    "FINAL PHASE-4 CLOSURE RECHECK"
)


saved_closure = load_json(
    phase_4_closure_path
)

saved_handoff = load_json(
    handoff_path
)


require(
    saved_closure[
        "status"
    ]
    == "COMPLETE_FROZEN",
    "Saved Phase-4 closure status invalid.",
)


require(
    saved_handoff[
        "status"
    ]
    == "FROZEN_HANDOFF",
    "Saved Phase-5 handoff status invalid.",
)


require(
    saved_closure[
        "model"
    ][
        "trainable_parameters"
    ]
    == FULL_PARAMETER_TOTAL,
    "Saved Phase-4 parameter total invalid.",
)


require(
    saved_handoff[
        "initialization"
    ][
        "canonical_state_sha256"
    ]
    == canonical_state_sha256,
    "Saved handoff initialization fingerprint invalid.",
)


require(
    set(
        saved_closure[
            "deferred_to_phase_5"
        ]
    )
    == DEFERRED_DECISIONS,
    "Saved closure deferred-decision set changed.",
)


require(
    set(
        saved_handoff[
            "phase_5_is_allowed_to_resolve"
        ]
    )
    == DEFERRED_DECISIONS,
    "Saved handoff Phase-5 decision set changed.",
)


require(
    len(
        saved_closure[
            "model_reconstruction_blockers_remaining"
        ]
    )
    == 0,
    "Saved Phase-4 closure contains reconstruction blocker.",
)


require(
    saved_closure[
        "training_performed"
    ]
    is False,
    "Saved Phase-4 closure unexpectedly records training.",
)


print(
    "Phase-4 closure status:          PASS"
)

print(
    "Phase-5 handoff status:          PASS"
)

print(
    "Parameter total:                 PASS"
)

print(
    "Initialization fingerprint:      PASS"
)

print(
    "Deferred-decision set:           PASS"
)

print(
    "Model reconstruction blockers:   0 PASS"
)

print(
    "Training boundary:               PASS"
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

banner(
    "PHASE 4.8 FINAL SUMMARY"
)


print(
    "Phase 4:"
)

print(
    "  name                           Model Reconstruction"
)

print(
    "  status                         COMPLETE / FROZEN"
)


print()
print(
    "Final ITRS model:"
)

print(
    f"  role nodes                     "
    f"{NUM_NODES:,}"
)

print(
    f"  Investors                      "
    f"{NUM_INVESTORS:,}"
)

print(
    f"  Startups                       "
    f"{NUM_STARTUPS:,}"
)

print(
    f"  structural edges               "
    f"{NUM_STRUCTURAL_EDGES:,}"
)

print(
    f"  relation channels              "
    f"{NUM_RELATIONS}"
)

print(
    f"  trainable parameter tensors    "
    f"{PARAMETER_TENSOR_COUNT}"
)

print(
    f"  trainable parameters           "
    f"{FULL_PARAMETER_TOTAL:,}"
)


print()
print(
    "Representations:"
)

print(
    "  latent                         40"
)

print(
    "  description                    40"
)

print(
    "  trend                          40"
)

print(
    "  structural                     40"
)

print(
    "  Investor scoring               160"
)

print(
    "  Startup scoring                120"
)

print(
    "  pair                           280"
)


print()
print(
    "Scoring architecture:"
)

print(
    "  280 -> 128 -> 64 -> 32 -> 16 -> 1"
)


print()
print(
    "Canonical initialization:"
)

print(
    "  Kaiming normal"
)

print(
    "  a = 0"
)

print(
    "  fan_in"
)

print(
    "  ReLU"
)

print(
    "  biases = zero"
)

print(
    "  global neural seed = 42"
)

print(
    "  canonical device = CPU"
)

print(
    "  reference PyTorch = 2.7.0"
)


print()
print(
    "Canonical initial-state SHA256:"
)

print(
    f"  {canonical_state_sha256}"
)


print()
print(
    "Phase-4 reconstruction blockers:"
)

print(
    "  NONE"
)


print()
print(
    "Deferred to Phase 5:"
)


for decision in sorted(
    DEFERRED_DECISIONS
):

    print(
        f"  - {decision}"
    )


print()
print(
    "Training performed:              NO"
)

print(
    "Evaluation performed:            NO"
)

print(
    "Phase 2 reopened:                NO"
)

print(
    "Phase 3 reopened:                NO"
)


print()
print(
    "Closure outputs:"
)


for path in [

    phase_4_closure_path,

    handoff_path,

    decision_path,

    deferred_path,

    limitation_path,

    artifact_registry_path,

    parameter_summary_path,

    status_path,

    reproduction_log_path,

    closure_hash_path,
]:

    print(
        f"  {path}"
    )


print()
print("=" * 120)

print(
    "PHASE 4 STATUS: COMPLETE / FROZEN — "
    "MODEL RECONSTRUCTION CLOSED"
)

print("=" * 120)


print()
print(
    "NEXT:"
)

print(
    "PHASE 5 — TRAINING AND EVALUATION"
)

print()
print(
    "First Phase-5 requirement:"
)

print(
    "  Freeze the training/evaluation contract "
    "before any optimizer step."
)