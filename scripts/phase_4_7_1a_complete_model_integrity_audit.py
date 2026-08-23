from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 4.7.1a — COMPLETE MODEL INTEGRITY AUDIT
#
# PURPOSE
# -------
# Verify that every frozen component of the reconstructed ITRS model remains
# mutually consistent after the successful Phase-4.6 end-to-end forward /
# BCE / backward audit.
#
# THIS IS A CROSS-CONTRACT AUDIT.
#
# It does NOT:
#
#   - train,
#   - initialize neural parameters,
#   - run another full neural forward,
#   - generate negatives,
#   - modify Phase 2,
#   - modify Phase 3,
#   - change any frozen Phase-4 decision.
#
#
# MAIN QUESTIONS
# --------------
#
# 1. Are all frozen Phase-4 contracts still frozen?
#
# 2. Do the authoritative static files still match their frozen hashes?
#
# 3. Does the integrated parameter namespace still equal 19,217,929?
#
# 4. Is the trend CSR internally exact?
#
# 5. Is the structural graph still exactly the frozen Phase-3 graph?
#
# 6. Did the successful 4.6.2 forward/backward audit exercise every expected
#    parameter family?
#
# 7. Is the selected integration event still a real T60 validation positive?
#
# 8. Which still-open decisions actually block Phase-4 closure, and which
#    belong to later training/evaluation phases?
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
# OUTPUTS
# =============================================================================

OUT_DIR = (
    PHASE_4_ROOT
    / "model_integrity_audit"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# PHASE-2 INPUT
# =============================================================================

TEMPORAL_SPLIT_PATH = (
    PHASE_2_ROOT
    / "model_ready"
    / "interactions_itrs_temporal_split.parquet"
)


# =============================================================================
# PHASE-3 INPUTS
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


# =============================================================================
# PHASE-4 CONTRACTS
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
# PHASE-4 STATIC ARTIFACTS
# =============================================================================

DOC2VEC_ALL_PATH = (
    PHASE_4_ROOT
    / "doc2vec"
    / "vectors"
    / "doc2vec_vectors_all.npy"
)

LABEL_MATRIX_PATH = (
    PHASE_4_ROOT
    / "description_labels"
    / "description_label_multihot.npz"
)

STATIC_HASH_AUDIT_PATH = (
    PHASE_4_ROOT
    / "full_model_contract"
    / "full_model_static_artifact_hashes.csv"
)

STATIC_ALIGNMENT_AUDIT_PATH = (
    PHASE_4_ROOT
    / "full_model_contract"
    / "full_model_static_input_alignment_audit.csv"
)


# =============================================================================
# PHASE-4 PARAMETER ARTIFACTS
# =============================================================================

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
# PHASE-4.6 OUTPUTS
# =============================================================================

SELECTED_VALIDATION_PAIR_PATH = (
    PHASE_4_ROOT
    / "full_model_forward_audit"
    / "phase_4_6_2_selected_validation_pair.json"
)

FORWARD_SHAPE_AUDIT_PATH = (
    PHASE_4_ROOT
    / "full_model_forward_audit"
    / "phase_4_6_2_forward_shape_audit.csv"
)

ATTENTION_AUDIT_PATH = (
    PHASE_4_ROOT
    / "full_model_forward_audit"
    / "phase_4_6_2_attention_audit.csv"
)

GRADIENT_AUDIT_PATH = (
    PHASE_4_ROOT
    / "full_model_forward_audit"
    / "phase_4_6_2_gradient_audit.csv"
)


# =============================================================================
# FROZEN POPULATION
# =============================================================================

NUM_INVESTORS = 165_975
NUM_STARTUPS = 311_589
NUM_NODES = 477_564

NUM_EDGES = 158_818
NUM_RELATIONS = 12

NUM_HISTORY_PERIODS = 60


# =============================================================================
# FROZEN TREND COUNTS
# =============================================================================

EXPECTED_TREND_POINTER_LENGTH = 9_958_501
EXPECTED_TREND_COUNT_LENGTH = 9_958_500
EXPECTED_TREND_MEMBERSHIPS = 1_145_364

EXPECTED_ACTIVE_INVESTOR_PERIODS = 554_171
EXPECTED_EMPTY_INVESTOR_PERIODS = 9_404_329


# =============================================================================
# FROZEN PARAMETER COUNTS
# =============================================================================

EXPECTED_PARAMETER_TOTAL = 19_217_929

EXPECTED_COMPONENT_PARAMETERS = {

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
# FROZEN TYPED RELATION COUNTS
# =============================================================================

EXPECTED_RELATION_COUNTS = {

    0:
        1_565,

    1:
        1_271,

    2:
        1_565,

    3:
        11_427,

    4:
        2_460,

    5:
        7_768,

    6:
        11_427,

    7:
        17_737,

    8:
        1_271,

    9:
        17_737,

    10:
        7_768,

    11:
        76_822,
}


# =============================================================================
# EXPECTED OPEN DECISIONS
# =============================================================================

EXPECTED_OPEN_DECISIONS = {

    "exact global Kaiming initialization variant",

    "global neural seed policy",

    "training negative:positive ratio",

    "training negative candidate eligibility",

    "training historical negative exclusion",

    "training epoch count",

    "early stopping",

    "weight decay",

    "evaluation candidate-generation runtime contract",
}


PHASE_4_CLOSURE_BLOCKERS = {

    "exact global Kaiming initialization variant",

    "global neural seed policy",
}


DEFER_TO_TRAINING_EVALUATION = (

    EXPECTED_OPEN_DECISIONS
    - PHASE_4_CLOSURE_BLOCKERS
)


# =============================================================================
# EXPECTED PARAMETER NAMESPACE
# =============================================================================

EXPECTED_PARAMETER_NAMES = {

    "investor_embedding.weight",

    "startup_embedding.weight",

    "description_encoder.text_projection.weight",
    "description_encoder.text_projection.bias",
    "description_encoder.label_projection.weight",
    "description_encoder.label_projection.bias",

    "trend_extractor.attention_weight",

    "trend_extractor.gru.weight_ih_l0",
    "trend_extractor.gru.weight_hh_l0",
    "trend_extractor.gru.bias_ih_l0",
    "trend_extractor.gru.bias_hh_l0",

    "trend_extractor.gru.weight_ih_l1",
    "trend_extractor.gru.weight_hh_l1",
    "trend_extractor.gru.bias_ih_l1",
    "trend_extractor.gru.bias_hh_l1",

    "trend_extractor.output_projection.weight",

    "preference_propagation.layer_1.bases",
    "preference_propagation.layer_1.coefficients",
    "preference_propagation.layer_1.root_weight",

    "preference_propagation.layer_2.bases",
    "preference_propagation.layer_2.coefficients",
    "preference_propagation.layer_2.root_weight",

    "scoring_mlp.hidden_1.weight",
    "scoring_mlp.hidden_1.bias",

    "scoring_mlp.hidden_2.weight",
    "scoring_mlp.hidden_2.bias",

    "scoring_mlp.hidden_3.weight",
    "scoring_mlp.hidden_3.bias",

    "scoring_mlp.hidden_4.weight",
    "scoring_mlp.hidden_4.bias",

    "scoring_mlp.output.weight",
    "scoring_mlp.output.bias",
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


def boolean_series(series):

    if series.dtype == bool:

        return series

    return (
        series
        .astype(str)
        .str.casefold()
        .map(
            {
                "true":
                    True,

                "false":
                    False,
            }
        )
    )


# =============================================================================
# START
# =============================================================================

banner(
    "PHASE 4.7.1a — "
    "COMPLETE MODEL INTEGRITY AUDIT"
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
# 2. REQUIRED ARTIFACT EXISTENCE
# =============================================================================

banner(
    "REQUIRED ARTIFACT EXISTENCE"
)


required_paths = [

    TEMPORAL_SPLIT_PATH,

    NODE_INDEX_PATH,
    RELATION_INDEX_PATH,
    EDGE_INDEX_PATH,
    EDGE_TYPE_PATH,

    DOC2VEC_ALL_PATH,
    LABEL_MATRIX_PATH,

    STATIC_HASH_AUDIT_PATH,
    STATIC_ALIGNMENT_AUDIT_PATH,

    PARAMETER_NAMESPACE_PATH,
    COMPONENT_PARAMETER_AUDIT_PATH,

    TREND_PERIOD_PTR_PATH,
    TREND_STARTUP_INDICES_PATH,
    TREND_PERIOD_COUNTS_PATH,

    SELECTED_VALIDATION_PAIR_PATH,
    FORWARD_SHAPE_AUDIT_PATH,
    ATTENTION_AUDIT_PATH,
    GRADIENT_AUDIT_PATH,
]


for path in required_paths:

    exists = path.exists()

    print(
        f"{str(path):<110} "
        f"{'FOUND' if exists else 'MISSING'}"
    )

    require(
        exists,
        f"Missing required artifact: {path}",
    )


# =============================================================================
# 3. FROZEN CONTRACT STATUS AUDIT
# =============================================================================

banner(
    "FROZEN CONTRACT STATUS AUDIT"
)


contract_expectations = [

    (
        "Doc2Vec",
        DOC2VEC_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Description input",
        DESCRIPTION_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Description neural",
        DESCRIPTION_NEURAL_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Trend history",
        TREND_HISTORY_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Trend neural",
        TREND_NEURAL_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Trend runtime",
        TREND_RUNTIME_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "R-GCN neural",
        RGCN_NEURAL_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "R-GCN integration",
        RGCN_INTEGRATION_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Phase 4.4 closure",
        PHASE_4_4_CLOSURE_PATH,
        "COMPLETE",
    ),

    (
        "Scoring input",
        SCORING_INPUT_CONTRACT_PATH,
        "FROZEN_INPUT_CONTRACT",
    ),

    (
        "Scoring neural",
        SCORING_NEURAL_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Scoring forward",
        SCORING_FORWARD_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Phase 4.5 closure",
        PHASE_4_5_CLOSURE_PATH,
        "COMPLETE",
    ),

    (
        "Full static input",
        STATIC_INPUT_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Full topology",
        FULL_MODEL_TOPOLOGY_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Phase 4.6.2 integration",
        PHASE_4_6_2_CONTRACT_PATH,
        "FROZEN",
    ),

    (
        "Phase 4.6 closure",
        PHASE_4_6_CLOSURE_PATH,
        "COMPLETE",
    ),
]


contract_records = []


for (
    name,
    path,
    expected_status,
) in contract_expectations:

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
        f"{name:<28} "
        f"expected={expected_status:<24} "
        f"actual={str(actual_status):<24} "
        f"{'PASS' if exact else 'FAIL'}"
    )

    require(
        exact,
        (
            "Frozen contract status mismatch: "
            f"{name}"
        ),
    )

    contract_records.append(
        {
            "contract":
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
# 4. FULL MODEL CONTRACT CROSS-CHECK
# =============================================================================

banner(
    "FULL MODEL CONTRACT CROSS-CHECK"
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

phase_4_6_closure = load_json(
    PHASE_4_6_CLOSURE_PATH
)


require(
    topology_contract[
        "parameter_budget"
    ][
        "total"
    ]
    == EXPECTED_PARAMETER_TOTAL,
    "Frozen full parameter budget changed.",
)


require(
    topology_contract[
        "latent_embeddings"
    ][
        "number_of_tables"
    ]
    == 2,
    "Frozen latent-table count changed.",
)


require(
    topology_contract[
        "scoring"
    ][
        "pair_dimension"
    ]
    == 280,
    "Frozen pair dimension changed.",
)


require(
    topology_contract[
        "scoring"
    ][
        "investor_representation"
    ][
        "dimension"
    ]
    == 160,
    "Frozen Investor scoring dimension changed.",
)


require(
    topology_contract[
        "scoring"
    ][
        "startup_representation"
    ][
        "dimension"
    ]
    == 120,
    "Frozen Startup scoring dimension changed.",
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
    integration_contract[
        "audit_case"
    ][
        "negative_generated"
    ]
    is False,
    "Phase 4.6 unexpectedly generated a negative.",
)


require(
    integration_contract[
        "audit_case"
    ][
        "test_data_used"
    ]
    is False,
    "Phase 4.6 unexpectedly used test data.",
)


print(
    "Parameter total:               "
    "19,217,929 PASS"
)

print(
    "Latent embedding tables:       "
    "2 PASS"
)

print(
    "Investor scoring dimension:    "
    "160 PASS"
)

print(
    "Startup scoring dimension:     "
    "120 PASS"
)

print(
    "Pair scoring dimension:        "
    "280 PASS"
)

print(
    "Phase-4.6 training performed:  "
    "NO PASS"
)

print(
    "Phase-4.6 negative generated:  "
    "NO PASS"
)

print(
    "Phase-4.6 test data used:      "
    "NO PASS"
)


# =============================================================================
# 5. PARAMETER NAMESPACE INTEGRITY
# =============================================================================

banner(
    "PARAMETER NAMESPACE INTEGRITY"
)


parameter_namespace = pd.read_csv(
    PARAMETER_NAMESPACE_PATH
)


require(
    {
        "parameter",
        "numel",
    }
    .issubset(
        parameter_namespace.columns
    ),
    "Parameter namespace CSV schema changed.",
)


actual_parameter_names = set(
    parameter_namespace[
        "parameter"
    ]
    .astype(str)
)


missing_parameters = (
    EXPECTED_PARAMETER_NAMES
    - actual_parameter_names
)

unexpected_parameters = (
    actual_parameter_names
    - EXPECTED_PARAMETER_NAMES
)


print(
    f"Expected parameter tensors: "
    f"{len(EXPECTED_PARAMETER_NAMES)}"
)

print(
    f"Actual parameter tensors:   "
    f"{len(actual_parameter_names)}"
)

print(
    f"Missing parameters:         "
    f"{len(missing_parameters)}"
)

print(
    f"Unexpected parameters:      "
    f"{len(unexpected_parameters)}"
)


require(
    len(
        missing_parameters
    )
    == 0,
    (
        "Missing integrated parameters: "
        f"{sorted(missing_parameters)}"
    ),
)


require(
    len(
        unexpected_parameters
    )
    == 0,
    (
        "Unexpected integrated parameters: "
        f"{sorted(unexpected_parameters)}"
    ),
)


namespace_total = int(
    parameter_namespace[
        "numel"
    ].sum()
)


print(
    f"Namespace parameter total:  "
    f"{namespace_total:,}"
)


require(
    namespace_total
    == EXPECTED_PARAMETER_TOTAL,
    "Parameter namespace total changed.",
)


# =============================================================================
# 6. COMPONENT PARAMETER BUDGET
# =============================================================================

banner(
    "COMPONENT PARAMETER BUDGET"
)


component_parameter_audit = pd.read_csv(
    COMPONENT_PARAMETER_AUDIT_PATH
)


require(
    {
        "component",
        "actual_parameters",
    }
    .issubset(
        component_parameter_audit.columns
    ),
    "Component parameter audit schema changed.",
)


for (
    component,
    expected_parameters,
) in EXPECTED_COMPONENT_PARAMETERS.items():

    rows = component_parameter_audit[
        component_parameter_audit[
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
            "Could not uniquely resolve "
            f"component: {component}"
        ),
    )

    actual_parameters = int(
        rows.iloc[
            0
        ][
            "actual_parameters"
        ]
    )

    exact = (
        actual_parameters
        == expected_parameters
    )

    print(
        f"{component:<34} "
        f"{actual_parameters:>12,} "
        f"{'PASS' if exact else 'FAIL'}"
    )

    require(
        exact,
        (
            "Component parameter count changed: "
            f"{component}"
        ),
    )


# =============================================================================
# 7. STATIC HASH RECHECK
# =============================================================================

banner(
    "AUTHORITATIVE STATIC ARTIFACT HASH RECHECK"
)


static_hash_audit = pd.read_csv(
    STATIC_HASH_AUDIT_PATH
)


require(
    {
        "artifact",
        "path",
        "sha256",
        "verified",
    }
    .issubset(
        static_hash_audit.columns
    ),
    "Static hash audit schema changed.",
)


static_hash_records = []


for _, row in static_hash_audit.iterrows():

    artifact = str(
        row[
            "artifact"
        ]
    )

    path = Path(
        row[
            "path"
        ]
    )

    frozen_hash = str(
        row[
            "sha256"
        ]
    ).strip().lower()


    require(
        path.exists(),
        (
            "Frozen static artifact missing: "
            f"{path}"
        ),
    )


    current_hash = sha256_file(
        path
    )


    exact = (
        current_hash
        == frozen_hash
    )


    print(
        f"{artifact:<32} "
        f"{'PASS' if exact else 'FAIL'}"
    )


    require(
        exact,
        (
            "Static artifact hash changed: "
            f"{artifact}"
        ),
    )


    static_hash_records.append(
        {
            "artifact":
                artifact,

            "path":
                str(
                    path
                ),

            "frozen_sha256":
                frozen_hash,

            "current_sha256":
                current_hash,

            "status":
                "PASS",
        }
    )


# =============================================================================
# 8. STATIC ALIGNMENT AUDIT RECHECK
# =============================================================================

banner(
    "STATIC ALIGNMENT AUDIT RECHECK"
)


static_alignment = pd.read_csv(
    STATIC_ALIGNMENT_AUDIT_PATH
)


require(
    {
        "check",
        "status",
    }
    .issubset(
        static_alignment.columns
    ),
    "Static alignment audit schema changed.",
)


failed_static_alignment = static_alignment[
    static_alignment[
        "status"
    ]
    .astype(str)
    .str.upper()
    != "PASS"
]


print(
    f"Static alignment checks: "
    f"{len(static_alignment)}"
)

print(
    f"Failed checks:            "
    f"{len(failed_static_alignment)}"
)


require(
    len(
        failed_static_alignment
    )
    == 0,
    "A frozen static alignment check no longer passes.",
)


# =============================================================================
# 9. TREND CSR INTEGRITY
# =============================================================================

banner(
    "TREND CSR INTERNAL INTEGRITY"
)


trend_ptr = np.load(
    TREND_PERIOD_PTR_PATH,
    mmap_mode="r",
)

trend_startups = np.load(
    TREND_STARTUP_INDICES_PATH,
    mmap_mode="r",
)

trend_counts = np.load(
    TREND_PERIOD_COUNTS_PATH,
    mmap_mode="r",
)


print(
    f"ptr shape:       "
    f"{trend_ptr.shape}"
)

print(
    f"startup entries: "
    f"{trend_startups.shape}"
)

print(
    f"counts shape:    "
    f"{trend_counts.shape}"
)


require(
    trend_ptr.shape
    == (
        EXPECTED_TREND_POINTER_LENGTH,
    ),
    "Trend pointer shape changed.",
)


require(
    trend_startups.shape
    == (
        EXPECTED_TREND_MEMBERSHIPS,
    ),
    "Trend membership shape changed.",
)


require(
    trend_counts.shape
    == (
        EXPECTED_TREND_COUNT_LENGTH,
    ),
    "Trend count-array shape changed.",
)


require(
    int(
        trend_ptr[
            0
        ]
    )
    == 0,
    "Trend pointer no longer starts at zero.",
)


require(
    int(
        trend_ptr[
            -1
        ]
    )
    == EXPECTED_TREND_MEMBERSHIPS,
    "Trend final pointer changed.",
)


require(
    int(
        trend_counts
        .astype(
            np.int64
        )
        .sum()
    )
    == EXPECTED_TREND_MEMBERSHIPS,
    "Trend counts no longer sum to membership count.",
)


# -----------------------------------------------------------------------------
# Chunked pointer/count equivalence.
#
# Avoid creating one unnecessary giant diff array.
# -----------------------------------------------------------------------------

chunk_size = 1_000_000

pointer_count_exact = True


for start in range(
    0,
    len(
        trend_counts
    ),
    chunk_size,
):

    end = min(
        start
        + chunk_size,
        len(
            trend_counts
        ),
    )


    ptr_diff = (
        trend_ptr[
            start + 1:
            end + 1
        ]
        -
        trend_ptr[
            start:
            end
        ]
    )


    expected_chunk = (
        trend_counts[
            start:end
        ]
        .astype(
            np.int64,
            copy=False,
        )
    )


    if not np.array_equal(
        ptr_diff,
        expected_chunk,
    ):

        pointer_count_exact = False
        break


print(
    f"ptr differences == counts: "
    f"{pointer_count_exact}"
)


require(
    pointer_count_exact,
    "Trend CSR pointer/count equivalence failed.",
)


active_investor_periods = int(
    np.count_nonzero(
        trend_counts
    )
)

empty_investor_periods = int(
    len(
        trend_counts
    )
    - active_investor_periods
)


print(
    f"Active investor-periods: "
    f"{active_investor_periods:,}"
)

print(
    f"Empty investor-periods:  "
    f"{empty_investor_periods:,}"
)


require(
    active_investor_periods
    == EXPECTED_ACTIVE_INVESTOR_PERIODS,
    "Active investor-period count changed.",
)


require(
    empty_investor_periods
    == EXPECTED_EMPTY_INVESTOR_PERIODS,
    "Empty investor-period count changed.",
)


startup_index_range_valid = bool(
    np.all(
        (
            trend_startups
            >= NUM_INVESTORS
        )
        &
        (
            trend_startups
            < NUM_NODES
        )
    )
)


print(
    "All trend memberships use global "
    f"Startup indices: {startup_index_range_valid}"
)


require(
    startup_index_range_valid,
    "Trend Startup index-space contract changed.",
)


# =============================================================================
# 10. GRAPH INTEGRITY
# =============================================================================

banner(
    "FROZEN STRUCTURAL GRAPH INTEGRITY"
)


edge_index = np.load(
    EDGE_INDEX_PATH,
    mmap_mode="r",
)

edge_type = np.load(
    EDGE_TYPE_PATH,
    mmap_mode="r",
)

relation_index = pd.read_csv(
    RELATION_INDEX_PATH
)


require(
    edge_index.shape
    == (
        2,
        NUM_EDGES,
    ),
    "edge_index shape changed.",
)


require(
    edge_type.shape
    == (
        NUM_EDGES,
    ),
    "edge_type shape changed.",
)


require(
    len(
        relation_index
    )
    == NUM_RELATIONS,
    "Typed relation vocabulary size changed.",
)


require(
    int(
        edge_index.min()
    )
    >= 0,
    "Graph contains negative node index.",
)


require(
    int(
        edge_index.max()
    )
    < NUM_NODES,
    "Graph contains out-of-range node index.",
)


require(
    int(
        edge_type.min()
    )
    == 0,
    "Minimum relation ID changed.",
)


require(
    int(
        edge_type.max()
    )
    == NUM_RELATIONS - 1,
    "Maximum relation ID changed.",
)


relation_counts = np.bincount(
    np.asarray(
        edge_type,
        dtype=np.int64,
    ),
    minlength=NUM_RELATIONS,
)


graph_integrity_records = []


for relation_id in range(
    NUM_RELATIONS
):

    actual_count = int(
        relation_counts[
            relation_id
        ]
    )

    expected_count = (
        EXPECTED_RELATION_COUNTS[
            relation_id
        ]
    )

    exact = (
        actual_count
        == expected_count
    )

    print(
        f"relation {relation_id:>2}: "
        f"{actual_count:>7,} "
        f"{'PASS' if exact else 'FAIL'}"
    )

    require(
        exact,
        (
            "Typed relation count changed for "
            f"relation {relation_id}."
        ),
    )

    graph_integrity_records.append(
        {
            "check":
                f"relation_{relation_id}_edge_count",

            "expected":
                expected_count,

            "actual":
                actual_count,

            "status":
                "PASS",
        }
    )


self_loop_count = int(
    np.count_nonzero(
        edge_index[
            0
        ]
        ==
        edge_index[
            1
        ]
    )
)


print()
print(
    f"Explicit structural self-loops: "
    f"{self_loop_count:,}"
)


require(
    self_loop_count
    == 0,
    "Structural self-loop count changed.",
)


typed_edges = np.column_stack(
    [
        np.asarray(
            edge_index[
                0
            ],
            dtype=np.int64,
        ),

        np.asarray(
            edge_index[
                1
            ],
            dtype=np.int64,
        ),

        np.asarray(
            edge_type,
            dtype=np.int64,
        ),
    ]
)


unique_typed_edges = np.unique(
    typed_edges,
    axis=0,
)


duplicate_typed_edges = (
    NUM_EDGES
    - len(
        unique_typed_edges
    )
)


print(
    f"Duplicate typed edges:          "
    f"{duplicate_typed_edges:,}"
)


require(
    duplicate_typed_edges
    == 0,
    "Duplicate typed graph edges appeared.",
)


connected_nodes = np.unique(
    np.concatenate(
        [
            np.asarray(
                edge_index[
                    0
                ],
                dtype=np.int64,
            ),

            np.asarray(
                edge_index[
                    1
                ],
                dtype=np.int64,
            ),
        ]
    )
)


print(
    f"Connected role nodes:           "
    f"{len(connected_nodes):,}"
)


require(
    len(
        connected_nodes
    )
    == 74_757,
    "Frozen connected-node count changed.",
)


# =============================================================================
# 11. PHASE-4.6 FORWARD SHAPE INTEGRITY
# =============================================================================

banner(
    "PHASE 4.6 FORWARD SHAPE AUDIT"
)


forward_shapes = pd.read_csv(
    FORWARD_SHAPE_AUDIT_PATH
)


require(
    {
        "feature",
        "status",
    }
    .issubset(
        forward_shapes.columns
    ),
    "Forward-shape audit schema changed.",
)


forward_failures = forward_shapes[
    forward_shapes[
        "status"
    ]
    .astype(str)
    .str.upper()
    != "PASS"
]


print(
    f"Recorded forward-shape checks: "
    f"{len(forward_shapes)}"
)

print(
    f"Failures:                      "
    f"{len(forward_failures)}"
)


require(
    len(
        forward_failures
    )
    == 0,
    "A Phase-4.6 forward shape check failed.",
)


# =============================================================================
# 12. PHASE-4.6 GRADIENT INTEGRITY
# =============================================================================

banner(
    "PHASE 4.6 END-TO-END GRADIENT INTEGRITY"
)


gradient_audit = pd.read_csv(
    GRADIENT_AUDIT_PATH
)


required_gradient_columns = {

    "parameter",

    "gradient_exists",

    "gradient_finite",

    "gradient_abs_sum",

    "gradient_nonzero",
}


require(
    required_gradient_columns.issubset(
        gradient_audit.columns
    ),
    "Gradient audit schema changed.",
)


gradient_parameter_names = set(
    gradient_audit[
        "parameter"
    ]
    .astype(str)
)


require(
    gradient_parameter_names
    == EXPECTED_PARAMETER_NAMES,
    (
        "Gradient-audit parameter namespace "
        "differs from integrated model namespace."
    ),
)


gradient_exists = boolean_series(
    gradient_audit[
        "gradient_exists"
    ]
)

gradient_finite = boolean_series(
    gradient_audit[
        "gradient_finite"
    ]
)

gradient_nonzero = boolean_series(
    gradient_audit[
        "gradient_nonzero"
    ]
)


require(
    gradient_exists.notna().all(),
    "Could not parse gradient_exists values.",
)


require(
    gradient_finite.notna().all(),
    "Could not parse gradient_finite values.",
)


require(
    gradient_nonzero.notna().all(),
    "Could not parse gradient_nonzero values.",
)


print(
    f"Parameters with gradients: "
    f"{int(gradient_exists.sum())}"
)

print(
    f"Finite gradients:          "
    f"{int(gradient_finite.sum())}"
)

print(
    f"Nonzero gradients:         "
    f"{int(gradient_nonzero.sum())}"
)


require(
    bool(
        gradient_exists.all()
    ),
    "At least one expected parameter lacks gradient.",
)


require(
    bool(
        gradient_finite.all()
    ),
    "At least one expected gradient is non-finite.",
)


require(
    bool(
        gradient_nonzero.all()
    ),
    "At least one expected parameter has zero gradient.",
)


# =============================================================================
# 13. ATTENTION AUDIT INTEGRITY
# =============================================================================

banner(
    "PHASE 4.6 ATTENTION INTEGRITY"
)


attention_audit = pd.read_csv(
    ATTENTION_AUDIT_PATH
)


require(
    {
        "period_index",
        "item_count",
        "attention_sum",
    }
    .issubset(
        attention_audit.columns
    ),
    "Attention audit schema changed.",
)


require(
    len(
        attention_audit
    )
    == NUM_HISTORY_PERIODS,
    "Attention audit no longer contains 60 periods.",
)


require(
    np.array_equal(
        attention_audit[
            "period_index"
        ]
        .to_numpy(
            dtype=np.int64
        ),
        np.arange(
            NUM_HISTORY_PERIODS,
            dtype=np.int64,
        ),
    ),
    "Attention period ordering changed.",
)


item_counts = (
    attention_audit[
        "item_count"
    ]
    .to_numpy(
        dtype=np.int64
    )
)


active_attention_periods = (
    attention_audit[
        attention_audit[
            "item_count"
        ]
        > 0
    ]
)


empty_attention_periods = (
    attention_audit[
        attention_audit[
            "item_count"
        ]
        == 0
    ]
)


multi_item_attention_periods = (
    attention_audit[
        attention_audit[
            "item_count"
        ]
        >= 2
    ]
)


print(
    f"History memberships:  "
    f"{int(item_counts.sum())}"
)

print(
    f"Active periods:        "
    f"{len(active_attention_periods)}"
)

print(
    f"Empty periods:         "
    f"{len(empty_attention_periods)}"
)

print(
    f"Multi-item periods:    "
    f"{len(multi_item_attention_periods)}"
)


require(
    int(
        item_counts.sum()
    )
    == 7,
    "Selected 4.6 audit history size changed.",
)


require(
    len(
        active_attention_periods
    )
    == 4,
    "Selected 4.6 active-period count changed.",
)


require(
    len(
        multi_item_attention_periods
    )
    >= 1,
    "Phase 4.6 no longer exercises multi-item attention.",
)


active_attention_sums = (
    active_attention_periods[
        "attention_sum"
    ]
    .to_numpy(
        dtype=float
    )
)


attention_normalized = bool(
    np.allclose(
        active_attention_sums,
        np.ones_like(
            active_attention_sums
        ),
        atol=1e-5,
        rtol=1e-5,
    )
)


print(
    f"Active attention sums == 1: "
    f"{attention_normalized}"
)


require(
    attention_normalized,
    "Phase 4.6 attention normalization changed.",
)


# =============================================================================
# 14. SELECTED REAL VALIDATION EVENT RECHECK
# =============================================================================

banner(
    "SELECTED REAL VALIDATION EVENT RECHECK"
)


selected_pair = load_json(
    SELECTED_VALIDATION_PAIR_PATH
)


require(
    selected_pair[
        "split"
    ]
    == "validation",
    "Selected Phase-4.6 pair is no longer validation.",
)


require(
    selected_pair[
        "segment"
    ]
    == "T60",
    "Selected Phase-4.6 pair is no longer T60.",
)


require(
    selected_pair[
        "target"
    ]
    == 1,
    "Selected Phase-4.6 target is no longer positive.",
)


require(
    selected_pair[
        "candidate_negative_generated"
    ]
    is False,
    "Selected Phase-4.6 audit generated a negative.",
)


require(
    selected_pair[
        "test_data_used"
    ]
    is False,
    "Selected Phase-4.6 audit used test data.",
)


selected_interaction_id = str(
    selected_pair[
        "interaction_id"
    ]
)


validation_match = pd.read_parquet(
    TEMPORAL_SPLIT_PATH,
    columns=[
        "interaction_id",
        "segment_number",
        "segment_label",
        "experiment_split",
    ],
    filters=[
        (
            "interaction_id",
            "==",
            selected_interaction_id,
        ),
    ],
)


print(
    f"interaction_id:"
)

print(
    f"  {selected_interaction_id}"
)

print(
    f"Rows found in frozen Phase-2 split: "
    f"{len(validation_match)}"
)


require(
    len(
        validation_match
    )
    == 1,
    (
        "Selected 4.6 interaction does not "
        "resolve uniquely in frozen Phase-2 split."
    ),
)


validation_row = validation_match.iloc[
    0
]


require(
    int(
        validation_row[
            "segment_number"
        ]
    )
    == 60,
    "Selected audit event is not T60 in Phase 2.",
)


require(
    str(
        validation_row[
            "segment_label"
        ]
    )
    == "T60",
    "Selected audit event segment label changed.",
)


require(
    str(
        validation_row[
            "experiment_split"
        ]
    )
    == "validation",
    "Selected audit event is not validation in Phase 2.",
)


print(
    "Frozen Phase-2 event identity: PASS"
)


# =============================================================================
# 15. T60 SPLIT SAFETY RECHECK
# =============================================================================

banner(
    "T60 SPLIT SAFETY RECHECK"
)


t60_validation = pd.read_parquet(
    TEMPORAL_SPLIT_PATH,
    columns=[
        "interaction_id",
    ],
    filters=[
        (
            "segment_number",
            "==",
            60,
        ),
        (
            "experiment_split",
            "==",
            "validation",
        ),
    ],
)


t60_test = pd.read_parquet(
    TEMPORAL_SPLIT_PATH,
    columns=[
        "interaction_id",
    ],
    filters=[
        (
            "segment_number",
            "==",
            60,
        ),
        (
            "experiment_split",
            "==",
            "test",
        ),
    ],
)


t60_train = pd.read_parquet(
    TEMPORAL_SPLIT_PATH,
    columns=[
        "interaction_id",
    ],
    filters=[
        (
            "segment_number",
            "==",
            60,
        ),
        (
            "experiment_split",
            "==",
            "train",
        ),
    ],
)


print(
    f"T60 validation: "
    f"{len(t60_validation):,}"
)

print(
    f"T60 test:       "
    f"{len(t60_test):,}"
)

print(
    f"T60 train:      "
    f"{len(t60_train):,}"
)


require(
    len(
        t60_validation
    )
    == 2_251,
    "Frozen T60 validation count changed.",
)


require(
    len(
        t60_test
    )
    == 20_264,
    "Frozen T60 test count changed.",
)


require(
    len(
        t60_train
    )
    == 0,
    "T60 interaction unexpectedly entered training.",
)


# =============================================================================
# 16. OPEN-DECISION INTEGRITY
# =============================================================================

banner(
    "STILL-OPEN DECISION INTEGRITY"
)


actual_open_decisions = set(
    phase_4_6_closure[
        "still_open"
    ]
)


missing_open_decisions = (
    EXPECTED_OPEN_DECISIONS
    - actual_open_decisions
)

unexpected_open_decisions = (
    actual_open_decisions
    - EXPECTED_OPEN_DECISIONS
)


print(
    f"Expected open decisions:   "
    f"{len(EXPECTED_OPEN_DECISIONS)}"
)

print(
    f"Actual open decisions:     "
    f"{len(actual_open_decisions)}"
)

print(
    f"Missing expected opens:    "
    f"{len(missing_open_decisions)}"
)

print(
    f"Unexpected open decisions: "
    f"{len(unexpected_open_decisions)}"
)


require(
    len(
        missing_open_decisions
    )
    == 0,
    (
        "A deliberately open decision was "
        "silently frozen or removed."
    ),
)


require(
    len(
        unexpected_open_decisions
    )
    == 0,
    (
        "Unexpected unresolved decision appeared."
    ),
)


decision_records = []


for decision in sorted(
    EXPECTED_OPEN_DECISIONS
):

    if decision in PHASE_4_CLOSURE_BLOCKERS:

        classification = (
            "PHASE_4_CLOSURE_BLOCKER"
        )

        next_action = (
            "Freeze in Phase 4.7.1b"
        )

    else:

        classification = (
            "DEFER_TO_TRAINING_OR_EVALUATION"
        )

        next_action = (
            "Preserve open after Phase 4"
        )


    print(
        f"{classification:<34} "
        f"{decision}"
    )


    decision_records.append(
        {
            "decision":
                decision,

            "classification":
                classification,

            "next_action":
                next_action,
        }
    )


# =============================================================================
# 17. GLOBAL INTEGRITY SUMMARY
# =============================================================================

banner(
    "GLOBAL RECONSTRUCTION INTEGRITY SUMMARY"
)


summary_checks = [

    (
        "Frozen contract statuses",
        True,
    ),

    (
        "Full model parameter total",
        namespace_total
        == EXPECTED_PARAMETER_TOTAL,
    ),

    (
        "Exact parameter namespace",
        actual_parameter_names
        == EXPECTED_PARAMETER_NAMES,
    ),

    (
        "Static artifact hashes",
        True,
    ),

    (
        "Static row alignment",
        len(
            failed_static_alignment
        )
        == 0,
    ),

    (
        "Trend CSR pointer/count exact",
        pointer_count_exact,
    ),

    (
        "Trend membership population",
        len(
            trend_startups
        )
        == EXPECTED_TREND_MEMBERSHIPS,
    ),

    (
        "Graph edge population",
        edge_index.shape[
            1
        ]
        == NUM_EDGES,
    ),

    (
        "Graph relation vocabulary",
        len(
            relation_index
        )
        == NUM_RELATIONS,
    ),

    (
        "Graph typed edge uniqueness",
        duplicate_typed_edges
        == 0,
    ),

    (
        "Graph explicit self-loops",
        self_loop_count
        == 0,
    ),

    (
        "Phase 4.6 forward shapes",
        len(
            forward_failures
        )
        == 0,
    ),

    (
        "Phase 4.6 all gradients exist",
        bool(
            gradient_exists.all()
        ),
    ),

    (
        "Phase 4.6 all gradients finite",
        bool(
            gradient_finite.all()
        ),
    ),

    (
        "Phase 4.6 all gradients nonzero",
        bool(
            gradient_nonzero.all()
        ),
    ),

    (
        "Phase 4.6 attention normalized",
        attention_normalized,
    ),

    (
        "Phase 4.6 audit event is frozen validation",
        True,
    ),

    (
        "T60 excluded from train",
        len(
            t60_train
        )
        == 0,
    ),

    (
        "Open decisions preserved",
        actual_open_decisions
        == EXPECTED_OPEN_DECISIONS,
    ),
]


integrity_records = []


for (
    check,
    passed,
) in summary_checks:

    status = (
        "PASS"
        if passed
        else "FAIL"
    )

    print(
        f"{check:<48} "
        f"{status}"
    )

    require(
        passed,
        (
            "Global model integrity failure: "
            f"{check}"
        ),
    )

    integrity_records.append(
        {
            "check":
                check,

            "status":
                status,
        }
    )


# =============================================================================
# 18. SAVE CONTRACT AUDIT
# =============================================================================

contract_df = pd.DataFrame(
    contract_records
)


contract_output_path = (
    OUT_DIR
    / "phase_4_7_1a_contract_status_audit.csv"
)


contract_df.to_csv(
    contract_output_path,
    index=False,
)


# =============================================================================
# 19. SAVE STATIC HASH RECHECK
# =============================================================================

static_hash_df = pd.DataFrame(
    static_hash_records
)


static_hash_output_path = (
    OUT_DIR
    / "phase_4_7_1a_static_hash_recheck.csv"
)


static_hash_df.to_csv(
    static_hash_output_path,
    index=False,
)


# =============================================================================
# 20. SAVE GRAPH INTEGRITY
# =============================================================================

graph_integrity_records.extend(
    [

        {
            "check":
                "self_loop_count",

            "expected":
                0,

            "actual":
                self_loop_count,

            "status":
                "PASS",
        },

        {
            "check":
                "duplicate_typed_edges",

            "expected":
                0,

            "actual":
                duplicate_typed_edges,

            "status":
                "PASS",
        },

        {
            "check":
                "connected_nodes",

            "expected":
                74_757,

            "actual":
                len(
                    connected_nodes
                ),

            "status":
                "PASS",
        },
    ]
)


graph_df = pd.DataFrame(
    graph_integrity_records
)


graph_output_path = (
    OUT_DIR
    / "phase_4_7_1a_graph_integrity_audit.csv"
)


graph_df.to_csv(
    graph_output_path,
    index=False,
)


# =============================================================================
# 21. SAVE TREND INTEGRITY
# =============================================================================

trend_records = [

    {
        "check":
            "pointer_length",

        "expected":
            EXPECTED_TREND_POINTER_LENGTH,

        "actual":
            len(
                trend_ptr
            ),

        "status":
            "PASS",
    },

    {
        "check":
            "count_length",

        "expected":
            EXPECTED_TREND_COUNT_LENGTH,

        "actual":
            len(
                trend_counts
            ),

        "status":
            "PASS",
    },

    {
        "check":
            "membership_count",

        "expected":
            EXPECTED_TREND_MEMBERSHIPS,

        "actual":
            len(
                trend_startups
            ),

        "status":
            "PASS",
    },

    {
        "check":
            "active_investor_periods",

        "expected":
            EXPECTED_ACTIVE_INVESTOR_PERIODS,

        "actual":
            active_investor_periods,

        "status":
            "PASS",
    },

    {
        "check":
            "empty_investor_periods",

        "expected":
            EXPECTED_EMPTY_INVESTOR_PERIODS,

        "actual":
            empty_investor_periods,

        "status":
            "PASS",
    },

    {
        "check":
            "pointer_count_equivalence",

        "expected":
            True,

        "actual":
            pointer_count_exact,

        "status":
            "PASS",
    },

    {
        "check":
            "startup_global_index_range",

        "expected":
            True,

        "actual":
            startup_index_range_valid,

        "status":
            "PASS",
    },
]


trend_df = pd.DataFrame(
    trend_records
)


trend_output_path = (
    OUT_DIR
    / "phase_4_7_1a_trend_runtime_integrity_audit.csv"
)


trend_df.to_csv(
    trend_output_path,
    index=False,
)


# =============================================================================
# 22. SAVE FORWARD / GRADIENT INTEGRITY
# =============================================================================

forward_gradient_records = [

    {
        "check":
            "forward_shape_failures",

        "actual":
            len(
                forward_failures
            ),

        "status":
            "PASS",
    },

    {
        "check":
            "all_parameter_gradients_exist",

        "actual":
            bool(
                gradient_exists.all()
            ),

        "status":
            "PASS",
    },

    {
        "check":
            "all_parameter_gradients_finite",

        "actual":
            bool(
                gradient_finite.all()
            ),

        "status":
            "PASS",
    },

    {
        "check":
            "all_parameter_gradients_nonzero",

        "actual":
            bool(
                gradient_nonzero.all()
            ),

        "status":
            "PASS",
    },

    {
        "check":
            "attention_normalized",

        "actual":
            attention_normalized,

        "status":
            "PASS",
    },

    {
        "check":
            "selected_event_validation",

        "actual":
            True,

        "status":
            "PASS",
    },

    {
        "check":
            "selected_event_T60",

        "actual":
            True,

        "status":
            "PASS",
    },

    {
        "check":
            "selected_event_positive",

        "actual":
            True,

        "status":
            "PASS",
    },

    {
        "check":
            "negative_generated",

        "actual":
            False,

        "status":
            "PASS",
    },

    {
        "check":
            "test_data_used",

        "actual":
            False,

        "status":
            "PASS",
    },
]


forward_gradient_df = pd.DataFrame(
    forward_gradient_records
)


forward_gradient_output_path = (
    OUT_DIR
    / "phase_4_7_1a_forward_gradient_integrity_audit.csv"
)


forward_gradient_df.to_csv(
    forward_gradient_output_path,
    index=False,
)


# =============================================================================
# 23. SAVE OPEN DECISION CLASSIFICATION
# =============================================================================

decision_df = pd.DataFrame(
    decision_records
)


decision_output_path = (
    OUT_DIR
    / "phase_4_7_1a_open_decision_classification.csv"
)


decision_df.to_csv(
    decision_output_path,
    index=False,
)


# =============================================================================
# 24. SAVE GLOBAL INTEGRITY TABLE
# =============================================================================

integrity_df = pd.DataFrame(
    integrity_records
)


integrity_output_path = (
    OUT_DIR
    / "phase_4_7_1a_global_integrity_audit.csv"
)


integrity_df.to_csv(
    integrity_output_path,
    index=False,
)


# =============================================================================
# 25. SAVE METADATA
# =============================================================================

metadata = {

    "phase":
        "4.7.1a",

    "status":
        "COMPLETE_AUDIT_ONLY",

    "component":
        "Complete cross-contract model integrity audit",

    "model":
        {

            "trainable_parameters":
                EXPECTED_PARAMETER_TOTAL,

            "latent_tables":
                2,

            "investor_representation_dim":
                160,

            "startup_representation_dim":
                120,

            "pair_representation_dim":
                280,
        },

    "trend":
        {

            "pointer_length":
                len(
                    trend_ptr
                ),

            "count_length":
                len(
                    trend_counts
                ),

            "memberships":
                len(
                    trend_startups
                ),

            "active_investor_periods":
                active_investor_periods,

            "empty_investor_periods":
                empty_investor_periods,

            "pointer_count_exact":
                pointer_count_exact,
        },

    "graph":
        {

            "nodes":
                NUM_NODES,

            "edges":
                NUM_EDGES,

            "relations":
                NUM_RELATIONS,

            "self_loops":
                self_loop_count,

            "duplicate_typed_edges":
                duplicate_typed_edges,

            "connected_nodes":
                len(
                    connected_nodes
                ),
        },

    "phase_4_6":
        {

            "forward_shape_failures":
                len(
                    forward_failures
                ),

            "all_gradients_exist":
                bool(
                    gradient_exists.all()
                ),

            "all_gradients_finite":
                bool(
                    gradient_finite.all()
                ),

            "all_gradients_nonzero":
                bool(
                    gradient_nonzero.all()
                ),

            "attention_normalized":
                attention_normalized,

            "selected_validation_positive":
                True,

            "negative_generated":
                False,

            "test_data_used":
                False,
        },

    "phase_4_closure_blockers":
        sorted(
            PHASE_4_CLOSURE_BLOCKERS
        ),

    "deferred_to_training_evaluation":
        sorted(
            DEFER_TO_TRAINING_EVALUATION
        ),

    "training_performed":
        False,

    "frozen_decisions_changed":
        False,

    "next_phase":
        {

            "phase":
                "4.7.1b",

            "name":
                (
                    "Freeze global neural "
                    "initialization and seed contract"
                ),
        },
}


metadata_path = (
    OUT_DIR
    / "phase_4_7_1a_integrity_metadata.json"
)


with open(
    metadata_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        ensure_ascii=False,
    )


# =============================================================================
# FINAL SUMMARY
# =============================================================================

banner(
    "PHASE 4.7.1a FINAL SUMMARY"
)


print(
    "Frozen contracts:"
)

print(
    f"  audited                        "
    f"{len(contract_records)}"
)

print(
    "  status failures                0"
)


print()
print(
    "Full model:"
)

print(
    f"  trainable parameters           "
    f"{namespace_total:,}"
)

print(
    f"  parameter tensors              "
    f"{len(actual_parameter_names)}"
)

print(
    "  unexpected tensors             0"
)

print(
    "  missing tensors                0"
)


print()
print(
    "Static inputs:"
)

print(
    f"  hash-rechecked artifacts       "
    f"{len(static_hash_records)}"
)

print(
    "  hash mismatches                0"
)

print(
    f"  alignment checks               "
    f"{len(static_alignment)}"
)

print(
    "  alignment failures             0"
)


print()
print(
    "Trend runtime:"
)

print(
    f"  memberships                    "
    f"{len(trend_startups):,}"
)

print(
    f"  active investor-periods        "
    f"{active_investor_periods:,}"
)

print(
    f"  empty investor-periods         "
    f"{empty_investor_periods:,}"
)

print(
    "  ptr/count equivalence          PASS"
)

print(
    "  Startup global index range     PASS"
)


print()
print(
    "Structural graph:"
)

print(
    f"  nodes                          "
    f"{NUM_NODES:,}"
)

print(
    f"  directed typed edges           "
    f"{NUM_EDGES:,}"
)

print(
    f"  relation channels              "
    f"{NUM_RELATIONS}"
)

print(
    "  relation counts                PASS"
)

print(
    "  explicit self-loops            0"
)

print(
    "  duplicate typed edges          0"
)

print(
    f"  connected nodes                "
    f"{len(connected_nodes):,}"
)


print()
print(
    "Phase 4.6 integration:"
)

print(
    "  forward shapes                 PASS"
)

print(
    "  all gradients exist            PASS"
)

print(
    "  all gradients finite           PASS"
)

print(
    "  all gradients nonzero          PASS"
)

print(
    "  attention normalized           PASS"
)

print(
    "  real validation positive       PASS"
)

print(
    "  test data used                 NO"
)

print(
    "  negative generated             NO"
)


print()
print(
    "T60 split:"
)

print(
    f"  validation                     "
    f"{len(t60_validation):,}"
)

print(
    f"  test                           "
    f"{len(t60_test):,}"
)

print(
    f"  train                          "
    f"{len(t60_train):,}"
)


print()
print(
    "Phase-4 closure blockers:"
)

for decision in sorted(
    PHASE_4_CLOSURE_BLOCKERS
):

    print(
        f"  - {decision}"
    )


print()
print(
    "Deferred to training/evaluation:"
)

for decision in sorted(
    DEFER_TO_TRAINING_EVALUATION
):

    print(
        f"  - {decision}"
    )


print()
print(
    "Training performed:              NO"
)

print(
    "Frozen architecture changed:     NO"
)

print(
    "Phase 2 reopened:                NO"
)

print(
    "Phase 3 reopened:                NO"
)


print()
print(
    "Outputs:"
)


for path in [

    contract_output_path,

    static_hash_output_path,

    graph_output_path,

    trend_output_path,

    forward_gradient_output_path,

    decision_output_path,

    integrity_output_path,

    metadata_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.7.1a STATUS: COMPLETE — "
    "CROSS-CONTRACT MODEL INTEGRITY AUDITED"
)


print()
print(
    "NEXT:"
)

print(
    "PHASE 4.7.1b — "
    "FREEZE GLOBAL NEURAL INITIALIZATION "
    "AND SEED CONTRACT"
)