from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 4.6.1a — FULL-MODEL ARTIFACT & PARAMETER INVENTORY AUDIT
#
# PURPOSE
# -------
# Before constructing the complete end-to-end ITRS model, identify and audit
# the exact persisted artifacts produced by the already-frozen Phase-4
# modules.
#
# WHY THIS STEP EXISTS
# --------------------
# We already know the semantics and dimensions of:
#
#   - Description Extraction
#   - Trend Extraction
#   - Preference Propagation
#   - Recommendation Scoring
#
# However, the end-to-end implementation must use the ACTUAL persisted
# artifact names and storage formats produced by earlier subphases.
#
# This audit therefore prevents us from inventing:
#
#   - tensor filenames,
#   - sparse-matrix filenames,
#   - path conventions,
#   - storage formats,
#   - duplicated/alternate artifacts.
#
# THIS SCRIPT DOES NOT:
#
#   - train anything,
#   - initialize neural parameters,
#   - alter Phase-2 decisions,
#   - alter Phase-3 graph artifacts,
#   - modify Phase-4 artifacts,
#   - choose negative sampling,
#   - freeze Kaiming details,
#   - instantiate the complete ITRS model.
#
# It is an inventory/audit only.
# =============================================================================


# =============================================================================
# ROOTS
# =============================================================================

PHASE_4_ROOT = Path(
    "data/experimental/phase_4"
)

PHASE_3_MODEL_READY = Path(
    "data/experimental/phase_3/"
    "model_ready"
)


# =============================================================================
# OUTPUTS
# =============================================================================

OUT_DIR = Path(
    "data/experimental/phase_4/"
    "full_model_integration_inventory"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# FROZEN POPULATION
# =============================================================================

NUM_INVESTORS = 165_975
NUM_STARTUPS = 311_589

NUM_NODES = 477_564

LATENT_DIM = 40

DOC2VEC_DIM = 32
LABEL_DIM = 802


# =============================================================================
# EXPECTED MODULE PARAMETER COUNTS
# =============================================================================

INVESTOR_LATENT_PARAMETERS = (
    NUM_INVESTORS
    * LATENT_DIM
)

STARTUP_LATENT_PARAMETERS = (
    NUM_STARTUPS
    * LATENT_DIM
)

LATENT_PARAMETERS = (
    INVESTOR_LATENT_PARAMETERS
    + STARTUP_LATENT_PARAMETERS
)

DESCRIPTION_PARAMETERS = 16_720

TREND_PARAMETERS = 32_480

RGCN_PARAMETERS = 19_320

SCORING_PARAMETERS = 46_849

EXPECTED_FULL_TRAINABLE_PARAMETERS = (
    LATENT_PARAMETERS
    + DESCRIPTION_PARAMETERS
    + TREND_PARAMETERS
    + RGCN_PARAMETERS
    + SCORING_PARAMETERS
)


assert (
    INVESTOR_LATENT_PARAMETERS
    == 6_639_000
)

assert (
    STARTUP_LATENT_PARAMETERS
    == 12_463_560
)

assert (
    LATENT_PARAMETERS
    == 19_102_560
)

assert (
    EXPECTED_FULL_TRAINABLE_PARAMETERS
    == 19_217_929
)


# =============================================================================
# KNOWN FROZEN CONTRACT PATHS
#
# These paths were created by earlier successful Phase-4 scripts.
# =============================================================================

KNOWN_CONTRACTS = {

    "description_neural_contract":
        Path(
            "data/experimental/phase_4/"
            "description_neural_contract/"
            "description_neural_contract.json"
        ),

    "trend_history_contract":
        Path(
            "data/experimental/phase_4/"
            "trend_contract/"
            "trend_history_semantics_contract.json"
        ),

    "trend_neural_contract":
        Path(
            "data/experimental/phase_4/"
            "trend_neural_contract/"
            "trend_neural_contract.json"
        ),

    "trend_runtime_contract":
        Path(
            "data/experimental/phase_4/"
            "trend_runtime/"
            "trend_runtime_contract.json"
        ),

    "rgcn_neural_contract":
        Path(
            "data/experimental/phase_4/"
            "rgcn_neural_contract/"
            "rgcn_neural_contract.json"
        ),

    "rgcn_integration_contract":
        Path(
            "data/experimental/phase_4/"
            "rgcn_integration/"
            "rgcn_integration_contract.json"
        ),

    "phase_4_4_closure":
        Path(
            "data/experimental/phase_4/"
            "rgcn_integration/"
            "phase_4_4_closure_manifest.json"
        ),

    "scoring_input_contract":
        Path(
            "data/experimental/phase_4/"
            "scoring_contract/"
            "scoring_input_contract.json"
        ),

    "scoring_neural_contract":
        Path(
            "data/experimental/phase_4/"
            "scoring_neural_contract/"
            "scoring_neural_contract.json"
        ),

    "scoring_forward_contract":
        Path(
            "data/experimental/phase_4/"
            "scoring_module/"
            "scoring_forward_contract.json"
        ),

    "phase_4_5_closure":
        Path(
            "data/experimental/phase_4/"
            "scoring_module/"
            "phase_4_5_closure_manifest.json"
        ),
}


# =============================================================================
# KNOWN TREND RUNTIME ARTIFACTS
# =============================================================================

KNOWN_TREND_RUNTIME = {

    "period_ptr":
        Path(
            "data/experimental/phase_4/"
            "trend_runtime/"
            "trend_period_ptr.npy"
        ),

    "startup_indices":
        Path(
            "data/experimental/phase_4/"
            "trend_runtime/"
            "trend_startup_node_indices.npy"
        ),

    "period_counts":
        Path(
            "data/experimental/phase_4/"
            "trend_runtime/"
            "trend_period_startup_counts.npy"
        ),
}


# =============================================================================
# KNOWN PHASE-3 MODEL-READY GRAPH
# =============================================================================

KNOWN_GRAPH_ARTIFACTS = {

    "node_index":
        PHASE_3_MODEL_READY
        / "node_index.parquet",

    "relation_index":
        PHASE_3_MODEL_READY
        / "relation_index.csv",

    "edge_index":
        PHASE_3_MODEL_READY
        / "edge_index.npy",

    "edge_type":
        PHASE_3_MODEL_READY
        / "edge_type.npy",

    "graph_masks":
        PHASE_3_MODEL_READY
        / "graph_variant_masks.npz",
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


def file_size_mib(path):

    return (
        path.stat().st_size
        / (
            1024
            * 1024
        )
    )


def recursive_collect_file_strings(
    value,
    prefix="root",
):

    """
    Find strings inside JSON structures that look like persisted artifact
    paths.

    This does NOT assume those paths are authoritative. It merely surfaces
    references already written into frozen contracts.
    """

    records = []


    if isinstance(
        value,
        dict,
    ):

        for key, child in (
            value.items()
        ):

            child_prefix = (
                f"{prefix}.{key}"
            )


            records.extend(
                recursive_collect_file_strings(
                    child,
                    child_prefix,
                )
            )


    elif isinstance(
        value,
        list,
    ):

        for index, child in enumerate(
            value
        ):

            child_prefix = (
                f"{prefix}[{index}]"
            )


            records.extend(
                recursive_collect_file_strings(
                    child,
                    child_prefix,
                )
            )


    elif isinstance(
        value,
        str,
    ):

        lowered = value.casefold()


        artifact_extensions = (
            ".npy",
            ".npz",
            ".parquet",
            ".csv",
            ".json",
            ".model",
            ".bin",
            ".pt",
            ".pth",
        )


        if any(
            extension in lowered

            for extension
            in artifact_extensions
        ):

            records.append(
                {
                    "json_key_path":
                        prefix,

                    "value":
                        value,
                }
            )


    return records


def inspect_npy(
    path,
):

    array = np.load(
        path,
        mmap_mode="r",
        allow_pickle=False,
    )


    return {

        "storage":
            "npy",

        "shape":
            str(
                tuple(
                    array.shape
                )
            ),

        "dtype":
            str(
                array.dtype
            ),

        "nnz":
            None,
    }


def inspect_npz(
    path,
):

    # -------------------------------------------------------------------------
    # First try scipy sparse format.
    # -------------------------------------------------------------------------

    try:

        from scipy import sparse


        matrix = sparse.load_npz(
            path
        )


        return {

            "storage":
                "scipy_sparse_npz",

            "shape":
                str(
                    tuple(
                        matrix.shape
                    )
                ),

            "dtype":
                str(
                    matrix.dtype
                ),

            "nnz":
                int(
                    matrix.nnz
                ),
        }


    except Exception:

        pass


    # -------------------------------------------------------------------------
    # Generic NumPy NPZ.
    # -------------------------------------------------------------------------

    archive = np.load(
        path,
        allow_pickle=False,
    )


    keys = list(
        archive.files
    )


    shape_string = None


    if "shape" in keys:

        try:

            shape_value = (
                archive[
                    "shape"
                ]
                .tolist()
            )


            shape_string = str(
                tuple(
                    int(value)
                    for value
                    in shape_value
                )
            )

        except Exception:

            shape_string = None


    return {

        "storage":
            "numpy_npz",

        "shape":
            shape_string,

        "dtype":
            None,

        "nnz":
            None,

        "archive_keys":
            str(
                keys
            ),
    }


def inspect_parquet(
    path,
):

    try:

        import pyarrow.parquet as pq


        parquet_file = pq.ParquetFile(
            path
        )


        metadata = (
            parquet_file.metadata
        )


        return {

            "storage":
                "parquet",

            "shape":
                (
                    f"({metadata.num_rows}, "
                    f"{metadata.num_columns})"
                ),

            "dtype":
                None,

            "nnz":
                None,

            "columns":
                str(
                    parquet_file
                    .schema_arrow
                    .names
                ),
        }


    except Exception as error:

        return {

            "storage":
                "parquet",

            "shape":
                None,

            "dtype":
                None,

            "nnz":
                None,

            "inspection_error":
                repr(
                    error
                ),
        }


def inspect_csv(
    path,
):

    try:

        frame = pd.read_csv(
            path,
            nrows=5,
        )


        return {

            "storage":
                "csv",

            "shape":
                "row_count_not_loaded",

            "dtype":
                None,

            "nnz":
                None,

            "columns":
                str(
                    frame.columns.tolist()
                ),
        }


    except Exception as error:

        return {

            "storage":
                "csv",

            "shape":
                None,

            "dtype":
                None,

            "nnz":
                None,

            "inspection_error":
                repr(
                    error
                ),
        }


def inspect_json(
    path,
):

    try:

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            payload = json.load(f)


        status = (
            payload.get(
                "status"
            )
            if isinstance(
                payload,
                dict,
            )
            else None
        )


        keys = (
            list(
                payload.keys()
            )
            if isinstance(
                payload,
                dict,
            )
            else None
        )


        return {

            "storage":
                "json",

            "shape":
                None,

            "dtype":
                None,

            "nnz":
                None,

            "status":
                status,

            "top_level_keys":
                str(
                    keys
                ),
        }


    except Exception as error:

        return {

            "storage":
                "json",

            "shape":
                None,

            "dtype":
                None,

            "nnz":
                None,

            "inspection_error":
                repr(
                    error
                ),
        }


def inspect_artifact(
    path,
):

    suffix = (
        path.suffix
        .casefold()
    )


    base = {

        "path":
            str(
                path
            ),

        "filename":
            path.name,

        "extension":
            suffix,

        "bytes":
            int(
                path.stat().st_size
            ),

        "mib":
            file_size_mib(
                path
            ),
    }


    try:

        if suffix == ".npy":

            details = inspect_npy(
                path
            )


        elif suffix == ".npz":

            details = inspect_npz(
                path
            )


        elif suffix == ".parquet":

            details = inspect_parquet(
                path
            )


        elif suffix == ".csv":

            details = inspect_csv(
                path
            )


        elif suffix == ".json":

            details = inspect_json(
                path
            )


        else:

            details = {

                "storage":
                    "opaque_file",

                "shape":
                    None,

                "dtype":
                    None,

                "nnz":
                    None,
            }


    except Exception as error:

        details = {

            "storage":
                "inspection_failed",

            "shape":
                None,

            "dtype":
                None,

            "nnz":
                None,

            "inspection_error":
                repr(
                    error
                ),
        }


    base.update(
        details
    )


    return base


# =============================================================================
# START
# =============================================================================

banner(
    "PHASE 4.6.1a — "
    "FULL-MODEL ARTIFACT & PARAMETER INVENTORY AUDIT"
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


try:

    import torch

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

except Exception:

    print(
        "PyTorch: NOT IMPORTABLE"
    )


# =============================================================================
# 2. ROOT EXISTENCE
# =============================================================================

banner(
    "PROJECT ARTIFACT ROOTS"
)


print(
    f"Phase-4 root:"
)

print(
    f"  {PHASE_4_ROOT}"
)

print(
    f"  exists={PHASE_4_ROOT.exists()}"
)


print()
print(
    f"Phase-3 model-ready root:"
)

print(
    f"  {PHASE_3_MODEL_READY}"
)

print(
    f"  exists={PHASE_3_MODEL_READY.exists()}"
)


require(
    PHASE_4_ROOT.exists(),
    "Phase-4 artifact directory not found.",
)


require(
    PHASE_3_MODEL_READY.exists(),
    "Phase-3 model-ready directory not found.",
)


# =============================================================================
# 3. REQUIRED FROZEN CONTRACTS
# =============================================================================

banner(
    "REQUIRED FROZEN CONTRACT INVENTORY"
)


contract_records = []


for (
    contract_name,
    path,
) in KNOWN_CONTRACTS.items():

    exists = path.exists()


    print(
        f"{contract_name:<32} "
        f"{'FOUND' if exists else 'MISSING'}"
    )


    require(
        exists,
        (
            "Required frozen contract missing: "
            f"{path}"
        ),
    )


    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        payload = json.load(f)


    status = (
        payload.get(
            "status"
        )
    )


    print(
        f"  status = {status}"
    )


    contract_records.append(
        {
            "contract":
                contract_name,

            "path":
                str(
                    path
                ),

            "status":
                status,

            "top_level_keys":
                str(
                    list(
                        payload.keys()
                    )
                ),
        }
    )


# =============================================================================
# 4. EXPECTED CONTRACT STATUS CHECKS
# =============================================================================

banner(
    "FROZEN CONTRACT STATUS CHECK"
)


expected_statuses = {

    "description_neural_contract":
        "FROZEN",

    "trend_history_contract":
        "FROZEN",

    "trend_neural_contract":
        "FROZEN",

    "trend_runtime_contract":
        "FROZEN",

    "rgcn_neural_contract":
        "FROZEN",

    "rgcn_integration_contract":
        "FROZEN",

    "phase_4_4_closure":
        "COMPLETE",

    "scoring_input_contract":
        "FROZEN_INPUT_CONTRACT",

    "scoring_neural_contract":
        "FROZEN",

    "scoring_forward_contract":
        "FROZEN",

    "phase_4_5_closure":
        "COMPLETE",
}


for record in contract_records:

    expected_status = (
        expected_statuses[
            record[
                "contract"
            ]
        ]
    )


    actual_status = (
        record[
            "status"
        ]
    )


    exact = (
        actual_status
        == expected_status
    )


    print(
        f"{record['contract']:<32} "
        f"expected={expected_status:<24} "
        f"actual={str(actual_status):<24} "
        f"{'PASS' if exact else 'FAIL'}"
    )


    require(
        exact,
        (
            "Frozen contract status changed for "
            f"{record['contract']}."
        ),
    )


# =============================================================================
# 5. KNOWN TREND RUNTIME ARRAYS
# =============================================================================

banner(
    "TREND RUNTIME ARTIFACTS"
)


expected_trend_shapes = {

    "period_ptr":
        (
            9_958_501,
        ),

    "startup_indices":
        (
            1_145_364,
        ),

    "period_counts":
        (
            9_958_500,
        ),
}


for (
    name,
    path,
) in KNOWN_TREND_RUNTIME.items():

    require(
        path.exists(),
        f"Missing trend runtime artifact: {path}",
    )


    array = np.load(
        path,
        mmap_mode="r",
    )


    print(
        f"{name:<20} "
        f"shape={array.shape} "
        f"dtype={array.dtype}"
    )


    require(
        tuple(
            array.shape
        )
        == expected_trend_shapes[
            name
        ],
        (
            "Trend runtime shape changed for "
            f"{name}."
        ),
    )


# =============================================================================
# 6. KNOWN PHASE-3 GRAPH ARTIFACTS
# =============================================================================

banner(
    "PHASE-3 GRAPH INPUTS"
)


for (
    name,
    path,
) in KNOWN_GRAPH_ARTIFACTS.items():

    exists = path.exists()


    print(
        f"{name:<20} "
        f"{'FOUND' if exists else 'MISSING'}"
    )


    require(
        exists,
        (
            "Frozen graph artifact missing: "
            f"{path}"
        ),
    )


edge_index = np.load(
    KNOWN_GRAPH_ARTIFACTS[
        "edge_index"
    ],
    mmap_mode="r",
)


edge_type = np.load(
    KNOWN_GRAPH_ARTIFACTS[
        "edge_type"
    ],
    mmap_mode="r",
)


print()
print(
    f"edge_index shape: "
    f"{edge_index.shape}"
)

print(
    f"edge_type shape:  "
    f"{edge_type.shape}"
)


require(
    edge_index.shape
    == (
        2,
        158_818,
    ),
    "Frozen graph edge_index changed.",
)


require(
    edge_type.shape
    == (
        158_818,
    ),
    "Frozen graph edge_type changed.",
)


# =============================================================================
# 7. RECURSIVE PHASE-4 ARTIFACT INVENTORY
#
# This is the key discovery step.
#
# We inspect persisted files without loading huge arrays into memory.
# =============================================================================

banner(
    "RECURSIVE PHASE-4 ARTIFACT INVENTORY"
)


supported_extensions = {

    ".npy",
    ".npz",
    ".parquet",
    ".csv",
    ".json",
    ".model",
    ".bin",
    ".pt",
    ".pth",
}


all_files = sorted(
    path

    for path
    in PHASE_4_ROOT.rglob(
        "*"
    )

    if (
        path.is_file()
        and path.suffix.casefold()
        in supported_extensions
        and OUT_DIR not in path.parents
    )
)


print(
    f"Inspectable Phase-4 artifacts: "
    f"{len(all_files)}"
)


artifact_records = []


for path in all_files:

    record = inspect_artifact(
        path
    )


    artifact_records.append(
        record
    )


    print()
    print(
        f"{record['path']}"
    )

    print(
        f"  storage: "
        f"{record.get('storage')}"
    )

    print(
        f"  size:    "
        f"{record['mib']:.3f} MiB"
    )


    if (
        record.get(
            "shape"
        )
        is not None
    ):

        print(
            f"  shape:   "
            f"{record.get('shape')}"
        )


    if (
        record.get(
            "dtype"
        )
        is not None
    ):

        print(
            f"  dtype:   "
            f"{record.get('dtype')}"
        )


    if (
        record.get(
            "nnz"
        )
        is not None
    ):

        print(
            f"  nnz:     "
            f"{record.get('nnz'):,}"
        )


    if (
        record.get(
            "status"
        )
        is not None
    ):

        print(
            f"  status:  "
            f"{record.get('status')}"
        )


    if (
        record.get(
            "inspection_error"
        )
        is not None
    ):

        print(
            f"  inspection error:"
        )

        print(
            f"    "
            f"{record.get('inspection_error')}"
        )


# =============================================================================
# 8. IDENTIFY DOC2VEC NODE-MATRIX CANDIDATES
#
# Frozen expected static input:
#
#   [477564, 32]
# =============================================================================

banner(
    "DOC2VEC NODE-MATRIX CANDIDATES"
)


doc2vec_candidates = []


for record in artifact_records:

    if (
        record.get(
            "shape"
        )
        == str(
            (
                NUM_NODES,
                DOC2VEC_DIM,
            )
        )
    ):

        doc2vec_candidates.append(
            record
        )


print(
    f"Candidates with shape "
    f"({NUM_NODES}, {DOC2VEC_DIM}): "
    f"{len(doc2vec_candidates)}"
)


for index, record in enumerate(
    doc2vec_candidates
):

    print(
        f"[{index}] "
        f"{record['path']}"
    )

    print(
        f"    storage="
        f"{record.get('storage')}"
    )

    print(
        f"    dtype="
        f"{record.get('dtype')}"
    )


require(
    len(
        doc2vec_candidates
    )
    >= 1,
    (
        "No persisted Phase-4 artifact with "
        "expected Doc2Vec node shape "
        f"({NUM_NODES}, {DOC2VEC_DIM}) "
        "was found."
    ),
)


# =============================================================================
# 9. IDENTIFY LABEL-MATRIX CANDIDATES
#
# Frozen expected static input:
#
#   [477564, 802]
#
# Expected sparse nnz from Phase 4.2.3a:
#
#   1,230,068
# =============================================================================

banner(
    "LABEL MATRIX CANDIDATES"
)


EXPECTED_LABEL_NNZ = 1_230_068


label_candidates = []


for record in artifact_records:

    if (
        record.get(
            "shape"
        )
        == str(
            (
                NUM_NODES,
                LABEL_DIM,
            )
        )
    ):

        label_candidates.append(
            record
        )


print(
    f"Candidates with shape "
    f"({NUM_NODES}, {LABEL_DIM}): "
    f"{len(label_candidates)}"
)


for index, record in enumerate(
    label_candidates
):

    print(
        f"[{index}] "
        f"{record['path']}"
    )

    print(
        f"    storage="
        f"{record.get('storage')}"
    )

    print(
        f"    dtype="
        f"{record.get('dtype')}"
    )

    print(
        f"    nnz="
        f"{record.get('nnz')}"
    )


require(
    len(
        label_candidates
    )
    >= 1,
    (
        "No persisted Phase-4 artifact with "
        "expected label matrix shape "
        f"({NUM_NODES}, {LABEL_DIM}) "
        "was found."
    ),
)


matching_label_nnz = [

    record

    for record
    in label_candidates

    if (
        record.get(
            "nnz"
        )
        == EXPECTED_LABEL_NNZ
    )
]


print()
print(
    f"Candidates matching frozen nnz "
    f"{EXPECTED_LABEL_NNZ:,}: "
    f"{len(matching_label_nnz)}"
)


for record in matching_label_nnz:

    print(
        f"  {record['path']}"
    )


require(
    len(
        matching_label_nnz
    )
    >= 1,
    (
        "No label-matrix candidate matches "
        f"frozen nnz={EXPECTED_LABEL_NNZ:,}."
    ),
)


# =============================================================================
# 10. CONTRACT-EMBEDDED ARTIFACT REFERENCES
#
# Search the frozen JSON contracts themselves for file/path references.
#
# This can help distinguish authoritative artifacts if more than one tensor
# with the same shape exists.
# =============================================================================

banner(
    "ARTIFACT REFERENCES EMBEDDED IN FROZEN CONTRACTS"
)


json_reference_records = []


for (
    contract_name,
    contract_path,
) in KNOWN_CONTRACTS.items():

    with open(
        contract_path,
        "r",
        encoding="utf-8",
    ) as f:

        payload = json.load(f)


    references = (
        recursive_collect_file_strings(
            payload
        )
    )


    if len(
        references
    ) == 0:

        continue


    print()
    print(
        f"{contract_name}:"
    )


    for reference in references:

        raw_value = (
            reference[
                "value"
            ]
        )


        candidate_path = Path(
            raw_value
        )


        exists = (
            candidate_path.exists()
        )


        print(
            f"  {reference['json_key_path']}"
        )

        print(
            f"    {raw_value}"
        )

        print(
            f"    exists={exists}"
        )


        json_reference_records.append(
            {
                "contract":
                    contract_name,

                "contract_path":
                    str(
                        contract_path
                    ),

                "json_key_path":
                    reference[
                        "json_key_path"
                    ],

                "referenced_value":
                    raw_value,

                "exists_as_path":
                    exists,
            }
        )


# =============================================================================
# 11. DESCRIPTION-RELATED ARTIFACT SHORTLIST
#
# Print only paths whose name/path indicates description/doc2vec/label.
#
# This is diagnostic and DOES NOT choose an authoritative artifact.
# =============================================================================

banner(
    "DESCRIPTION / DOC2VEC / LABEL SHORTLIST"
)


description_keywords = (
    "description",
    "doc2vec",
    "label",
)


description_shortlist = [

    record

    for record
    in artifact_records

    if any(
        keyword
        in record[
            "path"
        ].casefold()

        for keyword
        in description_keywords
    )
]


print(
    f"Description-related artifacts: "
    f"{len(description_shortlist)}"
)


for record in description_shortlist:

    print()
    print(
        f"{record['path']}"
    )

    print(
        f"  storage="
        f"{record.get('storage')}"
    )

    print(
        f"  shape="
        f"{record.get('shape')}"
    )

    print(
        f"  dtype="
        f"{record.get('dtype')}"
    )

    print(
        f"  nnz="
        f"{record.get('nnz')}"
    )


# =============================================================================
# 12. FULL TRAINABLE PARAMETER BUDGET
# =============================================================================

banner(
    "FULL ITRS TRAINABLE PARAMETER BUDGET"
)


parameter_records = [

    {
        "component":
            "Investor latent embeddings L_o",

        "parameters":
            INVESTOR_LATENT_PARAMETERS,

        "source":
            (
                f"{NUM_INVESTORS} x "
                f"{LATENT_DIM}"
            ),

        "frozen_count":
            True,
    },

    {
        "component":
            "Startup latent embeddings L_b",

        "parameters":
            STARTUP_LATENT_PARAMETERS,

        "source":
            (
                f"{NUM_STARTUPS} x "
                f"{LATENT_DIM}"
            ),

        "frozen_count":
            True,
    },

    {
        "component":
            "Description encoder",

        "parameters":
            DESCRIPTION_PARAMETERS,

        "source":
            "Phase 4.2",

        "frozen_count":
            True,
    },

    {
        "component":
            "Trend module",

        "parameters":
            TREND_PARAMETERS,

        "source":
            "Phase 4.3",

        "frozen_count":
            True,
    },

    {
        "component":
            "R-GCN",

        "parameters":
            RGCN_PARAMETERS,

        "source":
            "Phase 4.4",

        "frozen_count":
            True,
    },

    {
        "component":
            "Scoring MLP",

        "parameters":
            SCORING_PARAMETERS,

        "source":
            "Phase 4.5",

        "frozen_count":
            True,
    },
]


parameter_df = pd.DataFrame(
    parameter_records
)


for _, row in (
    parameter_df.iterrows()
):

    print(
        f"{row['component']:<36} "
        f"{int(row['parameters']):>12,}"
    )


print(
    "-" * 50
)

print(
    f"{'TOTAL TRAINABLE PARAMETERS':<36} "
    f"{int(parameter_df['parameters'].sum()):>12,}"
)


require(
    int(
        parameter_df[
            "parameters"
        ].sum()
    )
    == EXPECTED_FULL_TRAINABLE_PARAMETERS,
    (
        "Full ITRS parameter budget "
        "does not match expected total."
    ),
)


print()
print(
    "Pretrained Doc2Vec parameters:"
)

print(
    "  NOT included in neural trainable total"
)

print(
    "  Doc2Vec representations are static model inputs"
)


# =============================================================================
# 13. PARAMETER SHARE
# =============================================================================

latent_share = (
    LATENT_PARAMETERS
    /
    EXPECTED_FULL_TRAINABLE_PARAMETERS
)


print()
print(
    f"Latent-embedding share of trainable parameters: "
    f"{latent_share:.4%}"
)


# =============================================================================
# 14. END-TO-END MODULE BOUNDARY
# =============================================================================

banner(
    "END-TO-END MODULE BOUNDARY"
)


print(
    "Static persisted inputs:"
)

print(
    "  - Doc2Vec node representations [477564,32]"
)

print(
    "  - category label matrix [477564,802]"
)

print(
    "  - trend history CSR identities"
)

print(
    "  - Phase-3 graph edge_index / edge_type"
)


print()
print(
    "Trainable model state:"
)

print(
    "  - L_o"
)

print(
    "  - L_b"
)

print(
    "  - description text projection"
)

print(
    "  - description label projection"
)

print(
    "  - trend attention"
)

print(
    "  - trend GRU"
)

print(
    "  - trend output projection"
)

print(
    "  - R-GCN layer 1"
)

print(
    "  - R-GCN layer 2"
)

print(
    "  - scoring MLP"
)


print()
print(
    "Dynamically recomputed trainable features:"
)

print(
    "  - F_d"
)

print(
    "  - F_t"
)

print(
    "  - F_s"
)


print()
print(
    "Final pair scorer consumes:"
)

print(
    "  F_t || L_o || F_d,o || F_s,o"
)

print(
    "  || L_b || F_d,b || F_s,b"
)


# =============================================================================
# 15. ITEMS THAT REMAIN DELIBERATELY OPEN
# =============================================================================

banner(
    "STILL-OPEN DECISIONS"
)


still_open = [

    "authoritative description tensor path if multiple shape-matching copies exist",

    "authoritative label matrix path if multiple shape-matching copies exist",

    "exact global Kaiming initialization variant",

    "global neural seed policy",

    "training negative:positive ratio",

    "training negative candidate eligibility",

    "training historical negative exclusion",

    "training epoch count",

    "early stopping",

    "weight decay",

    "evaluation candidate-generation runtime contract",
]


for item in still_open:

    print(
        f"  - {item}"
    )


print()
print(
    "This audit freezes NONE of these decisions."
)


# =============================================================================
# 16. SAVE FULL ARTIFACT INVENTORY
# =============================================================================

artifact_df = pd.DataFrame(
    artifact_records
)


artifact_inventory_path = (
    OUT_DIR
    / "phase_4_full_artifact_inventory.csv"
)


artifact_df.to_csv(
    artifact_inventory_path,
    index=False,
)


# =============================================================================
# 17. SAVE CONTRACT INVENTORY
# =============================================================================

contract_df = pd.DataFrame(
    contract_records
)


contract_inventory_path = (
    OUT_DIR
    / "phase_4_contract_inventory.csv"
)


contract_df.to_csv(
    contract_inventory_path,
    index=False,
)


# =============================================================================
# 18. SAVE JSON-REFERENCE INVENTORY
# =============================================================================

json_reference_df = pd.DataFrame(
    json_reference_records
)


json_reference_path = (
    OUT_DIR
    / "phase_4_contract_artifact_references.csv"
)


json_reference_df.to_csv(
    json_reference_path,
    index=False,
)


# =============================================================================
# 19. SAVE PARAMETER BUDGET
# =============================================================================

parameter_budget_path = (
    OUT_DIR
    / "full_itrs_parameter_budget.csv"
)


parameter_df.to_csv(
    parameter_budget_path,
    index=False,
)


# =============================================================================
# 20. SAVE INVENTORY METADATA
# =============================================================================

metadata = {

    "phase":
        "4.6.1a",

    "status":
        "COMPLETE_AUDIT_ONLY",

    "component":
        (
            "Full ITRS artifact and "
            "parameter inventory"
        ),

    "population":
        {

            "investors":
                NUM_INVESTORS,

            "startups":
                NUM_STARTUPS,

            "role_nodes":
                NUM_NODES,
        },

    "expected_static_inputs":
        {

            "doc2vec":
                {

                    "shape":
                        [
                            NUM_NODES,
                            DOC2VEC_DIM,
                        ],

                    "candidate_count":
                        len(
                            doc2vec_candidates
                        ),

                    "candidates":
                        [
                            record[
                                "path"
                            ]

                            for record
                            in doc2vec_candidates
                        ],
                },

            "labels":
                {

                    "shape":
                        [
                            NUM_NODES,
                            LABEL_DIM,
                        ],

                    "expected_nnz":
                        EXPECTED_LABEL_NNZ,

                    "candidate_count":
                        len(
                            label_candidates
                        ),

                    "matching_nnz_count":
                        len(
                            matching_label_nnz
                        ),

                    "candidates":
                        [
                            record[
                                "path"
                            ]

                            for record
                            in label_candidates
                        ],
                },
        },

    "trainable_parameter_budget":
        {

            "investor_latents":
                INVESTOR_LATENT_PARAMETERS,

            "startup_latents":
                STARTUP_LATENT_PARAMETERS,

            "all_latents":
                LATENT_PARAMETERS,

            "description":
                DESCRIPTION_PARAMETERS,

            "trend":
                TREND_PARAMETERS,

            "rgcn":
                RGCN_PARAMETERS,

            "scoring":
                SCORING_PARAMETERS,

            "total":
                EXPECTED_FULL_TRAINABLE_PARAMETERS,

            "doc2vec_included":
                False,
        },

    "contract_statuses":
        {

            record[
                "contract"
            ]:
                record[
                    "status"
                ]

            for record
            in contract_records
        },

    "frozen_decisions_changed":
        False,

    "training_performed":
        False,

    "model_instantiated":
        False,

    "negative_sampling_changed":
        False,

    "initialization_changed":
        False,

    "next_phase":
        {

            "phase":
                "4.6.1b",

            "purpose":
                (
                    "Freeze authoritative full-model "
                    "static input paths and construct "
                    "the complete integrated ITRS "
                    "forward contract."
                ),
        },
}


metadata_path = (
    OUT_DIR
    / "phase_4_6_1a_inventory_metadata.json"
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
    "PHASE 4.6.1a FINAL SUMMARY"
)


print(
    "Frozen module contracts:"
)

print(
    "  Description                         PASS"
)

print(
    "  Trend                               PASS"
)

print(
    "  Preference propagation              PASS"
)

print(
    "  Recommendation scoring              PASS"
)


print()
print(
    "Static model-input candidates:"
)

print(
    f"  Doc2Vec [477564,32] candidates      "
    f"{len(doc2vec_candidates)}"
)

print(
    f"  Label [477564,802] candidates       "
    f"{len(label_candidates)}"
)

print(
    f"  Label nnz-matching candidates       "
    f"{len(matching_label_nnz)}"
)


print()
print(
    "Frozen runtime inputs:"
)

print(
    "  Trend CSR arrays                    PASS"
)

print(
    "  Phase-3 graph arrays                PASS"
)


print()
print(
    "Trainable parameter budget:"
)

print(
    f"  Investor latent embeddings          "
    f"{INVESTOR_LATENT_PARAMETERS:,}"
)

print(
    f"  Startup latent embeddings           "
    f"{STARTUP_LATENT_PARAMETERS:,}"
)

print(
    f"  Description encoder                 "
    f"{DESCRIPTION_PARAMETERS:,}"
)

print(
    f"  Trend module                        "
    f"{TREND_PARAMETERS:,}"
)

print(
    f"  R-GCN                               "
    f"{RGCN_PARAMETERS:,}"
)

print(
    f"  Scoring MLP                         "
    f"{SCORING_PARAMETERS:,}"
)

print(
    "                                        "
    "------------"
)

print(
    f"  FULL ITRS                            "
    f"{EXPECTED_FULL_TRAINABLE_PARAMETERS:,}"
)


print()
print(
    "Doc2Vec trainable in neural model:     NO"
)

print(
    "Training performed:                    NO"
)

print(
    "Full model instantiated:               NO"
)

print(
    "Negative sampling changed:             NO"
)

print(
    "Kaiming variant frozen:                NO"
)


print()
print("Outputs:")

for path in [

    artifact_inventory_path,

    contract_inventory_path,

    json_reference_path,

    parameter_budget_path,

    metadata_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.6.1a STATUS: COMPLETE — "
    "FULL-MODEL ARTIFACT INVENTORY AUDITED ONLY"
)


print()
print(
    "NEXT:"
)

print(
    "PHASE 4.6.1b — "
    "FREEZE AUTHORITATIVE STATIC INPUTS "
    "AND COMPLETE ITRS FORWARD CONTRACT"
)