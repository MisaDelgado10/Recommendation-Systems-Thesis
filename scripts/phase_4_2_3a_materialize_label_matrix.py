from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from scipy import sparse


# =============================================================================
# PHASE 4.2.3a — MATERIALIZE AND AUDIT FROZEN LABEL MATRIX
# =============================================================================

DESCRIPTION_DIR = Path(
    "data/experimental/phase_4/"
    "description_contract"
)

MANIFEST_PATH = (
    DESCRIPTION_DIR
    / "description_input_manifest.parquet"
)

VOCAB_PATH = (
    DESCRIPTION_DIR
    / "description_label_vocabulary.csv"
)


DOC2VEC_MANIFEST_PATH = Path(
    "data/experimental/phase_4/"
    "doc2vec/"
    "vectors/"
    "doc2vec_vector_manifest.parquet"
)


OUT_DIR = Path(
    "data/experimental/phase_4/"
    "description_labels"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# Frozen expectations
# =============================================================================

EXPECTED_ROLE_NODES = 477_564

EXPECTED_INVESTORS = 165_975
EXPECTED_STARTUPS = 311_589

EXPECTED_VOCABULARY = 802

EXPECTED_INVESTORS_WITH_LABELS = 61_957
EXPECTED_STARTUPS_WITH_LABELS = 303_688

EXPECTED_LABELED_TOTAL = 365_645

EXPECTED_ZERO_LABEL_ROWS = 111_919


# These are corrected Phase-4.1.3c counts after treating
# "Heating, Ventilation, and Air Conditioning (HVAC)"
# as ONE category rather than three provisional fragments.

EXPECTED_INVESTOR_NNZ = 186_437
EXPECTED_STARTUP_NNZ = 1_043_631

EXPECTED_TOTAL_NNZ = 1_230_068


# =============================================================================
# Helpers
# =============================================================================

def banner(title):

    print()
    print("=" * 118)
    print(title)
    print("=" * 118)


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
# 1. Load frozen description manifest
# =============================================================================

banner(
    "PHASE 4.2.3a — "
    "MATERIALIZE AND AUDIT FROZEN LABEL MATRIX"
)


manifest = pd.read_parquet(
    MANIFEST_PATH,
    columns=[
        "node_id",
        "node_type",
        "raw_entity_id",
        "has_labels",
        "label_count",
        "labels_json",
    ],
)


if len(manifest) != EXPECTED_ROLE_NODES:

    raise AssertionError(
        "Frozen description-manifest "
        "population changed."
    )


if manifest[
    "node_id"
].duplicated().any():

    raise AssertionError(
        "Duplicate node_id detected."
    )


role_counts = (
    manifest[
        "node_type"
    ]
    .value_counts()
    .to_dict()
)


assert (
    role_counts.get(
        "investor",
        0,
    )
    == EXPECTED_INVESTORS
)

assert (
    role_counts.get(
        "startup",
        0,
    )
    == EXPECTED_STARTUPS
)


manifest[
    "label_feature_row"
] = np.arange(
    len(manifest),
    dtype=np.int64,
)


print(
    f"Frozen role nodes: "
    f"{len(manifest):,}"
)


# =============================================================================
# 2. Load frozen 802-label vocabulary
#
# Disable Pandas NA interpretation because arbitrary lexical strings must
# remain literal strings — same principle discovered in Phase 4.2.2a.
# =============================================================================

banner(
    "LOADING FROZEN SHARED LABEL VOCABULARY"
)


vocab = pd.read_csv(
    VOCAB_PATH,
    keep_default_na=False,
    na_filter=False,
    dtype={
        "label_index":
            "int64",

        "rank_by_entity_frequency":
            "int64",

        "label":
            "string",

        "entity_frequency":
            "int64",

        "selected_by_paper_top_k":
            "boolean",
    },
)


if len(vocab) != EXPECTED_VOCABULARY:

    raise AssertionError(
        f"Expected "
        f"{EXPECTED_VOCABULARY:,} "
        f"labels; found "
        f"{len(vocab):,}."
    )


if vocab[
    "label"
].isna().any():

    raise AssertionError(
        "Parsed NA value in frozen "
        "label vocabulary."
    )


if vocab[
    "label"
].duplicated().any():

    raise AssertionError(
        "Duplicate label in frozen vocabulary."
    )


expected_indices = np.arange(
    EXPECTED_VOCABULARY,
    dtype=np.int64,
)


actual_indices = (
    vocab[
        "label_index"
    ]
    .to_numpy(
        dtype=np.int64
    )
)


if not np.array_equal(
    expected_indices,
    actual_indices,
):

    raise AssertionError(
        "Frozen label_index is not "
        "contiguous 0..801."
    )


label_to_index = dict(
    zip(
        vocab[
            "label"
        ].astype(str),

        vocab[
            "label_index"
        ].astype(int),
    )
)


print(
    f"Frozen shared vocabulary: "
    f"{len(vocab):,}"
)


# =============================================================================
# 3. Parse frozen labels_json
# =============================================================================

banner(
    "PARSING FROZEN ENTITY LABEL SETS"
)


row_indices = []
column_indices = []

unmapped_labels = []

duplicate_labels_within_entity = []

observed_label_count = np.zeros(
    EXPECTED_ROLE_NODES,
    dtype=np.int32,
)


for row in manifest.itertuples(
    index=False
):

    labels = json.loads(
        row.labels_json
    )


    if not isinstance(
        labels,
        list,
    ):

        raise AssertionError(
            f"{row.node_id}: labels_json "
            "did not decode to list."
        )


    if (
        len(labels)
        != len(set(labels))
    ):

        duplicate_labels_within_entity.append(
            row.node_id
        )


    observed_label_count[
        row.label_feature_row
    ] = len(labels)


    for label in labels:

        if label not in label_to_index:

            unmapped_labels.append(
                (
                    row.node_id,
                    label,
                )
            )

            continue


        row_indices.append(
            row.label_feature_row
        )

        column_indices.append(
            label_to_index[
                label
            ]
        )


print(
    f"Duplicate-label entities: "
    f"{len(duplicate_labels_within_entity):,}"
)

print(
    f"Unmapped entity-label pairs: "
    f"{len(unmapped_labels):,}"
)


if duplicate_labels_within_entity:

    raise AssertionError(
        "At least one entity contains "
        "duplicate labels."
    )


if unmapped_labels:

    print()
    print(
        "First unmapped labels:"
    )

    for record in (
        unmapped_labels[
            :20
        ]
    ):

        print(
            repr(record)
        )

    raise AssertionError(
        "Frozen entity labels do not "
        "map exactly to frozen vocabulary."
    )


# =============================================================================
# 4. Verify frozen label_count metadata
# =============================================================================

banner(
    "LABEL-COUNT METADATA INTEGRITY"
)


frozen_label_count = (
    manifest[
        "label_count"
    ]
    .to_numpy(
        dtype=np.int32
    )
)


label_count_mismatches = int(
    np.sum(
        observed_label_count
        != frozen_label_count
    )
)


print(
    f"label_count mismatches: "
    f"{label_count_mismatches:,}"
)


if label_count_mismatches:

    raise AssertionError(
        "Frozen labels_json disagrees "
        "with frozen label_count."
    )


# =============================================================================
# 5. Build sparse binary matrix
# =============================================================================

banner(
    "BUILDING CSR MULTI-HOT MATRIX"
)


data = np.ones(
    len(row_indices),
    dtype=np.float32,
)


label_matrix = sparse.csr_matrix(
    (
        data,
        (
            np.asarray(
                row_indices,
                dtype=np.int64,
            ),
            np.asarray(
                column_indices,
                dtype=np.int64,
            ),
        ),
    ),
    shape=(
        EXPECTED_ROLE_NODES,
        EXPECTED_VOCABULARY,
    ),
    dtype=np.float32,
)


label_matrix.sum_duplicates()
label_matrix.sort_indices()


print(
    f"Matrix shape: "
    f"{label_matrix.shape}"
)

print(
    f"Matrix dtype: "
    f"{label_matrix.dtype}"
)

print(
    f"Nonzero entries: "
    f"{label_matrix.nnz:,}"
)


if label_matrix.shape != (
    EXPECTED_ROLE_NODES,
    EXPECTED_VOCABULARY,
):

    raise AssertionError(
        "Label matrix shape mismatch."
    )


if label_matrix.nnz != (
    EXPECTED_TOTAL_NNZ
):

    raise AssertionError(
        f"Expected "
        f"{EXPECTED_TOTAL_NNZ:,} "
        f"nonzero entries; found "
        f"{label_matrix.nnz:,}."
    )


if not np.all(
    label_matrix.data
    == 1.0
):

    raise AssertionError(
        "Label matrix is not exactly binary."
    )


# =============================================================================
# 6. Row-sum exactness
# =============================================================================

banner(
    "ROW-SUM / MULTI-HOT INTEGRITY"
)


row_sums = np.asarray(
    label_matrix.sum(
        axis=1
    )
).ravel().astype(
    np.int32
)


row_sum_mismatches = int(
    np.sum(
        row_sums
        != frozen_label_count
    )
)


print(
    f"Row-sum mismatches: "
    f"{row_sum_mismatches:,}"
)


if row_sum_mismatches:

    raise AssertionError(
        "Sparse matrix row sums disagree "
        "with frozen label counts."
    )


actual_has_labels = (
    row_sums > 0
)


frozen_has_labels = (
    manifest[
        "has_labels"
    ]
    .to_numpy(
        dtype=bool
    )
)


if not np.array_equal(
    actual_has_labels,
    frozen_has_labels,
):

    raise AssertionError(
        "Sparse label matrix disagrees "
        "with frozen has_labels flag."
    )


# =============================================================================
# 7. Role-specific integrity
# =============================================================================

banner(
    "ROLE-SPECIFIC LABEL INTEGRITY"
)


investor_mask = (
    manifest[
        "node_type"
    ]
    .eq("investor")
    .to_numpy()
)


startup_mask = (
    manifest[
        "node_type"
    ]
    .eq("startup")
    .to_numpy()
)


investor_labeled = int(
    np.sum(
        actual_has_labels[
            investor_mask
        ]
    )
)

startup_labeled = int(
    np.sum(
        actual_has_labels[
            startup_mask
        ]
    )
)


investor_nnz = int(
    label_matrix[
        investor_mask
    ].nnz
)

startup_nnz = int(
    label_matrix[
        startup_mask
    ].nnz
)


print(
    f"Investors with labels: "
    f"{investor_labeled:,}"
)

print(
    f"Investor nonzero entries: "
    f"{investor_nnz:,}"
)

print()

print(
    f"Startups with labels: "
    f"{startup_labeled:,}"
)

print(
    f"Startup nonzero entries: "
    f"{startup_nnz:,}"
)


assert (
    investor_labeled
    == EXPECTED_INVESTORS_WITH_LABELS
)

assert (
    startup_labeled
    == EXPECTED_STARTUPS_WITH_LABELS
)

assert (
    investor_nnz
    == EXPECTED_INVESTOR_NNZ
)

assert (
    startup_nnz
    == EXPECTED_STARTUP_NNZ
)


labeled_total = int(
    np.sum(
        actual_has_labels
    )
)

zero_rows = (
    EXPECTED_ROLE_NODES
    - labeled_total
)


assert (
    labeled_total
    == EXPECTED_LABELED_TOTAL
)

assert (
    zero_rows
    == EXPECTED_ZERO_LABEL_ROWS
)


print()
print(
    f"Total labeled role nodes: "
    f"{labeled_total:,}"
)

print(
    f"All-zero label rows:      "
    f"{zero_rows:,}"
)


# =============================================================================
# 8. Doc2Vec / label row-alignment audit
# =============================================================================

banner(
    "DOC2VEC / LABEL ROW ALIGNMENT"
)


doc2vec_manifest = (
    pd.read_parquet(
        DOC2VEC_MANIFEST_PATH,
        columns=[
            "doc2vec_feature_row",
            "node_id",
            "node_type",
            "raw_entity_id",
        ],
    )
)


if len(
    doc2vec_manifest
) != EXPECTED_ROLE_NODES:

    raise AssertionError(
        "Doc2Vec manifest population changed."
    )


same_node_order = np.array_equal(
    manifest[
        "node_id"
    ].astype(str).to_numpy(),

    doc2vec_manifest[
        "node_id"
    ].astype(str).to_numpy(),
)


same_role_order = np.array_equal(
    manifest[
        "node_type"
    ].astype(str).to_numpy(),

    doc2vec_manifest[
        "node_type"
    ].astype(str).to_numpy(),
)


same_raw_id_order = np.array_equal(
    manifest[
        "raw_entity_id"
    ].astype(str).to_numpy(),

    doc2vec_manifest[
        "raw_entity_id"
    ].astype(str).to_numpy(),
)


doc2vec_rows_contiguous = (
    np.array_equal(
        doc2vec_manifest[
            "doc2vec_feature_row"
        ].to_numpy(
            dtype=np.int64
        ),

        np.arange(
            EXPECTED_ROLE_NODES,
            dtype=np.int64,
        ),
    )
)


print(
    f"Same node order:      "
    f"{same_node_order}"
)

print(
    f"Same role order:      "
    f"{same_role_order}"
)

print(
    f"Same raw-ID order:    "
    f"{same_raw_id_order}"
)

print(
    f"Doc2Vec rows 0..N-1:  "
    f"{doc2vec_rows_contiguous}"
)


if not (
    same_node_order
    and same_role_order
    and same_raw_id_order
    and doc2vec_rows_contiguous
):

    raise AssertionError(
        "Description text and label "
        "feature rows are not exactly aligned."
    )


# =============================================================================
# 9. Save artifacts
# =============================================================================

matrix_path = (
    OUT_DIR
    / "description_label_multihot.npz"
)


manifest_path = (
    OUT_DIR
    / "description_label_vector_manifest.parquet"
)


audit_path = (
    OUT_DIR
    / "description_label_matrix_audit.csv"
)


metadata_path = (
    OUT_DIR
    / "description_label_matrix_metadata.json"
)


sparse.save_npz(
    matrix_path,
    label_matrix,
    compressed=True,
)


label_manifest = manifest[
    [
        "label_feature_row",
        "node_id",
        "node_type",
        "raw_entity_id",
        "has_labels",
        "label_count",
    ]
].copy()


label_manifest.to_parquet(
    manifest_path,
    index=False,
)


audit_df = pd.DataFrame(
    [
        {
            "metric":
                "role_nodes",
            "value":
                EXPECTED_ROLE_NODES,
        },
        {
            "metric":
                "label_vocabulary",
            "value":
                EXPECTED_VOCABULARY,
        },
        {
            "metric":
                "matrix_nonzero_entries",
            "value":
                label_matrix.nnz,
        },
        {
            "metric":
                "investors_with_labels",
            "value":
                investor_labeled,
        },
        {
            "metric":
                "startups_with_labels",
            "value":
                startup_labeled,
        },
        {
            "metric":
                "all_zero_label_rows",
            "value":
                zero_rows,
        },
    ]
)


audit_df.to_csv(
    audit_path,
    index=False,
)


metadata = {
    "phase":
        "4.2.3a",

    "status":
        "COMPLETE",

    "representation":
        "CSR multi-hot binary matrix",

    "shape": [
        EXPECTED_ROLE_NODES,
        EXPECTED_VOCABULARY,
    ],

    "dtype":
        "float32",

    "nnz":
        int(
            label_matrix.nnz
        ),

    "label_semantics":
        "Crunchbase categories",

    "shared_vocabulary":
        True,

    "vocabulary_size":
        EXPECTED_VOCABULARY,

    "paper_mapping":
        (
            "ITRS calls Labels_o and Labels_b "
            "one-hot encodings; because entities "
            "can possess multiple labels, the "
            "reproduction represents the set as "
            "a binary multi-hot vector."
        ),

    "missing_label_policy":
        "all-zero 802-dimensional row",

    "text_label_row_alignment":
        "exact",

    "phase_4_1_3_reopened":
        False,

    "doc2vec_contract_reopened":
        False,
}


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
# 10. Persistence reload audit
# =============================================================================

banner(
    "PERSISTENCE RELOAD AUDIT"
)


reloaded = sparse.load_npz(
    matrix_path
)


if reloaded.shape != (
    EXPECTED_ROLE_NODES,
    EXPECTED_VOCABULARY,
):

    raise AssertionError(
        "Reloaded sparse-matrix shape changed."
    )


if reloaded.nnz != (
    EXPECTED_TOTAL_NNZ
):

    raise AssertionError(
        "Reloaded sparse-matrix nnz changed."
    )


if not np.array_equal(
    label_matrix.indptr,
    reloaded.indptr,
):

    raise AssertionError(
        "CSR indptr changed after persistence."
    )


if not np.array_equal(
    label_matrix.indices,
    reloaded.indices,
):

    raise AssertionError(
        "CSR indices changed after persistence."
    )


if not np.array_equal(
    label_matrix.data,
    reloaded.data,
):

    raise AssertionError(
        "CSR data changed after persistence."
    )


matrix_hash = sha256_file(
    matrix_path
)

manifest_hash = sha256_file(
    manifest_path
)


print(
    f"Matrix SHA256:   "
    f"{matrix_hash}"
)

print(
    f"Manifest SHA256: "
    f"{manifest_hash}"
)


# =============================================================================
# 11. Final summary
# =============================================================================

banner(
    "PHASE 4.2.3a FINAL SUMMARY"
)


print(
    f"Label matrix shape:      "
    f"{label_matrix.shape}"
)

print(
    f"Label matrix nnz:        "
    f"{label_matrix.nnz:,}"
)

print(
    f"Shared vocabulary:       "
    f"{EXPECTED_VOCABULARY:,}"
)

print(
    f"Investors with labels:   "
    f"{investor_labeled:,}"
)

print(
    f"Startups with labels:    "
    f"{startup_labeled:,}"
)

print(
    f"All-zero label rows:     "
    f"{zero_rows:,}"
)

print()
print(
    "Binary matrix integrity:   PASS"
)

print(
    "Frozen label counts:       PASS"
)

print(
    "Vocabulary mapping:        PASS"
)

print(
    "Doc2Vec row alignment:     PASS"
)

print(
    "Persistence reload:        PASS"
)

print()
print("Outputs:")

for path in [
    matrix_path,
    manifest_path,
    audit_path,
    metadata_path,
]:

    print(
        f"  {path}"
    )


print()
print(
    "PHASE 4.2.3a STATUS: COMPLETE — "
    "LABEL INPUT MATRIX MATERIALIZED"
)