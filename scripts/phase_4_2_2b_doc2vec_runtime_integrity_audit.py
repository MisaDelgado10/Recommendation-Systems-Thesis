from pathlib import Path
import contextlib
import io
import json
import os
import platform

import gensim
import numpy as np
import scipy
from scipy.linalg import blas

from gensim.models.doc2vec import Doc2Vec


ROOT = Path(
    "data/experimental/phase_4/doc2vec"
)

INV_MODEL = (
    ROOT / "models/investor_doc2vec.model"
)

START_MODEL = (
    ROOT / "models/startup_doc2vec.model"
)

INV_VEC = (
    ROOT / "vectors/investor_doc2vec_vectors.npy"
)

START_VEC = (
    ROOT / "vectors/startup_doc2vec_vectors.npy"
)

ALL_VEC = (
    ROOT / "vectors/doc2vec_vectors_all.npy"
)

MANIFEST = (
    ROOT / "vectors/doc2vec_vector_manifest.parquet"
)


def banner(title):
    print()
    print("=" * 115)
    print(title)
    print("=" * 115)


banner(
    "PHASE 4.2.2b — "
    "DOC2VEC SAVED-ARTIFACT AND BLAS RUNTIME AUDIT"
)


# =============================================================================
# 1. Environment
# =============================================================================

banner("ENVIRONMENT")

print("Platform:", platform.platform())
print("NumPy:", np.__version__)
print("SciPy:", scipy.__version__)
print("Gensim:", gensim.__version__)
print(
    "PYTHONHASHSEED:",
    os.environ.get("PYTHONHASHSEED"),
)


# =============================================================================
# 2. Direct BLAS SDOT numerical check
# =============================================================================

banner("SCIPY BLAS SDOT CHECK")

rng = np.random.default_rng(42)

max_abs_error = 0.0

for _ in range(1000):

    x = rng.standard_normal(
        32
    ).astype(np.float32)

    y = rng.standard_normal(
        32
    ).astype(np.float32)

    expected = float(
        np.dot(x, y)
    )

    actual = float(
        blas.sdot(x, y)
    )

    error = abs(
        expected - actual
    )

    max_abs_error = max(
        max_abs_error,
        error,
    )


print(
    "Maximum absolute difference "
    "NumPy dot vs scipy BLAS sdot:",
    max_abs_error,
)

if not np.isfinite(
    max_abs_error
):
    raise AssertionError(
        "BLAS comparison produced "
        "non-finite values."
    )


# =============================================================================
# 3. Reload saved models
# =============================================================================

banner("MODEL RELOAD")

investor_model = Doc2Vec.load(
    str(INV_MODEL)
)

startup_model = Doc2Vec.load(
    str(START_MODEL)
)

print(
    "Investor vocabulary:",
    f"{len(investor_model.wv):,}",
)

print(
    "Investor documents:",
    f"{len(investor_model.dv):,}",
)

print(
    "Startup vocabulary:",
    f"{len(startup_model.wv):,}",
)

print(
    "Startup documents:",
    f"{len(startup_model.dv):,}",
)


assert len(investor_model.wv) == 36_505
assert len(investor_model.dv) == 163_531

assert len(startup_model.wv) == 57_605
assert len(startup_model.dv) == 311_363


# =============================================================================
# 4. Reload vector arrays
# =============================================================================

banner("VECTOR ARRAY RELOAD")

investor_vectors = np.load(
    INV_VEC
)

startup_vectors = np.load(
    START_VEC
)

combined_vectors = np.load(
    ALL_VEC
)


print(
    "Investor vectors:",
    investor_vectors.shape,
)

print(
    "Startup vectors:",
    startup_vectors.shape,
)

print(
    "Combined vectors:",
    combined_vectors.shape,
)


assert investor_vectors.shape == (
    165_975,
    32,
)

assert startup_vectors.shape == (
    311_589,
    32,
)

assert combined_vectors.shape == (
    477_564,
    32,
)


for name, array in [
    (
        "investor",
        investor_vectors,
    ),
    (
        "startup",
        startup_vectors,
    ),
    (
        "combined",
        combined_vectors,
    ),
]:

    if not np.isfinite(
        array
    ).all():

        raise AssertionError(
            f"{name}: NaN/Inf detected."
        )


# =============================================================================
# 5. Verify combined matrix concatenation
# =============================================================================

banner("COMBINED MATRIX EXACTNESS")

expected_combined = np.concatenate(
    [
        investor_vectors,
        startup_vectors,
    ],
    axis=0,
)


exact_combined = np.array_equal(
    expected_combined,
    combined_vectors,
)


print(
    "Exact Investor + Startup concatenation:",
    exact_combined,
)


if not exact_combined:
    raise AssertionError(
        "Combined Doc2Vec matrix "
        "does not exactly equal role matrices."
    )


# =============================================================================
# 6. Reload-model ↔ saved-vector exactness
# =============================================================================

banner("MODEL / SAVED VECTOR EXACTNESS")

import pandas as pd

manifest = pd.read_parquet(
    MANIFEST
)


checks = []


for role, model, vectors in [
    (
        "investor",
        investor_model,
        investor_vectors,
    ),
    (
        "startup",
        startup_model,
        startup_vectors,
    ),
]:

    role_manifest = (
        manifest.loc[
            manifest[
                "node_type"
            ].eq(role)
        ]
        .reset_index(drop=True)
    )

    eligible = role_manifest.loc[
        ~role_manifest[
            "doc2vec_zero_vector"
        ]
    ]


    # Deterministic sample distributed through
    # the eligible-role manifest.
    sample_positions = np.linspace(
        0,
        len(eligible) - 1,
        num=min(
            10_000,
            len(eligible),
        ),
        dtype=np.int64,
    )


    sample = eligible.iloc[
        sample_positions
    ]


    mismatch = 0


    for row in sample.itertuples():

        node_id = row.node_id

        local_row = int(
            role_manifest.index[
                role_manifest[
                    "node_id"
                ].eq(node_id)
            ][0]
        )

        saved = vectors[
            local_row
        ]

        model_vector = model.dv[
            node_id
        ]


        if not np.array_equal(
            saved,
            model_vector,
        ):
            mismatch += 1


    print(
        f"{role}: exact sampled "
        f"model/vector mismatches = "
        f"{mismatch:,}"
    )


    if mismatch:
        raise AssertionError(
            f"{role}: model and saved "
            "vectors differ."
        )


    checks.append(
        {
            "node_type": role,
            "sampled_documents":
                len(sample),
            "exact_mismatches":
                mismatch,
        }
    )


# =============================================================================
# 7. Norm sanity
# =============================================================================

banner("VECTOR NORM SANITY")

for role, array in [
    (
        "investor",
        investor_vectors,
    ),
    (
        "startup",
        startup_vectors,
    ),
]:

    norms = np.linalg.norm(
        array,
        axis=1,
    )

    nonzero = norms[
        norms > 0
    ]

    print()
    print(role.upper())

    print(
        "zero vectors:",
        f"{np.sum(norms == 0):,}",
    )

    print(
        "minimum nonzero norm:",
        float(nonzero.min()),
    )

    print(
        "median nonzero norm:",
        float(np.median(nonzero)),
    )

    print(
        "maximum nonzero norm:",
        float(nonzero.max()),
    )


# =============================================================================
# Final
# =============================================================================

banner("PHASE 4.2.2b SUMMARY")

print(
    "Saved models reload:             PASS"
)

print(
    "Saved arrays reload:             PASS"
)

print(
    "Combined matrix exactness:       PASS"
)

print(
    "Model/vector sampled exactness:  PASS"
)

print(
    "Finite-vector integrity:         PASS"
)

print()
print(
    "NOTE: This audit does not by itself "
    "prove that the observed Gensim Cython "
    "warning is harmless."
)

print()
print(
    "PHASE 4.2.2b STATUS: COMPLETE"
)