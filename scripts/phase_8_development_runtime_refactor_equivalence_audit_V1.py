#!/usr/bin/env python3

from __future__ import annotations

import ast
import hashlib
import json
import textwrap
from pathlib import Path


SOURCE = Path(
    "scripts/"
    "phase_8_2_7_dev_t57_frozen_batch_one_step_preflight_V1.py"
)

REFERENCE = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_7_autograd_optimizer_preflight/"
    "phase_8_2_7_dev_t57_frozen_batch_one_step_results_V1.json"
)

OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "comparative_development_training_preflight/"
    "runtime_refactor_equivalence_V1"
)

OUTPUT_RESULT = (
    OUTPUT_DIR
    / "phase_8_development_runtime_refactor_equivalence_V1.json"
)

OUTPUT_HASHES = (
    OUTPUT_DIR
    / "phase_8_development_runtime_refactor_equivalence_sha256_V1.json"
)

EXPECTED_SOURCE_SHA256 = (
    "06cfea074a447a11cbdc83c7c53fa213"
    "f3f870730440d69c84b25039b19696a3"
)


def require(condition, message):

    if not bool(condition):
        raise RuntimeError(message)


def sha256_bytes(data: bytes) -> str:

    return hashlib.sha256(
        data
    ).hexdigest()


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def canonical_json_sha256(obj) -> str:

    payload = json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode(
        "utf-8"
    )

    return sha256_bytes(
        payload
    )


print("=" * 120)
print(
    "PHASE 8 — DEVELOPMENT TRAINING RUNTIME "
    "REFACTOR EQUIVALENCE AUDIT V1"
)
print("=" * 120)

print()
print("Scientific role:")
print(
    "  Mechanical refactor-equivalence audit only."
)
print(
    "  NO full epoch, NO checkpoint retention, "
    "NO ranking metrics, NO T60, NO April."
)


# =============================================================================
# 1. SOURCE + REFERENCE
# =============================================================================

print()
print("1. AUTHORITATIVE INPUTS")

require(
    SOURCE.exists(),
    f"Missing authoritative source: {SOURCE}",
)

require(
    REFERENCE.exists(),
    f"Missing frozen 8.2.7 result: {REFERENCE}",
)

source_bytes = SOURCE.read_bytes()

source_sha256 = sha256_bytes(
    source_bytes
)

require(
    source_sha256
    == EXPECTED_SOURCE_SHA256,
    (
        "8.2.7 source hash differs from the uploaded "
        "authoritative source.\n"
        f"Expected: {EXPECTED_SOURCE_SHA256}\n"
        f"Observed: {source_sha256}"
    ),
)

reference = json.loads(
    REFERENCE.read_text(
        encoding="utf-8"
    )
)

require(
    reference["status"]
    == (
        "ONE_STEP_AUTOGRAD_OPTIMIZER_"
        "PREFLIGHT_EXECUTED_AND_AUDITED"
    ),
    "Frozen 8.2.7 reference status changed.",
)

require(
    set(
        reference["families"].keys()
    )
    == {
        "B1a",
        "B1b",
        "M1",
        "M2",
    },
    "Frozen 8.2.7 reference family set changed.",
)

print(
    f"  Source SHA256:      {source_sha256}  PASS"
)
print(
    "  Frozen result:      PASS"
)


# =============================================================================
# 2. AST — LOCATE EXACT AUTHORITATIVE FAMILY LOOP
# =============================================================================

print()
print("2. LOCATE AUTHORITATIVE FAMILY EXECUTION LOOP")

source_text = source_bytes.decode(
    "utf-8"
)

tree = ast.parse(
    source_text,
    filename=str(
        SOURCE
    ),
)

candidate_loops = []

for node in tree.body:

    if not isinstance(
        node,
        ast.For,
    ):
        continue

    target_text = ast.unparse(
        node.target
    )

    iter_text = ast.unparse(
        node.iter
    )

    if (
        target_text == "family"
        and iter_text == "FAMILIES"
        and node.lineno < 2600
    ):

        candidate_loops.append(
            node
        )

require(
    len(
        candidate_loops
    )
    == 1,
    (
        "Could not identify exactly one authoritative "
        "family execution loop."
    ),
)

family_loop = candidate_loops[
    0
]

require(
    family_loop.lineno == 1478,
    (
        "Authoritative family loop start changed: "
        f"{family_loop.lineno}"
    ),
)

require(
    family_loop.end_lineno == 2670,
    (
        "Authoritative family loop end changed: "
        f"{family_loop.end_lineno}"
    ),
)

require(
    len(
        family_loop.body
    )
    > 0,
    "Authoritative family loop has empty body.",
)

print(
    "  Loop target:        family"
)
print(
    "  Loop iterator:      FAMILIES"
)
print(
    f"  Source lines:       "
    f"{family_loop.lineno}..{family_loop.end_lineno}"
)
print(
    "  AST binding:        PASS"
)


# =============================================================================
# 3. MECHANICAL EXTRACTION
# =============================================================================

print()
print("3. MECHANICAL FUNCTION EXTRACTION")

source_lines = source_text.splitlines()

# Everything before the actual for-family loop is kept byte-for-byte
# at source-line granularity. This includes:
#
#   - prerequisite checks
#   - frozen batch
#   - feature registry
#   - Investor inputs
#   - strict pre-target history
#   - exact two-hop graph closure
#   - raw Startup feature tensors
#   - canonical initialization namespace
#   - family_results = {}
#
preamble = (
    "\n".join(
        source_lines[
            :family_loop.lineno - 1
        ]
    )
    + "\n"
)

body_start = (
    family_loop.body[
        0
    ].lineno
)

body_end = (
    family_loop.body[
        -1
    ].end_lineno
)

raw_body = "\n".join(
    source_lines[
        body_start - 1:
        body_end
    ]
)

dedented_body = textwrap.dedent(
    raw_body
)

function_source = (
    "def execute_family_refactored(family):\n"
    + textwrap.indent(
        dedented_body,
        "    ",
    )
    + "\n"
    + "    return family_results[family]\n"
)

extracted_body_sha256 = sha256_bytes(
    raw_body.encode(
        "utf-8"
    )
)

generated_function_sha256 = sha256_bytes(
    function_source.encode(
        "utf-8"
    )
)

# Compile before touching any model runtime.
compile(
    function_source,
    "<phase8_refactored_family_runtime>",
    "exec",
)

print(
    f"  Original body:      "
    f"lines {body_start}..{body_end}"
)
print(
    f"  Body SHA256:        "
    f"{extracted_body_sha256}"
)
print(
    f"  Function SHA256:    "
    f"{generated_function_sha256}"
)
print(
    "  Compilation:        PASS"
)


# =============================================================================
# 4. EXECUTE ORIGINAL PREAMBLE
# =============================================================================

print()
print("4. EXECUTE UNCHANGED 8.2.7 PREAMBLE")

runtime_namespace = {
    "__file__":
        str(
            SOURCE.resolve()
        ),

    "__name__":
        "__phase8_runtime_refactor_equivalence__",
}

exec(
    compile(
        preamble,
        str(
            SOURCE
        ),
        "exec",
    ),
    runtime_namespace,
)

require(
    "family_results"
    in runtime_namespace,
    "family_results missing after authoritative preamble.",
)

require(
    runtime_namespace[
        "family_results"
    ]
    == {},
    (
        "Authoritative preamble unexpectedly "
        "executed a family."
    ),
)

print(
    "  Data preparation:   PASS"
)
print(
    "  Model execution:    NONE YET"
)


# =============================================================================
# 5. INSTALL MECHANICALLY EXTRACTED FUNCTION
# =============================================================================

exec(
    compile(
        function_source,
        "<phase8_refactored_family_runtime>",
        "exec",
    ),
    runtime_namespace,
)

execute_family = runtime_namespace[
    "execute_family_refactored"
]

families = list(
    runtime_namespace[
        "FAMILIES"
    ]
)

require(
    families
    == [
        "B1a",
        "B1b",
        "M1",
        "M2",
    ],
    "Family execution order changed.",
)


# =============================================================================
# 6. REAL ONE-BATCH EQUIVALENCE EXECUTION
# =============================================================================

print()
print("=" * 120)
print("5. EXACT ONE-BATCH REFACTOR EQUIVALENCE")
print("=" * 120)

observed = {}
comparison = {}

for family in families:

    print()
    print("#" * 120)
    print(
        f"REFACTORED EXECUTION — {family}"
    )
    print("#" * 120)

    result = execute_family(
        family
    )

    observed[
        family
    ] = result

    expected = reference[
        "families"
    ][
        family
    ]

    observed_sha = canonical_json_sha256(
        result
    )

    expected_sha = canonical_json_sha256(
        expected
    )

    exact_equal = (
        result == expected
    )

    require(
        exact_equal,
        (
            f"{family}: mechanically refactored "
            "runtime does NOT exactly reproduce "
            "the frozen 8.2.7 family result.\n"
            f"Expected result SHA: {expected_sha}\n"
            f"Observed result SHA: {observed_sha}"
        ),
    )

    require(
        result[
            "state_audit"
        ][
            "initial_parameter_sha256"
        ]
        ==
        expected[
            "state_audit"
        ][
            "initial_parameter_sha256"
        ],
        f"{family}: initial model hash mismatch.",
    )

    require(
        result[
            "state_audit"
        ][
            "post_step_parameter_sha256"
        ]
        ==
        expected[
            "state_audit"
        ][
            "post_step_parameter_sha256"
        ],
        f"{family}: post-step model hash mismatch.",
    )

    require(
        result[
            "state_audit"
        ][
            "optimizer_step_calls"
        ]
        == 1,
        f"{family}: optimizer-step count changed.",
    )

    require(
        result[
            "post_step_policy"
        ][
            "checkpoint_written"
        ]
        is False,
        f"{family}: checkpoint unexpectedly written.",
    )

    require(
        result[
            "post_step_policy"
        ][
            "updated_state_retained"
        ]
        is False,
        f"{family}: updated state unexpectedly retained.",
    )

    comparison[
        family
    ] = {
        "exact_family_result_equal":
            True,

        "reference_family_result_sha256":
            expected_sha,

        "observed_family_result_sha256":
            observed_sha,

        "initial_parameter_sha256_equal":
            True,

        "post_step_parameter_sha256_equal":
            True,

        "loss_exact_equal":
            (
                result[
                    "numerical_audit"
                ][
                    "loss_bce_with_logits"
                ]
                ==
                expected[
                    "numerical_audit"
                ][
                    "loss_bce_with_logits"
                ]
            ),

        "gradient_components_exact_equal":
            (
                result[
                    "numerical_audit"
                ][
                    "gradient_components"
                ]
                ==
                expected[
                    "numerical_audit"
                ][
                    "gradient_components"
                ]
            ),

        "changed_parameter_tensors_equal":
            (
                result[
                    "state_audit"
                ][
                    "changed_parameter_tensors"
                ]
                ==
                expected[
                    "state_audit"
                ][
                    "changed_parameter_tensors"
                ]
            ),
    }

    require(
        comparison[
            family
        ][
            "loss_exact_equal"
        ],
        f"{family}: BCE loss changed.",
    )

    require(
        comparison[
            family
        ][
            "gradient_components_exact_equal"
        ],
        f"{family}: gradient-component audit changed.",
    )

    require(
        comparison[
            family
        ][
            "changed_parameter_tensors_equal"
        ],
        f"{family}: changed-parameter count changed.",
    )

    print()
    print(
        "  Complete family result:            EXACT MATCH"
    )
    print(
        "  Initial parameter SHA256:          EXACT MATCH"
    )
    print(
        "  BCEWithLogitsLoss:                 EXACT MATCH"
    )
    print(
        "  Gradient components:               EXACT MATCH"
    )
    print(
        "  Post-step parameter SHA256:        EXACT MATCH"
    )
    print(
        "  Updated state retained:            NO"
    )


# =============================================================================
# 7. GLOBAL RESULT
# =============================================================================

require(
    runtime_namespace[
        "family_results"
    ]
    == observed,
    "Runtime family-result registry changed unexpectedly.",
)

reference_families_sha256 = (
    canonical_json_sha256(
        reference[
            "families"
        ]
    )
)

observed_families_sha256 = (
    canonical_json_sha256(
        observed
    )
)

require(
    observed_families_sha256
    == reference_families_sha256,
    "Combined four-family result fingerprint changed.",
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

audit = {
    "schema_version":
        (
            "PHASE_8_DEVELOPMENT_RUNTIME_"
            "REFACTOR_EQUIVALENCE_V1"
        ),

    "status":
        "EXACT_REFACTOR_EQUIVALENCE_PASS",

    "scientific_role":
        (
            "POST_CLOSURE_IMPLEMENTATION_REFACTOR_"
            "EQUIVALENCE_AUDIT_ONLY"
        ),

    "authoritative_source":
        str(
            SOURCE
        ),

    "authoritative_source_sha256":
        source_sha256,

    "frozen_reference":
        str(
            REFERENCE
        ),

    "mechanical_extraction": {
        "family_loop_start_line":
            int(
                family_loop.lineno
            ),

        "family_loop_end_line":
            int(
                family_loop.end_lineno
            ),

        "body_start_line":
            int(
                body_start
            ),

        "body_end_line":
            int(
                body_end
            ),

        "body_sha256":
            extracted_body_sha256,

        "generated_function_sha256":
            generated_function_sha256,

        "semantic_edits_to_family_body":
            False,
    },

    "families":
        comparison,

    "combined_reference_family_result_sha256":
        reference_families_sha256,

    "combined_observed_family_result_sha256":
        observed_families_sha256,

    "execution_boundary": {
        "full_epoch_executed":
            False,

        "ranking_metrics_computed":
            False,

        "model_selection_performed":
            False,

        "checkpoint_written":
            False,

        "updated_model_state_retained":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,
    },

    "interpretation":
        (
            "The authoritative 8.2.7 family execution body "
            "can be mechanically encapsulated as a callable "
            "function without any numerical or state change. "
            "This authorizes construction of a reusable "
            "development-training runtime around this exact "
            "family computation path; it does not itself "
            "execute a full training epoch."
        ),
}

OUTPUT_RESULT.write_text(
    json.dumps(
        audit,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)

hash_registry = {
    str(
        SOURCE
    ):
        sha256_file(
            SOURCE
        ),

    str(
        REFERENCE
    ):
        sha256_file(
            REFERENCE
        ),

    str(
        OUTPUT_RESULT
    ):
        sha256_file(
            OUTPUT_RESULT
        ),
}

OUTPUT_HASHES.write_text(
    json.dumps(
        hash_registry,
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)


print()
print("=" * 120)
print("RUNTIME REFACTOR EQUIVALENCE — GLOBAL AUDIT")
print("=" * 120)

print(
    "Authoritative source SHA:            PASS"
)
print(
    "Mechanical family-body extraction:  PASS"
)
print(
    "B1a exact numerical/state match:     PASS"
)
print(
    "B1b exact numerical/state match:     PASS"
)
print(
    "M1 exact numerical/state match:      PASS"
)
print(
    "M2 exact numerical/state match:      PASS"
)
print(
    "Combined four-family fingerprint:    PASS"
)
print(
    "Full epoch executed:                 NO"
)
print(
    "Ranking metrics computed:            NO"
)
print(
    "Checkpoint written:                  NO"
)
print(
    "Updated state retained:              NO"
)
print(
    "T60 scored:                          NO"
)
print(
    "April analyzed:                      NO"
)

print()
print(
    f"Audit:  {OUTPUT_RESULT}"
)
print(
    f"Hashes: {OUTPUT_HASHES}"
)

print()
print(
    "STATUS: EXACT_REFACTOR_EQUIVALENCE_PASS"
)
print(
    "NEXT:   Generalize this exact callable path "
    "from the frozen first batch to arbitrary "
    "development-stream batches."
)
print("=" * 120)
