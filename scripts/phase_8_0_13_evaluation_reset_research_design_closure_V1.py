from pathlib import Path
import hashlib
import json
import re

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 8.0.13 — EVALUATION RESET & RESEARCH DESIGN CLOSURE V1
# =============================================================================
#
# Scientific role:
#
#   INTEGRITY / PROVENANCE / PHASE-CLOSURE AUDIT ONLY
#
# This script:
#
#   - verifies authoritative Phase-8.0 artifacts;
#   - recomputes hashes from all authoritative derived-hash registers;
#   - verifies graph-mask counts;
#   - verifies candidate-universe / target counts;
#   - verifies frozen 99-negative matrices;
#   - verifies research-design / statistical contracts;
#   - records superseded Phase-8.0.9 partial artifacts;
#   - writes the Phase-8.0 closure evidence package.
#
# This script DOES NOT:
#
#   - construct a model;
#   - load a model;
#   - load a checkpoint;
#   - train;
#   - infer;
#   - score rankings;
#   - select a model;
#   - regenerate negatives;
#   - resample negatives;
#   - inspect April performance.
#
# =============================================================================


# =============================================================================
# ROOTS
# =============================================================================

PHASE8_ROOT = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset"
)

OUTPUT_DIR = (
    PHASE8_ROOT
    / "phase_8_0_13_phase_closure"
)

CANONICAL_PATH = Path(
    "data/processed/interactions.parquet"
)

TEMPORAL_PATH = Path(
    "data/experimental/phase_2/"
    "temporal/interactions_itrs_temporal.parquet"
)

PHASE3_EDGES_PATH = Path(
    "data/experimental/phase_3/graph/edges.parquet"
)

PHASE3_NODES_PATH = Path(
    "data/experimental/phase_3/graph/nodes.parquet"
)

PHASE3_NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/model_ready/"
    "node_index.parquet"
)


# =============================================================================
# EXPECTED FROZEN SOURCE HASHES
# =============================================================================

EXPECTED_SOURCE_HASHES = {
    str(CANONICAL_PATH):
        (
            "03d02c48dc8bc67981fa27b66714641f6"
            "fdf1d8dd907348862517740ba77db8b"
        ),

    str(TEMPORAL_PATH):
        (
            "8846c044f27d1ab317fd9d05ff75442a7"
            "fd9759351c5906fc9e2483e755fcb97"
        ),

    str(PHASE3_EDGES_PATH):
        (
            "895871308de416b12d001550838cdb276"
            "3c87cd897774c6a55539a0ebb9a1ee2"
        ),

    str(PHASE3_NODES_PATH):
        (
            "84ed5bb49bb7ee3a4b40ee21fab1f493"
            "10ffb892707088ba537d1dc4fcc8c551"
        ),

    str(PHASE3_NODE_INDEX_PATH):
        (
            "e1c21502373ea1808544ad4eaaf4d6e7"
            "3c215a212258db462bda1c83423dc3d1"
        ),
}


# =============================================================================
# AUTHORITATIVE DERIVED-HASH REGISTERS
# =============================================================================

HASH_REGISTERS = [
    PHASE8_ROOT
    / "phase_8_0_1_post_endpoint_availability"
    / "phase_8_0_1_derived_artifact_sha256_V1.json",

    PHASE8_ROOT
    / "phase_8_0_2_candidate_confirmatory_structural_isolation"
    / "phase_8_0_2_derived_artifact_sha256_V1.json",

    PHASE8_ROOT
    / "phase_8_0_3_confirmatory_safe_graph_masks"
    / "phase_8_0_3_derived_artifact_sha256_V1.json",

    PHASE8_ROOT
    / "phase_8_0_4_rolling_origin_development_feasibility"
    / "phase_8_0_4_derived_artifact_sha256_V1.json",

    PHASE8_ROOT
    / "phase_8_0_5_founder_temporal_observability"
    / "phase_8_0_5_derived_artifact_sha256_V1.json",

    PHASE8_ROOT
    / "phase_8_0_6_dual_track_graph_policy"
    / "phase_8_0_6_derived_artifact_sha256_V1.json",

    PHASE8_ROOT
    / "phase_8_0_7_startup_origin_eligibility"
    / "phase_8_0_7_derived_artifact_sha256_V1.json",

    PHASE8_ROOT
    / "phase_8_0_8_evidence_based_startup_existence"
    / "phase_8_0_8_derived_artifact_sha256_V1.json",

    PHASE8_ROOT
    / "phase_8_0_9_origin_eligibility_candidate_universe"
    / "phase_8_0_9_derived_artifact_sha256_V5.json",

    PHASE8_ROOT
    / "phase_8_0_10_evaluation_protocol"
    / "phase_8_0_10_derived_artifact_sha256_V1.json",

    PHASE8_ROOT
    / "phase_8_0_11_evaluation_candidates"
    / "phase_8_0_11_derived_artifact_sha256_V1.json",

    PHASE8_ROOT
    / "phase_8_0_12_research_design_contract"
    / "phase_8_0_12_derived_artifact_sha256_V1.json",
]


# =============================================================================
# AUTHORITATIVE KEY ARTIFACTS
# =============================================================================

PHASE803_DIR = (
    PHASE8_ROOT
    / "phase_8_0_3_confirmatory_safe_graph_masks"
)

PHASE803_MASKS = (
    PHASE803_DIR
    / "phase_8_0_3_confirmatory_safe_graph_masks_V1.npz"
)


PHASE806_DIR = (
    PHASE8_ROOT
    / "phase_8_0_6_dual_track_graph_policy"
)

PHASE806_MASKS = (
    PHASE806_DIR
    / "phase_8_0_6_dual_track_graph_masks_V1.npz"
)

PHASE806_SUMMARY = (
    PHASE806_DIR
    / "phase_8_0_6_origin_graph_summary_V1.csv"
)

PHASE806_POLICY = (
    PHASE806_DIR
    / "phase_8_0_6_graph_information_policy_V1.json"
)


PHASE809_DIR = (
    PHASE8_ROOT
    / "phase_8_0_9_origin_eligibility_candidate_universe"
)

PHASE809_REGISTRY = (
    PHASE809_DIR
    / "phase_8_0_9_startup_existence_registry_V5.parquet"
)

PHASE809_CANDIDATES = (
    PHASE809_DIR
    / "phase_8_0_9_origin_candidate_universe_V5.parquet"
)

PHASE809_TARGETS = (
    PHASE809_DIR
    / "phase_8_0_9_origin_eligible_targets_V5.parquet"
)

PHASE809_EXCLUDED = (
    PHASE809_DIR
    / "phase_8_0_9_noneligible_target_register_V5.parquet"
)

PHASE809_SUMMARY = (
    PHASE809_DIR
    / "phase_8_0_9_origin_contract_summary_V5.csv"
)

PHASE809_CONTRACT = (
    PHASE809_DIR
    / "phase_8_0_9_origin_eligibility_contract_V5.json"
)


PHASE810_DIR = (
    PHASE8_ROOT
    / "phase_8_0_10_evaluation_protocol"
)

PHASE810_PROTOCOL = (
    PHASE810_DIR
    / "phase_8_0_10_evaluation_protocol_V1.json"
)

PHASE810_SELECTION = (
    PHASE810_DIR
    / "phase_8_0_10_model_selection_contract_V1.json"
)

PHASE810_FOLDS = (
    PHASE810_DIR
    / "phase_8_0_10_fold_role_register_V1.csv"
)


PHASE811_DIR = (
    PHASE8_ROOT
    / "phase_8_0_11_evaluation_candidates"
)

PHASE811_CASES = (
    PHASE811_DIR
    / "phase_8_0_11_evaluation_case_manifest_V1.parquet"
)

PHASE811_SUMMARY = (
    PHASE811_DIR
    / "phase_8_0_11_candidate_generation_summary_V1.csv"
)

PHASE811_CONTRACT = (
    PHASE811_DIR
    / "phase_8_0_11_candidate_runtime_contract_V1.json"
)

PHASE811_MANIFEST = (
    PHASE811_DIR
    / "phase_8_0_11_manifest_V1.json"
)

PHASE811_MATRICES = {
    "T57":
        PHASE811_DIR
        / "phase_8_0_11_t57_negative_node_indices_V1.npy",

    "T58":
        PHASE811_DIR
        / "phase_8_0_11_t58_negative_node_indices_V1.npy",

    "T59":
        PHASE811_DIR
        / "phase_8_0_11_t59_negative_node_indices_V1.npy",

    "APRIL_CONFIRMATORY":
        PHASE811_DIR
        / "phase_8_0_11_april_confirmatory_negative_node_indices_V1.npy",
}


PHASE812_DIR = (
    PHASE8_ROOT
    / "phase_8_0_12_research_design_contract"
)

PHASE812_RQ = (
    PHASE812_DIR
    / "phase_8_0_12_research_questions_V1.csv"
)

PHASE812_HYPOTHESES = (
    PHASE812_DIR
    / "phase_8_0_12_hypotheses_V1.csv"
)

PHASE812_BASELINES = (
    PHASE812_DIR
    / "phase_8_0_12_baseline_model_matrix_V1.csv"
)

PHASE812_ABLATIONS = (
    PHASE812_DIR
    / "phase_8_0_12_required_ablation_matrix_V1.csv"
)

PHASE812_ENDPOINTS = (
    PHASE812_DIR
    / "phase_8_0_12_endpoint_register_V1.csv"
)

PHASE812_STATS = (
    PHASE812_DIR
    / "phase_8_0_12_statistical_analysis_contract_V1.json"
)

PHASE812_CONTRACT = (
    PHASE812_DIR
    / "phase_8_0_12_research_design_contract_V1.json"
)

PHASE812_MANIFEST = (
    PHASE812_DIR
    / "phase_8_0_12_manifest_V1.json"
)


# =============================================================================
# EXPECTED FROZEN COUNTS
# =============================================================================

EXPECTED_GRAPH_COUNTS = {
    "T57": {
        "strict_temporal_edges": 60_144,
        "founder_proxy_edges": 153_034,
    },
    "T58": {
        "strict_temporal_edges": 61_358,
        "founder_proxy_edges": 154_392,
    },
    "T59": {
        "strict_temporal_edges": 62_554,
        "founder_proxy_edges": 155_736,
    },
    "APRIL_CONFIRMATORY": {
        "strict_temporal_edges": 63_964,
        "founder_proxy_edges": 157_444,
    },
}

EXPECTED_EVALUATION = {
    "T57": {
        "candidate_startups": 308_027,
        "eligible_targets": 21_924,
        "primary": 6_850,
    },
    "T58": {
        "candidate_startups": 308_854,
        "eligible_targets": 22_384,
        "primary": 6_872,
    },
    "T59": {
        "candidate_startups": 309_611,
        "eligible_targets": 22_650,
        "primary": 6_764,
    },
    "APRIL_CONFIRMATORY": {
        "candidate_startups": 311_266,
        "eligible_targets": 6_313,
        "primary": 1_690,
    },
}

EXPECTED_TOTAL_CASES = 73_271
EXPECTED_TOTAL_PRIMARY = 22_176
EXPECTED_NEGATIVES_PER_CASE = 99


# =============================================================================
# SUPERSEDED / FAILED-ATTEMPT ARTIFACTS
# =============================================================================
#
# These are preserved for provenance but are NOT authoritative inputs.
#
# V1 path may have been rewritten by the failed V2 attempt because V2
# inherited V1 output naming. It remains explicitly superseded.
# =============================================================================

SUPERSEDED_ARTIFACTS = [
    {
        "artifact":
            PHASE809_DIR
            / "phase_8_0_9_startup_existence_registry_V1.parquet",

        "status":
            "SUPERSEDED_FAILED_DEVELOPMENT_ARTIFACT",

        "reason":
            (
                "Produced before successful Phase-8.0.9 V5 freeze; "
                "V1/V2 execution path was not authoritative."
            ),
    },
    {
        "artifact":
            PHASE809_DIR
            / "phase_8_0_9_startup_existence_registry_V3.parquet",

        "status":
            "SUPERSEDED_DIAGNOSTIC_ARTIFACT",

        "reason":
            (
                "V3 stopped on datetime-rule disagreement before "
                "contract freeze."
            ),
    },
    {
        "artifact":
            PHASE809_DIR
            / "phase_8_0_9_startup_existence_registry_V4.parquet",

        "status":
            "SUPERSEDED_DIAGNOSTIC_ARTIFACT",

        "reason":
            (
                "V4 deliberately stopped after exposing the three "
                "ancient-date datetime-resolution disagreements."
            ),
    },
]


# =============================================================================
# CLOSURE OUTPUTS
# =============================================================================

AUTHORITATIVE_REGISTER_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_13_authoritative_artifact_register_V1.csv"
)

SUPERSEDED_REGISTER_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_13_superseded_artifact_register_V1.csv"
)

CLOSURE_EVIDENCE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_13_phase_closure_evidence_V1.json"
)

CLOSURE_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_13_phase_closure_summary_V1.md"
)

CLOSURE_MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_13_closure_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_13_derived_artifact_sha256_V1.json"
)


# =============================================================================
# HELPERS
# =============================================================================

SHA256_RE = re.compile(
    r"^[0-9a-f]{64}$"
)


def separator(
    char="=",
    width=118,
):

    print(
        char * width
    )


def require(
    condition,
    message,
):

    if not condition:

        raise ValueError(
            message
        )


def sha256_file(
    path,
    chunk_size=1024 * 1024,
):

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:

                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def read_json(
    path,
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def write_json(
    path,
    obj,
):

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


def verify_flat_hash_register(
    register_path,
):

    data = read_json(
        register_path
    )

    require(
        isinstance(
            data,
            dict,
        ),
        (
            f"Hash register is not a JSON object: "
            f"{register_path}"
        ),
    )

    verified = []

    for raw_path, expected_hash in (
        data.items()
    ):

        if not (
            isinstance(
                raw_path,
                str,
            )
            and
            isinstance(
                expected_hash,
                str,
            )
            and
            SHA256_RE.fullmatch(
                expected_hash
            )
        ):

            continue

        artifact_path = Path(
            raw_path
        )

        require(
            artifact_path.exists(),
            (
                "Hash register references missing artifact: "
                f"{artifact_path}"
            ),
        )

        observed_hash = sha256_file(
            artifact_path
        )

        require(
            observed_hash
            ==
            expected_hash,
            (
                "Derived artifact fingerprint drift:\n"
                f"  artifact: {artifact_path}\n"
                f"  expected: {expected_hash}\n"
                f"  observed: {observed_hash}"
            ),
        )

        verified.append(
            {
                "artifact":
                    str(
                        artifact_path
                    ),

                "sha256":
                    observed_hash,
            }
        )

    require(
        len(
            verified
        )
        > 0,
        (
            "No path->SHA256 entries were verified "
            f"from {register_path}"
        ),
    )

    return verified


# =============================================================================
# MAIN
# =============================================================================

def main():

    separator()

    print(
        "PHASE 8.0.13 — "
        "EVALUATION RESET & RESEARCH DESIGN "
        "CLOSURE AUDIT V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "INTEGRITY / PROVENANCE / PHASE CLOSURE ONLY"
    )

    print("Model constructed:           NO")
    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Training performed:          NO")
    print("Inference executed:          NO")
    print("Ranking metrics scored:      NO")
    print("Negatives regenerated:       NO")
    print("Model selection performed:   NO")
    print("April performance inspected: NO")

    # =========================================================================
    # 1. Frozen source fingerprint gate
    # =========================================================================

    separator("-")

    print(
        "8.0.13.1 — FROZEN SOURCE "
        "FINGERPRINT GATE"
    )

    separator("-")

    for raw_path, expected_hash in (
        EXPECTED_SOURCE_HASHES.items()
    ):

        path = Path(
            raw_path
        )

        require(
            path.exists(),
            f"Missing frozen source: {path}",
        )

        observed_hash = sha256_file(
            path
        )

        require(
            observed_hash
            ==
            expected_hash,
            (
                f"Frozen source drift: {path}\n"
                f"expected={expected_hash}\n"
                f"observed={observed_hash}"
            ),
        )

        print(
            f"PASS  {path}"
        )

        print(
            f"      {observed_hash}"
        )

    # =========================================================================
    # 2. Verify all Phase-8.0 authoritative hash registers
    # =========================================================================

    separator("-")

    print(
        "8.0.13.2 — AUTHORITATIVE "
        "DERIVED-HASH CHAIN"
    )

    separator("-")

    hash_verified_rows = []

    for register in HASH_REGISTERS:

        require(
            register.exists(),
            (
                "Missing authoritative hash register: "
                f"{register}"
            ),
        )

        verified = (
            verify_flat_hash_register(
                register
            )
        )

        hash_verified_rows.extend(
            [
                {
                    "hash_register":
                        str(
                            register
                        ),

                    **row,
                }
                for row
                in verified
            ]
        )

        print(
            f"PASS  {register}"
        )

        print(
            f"      verified artifacts: "
            f"{len(verified):,}"
        )

    print()

    print(
        f"Total derived artifacts verified "
        f"through authoritative hash registers: "
        f"{len(hash_verified_rows):,}"
    )

    # =========================================================================
    # 3. April direct-pair graph safety
    # =========================================================================

    separator("-")

    print(
        "8.0.13.3 — APRIL DIRECT-PAIR "
        "GRAPH SAFETY"
    )

    separator("-")

    require(
        PHASE803_MASKS.exists(),
        (
            "Missing Phase-8.0.3 "
            "confirmatory-safe masks."
        ),
    )

    masks803 = np.load(
        PHASE803_MASKS,
        allow_pickle=False,
    )

    expected_803_keys = {
        "direct_pair_exclusion",
        "core_confirmatory_safe",
        "founder_only_confirmatory_safe",
        "acquisition_only_confirmatory_safe",
    }

    require(
        set(
            masks803.files
        )
        ==
        expected_803_keys,
        (
            "Phase-8.0.3 mask-key drift. "
            f"Observed: {masks803.files}"
        ),
    )

    count_803 = {
        key:
            int(
                np.asarray(
                    masks803[
                        key
                    ],
                    dtype=bool,
                ).sum()
            )
        for key
        in masks803.files
    }

    require(
        count_803[
            "direct_pair_exclusion"
        ]
        ==
        40,
        "Phase-8.0.3 exclusion count drift.",
    )

    require(
        count_803[
            "core_confirmatory_safe"
        ]
        ==
        158_778,
        "Phase-8.0.3 safe-core count drift.",
    )

    require(
        count_803[
            "founder_only_confirmatory_safe"
        ]
        ==
        94_814,
        "Phase-8.0.3 founder-safe count drift.",
    )

    require(
        count_803[
            "acquisition_only_confirmatory_safe"
        ]
        ==
        63_964,
        "Phase-8.0.3 acquisition-safe count drift.",
    )

    print(
        "Direct target-pair exclusion edges: 40 PASS"
    )

    print(
        "Confirmatory-safe core:         "
        "158,778 PASS"
    )

    print(
        "Confirmatory-safe founder-only: "
        "94,814 PASS"
    )

    print(
        "Confirmatory-safe acquisition:  "
        "63,964 PASS"
    )

    # =========================================================================
    # 4. Origin-specific graph policy
    # =========================================================================

    separator("-")

    print(
        "8.0.13.4 — ORIGIN-SPECIFIC "
        "GRAPH POLICY"
    )

    separator("-")

    require(
        PHASE806_MASKS.exists(),
        "Missing Phase-8.0.6 masks.",
    )

    require(
        PHASE806_SUMMARY.exists(),
        "Missing Phase-8.0.6 graph summary.",
    )

    require(
        PHASE806_POLICY.exists(),
        "Missing Phase-8.0.6 graph policy.",
    )

    graph_summary = pd.read_csv(
        PHASE806_SUMMARY
    )

    graph_policy = read_json(
        PHASE806_POLICY
    )

    require(
        graph_policy[
            "strict_temporal_track"
        ][
            "status"
        ]
        ==
        "FROZEN",
        "Strict graph track is not frozen.",
    )

    require(
        graph_policy[
            "founder_proxy_track"
        ][
            "status"
        ]
        ==
        "FROZEN",
        "Founder graph track is not frozen.",
    )

    masks806 = np.load(
        PHASE806_MASKS,
        allow_pickle=False,
    )

    for origin, expected in (
        EXPECTED_GRAPH_COUNTS.items()
    ):

        row = graph_summary.loc[
            graph_summary[
                "origin"
            ]
            ==
            origin
        ]

        require(
            len(
                row
            )
            ==
            1,
            (
                f"{origin}: expected exactly "
                "one graph-summary row."
            ),
        )

        row = row.iloc[0]

        for field, expected_count in (
            expected.items()
        ):

            observed_count = int(
                row[
                    field
                ]
            )

            require(
                observed_count
                ==
                expected_count,
                (
                    f"{origin}: {field} drift; "
                    f"expected {expected_count:,}, "
                    f"observed {observed_count:,}"
                ),
            )

        strict_key = (
            f"{origin}_strict_temporal"
        )

        founder_key = (
            f"{origin}_founder_proxy"
        )

        require(
            strict_key
            in
            masks806.files,
            f"Missing mask: {strict_key}",
        )

        require(
            founder_key
            in
            masks806.files,
            f"Missing mask: {founder_key}",
        )

        strict_sum = int(
            np.asarray(
                masks806[
                    strict_key
                ],
                dtype=bool,
            ).sum()
        )

        founder_sum = int(
            np.asarray(
                masks806[
                    founder_key
                ],
                dtype=bool,
            ).sum()
        )

        require(
            strict_sum
            ==
            expected[
                "strict_temporal_edges"
            ],
            (
                f"{origin}: strict mask "
                "count drift."
            ),
        )

        require(
            founder_sum
            ==
            expected[
                "founder_proxy_edges"
            ],
            (
                f"{origin}: founder mask "
                "count drift."
            ),
        )

        print(
            f"{origin}: strict={strict_sum:,} | "
            f"founder-track={founder_sum:,} PASS"
        )

    # =========================================================================
    # 5. Startup origin-eligibility freeze
    # =========================================================================

    separator("-")

    print(
        "8.0.13.5 — ORIGIN-ELIGIBILITY "
        "AND TARGET FREEZE"
    )

    separator("-")

    for path in [
        PHASE809_REGISTRY,
        PHASE809_CANDIDATES,
        PHASE809_TARGETS,
        PHASE809_EXCLUDED,
        PHASE809_SUMMARY,
        PHASE809_CONTRACT,
    ]:

        require(
            path.exists(),
            f"Missing Phase-8.0.9 artifact: {path}",
        )

    contract809 = read_json(
        PHASE809_CONTRACT
    )

    require(
        contract809.get(
            "status"
        )
        ==
        "FROZEN",
        "Phase-8.0.9 contract not frozen.",
    )

    registry809 = pd.read_parquet(
        PHASE809_REGISTRY
    )

    require(
        len(
            registry809
        )
        ==
        311_589,
        "Startup existence registry row-count drift.",
    )

    summary809 = pd.read_csv(
        PHASE809_SUMMARY
    )

    for origin, expected in (
        EXPECTED_EVALUATION.items()
    ):

        row = summary809.loc[
            summary809[
                "origin"
            ]
            ==
            origin
        ]

        require(
            len(
                row
            )
            ==
            1,
            f"{origin}: missing Phase-8.0.9 summary row.",
        )

        row = row.iloc[0]

        require(
            int(
                row[
                    "candidate_startups"
                ]
            )
            ==
            expected[
                "candidate_startups"
            ],
            f"{origin}: candidate count drift.",
        )

        require(
            int(
                row[
                    "origin_eligible_targets"
                ]
            )
            ==
            expected[
                "eligible_targets"
            ],
            f"{origin}: target count drift.",
        )

        require(
            int(
                row[
                    "eligible_primary_events"
                ]
            )
            ==
            expected[
                "primary"
            ],
            f"{origin}: primary count drift.",
        )

        print(
            f"{origin}: candidates="
            f"{expected['candidate_startups']:,} | "
            f"eligible={expected['eligible_targets']:,} | "
            f"primary={expected['primary']:,} PASS"
        )

    # =========================================================================
    # 6. Evaluation protocol freeze
    # =========================================================================

    separator("-")

    print(
        "8.0.13.6 — TEMPORAL / "
        "MODEL-SELECTION PROTOCOL"
    )

    separator("-")

    protocol810 = read_json(
        PHASE810_PROTOCOL
    )

    selection810 = read_json(
        PHASE810_SELECTION
    )

    folds810 = pd.read_csv(
        PHASE810_FOLDS
    )

    require(
        protocol810.get(
            "status"
        )
        ==
        "FROZEN",
        "Phase-8.0.10 protocol not frozen.",
    )

    require(
        selection810.get(
            "status"
        )
        ==
        "FROZEN",
        "Phase-8.0.10 selection contract not frozen.",
    )

    require(
        selection810[
            "development_folds"
        ]
        ==
        [
            "T57",
            "T58",
            "T59",
        ],
        "Development fold set drift.",
    )

    require(
        selection810[
            "T60_used_for_model_selection"
        ]
        is False,
        "T60 must remain forbidden for model selection.",
    )

    require(
        selection810[
            "April_used_for_model_selection"
        ]
        is False,
        "April must remain forbidden for model selection.",
    )

    require(
        int(
            selection810[
                "maximum_training_epochs"
            ]
        )
        ==
        20,
        "Maximum epoch contract drift.",
    )

    expected_training_segments = {
        "T57": 56,
        "T58": 57,
        "T59": 58,
    }

    for fold, last_segment in (
        expected_training_segments.items()
    ):

        row = folds810.loc[
            folds810[
                "fold"
            ]
            ==
            fold
        ]

        require(
            len(
                row
            )
            ==
            1,
            f"{fold}: missing fold register row.",
        )

        row = row.iloc[0]

        require(
            int(
                row[
                    "training_segment_first"
                ]
            )
            ==
            1,
            f"{fold}: training start drift.",
        )

        require(
            int(
                row[
                    "training_segment_last"
                ]
            )
            ==
            last_segment,
            f"{fold}: training end drift.",
        )

    print(
        "Development folds: T57 / T58 / T59 PASS"
    )

    print(
        "T60 Phase-8 model selection: FORBIDDEN PASS"
    )

    print(
        "April model selection:       FORBIDDEN PASS"
    )

    print(
        "Maximum epochs:              20 PASS"
    )

    # =========================================================================
    # 7. Frozen evaluation candidate lists
    # =========================================================================

    separator("-")

    print(
        "8.0.13.7 — FROZEN 100-CANDIDATE "
        "EVALUATION CASES"
    )

    separator("-")

    cases811 = pd.read_parquet(
        PHASE811_CASES,
        columns=[
            "origin",
            "interaction_id",
            "is_primary_stratum",
            "negative_count",
        ],
    )

    require(
        len(
            cases811
        )
        ==
        EXPECTED_TOTAL_CASES,
        "Evaluation case total drift.",
    )

    require(
        int(
            cases811[
                "is_primary_stratum"
            ].sum()
        )
        ==
        EXPECTED_TOTAL_PRIMARY,
        "Primary evaluation case total drift.",
    )

    require(
        (
            cases811[
                "negative_count"
            ]
            ==
            EXPECTED_NEGATIVES_PER_CASE
        ).all(),
        "At least one case does not have 99 negatives.",
    )

    require(
        not cases811[
            [
                "origin",
                "interaction_id",
            ]
        ]
        .duplicated()
        .any(),
        "Duplicate evaluation case detected.",
    )

    contract811 = read_json(
        PHASE811_CONTRACT
    )

    manifest811 = read_json(
        PHASE811_MANIFEST
    )

    require(
        contract811.get(
            "status"
        )
        ==
        "FROZEN",
        "Phase-8.0.11 runtime contract not frozen.",
    )

    require(
        manifest811.get(
            "status"
        )
        ==
        "FROZEN",
        "Phase-8.0.11 manifest not frozen.",
    )

    require(
        contract811[
            "resampling_after_freeze"
        ]
        is False,
        "Post-freeze candidate resampling became allowed.",
    )

    for origin, expected in (
        EXPECTED_EVALUATION.items()
    ):

        matrix_path = (
            PHASE811_MATRICES[
                origin
            ]
        )

        require(
            matrix_path.exists(),
            f"Missing negative matrix: {matrix_path}",
        )

        matrix = np.load(
            matrix_path,
            mmap_mode="r",
            allow_pickle=False,
        )

        expected_shape = (
            expected[
                "eligible_targets"
            ],
            EXPECTED_NEGATIVES_PER_CASE,
        )

        require(
            matrix.shape
            ==
            expected_shape,
            (
                f"{origin}: negative matrix shape drift; "
                f"expected {expected_shape}, "
                f"observed {matrix.shape}"
            ),
        )

        require(
            np.issubdtype(
                matrix.dtype,
                np.integer,
            ),
            (
                f"{origin}: negative matrix "
                "dtype is not integer."
            ),
        )

        print(
            f"{origin}: matrix "
            f"{matrix.shape[0]:,} x "
            f"{matrix.shape[1]} PASS"
        )

    print()

    print(
        f"Total frozen cases:   "
        f"{len(cases811):,} PASS"
    )

    print(
        f"Total primary cases:  "
        f"{int(cases811['is_primary_stratum'].sum()):,} PASS"
    )

    # =========================================================================
    # 8. Research design / preregistration freeze
    # =========================================================================

    separator("-")

    print(
        "8.0.13.8 — RESEARCH DESIGN "
        "PREREGISTRATION"
    )

    separator("-")

    rq = pd.read_csv(
        PHASE812_RQ
    )

    hypotheses = pd.read_csv(
        PHASE812_HYPOTHESES
    )

    baselines = pd.read_csv(
        PHASE812_BASELINES
    )

    ablations = pd.read_csv(
        PHASE812_ABLATIONS
    )

    endpoints = pd.read_csv(
        PHASE812_ENDPOINTS
    )

    stats812 = read_json(
        PHASE812_STATS
    )

    contract812 = read_json(
        PHASE812_CONTRACT
    )

    require(
        set(
            rq[
                "rq"
            ]
        )
        ==
        {
            "RQ1",
            "RQ2",
            "RQ3",
            "RQ4",
            "RQ5",
        },
        "Research-question register drift.",
    )

    require(
        set(
            hypotheses[
                "hypothesis"
            ]
        )
        ==
        {
            "H1",
            "H2",
            "H3",
            "H4",
            "H5",
        },
        "Hypothesis register drift.",
    )

    require(
        set(
            baselines[
                "model_id"
            ]
        )
        ==
        {
            "B0",
            "B1",
            "M1",
            "M2",
        },
        "Baseline/model matrix drift.",
    )

    require(
        set(
            ablations[
                "ablation_id"
            ]
        )
        ==
        {
            "A0",
            "A1",
            "A2",
            "A3",
            "A4",
        },
        "Ablation matrix drift.",
    )

    require(
        (
            ablations[
                "April_scoring_allowed"
            ]
            ==
            False
        ).all(),
        "At least one development ablation permits April scoring.",
    )

    e1 = endpoints.loc[
        endpoints[
            "endpoint_id"
        ]
        ==
        "E1"
    ]

    require(
        len(
            e1
        )
        ==
        1,
        "Primary endpoint E1 missing.",
    )

    e1 = e1.iloc[0]

    require(
        e1[
            "role"
        ]
        ==
        "PRIMARY_CONFIRMATORY",
        "E1 role drift.",
    )

    require(
        e1[
            "metric"
        ]
        ==
        "NDCG@10",
        "Primary endpoint metric drift.",
    )

    require(
        stats812.get(
            "status"
        )
        ==
        "FROZEN",
        "Statistical analysis contract not frozen.",
    )

    require(
        stats812[
            "primary_uncertainty"
        ][
            "method"
        ]
        ==
        (
            "two-way investor/startup "
            "clustered bootstrap"
        ),
        "Primary uncertainty method drift.",
    )

    require(
        int(
            stats812[
                "primary_uncertainty"
            ][
                "replicates"
            ]
        )
        ==
        10_000,
        "Bootstrap replicate count drift.",
    )

    require(
        contract812.get(
            "status"
        )
        ==
        "FROZEN",
        "Research design contract not frozen.",
    )

    require(
        contract812[
            "confirmatory_primary_cases"
        ]
        ==
        1_690,
        "April primary case count drift.",
    )

    require(
        contract812[
            "April_ablation_scoring_allowed"
        ]
        is False,
        "April ablation scoring became allowed.",
    )

    require(
        contract812[
            "new_model_training_started"
        ]
        is False,
        "Research-design artifact claims training started.",
    )

    require(
        contract812[
            "April_performance_inspected"
        ]
        is False,
        "Research-design artifact claims April was inspected.",
    )

    print(
        "RQ1-RQ5:                    FROZEN PASS"
    )

    print(
        "H1-H5:                      FROZEN PASS"
    )

    print(
        "B0/B1/M1/M2:                FROZEN PASS"
    )

    print(
        "A0-A4:                      FROZEN PASS"
    )

    print(
        "Primary endpoint:            April NDCG@10 PASS"
    )

    print(
        "Primary uncertainty:         "
        "two-way clustered bootstrap PASS"
    )

    print(
        "April ablation search:       FORBIDDEN PASS"
    )

    # =========================================================================
    # 9. Superseded provenance artifacts
    # =========================================================================

    separator("-")

    print(
        "8.0.13.9 — SUPERSEDED "
        "PHASE-8.0.9 PROVENANCE"
    )

    separator("-")

    superseded_rows = []

    for item in (
        SUPERSEDED_ARTIFACTS
    ):

        path = item[
            "artifact"
        ]

        exists = path.exists()

        observed_hash = (
            sha256_file(
                path
            )
            if exists
            else None
        )

        superseded_rows.append(
            {
                "artifact":
                    str(
                        path
                    ),

                "exists":
                    exists,

                "sha256":
                    observed_hash,

                "status":
                    item[
                        "status"
                    ],

                "reason":
                    item[
                        "reason"
                    ],

                "authoritative":
                    False,
            }
        )

        print(
            f"{'FOUND' if exists else 'NOT FOUND'}  "
            f"{path.name}"
        )

        print(
            f"  status: {item['status']}"
        )

    superseded_register = pd.DataFrame(
        superseded_rows
    )

    # =========================================================================
    # 10. Construct authoritative artifact register
    # =========================================================================

    separator("-")

    print(
        "8.0.13.10 — AUTHORITATIVE "
        "ARTIFACT REGISTER"
    )

    separator("-")

    authoritative_paths = [
        *HASH_REGISTERS,

        PHASE803_MASKS,

        PHASE806_MASKS,
        PHASE806_SUMMARY,
        PHASE806_POLICY,

        PHASE809_REGISTRY,
        PHASE809_CANDIDATES,
        PHASE809_TARGETS,
        PHASE809_EXCLUDED,
        PHASE809_SUMMARY,
        PHASE809_CONTRACT,

        PHASE810_PROTOCOL,
        PHASE810_SELECTION,
        PHASE810_FOLDS,

        PHASE811_CASES,
        PHASE811_SUMMARY,
        PHASE811_CONTRACT,
        PHASE811_MANIFEST,

        *PHASE811_MATRICES.values(),

        PHASE812_RQ,
        PHASE812_HYPOTHESES,
        PHASE812_BASELINES,
        PHASE812_ABLATIONS,
        PHASE812_ENDPOINTS,
        PHASE812_STATS,
        PHASE812_CONTRACT,
        PHASE812_MANIFEST,
    ]

    # Remove duplicate paths while preserving order.
    authoritative_paths = list(
        dict.fromkeys(
            authoritative_paths
        )
    )

    authoritative_rows = []

    superseded_path_set = {
        str(
            item[
                "artifact"
            ]
        )
        for item
        in SUPERSEDED_ARTIFACTS
    }

    for path in (
        authoritative_paths
    ):

        require(
            path.exists(),
            (
                "Missing authoritative Phase-8.0 "
                f"artifact: {path}"
            ),
        )

        require(
            str(
                path
            )
            not in
            superseded_path_set,
            (
                "Superseded artifact entered "
                f"authoritative register: {path}"
            ),
        )

        relative_parts = path.parts

        phase_dir = next(
            (
                part
                for part
                in relative_parts
                if part.startswith(
                    "phase_8_0_"
                )
            ),
            "external_source",
        )

        authoritative_rows.append(
            {
                "subphase":
                    phase_dir,

                "artifact":
                    str(
                        path
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),

                "authoritative":
                    True,
            }
        )

    authoritative_register = pd.DataFrame(
        authoritative_rows
    )

    print(
        f"Authoritative registered artifacts: "
        f"{len(authoritative_register):,}"
    )

    # =========================================================================
    # 11. Persist closure evidence
    # =========================================================================

    separator("-")

    print(
        "8.0.13.11 — PERSIST "
        "PHASE-8.0 CLOSURE EVIDENCE"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    authoritative_register.to_csv(
        AUTHORITATIVE_REGISTER_OUTPUT,
        index=False,
    )

    superseded_register.to_csv(
        SUPERSEDED_REGISTER_OUTPUT,
        index=False,
    )

    closure_evidence = {
        "schema_version":
            "PHASE_8_0_13_CLOSURE_EVIDENCE_V1",

        "status":
            "COMPLETE",

        "phase":
            "8.0",

        "phase_name":
            (
                "Evaluation Reset & "
                "Research Design Freeze"
            ),

        "scientific_status":
            "CLOSED",

        "frozen_source_integrity":
            "PASS",

        "derived_hash_chain_integrity":
            "PASS",

        "april_confirmatory_window": {
            "prediction_origin":
                "2026-03-31",

            "evaluation_start":
                "2026-04-01",

            "evaluation_end":
                "2026-04-30",

            "eligible_cases":
                6_313,

            "primary_cases":
                1_690,

            "performance_status":
                "UNTOUCHED",
        },

        "development_protocol": {
            "folds": [
                "T57",
                "T58",
                "T59",
            ],

            "model_selection_metric":
                (
                    "equal-weight mean primary-stratum "
                    "NDCG@10 across T57/T58/T59"
                ),

            "maximum_epochs":
                20,

            "T60_model_selection":
                "FORBIDDEN",

            "T60_final_fit_history":
                (
                    "ALLOWED only after "
                    "model-selection freeze"
                ),
        },

        "startup_origin_eligibility": {
            "status":
                "FROZEN",

            "rule":
                (
                    "existence proven by founded_on "
                    "OR pre-origin investment "
                    "OR pre-origin timestamped "
                    "acquisition evidence"
                ),
        },

        "graph_policy": {
            "strict_temporal_track":
                "FROZEN",

            "founder_track":
                "FROZEN",

            "founder_scientific_interpretation":
                (
                    "temporally eligible shared-founder "
                    "side information; database observation "
                    "time is not claimed"
                ),

            "April_direct_target_pair_edges_after_mask":
                0,
        },

        "evaluation_candidate_contract": {
            "status":
                "FROZEN",

            "candidate_count_per_case":
                100,

            "positive_count":
                1,

            "negative_count":
                99,

            "total_cases":
                EXPECTED_TOTAL_CASES,

            "total_primary_cases":
                EXPECTED_TOTAL_PRIMARY,

            "resampling_after_freeze":
                False,
        },

        "research_design": {
            "research_questions":
                [
                    "RQ1",
                    "RQ2",
                    "RQ3",
                    "RQ4",
                    "RQ5",
                ],

            "hypotheses":
                [
                    "H1",
                    "H2",
                    "H3",
                    "H4",
                    "H5",
                ],

            "baseline_models":
                [
                    "B0",
                    "B1",
                    "M1",
                    "M2",
                ],

            "required_ablations":
                [
                    "A0",
                    "A1",
                    "A2",
                    "A3",
                    "A4",
                ],

            "primary_endpoint":
                "April primary-stratum NDCG@10",

            "primary_comparison":
                "M* minus B0",

            "primary_uncertainty":
                (
                    "10,000-replicate two-way "
                    "investor/startup clustered bootstrap"
                ),

            "April_models_allowed":
                (
                    "B0 plus one single final "
                    "selected M* model"
                ),

            "April_ablation_search":
                "FORBIDDEN",
        },

        "operations_performed_in_phase_8_0_13": {
            "model_constructed":
                False,

            "model_loaded":
                False,

            "checkpoint_loaded":
                False,

            "training":
                False,

            "inference":
                False,

            "ranking_scoring":
                False,

            "negative_generation":
                False,

            "negative_resampling":
                False,

            "model_selection":
                False,

            "April_performance_inspection":
                False,
        },

        "authoritative_artifact_count":
            len(
                authoritative_register
            ),

        "superseded_artifact_count":
            len(
                superseded_register
            ),

        "next_scientific_stage":
            (
                "Phase 8.1 — inductive startup "
                "representation development under "
                "the frozen Phase-8.0 protocol"
            ),
    }

    write_json(
        CLOSURE_EVIDENCE_OUTPUT,
        closure_evidence,
    )

    closure_summary = f"""# Phase 8.0 — Evaluation Reset & Research Design Freeze

## Status

**CLOSED**

Phase 8.0 was completed before construction or training of the new
Phase-8 model.

## Confirmatory holdout

- Prediction origin: 2026-03-31
- Confirmatory period: 2026-04-01 through 2026-04-30
- Origin-eligible evaluation cases: 6,313
- Primary thesis cases: 1,690
- April ranking performance inspected: NO

## Development protocol

Model development and selection are restricted to rolling-origin folds:

- T57
- T58
- T59

T60 is prohibited from Phase-8 model selection. It may enter the
final training history only after architecture, hyperparameters, epoch
count, relation configuration, and analysis plan are frozen.

## Primary model-selection target

Equal-weight mean primary-stratum NDCG@10 across T57, T58, and T59.

Maximum candidate epoch: 20.

## Startup origin eligibility

A startup is eligible at an origin when existence is established by at
least one audited pre-origin evidence source:

1. founded_on;
2. canonical investment history;
3. timestamped acquisition evidence.

Unresolved existence is not interpreted as proven nonexistence.

## Graph policy

Two graph-information tracks are frozen:

1. strict timestamped acquisition graph;
2. temporally eligible shared-founder plus acquisition graph.

Shared-founder eligibility uses company founding dates as semantic
entity-existence constraints. The date Crunchbase first recorded the
founder relationship is not claimed to be known.

Direct target-pair structural leakage is masked.

## Evaluation candidates

Every frozen evaluation case contains:

- 1 focal positive;
- 99 immutable uniformly sampled negatives.

All models must use identical candidate lists.

Total frozen cases: 73,271.

Total primary cases: 22,176.

Negative resampling after freeze is forbidden.

## Confirmatory research question

The primary confirmatory question is whether one single Phase-8 model
selected using T57/T58/T59 improves April primary-stratum NDCG@10
relative to the reproduced ITRS baseline scored under the same
Phase-8 evaluation protocol.

## Primary inference

Primary estimand:

mean paired case-level NDCG@10 difference, M* minus B0.

Primary uncertainty:

10,000-replicate two-way investor/startup clustered bootstrap.

April ablation search is forbidden.

## Provenance

Failed / diagnostic Phase-8.0.9 artifacts are preserved but explicitly
excluded from the authoritative evidence chain. Phase-8.0.9 V5 is the
authoritative origin-eligibility freeze.

## Phase transition

Phase 8.0 is closed.

The next scientific stage is model development under the frozen
evaluation contract.
"""

    CLOSURE_SUMMARY_OUTPUT.write_text(
        closure_summary,
        encoding="utf-8",
    )

    # =========================================================================
    # 12. Closure manifest and final hashes
    # =========================================================================

    separator("-")

    print(
        "8.0.13.12 — CLOSURE MANIFEST "
        "AND FINAL FINGERPRINTS"
    )

    separator("-")

    closure_manifest = {
        "schema_version":
            "PHASE_8_0_13_CLOSURE_MANIFEST_V1",

        "status":
            "COMPLETE",

        "phase_8_0_status":
            "CLOSED",

        "authoritative_artifact_register":
            str(
                AUTHORITATIVE_REGISTER_OUTPUT
            ),

        "superseded_artifact_register":
            str(
                SUPERSEDED_REGISTER_OUTPUT
            ),

        "closure_evidence":
            str(
                CLOSURE_EVIDENCE_OUTPUT
            ),

        "closure_summary":
            str(
                CLOSURE_SUMMARY_OUTPUT
            ),

        "frozen_source_hashes":
            EXPECTED_SOURCE_HASHES,

        "authoritative_derived_hash_registers":
            [
                str(
                    path
                )
                for path
                in HASH_REGISTERS
            ],

        "phase_8_0_9_authoritative_version":
            "V5",

        "April_performance_inspected":
            False,

        "new_model_training_started":
            False,

        "closure_gate":
            "PASS",
    }

    write_json(
        CLOSURE_MANIFEST_OUTPUT,
        closure_manifest,
    )

    final_hashes = {
        str(
            AUTHORITATIVE_REGISTER_OUTPUT
        ):
            sha256_file(
                AUTHORITATIVE_REGISTER_OUTPUT
            ),

        str(
            SUPERSEDED_REGISTER_OUTPUT
        ):
            sha256_file(
                SUPERSEDED_REGISTER_OUTPUT
            ),

        str(
            CLOSURE_EVIDENCE_OUTPUT
        ):
            sha256_file(
                CLOSURE_EVIDENCE_OUTPUT
            ),

        str(
            CLOSURE_SUMMARY_OUTPUT
        ):
            sha256_file(
                CLOSURE_SUMMARY_OUTPUT
            ),

        str(
            CLOSURE_MANIFEST_OUTPUT
        ):
            sha256_file(
                CLOSURE_MANIFEST_OUTPUT
            ),
    }

    write_json(
        HASH_OUTPUT,
        final_hashes,
    )

    print(
        f"Closure artifacts fingerprinted: "
        f"{len(final_hashes):,}"
    )

    # =========================================================================
    # FINAL STATUS
    # =========================================================================

    separator()

    print(
        "PHASE 8.0.13 CLOSURE AUDIT RESULT"
    )

    separator()

    print()

    print(
        "Frozen source fingerprints:             PASS"
    )

    print(
        "Authoritative derived-hash chain:       PASS"
    )

    print(
        "April structural leakage protection:   PASS"
    )

    print(
        "Origin-specific graph policy:           PASS"
    )

    print(
        "Startup origin-eligibility contract:    PASS"
    )

    print(
        "Rolling-origin selection protocol:      PASS"
    )

    print(
        "Frozen evaluation candidate lists:      PASS"
    )

    print(
        "Research-design preregistration:        PASS"
    )

    print(
        "Superseded artifact isolation:          PASS"
    )

    print(
        "April performance inspected:            NO"
    )

    print(
        "New model training performed:           NO"
    )

    print()

    print(
        "PHASE 8.0 — EVALUATION RESET & "
        "RESEARCH DESIGN FREEZE: CLOSED"
    )

    print()

    print(
        "Next scientific stage:"
    )

    print(
        "Phase 8.1 — inductive startup "
        "representation development under "
        "the frozen Phase-8.0 protocol."
    )

    print()

    print(
        f"Closure outputs written to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()