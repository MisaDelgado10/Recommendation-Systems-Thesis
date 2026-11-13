from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.0.12 — FREEZE RESEARCH DESIGN CONTRACT V1
# =============================================================================
#
# Scientific role:
#
#   OUTCOME-BLIND THESIS RESEARCH-DESIGN / ANALYSIS FREEZE
#
# PURPOSE
# -------
#
# Close Phase 8.0 by freezing, before any new model development:
#
#   - research questions;
#   - hypotheses;
#   - baseline/model-family matrix;
#   - required relation/trend ablations;
#   - primary and secondary endpoints;
#   - confirmatory statistical analysis plan;
#   - April access restrictions.
#
# IMPORTANT
# ---------
#
# April 2026 remains untouched.
#
# Only TWO models may enter the primary April confirmatory comparison:
#
#   B0 = reproduced ITRS under the Phase-8 protocol
#   M* = one final model selected using T57/T58/T59 only
#
# Development ablations are NOT used to search April results.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

PHASE810_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_10_evaluation_protocol"
)

PHASE810_PROTOCOL = (
    PHASE810_DIR
    / "phase_8_0_10_evaluation_protocol_V1.json"
)

PHASE810_SELECTION = (
    PHASE810_DIR
    / "phase_8_0_10_model_selection_contract_V1.json"
)


PHASE811_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_11_evaluation_candidates"
)

PHASE811_SUMMARY = (
    PHASE811_DIR
    / "phase_8_0_11_candidate_generation_summary_V1.csv"
)

PHASE811_COLLISION = (
    PHASE811_DIR
    / "phase_8_0_11_same_fold_positive_collision_summary_V1.csv"
)

PHASE811_CONTRACT = (
    PHASE811_DIR
    / "phase_8_0_11_candidate_runtime_contract_V1.json"
)

PHASE811_CASES = (
    PHASE811_DIR
    / "phase_8_0_11_evaluation_case_manifest_V1.parquet"
)


PHASE809_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_9_origin_eligibility_candidate_universe"
)

PHASE809_CONTRACT = (
    PHASE809_DIR
    / "phase_8_0_9_origin_eligibility_contract_V5.json"
)


PHASE806_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_6_dual_track_graph_policy"
)

PHASE806_POLICY = (
    PHASE806_DIR
    / "phase_8_0_6_graph_information_policy_V1.json"
)


PHASE7_REPORT = Path(
    "docs/Phase_7_Closure_Report_ITRS_Crunchbase.md"
)


OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_12_research_design_contract"
)

RQ_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_12_research_questions_V1.csv"
)

HYPOTHESIS_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_12_hypotheses_V1.csv"
)

BASELINE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_12_baseline_model_matrix_V1.csv"
)

ABLATION_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_12_required_ablation_matrix_V1.csv"
)

ENDPOINT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_12_endpoint_register_V1.csv"
)

STATISTICAL_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_12_statistical_analysis_contract_V1.json"
)

CONTRACT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_12_research_design_contract_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_12_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_12_derived_artifact_sha256_V1.json"
)


# =============================================================================
# FROZEN PHASE-8 ANCHORS
# =============================================================================

EXPECTED_CASE_COUNTS = {
    "T57": {
        "evaluation_cases": 21_924,
        "primary_cases": 6_850,
    },
    "T58": {
        "evaluation_cases": 22_384,
        "primary_cases": 6_872,
    },
    "T59": {
        "evaluation_cases": 22_650,
        "primary_cases": 6_764,
    },
    "APRIL_CONFIRMATORY": {
        "evaluation_cases": 6_313,
        "primary_cases": 1_690,
    },
}

EXPECTED_TOTAL_CASES = 73_271
EXPECTED_TOTAL_PRIMARY = 22_176

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 42


# =============================================================================
# UTILITIES
# =============================================================================

def separator(char="=", width=118):

    print(
        char * width
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


# =============================================================================
# MAIN
# =============================================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    separator()

    print(
        "PHASE 8.0.12 — "
        "FREEZE RESEARCH DESIGN CONTRACT V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "OUTCOME-BLIND RESEARCH-DESIGN FREEZE"
    )

    print("New model constructed:       NO")
    print("New model trained:           NO")
    print("Checkpoint loaded:           NO")
    print("Inference executed:          NO")
    print("Ranking metrics scored:      NO")
    print("Model selection performed:   NO")
    print("April performance inspected: NO")

    # =========================================================================
    # 1. Source integrity
    # =========================================================================

    separator("-")

    print(
        "8.0.12.1 — AUTHORITATIVE "
        "INPUT INTEGRITY"
    )

    separator("-")

    required_paths = [
        PHASE810_PROTOCOL,
        PHASE810_SELECTION,
        PHASE811_SUMMARY,
        PHASE811_COLLISION,
        PHASE811_CONTRACT,
        PHASE811_CASES,
        PHASE809_CONTRACT,
        PHASE806_POLICY,
        PHASE7_REPORT,
    ]

    source_hashes = {}

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        observed_hash = sha256_file(
            path
        )

        source_hashes[
            str(path)
        ] = observed_hash

        print(
            f"FOUND  {path}"
        )

        print(
            f"SHA256 {observed_hash}"
        )

    # =========================================================================
    # 2. Verify prior frozen contracts
    # =========================================================================

    separator("-")

    print(
        "8.0.12.2 — PRIOR CONTRACT "
        "VERIFICATION"
    )

    separator("-")

    with open(
        PHASE810_PROTOCOL,
        "r",
        encoding="utf-8",
    ) as f:

        protocol810 = json.load(
            f
        )

    with open(
        PHASE810_SELECTION,
        "r",
        encoding="utf-8",
    ) as f:

        selection810 = json.load(
            f
        )

    with open(
        PHASE811_CONTRACT,
        "r",
        encoding="utf-8",
    ) as f:

        candidate811 = json.load(
            f
        )

    with open(
        PHASE809_CONTRACT,
        "r",
        encoding="utf-8",
    ) as f:

        origin809 = json.load(
            f
        )

    with open(
        PHASE806_POLICY,
        "r",
        encoding="utf-8",
    ) as f:

        graph806 = json.load(
            f
        )

    frozen_checks = {
        "phase_8_0_10_protocol":
            protocol810.get(
                "status"
            )
            ==
            "FROZEN",

        "phase_8_0_10_selection":
            selection810.get(
                "status"
            )
            ==
            "FROZEN",

        "phase_8_0_11_candidates":
            candidate811.get(
                "status"
            )
            ==
            "FROZEN",

        "phase_8_0_9_origin_contract":
            origin809.get(
                "status"
            )
            ==
            "FROZEN",

        "phase_8_0_6_strict_graph":
            graph806[
                "strict_temporal_track"
            ][
                "status"
            ]
            ==
            "FROZEN",

        "phase_8_0_6_founder_graph":
            graph806[
                "founder_proxy_track"
            ][
                "status"
            ]
            ==
            "FROZEN",
    }

    if not all(
        frozen_checks.values()
    ):

        raise ValueError(
            "At least one prerequisite "
            "Phase-8 contract is not frozen."
        )

    print(
        "All prerequisite contracts: PASS"
    )

    # =========================================================================
    # 3. Verify immutable candidate cases
    # =========================================================================

    separator("-")

    print(
        "8.0.12.3 — FROZEN EVALUATION "
        "CASE VERIFICATION"
    )

    separator("-")

    summary811 = pd.read_csv(
        PHASE811_SUMMARY
    )

    case_manifest = pd.read_parquet(
        PHASE811_CASES,
        columns=[
            "origin",
            "interaction_id",
            "investor_id",
            "positive_startup_id",
            "is_primary_stratum",
        ],
    )

    if (
        len(
            case_manifest
        )
        !=
        EXPECTED_TOTAL_CASES
    ):

        raise ValueError(
            "Total evaluation case count drift."
        )

    if (
        int(
            case_manifest[
                "is_primary_stratum"
            ].sum()
        )
        !=
        EXPECTED_TOTAL_PRIMARY
    ):

        raise ValueError(
            "Total primary case count drift."
        )

    if (
        case_manifest[
            [
                "origin",
                "interaction_id",
            ]
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            "Duplicate origin/interaction "
            "evaluation case."
        )

    for origin, expected in (
        EXPECTED_CASE_COUNTS.items()
    ):

        subset = case_manifest[
            case_manifest[
                "origin"
            ]
            ==
            origin
        ]

        if (
            len(
                subset
            )
            !=
            expected[
                "evaluation_cases"
            ]
        ):

            raise ValueError(
                f"{origin}: evaluation "
                "case count drift."
            )

        if (
            int(
                subset[
                    "is_primary_stratum"
                ].sum()
            )
            !=
            expected[
                "primary_cases"
            ]
        ):

            raise ValueError(
                f"{origin}: primary "
                "case count drift."
            )

    print(
        f"Evaluation cases: "
        f"{len(case_manifest):,} PASS"
    )

    print(
        f"Primary cases:    "
        f"{int(case_manifest['is_primary_stratum'].sum()):,} PASS"
    )

    # =========================================================================
    # 4. Freeze research questions
    # =========================================================================

    separator("-")

    print(
        "8.0.12.4 — RESEARCH QUESTIONS"
    )

    separator("-")

    research_questions = pd.DataFrame(
        [
            {
                "rq":
                    "RQ1",

                "role":
                    "PRIMARY_CONFIRMATORY",

                "question":
                    (
                        "Can an inductive startup "
                        "representation improve "
                        "new-to-investor ranking for "
                        "origin-eligible cold-start "
                        "startups relative to reproduced "
                        "ITRS?"
                    ),

                "evaluation_scope":
                    (
                        "T57/T58/T59 selection; "
                        "April confirmatory"
                    ),
            },
            {
                "rq":
                    "RQ2",

                "role":
                    "SECONDARY_DEVELOPMENT",

                "question":
                    (
                        "Does founder-related side "
                        "information improve ranking "
                        "for origin-eligible cold-start "
                        "startups?"
                    ),

                "evaluation_scope":
                    "T57/T58/T59 only",
            },
            {
                "rq":
                    "RQ3",

                "role":
                    "SECONDARY_DEVELOPMENT",

                "question":
                    (
                        "Does timestamped acquisition "
                        "structure provide incremental "
                        "ranking value?"
                    ),

                "evaluation_scope":
                    "T57/T58/T59 only",
            },
            {
                "rq":
                    "RQ4",

                "role":
                    "SECONDARY_DEVELOPMENT",

                "question":
                    (
                        "Does the trend-aware investor "
                        "component provide incremental "
                        "value after introducing an "
                        "inductive startup representation?"
                    ),

                "evaluation_scope":
                    "T57/T58/T59 only",
            },
            {
                "rq":
                    "RQ5",

                "role":
                    "SECONDARY_DEVELOPMENT",

                "question":
                    (
                        "Is the benefit of the proposed "
                        "startup representation larger "
                        "for startup cold-start discovery "
                        "than for new warm/warm discovery?"
                    ),

                "evaluation_scope":
                    "T57/T58/T59 only",
            },
        ]
    )

    research_questions.to_csv(
        RQ_OUTPUT,
        index=False,
    )

    print(
        research_questions[
            [
                "rq",
                "role",
                "question",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 5. Freeze hypotheses
    # =========================================================================

    separator("-")

    print(
        "8.0.12.5 — HYPOTHESES"
    )

    separator("-")

    hypotheses = pd.DataFrame(
        [
            {
                "hypothesis":
                    "H1",

                "linked_rq":
                    "RQ1",

                "role":
                    "PRIMARY_CONFIRMATORY",

                "direction":
                    "positive",

                "statement":
                    (
                        "The single Phase-8 model "
                        "selected using T57/T58/T59 "
                        "has higher primary-stratum "
                        "NDCG@10 than B0 reproduced "
                        "ITRS on the untouched April "
                        "confirmatory set."
                    ),

                "formal_support_rule":
                    (
                        "paired mean delta NDCG@10 > 0 "
                        "and two-way investor/startup "
                        "clustered 95% CI lower bound > 0"
                    ),
            },
            {
                "hypothesis":
                    "H2",

                "linked_rq":
                    "RQ2",

                "role":
                    "SECONDARY_DEVELOPMENT",

                "direction":
                    "positive",

                "statement":
                    (
                        "Founder-containing variants "
                        "improve mean primary-stratum "
                        "NDCG@10 over the otherwise "
                        "matched no-founder variant."
                    ),

                "formal_support_rule":
                    (
                        "direction and uncertainty "
                        "reported on T57/T58/T59; "
                        "not an April confirmatory claim"
                    ),
            },
            {
                "hypothesis":
                    "H3",

                "linked_rq":
                    "RQ3",

                "role":
                    "SECONDARY_DEVELOPMENT",

                "direction":
                    "two-sided",

                "statement":
                    (
                        "Timestamped acquisition "
                        "relations may change ranking "
                        "performance after controlling "
                        "the startup encoder."
                    ),

                "formal_support_rule":
                    (
                        "effect estimate and uncertainty "
                        "reported; no directional claim "
                        "pre-specified"
                    ),
            },
            {
                "hypothesis":
                    "H4",

                "linked_rq":
                    "RQ4",

                "role":
                    "SECONDARY_DEVELOPMENT",

                "direction":
                    "two-sided",

                "statement":
                    (
                        "The trend-aware investor "
                        "component may add incremental "
                        "ranking value after the "
                        "startup-side inductive encoder "
                        "is introduced."
                    ),

                "formal_support_rule":
                    (
                        "matched trend-on versus "
                        "trend-off comparison on "
                        "T57/T58/T59"
                    ),
            },
            {
                "hypothesis":
                    "H5",

                "linked_rq":
                    "RQ5",

                "role":
                    "SECONDARY_DEVELOPMENT",

                "direction":
                    "positive_interaction",

                "statement":
                    (
                        "The improvement relative to B0 "
                        "is larger in the primary "
                        "startup-cold stratum than in "
                        "new warm/warm cases."
                    ),

                "formal_support_rule":
                    (
                        "report difference-in-improvement "
                        "with dependence-aware uncertainty"
                    ),
            },
        ]
    )

    hypotheses.to_csv(
        HYPOTHESIS_OUTPUT,
        index=False,
    )

    print(
        hypotheses[
            [
                "hypothesis",
                "role",
                "direction",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 6. Baseline / model-family matrix
    # =========================================================================

    separator("-")

    print(
        "8.0.12.6 — BASELINE / "
        "MODEL-FAMILY MATRIX"
    )

    separator("-")

    baseline_matrix = pd.DataFrame(
        [
            {
                "model_id":
                    "B0",

                "name":
                    "Reproduced ITRS",

                "startup_representation":
                    "original reproduced ITRS",

                "heterogeneous_graph":
                    "original Phase-8-compatible ITRS graph input",

                "trend_investor_branch":
                    True,

                "purpose":
                    "mandatory primary baseline",

                "development_required":
                    True,

                "april_role":
                    "MANDATORY_CONFIRMATORY_BASELINE",
            },
            {
                "model_id":
                    "B1",

                "name":
                    "Inductive feature-only startup baseline",

                "startup_representation":
                    (
                        "inductive non-graph encoder "
                        "using the same origin-safe "
                        "startup inputs available to "
                        "the proposed encoder"
                    ),

                "heterogeneous_graph":
                    "none",

                "trend_investor_branch":
                    "matched to comparison",

                "purpose":
                    (
                        "separate inductive feature "
                        "benefit from graph benefit"
                    ),

                "development_required":
                    True,

                "april_role":
                    "ONLY_IF_FINAL_SELECTED_MODEL",
            },
            {
                "model_id":
                    "M1",

                "name":
                    "Inductive heterogeneous startup model",

                "startup_representation":
                    "inductive heterogeneous encoder",

                "heterogeneous_graph":
                    (
                        "origin-safe relation variant "
                        "selected through development"
                    ),

                "trend_investor_branch":
                    False,

                "purpose":
                    (
                        "measure startup-side "
                        "heterogeneous representation"
                    ),

                "development_required":
                    True,

                "april_role":
                    "ONLY_IF_FINAL_SELECTED_MODEL",
            },
            {
                "model_id":
                    "M2",

                "name":
                    (
                        "Inductive heterogeneous startup "
                        "+ trend-aware investor model"
                    ),

                "startup_representation":
                    "same inductive encoder as M1",

                "heterogeneous_graph":
                    "same selected relation configuration",

                "trend_investor_branch":
                    True,

                "purpose":
                    "full proposed model family",

                "development_required":
                    True,

                "april_role":
                    "ONLY_IF_FINAL_SELECTED_MODEL",
            },
        ]
    )

    baseline_matrix.to_csv(
        BASELINE_OUTPUT,
        index=False,
    )

    print(
        baseline_matrix[
            [
                "model_id",
                "name",
                "purpose",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 7. Required ablations
    # =========================================================================

    separator("-")

    print(
        "8.0.12.7 — REQUIRED ABLATION MATRIX"
    )

    separator("-")

    ablations = pd.DataFrame(
        [
            {
                "ablation_id":
                    "A0",

                "relation_configuration":
                    "no_heterogeneous_side_relations",

                "founder":
                    False,

                "acquisition":
                    False,

                "trend":
                    True,

                "purpose":
                    "non-graph reference condition",
            },
            {
                "ablation_id":
                    "A1",

                "relation_configuration":
                    "acquisition_only_strict_temporal",

                "founder":
                    False,

                "acquisition":
                    True,

                "trend":
                    True,

                "purpose":
                    (
                        "measure timestamp-auditable "
                        "acquisition contribution"
                    ),
            },
            {
                "ablation_id":
                    "A2",

                "relation_configuration":
                    "founder_only_time_eligible",

                "founder":
                    True,

                "acquisition":
                    False,

                "trend":
                    True,

                "purpose":
                    "measure founder contribution",
            },
            {
                "ablation_id":
                    "A3",

                "relation_configuration":
                    "founder_plus_acquisition",

                "founder":
                    True,

                "acquisition":
                    True,

                "trend":
                    True,

                "purpose":
                    "full structural relation condition",
            },
            {
                "ablation_id":
                    "A4",

                "relation_configuration":
                    "founder_plus_acquisition",

                "founder":
                    True,

                "acquisition":
                    True,

                "trend":
                    False,

                "purpose":
                    (
                        "matched trend-off condition "
                        "for incremental trend test"
                    ),
            },
        ]
    )

    ablations[
        "evaluation_scope"
    ] = "T57/T58/T59_ONLY"

    ablations[
        "April_scoring_allowed"
    ] = False

    ablations.to_csv(
        ABLATION_OUTPUT,
        index=False,
    )

    print(
        ablations.to_string(
            index=False
        )
    )

    # =========================================================================
    # 8. Endpoint register
    # =========================================================================

    separator("-")

    print(
        "8.0.12.8 — ENDPOINT REGISTER"
    )

    separator("-")

    endpoints = pd.DataFrame(
        [
            {
                "endpoint_id":
                    "E1",

                "role":
                    "PRIMARY_CONFIRMATORY",

                "population":
                    (
                        "April; new-to-investor; "
                        "warm investor; investment-cold "
                        "startup; startup origin eligible"
                    ),

                "metric":
                    "NDCG@10",

                "comparison":
                    "M* minus B0",

                "model_selection_use":
                    False,
            },
            {
                "endpoint_id":
                    "E2",

                "role":
                    "SECONDARY_CONFIRMATORY",

                "population":
                    "same as E1",

                "metric":
                    "HR@10",

                "comparison":
                    "M* minus B0",

                "model_selection_use":
                    False,
            },
            {
                "endpoint_id":
                    "E3",

                "role":
                    "SECONDARY_GUARDRAIL",

                "population":
                    (
                        "April new-to-investor "
                        "warm-investor/warm-startup"
                    ),

                "metric":
                    "NDCG@10 and HR@10",

                "comparison":
                    "M* minus B0",

                "model_selection_use":
                    False,
            },
            {
                "endpoint_id":
                    "E4",

                "role":
                    "SECONDARY_AGGREGATE",

                "population":
                    "all April origin-eligible cases",

                "metric":
                    "NDCG@10 and HR@10",

                "comparison":
                    "M* minus B0",

                "model_selection_use":
                    False,
            },
            {
                "endpoint_id":
                    "E5",

                "role":
                    "DEVELOPMENT_SELECTION",

                "population":
                    (
                        "primary stratum in each "
                        "T57/T58/T59 fold"
                    ),

                "metric":
                    "NDCG@10",

                "comparison":
                    "candidate configurations",

                "model_selection_use":
                    True,
            },
        ]
    )

    endpoints.to_csv(
        ENDPOINT_OUTPUT,
        index=False,
    )

    print(
        endpoints.to_string(
            index=False
        )
    )

    # =========================================================================
    # 9. Freeze statistical analysis contract
    # =========================================================================

    separator("-")

    print(
        "8.0.12.9 — STATISTICAL "
        "ANALYSIS CONTRACT"
    )

    separator("-")

    statistical_contract = {
        "schema_version":
            "PHASE_8_0_12_STATISTICAL_ANALYSIS_V1",

        "status":
            "FROZEN",

        "primary_confirmatory_comparison": {
            "models":
                [
                    "B0",
                    "M_STAR",
                ],

            "population":
                (
                    "April primary thesis stratum; "
                    "N=1,690 frozen cases"
                ),

            "metric":
                "NDCG@10",

            "estimand":
                (
                    "mean paired case-level difference: "
                    "NDCG@10(M*) - NDCG@10(B0)"
                ),

            "pairing":
                (
                    "same event and identical frozen "
                    "candidate list for both models"
                ),
        },

        "primary_uncertainty": {
            "method":
                (
                    "two-way investor/startup "
                    "clustered bootstrap"
                ),

            "confidence_level":
                0.95,

            "replicates":
                BOOTSTRAP_REPLICATES,

            "seed":
                BOOTSTRAP_SEED,

            "cluster_1":
                "investor_id",

            "cluster_2":
                "positive_startup_id",

            "paired_model_difference":
                True,
        },

        "primary_hypothesis_support_rule": (
            "H1 is statistically supported only if "
            "the paired mean April primary-stratum "
            "delta NDCG@10 is positive and the lower "
            "bound of the frozen 95% two-way clustered "
            "confidence interval is greater than zero."
        ),

        "secondary_confirmatory_metrics": [
            "primary-stratum HR@10 difference",
            "new warm/warm NDCG@10 difference",
            "new warm/warm HR@10 difference",
            "aggregate eligible NDCG@10 difference",
            "aggregate eligible HR@10 difference",
        ],

        "secondary_inference_policy": (
            "Secondary endpoints receive effect estimates "
            "and 95% dependence-aware intervals. They do "
            "not override or redefine the primary H1 result."
        ),

        "dependence_sensitivity_analyses": [
            "investor-clustered bootstrap",
            "startup-clustered bootstrap",
            "investor-startup-pair-clustered bootstrap",
            "two-way investor/startup clustered bootstrap",
        ],

        "multiple_testing": {
            "single_primary_confirmatory_endpoint":
                True,

            "formal_multiplicity_adjustment":
                False,

            "reason":
                (
                    "Only E1 is designated as the primary "
                    "confirmatory hypothesis test. Other "
                    "endpoints are secondary/descriptive "
                    "and cannot replace E1."
                ),
        },

        "development_fold_inference": {
            "role":
                (
                    "development evidence and controlled "
                    "ablation interpretation"
                ),

            "selection_metric":
                (
                    "equal-weight mean primary NDCG@10 "
                    "across T57/T58/T59"
                ),

            "confirmatory_p_values":
                False,

            "April_ablation_search":
                False,
        },

        "warm_warm_guardrail": {
            "formal_noninferiority_margin":
                None,

            "interpretation":
                (
                    "Report effect size and uncertainty. "
                    "Do not claim formal non-inferiority "
                    "or 'no material degradation' without "
                    "a separately justified margin."
                ),
        },

        "April_rescoring_policy": {
            "primary_scoring_pass":
                "ONE",

            "candidate_resampling":
                False,

            "post_April_architecture_change":
                False,

            "post_April_hyperparameter_change":
                False,

            "post_April_checkpoint_change":
                False,

            "post_April_model_reselection":
                False,
        },

        "causal_claims_allowed":
            False,

        "approved_language": (
            "performance difference, association, "
            "incremental predictive value"
        ),

        "prohibited_language_without_additional_design": [
            "causes improvement",
            "founder relations cause investment success",
            "graph structure causes investor decisions",
        ],
    }

    write_json(
        STATISTICAL_OUTPUT,
        statistical_contract,
    )

    print(
        "Primary April estimand:"
    )

    print(
        "  mean paired delta NDCG@10 "
        "(M* - B0)"
    )

    print()

    print(
        "Primary uncertainty:"
    )

    print(
        "  two-way investor/startup "
        "clustered bootstrap"
    )

    print(
        f"  replicates: {BOOTSTRAP_REPLICATES:,}"
    )

    print()

    print(
        "April ablation search: NO"
    )

    # =========================================================================
    # 10. Complete Phase-8.0 research design contract
    # =========================================================================

    separator("-")

    print(
        "8.0.12.10 — COMPLETE "
        "RESEARCH DESIGN CONTRACT"
    )

    separator("-")

    contract = {
        "schema_version":
            "PHASE_8_0_12_RESEARCH_DESIGN_CONTRACT_V1",

        "status":
            "FROZEN",

        "thesis_title":
            (
                "Improving Trend-Aware Heterogeneous "
                "Graph Recommendation for New-to-Investor "
                "Startup Discovery under Cold Start"
            ),

        "primary_task":
            (
                "Rank origin-eligible startups for "
                "warm investors where the investor-startup "
                "pair is new and the startup has no prior "
                "investment history."
            ),

        "primary_confirmatory_question":
            research_questions.loc[
                research_questions[
                    "rq"
                ]
                ==
                "RQ1",
                "question",
            ].iloc[0],

        "primary_hypothesis":
            hypotheses.loc[
                hypotheses[
                    "hypothesis"
                ]
                ==
                "H1",
                "statement",
            ].iloc[0],

        "development_windows": [
            "T57",
            "T58",
            "T59",
        ],

        "confirmatory_window":
            "APRIL_CONFIRMATORY",

        "confirmatory_primary_cases":
            1_690,

        "confirmatory_models_allowed": [
            "B0",
            "one_single_final_selected_model_M_STAR",
        ],

        "development_ablation_scoring_allowed":
            True,

        "April_ablation_scoring_allowed":
            False,

        "Phase6_test_metric_reused_as_Phase8_baseline":
            False,

        "B0_must_be_scored_under_Phase8_protocol":
            True,

        "founder_relation_interpretation": {
            "artifact_track_name":
                "founder_proxy",

            "Phase8_scientific_term":
                (
                    "temporally eligible shared-founder "
                    "side information"
                ),

            "eligibility_rule":
                (
                    "both endpoint founded_on known "
                    "and <= prediction origin, with "
                    "direct target-pair leakage removed"
                ),

            "semantic_effective_time":
                (
                    "max(src.founded_on, dst.founded_on)"
                ),

            "Crunchbase_database_observation_time_known":
                False,

            "historical_snapshot_claim":
                False,
        },

        "strict_graph_robustness_track":
            (
                "timestamped acquisition-only "
                "structural graph"
            ),

        "model_selection_contract":
            str(
                PHASE810_SELECTION
            ),

        "candidate_runtime_contract":
            str(
                PHASE811_CONTRACT
            ),

        "statistical_analysis_contract":
            str(
                STATISTICAL_OUTPUT
            ),

        "new_model_training_started":
            False,

        "April_performance_inspected":
            False,

        "phase_8_0_status":
            "READY_FOR_CLOSURE_IF_INTEGRITY_CHECKS_PASS",
    }

    write_json(
        CONTRACT_OUTPUT,
        contract,
    )

    # =========================================================================
    # 11. Manifest + hashes
    # =========================================================================

    manifest = {
        "phase":
            "8.0.12",

        "authoritative_inputs":
            source_hashes,

        "outputs": [
            str(
                RQ_OUTPUT
            ),
            str(
                HYPOTHESIS_OUTPUT
            ),
            str(
                BASELINE_OUTPUT
            ),
            str(
                ABLATION_OUTPUT
            ),
            str(
                ENDPOINT_OUTPUT
            ),
            str(
                STATISTICAL_OUTPUT
            ),
            str(
                CONTRACT_OUTPUT
            ),
        ],

        "prohibited_operations": [
            "new_model_training",
            "model_inference",
            "ranking_metric_scoring",
            "model_selection",
            "April_performance_scoring",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    output_hashes = {
        str(
            RQ_OUTPUT
        ):
            sha256_file(
                RQ_OUTPUT
            ),

        str(
            HYPOTHESIS_OUTPUT
        ):
            sha256_file(
                HYPOTHESIS_OUTPUT
            ),

        str(
            BASELINE_OUTPUT
        ):
            sha256_file(
                BASELINE_OUTPUT
            ),

        str(
            ABLATION_OUTPUT
        ):
            sha256_file(
                ABLATION_OUTPUT
            ),

        str(
            ENDPOINT_OUTPUT
        ):
            sha256_file(
                ENDPOINT_OUTPUT
            ),

        str(
            STATISTICAL_OUTPUT
        ):
            sha256_file(
                STATISTICAL_OUTPUT
            ),

        str(
            CONTRACT_OUTPUT
        ):
            sha256_file(
                CONTRACT_OUTPUT
            ),

        str(
            MANIFEST_OUTPUT
        ):
            sha256_file(
                MANIFEST_OUTPUT
            ),
    }

    write_json(
        HASH_OUTPUT,
        output_hashes,
    )

    # =========================================================================
    # 12. Final
    # =========================================================================

    separator()

    print(
        "PHASE 8.0.12 RESEARCH "
        "DESIGN FREEZE COMPLETE"
    )

    separator()

    print()

    print(
        "RESEARCH QUESTIONS:             FROZEN"
    )

    print(
        "HYPOTHESES:                     FROZEN"
    )

    print(
        "BASELINE MODEL MATRIX:           FROZEN"
    )

    print(
        "REQUIRED ABLATIONS:              FROZEN"
    )

    print(
        "PRIMARY ENDPOINT:                "
        "FROZEN -> April primary NDCG@10"
    )

    print(
        "PRIMARY COMPARISON:              "
        "FROZEN -> M* minus B0"
    )

    print(
        "PRIMARY UNCERTAINTY:             "
        "FROZEN -> two-way clustered bootstrap"
    )

    print(
        "APRIL MODELS ALLOWED:            "
        "B0 + ONE selected final model"
    )

    print(
        "APRIL ABLATION SEARCH:           FORBIDDEN"
    )

    print(
        "NEW MODEL TRAINING PERFORMED:    NO"
    )

    print(
        "APRIL PERFORMANCE INSPECTED:     NO"
    )

    print()

    print(
        "If all outputs above are PASS, "
        "Phase 8.0 is scientifically ready "
        "for closure."
    )

    print()

    print(
        f"Outputs written to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()