from pathlib import Path
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.1.11A — MODEL-FAMILY BRANCH SEMANTICS CLOSURE V1
# =============================================================================
#
# PURPOSE
# -------
# Remove the final ambiguity in Phase 8.1.11:
#
# Phase 8.1.11 explicitly froze Startup graph activation but did not explicitly
# state whether the Investor structural R-GCN branch is active in B1a/B1b.
#
# Earlier model-family contracts define B1a/B1b as NO-GRAPH comparators.
# Therefore BOTH Investor and Startup structural slots must be ZERO_40 in
# B1a/B1b.
#
# No new feature, architecture, node, edge, hyperparameter or model family
# is introduced.
# =============================================================================


BASE = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation"
)

MODEL_FAMILY_V2 = (
    BASE
    / "phase_8_1_7_input_contract_amendment"
    / "phase_8_1_7_model_family_contract_V2.csv"
)

ARCHITECTURE_CONTRACT = (
    BASE
    / "phase_8_1_11_exact_neural_architecture"
    / "phase_8_1_11_architecture_contract_V1.json"
)

ARCHITECTURE_MODELS = (
    BASE
    / "phase_8_1_11_exact_neural_architecture"
    / "phase_8_1_11_model_family_architecture_V1.csv"
)

OUTPUT_DIR = (
    BASE
    / "phase_8_1_11a_model_family_branch_semantics"
)

BRANCH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_11a_branch_activation_V1.csv"
)

AMENDMENT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_11a_branch_semantics_amendment_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_11a_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_11a_artifact_sha256_V1.json"
)


def separator(char="=", width=118):
    print(char * width)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
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
        "PHASE 8.1.11A — "
        "MODEL-FAMILY BRANCH SEMANTICS CLOSURE V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "PRE-EXECUTION ARCHITECTURAL CLARIFICATION"
    )

    print("New feature introduced:      NO")
    print("Architecture topology changed:NO")
    print("Model family added:          NO")
    print("Model trained:               NO")
    print("April analyzed:              NO")

    # =========================================================================
    # 1. Gates
    # =========================================================================

    separator("-")
    print("8.1.11A.1 — PREREQUISITE GATES")
    separator("-")

    for path in [
        MODEL_FAMILY_V2,
        ARCHITECTURE_CONTRACT,
        ARCHITECTURE_MODELS,
    ]:

        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

        print(f"FOUND {path}")

    architecture = read_json(
        ARCHITECTURE_CONTRACT
    )

    models = pd.read_csv(
        ARCHITECTURE_MODELS
    )

    require(
        architecture.get("status")
        ==
        "FROZEN",
        "Phase 8.1.11 is not frozen.",
    )

    require(
        architecture.get(
            "architecture_selected"
        )
        is True,
        "Architecture not selected.",
    )

    require(
        architecture.get(
            "architecture_selection_used_performance"
        )
        is False,
        "Architecture unexpectedly used performance.",
    )

    require(
        architecture.get(
            "model_trained"
        )
        is False,
        "Model unexpectedly trained.",
    )

    require(
        architecture.get(
            "April_analyzed"
        )
        is False,
        "April unexpectedly analyzed.",
    )

    expected_models = {
        "B0",
        "B1a",
        "B1b",
        "M1",
        "M2",
    }

    require(
        set(models["model"].astype(str))
        ==
        expected_models,
        "Model-family set drift.",
    )

    print()
    print("Phase 8.1.11 gate: PASS")

    # =========================================================================
    # 2. Freeze exact branch semantics
    # =========================================================================

    separator("-")
    print("8.1.11A.2 — EXACT BRANCH ACTIVATION")
    separator("-")

    rows = [
        {
            "model": "B0",
            "investor_trend": "ITRS_TREND_40",
            "investor_latent": "BASELINE_L_O_40",
            "investor_description": "BASELINE_DESCRIPTION_40",
            "investor_structural": "BASELINE_PHASE4_RGCN_40",
            "startup_structured": "BASELINE_PHASE4_LATENT_40",
            "startup_content": "BASELINE_PHASE4_DESCRIPTION_40",
            "startup_structural": "BASELINE_PHASE4_RGCN_40",
            "graph_active": True,
            "graph_family": "FROZEN_PHASE4_GRAPH",
            "pair_dim": 280,
        },
        {
            "model": "B1a",
            "investor_trend": "ITRS_TREND_40",
            "investor_latent": "BASELINE_L_O_40",
            "investor_description": "BASELINE_DESCRIPTION_40",
            "investor_structural": "ZERO_40",
            "startup_structured": "ZERO_40",
            "startup_content": "INDUCTIVE_CONTENT_40",
            "startup_structural": "ZERO_40",
            "graph_active": False,
            "graph_family": "NONE",
            "pair_dim": 280,
        },
        {
            "model": "B1b",
            "investor_trend": "ITRS_TREND_40",
            "investor_latent": "BASELINE_L_O_40",
            "investor_description": "BASELINE_DESCRIPTION_40",
            "investor_structural": "ZERO_40",
            "startup_structured": "INDUCTIVE_BASE_STRUCTURED_40",
            "startup_content": "INDUCTIVE_CONTENT_40",
            "startup_structural": "ZERO_40",
            "graph_active": False,
            "graph_family": "NONE",
            "pair_dim": 280,
        },
        {
            "model": "M1",
            "investor_trend": "ZERO_40",
            "investor_latent": "BASELINE_L_O_40",
            "investor_description": "BASELINE_DESCRIPTION_40",
            "investor_structural": "HETEROGENEOUS_RGCN_40",
            "startup_structured": "INDUCTIVE_EXTENDED_STRUCTURED_40",
            "startup_content": "INDUCTIVE_CONTENT_40",
            "startup_structural": "HETEROGENEOUS_RGCN_40",
            "graph_active": True,
            "graph_family": "G_PRIMARY",
            "pair_dim": 280,
        },
        {
            "model": "M2",
            "investor_trend": "ITRS_TREND_40",
            "investor_latent": "BASELINE_L_O_40",
            "investor_description": "BASELINE_DESCRIPTION_40",
            "investor_structural": "HETEROGENEOUS_RGCN_40",
            "startup_structured": "INDUCTIVE_EXTENDED_STRUCTURED_40",
            "startup_content": "INDUCTIVE_CONTENT_40",
            "startup_structural": "HETEROGENEOUS_RGCN_40",
            "graph_active": True,
            "graph_family": "G_PRIMARY",
            "pair_dim": 280,
        },
    ]

    branch_df = pd.DataFrame(rows)

    require(
        branch_df["pair_dim"]
        .eq(280)
        .all(),
        "All models must retain pair_dim=280.",
    )

    print(
        branch_df[
            [
                "model",
                "investor_trend",
                "investor_structural",
                "startup_structured",
                "startup_content",
                "startup_structural",
                "graph_active",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 3. Scientific comparison semantics
    # =========================================================================

    separator("-")
    print("8.1.11A.3 — COMPARISON SEMANTICS")
    separator("-")

    comparisons = {
        "B0_vs_M2":
            (
                "Primary reproduced-ITRS versus proposed "
                "trend-aware inductive heterogeneous model."
            ),

        "B1a_vs_B1b":
            (
                "Incremental value of stable structured Startup "
                "attributes beyond inductive content."
            ),

        "M1_vs_M2":
            (
                "Incremental value of ITRS trend-aware Investor "
                "representation with identical heterogeneous "
                "Startup/graph architecture."
            ),

        "M2_no_heterogeneous_branch":
            (
                "Predeclared development ablation: retain M2 content, "
                "extended structured Startup features and trend, but "
                "replace BOTH Investor and Startup R-GCN outputs with "
                "ZERO_40."
            ),

        "B1a_B1b_graph_semantics":
            (
                "No graph means no R-GCN contribution on either side "
                "of the scored Investor-Startup pair."
            ),
    }

    for key, value in comparisons.items():
        print(f"{key:<32} {value}")

    # =========================================================================
    # 4. Inductive graph inference guard
    # =========================================================================

    separator("-")
    print("8.1.11A.4 — HELD-OUT NODE MESSAGE GUARD")
    separator("-")

    heldout_message_policy = {
        "training":
            (
                "Held-out primary cold-start Startup features are zeroed "
                "and all incident structural edges are masked."
            ),

        "inference_features":
            "Restore allowed held-out Startup features.",

        "inference_incoming_messages":
            True,

        "inference_outgoing_messages":
            False,

        "inference_rule":
            (
                "Held-out primary cold-start Startups are inference sinks: "
                "retain origin-eligible edges whose destination is the "
                "held-out Startup; suppress structural messages sourced "
                "from held-out primary Startups."
            ),

        "reason":
            (
                "Prevents one held-out Startup from altering Person, "
                "Investor, or other Startup representations during "
                "confirmatory inference while still allowing the unseen "
                "Startup to aggregate from the frozen fitted graph."
            ),

        "parameters_updated_at_inference":
            False,

        "evaluation_case_specific_graph":
            False,

        "negative_sample_specific_graph":
            False,
    }

    for key, value in heldout_message_policy.items():
        print(f"{key:<44} {value}")

    # =========================================================================
    # 5. Persist
    # =========================================================================

    separator("-")
    print("8.1.11A.5 — PERSIST CLOSURE")
    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    branch_df.to_csv(
        BRANCH_OUTPUT,
        index=False,
    )

    amendment = {
        "schema_version":
            "PHASE_8_1_11A_MODEL_FAMILY_BRANCH_SEMANTICS_V1",

        "status":
            "FROZEN",

        "scientific_role":
            "PRE_EXECUTION_ARCHITECTURAL_CLARIFICATION",

        "new_information_channels_added":
            False,

        "architecture_topology_changed":
            False,

        "model_family_added":
            False,

        "clarification":
            (
                "B1a/B1b are strict no-graph comparators: both "
                "Investor and Startup structural slots are ZERO_40. "
                "M1/M2 use the heterogeneous R-GCN on both sides."
            ),

        "model_family_branch_activation":
            branch_df.to_dict(
                orient="records"
            ),

        "comparison_semantics":
            comparisons,

        "heldout_message_policy":
            heldout_message_policy,

        "pair_dimension":
            280,

        "feature_discovery_closed":
            True,

        "architecture_selected":
            True,

        "performance_based_architecture_change":
            False,

        "model_trained":
            False,

        "April_analyzed":
            False,

        "April_performance_inspected":
            False,

        "next_step":
            (
                "Phase 8.1.12 — freeze randomness, initialization, "
                "optimizer, development checkpoint selection and "
                "execution contract."
            ),
    }

    write_json(
        AMENDMENT_OUTPUT,
        amendment,
    )

    manifest = {
        "phase": "8.1.11A",
        "status": "FROZEN",
        "outputs": [
            str(BRANCH_OUTPUT),
            str(AMENDMENT_OUTPUT),
        ],
        "new_information_channels_added": False,
        "architecture_topology_changed": False,
        "model_trained": False,
        "April_analyzed": False,
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    outputs = [
        BRANCH_OUTPUT,
        AMENDMENT_OUTPUT,
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
        "PHASE 8.1.11A MODEL-FAMILY "
        "BRANCH SEMANTICS FROZEN"
    )

    separator()

    print()
    print("B1a/B1b INVESTOR STRUCTURAL: ZERO_40")
    print("B1a/B1b STARTUP STRUCTURAL:  ZERO_40")
    print("M1/M2 INVESTOR STRUCTURAL:   HETERO_RGCN_40")
    print("M1/M2 STARTUP STRUCTURAL:    HETERO_RGCN_40")
    print("M1 TREND:                    ZERO_40")
    print("M2 TREND:                    ITRS_TREND_40")
    print("PAIR DIMENSION:              280")
    print()
    print("HELD-OUT INFERENCE NODES:    SINK-ONLY MESSAGING")
    print("NEW FEATURES:                NO")
    print("ARCHITECTURE TOPOLOGY CHANGE:NO")
    print("MODEL TRAINED:               NO")
    print("APRIL ANALYZED:              NO")
    print()
    print("NEXT:")
    print("Phase 8.1.12 — execution contract.")


if __name__ == "__main__":
    main()
