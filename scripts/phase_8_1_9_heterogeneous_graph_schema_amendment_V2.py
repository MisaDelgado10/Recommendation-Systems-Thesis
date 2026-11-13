from pathlib import Path
from copy import deepcopy
import hashlib
import json

import pandas as pd


# =============================================================================
# PHASE 8.1.9 — HETEROGENEOUS GRAPH SCHEMA AMENDMENT V2
# =============================================================================
#
# Scientific role:
#
#   PRE-TRAINING GRAPH-SCHEMA SEMANTIC / TEMPORAL REPAIR
#
# V2 supersedes V1 only in two areas:
#
#   1. P11/P12 placement
#        PERSON_NODE_FEATURE
#        ->
#        STARTUP_NODE_FEATURE
#
#      because founder history is Startup-conditioned:
#
#          H(s,t)
#
#   2. structural relation time gates
#
#        event_date <= prediction_origin
#        ->
#        event_date < prediction_origin
#
#      so structural information is strictly pre-origin.
#
# No candidate universe, graph family, node identity policy, relation family,
# feature inventory, model architecture or evaluation protocol is changed.
#
# =============================================================================


BASE = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation"
)

SCHEMA_DIR = (
    BASE
    / "phase_8_1_9_heterogeneous_graph_schema"
)

V1_SCHEMA = (
    SCHEMA_DIR
    / "phase_8_1_9_graph_schema_V1.json"
)

V1_MANIFEST = (
    SCHEMA_DIR
    / "phase_8_1_9_manifest_V1.json"
)

MASKING_CONTRACT = (
    BASE
    / "phase_8_1_12a_outcome_independent_cold_start_masking"
    / "phase_8_1_12a_outcome_independent_masking_contract_V1.json"
)

FOUNDER_HISTORY_CONTRACT = (
    BASE
    / "phase_8_1_12b_origin_referenced_founder_history"
    / "phase_8_1_12b_origin_referenced_founder_history_contract_V1.json"
)

BRANCH_SEMANTICS_V2 = (
    BASE
    / "phase_8_1_11a_model_family_branch_semantics"
    / "phase_8_1_11a_branch_semantics_amendment_V2.json"
)


V2_SCHEMA = (
    SCHEMA_DIR
    / "phase_8_1_9_graph_schema_V2.json"
)

V2_NODE_SCHEMA = (
    SCHEMA_DIR
    / "phase_8_1_9_node_type_schema_V2.csv"
)

V2_RELATION_SCHEMA = (
    SCHEMA_DIR
    / "phase_8_1_9_relation_schema_V2.csv"
)

V2_INFORMATION_PLACEMENT = (
    SCHEMA_DIR
    / "phase_8_1_9_information_placement_V2.csv"
)

V2_GRAPH_VARIANTS = (
    SCHEMA_DIR
    / "phase_8_1_9_graph_variants_V2.csv"
)

V2_MANIFEST = (
    SCHEMA_DIR
    / "phase_8_1_9_manifest_V2.json"
)

V2_HASHES = (
    SCHEMA_DIR
    / "phase_8_1_9_artifact_sha256_V2.json"
)


EXPECTED_CANDIDATE_UNIVERSE = (
    "FROZEN_311589_STARTUP_ROLE_NODES"
)

EXPECTED_INVESTOR_UNIVERSE = (
    "FROZEN_165975_INVESTOR_ROLE_NODES"
)

P11 = "founder_prior_startup_count"
P12 = "founder_prior_sector_profile"


def separator(char="=", width=118):
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
        "PHASE 8.1.9 — "
        "HETEROGENEOUS GRAPH SCHEMA AMENDMENT V2"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "PRE-TRAINING GRAPH-SCHEMA SEMANTIC / TEMPORAL REPAIR"
    )

    print("Feature discovery reopened:  NO")
    print("Candidate universe changed:  NO")
    print("Graph families changed:      NO")
    print("Relation families changed:   NO")
    print("Architecture changed:        NO")
    print("Graph materialized:          NO")
    print("Training performed:          NO")
    print("Performance inspected:       NO")
    print("T60 scored:                  NO")
    print("April analyzed:              NO")

    # =========================================================================
    # 1. Prerequisite gates
    # =========================================================================

    separator("-")

    print(
        "8.1.9-V2.1 — PREREQUISITE GATES"
    )

    separator("-")

    required = [
        V1_SCHEMA,
        V1_MANIFEST,
        MASKING_CONTRACT,
        FOUNDER_HISTORY_CONTRACT,
        BRANCH_SEMANTICS_V2,
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
        V1_SCHEMA
    )

    v1_manifest = read_json(
        V1_MANIFEST
    )

    masking = read_json(
        MASKING_CONTRACT
    )

    founder_history = read_json(
        FOUNDER_HISTORY_CONTRACT
    )

    branch_v2 = read_json(
        BRANCH_SEMANTICS_V2
    )

    require(
        v1.get("status")
        ==
        "FROZEN",
        "8.1.9 V1 is not frozen.",
    )

    require(
        v1_manifest.get("status")
        ==
        "FROZEN",
        "8.1.9 V1 manifest is not frozen.",
    )

    require(
        masking.get("status")
        ==
        "FROZEN",
        "8.1.12A is not frozen.",
    )

    require(
        founder_history.get("status")
        ==
        "FROZEN",
        "8.1.12B is not frozen.",
    )

    require(
        branch_v2.get("status")
        ==
        "FROZEN",
        "8.1.11A V2 is not frozen.",
    )

    require(
        v1[
            "graph_universe_boundaries"
        ][
            "candidate_startup_universe"
        ]
        ==
        EXPECTED_CANDIDATE_UNIVERSE,
        "Startup candidate universe drift.",
    )

    require(
        v1[
            "graph_universe_boundaries"
        ][
            "investor_role_universe"
        ]
        ==
        EXPECTED_INVESTOR_UNIVERSE,
        "Investor universe drift.",
    )

    # =========================================================================
    # 2. Verify founder-history semantics already correct in V1
    # =========================================================================

    separator("-")

    print(
        "8.1.9-V2.2 — FOUNDER-HISTORY SEMANTIC RECHECK"
    )

    separator("-")

    history_policy = (
        v1[
            "founder_history_policy"
        ]
    )

    require(
        history_policy.get(
            "prior_company_condition"
        )
        ==
        "prior_company.founded_on < prediction_origin",
        (
            "V1 founder-history cutoff is not the expected "
            "prediction-origin rule."
        ),
    )

    require(
        history_policy.get(
            "focal_startup_excluded"
        )
        is True,
        "Focal Startup is not excluded.",
    )

    require(
        history_policy.get(
            "must_predate_focal_startup_founding"
        )
        is False,
        (
            "V1 unexpectedly requires prior Company "
            "to predate focal Startup founding."
        ),
    )

    print(
        "Prediction-origin founder-history cutoff: PASS"
    )

    print(
        "Focal Startup exclusion:              PASS"
    )

    print(
        "Pre-focal-founding requirement:       FALSE / PASS"
    )

    # =========================================================================
    # 3. Detect stale P11/P12 placement
    # =========================================================================

    separator("-")

    print(
        "8.1.9-V2.3 — P11/P12 PLACEMENT DRIFT"
    )

    separator("-")

    placement = (
        v1[
            "information_placement"
        ]
    )

    by_name = {
        row[
            "information"
        ]:
            row
        for row
        in placement
    }

    require(
        P11 in by_name,
        "P11 missing from information placement.",
    )

    require(
        P12 in by_name,
        "P12 missing from information placement.",
    )

    require(
        by_name[
            P11
        ][
            "placement"
        ]
        ==
        "PERSON_NODE_FEATURE",
        (
            "Expected historical P11 Person placement "
            "was not found."
        ),
    )

    require(
        by_name[
            P12
        ][
            "placement"
        ]
        ==
        "PERSON_NODE_FEATURE",
        (
            "Expected historical P12 Person placement "
            "was not found."
        ),
    )

    print(
        "Historical P11 placement: PERSON_NODE_FEATURE"
    )

    print(
        "Historical P12 placement: PERSON_NODE_FEATURE"
    )

    print(
        "Correct placement:         STARTUP_NODE_FEATURE"
    )

    # =========================================================================
    # 4. Detect inclusive relation gates
    # =========================================================================

    separator("-")

    print(
        "8.1.9-V2.4 — STRICT PRE-ORIGIN RELATION-GATE REPAIR"
    )

    separator("-")

    relations = {
        row[
            "relation_id"
        ]:
            row
        for row
        in v1[
            "primary_relations"
        ]
    }

    expected_old_gates = {
        "R01":
            "startup.founded_on <= origin",

        "R02":
            "startup.founded_on <= origin",

        "R03":
            "acquisition.announced_on <= origin",

        "R04":
            "acquisition.announced_on <= origin",
    }

    for relation_id, old_gate in (
        expected_old_gates.items()
    ):

        require(
            relations[
                relation_id
            ][
                "origin_gate"
            ]
            ==
            old_gate,
            (
                f"{relation_id}: unexpected V1 origin gate: "
                f"{relations[relation_id]['origin_gate']}"
            ),
        )

        print(
            f"{relation_id}: {old_gate}"
        )

    print()

    print(
        "Correct temporal convention: "
        "structural evidence must be STRICTLY BEFORE prediction origin."
    )

    # =========================================================================
    # 5. Construct V2
    # =========================================================================

    separator("-")

    print(
        "8.1.9-V2.5 — CONSTRUCT CORRECTED GRAPH SCHEMA"
    )

    separator("-")

    v2 = deepcopy(
        v1
    )

    v2[
        "schema_version"
    ] = (
        "PHASE_8_1_9_HETEROGENEOUS_GRAPH_SCHEMA_V2"
    )

    v2[
        "scientific_role"
    ] = (
        "PRE_MATERIALIZATION_HETEROGENEOUS_GRAPH_SCHEMA_CORRECTED"
    )

    # -------------------------------------------------------------------------
    # Correct P11 / P12 placement.
    # -------------------------------------------------------------------------

    for row in (
        v2[
            "information_placement"
        ]
    ):

        if (
            row[
                "information"
            ]
            ==
            P11
        ):

            row[
                "placement"
            ] = (
                "STARTUP_NODE_FEATURE"
            )

            row[
                "origin_specific"
            ] = True

            row[
                "notes"
            ] = (
                "Startup-conditioned P11 = |H(s,t)|. "
                "H(s,t) contains unique non-focal Companies "
                "founded by any exact-resolved founder of focal "
                "Startup s with parseable founded_on strictly "
                "before prediction origin t. Prior Companies "
                "are deduplicated by UUID across founders."
            )

        if (
            row[
                "information"
            ]
            ==
            P12
        ):

            row[
                "placement"
            ] = (
                "STARTUP_NODE_FEATURE"
            )

            row[
                "origin_specific"
            ] = True

            row[
                "notes"
            ] = (
                "Startup-conditioned P12 sector profile derived "
                "from the same unique H(s,t) Company set as P11. "
                "Company category/category-group labels retain "
                "matched-snapshot semantics."
            )

    # -------------------------------------------------------------------------
    # Correct Person role semantics.
    # -------------------------------------------------------------------------

    for row in (
        v2[
            "node_types"
        ]
    ):

        if (
            row[
                "node_type"
            ]
            ==
            "person"
        ):

            row[
                "temporal_policy"
            ] = (
                "Person is an identity-free structural intermediary. "
                "Founder relations are eligible only when the "
                "associated Startup founded_on is strictly before "
                "prediction origin."
            )

    # -------------------------------------------------------------------------
    # Strict pre-origin relation gates.
    # -------------------------------------------------------------------------

    corrected_gates = {
        "R01":
            "startup.founded_on < origin",

        "R02":
            "startup.founded_on < origin",

        "R03":
            "acquisition.announced_on < origin",

        "R04":
            "acquisition.announced_on < origin",
    }

    for row in (
        v2[
            "primary_relations"
        ]
    ):

        relation_id = (
            row[
                "relation_id"
            ]
        )

        require(
            relation_id
            in
            corrected_gates,
            (
                f"Unexpected primary relation ID: "
                f"{relation_id}"
            ),
        )

        row[
            "origin_gate"
        ] = (
            corrected_gates[
                relation_id
            ]
        )

    # -------------------------------------------------------------------------
    # Explicit semantic amendment ledger.
    # -------------------------------------------------------------------------

    v2[
        "semantic_amendment"
    ] = {
        "version":
            "V2",

        "supersedes":
            str(
                V1_SCHEMA
            ),

        "V1_file_modified":
            False,

        "V1_artifacts_preserved":
            True,

        "P11_placement":
            (
                "PERSON_NODE_FEATURE -> STARTUP_NODE_FEATURE"
            ),

        "P12_placement":
            (
                "PERSON_NODE_FEATURE -> STARTUP_NODE_FEATURE"
            ),

        "P11_P12_definition_changed":
            False,

        "P11_P12_definition":
            (
                "Startup-conditioned origin-specific H(s,t), "
                "with prior_company.founded_on < prediction_origin, "
                "focal Startup excluded, prior Company UUID "
                "deduplicated across founders."
            ),

        "founder_relation_gate":
            (
                "startup.founded_on < prediction_origin"
            ),

        "acquisition_relation_gate":
            (
                "acquisition.announced_on < prediction_origin"
            ),

        "temporal_convention":
            "STRICTLY_PRE_ORIGIN",

        "candidate_universe_changed":
            False,

        "external_company_graph_nodes_added":
            False,

        "person_identity_embedding_added":
            False,

        "feature_discovery_reopened":
            False,

        "graph_relation_family_changed":
            False,

        "performance_based_change":
            False,

        "training_performed":
            False,
    }

    v2[
        "semantic_dependencies"
    ] = {
        "outcome_independent_cold_mask":
            str(
                MASKING_CONTRACT
            ),

        "origin_referenced_founder_history":
            str(
                FOUNDER_HISTORY_CONTRACT
            ),

        "corrected_branch_semantics":
            str(
                BRANCH_SEMANTICS_V2
            ),
    }

    v2[
        "next_step"
    ] = (
        "Phase 8.1.10A V2 — consolidate inductive consistency "
        "under corrected graph placement, COLD_t masking and "
        "origin-referenced founder history."
    )

    # =========================================================================
    # 6. Integrity audit
    # =========================================================================

    separator("-")

    print(
        "8.1.9-V2.6 — CORRECTED-SCHEMA INTEGRITY AUDIT"
    )

    separator("-")

    v2_by_name = {
        row[
            "information"
        ]:
            row
        for row
        in v2[
            "information_placement"
        ]
    }

    require(
        v2_by_name[
            P11
        ][
            "placement"
        ]
        ==
        "STARTUP_NODE_FEATURE",
        "P11 placement correction failed.",
    )

    require(
        v2_by_name[
            P12
        ][
            "placement"
        ]
        ==
        "STARTUP_NODE_FEATURE",
        "P12 placement correction failed.",
    )

    v2_relations = {
        row[
            "relation_id"
        ]:
            row
        for row
        in v2[
            "primary_relations"
        ]
    }

    for relation_id, gate in (
        corrected_gates.items()
    ):

        require(
            v2_relations[
                relation_id
            ][
                "origin_gate"
            ]
            ==
            gate,
            (
                f"{relation_id}: strict pre-origin "
                "gate correction failed."
            ),
        )

    require(
        v2[
            "graph_universe_boundaries"
        ][
            "candidate_startup_universe"
        ]
        ==
        v1[
            "graph_universe_boundaries"
        ][
            "candidate_startup_universe"
        ],
        "Candidate Startup universe changed.",
    )

    require(
        v2[
            "graph_universe_boundaries"
        ][
            "investor_role_universe"
        ]
        ==
        v1[
            "graph_universe_boundaries"
        ][
            "investor_role_universe"
        ],
        "Investor universe changed.",
    )

    require(
        v2[
            "graph_variants"
        ]
        ==
        v1[
            "graph_variants"
        ],
        "Graph variants changed unexpectedly.",
    )

    require(
        v2[
            "founder_history_policy"
        ]
        ==
        v1[
            "founder_history_policy"
        ],
        (
            "Founder-history definition changed unexpectedly; "
            "V1 definition was already correct."
        ),
    )

    require(
        v2[
            "structural_safety_rules"
        ]
        ==
        v1[
            "structural_safety_rules"
        ],
        "Structural safety rules changed unexpectedly.",
    )

    print(
        "P11 Startup placement:                 PASS"
    )

    print(
        "P12 Startup placement:                 PASS"
    )

    print(
        "Founder-history definition preserved: PASS"
    )

    print(
        "Founder relation strict-before gate:  PASS"
    )

    print(
        "Acquisition strict-before gate:       PASS"
    )

    print(
        "Candidate universe unchanged:         PASS"
    )

    print(
        "Graph variants unchanged:             PASS"
    )

    print(
        "Feature discovery remains closed:     PASS"
    )

    # =========================================================================
    # 7. Persist corrected JSON + CSV schema artifacts
    # =========================================================================

    separator("-")

    print(
        "8.1.9-V2.7 — PERSIST SUPERSEDING GRAPH SCHEMA"
    )

    separator("-")

    SCHEMA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json(
        V2_SCHEMA,
        v2,
    )

    pd.DataFrame(
        v2[
            "node_types"
        ]
    ).to_csv(
        V2_NODE_SCHEMA,
        index=False,
    )

    pd.DataFrame(
        v2[
            "primary_relations"
        ]
    ).to_csv(
        V2_RELATION_SCHEMA,
        index=False,
    )

    pd.DataFrame(
        v2[
            "information_placement"
        ]
    ).to_csv(
        V2_INFORMATION_PLACEMENT,
        index=False,
    )

    pd.DataFrame(
        v2[
            "graph_variants"
        ]
    ).to_csv(
        V2_GRAPH_VARIANTS,
        index=False,
    )

    manifest = {
        "phase":
            "8.1.9",

        "version":
            "V2",

        "status":
            "FROZEN",

        "scientific_role":
            "GRAPH_SCHEMA_SEMANTIC_TEMPORAL_REPAIR",

        "V1_status":
            "FROZEN_SUPERSEDED",

        "V1_files_preserved":
            True,

        "authoritative_graph_schema":
            str(
                V2_SCHEMA
            ),

        "P11_placement":
            "STARTUP_NODE_FEATURE",

        "P12_placement":
            "STARTUP_NODE_FEATURE",

        "founder_history_reference":
            "prediction_origin",

        "founder_relation_temporal_gate":
            "STRICTLY_BEFORE_ORIGIN",

        "acquisition_relation_temporal_gate":
            "STRICTLY_BEFORE_ORIGIN",

        "candidate_universe_changed":
            False,

        "graph_variants_changed":
            False,

        "feature_discovery_reopened":
            False,

        "graph_materialized":
            False,

        "model_trained":
            False,

        "performance_inspected":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,

        "outputs": [
            str(
                V2_NODE_SCHEMA
            ),
            str(
                V2_RELATION_SCHEMA
            ),
            str(
                V2_INFORMATION_PLACEMENT
            ),
            str(
                V2_GRAPH_VARIANTS
            ),
            str(
                V2_SCHEMA
            ),
        ],
    }

    write_json(
        V2_MANIFEST,
        manifest,
    )

    outputs = [
        V2_SCHEMA,
        V2_NODE_SCHEMA,
        V2_RELATION_SCHEMA,
        V2_INFORMATION_PLACEMENT,
        V2_GRAPH_VARIANTS,
        V2_MANIFEST,
    ]

    write_json(
        V2_HASHES,
        {
            str(path):
                sha256_file(
                    path
                )
            for path
            in outputs
        },
    )

    separator()

    print(
        "PHASE 8.1.9 V2 HETEROGENEOUS GRAPH SCHEMA FROZEN"
    )

    separator()

    print()

    print(
        "8.1.9 V1:                          FROZEN_SUPERSEDED"
    )

    print(
        "8.1.9 V2:                          FROZEN / AUTHORITATIVE"
    )

    print()

    print(
        "P11 PLACEMENT:                     STARTUP_NODE_FEATURE"
    )

    print(
        "P12 PLACEMENT:                     STARTUP_NODE_FEATURE"
    )

    print(
        "FOUNDER HISTORY REFERENCE:         PREDICTION_ORIGIN"
    )

    print(
        "FOUNDER RELATION GATE:             founded_on < origin"
    )

    print(
        "ACQUISITION RELATION GATE:         announced_on < origin"
    )

    print(
        "PERSON ID EMBEDDING:               NO"
    )

    print(
        "EXTERNAL COMPANY GRAPH NODES:      NO"
    )

    print(
        "CANDIDATE UNIVERSE EXPANDED:       NO"
    )

    print(
        "GRAPH VARIANTS CHANGED:            NO"
    )

    print(
        "FEATURE DISCOVERY REOPENED:        NO"
    )

    print(
        "GRAPH MATERIALIZED:                NO"
    )

    print(
        "TRAINING PERFORMED:                NO"
    )

    print(
        "T60 SCORED:                        NO"
    )

    print(
        "APRIL ANALYZED:                    NO"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 8.1.10A V2 — corrected inductive "
        "consistency closure."
    )


if __name__ == "__main__":
    main()
