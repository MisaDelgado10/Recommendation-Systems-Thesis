from pathlib import Path
import hashlib
import json


# =============================================================================
# PHASE 8.1.12B — ORIGIN-REFERENCED FOUNDER-HISTORY AMENDMENT V1
# =============================================================================
#
# Scientific role:
#
#   PRE-TRAINING TEMPORAL-SEMANTICS CORRECTION
#
# PURPOSE
# -------
#
# Restore the founder-history semantics frozen in Phase 8.1.7 V2:
#
#   founder experience must be defined relative to prediction origin t,
#   NOT relative to the focal Startup's own founding date.
#
# AUTHORITATIVE DEFINITION
# ------------------------
#
# For focal Startup s at origin t:
#
#   F(s) =
#       exact resolved Founder Person IDs of s
#
#   H(s,t) =
#       unique Companies c such that:
#
#           c != s
#
#           AND
#
#           at least one p in F(s) is a founder of c
#
#           AND
#
#           founded_on(c) is parseable
#
#           AND
#
#           founded_on(c) < t
#
# Then:
#
#   founder_prior_startup_count(s,t)
#       = |H(s,t)|
#
#   founder_prior_sector_profile(s,t)
#       = sector/category evidence from H(s,t)
#
# Companies are deduplicated by company UUID across multiple shared founders.
#
# External Companies remain aggregation context only:
#
#   - NOT candidate Startups;
#   - NOT graph Company nodes;
#   - NOT scored entities.
#
# Person nodes remain ZERO_40 and identity-free.
#
# No feature discovery, model family, architecture, optimizer, candidate
# universe, evaluation protocol or performance-based choice is changed.
#
# =============================================================================


BASE = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation"
)

INPUT_CONTRACT = (
    BASE
    / "phase_8_1_7_input_contract_amendment"
    / "phase_8_1_7_contract_amendment_V2.json"
)

CONSISTENCY_CONTRACT = (
    BASE
    / "phase_8_1_10a_inductive_consistency_closure"
    / "phase_8_1_10a_consistency_amendment_V1.json"
)

MASKING_AMENDMENT = (
    BASE
    / "phase_8_1_12a_outcome_independent_cold_start_masking"
    / "phase_8_1_12a_outcome_independent_masking_contract_V1.json"
)

EXECUTION_CONTRACT = (
    BASE
    / "phase_8_1_12_execution_contract"
    / "phase_8_1_12_execution_contract_V3.json"
)

FOUNDER_HISTORY_MANIFEST = Path(
    "data/experimental/phase_8/"
    "phase_8_2_framework_development/"
    "phase_8_2_1_feature_materialization/"
    "phase_8_2_1b_manifest_V1.json"
)


OUTPUT_DIR = (
    BASE
    / "phase_8_1_12b_origin_referenced_founder_history"
)

CONTRACT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_12b_origin_referenced_founder_history_contract_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_12b_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_12b_artifact_sha256_V1.json"
)


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
        "PHASE 8.1.12B — "
        "ORIGIN-REFERENCED FOUNDER-HISTORY AMENDMENT V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "PRE-TRAINING TEMPORAL-SEMANTICS CORRECTION"
    )

    print("Feature discovery reopened:  NO")
    print("Graph schema reopened:       NO")
    print("Architecture changed:        NO")
    print("Model family changed:        NO")
    print("Optimizer changed:           NO")
    print("Training performed:          NO")
    print("Performance inspected:       NO")
    print("T60 scored:                  NO")
    print("April analyzed:              NO")

    # =========================================================================
    # 1. Prerequisite gates
    # =========================================================================

    separator("-")

    print(
        "8.1.12B.1 — PREREQUISITE GATES"
    )

    separator("-")

    required = [
        INPUT_CONTRACT,
        CONSISTENCY_CONTRACT,
        MASKING_AMENDMENT,
        EXECUTION_CONTRACT,
        FOUNDER_HISTORY_MANIFEST,
    ]

    for path in required:

        require(
            path.exists(),
            f"Missing prerequisite: {path}",
        )

        print(
            f"FOUND {path}"
        )

    input_contract = read_json(
        INPUT_CONTRACT
    )

    consistency = read_json(
        CONSISTENCY_CONTRACT
    )

    masking = read_json(
        MASKING_AMENDMENT
    )

    execution = read_json(
        EXECUTION_CONTRACT
    )

    founder_manifest = read_json(
        FOUNDER_HISTORY_MANIFEST
    )

    require(
        input_contract.get("status")
        ==
        "FROZEN",
        "Phase 8.1.7 V2 is not frozen.",
    )

    require(
        consistency.get("status")
        ==
        "FROZEN",
        "Phase 8.1.10A is not frozen.",
    )

    require(
        masking.get("status")
        ==
        "FROZEN",
        "Phase 8.1.12A is not frozen.",
    )

    require(
        execution.get("status")
        ==
        "FROZEN",
        "Phase 8.1.12 V3 is not frozen.",
    )

    require(
        founder_manifest.get("status")
        ==
        "MATERIALIZED_AND_AUDITED",
        "Phase 8.2.1B V1 is not materialized/audited.",
    )

    require(
        execution.get(
            "training_performed"
        )
        is False,
        "Training already occurred."
    )

    require(
        execution.get(
            "T60_scored"
        )
        is False,
        "T60 already scored."
    )

    require(
        execution.get(
            "April_analyzed"
        )
        is False,
        "April already analyzed."
    )

    # =========================================================================
    # 2. Confirm founder-history channels remain the frozen channels
    # =========================================================================

    separator("-")

    print(
        "8.1.12B.2 — FROZEN CHANNEL RECHECK"
    )

    separator("-")

    input_text = json.dumps(
        input_contract,
        sort_keys=True,
    ).casefold()

    require(
        "founder_prior_startup_count"
        in
        input_text,
        (
            "Frozen P11 founder_prior_startup_count "
            "not found in 8.1.7 V2."
        ),
    )

    require(
        "founder_prior_sector_profile"
        in
        input_text,
        (
            "Frozen P12 founder_prior_sector_profile "
            "not found in 8.1.7 V2."
        ),
    )

    print(
        "P11 founder_prior_startup_count:  FOUND"
    )

    print(
        "P12 founder_prior_sector_profile: FOUND"
    )

    print(
        "New founder feature introduced:   NO"
    )

    # =========================================================================
    # 3. Confirm materialized V1 semantic drift
    # =========================================================================

    separator("-")

    print(
        "8.1.12B.3 — MATERIALIZED V1 SEMANTIC DRIFT"
    )

    separator("-")

    old_cutoff = founder_manifest.get(
        "founder_history_cutoff"
    )

    print(
        f"8.2.1B V1 cutoff: {old_cutoff}"
    )

    require(
        old_cutoff
        ==
        (
            "prior_company.founded_on "
            "< focal_startup.founded_on"
        ),
        (
            "Unexpected Phase 8.2.1B V1 cutoff; "
            "manual inspection required."
        ),
    )

    issue = {
        "materialized_V1_cutoff":
            old_cutoff,

        "scientific_problem":
            (
                "The focal Startup founding date is earlier than "
                "the prediction origin and therefore discards founder "
                "experience that became observable after the focal "
                "Startup was founded but before prediction."
            ),

        "example_logic":
            (
                "If focal Startup A was founded in 2022, founder P "
                "founded Company B in 2023, and prediction origin is "
                "2025, Company B is valid pre-origin founder experience "
                "and must not be excluded solely because it postdates "
                "A's founding."
            ),

        "classification":
            "TEMPORAL_REFERENCE_SEMANTIC_DRIFT",

        "performance_tuning_issue":
            False,

        "new_feature_issue":
            False,

        "requires_pretraining_correction":
            True,
    }

    for key, value in issue.items():

        print(
            f"{key:<36} {value}"
        )

    # =========================================================================
    # 4. Authoritative founder-history definition
    # =========================================================================

    separator("-")

    print(
        "8.1.12B.4 — AUTHORITATIVE ORIGIN-REFERENCED DEFINITION"
    )

    separator("-")

    definition = {
        "placement":
            "STARTUP_NODE_FEATURE_ORIGIN_SPECIFIC",

        "reference_time":
            "PREDICTION_ORIGIN_t",

        "resolved_founders":
            (
                "Exact Founder Person IDs attached to focal Startup s."
            ),

        "eligible_prior_company_rule":
            (
                "Unique Company c where c != s, at least one resolved "
                "Founder Person of s founded c, founded_on(c) is "
                "parseable, and founded_on(c) < prediction origin t."
            ),

        "focal_startup_excluded":
            True,

        "deduplicate_prior_company_by":
            "company_uuid",

        "deduplicate_across_multiple_founders":
            True,

        "missing_prior_company_founded_on":
            "EXCLUDE_FROM_HISTORY",

        "P11_founder_prior_startup_count":
            (
                "Number of unique eligible prior Companies H(s,t)."
            ),

        "P12_founder_prior_sector_profile":
            (
                "Sector/category evidence from the same unique "
                "eligible prior Company set H(s,t)."
            ),

        "sector_source_temporal_caveat":
            (
                "Company category labels are matched-snapshot "
                "semantics; Company eligibility itself is gated "
                "by founded_on < origin."
            ),

        "external_companies_allowed_for_aggregation":
            True,

        "external_companies_added_to_candidate_universe":
            False,

        "external_companies_added_as_graph_nodes":
            False,

        "person_identity_embedding":
            False,

        "person_initial_feature":
            "ZERO_40",

        "edge_attribute_required":
            False,

        "startup_level_aggregation":
            True,
    }

    for key, value in definition.items():

        print(
            f"{key:<46} {value}"
        )

    # =========================================================================
    # 5. Origin-specific behavior
    # =========================================================================

    separator("-")

    print(
        "8.1.12B.5 — ORIGIN-SPECIFIC FEATURE SNAPSHOTS"
    )

    separator("-")

    origins = {
        "T57":
            (
                "Compute H(s,T57) using Companies founded strictly "
                "before the frozen T57 prediction origin."
            ),

        "T58":
            (
                "Compute H(s,T58) using Companies founded strictly "
                "before the frozen T58 prediction origin."
            ),

        "T59":
            (
                "Compute H(s,T59) using Companies founded strictly "
                "before the frozen T59 prediction origin."
            ),

        "T60":
            (
                "Compute H(s,T60) using Companies founded strictly "
                "before the frozen T60 prediction origin."
            ),

        "APRIL_CONFIRMATORY":
            (
                "Compute H(s,April) using Companies founded strictly "
                "before the frozen April confirmatory origin."
            ),
    }

    for origin, rule in origins.items():

        print(
            f"{origin:<22} {rule}"
        )

    print()

    print(
        "One global founder-history vector reused across origins: NO"
    )

    print(
        "Founder-history information may increase over origins:  YES"
    )

    # =========================================================================
    # 6. Interaction with outcome-independent cold masking
    # =========================================================================

    separator("-")

    print(
        "8.1.12B.6 — INTERACTION WITH 8.1.12A COLD MASK"
    )

    separator("-")

    masking_interaction = {
        "8_1_12A_remains_authoritative":
            True,

        "cold_population":
            (
                "ALL origin-eligible Startups with zero canonical "
                "investment interactions strictly before origin."
            ),

        "founder_history_computed_for_cold_nodes":
            True,

        "founder_history_visible_during_parameter_fit_for_cold_nodes":
            False,

        "founder_history_restored_at_inductive_inference":
            True,

        "positive_target_ids_used":
            False,

        "negative_candidate_ids_used":
            False,

        "candidate_matrix_used":
            False,

        "future_outcomes_used":
            False,
    }

    for key, value in masking_interaction.items():

        print(
            f"{key:<56} {value}"
        )

    # =========================================================================
    # 7. Supersession scope
    # =========================================================================

    separator("-")

    print(
        "8.1.12B.7 — CLAUSE-LEVEL SUPERSESSION"
    )

    separator("-")

    supersession = {
        "files_deleted_or_overwritten":
            False,

        "phase_8_1_7_V2":
            (
                "REMAINS AUTHORITATIVE; this amendment restores "
                "its pre-origin founder-history semantics."
            ),

        "phase_8_1_10A_V1":
            (
                "SUPERSEDED ONLY where founder history uses "
                "focal Startup founded_on as the reference cutoff."
            ),

        "phase_8_1_12A_V1":
            (
                "OUTCOME-INDEPENDENT COLD MASK REMAINS AUTHORITATIVE. "
                "Its retained 'pre-focal founder-history cutoff' clause "
                "is superseded by this amendment."
            ),

        "phase_8_2_1B_V1_identity_resolution":
            "REUSABLE_AND_AUTHORITATIVE",

        "phase_8_2_1B_V1_company_founder_edges":
            "REUSABLE_AND_AUTHORITATIVE",

        "phase_8_2_1B_V1_startup_founder_edges":
            "REUSABLE_AND_AUTHORITATIVE",

        "phase_8_2_1B_V1_person_node_source":
            "REUSABLE_AND_AUTHORITATIVE",

        "phase_8_2_1B_V1_prefocal_history_pairs":
            "SUPERSEDED_FOR_PREDICTIVE_USE",

        "phase_8_2_1B_V1_founder_history_source":
            "SUPERSEDED_FOR_PREDICTIVE_USE",

        "V1_artifacts_preserved_for_audit":
            True,
    }

    for key, value in supersession.items():

        print(
            f"{key:<48} {value}"
        )

    # =========================================================================
    # 8. Scientific status
    # =========================================================================

    separator("-")

    print(
        "8.1.12B.8 — SCIENTIFIC STATUS"
    )

    separator("-")

    scientific_status = {
        "feature_discovery_closed":
            True,

        "graph_schema_closed":
            True,

        "architecture_closed":
            True,

        "optimizer_closed":
            True,

        "evaluation_protocol_closed":
            True,

        "method_design_closed":
            True,

        "correction_class":
            "TEMPORAL_SEMANTICS_INTEGRITY_AMENDMENT",

        "performance_based_change":
            False,

        "new_information_channel_added":
            False,

        "new_model_family_added":
            False,

        "candidate_universe_changed":
            False,

        "training_performed":
            False,

        "performance_inspected":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,
    }

    for key, value in scientific_status.items():

        print(
            f"{key:<40} {value}"
        )

    # =========================================================================
    # 9. Persist
    # =========================================================================

    separator("-")

    print(
        "8.1.12B.9 — PERSIST TEMPORAL-SEMANTICS AMENDMENT"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    contract = {
        "schema_version":
            "PHASE_8_1_12B_ORIGIN_REFERENCED_FOUNDER_HISTORY_V1",

        "phase":
            "8.1.12B",

        "status":
            "FROZEN",

        "scientific_role":
            "PRETRAINING_TEMPORAL_SEMANTICS_CORRECTION",

        "issue":
            issue,

        "authoritative_founder_history_definition":
            definition,

        "origin_specific_snapshots":
            origins,

        "cold_mask_interaction":
            masking_interaction,

        "supersession":
            supersession,

        "scientific_status":
            scientific_status,

        "authoritative_interpretation":
            (
                "Founder-history features are Startup-conditioned and "
                "origin-specific. For focal Startup s at prediction origin t, "
                "the historical Company set contains unique non-focal "
                "Companies founded by any resolved founder of s with "
                "parseable founded_on strictly before t. The Company set "
                "is deduplicated by UUID across founders. P11 counts this "
                "set; P12 derives sector evidence from the same set."
            ),

        "training_allowed_now":
            False,

        "next_step":
            (
                "Phase 8.2.1B V2 — reuse V1 founder identity artifacts "
                "and rematerialize only origin-specific founder-history "
                "pairs/features for T57/T58/T59/T60/April."
            ),
    }

    write_json(
        CONTRACT_OUTPUT,
        contract,
    )

    manifest = {
        "phase":
            "8.1.12B",

        "status":
            "FROZEN",

        "correction_class":
            "TEMPORAL_SEMANTICS_INTEGRITY_AMENDMENT",

        "upstream_files_preserved":
            True,

        "8_2_1B_V1_identity_artifacts_reusable":
            True,

        "8_2_1B_V1_history_artifacts_superseded":
            True,

        "feature_discovery_reopened":
            False,

        "architecture_changed":
            False,

        "training_performed":
            False,

        "performance_inspected":
            False,

        "T60_scored":
            False,

        "April_analyzed":
            False,

        "contract":
            str(
                CONTRACT_OUTPUT
            ),
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    write_json(
        HASH_OUTPUT,
        {
            str(CONTRACT_OUTPUT):
                sha256_file(
                    CONTRACT_OUTPUT
                ),

            str(MANIFEST_OUTPUT):
                sha256_file(
                    MANIFEST_OUTPUT
                ),
        },
    )

    separator()

    print(
        "PHASE 8.1.12B ORIGIN-REFERENCED "
        "FOUNDER HISTORY FROZEN"
    )

    separator()

    print()

    print(
        "SUPERSEDED HISTORY CUTOFF:"
    )

    print(
        "  prior_company.founded_on "
        "< focal_startup.founded_on"
    )

    print()

    print(
        "AUTHORITATIVE HISTORY CUTOFF:"
    )

    print(
        "  prior_company.founded_on "
        "< prediction_origin"
    )

    print()

    print(
        "FOCAL STARTUP EXCLUDED:              YES"
    )

    print(
        "PRIOR COMPANY UUID DEDUP:            YES"
    )

    print(
        "FEATURE IS ORIGIN-SPECIFIC:          YES"
    )

    print(
        "EXTERNAL COMPANY CANDIDATES ADDED:   NO"
    )

    print(
        "PERSON ID EMBEDDINGS:                NO"
    )

    print(
        "8.1.12A COLD MASK RETAINED:          YES"
    )

    print(
        "FEATURE DISCOVERY REOPENED:          NO"
    )

    print(
        "TRAINING PERFORMED:                  NO"
    )

    print(
        "T60 SCORED:                          NO"
    )

    print(
        "APRIL ANALYZED:                      NO"
    )

    print()

    print(
        "NEXT:"
    )

    print(
        "Phase 8.2.1B V2 — origin-specific founder-history "
        "rematerialization using the already-audited V1 "
        "identity-resolution artifacts."
    )


if __name__ == "__main__":
    main()
