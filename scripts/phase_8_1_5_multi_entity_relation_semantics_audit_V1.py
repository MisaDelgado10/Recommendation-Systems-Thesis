from pathlib import Path
import hashlib
import json
import re

import pandas as pd


# =============================================================================
# PHASE 8.1.5 — MULTI-ENTITY RELATION SEMANTICS & RAW-FORMAT AUDIT V1
# =============================================================================
#
# Scientific role:
#
#   RELATION-DISCOVERY / FORMAT / COVERAGE AUDIT ONLY
#
# PURPOSE
# -------
#
# Examine the most scientifically plausible cross-entity fields discovered
# in Phase 8.1.4.
#
# This script does NOT construct edges.
#
# It asks:
#
#   - Is the field actually populated?
#   - Does it contain UUIDs?
#   - URLs?
#   - Crunchbase person/organization/school paths?
#   - Multi-value strings?
#   - Does its raw representation look resolvable?
#
# No actual contact/person values are printed.
#
# =============================================================================


RAW_DIR = Path(
    "data/raw"
)

PHASE814_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_4_multi_entity_schema_inventory"
)

PHASE814_SUMMARY = (
    PHASE814_DIR
    / "phase_8_1_4_schema_inventory_summary_V1.json"
)


OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_1_inductive_startup_representation/"
    "phase_8_1_5_relation_semantics"
)

PROFILE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_5_relation_field_profile_V1.csv"
)

CANDIDATE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_5_relation_candidate_register_V1.csv"
)

SURFACE_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_5_entity_linkage_surface_V1.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_5_relation_semantics_summary_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_5_analysis_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_1_5_derived_artifact_sha256_V1.json"
)


CHUNK_SIZE = 100_000


# =============================================================================
# FILE GROUPS
# =============================================================================

FILE_GLOBS = {
    "companies":
        "companies*.csv",

    "people":
        "people*.csv",

    "contacts":
        "CRUNCHBASE_contact*.csv",

    "funding":
        "CRUNCHBASE_funding*.csv",

    "acquisitions":
        "CRUNCHBASE_acquisition*.csv",

    "investors":
        "CRUNCHBASE_investor*.csv",

    "events":
        "CRUNCHBASE_event*.csv",

    "hubs":
        "CRUNCHBASE_hub*.csv",

    "schools":
        "CRUNCHBASE_school*.csv",
}


# =============================================================================
# SCIENTIFICALLY PLAUSIBLE RELATION FIELDS
# =============================================================================
#
# These are hypotheses only.
# No candidate becomes a Phase-8 graph edge in this script.
# =============================================================================

RELATION_CANDIDATES = [
    {
        "candidate_id":
            "R01",

        "source_entity":
            "companies",

        "field":
            "founder_links",

        "proposed_target":
            "people",

        "candidate_semantics":
            "COMPANY_FOUNDED_BY_PERSON",

        "priority":
            "HIGH",

        "temporal_risk":
            "SNAPSHOT_UNVERSIONED",

        "notes":
            (
                "Potential explicit replacement for the current "
                "projected shared_founder relation."
            ),
    },
    {
        "candidate_id":
            "R02",

        "source_entity":
            "companies",

        "field":
            "founders",

        "proposed_target":
            "people",

        "candidate_semantics":
            "COMPANY_FOUNDED_BY_PERSON_NAME",

        "priority":
            "MEDIUM",

        "temporal_risk":
            "SNAPSHOT_UNVERSIONED",

        "notes":
            (
                "Likely display-name representation; founder_links "
                "is preferred if resolvable."
            ),
    },
    {
        "candidate_id":
            "R03",

        "source_entity":
            "people",

        "field":
            "schools",

        "proposed_target":
            "schools",

        "candidate_semantics":
            "PERSON_EDUCATED_AT_SCHOOL",

        "priority":
            "HIGH",

        "temporal_risk":
            "SNAPSHOT_UNVERSIONED",

        "notes":
            (
                "Potential founder-background signal for "
                "investment-cold startups."
            ),
    },
    {
        "candidate_id":
            "R04",

        "source_entity":
            "people",

        "field":
            "current_organizations",

        "proposed_target":
            "companies_or_investors",

        "candidate_semantics":
            "PERSON_CURRENTLY_ASSOCIATED_WITH_ORGANIZATION",

        "priority":
            "MEDIUM",

        "temporal_risk":
            "HIGH_SNAPSHOT_RISK",

        "notes":
            (
                "Current affiliation can change after prediction origin; "
                "requires strong temporal caveat."
            ),
    },
    {
        "candidate_id":
            "R05",

        "source_entity":
            "companies",

        "field":
            "hub_tags",

        "proposed_target":
            "hubs_or_tags",

        "candidate_semantics":
            "COMPANY_ASSOCIATED_WITH_HUB",

        "priority":
            "MEDIUM",

        "temporal_risk":
            "SNAPSHOT_UNVERSIONED",

        "notes":
            (
                "Need to determine whether values correspond to "
                "actual Hub entities or only descriptive tags."
            ),
    },
    {
        "candidate_id":
            "R06",

        "source_entity":
            "contacts",

        "field":
            "organization_uuid",

        "proposed_target":
            "companies_or_investors",

        "candidate_semantics":
            "CONTACT_ASSOCIATED_WITH_ORGANIZATION",

        "priority":
            "LOW",

        "temporal_risk":
            "SNAPSHOT_UNVERSIONED",

        "notes":
            (
                "Explicit UUID relation but likely limited research value. "
                "No personal contact values will be exported."
            ),
    },
    {
        "candidate_id":
            "R07",

        "source_entity":
            "funding",

        "field":
            "investors",

        "proposed_target":
            "investors",

        "candidate_semantics":
            "FUNDING_ROUND_HAS_INVESTOR",

        "priority":
            "EXISTING",

        "temporal_risk":
            "TIMESTAMPED_BY_ANNOUNCED_ON",

        "notes":
            (
                "Already represented through audited investment "
                "interactions; not a new cold-start relation."
            ),
    },
    {
        "candidate_id":
            "R08",

        "source_entity":
            "funding",

        "field":
            "lead_investors",

        "proposed_target":
            "investors",

        "candidate_semantics":
            "FUNDING_ROUND_HAS_LEAD_INVESTOR",

        "priority":
            "SECONDARY",

        "temporal_risk":
            "TIMESTAMPED_BY_ANNOUNCED_ON",

        "notes":
            (
                "Potential investment-edge subtype; focal cold-start "
                "startup has no pre-origin investment history."
            ),
    },
    {
        "candidate_id":
            "R09",

        "source_entity":
            "acquisitions",

        "field":
            "acquirer_name",

        "proposed_target":
            "companies",

        "candidate_semantics":
            "ACQUIRER_COMPANY",

        "priority":
            "EXISTING",

        "temporal_risk":
            "TIMESTAMPED_BY_ANNOUNCED_ON",

        "notes":
            "Already covered by Phase-8 acquisition graph.",
    },
    {
        "candidate_id":
            "R10",

        "source_entity":
            "acquisitions",

        "field":
            "acquiree_name",

        "proposed_target":
            "companies",

        "candidate_semantics":
            "ACQUIRED_COMPANY",

        "priority":
            "EXISTING",

        "temporal_risk":
            "TIMESTAMPED_BY_ANNOUNCED_ON",

        "notes":
            "Already covered by Phase-8 acquisition graph.",
    },
    {
        "candidate_id":
            "R11",

        "source_entity":
            "investors",

        "field":
            "founders",

        "proposed_target":
            "people",

        "candidate_semantics":
            "INVESTOR_ORGANIZATION_FOUNDED_BY_PERSON",

        "priority":
            "LOW_CURRENT_SCOPE",

        "temporal_risk":
            "SNAPSHOT_UNVERSIONED",

        "notes":
            (
                "Could enrich investor representation, but Phase-8 "
                "current research focus is startup-side induction."
            ),
    },
]


# =============================================================================
# FORMAT PATTERNS
# =============================================================================

UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}\b"
)

URL_RE = re.compile(
    r"https?://",
    flags=re.IGNORECASE,
)

PERSON_PATH_RE = re.compile(
    r"/person/",
    flags=re.IGNORECASE,
)

ORG_PATH_RE = re.compile(
    r"/organization/",
    flags=re.IGNORECASE,
)

SCHOOL_PATH_RE = re.compile(
    r"/school/",
    flags=re.IGNORECASE,
)

HUB_PATH_RE = re.compile(
    r"/hub/",
    flags=re.IGNORECASE,
)


# =============================================================================
# HELPERS
# =============================================================================

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


def pct(
    numerator,
    denominator,
):

    if denominator == 0:

        return float("nan")

    return (
        100.0
        * float(
            numerator
        )
        / float(
            denominator
        )
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    separator()

    print(
        "PHASE 8.1.5 — "
        "MULTI-ENTITY RELATION SEMANTICS "
        "& RAW-FORMAT AUDIT V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "RELATION DISCOVERY / FORMAT AUDIT ONLY"
    )

    print(
        "Edges constructed:           NO"
    )

    print(
        "Graph changed:               NO"
    )

    print(
        "Relation selected:           NO"
    )

    print(
        "Architecture selected:       NO"
    )

    print(
        "Model trained:               NO"
    )

    print(
        "April performance inspected: NO"
    )

    # =========================================================================
    # 1. Phase-8.1.4 gate
    # =========================================================================

    separator("-")

    print(
        "8.1.5.1 — PHASE-8.1.4 GATE"
    )

    separator("-")

    require(
        PHASE814_SUMMARY.exists(),
        "Missing Phase-8.1.4 summary.",
    )

    previous = read_json(
        PHASE814_SUMMARY
    )

    require(
        previous.get(
            "status"
        )
        ==
        "COMPLETE",
        "Phase 8.1.4 is not complete.",
    )

    require(
        previous.get(
            "relations_constructed"
        )
        is False,
        (
            "Phase 8.1.4 unexpectedly "
            "constructed relations."
        ),
    )

    print(
        "Phase 8.1.4: COMPLETE PASS"
    )

    # =========================================================================
    # 2. Resolve actual file groups
    # =========================================================================

    separator("-")

    print(
        "8.1.5.2 — RAW FILE GROUPS"
    )

    separator("-")

    entity_files = {}

    for entity, pattern in (
        FILE_GLOBS.items()
    ):

        paths = sorted(
            RAW_DIR.glob(
                pattern
            )
        )

        require(
            len(
                paths
            )
            >
            0,
            (
                f"No raw files resolved for "
                f"{entity}: {pattern}"
            ),
        )

        entity_files[
            entity
        ] = paths

        print(
            f"{entity:<14} "
            f"{len(paths):>2} file(s)"
        )

    # =========================================================================
    # 3. Candidate register
    # =========================================================================

    separator("-")

    print(
        "8.1.5.3 — SCIENTIFIC RELATION "
        "CANDIDATE REGISTER"
    )

    separator("-")

    candidates = pd.DataFrame(
        RELATION_CANDIDATES
    )

    print(
        candidates[
            [
                "candidate_id",
                "source_entity",
                "field",
                "proposed_target",
                "priority",
                "temporal_risk",
            ]
        ].to_string(
            index=False
        )
    )

    # =========================================================================
    # 4. Profile raw relation-field representation
    # =========================================================================

    separator("-")

    print(
        "8.1.5.4 — RAW RELATION "
        "FIELD PROFILE"
    )

    separator("-")

    profile_rows = []

    for spec in (
        RELATION_CANDIDATES
    ):

        entity = spec[
            "source_entity"
        ]

        field = spec[
            "field"
        ]

        total_rows = 0
        files_with_field = 0

        non_null = 0
        char_total = 0

        contains_uuid = 0
        contains_url = 0

        contains_person_path = 0
        contains_org_path = 0
        contains_school_path = 0
        contains_hub_path = 0

        contains_comma = 0
        contains_semicolon = 0
        contains_pipe = 0
        contains_json_bracket = 0

        for path in (
            entity_files[
                entity
            ]
        ):

            header = pd.read_csv(
                path,
                nrows=0,
            )

            if (
                field
                not in
                header.columns
            ):

                continue

            files_with_field += 1

            for chunk in pd.read_csv(
                path,
                usecols=[
                    field
                ],
                dtype="string",
                chunksize=CHUNK_SIZE,
                low_memory=False,
            ):

                total_rows += len(
                    chunk
                )

                series = (
                    chunk[
                        field
                    ]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )

                series = series[
                    series.ne("")
                ]

                n = len(
                    series
                )

                if n == 0:
                    continue

                non_null += n

                lengths = (
                    series
                    .str.len()
                )

                char_total += int(
                    lengths.sum()
                )

                contains_uuid += int(
                    series.str.contains(
                        UUID_RE,
                        regex=True,
                    ).sum()
                )

                contains_url += int(
                    series.str.contains(
                        URL_RE,
                        regex=True,
                    ).sum()
                )

                contains_person_path += int(
                    series.str.contains(
                        PERSON_PATH_RE,
                        regex=True,
                    ).sum()
                )

                contains_org_path += int(
                    series.str.contains(
                        ORG_PATH_RE,
                        regex=True,
                    ).sum()
                )

                contains_school_path += int(
                    series.str.contains(
                        SCHOOL_PATH_RE,
                        regex=True,
                    ).sum()
                )

                contains_hub_path += int(
                    series.str.contains(
                        HUB_PATH_RE,
                        regex=True,
                    ).sum()
                )

                contains_comma += int(
                    series.str.contains(
                        ",",
                        regex=False,
                    ).sum()
                )

                contains_semicolon += int(
                    series.str.contains(
                        ";",
                        regex=False,
                    ).sum()
                )

                contains_pipe += int(
                    series.str.contains(
                        "|",
                        regex=False,
                    ).sum()
                )

                contains_json_bracket += int(
                    (
                        series.str.contains(
                            "[",
                            regex=False,
                        )
                        |
                        series.str.contains(
                            "]",
                            regex=False,
                        )
                    ).sum()
                )

        require(
            files_with_field
            >
            0,
            (
                f"{entity}.{field}: field "
                "not found in resolved files."
            ),
        )

        row = {
            "candidate_id":
                spec[
                    "candidate_id"
                ],

            "source_entity":
                entity,

            "field":
                field,

            "files_with_field":
                files_with_field,

            "total_rows":
                total_rows,

            "non_null_rows":
                non_null,

            "non_null_pct":
                pct(
                    non_null,
                    total_rows,
                ),

            "mean_chars_non_null":
                (
                    float(
                        char_total
                    )
                    /
                    float(
                        non_null
                    )
                    if
                    non_null
                    else
                    float("nan")
                ),

            "rows_containing_uuid":
                contains_uuid,

            "uuid_pct_of_non_null":
                pct(
                    contains_uuid,
                    non_null,
                ),

            "rows_containing_url":
                contains_url,

            "url_pct_of_non_null":
                pct(
                    contains_url,
                    non_null,
                ),

            "rows_with_person_path":
                contains_person_path,

            "person_path_pct":
                pct(
                    contains_person_path,
                    non_null,
                ),

            "rows_with_organization_path":
                contains_org_path,

            "organization_path_pct":
                pct(
                    contains_org_path,
                    non_null,
                ),

            "rows_with_school_path":
                contains_school_path,

            "school_path_pct":
                pct(
                    contains_school_path,
                    non_null,
                ),

            "rows_with_hub_path":
                contains_hub_path,

            "hub_path_pct":
                pct(
                    contains_hub_path,
                    non_null,
                ),

            "rows_with_comma":
                contains_comma,

            "comma_pct":
                pct(
                    contains_comma,
                    non_null,
                ),

            "rows_with_semicolon":
                contains_semicolon,

            "semicolon_pct":
                pct(
                    contains_semicolon,
                    non_null,
                ),

            "rows_with_pipe":
                contains_pipe,

            "pipe_pct":
                pct(
                    contains_pipe,
                    non_null,
                ),

            "rows_with_brackets":
                contains_json_bracket,

            "bracket_pct":
                pct(
                    contains_json_bracket,
                    non_null,
                ),
        }

        profile_rows.append(
            row
        )

        print()

        print(
            f"{spec['candidate_id']} "
            f"{entity}.{field}"
        )

        print(
            f"  populated:     "
            f"{non_null:,}/{total_rows:,} "
            f"({pct(non_null, total_rows):.4f}%)"
        )

        print(
            f"  UUID rows:     "
            f"{contains_uuid:,} "
            f"({pct(contains_uuid, non_null):.4f}%)"
        )

        print(
            f"  URL rows:      "
            f"{contains_url:,} "
            f"({pct(contains_url, non_null):.4f}%)"
        )

        print(
            f"  person path:   "
            f"{contains_person_path:,}"
        )

        print(
            f"  org path:      "
            f"{contains_org_path:,}"
        )

        print(
            f"  school path:   "
            f"{contains_school_path:,}"
        )

        print(
            f"  hub path:      "
            f"{contains_hub_path:,}"
        )

        print(
            f"  comma rows:    "
            f"{contains_comma:,}"
        )

    profile_df = pd.DataFrame(
        profile_rows
    )

    # =========================================================================
    # 5. Entity linkage surface
    # =========================================================================

    separator("-")

    print(
        "8.1.5.5 — ENTITY LINKAGE SURFACE"
    )

    separator("-")

    linkage_surface = pd.DataFrame(
        [
            {
                "entity":
                    "companies",

                "outgoing_candidate":
                    True,

                "incoming_candidate":
                    True,

                "highest_priority_path":
                    (
                        "company -> founder person"
                    ),
            },
            {
                "entity":
                    "people",

                "outgoing_candidate":
                    True,

                "incoming_candidate":
                    True,

                "highest_priority_path":
                    (
                        "person -> school; "
                        "company -> person"
                    ),
            },
            {
                "entity":
                    "schools",

                "outgoing_candidate":
                    False,

                "incoming_candidate":
                    True,

                "highest_priority_path":
                    "person -> school",
            },
            {
                "entity":
                    "hubs",

                "outgoing_candidate":
                    False,

                "incoming_candidate":
                    True,

                "highest_priority_path":
                    "company -> hub/tag candidate",
            },
            {
                "entity":
                    "contacts",

                "outgoing_candidate":
                    True,

                "incoming_candidate":
                    False,

                "highest_priority_path":
                    "contact -> organization",
            },
            {
                "entity":
                    "events",

                "outgoing_candidate":
                    False,

                "incoming_candidate":
                    False,

                "highest_priority_path":
                    (
                        "NO explicit participant "
                        "relationship established "
                        "from Phase-8.1.4 schema"
                    ),
            },
            {
                "entity":
                    "funding",

                "outgoing_candidate":
                    True,

                "incoming_candidate":
                    True,

                "highest_priority_path":
                    (
                        "existing timestamped "
                        "investment interaction"
                    ),
            },
            {
                "entity":
                    "acquisitions",

                "outgoing_candidate":
                    True,

                "incoming_candidate":
                    True,

                "highest_priority_path":
                    (
                        "existing timestamped "
                        "company acquisition relation"
                    ),
            },
            {
                "entity":
                    "investors",

                "outgoing_candidate":
                    True,

                "incoming_candidate":
                    True,

                "highest_priority_path":
                    (
                        "existing investment role; "
                        "investor-founder path secondary"
                    ),
            },
        ]
    )

    print(
        linkage_surface.to_string(
            index=False
        )
    )

    # =========================================================================
    # 6. Persist
    # =========================================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        CANDIDATE_OUTPUT,
        index=False,
    )

    profile_df.to_csv(
        PROFILE_OUTPUT,
        index=False,
    )

    linkage_surface.to_csv(
        SURFACE_OUTPUT,
        index=False,
    )

    summary = {
        "schema_version":
            "PHASE_8_1_5_RELATION_SEMANTICS_V1",

        "status":
            "COMPLETE",

        "scientific_role":
            (
                "RELATION_DISCOVERY_FORMAT_"
                "COVERAGE_AUDIT_ONLY"
            ),

        "candidate_relation_count":
            len(
                candidates
            ),

        "high_priority_candidates": [
            "R01",
            "R03",
        ],

        "medium_priority_candidates": [
            "R02",
            "R04",
            "R05",
        ],

        "existing_relation_candidates": [
            "R07",
            "R09",
            "R10",
        ],

        "contact_values_exported":
            False,

        "edges_constructed":
            False,

        "graph_changed":
            False,

        "relation_selected":
            False,

        "architecture_selected":
            False,

        "model_trained":
            False,

        "April_performance_inspected":
            False,

        "next_step":
            (
                "Build deterministic resolution audits "
                "for high-priority relation candidates "
                "before adding any graph edge."
            ),
    }

    write_json(
        SUMMARY_OUTPUT,
        summary,
    )

    manifest = {
        "phase":
            "8.1.5",

        "status":
            "COMPLETE",

        "inputs": {
            "phase_8_1_4_summary":
                str(
                    PHASE814_SUMMARY
                ),
        },

        "outputs": [
            str(
                PROFILE_OUTPUT
            ),
            str(
                CANDIDATE_OUTPUT
            ),
            str(
                SURFACE_OUTPUT
            ),
            str(
                SUMMARY_OUTPUT
            ),
        ],

        "prohibited_operations": [
            "edge_construction",
            "graph_modification",
            "relation_selection",
            "architecture_selection",
            "training",
            "inference",
            "ranking_scoring",
            "April_performance_inspection",
        ],
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    hashes = {
        str(PROFILE_OUTPUT):
            sha256_file(
                PROFILE_OUTPUT
            ),

        str(CANDIDATE_OUTPUT):
            sha256_file(
                CANDIDATE_OUTPUT
            ),

        str(SURFACE_OUTPUT):
            sha256_file(
                SURFACE_OUTPUT
            ),

        str(SUMMARY_OUTPUT):
            sha256_file(
                SUMMARY_OUTPUT
            ),

        str(MANIFEST_OUTPUT):
            sha256_file(
                MANIFEST_OUTPUT
            ),
    }

    write_json(
        HASH_OUTPUT,
        hashes,
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    separator()

    print(
        "PHASE 8.1.5 MULTI-ENTITY "
        "RELATION SEMANTICS AUDIT COMPLETE"
    )

    separator()

    print()

    print(
        "RELATION CANDIDATES PROFILED: YES"
    )

    print(
        "EDGES CONSTRUCTED:            NO"
    )

    print(
        "GRAPH MODIFIED:               NO"
    )

    print(
        "RELATION SELECTED:            NO"
    )

    print(
        "ARCHITECTURE SELECTED:        NO"
    )

    print(
        "MODEL TRAINED:                NO"
    )

    print(
        "APRIL PERFORMANCE INSPECTED:  NO"
    )

    print()

    print(
        "Next:"
    )

    print(
        "Resolve the high-priority "
        "Company→Person and Person→School "
        "relations against canonical entity IDs."
    )

    print()

    print(
        f"Outputs written to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()