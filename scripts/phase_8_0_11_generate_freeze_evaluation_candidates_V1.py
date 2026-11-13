from pathlib import Path
from collections import defaultdict
import hashlib
import json

import numpy as np
import pandas as pd


# =============================================================================
# PHASE 8.0.11 — GENERATE & FREEZE EVALUATION CANDIDATES V1
# =============================================================================
#
# Scientific role:
#
#   OUTCOME-BLIND FIXED EVALUATION-CANDIDATE GENERATION
#
# For every Phase-8 eligible evaluation event:
#
#     1 focal positive
#     +
#     99 distinct uniformly sampled negatives
#
# Negative universe:
#
#     origin-specific Startup candidate universe
#
# Exclude:
#
#     - focal positive Startup;
#     - Startups historically positive for the focal Investor
#       at or before the prediction origin.
#
# DO NOT exclude:
#
#     - other positives from the same evaluation fold;
#     - future-fold positives;
#     - April labels when constructing historical folds.
#
# All candidate lists are generated exactly once and frozen before
# model construction/training/selection.
#
# NO:
#
#     - model loading
#     - training
#     - inference
#     - ranking scoring
#     - checkpoint selection
#     - negative resampling after this freeze
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

CANONICAL_PATH = Path(
    "data/processed/interactions.parquet"
)

TEMPORAL_PATH = Path(
    "data/experimental/phase_2/"
    "temporal/interactions_itrs_temporal.parquet"
)

NODE_INDEX_PATH = Path(
    "data/experimental/phase_3/model_ready/"
    "node_index.parquet"
)


PHASE809_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_9_origin_eligibility_candidate_universe"
)

CANDIDATE_UNIVERSE_PATH = (
    PHASE809_DIR
    / "phase_8_0_9_origin_candidate_universe_V5.parquet"
)

ELIGIBLE_TARGETS_PATH = (
    PHASE809_DIR
    / "phase_8_0_9_origin_eligible_targets_V5.parquet"
)


PHASE810_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_10_evaluation_protocol"
)

PROTOCOL_PATH = (
    PHASE810_DIR
    / "phase_8_0_10_evaluation_protocol_V1.json"
)


OUTPUT_DIR = Path(
    "data/experimental/phase_8/"
    "phase_8_0_evaluation_reset/"
    "phase_8_0_11_evaluation_candidates"
)

CASE_MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_11_evaluation_case_manifest_V1.parquet"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_11_candidate_generation_summary_V1.csv"
)

COLLISION_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_11_same_fold_positive_collision_summary_V1.csv"
)

CONTRACT_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_11_candidate_runtime_contract_V1.json"
)

MANIFEST_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_11_manifest_V1.json"
)

HASH_OUTPUT = (
    OUTPUT_DIR
    / "phase_8_0_11_derived_artifact_sha256_V1.json"
)


# =============================================================================
# FROZEN INPUT HASHES
# =============================================================================

EXPECTED_TEMPORAL_SHA256 = (
    "8846c044f27d1ab317fd9d05ff75442a7"
    "fd9759351c5906fc9e2483e755fcb97"
)

EXPECTED_CANONICAL_SHA256 = (
    "03d02c48dc8bc67981fa27b66714641f6"
    "fdf1d8dd907348862517740ba77db8b"
)

EXPECTED_CANDIDATE_UNIVERSE_SHA256 = (
    "a9a52491ca04c5591ce7b912e7a43ff"
    "022c2781a9c4cbeae2748f8cffeb6e02c"
)

EXPECTED_ELIGIBLE_TARGETS_SHA256 = (
    "6d4eaffdcc8991e79ee2caf89e1ba500"
    "869ab482699f771107b89be4eda4d4a7"
)


# =============================================================================
# FROZEN ORIGIN CONTRACT
# =============================================================================

ORIGINS = [
    {
        "origin": "T57",
        "history_end": "2025-03-31",
        "expected_cases": 21_924,
        "expected_primary": 6_850,
    },
    {
        "origin": "T58",
        "history_end": "2025-06-30",
        "expected_cases": 22_384,
        "expected_primary": 6_872,
    },
    {
        "origin": "T59",
        "history_end": "2025-09-30",
        "expected_cases": 22_650,
        "expected_primary": 6_764,
    },
    {
        "origin": "APRIL_CONFIRMATORY",
        "history_end": "2026-03-31",
        "expected_cases": 6_313,
        "expected_primary": 1_690,
    },
]


EXPECTED_CANDIDATE_UNIVERSE = {
    "T57": 308_027,
    "T58": 308_854,
    "T59": 309_611,
    "APRIL_CONFIRMATORY": 311_266,
}


# =============================================================================
# SAMPLING CONTRACT
# =============================================================================

BASE_SEED = 42

NEGATIVES_PER_CASE = 99

RNG_NAMESPACE = (
    "ITRS_PHASE8_EVAL_NEGATIVE"
)


# =============================================================================
# UTILITIES
# =============================================================================

def separator(
    char="=",
    width=118,
):
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


def case_seed(
    origin,
    interaction_id,
):

    payload = (
        f"{RNG_NAMESPACE}|"
        f"{BASE_SEED}|"
        f"{origin}|"
        f"{interaction_id}"
    )

    digest = hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).digest()

    # Deterministic unsigned 64-bit seed.
    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    )


def build_history_map(
    interactions,
    history_end,
):

    history = interactions[
        interactions[
            "announced_on"
        ]
        <= history_end
    ]

    mapping = defaultdict(
        set
    )

    for row in history[
        [
            "investor_id",
            "startup_id",
        ]
    ].itertuples(
        index=False
    ):

        mapping[
            str(
                row.investor_id
            )
        ].add(
            str(
                row.startup_id
            )
        )

    return mapping


# =============================================================================
# MAIN
# =============================================================================

def main():

    separator()

    print(
        "PHASE 8.0.11 — "
        "GENERATE & FREEZE EVALUATION "
        "CANDIDATES V1"
    )

    separator()

    print()

    print(
        "Scientific role:             "
        "OUTCOME-BLIND FIXED-CANDIDATE FREEZE"
    )

    print("Model loaded:                NO")
    print("Checkpoint loaded:           NO")
    print("Training performed:          NO")
    print("Inference executed:          NO")
    print("Ranking metrics scored:      NO")
    print("Model selection performed:   NO")
    print("Evaluation negatives:        GENERATE ONCE")
    print("April performance inspected: NO")

    # =========================================================================
    # 1. Input integrity
    # =========================================================================

    separator("-")

    print(
        "8.0.11.1 — AUTHORITATIVE "
        "INPUT INTEGRITY"
    )

    separator("-")

    required_paths = [
        CANONICAL_PATH,
        TEMPORAL_PATH,
        NODE_INDEX_PATH,
        CANDIDATE_UNIVERSE_PATH,
        ELIGIBLE_TARGETS_PATH,
        PROTOCOL_PATH,
    ]

    source_hashes = {}

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

        observed = sha256_file(
            path
        )

        source_hashes[
            str(path)
        ] = observed

        print(
            f"FOUND  {path}"
        )

        print(
            f"SHA256 {observed}"
        )

    if (
        source_hashes[
            str(TEMPORAL_PATH)
        ]
        != EXPECTED_TEMPORAL_SHA256
    ):

        raise ValueError(
            "Temporal source fingerprint drift."
        )

    if (
        source_hashes[
            str(CANONICAL_PATH)
        ]
        != EXPECTED_CANONICAL_SHA256
    ):

        raise ValueError(
            "Canonical source fingerprint drift."
        )

    if (
        source_hashes[
            str(CANDIDATE_UNIVERSE_PATH)
        ]
        != EXPECTED_CANDIDATE_UNIVERSE_SHA256
    ):

        raise ValueError(
            "Phase-8.0.9 candidate-universe "
            "fingerprint drift."
        )

    if (
        source_hashes[
            str(ELIGIBLE_TARGETS_PATH)
        ]
        != EXPECTED_ELIGIBLE_TARGETS_SHA256
    ):

        raise ValueError(
            "Phase-8.0.9 eligible-target "
            "fingerprint drift."
        )

    with open(
        PROTOCOL_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        protocol = json.load(
            f
        )

    if (
        protocol.get(
            "status"
        )
        !=
        "FROZEN"
    ):

        raise ValueError(
            "Phase-8.0.10 protocol "
            "is not frozen."
        )

    print()

    print(
        "Authoritative inputs: PASS"
    )

    # =========================================================================
    # 2. Startup node-index integrity
    # =========================================================================

    separator("-")

    print(
        "8.0.11.2 — STARTUP NODE-INDEX "
        "INTEGRITY"
    )

    separator("-")

    nodes = pd.read_parquet(
        NODE_INDEX_PATH,
        columns=[
            "node_index",
            "node_type",
            "raw_entity_id",
        ],
    )

    startups = nodes[
        nodes[
            "node_type"
        ]
        ==
        "startup"
    ].copy()

    startups[
        "startup_id"
    ] = (
        startups[
            "raw_entity_id"
        ].astype(str)
    )

    if (
        startups[
            "startup_id"
        ].duplicated().any()
    ):

        raise ValueError(
            "Duplicate Startup IDs in "
            "node index."
        )

    if (
        startups[
            "node_index"
        ].duplicated().any()
    ):

        raise ValueError(
            "Duplicate Startup node indices."
        )

    if (
        len(startups)
        != 311_589
    ):

        raise ValueError(
            "Startup node count drift."
        )

    startup_to_node = (
        startups
        .set_index(
            "startup_id"
        )[
            "node_index"
        ]
        .astype(
            np.int64
        )
        .to_dict()
    )

    print(
        f"Startup node mappings: "
        f"{len(startup_to_node):,} PASS"
    )

    # =========================================================================
    # 3. Load frozen candidate and target universes
    # =========================================================================

    separator("-")

    print(
        "8.0.11.3 — LOAD FROZEN "
        "EVALUATION UNIVERSES"
    )

    separator("-")

    candidate_universe = (
        pd.read_parquet(
            CANDIDATE_UNIVERSE_PATH
        )
    )

    targets = pd.read_parquet(
        ELIGIBLE_TARGETS_PATH
    )

    candidate_universe[
        "startup_id"
    ] = (
        candidate_universe[
            "startup_id"
        ].astype(str)
    )

    targets[
        "investor_id"
    ] = (
        targets[
            "investor_id"
        ].astype(str)
    )

    targets[
        "startup_id_str"
    ] = (
        targets[
            "startup_id_str"
        ].astype(str)
    )

    # Every origin must be present exactly once as a universe.
    for spec in ORIGINS:

        origin = spec[
            "origin"
        ]

        n_candidates = int(
            (
                candidate_universe[
                    "origin"
                ]
                ==
                origin
            ).sum()
        )

        if (
            n_candidates
            !=
            EXPECTED_CANDIDATE_UNIVERSE[
                origin
            ]
        ):

            raise ValueError(
                f"{origin}: frozen candidate "
                "universe count drift."
            )

        origin_targets = targets[
            targets[
                "origin"
            ]
            ==
            origin
        ]

        if (
            len(origin_targets)
            !=
            spec[
                "expected_cases"
            ]
        ):

            raise ValueError(
                f"{origin}: eligible target "
                "count drift."
            )

        if (
            int(
                origin_targets[
                    "is_primary_stratum"
                ].sum()
            )
            !=
            spec[
                "expected_primary"
            ]
        ):

            raise ValueError(
                f"{origin}: primary target "
                "count drift."
            )

    print(
        "Frozen universe counts: PASS"
    )

    # =========================================================================
    # 4. Load historical interaction evidence
    # =========================================================================

    separator("-")

    print(
        "8.0.11.4 — HISTORICAL "
        "POSITIVE-PAIR SOURCE"
    )

    separator("-")

    canonical = pd.read_parquet(
        CANONICAL_PATH,
        columns=[
            "interaction_id",
            "announced_on",
            "investor_id",
            "startup_id",
        ],
    )

    canonical[
        "announced_on"
    ] = pd.to_datetime(
        canonical[
            "announced_on"
        ],
        errors="raise",
    )

    canonical[
        "investor_id"
    ] = (
        canonical[
            "investor_id"
        ].astype(str)
    )

    canonical[
        "startup_id"
    ] = (
        canonical[
            "startup_id"
        ].astype(str)
    )

    print(
        f"Canonical interactions: "
        f"{len(canonical):,}"
    )

    # =========================================================================
    # 5. Generate candidates origin by origin
    # =========================================================================

    separator("-")

    print(
        "8.0.11.5 — FIXED NEGATIVE "
        "GENERATION"
    )

    separator("-")

    matrices = {}

    case_frames = []

    summary_rows = []

    collision_rows = []

    for spec in ORIGINS:

        origin = spec[
            "origin"
        ]

        history_end = pd.Timestamp(
            spec[
                "history_end"
            ]
        )

        print()

        separator("-")

        print(
            f"ORIGIN {origin}"
        )

        separator("-")

        universe = (
            candidate_universe[
                candidate_universe[
                    "origin"
                ]
                ==
                origin
            ][
                [
                    "startup_id",
                ]
            ]
            .drop_duplicates()
            .sort_values(
                "startup_id"
            )
            .reset_index(
                drop=True
            )
        )

        if (
            len(universe)
            !=
            EXPECTED_CANDIDATE_UNIVERSE[
                origin
            ]
        ):

            raise ValueError(
                f"{origin}: candidate "
                "deduplication changed size."
            )

        universe_ids = (
            universe[
                "startup_id"
            ].to_numpy(
                dtype=object
            )
        )

        universe_id_set = set(
            universe_ids.tolist()
        )

        missing_node_ids = [
            startup_id
            for startup_id
            in universe_ids
            if startup_id
            not in startup_to_node
        ]

        if missing_node_ids:

            raise ValueError(
                f"{origin}: "
                f"{len(missing_node_ids):,} "
                "candidate startups missing "
                "from node index."
            )

        universe_node_indices = np.array(
            [
                startup_to_node[
                    startup_id
                ]
                for startup_id
                in universe_ids
            ],
            dtype=np.int64,
        )

        fold_targets = (
            targets[
                targets[
                    "origin"
                ]
                ==
                origin
            ]
            .copy()
            .sort_values(
                [
                    "interaction_id",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        if (
            len(fold_targets)
            !=
            spec[
                "expected_cases"
            ]
        ):

            raise ValueError(
                f"{origin}: target count "
                "changed after ordering."
            )

        # ---------------------------------------------------------------------
        # Historical positives available at origin
        # ---------------------------------------------------------------------

        historical_positive_map = (
            build_history_map(
                canonical,
                history_end,
            )
        )

        # ---------------------------------------------------------------------
        # Same-fold positives are built ONLY for a post-generation
        # collision diagnostic.
        #
        # They are NEVER consulted when deciding whether a sampled
        # Startup is eligible.
        # ---------------------------------------------------------------------

        same_fold_positive_map = defaultdict(
            set
        )

        for row in fold_targets[
            [
                "investor_id",
                "startup_id_str",
            ]
        ].itertuples(
            index=False
        ):

            same_fold_positive_map[
                str(
                    row.investor_id
                )
            ].add(
                str(
                    row.startup_id_str
                )
            )

        negative_matrix = np.empty(
            (
                len(
                    fold_targets
                ),
                NEGATIVES_PER_CASE,
            ),
            dtype=np.int64,
        )

        pool_sizes = np.empty(
            len(
                fold_targets
            ),
            dtype=np.int64,
        )

        collision_case_count = 0
        collision_slot_count = 0

        manifest_rows = []

        for case_index, row in enumerate(
            fold_targets.itertuples(
                index=False
            )
        ):

            investor_id = str(
                row.investor_id
            )

            focal_startup = str(
                row.startup_id_str
            )

            interaction_id = str(
                row.interaction_id
            )

            if (
                focal_startup
                not in universe_id_set
            ):

                raise ValueError(
                    f"{origin}: focal positive "
                    f"{interaction_id} outside "
                    "candidate universe."
                )

            historical_positives = (
                historical_positive_map.get(
                    investor_id,
                    set(),
                )
            )

            # Count the exact exclusion union inside the
            # origin-specific candidate universe.
            historical_in_universe = (
                historical_positives
                &
                universe_id_set
            )

            excluded_count = len(
                historical_in_universe
                |
                {
                    focal_startup
                }
            )

            eligible_pool_size = (
                len(
                    universe_ids
                )
                -
                excluded_count
            )

            if (
                eligible_pool_size
                <
                NEGATIVES_PER_CASE
            ):

                raise ValueError(
                    f"{origin}: case "
                    f"{interaction_id} has only "
                    f"{eligible_pool_size:,} "
                    "eligible negatives."
                )

            pool_sizes[
                case_index
            ] = eligible_pool_size

            rng = np.random.Generator(
                np.random.PCG64(
                    case_seed(
                        origin,
                        interaction_id,
                    )
                )
            )

            accepted_positions = []
            accepted_ids = set()

            # Uniform rejection sampling.
            #
            # Draw from the full frozen universe and reject:
            #
            #   - focal Startup
            #   - historical positive for Investor
            #   - duplicate accepted negative
            #
            # Sequential rejection from uniform universe produces
            # a uniform without-replacement sample from the eligible pool.
            while (
                len(
                    accepted_positions
                )
                <
                NEGATIVES_PER_CASE
            ):

                candidate_position = int(
                    rng.integers(
                        0,
                        len(
                            universe_ids
                        ),
                    )
                )

                candidate_id = str(
                    universe_ids[
                        candidate_position
                    ]
                )

                if (
                    candidate_id
                    ==
                    focal_startup
                ):
                    continue

                if (
                    candidate_id
                    in
                    historical_positives
                ):
                    continue

                if (
                    candidate_id
                    in
                    accepted_ids
                ):
                    continue

                accepted_ids.add(
                    candidate_id
                )

                accepted_positions.append(
                    candidate_position
                )

            accepted_positions = np.asarray(
                accepted_positions,
                dtype=np.int64,
            )

            negative_node_indices = (
                universe_node_indices[
                    accepted_positions
                ]
            )

            if (
                len(
                    np.unique(
                        negative_node_indices
                    )
                )
                !=
                NEGATIVES_PER_CASE
            ):

                raise ValueError(
                    f"{origin}: duplicate "
                    "negative node indices."
                )

            positive_node_index = int(
                startup_to_node[
                    focal_startup
                ]
            )

            if (
                positive_node_index
                in
                set(
                    negative_node_indices.tolist()
                )
            ):

                raise ValueError(
                    f"{origin}: focal positive "
                    "sampled as negative."
                )

            negative_matrix[
                case_index,
                :
            ] = negative_node_indices

            # -------------------------------------------------------------
            # Post-generation same-fold collision diagnostic.
            #
            # This information NEVER changes the sampled list.
            # -------------------------------------------------------------

            same_fold_others = (
                same_fold_positive_map.get(
                    investor_id,
                    set(),
                )
                -
                {
                    focal_startup
                }
            )

            sampled_ids = {
                str(
                    universe_ids[
                        position
                    ]
                )
                for position
                in accepted_positions
            }

            collision_count = len(
                sampled_ids
                &
                same_fold_others
            )

            if collision_count > 0:

                collision_case_count += 1

                collision_slot_count += (
                    collision_count
                )

            manifest_rows.append(
                {
                    "origin":
                        origin,

                    "case_index":
                        case_index,

                    "interaction_id":
                        interaction_id,

                    "investor_id":
                        investor_id,

                    "positive_startup_id":
                        focal_startup,

                    "positive_node_index":
                        positive_node_index,

                    "is_primary_stratum":
                        bool(
                            row.is_primary_stratum
                        ),

                    "eligible_negative_pool_size":
                        eligible_pool_size,

                    "negative_count":
                        NEGATIVES_PER_CASE,

                    "case_seed_uint64":
                        str(
                            case_seed(
                                origin,
                                interaction_id,
                            )
                        ),

                    "same_fold_positive_collision_count":
                        collision_count,
                }
            )

        # ---------------------------------------------------------------------
        # Full-origin candidate integrity
        # ---------------------------------------------------------------------

        if (
            negative_matrix.shape
            !=
            (
                spec[
                    "expected_cases"
                ],
                NEGATIVES_PER_CASE,
            )
        ):

            raise ValueError(
                f"{origin}: negative matrix "
                "shape drift."
            )

        matrices[
            origin
        ] = negative_matrix

        origin_manifest = pd.DataFrame(
            manifest_rows
        )

        if (
            origin_manifest[
                "interaction_id"
            ].duplicated().any()
        ):

            raise ValueError(
                f"{origin}: duplicate "
                "interaction cases."
            )

        if (
            int(
                origin_manifest[
                    "is_primary_stratum"
                ].sum()
            )
            !=
            spec[
                "expected_primary"
            ]
        ):

            raise ValueError(
                f"{origin}: primary-case "
                "manifest drift."
            )

        case_frames.append(
            origin_manifest
        )

        summary_rows.append(
            {
                "origin":
                    origin,

                "evaluation_cases":
                    len(
                        fold_targets
                    ),

                "primary_cases":
                    int(
                        origin_manifest[
                            "is_primary_stratum"
                        ].sum()
                    ),

                "candidate_universe_size":
                    len(
                        universe_ids
                    ),

                "negatives_per_case":
                    NEGATIVES_PER_CASE,

                "negative_pool_min":
                    int(
                        pool_sizes.min()
                    ),

                "negative_pool_mean":
                    float(
                        pool_sizes.mean()
                    ),

                "negative_pool_max":
                    int(
                        pool_sizes.max()
                    ),

                "cases_with_same_fold_positive_collision":
                    collision_case_count,

                "same_fold_positive_collision_slots":
                    collision_slot_count,
            }
        )

        collision_rows.append(
            {
                "origin":
                    origin,

                "evaluation_cases":
                    len(
                        fold_targets
                    ),

                "cases_with_collision":
                    collision_case_count,

                "case_collision_rate_pct":
                    (
                        100.0
                        *
                        collision_case_count
                        /
                        len(
                            fold_targets
                        )
                    ),

                "total_negative_slots":
                    (
                        len(
                            fold_targets
                        )
                        *
                        NEGATIVES_PER_CASE
                    ),

                "collision_slots":
                    collision_slot_count,

                "slot_collision_rate_pct":
                    (
                        100.0
                        *
                        collision_slot_count
                        /
                        (
                            len(
                                fold_targets
                            )
                            *
                            NEGATIVES_PER_CASE
                        )
                    ),
            }
        )

        print(
            f"Evaluation cases:                   "
            f"{len(fold_targets):,}"
        )

        print(
            f"Primary cases:                      "
            f"{spec['expected_primary']:,}"
        )

        print(
            f"Candidate universe:                 "
            f"{len(universe_ids):,}"
        )

        print(
            f"Negative pool minimum:              "
            f"{int(pool_sizes.min()):,}"
        )

        print(
            f"Negative pool mean:                 "
            f"{float(pool_sizes.mean()):,.2f}"
        )

        print(
            f"Cases with same-fold + collision:   "
            f"{collision_case_count:,}"
        )

        print(
            f"Same-fold + negative slots:         "
            f"{collision_slot_count:,}"
        )

    # =========================================================================
    # 6. Persist only after every origin passes
    # =========================================================================

    separator("-")

    print(
        "8.0.11.6 — PERSIST FROZEN "
        "CANDIDATE ARTIFACTS"
    )

    separator("-")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matrix_paths = {}

    for spec in ORIGINS:

        origin = spec[
            "origin"
        ]

        safe_name = (
            origin.lower()
            .replace(
                "april_confirmatory",
                "april_confirmatory",
            )
        )

        path = (
            OUTPUT_DIR
            /
            (
                f"phase_8_0_11_"
                f"{safe_name}_negative_node_indices_V1.npy"
            )
        )

        np.save(
            path,
            matrices[
                origin
            ],
            allow_pickle=False,
        )

        matrix_paths[
            origin
        ] = path

    case_manifest = pd.concat(
        case_frames,
        ignore_index=True,
    )

    summary = pd.DataFrame(
        summary_rows
    )

    collision_summary = pd.DataFrame(
        collision_rows
    )

    case_manifest.to_parquet(
        CASE_MANIFEST_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    collision_summary.to_csv(
        COLLISION_OUTPUT,
        index=False,
    )

    print(
        summary.to_string(
            index=False
        )
    )

    # =========================================================================
    # 7. Freeze runtime contract
    # =========================================================================

    separator("-")

    print(
        "8.0.11.7 — CANDIDATE "
        "RUNTIME CONTRACT"
    )

    separator("-")

    contract = {
        "schema_version":
            "PHASE_8_0_11_EVALUATION_CANDIDATE_RUNTIME_V1",

        "status":
            "FROZEN",

        "positive_candidates_per_case":
            1,

        "negative_candidates_per_case":
            99,

        "total_candidates_per_case":
            100,

        "candidate_universe":
            (
                "origin-specific Phase-8.0.9 "
                "Startup universe"
            ),

        "negative_sampling": {
            "distribution":
                "uniform",

            "replacement":
                False,

            "exclude": [
                (
                    "focal positive Startup"
                ),
                (
                    "Startup historically positive "
                    "for focal Investor at or before "
                    "prediction origin"
                ),
            ],

            "do_not_exclude": [
                (
                    "other positive Startups in "
                    "same evaluation fold"
                ),
                (
                    "future-fold positive Startups"
                ),
            ],
        },

        "rng": {
            "generator":
                "numpy.random.Generator",

            "bit_generator":
                "PCG64",

            "base_seed":
                BASE_SEED,

            "namespace":
                RNG_NAMESPACE,

            "case_seed_derivation":
                (
                    "little-endian uint64 from first "
                    "8 bytes of SHA256("
                    "namespace|base_seed|origin|"
                    "interaction_id)"
                ),
        },

        "candidate_order_storage": {
            "positive":
                (
                    "stored separately in "
                    "case manifest"
                ),

            "negatives":
                (
                    "99 node indices stored in "
                    "frozen row order"
                ),
        },

        "same_fold_positive_collision_policy":
            (
                "diagnose after generation; "
                "never resample or modify list"
            ),

        "all_models_must_use_identical_lists":
            True,

        "resampling_after_freeze":
            False,

        "metric_interpretation":
            (
                "100-candidate sampled ranking; "
                "not full-catalog recommendation"
            ),

        "April_performance_inspected":
            False,

        "model_selection_performed":
            False,
    }

    write_json(
        CONTRACT_OUTPUT,
        contract,
    )

    # =========================================================================
    # 8. Manifest and fingerprints
    # =========================================================================

    artifact_hashes = {
        str(
            CASE_MANIFEST_OUTPUT
        ):
            sha256_file(
                CASE_MANIFEST_OUTPUT
            ),

        str(
            SUMMARY_OUTPUT
        ):
            sha256_file(
                SUMMARY_OUTPUT
            ),

        str(
            COLLISION_OUTPUT
        ):
            sha256_file(
                COLLISION_OUTPUT
            ),

        str(
            CONTRACT_OUTPUT
        ):
            sha256_file(
                CONTRACT_OUTPUT
            ),
    }

    for origin, path in (
        matrix_paths.items()
    ):

        artifact_hashes[
            str(
                path
            )
        ] = sha256_file(
            path
        )

    manifest = {
        "phase":
            "8.0.11",

        "status":
            "FROZEN",

        "authoritative_inputs":
            source_hashes,

        "candidate_matrices": {
            origin:
                {
                    "path":
                        str(
                            path
                        ),

                    "sha256":
                        artifact_hashes[
                            str(
                                path
                            )
                        ],

                    "shape":
                        list(
                            matrices[
                                origin
                            ].shape
                        ),

                    "dtype":
                        str(
                            matrices[
                                origin
                            ].dtype
                        ),
                }
            for origin, path
            in matrix_paths.items()
        },

        "outputs": [
            str(
                CASE_MANIFEST_OUTPUT
            ),
            str(
                SUMMARY_OUTPUT
            ),
            str(
                COLLISION_OUTPUT
            ),
            str(
                CONTRACT_OUTPUT
            ),
        ],

        "model_training_performed":
            False,

        "ranking_metrics_scored":
            False,

        "April_performance_inspected":
            False,
    }

    write_json(
        MANIFEST_OUTPUT,
        manifest,
    )

    artifact_hashes[
        str(
            MANIFEST_OUTPUT
        )
    ] = sha256_file(
        MANIFEST_OUTPUT
    )

    write_json(
        HASH_OUTPUT,
        artifact_hashes,
    )

    # =========================================================================
    # 9. Final freeze
    # =========================================================================

    separator()

    print(
        "PHASE 8.0.11 EVALUATION "
        "CANDIDATE FREEZE COMPLETE"
    )

    separator()

    print()

    print(
        f"Total evaluation cases:       "
        f"{len(case_manifest):,}"
    )

    print(
        f"Total primary cases:          "
        f"{int(case_manifest['is_primary_stratum'].sum()):,}"
    )

    print(
        f"Negatives per case:           "
        f"{NEGATIVES_PER_CASE}"
    )

    print(
        "Candidate lists frozen:      YES"
    )

    print(
        "Same-fold labels used "
        "for exclusion:              NO"
    )

    print(
        "Post-freeze resampling:       FORBIDDEN"
    )

    print(
        "Model training performed:     NO"
    )

    print(
        "Ranking metrics scored:       NO"
    )

    print(
        "April performance inspected:  NO"
    )

    print()

    print(
        "Next:"
    )

    print(
        "Close Phase 8.0 by freezing "
        "the research questions, hypotheses, "
        "baseline matrix, ablation matrix, "
        "and statistical analysis contract."
    )

    print()

    print(
        f"Outputs written to:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()