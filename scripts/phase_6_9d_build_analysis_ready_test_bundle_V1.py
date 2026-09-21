#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


# =============================================================================
# Phase 6.9d — Analysis-ready final-test bundle
# Post-hoc only: no model loading, no inference, no optimizer, no test rescoring.
# =============================================================================

FINAL_CASE_METRICS = Path(
    "data/experimental/phase_6/full_training/100pct/final_test/"
    "final_test_case_metrics.parquet"
)

CASE_MANIFEST = Path(
    "data/experimental/phase_5/model_ready/evaluation/"
    "t60_evaluation_case_manifest.parquet"
)

T60_SPLIT = Path(
    "data/experimental/phase_2/model_ready/"
    "t60_validation_test_split.parquet"
)

STRUCTURAL_COVERAGE = Path(
    "data/experimental/phase_3/model_ready/"
    "node_structural_coverage.parquet"
)

OUT_DIR = Path(
    "data/experimental/phase_6/full_training/100pct/final_test"
)

OUT_PARQUET = OUT_DIR / "analysis_ready_test_cases.parquet"
OUT_MANIFEST = OUT_DIR / "analysis_ready_test_cases_manifest.json"

EXPECTED_ROWS = 20_264

EXPECTED_SOURCE_HASHES = {
    "final_test_case_metrics.parquet":
        "e09895011115c570bd028f8fa3713df2cfdf8bc8020c3b5ea51ffd879f6f7d7a",
}


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main():
    print("=" * 110)
    print("PHASE 6.9d — BUILD ANALYSIS-READY FINAL-TEST BUNDLE V1")
    print("=" * 110)
    print("Model inference:      NO")
    print("Test rescoring:       NO")
    print("Optimizer/backward:   NO")
    print("Role:                 post-hoc metadata join only")
    print()

    for path in (
        FINAL_CASE_METRICS,
        CASE_MANIFEST,
        T60_SPLIT,
        STRUCTURAL_COVERAGE,
    ):
        require(path.exists(), f"Missing required source: {path}")

    # Protect the authoritative final-test per-case result against drift.
    require(
        file_sha256(FINAL_CASE_METRICS)
        == EXPECTED_SOURCE_HASHES["final_test_case_metrics.parquet"],
        "Final-test case metrics fingerprint drift.",
    )

    metrics = pd.read_parquet(FINAL_CASE_METRICS)
    cases = pd.read_parquet(CASE_MANIFEST)
    split = pd.read_parquet(T60_SPLIT)
    structural = pd.read_parquet(STRUCTURAL_COVERAGE)

    require(len(metrics) == EXPECTED_ROWS, "Final metrics row-count drift.")
    require(metrics["interaction_id"].is_unique, "Final metrics interaction_id is not unique.")
    require(cases["interaction_id"].is_unique, "Case manifest interaction_id is not unique.")
    require(split["interaction_id"].is_unique, "T60 split interaction_id is not unique.")
    require(structural["node_index"].is_unique, "Structural node_index is not unique.")

    require(
        set(metrics["positive_rank"].between(1, 100).unique()) <= {True},
        "Positive rank outside 1..100.",
    )

    # -------------------------------------------------------------------------
    # 1) Bind final test metrics to the frozen evaluation case manifest.
    # -------------------------------------------------------------------------
    case_cols = [
        "case_index",
        "interaction_id",
        "funding_round_id",
        "experiment_split",
        "investor_id",
        "investor_node_index",
        "positive_startup_id",
        "positive_startup_node_index",
        "evaluation_seed",
        "negative_pool_size",
        "negative_list_sha256",
        "has_realized_other_T60_positive_collision",
    ]

    bundle = metrics.merge(
        cases[case_cols],
        on="interaction_id",
        how="left",
        validate="one_to_one",
    )

    require(
        bundle["case_index"].notna().all(),
        "At least one final-test metric row failed case-manifest binding.",
    )
    require(
        (bundle["experiment_split"] == "test").all(),
        "Final-test metrics contain non-test evaluation cases.",
    )
    require(
        (bundle["matrix_row_index"] == bundle["case_index"]).all(),
        "matrix_row_index != frozen case_index.",
    )
    require(
        (bundle["investor_global"] == bundle["investor_node_index"]).all(),
        "Investor global index mismatch.",
    )

    # -------------------------------------------------------------------------
    # 2) Bind temporal/cold-start/new-to-investor metadata from Phase 2.
    # -------------------------------------------------------------------------
    split_cols = [
        "interaction_id",
        "startup_id",
        "startup_name",
        "investor_name",
        "announced_on",
        "investment_type",
        "segment_number",
        "segment_label",
        "temporal_role",
        "investor_seen_before_t60",
        "investor_seen_in_t1_t59",
        "investor_seen_in_t0",
        "investor_t0_only_history",
        "startup_seen_before_t60",
        "startup_seen_in_t1_t59",
        "startup_seen_in_t0",
        "startup_t0_only_history",
        "pair_seen_before_t60",
        "pair_seen_in_t1_t59",
        "new_to_investor_pair",
        "interaction_cold_start_status",
        "t60_pair_event_count",
        "pair_repeats_within_t60",
        "evaluation_split",
    ]

    bundle = bundle.merge(
        split[split_cols],
        on="interaction_id",
        how="left",
        validate="one_to_one",
    )

    require(
        bundle["evaluation_split"].notna().all(),
        "At least one final-test row failed Phase-2 split binding.",
    )
    require(
        (bundle["evaluation_split"] == "test").all(),
        "Phase-2 split binding contains non-test rows.",
    )
    require(
        (bundle["positive_startup_id"] == bundle["startup_id"]).all(),
        "Startup ID mismatch between evaluation manifest and Phase-2 split.",
    )

    # Explicit convenience aliases using already-audited semantics.
    bundle["cold_start_investor"] = ~bundle["investor_seen_before_t60"].astype(bool)
    bundle["cold_start_startup"] = ~bundle["startup_seen_before_t60"].astype(bool)
    bundle["prior_investor_startup_relationship"] = bundle["pair_seen_before_t60"].astype(bool)

    # -------------------------------------------------------------------------
    # 3) Attach structural coverage for investor and positive startup.
    # -------------------------------------------------------------------------
    structural_cols = [
        "node_index",
        "node_type",
        "core_out_degree",
        "core_in_degree",
        "core_total_degree",
        "core_connected",
        "founder_only_ablation_out_degree",
        "founder_only_ablation_in_degree",
        "founder_only_ablation_total_degree",
        "founder_only_ablation_connected",
        "acquisition_only_ablation_out_degree",
        "acquisition_only_ablation_in_degree",
        "acquisition_only_ablation_total_degree",
        "acquisition_only_ablation_connected",
        "structural_source_class",
    ]

    investor_struct = structural[structural_cols].rename(
        columns={
            c: f"investor_{c}"
            for c in structural_cols
            if c != "node_index"
        }
    ).rename(columns={"node_index": "investor_node_index"})

    startup_struct = structural[structural_cols].rename(
        columns={
            c: f"startup_{c}"
            for c in structural_cols
            if c != "node_index"
        }
    ).rename(columns={"node_index": "positive_startup_node_index"})

    bundle = bundle.merge(
        investor_struct,
        on="investor_node_index",
        how="left",
        validate="many_to_one",
    )

    bundle = bundle.merge(
        startup_struct,
        on="positive_startup_node_index",
        how="left",
        validate="many_to_one",
    )

    require(
        bundle["investor_node_type"].eq("investor").all(),
        "Investor structural binding node_type mismatch.",
    )
    require(
        bundle["startup_node_type"].eq("startup").all(),
        "Startup structural binding node_type mismatch.",
    )

    # -------------------------------------------------------------------------
    # 4) Final closure checks.
    # -------------------------------------------------------------------------
    require(len(bundle) == EXPECTED_ROWS, "Analysis bundle row-count drift.")
    require(bundle["interaction_id"].is_unique, "Bundle interaction_id is not unique.")
    require(bundle["test_position"].is_unique, "Bundle test_position is not unique.")
    require(
        bundle["test_position"].tolist() == list(range(EXPECTED_ROWS)),
        "Test position is not exactly 0..20263 in frozen evaluation order.",
    )

    require(
        int(bundle["HR@10"].sum()) == 11_072,
        "Frozen final-test hit count drift.",
    )

    hr10 = float(bundle["HR@10"].mean())
    ndcg10 = float(bundle["NDCG@10"].mean())

    require(
        abs(hr10 - 0.546387682590) <= 5e-13,
        "Frozen final-test HR@10 drift.",
    )
    require(
        abs(ndcg10 - 0.398709730124) <= 5e-13,
        "Frozen final-test NDCG@10 drift.",
    )

    # Put identifiers/metrics/core thesis subgroup fields first.
    preferred = [
        "test_position",
        "case_index",
        "matrix_row_index",
        "interaction_id",
        "funding_round_id",
        "investor_id",
        "investor_name",
        "startup_id",
        "startup_name",
        "investor_node_index",
        "positive_startup_node_index",
        "investor_global",
        "positive_startup_local",
        "announced_on",
        "investment_type",
        "positive_rank",
        "HR@10",
        "NDCG@10",
        "new_to_investor_pair",
        "prior_investor_startup_relationship",
        "cold_start_investor",
        "cold_start_startup",
        "interaction_cold_start_status",
        "investor_seen_before_t60",
        "startup_seen_before_t60",
        "pair_seen_before_t60",
        "pair_repeats_within_t60",
        "t60_pair_event_count",
        "investor_core_connected",
        "startup_core_connected",
        "investor_core_total_degree",
        "startup_core_total_degree",
        "investor_structural_source_class",
        "startup_structural_source_class",
        "has_realized_other_T60_positive_collision",
        "negative_pool_size",
        "negative_list_sha256",
        "evaluation_seed",
        "chunk_index",
    ]
    remaining = [c for c in bundle.columns if c not in preferred]
    bundle = bundle[preferred + remaining]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle.to_parquet(OUT_PARQUET, index=False)

    source_hashes = {
        str(path): file_sha256(path)
        for path in (
            FINAL_CASE_METRICS,
            CASE_MANIFEST,
            T60_SPLIT,
            STRUCTURAL_COVERAGE,
        )
    }

    output_hash = file_sha256(OUT_PARQUET)

    manifest = {
        "phase": "6.9d",
        "schema_version": "ITRS_PHASE6_9D_ANALYSIS_READY_TEST_BUNDLE_V1",
        "status": "FROZEN_POSTHOC_ANALYSIS_READY",
        "scientific_role": (
            "Post-hoc metadata binding for analysis of the already-frozen "
            "one-shot final test. No model inference or test rescoring."
        ),
        "rows": int(len(bundle)),
        "columns": list(bundle.columns),
        "metrics": {
            "HR@10": hr10,
            "NDCG@10": ndcg10,
            "hit_count": int(bundle["HR@10"].sum()),
            "mean_positive_rank": float(bundle["positive_rank"].mean()),
            "median_positive_rank": float(bundle["positive_rank"].median()),
        },
        "subgroup_fields": [
            "new_to_investor_pair",
            "prior_investor_startup_relationship",
            "cold_start_investor",
            "cold_start_startup",
            "interaction_cold_start_status",
            "investor_core_connected",
            "startup_core_connected",
            "investor_structural_source_class",
            "startup_structural_source_class",
        ],
        "source_file_sha256": source_hashes,
        "output": {
            "path": str(OUT_PARQUET),
            "sha256": output_hash,
        },
        "integrity": {
            "expected_test_cases": EXPECTED_ROWS,
            "interaction_id_unique": True,
            "test_position_contiguous_0_20263": True,
            "evaluation_split_all_test": True,
            "case_manifest_binding": "PASS",
            "phase2_temporal_binding": "PASS",
            "phase3_structural_binding": "PASS",
            "test_rescored": False,
        },
    }

    OUT_MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"rows:                 {len(bundle):,}")
    print(f"columns:              {len(bundle.columns):,}")
    print(f"HR@10:                {hr10:.12f}")
    print(f"NDCG@10:              {ndcg10:.12f}")
    print(f"hits@10:              {int(bundle['HR@10'].sum()):,}")
    print(f"new-to-investor:      {int(bundle['new_to_investor_pair'].sum()):,}")
    print(f"cold investors:       {int(bundle['cold_start_investor'].sum()):,}")
    print(f"cold startups:        {int(bundle['cold_start_startup'].sum()):,}")
    print(f"investor connected:   {int(bundle['investor_core_connected'].sum()):,}")
    print(f"startup connected:    {int(bundle['startup_core_connected'].sum()):,}")
    print()
    print(f"WROTE {OUT_PARQUET}")
    print(f"sha256 {output_hash}")
    print(f"WROTE {OUT_MANIFEST}")
    print(f"manifest_sha256 {file_sha256(OUT_MANIFEST)}")
    print()
    print("PHASE 6.9d: PASS / ANALYSIS-READY TEST BUNDLE FROZEN")


if __name__ == "__main__":
    main()
