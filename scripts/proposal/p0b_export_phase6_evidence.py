from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd


# =============================================================================
# P0.B — EXPORT AUTHORITATIVE PHASE-6 EVIDENCE
# =============================================================================
#
# Purpose:
#   Convert already-computed frozen Phase-6 results into clean proposal-facing
#   CSV/JSON artifacts.
#
# IMPORTANT:
#   - NO training
#   - NO inference
#   - NO test rescoring
#   - NO checkpoint modification
#   - NO Phase-6 artifact modification
#
# =============================================================================


ROOT = Path(__file__).resolve().parents[2]

PHASE6 = ROOT / "data" / "experimental" / "phase_6"

TRAINING_ROOT = (
    PHASE6
    / "reduced_training"
    / "10pct"
    / "run_20epoch"
)

FINAL_TEST_ROOT = PHASE6 / "final_test"
CONTRACT_ROOT = PHASE6 / "contracts"

OUTPUT = (
    ROOT
    / "data"
    / "experimental"
    / "proposal_evidence"
)

OUTPUT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Authoritative input paths
# =============================================================================

EPOCH_METRICS = TRAINING_ROOT / "epoch_metrics.csv"

BEST_CHECKPOINT = (
    TRAINING_ROOT
    / "checkpoints"
    / "best.pt"
)

TRAINING_RESULT_JSON = (
    CONTRACT_ROOT
    / "phase_6_7d_10pct_20epoch_pilot_result.json"
)

FINAL_RESULT_JSON = (
    CONTRACT_ROOT
    / "phase_6_8d_final_t60_test_result.json"
)

CASE_METRICS = (
    FINAL_TEST_ROOT
    / "final_t60_test_case_metrics.parquet"
)

CASE_BINDING = (
    FINAL_TEST_ROOT
    / "final_t60_test_case_binding.parquet"
)


# =============================================================================
# Output paths
# =============================================================================

OUT_TRAINING_HISTORY = (
    OUTPUT / "01_phase6_training_history.csv"
)

OUT_FINAL_METRICS = (
    OUTPUT / "02_phase6_final_metrics.csv"
)

OUT_CASE_METRICS = (
    OUTPUT / "03_final_test_case_metrics.csv"
)

OUT_CASE_BINDING = (
    OUTPUT / "04_final_test_case_binding.csv"
)

OUT_COMBINED_CASES = (
    OUTPUT / "05_final_test_event_predictions.csv"
)

OUT_SUMMARY = (
    OUTPUT / "proposal_evidence_summary.json"
)


# =============================================================================
# Helpers
# =============================================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required authoritative artifact missing:\n{path}"
        )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_json(
    obj,
    prefix: str = "",
    out: dict | None = None,
) -> dict:
    if out is None:
        out = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = (
                f"{prefix}.{key}"
                if prefix
                else str(key)
            )
            flatten_json(
                value,
                new_key,
                out,
            )

    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        out[prefix] = obj

    return out


def print_dataframe_summary(
    name: str,
    df: pd.DataFrame,
) -> None:
    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print()
    print("Columns:")
    for col in df.columns:
        print(f"  - {col}")

    print()
    print("First 3 rows:")
    print(df.head(3).to_string(index=False))


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    print("=" * 100)
    print("P0.B — EXPORT AUTHORITATIVE PHASE-6 PROPOSAL EVIDENCE")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------

    inputs = [
        EPOCH_METRICS,
        BEST_CHECKPOINT,
        TRAINING_RESULT_JSON,
        FINAL_RESULT_JSON,
        CASE_METRICS,
        CASE_BINDING,
    ]

    print()
    print("AUTHORITATIVE INPUTS")
    print("-" * 100)

    for path in inputs:
        require_file(path)

        print(
            f"FOUND  "
            f"{path.relative_to(ROOT)}"
        )

    # =========================================================================
    # 1. Training history
    # =========================================================================

    training_df = pd.read_csv(
        EPOCH_METRICS
    )

    print_dataframe_summary(
        "PHASE-6 10% TRAINING HISTORY",
        training_df,
    )

    if len(training_df) != 20:
        raise AssertionError(
            "Expected exactly 20 completed epochs, "
            f"found {len(training_df):,}."
        )

    training_df.to_csv(
        OUT_TRAINING_HISTORY,
        index=False,
    )

    # =========================================================================
    # 2. Final result contracts
    # =========================================================================

    training_result = load_json(
        TRAINING_RESULT_JSON
    )

    final_result = load_json(
        FINAL_RESULT_JSON
    )

    flat_training = flatten_json(
        training_result,
        prefix="training",
    )

    flat_final = flatten_json(
        final_result,
        prefix="test",
    )

    final_metrics_row = {
        **flat_training,
        **flat_final,
        "best_checkpoint_path": str(
            BEST_CHECKPOINT.relative_to(ROOT)
        ),
        "best_checkpoint_size_bytes": (
            BEST_CHECKPOINT.stat().st_size
        ),
    }

    final_metrics_df = pd.DataFrame(
        [final_metrics_row]
    )

    final_metrics_df.to_csv(
        OUT_FINAL_METRICS,
        index=False,
    )

    # =========================================================================
    # 3. Per-test-case metrics
    # =========================================================================

    metrics_df = pd.read_parquet(
        CASE_METRICS
    )

    binding_df = pd.read_parquet(
        CASE_BINDING
    )

    print_dataframe_summary(
        "FINAL T60 TEST — CASE METRICS",
        metrics_df,
    )

    print_dataframe_summary(
        "FINAL T60 TEST — CASE BINDING",
        binding_df,
    )

    if len(metrics_df) != 20_264:
        raise AssertionError(
            "Expected 20,264 final test metric rows, "
            f"found {len(metrics_df):,}."
        )

    if len(binding_df) != 20_264:
        raise AssertionError(
            "Expected 20,264 final test binding rows, "
            f"found {len(binding_df):,}."
        )

    metrics_df.to_csv(
        OUT_CASE_METRICS,
        index=False,
    )

    binding_df.to_csv(
        OUT_CASE_BINDING,
        index=False,
    )

    # =========================================================================
    # 4. Safely produce combined event-level table
    # =========================================================================
    #
    # We only merge automatically when an explicit shared unique identifier
    # exists.
    #
    # We DO NOT assume row ordering unless necessary.
    # =========================================================================

    candidate_keys = [
        "case_index",
        "test_case_index",
        "evaluation_case_index",
        "split_row_index",
        "event_index",
        "interaction_id",
    ]

    common_keys = [
        key
        for key in candidate_keys
        if (
            key in metrics_df.columns
            and key in binding_df.columns
        )
    ]

    merge_key = None

    for key in common_keys:
        if (
            metrics_df[key].is_unique
            and binding_df[key].is_unique
            and not metrics_df[key].isna().any()
            and not binding_df[key].isna().any()
        ):
            merge_key = key
            break

    merge_method = None

    if merge_key is not None:

        combined_df = binding_df.merge(
            metrics_df,
            on=merge_key,
            how="inner",
            validate="one_to_one",
            suffixes=("_binding", "_metric"),
        )

        if len(combined_df) != 20_264:
            raise AssertionError(
                "Explicit-key merge changed test-case count."
            )

        merge_method = (
            f"explicit_unique_key:{merge_key}"
        )

    else:
        # ---------------------------------------------------------------------
        # Fallback:
        #
        # The Phase-6 scorer emitted case_binding and case_metrics from the
        # same frozen evaluation stream. If no explicit ID is shared, preserve
        # both row indices and combine by row position, but label this clearly.
        # ---------------------------------------------------------------------

        binding_copy = (
            binding_df
            .reset_index(drop=True)
            .add_prefix("binding__")
        )

        metrics_copy = (
            metrics_df
            .reset_index(drop=True)
            .add_prefix("metric__")
        )

        combined_df = pd.concat(
            [
                binding_copy,
                metrics_copy,
            ],
            axis=1,
        )

        combined_df.insert(
            0,
            "proposal_case_row",
            range(len(combined_df)),
        )

        merge_method = (
            "row_position_from_same_frozen_phase6_evaluation_stream"
        )

    combined_df.to_csv(
        OUT_COMBINED_CASES,
        index=False,
    )

    print_dataframe_summary(
        "COMBINED FINAL TEST EVENT TABLE",
        combined_df,
    )

    # =========================================================================
    # 5. Detect likely key metric columns
    # =========================================================================

    lower_metric_columns = {
        str(col).lower(): col
        for col in metrics_df.columns
    }

    detected = {}

    searches = {
        "rank_column": [
            "positive_rank",
            "rank",
        ],
        "hit10_column": [
            "hit_at_10",
            "hr_at_10",
            "hit10",
        ],
        "ndcg10_column": [
            "ndcg_at_10",
            "ndcg10",
        ],
    }

    for label, possibilities in searches.items():
        detected[label] = None

        for candidate in possibilities:
            if candidate in lower_metric_columns:
                detected[label] = str(
                    lower_metric_columns[candidate]
                )
                break

    # =========================================================================
    # 6. Proposal evidence summary
    # =========================================================================

    summary = {
        "schema_version": (
            "PROPOSAL_EVIDENCE_P0_V1"
        ),
        "status": "P0_COMPLETE",
        "scientific_role": (
            "post-hoc export of frozen Phase-6 results; "
            "no model selection or retraining"
        ),
        "selected_training_budget": "10pct",
        "training_epochs": int(
            len(training_df)
        ),
        "final_test_cases": int(
            len(metrics_df)
        ),
        "combined_case_merge_method": (
            merge_method
        ),
        "detected_metric_columns": detected,
        "authoritative_inputs": {
            "training_history": str(
                EPOCH_METRICS.relative_to(ROOT)
            ),
            "training_result": str(
                TRAINING_RESULT_JSON.relative_to(ROOT)
            ),
            "best_checkpoint": str(
                BEST_CHECKPOINT.relative_to(ROOT)
            ),
            "test_result": str(
                FINAL_RESULT_JSON.relative_to(ROOT)
            ),
            "test_case_metrics": str(
                CASE_METRICS.relative_to(ROOT)
            ),
            "test_case_binding": str(
                CASE_BINDING.relative_to(ROOT)
            ),
        },
        "proposal_outputs": {
            "training_history": str(
                OUT_TRAINING_HISTORY.relative_to(ROOT)
            ),
            "final_metrics": str(
                OUT_FINAL_METRICS.relative_to(ROOT)
            ),
            "test_case_metrics": str(
                OUT_CASE_METRICS.relative_to(ROOT)
            ),
            "test_case_binding": str(
                OUT_CASE_BINDING.relative_to(ROOT)
            ),
            "combined_test_events": str(
                OUT_COMBINED_CASES.relative_to(ROOT)
            ),
        },
    }

    with OUT_SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    # =========================================================================
    # Final
    # =========================================================================

    print()
    print("=" * 100)
    print("P0.B OUTPUTS")
    print("=" * 100)

    outputs = [
        OUT_TRAINING_HISTORY,
        OUT_FINAL_METRICS,
        OUT_CASE_METRICS,
        OUT_CASE_BINDING,
        OUT_COMBINED_CASES,
        OUT_SUMMARY,
    ]

    for path in outputs:
        print(
            "WROTE ",
            path.relative_to(ROOT),
        )

    print()
    print(
        "Combined-case merge method:",
        merge_method,
    )

    print()
    print(
        "Detected metric columns:",
        json.dumps(
            detected,
            indent=2,
        ),
    )

    print()
    print("=" * 100)
    print(
        "P0 COMPLETE — NO TRAINING, "
        "NO INFERENCE, NO TEST RESCORING"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
