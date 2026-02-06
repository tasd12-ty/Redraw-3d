"""
Evaluation metrics and tools for ORDINAL-SPATIAL benchmark.

This module provides:
- Consistency checking for constraint sets
- Metrics for T1 classification tasks
- Metrics for T2 extraction tasks
- Procrustes alignment for T3 reconstruction
- Failure analysis and diagnostics
"""

from ordinal_spatial.evaluation.consistency import (
    ConsistencyStatus,
    Cycle,
    ConsistencyReport,
    ConsistencyChecker,
    check_qrr_consistency,
    check_trr_consistency,
    check_full_consistency,
    find_minimal_conflict,
)

from ordinal_spatial.evaluation.metrics import (
    T1QRRMetrics,
    T1TRRMetrics,
    T2Metrics,
    T3Metrics,
    BenchmarkMetrics,
    compute_t1_qrr_metrics,
    compute_t1_trr_metrics,
    compute_t2_metrics,
    compute_t2_metrics_batch,
    compute_t3_metrics,
    procrustes_align,
)

from ordinal_spatial.evaluation.constraint_diff import (
    ConstraintDiffMetrics,
    compute_constraint_diff,
    compute_constraint_diff_batch,
)

__all__ = [
    # Consistency
    "ConsistencyStatus",
    "Cycle",
    "ConsistencyReport",
    "ConsistencyChecker",
    "check_qrr_consistency",
    "check_trr_consistency",
    "check_full_consistency",
    "find_minimal_conflict",
    # Metrics
    "T1QRRMetrics",
    "T1TRRMetrics",
    "T2Metrics",
    "T3Metrics",
    "BenchmarkMetrics",
    "compute_t1_qrr_metrics",
    "compute_t1_trr_metrics",
    "compute_t2_metrics",
    "compute_t2_metrics_batch",
    "compute_t3_metrics",
    "procrustes_align",
    # Constraint-Diff
    "ConstraintDiffMetrics",
    "compute_constraint_diff",
    "compute_constraint_diff_batch",
]
