"""
Task runners for ORDINAL-SPATIAL benchmark.

Available tasks:
- T1: Ordinal Classification (QRR and TRR)
- T2: Constraint Extraction
- T3: Ordinal Reconstruction
"""

from ordinal_spatial.tasks.t1_classification import (
    T1Config,
    T1Result,
    T1QRRRunner,
    T1TRRRunner,
    run_t1_qrr_evaluation,
    run_t1_trr_evaluation,
)

from ordinal_spatial.tasks.t2_extraction import (
    T2Runner,
    run_t2_evaluation,
)

from ordinal_spatial.tasks.t3_reconstruction import (
    T3Runner,
    run_t3_evaluation,
)

__all__ = [
    # T1
    "T1Config",
    "T1Result",
    "T1QRRRunner",
    "T1TRRRunner",
    "run_t1_qrr_evaluation",
    "run_t1_trr_evaluation",
    # T2
    "T2Runner",
    "run_t2_evaluation",
    # T3
    "T3Runner",
    "run_t3_evaluation",
]
