"""
Ordinal Constraint Language (DSL) for spatial relations.

This module provides the core language constructs for expressing
ordinal spatial constraints:

- Comparators: Tolerance-based comparison algebra
- Predicates: QRR (Quaternary) and TRR (Ternary) constraint types
- Schemas: Data models for constraint representation
"""

from ordinal_spatial.dsl.comparators import (
    Comparator,
    TolerancePreset,
    TOLERANCE_STRICT,
    TOLERANCE_STANDARD,
    TOLERANCE_RELAXED,
    TOLERANCE_PRESETS,
    compare,
    compare_ratio,
    ordinal_distance,
    is_flip,
    difficulty_from_ratio,
    ComparatorChain,
)

from ordinal_spatial.dsl.predicates import (
    MetricType,
    QRRConstraint,
    TRRConstraint,
    compute_qrr,
    compute_trr,
    extract_all_qrr,
    extract_all_trr,
    clock_angular_error,
    angle_to_hour,
    hour_to_quadrant,
)

from ordinal_spatial.dsl.schema import (
    ObjectSpec,
    OrdinalSceneDescription,
    WorldConstraints,
    ViewConstraints,
    CameraParams,
    QRRQuery,
    TRRQuery,
    QRRPrediction,
    TRRPrediction,
    OSDPrediction,
)

__all__ = [
    # Comparators
    "Comparator",
    "TolerancePreset",
    "TOLERANCE_STRICT",
    "TOLERANCE_STANDARD",
    "TOLERANCE_RELAXED",
    "TOLERANCE_PRESETS",
    "compare",
    "compare_ratio",
    "ordinal_distance",
    "is_flip",
    "difficulty_from_ratio",
    "ComparatorChain",
    # Predicates
    "MetricType",
    "QRRConstraint",
    "TRRConstraint",
    "compute_qrr",
    "compute_trr",
    "extract_all_qrr",
    "extract_all_trr",
    "clock_angular_error",
    "angle_to_hour",
    "hour_to_quadrant",
    # Schemas
    "ObjectSpec",
    "OrdinalSceneDescription",
    "WorldConstraints",
    "ViewConstraints",
    "CameraParams",
    "QRRQuery",
    "TRRQuery",
    "QRRPrediction",
    "TRRPrediction",
    "OSDPrediction",
]
