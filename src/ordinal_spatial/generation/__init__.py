"""
Scene generation and constraint extraction for ORDINAL-SPATIAL benchmark.

This module provides:
- Constraint extraction from 3D scene data
- Difficulty control for generated constraints
- Degeneracy checking and rejection
- Scene generation orchestration
"""

from ordinal_spatial.generation.constraint_extractor import (
    ConstraintExtractor,
    extract_scene_constraints,
    extract_qrr_from_scene,
    extract_trr_from_scene,
)

from ordinal_spatial.generation.difficulty_control import (
    DifficultyController,
    DifficultyLevel,
    filter_by_difficulty,
    compute_difficulty_distribution,
)

from ordinal_spatial.generation.degeneracy_checker import (
    DegeneracyType,
    DegeneracyReport,
    check_scene_degeneracy,
    is_scene_valid,
)

__all__ = [
    # Constraint Extraction
    "ConstraintExtractor",
    "extract_scene_constraints",
    "extract_qrr_from_scene",
    "extract_trr_from_scene",
    # Difficulty Control
    "DifficultyController",
    "DifficultyLevel",
    "filter_by_difficulty",
    "compute_difficulty_distribution",
    # Degeneracy Checking
    "DegeneracyType",
    "DegeneracyReport",
    "check_scene_degeneracy",
    "is_scene_valid",
]
