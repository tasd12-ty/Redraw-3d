"""
ORDINAL-SPATIAL 基准测试的难度控制。

本模块提供工具来控制和分析生成约束的难度分布。

难度等级（基于度量比率）：
- Level 1 (简单)：明显差异（比率 > 2.0）
- Level 2 (中等偏易)：清晰差异（比率 1.5-2.0）
- Level 3 (中等)：适中差异（比率 1.3-1.5）
- Level 4 (中等偏难)：微小差异（比率 1.15-1.3）
- Level 5 (困难)：极小差异（比率 1.05-1.15）
- Level 6 (极难)：接近阈值（比率 ~1.0）

功能：
- 根据比率计算难度等级
- 生成平衡的难度分布
- 统计难度直方图
- 采样特定难度的约束

Difficulty control for ORDINAL-SPATIAL benchmark.

This module provides tools to control and analyze the difficulty
distribution of generated constraints.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum
from collections import Counter
import numpy as np

from ordinal_spatial.dsl.predicates import QRRConstraint
from ordinal_spatial.dsl.comparators import difficulty_from_ratio


class DifficultyLevel(IntEnum):
    """
    Difficulty levels based on metric ratios.

    Level 1 (Easy): Clear differences (ratio > 2.0)
    Level 6 (Extreme): Near-threshold (ratio ~1.0)
    """
    EASY = 1
    MEDIUM_EASY = 2
    MEDIUM = 3
    MEDIUM_HARD = 4
    HARD = 5
    EXTREME = 6


@dataclass
class DifficultyBand:
    """Definition of a difficulty band."""
    level: DifficultyLevel
    ratio_min: float
    ratio_max: float
    name: str
    description: str


# Standard difficulty bands
DIFFICULTY_BANDS = {
    DifficultyLevel.EASY: DifficultyBand(
        level=DifficultyLevel.EASY,
        ratio_min=2.0,
        ratio_max=float('inf'),
        name="Easy",
        description="Obvious difference (ratio > 2:1)"
    ),
    DifficultyLevel.MEDIUM_EASY: DifficultyBand(
        level=DifficultyLevel.MEDIUM_EASY,
        ratio_min=1.5,
        ratio_max=2.0,
        name="Medium-Easy",
        description="Clear difference (ratio 1.5-2.0)"
    ),
    DifficultyLevel.MEDIUM: DifficultyBand(
        level=DifficultyLevel.MEDIUM,
        ratio_min=1.3,
        ratio_max=1.5,
        name="Medium",
        description="Moderate difference (ratio 1.3-1.5)"
    ),
    DifficultyLevel.MEDIUM_HARD: DifficultyBand(
        level=DifficultyLevel.MEDIUM_HARD,
        ratio_min=1.15,
        ratio_max=1.3,
        name="Medium-Hard",
        description="Subtle difference (ratio 1.15-1.3)"
    ),
    DifficultyLevel.HARD: DifficultyBand(
        level=DifficultyLevel.HARD,
        ratio_min=1.05,
        ratio_max=1.15,
        name="Hard",
        description="Fine discrimination (ratio 1.05-1.15)"
    ),
    DifficultyLevel.EXTREME: DifficultyBand(
        level=DifficultyLevel.EXTREME,
        ratio_min=1.0,
        ratio_max=1.05,
        name="Extreme",
        description="Near-threshold (ratio 1.0-1.05)"
    ),
}


class DifficultyController:
    """
    Controller for managing constraint difficulty distribution.

    Used during scene generation to ensure appropriate difficulty
    mix in the benchmark dataset.
    """

    def __init__(
        self,
        target_distribution: Dict[DifficultyLevel, float] = None,
        min_per_level: int = 0,
    ):
        """
        Initialize the controller.

        Args:
            target_distribution: Target proportion for each difficulty level
            min_per_level: Minimum constraints per difficulty level
        """
        if target_distribution is None:
            # Default: balanced distribution
            target_distribution = {
                DifficultyLevel.EASY: 0.15,
                DifficultyLevel.MEDIUM_EASY: 0.20,
                DifficultyLevel.MEDIUM: 0.25,
                DifficultyLevel.MEDIUM_HARD: 0.20,
                DifficultyLevel.HARD: 0.15,
                DifficultyLevel.EXTREME: 0.05,
            }

        self.target_distribution = target_distribution
        self.min_per_level = min_per_level

    def classify(self, constraint: QRRConstraint) -> DifficultyLevel:
        """
        分类约束的难度等级。

        Classify a constraint's difficulty level.
        """
        return DifficultyLevel(constraint.difficulty)

    def filter_by_level(
        self,
        constraints: List[QRRConstraint],
        levels: List[DifficultyLevel]
    ) -> List[QRRConstraint]:
        """
        过滤约束，仅包含指定的难度等级。

        Filter constraints to only include specified difficulty levels.
        """
        level_set = set(levels)
        return [c for c in constraints if DifficultyLevel(c.difficulty) in level_set]

    def sample_balanced(
        self,
        constraints: List[QRRConstraint],
        n: int,
        strict: bool = False
    ) -> List[QRRConstraint]:
        """
        Sample constraints to achieve balanced difficulty distribution.

        Args:
            constraints: Pool of constraints to sample from
            n: Target number of constraints
            strict: If True, enforce exact distribution

        Returns:
            Sampled constraints with balanced difficulties
        """
        # Group by difficulty
        by_level = {}
        for c in constraints:
            level = DifficultyLevel(c.difficulty)
            if level not in by_level:
                by_level[level] = []
            by_level[level].append(c)

        result = []

        # Calculate target counts
        target_counts = {
            level: max(self.min_per_level, int(n * prop))
            for level, prop in self.target_distribution.items()
        }

        # Sample from each level
        for level, target in target_counts.items():
            available = by_level.get(level, [])
            sample_n = min(target, len(available))

            if sample_n > 0:
                indices = np.random.choice(len(available), sample_n, replace=False)
                result.extend([available[i] for i in indices])

        # Fill remaining with random samples
        while len(result) < n:
            remaining = [c for c in constraints if c not in result]
            if not remaining:
                break
            result.append(remaining[np.random.randint(len(remaining))])

        np.random.shuffle(result)
        return result[:n]

    def analyze_distribution(
        self,
        constraints: List[QRRConstraint]
    ) -> Dict[str, Any]:
        """
        Analyze the difficulty distribution of constraints.

        Returns:
            Analysis report with distribution statistics
        """
        if not constraints:
            return {"n_constraints": 0, "distribution": {}}

        # Count by level
        counts = Counter(DifficultyLevel(c.difficulty) for c in constraints)
        total = len(constraints)

        distribution = {
            level.name: {
                "count": counts.get(level, 0),
                "proportion": counts.get(level, 0) / total,
                "target": self.target_distribution.get(level, 0),
            }
            for level in DifficultyLevel
        }

        # Calculate deviation from target
        deviation = sum(
            abs(d["proportion"] - d["target"])
            for d in distribution.values()
        ) / 2  # Normalize to [0, 1]

        return {
            "n_constraints": total,
            "distribution": distribution,
            "deviation_from_target": deviation,
            "is_balanced": deviation < 0.1,
        }


# =============================================================================
# Standalone Functions
# =============================================================================

def filter_by_difficulty(
    constraints: List[QRRConstraint],
    min_level: DifficultyLevel = DifficultyLevel.EASY,
    max_level: DifficultyLevel = DifficultyLevel.EXTREME,
) -> List[QRRConstraint]:
    """
    Filter constraints by difficulty range.

    Args:
        constraints: List of constraints
        min_level: Minimum difficulty (inclusive)
        max_level: Maximum difficulty (inclusive)

    Returns:
        Filtered constraints within difficulty range
    """
    return [
        c for c in constraints
        if min_level.value <= c.difficulty <= max_level.value
    ]


def compute_difficulty_distribution(
    constraints: List[QRRConstraint]
) -> Dict[DifficultyLevel, int]:
    """
    Compute difficulty distribution of constraints.

    Args:
        constraints: List of constraints

    Returns:
        Dictionary mapping difficulty levels to counts
    """
    return Counter(DifficultyLevel(c.difficulty) for c in constraints)


def ratio_to_difficulty(ratio: float) -> DifficultyLevel:
    """
    Convert a metric ratio to difficulty level.

    Args:
        ratio: Metric ratio (>= 1.0)

    Returns:
        Corresponding difficulty level
    """
    level = difficulty_from_ratio(ratio)
    return DifficultyLevel(level)


def suggest_difficulty_improvement(
    current: Dict[DifficultyLevel, int],
    target: Dict[DifficultyLevel, float],
    total_target: int
) -> Dict[str, Any]:
    """
    Suggest how to improve difficulty distribution.

    Args:
        current: Current counts by difficulty
        target: Target proportions by difficulty
        total_target: Desired total count

    Returns:
        Suggestions for adding/removing constraints
    """
    suggestions = []
    total_current = sum(current.values())

    for level in DifficultyLevel:
        current_count = current.get(level, 0)
        target_count = int(total_target * target.get(level, 0))

        diff = target_count - current_count
        if diff > 0:
            suggestions.append({
                "action": "add",
                "level": level.name,
                "count": diff,
            })
        elif diff < 0:
            suggestions.append({
                "action": "remove",
                "level": level.name,
                "count": -diff,
            })

    return {
        "current_total": total_current,
        "target_total": total_target,
        "suggestions": suggestions,
    }
