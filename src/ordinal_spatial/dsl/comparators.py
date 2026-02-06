"""
基于容差的序关系比较代数。

本模块实现了 ORDINAL-SPATIAL 基准测试中使用的核心比较器逻辑。
核心思想是通过相对比较而非绝对值来评估空间理解能力。

比较器代数：
    - a <_tau b  当且仅当  a < b * (1 - tau)     [严格小于]
    - a ~=_tau b 当且仅当  |a - b| <= tau * max(a, b)  [约等于]
    - a >_tau b  当且仅当  a > b * (1 + tau)     [严格大于]

Tolerance-based comparison algebra for ordinal spatial relations.

This module implements the core comparator logic used throughout the
ORDINAL-SPATIAL benchmark. The key insight is that spatial understanding
should be evaluated through relative comparisons rather than absolute values.

Comparator Algebra:
    - a <_tau b  iff  a < b * (1 - tau)
    - a ~=_tau b iff  |a - b| <= tau * max(a, b)
    - a >_tau b  iff  a > b * (1 + tau)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Union, Tuple


class Comparator(Enum):
    """
    三值序关系比较器。

    比较器形成全序关系：LT < APPROX < GT
    这使得可以计算预测之间的序距离。

    Three-way comparator for ordinal relations.

    The comparators form a total order: LT < APPROX < GT
    This allows ordinal distance calculations between predictions.
    """
    LT = "<"       # strictly less than
    APPROX = "~="  # approximately equal (within tolerance)
    GT = ">"       # strictly greater than

    def __str__(self) -> str:
        return self.value

    def __repr__(self) -> str:
        return f"Comparator.{self.name}"

    @property
    def ordinal(self) -> int:
        """
        返回用于距离计算的序数值。

        Return ordinal value for distance calculations.
        """
        return {Comparator.LT: 0, Comparator.APPROX: 1, Comparator.GT: 2}[self]

    def flip(self) -> "Comparator":
        """
        返回相反的比较器（用于翻转对的顺序）。

        Return the opposite comparator (for reversing pair order).
        """
        if self == Comparator.LT:
            return Comparator.GT
        elif self == Comparator.GT:
            return Comparator.LT
        return Comparator.APPROX

    @classmethod
    def from_string(cls, s: str) -> "Comparator":
        """
        从字符串表示解析比较器。

        Parse comparator from string representation.
        """
        s = s.strip()
        mapping = {
            "<": cls.LT,
            "~=": cls.APPROX,
            "≈": cls.APPROX,
            "=": cls.APPROX,
            ">": cls.GT,
            "lt": cls.LT,
            "eq": cls.APPROX,
            "approx": cls.APPROX,
            "gt": cls.GT,
        }
        if s.lower() in mapping:
            return mapping[s.lower()]
        raise ValueError(f"Unknown comparator: {s}")


@dataclass
class TolerancePreset:
    """
    预定义的容差等级，用于不同难度设置。

    属性:
        name: 预设的人类可读名称
        tau: 容差值 (0 < tau < 1)
        description: 何时使用此预设的说明

    Predefined tolerance levels for different difficulty settings.

    Attributes:
        name: Human-readable name for the preset
        tau: Tolerance value (0 < tau < 1)
        description: Explanation of when to use this preset
    """
    name: str
    tau: float
    description: str

    def __post_init__(self):
        if not 0 < self.tau < 1:
            raise ValueError(f"Tolerance tau must be in (0, 1), got {self.tau}")


# Standard tolerance presets
TOLERANCE_STRICT = TolerancePreset(
    name="strict",
    tau=0.05,
    description="Fine discrimination required (~10% approx frequency)"
)

TOLERANCE_STANDARD = TolerancePreset(
    name="standard",
    tau=0.10,
    description="Balanced difficulty (~19% approx frequency)"
)

TOLERANCE_RELAXED = TolerancePreset(
    name="relaxed",
    tau=0.20,
    description="Coarse discrimination (~33% approx frequency)"
)

# Mapping for easy access
TOLERANCE_PRESETS = {
    "strict": TOLERANCE_STRICT,
    "standard": TOLERANCE_STANDARD,
    "relaxed": TOLERANCE_RELAXED,
}


def compare(a: float, b: float, tau: float = 0.10) -> Comparator:
    """
    使用基于容差的近似相等比较两个值。

    比较规则：
    - a <_tau b  当且仅当  a < b * (1 - tau)  [严格小于]
    - a ~=_tau b 当且仅当  |a - b| <= tau * max(a, b)  [约等于]
    - a >_tau b  当且仅当  a > b * (1 + tau)  [严格大于]

    参数:
        a: 第一个待比较的值
        b: 第二个待比较的值
        tau: 容差参数（默认 0.10 = 10%）

    返回:
        表示 a 和 b 关系的 Comparator 枚举值

    示例:
        >>> compare(1.0, 2.0, 0.10)
        Comparator.LT
        >>> compare(1.0, 1.05, 0.10)
        Comparator.APPROX
        >>> compare(2.0, 1.0, 0.10)
        Comparator.GT

    Compare two values with tolerance-based approximate equality.

    The comparison follows these rules:
    - a <_tau b  iff  a < b * (1 - tau)  [strictly less]
    - a ~=_tau b iff  |a - b| <= tau * max(a, b)  [approximately equal]
    - a >_tau b  iff  a > b * (1 + tau)  [strictly greater]

    Args:
        a: First value to compare
        b: Second value to compare
        tau: Tolerance parameter (default 0.10 = 10%)

    Returns:
        Comparator indicating the relationship between a and b

    Examples:
        >>> compare(1.0, 2.0, 0.10)
        Comparator.LT
        >>> compare(1.0, 1.05, 0.10)
        Comparator.APPROX
        >>> compare(2.0, 1.0, 0.10)
        Comparator.GT
    """
    if tau <= 0 or tau >= 1:
        raise ValueError(f"Tolerance tau must be in (0, 1), got {tau}")

    # Handle edge cases
    if a < 0 or b < 0:
        raise ValueError(f"Values must be non-negative, got a={a}, b={b}")

    # Both zero case
    if a == 0 and b == 0:
        return Comparator.APPROX

    # One zero case
    if a == 0:
        return Comparator.LT
    if b == 0:
        return Comparator.GT

    max_val = max(a, b)
    threshold = tau * max_val
    diff = a - b

    if abs(diff) <= threshold:
        return Comparator.APPROX
    elif diff < 0:
        return Comparator.LT
    else:
        return Comparator.GT


def compare_ratio(a: float, b: float, tau: float = 0.10) -> Tuple[Comparator, float]:
    """
    比较两个值并返回比较器和比率。

    比率 a/b 用于难度分类：
    - 比率 > 2.0: 简单（明显差异）
    - 比率 1.3-2.0: 中等
    - 比率 1.05-1.3: 困难
    - 比率 ~1.0: 极难（接近阈值）

    参数:
        a: 第一个值（分子）
        b: 第二个值（分母）
        tau: 容差参数

    返回:
        元组 (comparator, ratio)，其中 ratio = a/b

    Compare two values and return both the comparator and the ratio.

    The ratio a/b is useful for difficulty classification:
    - ratio > 2.0: Easy (obvious difference)
    - ratio 1.3-2.0: Medium
    - ratio 1.05-1.3: Hard
    - ratio ~1.0: Very hard (near threshold)

    Args:
        a: First value (numerator)
        b: Second value (denominator)
        tau: Tolerance parameter

    Returns:
        Tuple of (comparator, ratio) where ratio = a/b
    """
    comparator = compare(a, b, tau)
    ratio = a / b if b > 0 else float('inf')
    return comparator, ratio


def ordinal_distance(pred: Comparator, gt: Comparator) -> int:
    """
    计算预测和真值比较器之间的序距离。

    距离基于序位置：
    - LT = 0, APPROX = 1, GT = 2
    - 距离 = |pred.ordinal - gt.ordinal|

    这使得 LT<->GT 翻转（距离=2）比 LT<->APPROX（距离=1）惩罚更重。

    参数:
        pred: 预测的比较器
        gt: 真值比较器

    返回:
        序距离（0、1 或 2）

    Calculate ordinal distance between predicted and ground-truth comparators.

    Distance is based on the ordinal positions:
    - LT = 0, APPROX = 1, GT = 2
    - Distance = |pred.ordinal - gt.ordinal|

    This penalizes LT<->GT flips (distance=2) more than LT<->APPROX (distance=1).

    Args:
        pred: Predicted comparator
        gt: Ground-truth comparator

    Returns:
        Ordinal distance (0, 1, or 2)
    """
    return abs(pred.ordinal - gt.ordinal)


def is_flip(pred: Comparator, gt: Comparator) -> bool:
    """
    检查预测是否为完全翻转（LT <-> GT）。

    翻转是最严重的错误，表示对相对顺序的根本误解。

    参数:
        pred: 预测的比较器
        gt: 真值比较器

    返回:
        如果 pred 和 gt 互为相反（LT vs GT）则返回 True

    Check if prediction is a complete flip (LT <-> GT).

    Flips are the most severe errors, representing a fundamental
    misunderstanding of the relative ordering.

    Args:
        pred: Predicted comparator
        gt: Ground-truth comparator

    Returns:
        True if pred and gt are opposites (LT vs GT)
    """
    return (pred == Comparator.LT and gt == Comparator.GT) or \
           (pred == Comparator.GT and gt == Comparator.LT)


def difficulty_from_ratio(ratio: float) -> int:
    """
    从度量比率确定难度等级（1-6）。

    难度分级：
        1: 比率 > 2.0（简单 - 明显差异）
        2: 1.5 < 比率 <= 2.0
        3: 1.3 < 比率 <= 1.5
        4: 1.15 < 比率 <= 1.3
        5: 1.05 < 比率 <= 1.15（困难）
        6: 比率 <= 1.05（极难 - 接近阈值）

    参数:
        ratio: 比率 a/b（按惯例总是 >= 1.0）

    返回:
        难度等级，从 1（最简单）到 6（最困难）

    Determine difficulty level (1-6) from metric ratio.

    Difficulty bands:
        1: ratio > 2.0 (easy - obvious difference)
        2: 1.5 < ratio <= 2.0
        3: 1.3 < ratio <= 1.5
        4: 1.15 < ratio <= 1.3
        5: 1.05 < ratio <= 1.15 (hard)
        6: ratio <= 1.05 (extreme - near threshold)

    Args:
        ratio: The ratio a/b (always >= 1.0 by convention)

    Returns:
        Difficulty level from 1 (easiest) to 6 (hardest)
    """
    # Normalize ratio to be >= 1.0
    if ratio < 1.0:
        ratio = 1.0 / ratio

    if ratio > 2.0:
        return 1
    elif ratio > 1.5:
        return 2
    elif ratio > 1.3:
        return 3
    elif ratio > 1.15:
        return 4
    elif ratio > 1.05:
        return 5
    else:
        return 6


class ComparatorChain:
    """
    用于传递性检查的比较链。

    用于验证一致性：如果 A < B 且 B < C，则 A < C。
    违反表示存在矛盾的约束。

    A chain of comparisons for transitivity checking.

    Used to verify consistency: if A < B and B < C, then A < C.
    Violations indicate contradictory constraints.
    """

    def __init__(self):
        self._comparisons = []

    def add(self, comparator: Comparator) -> "ComparatorChain":
        """
        向链中添加一个比较。

        Add a comparison to the chain.
        """
        self._comparisons.append(comparator)
        return self

    def implies(self) -> Comparator:
        """
        计算整个链暗示的关系。

        返回:
            由传递性暗示的比较器，如果不一致则返回 None

        Compute the implied relationship across the entire chain.

        Returns:
            The comparator implied by transitivity, or None if inconsistent
        """
        if not self._comparisons:
            return Comparator.APPROX

        # Track net direction
        lt_count = sum(1 for c in self._comparisons if c == Comparator.LT)
        gt_count = sum(1 for c in self._comparisons if c == Comparator.GT)

        # Any strict inequality dominates approx
        if lt_count > 0 and gt_count == 0:
            return Comparator.LT
        elif gt_count > 0 and lt_count == 0:
            return Comparator.GT
        elif lt_count == 0 and gt_count == 0:
            return Comparator.APPROX
        else:
            # Mixed directions - inconsistent
            return None

    def is_consistent(self) -> bool:
        """
        检查链是否内部一致。

        Check if the chain is internally consistent.
        """
        return self.implies() is not None
