"""
序空间谓词：QRR（四元）和 TRR（三元）关系。

QRR（四元相对关系 Quaternary Relative Relations）：
    比较四个物体之间的成对度量：
    "dist(A,B) 是否小于 dist(C,D)?"

    支持的度量类型：
    - DIST_3D: 3D欧氏距离（视角不变）
    - DIST_2D: 2D图像平面距离（视角相关）
    - DEPTH_GAP: 深度差
    - SIZE_RATIO: 尺寸比

TRR（三元时钟关系 Ternary Clock Relations）：
    使用时钟表盘方向的方向关系：
    "A 相对于 B->C 轴的位置是几点钟?"
    - 12 个小时位置（每个 30°）
    - 4 个象限（每个 90°）

Ordinal spatial predicates: QRR (Quaternary) and TRR (Ternary) relations.

QRR (Quaternary Relative Relations):
    Compare pairwise metrics across four objects:
    "Is dist(A,B) < dist(C,D)?"

TRR (Ternary Clock Relations):
    Directional relations using clock-face orientation:
    "What is A's position relative to the B->C axis?"
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, List, Dict, Optional, Any, Union
import math
import numpy as np

from ordinal_spatial.dsl.comparators import Comparator, compare, compare_ratio


class MetricType(Enum):
    """
    QRR 比较的成对度量类型。

    每个度量代表空间关系的不同方面：
    - DIST_3D: 真实 3D 欧氏距离（视角不变）
    - DIST_2D: 投影的 2D 图像平面距离（视角相关）
    - DEPTH_GAP: 深度/z 坐标的绝对差
    - SIZE_RATIO: 物体尺寸比

    Types of pairwise metrics for QRR comparisons.

    Each metric represents a different aspect of spatial relationships:
    - DIST_3D: True 3D Euclidean distance (view-invariant)
    - DIST_2D: Projected 2D image-plane distance (view-dependent)
    - DEPTH_GAP: Absolute difference in depth/z-coordinate
    - SIZE_RATIO: Ratio of object sizes
    """
    DIST_3D = "dist3D"
    DIST_2D = "dist2D"
    DEPTH_GAP = "depthGap"
    SIZE_RATIO = "sizeRatio"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def from_string(cls, s: str) -> "MetricType":
        """
        从字符串解析度量类型。

        Parse metric type from string.
        """
        mapping = {
            "dist3d": cls.DIST_3D,
            "dist3D": cls.DIST_3D,
            "dist_3d": cls.DIST_3D,
            "dist2d": cls.DIST_2D,
            "dist2D": cls.DIST_2D,
            "dist_2d": cls.DIST_2D,
            "depthgap": cls.DEPTH_GAP,
            "depthGap": cls.DEPTH_GAP,
            "depth_gap": cls.DEPTH_GAP,
            "sizeratio": cls.SIZE_RATIO,
            "sizeRatio": cls.SIZE_RATIO,
            "size_ratio": cls.SIZE_RATIO,
        }
        if s in mapping:
            return mapping[s]
        raise ValueError(f"Unknown metric type: {s}")


@dataclass
class QRRConstraint:
    """
    四元相对关系约束。

    比较两对物体之间的度量：
    metric(pair1[0], pair1[1]) <comparator> metric(pair2[0], pair2[1])

    属性:
        pair1: 第一对物体 ID (A, B)
        pair2: 第二对物体 ID (C, D)
        metric: 被比较的度量类型
        comparator: 比较结果 (<, ~=, >)
        ratio: 实际度量比率（用于难度评估）
        difficulty: 基于比率的难度等级 (1-6)
        boundary_flag: 如果接近容差阈值则为 True

    Quaternary Relative Relation constraint.

    Compares a metric between two pairs of objects:
    metric(pair1[0], pair1[1]) <comparator> metric(pair2[0], pair2[1])

    Attributes:
        pair1: First pair of object IDs (A, B)
        pair2: Second pair of object IDs (C, D)
        metric: Type of metric being compared
        comparator: The comparison result (<, ~=, >)
        ratio: Actual ratio of metrics (for difficulty assessment)
        difficulty: Difficulty level (1-6) based on ratio
        boundary_flag: True if near tolerance threshold
    """
    pair1: Tuple[str, str]
    pair2: Tuple[str, str]
    metric: MetricType
    comparator: Comparator
    ratio: float = 1.0
    difficulty: int = 3
    boundary_flag: bool = False

    def __post_init__(self):
        # Ensure pairs are sorted for canonical representation
        self.pair1 = tuple(sorted(self.pair1))
        self.pair2 = tuple(sorted(self.pair2))

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为 JSON 可序列化的字典。

        Convert to JSON-serializable dictionary.
        """
        return {
            "pair1": list(self.pair1),
            "pair2": list(self.pair2),
            "metric": str(self.metric),
            "comparator": str(self.comparator),
            "ratio": self.ratio,
            "difficulty": self.difficulty,
            "boundary_flag": self.boundary_flag,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QRRConstraint":
        """
        从字典创建约束。

        Create from dictionary.
        """
        return cls(
            pair1=tuple(d["pair1"]),
            pair2=tuple(d["pair2"]),
            metric=MetricType.from_string(d["metric"]),
            comparator=Comparator.from_string(d["comparator"]),
            ratio=d.get("ratio", 1.0),
            difficulty=d.get("difficulty", 3),
            boundary_flag=d.get("boundary_flag", False),
        )

    def flip(self) -> "QRRConstraint":
        """
        返回交换对后的约束（并翻转比较器）。

        Return constraint with pairs swapped (and comparator flipped).
        """
        return QRRConstraint(
            pair1=self.pair2,
            pair2=self.pair1,
            metric=self.metric,
            comparator=self.comparator.flip(),
            ratio=1.0 / self.ratio if self.ratio > 0 else float('inf'),
            difficulty=self.difficulty,
            boundary_flag=self.boundary_flag,
        )

    def canonical_key(self) -> Tuple:
        """
        返回用于去重的规范键。

        Return canonical key for deduplication.
        """
        # Always put smaller pair first
        if self.pair1 < self.pair2:
            return (self.pair1, self.pair2, self.metric)
        else:
            return (self.pair2, self.pair1, self.metric)


@dataclass
class TRRConstraint:
    """
    三元时钟关系约束。

    描述目标物体相对于由另外两个物体定义的参考轴的角度位置。

    时钟模型：
        - 原点在 ref1 (B)
        - 12 点钟方向指向 ref2 (C)
        - 小时数顺时针增加
        - 每小时跨度 30 度

    属性:
        target: 目标物体 ID (A)，我们描述其位置
        ref1: 原点物体 ID (B)
        ref2: 方向参考物体 ID (C)
        hour: 时钟小时数 (1-12)
        quadrant: 粗略象限 (1-4)
        angle_deg: 精确角度（度数，用于调试）

    Ternary Clock Relation constraint.

    Describes the angular position of a target object relative to
    a reference axis defined by two other objects.

    Clock Model:
        - Origin at ref1 (B)
        - 12 o'clock points toward ref2 (C)
        - Hours increase clockwise
        - Each hour spans 30 degrees

    Attributes:
        target: Target object ID (A) whose position we're describing
        ref1: Origin object ID (B)
        ref2: Direction reference object ID (C)
        hour: Clock hour (1-12)
        quadrant: Coarse quadrant (1-4)
        angle_deg: Exact angle in degrees (for debugging)
    """
    target: str
    ref1: str
    ref2: str
    hour: int
    quadrant: int = 0  # 0 means auto-calculate
    angle_deg: float = 0.0

    def __post_init__(self):
        if not 1 <= self.hour <= 12:
            raise ValueError(f"Hour must be 1-12, got {self.hour}")
        # Auto-calculate quadrant if not provided
        if self.quadrant == 0:
            object.__setattr__(self, 'quadrant', hour_to_quadrant(self.hour))
        if not 1 <= self.quadrant <= 4:
            raise ValueError(f"Quadrant must be 1-4, got {self.quadrant}")

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为 JSON 可序列化的字典。

        Convert to JSON-serializable dictionary.
        """
        return {
            "target": self.target,
            "ref1": self.ref1,
            "ref2": self.ref2,
            "hour": self.hour,
            "quadrant": self.quadrant,
            "angle_deg": self.angle_deg,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TRRConstraint":
        """
        从字典创建约束。

        Create from dictionary.
        """
        return cls(
            target=d["target"],
            ref1=d["ref1"],
            ref2=d["ref2"],
            hour=d["hour"],
            quadrant=d.get("quadrant", hour_to_quadrant(d["hour"])),
            angle_deg=d.get("angle_deg", 0.0),
        )

    def triple_key(self) -> Tuple[str, str, str]:
        """
        返回此约束的有序三元组。

        Return the ordered triple for this constraint.
        """
        return (self.target, self.ref1, self.ref2)


def hour_to_quadrant(hour: int) -> int:
    """
    将时钟小时数（1-12）转换为象限（1-4）。

    Convert clock hour (1-12) to quadrant (1-4).
    """
    if hour in (12, 1, 2):
        return 1  # Top-right (toward ref2)
    elif hour in (3, 4, 5):
        return 2  # Bottom-right
    elif hour in (6, 7, 8):
        return 3  # Bottom-left
    else:  # 9, 10, 11
        return 4  # Top-left


def angle_to_hour(angle_deg: float) -> int:
    """
    Convert angle (in degrees) to clock hour.

    Angle convention:
        - 0 degrees = 12 o'clock (toward ref2)
        - Angles increase clockwise
        - Each hour spans 30 degrees

    Args:
        angle_deg: Angle in degrees [0, 360)

    Returns:
        Clock hour 1-12
    """
    # Normalize angle to [0, 360)
    angle_deg = angle_deg % 360

    # Each hour is 30 degrees
    # Hour 12 is centered at 0 degrees (-15 to +15)
    # Hour 1 is centered at 30 degrees (15 to 45)
    # etc.

    # Shift so hour boundaries align with multiples of 30
    shifted = (angle_deg + 15) % 360

    # Calculate hour (0-11 then map to 1-12)
    hour_idx = int(shifted // 30)
    hour = (hour_idx % 12)
    if hour == 0:
        hour = 12

    return hour


def hour_to_angle_center(hour: int) -> float:
    """
    获取时钟小时数对应的中心角度（度数）。

    Get the center angle (in degrees) for a clock hour.
    """
    # Hour 12 -> 0 degrees
    # Hour 1 -> 30 degrees
    # etc.
    return ((hour % 12) * 30) % 360


def compute_angle_2d(
    target_pos: np.ndarray,
    ref1_pos: np.ndarray,
    ref2_pos: np.ndarray
) -> float:
    """
    Compute the clock angle of target relative to ref1->ref2 axis.

    Args:
        target_pos: 2D position of target [x, y]
        ref1_pos: 2D position of origin/ref1 [x, y]
        ref2_pos: 2D position of direction reference/ref2 [x, y]

    Returns:
        Angle in degrees [0, 360), where 0 = toward ref2 (12 o'clock)
    """
    # Vector from ref1 to ref2 (defines 12 o'clock direction)
    ref_vec = ref2_pos - ref1_pos
    ref_angle = math.atan2(ref_vec[1], ref_vec[0])

    # Vector from ref1 to target
    target_vec = target_pos - ref1_pos

    # Handle zero-length vectors
    if np.linalg.norm(target_vec) < 1e-10:
        return 0.0

    target_angle = math.atan2(target_vec[1], target_vec[0])

    # Compute relative angle (clockwise from ref direction)
    # Note: In image coordinates, Y increases downward, so clockwise
    # corresponds to increasing angle in standard math coordinates
    rel_angle = target_angle - ref_angle

    # Convert to degrees and normalize to [0, 360)
    angle_deg = math.degrees(rel_angle) % 360

    return angle_deg


# =============================================================================
# Metric Computation Functions
# =============================================================================

def compute_dist_3d(obj_a: Dict, obj_b: Dict) -> float:
    """
    计算两个物体之间的 3D 欧氏距离。

    Compute 3D Euclidean distance between two objects.
    """
    pos_a = np.array(obj_a.get("position_3d", obj_a.get("3d_coords", [0, 0, 0])))
    pos_b = np.array(obj_b.get("position_3d", obj_b.get("3d_coords", [0, 0, 0])))
    return float(np.linalg.norm(pos_a - pos_b))


def compute_dist_2d(obj_a: Dict, obj_b: Dict) -> float:
    """
    计算两个物体之间的 2D 图像平面距离。

    Compute 2D image-plane distance between two objects.
    """
    pos_a = np.array(obj_a.get("position_2d", obj_a.get("pixel_coords", [0, 0])[:2]))
    pos_b = np.array(obj_b.get("position_2d", obj_b.get("pixel_coords", [0, 0])[:2]))
    return float(np.linalg.norm(pos_a - pos_b))


def compute_depth_gap(obj_a: Dict, obj_b: Dict) -> float:
    """
    计算两个物体之间的绝对深度差。

    Compute absolute depth difference between two objects.
    """
    depth_a = obj_a.get("depth", obj_a.get("pixel_coords", [0, 0, 0])[2] if len(obj_a.get("pixel_coords", [])) > 2 else 0)
    depth_b = obj_b.get("depth", obj_b.get("pixel_coords", [0, 0, 0])[2] if len(obj_b.get("pixel_coords", [])) > 2 else 0)
    return abs(float(depth_a) - float(depth_b))


def compute_size_ratio(obj_a: Dict, obj_b: Dict) -> float:
    """
    计算两个物体之间的尺寸比（a/b）。

    Compute size ratio between two objects (a/b).
    """
    size_a = obj_a.get("size", 1.0)
    size_b = obj_b.get("size", 1.0)

    # Handle size as string (e.g., "large", "medium", "small")
    size_map = {"large": 0.7, "medium": 0.5, "small": 0.35}
    if isinstance(size_a, str):
        size_a = size_map.get(size_a.lower(), 0.5)
    if isinstance(size_b, str):
        size_b = size_map.get(size_b.lower(), 0.5)

    return float(size_a) / float(size_b) if size_b > 0 else float('inf')


METRIC_FUNCTIONS = {
    MetricType.DIST_3D: compute_dist_3d,
    MetricType.DIST_2D: compute_dist_2d,
    MetricType.DEPTH_GAP: compute_depth_gap,
    MetricType.SIZE_RATIO: compute_size_ratio,
}


# =============================================================================
# High-Level Constraint Computation
# =============================================================================

def compute_qrr(
    objects: Dict[str, Dict],
    pair1: Tuple[str, str],
    pair2: Tuple[str, str],
    metric: MetricType,
    tau: float = 0.10
) -> QRRConstraint:
    """
    从物体数据计算 QRR（四元相对关系）约束。

    比较两对物体之间的度量关系，例如：
    "dist(A,B) 与 dist(C,D) 的关系是 < | ~= | >"

    参数:
        objects: 物体ID到物体数据的字典映射
        pair1: 第一对物体 (A, B)
        pair2: 第二对物体 (C, D)
        metric: 要比较的度量类型（DIST_3D/DIST_2D/DEPTH_GAP/SIZE_RATIO）
        tau: 近似相等的容差（默认 0.10）

    返回:
        包含计算的比较器和元数据的 QRRConstraint 对象

    示例:
        >>> objects = {
        ...     "A": {"position": [0, 0, 0]},
        ...     "B": {"position": [1, 0, 0]},
        ...     "C": {"position": [0, 0, 0]},
        ...     "D": {"position": [2, 0, 0]},
        ... }
        >>> constraint = compute_qrr(objects, ("A", "B"), ("C", "D"), MetricType.DIST_3D)
        >>> print(constraint.comparator)  # "<" (因为 dist(A,B)=1 < dist(C,D)=2)

    Compute a QRR constraint from object data.

    Args:
        objects: Dictionary mapping object IDs to object data
        pair1: First object pair (A, B)
        pair2: Second object pair (C, D)
        metric: Type of metric to compare
        tau: Tolerance for approximate equality

    Returns:
        QRRConstraint with computed comparator and metadata
    """
    metric_func = METRIC_FUNCTIONS[metric]

    # Compute metrics for each pair
    m1 = metric_func(objects[pair1[0]], objects[pair1[1]])
    m2 = metric_func(objects[pair2[0]], objects[pair2[1]])

    # Compare with tolerance
    comparator, ratio = compare_ratio(m1, m2, tau)

    # Check if near boundary
    max_val = max(m1, m2)
    threshold = tau * max_val
    diff = abs(m1 - m2)
    boundary_flag = diff > 0.8 * threshold and diff < 1.2 * threshold

    # Compute difficulty
    from ordinal_spatial.dsl.comparators import difficulty_from_ratio
    difficulty = difficulty_from_ratio(ratio)

    return QRRConstraint(
        pair1=pair1,
        pair2=pair2,
        metric=metric,
        comparator=comparator,
        ratio=ratio,
        difficulty=difficulty,
        boundary_flag=boundary_flag,
    )


def compute_trr(
    objects: Dict[str, Dict],
    target: str,
    ref1: str,
    ref2: str,
    use_3d: bool = False
) -> TRRConstraint:
    """
    从物体数据计算 TRR（三元时钟关系）约束。

    使用时钟表盘模型描述目标物体相对于参考轴的方向：
    - 站在 ref1（原点）
    - 面向 ref2（参考方向 = 12点）
    - 确定 target 在几点钟方向

    参数:
        objects: 物体ID到物体数据的字典映射
        target: 目标物体ID（要确定位置的物体）
        ref1: 原点/参考物体ID（观察者站立位置）
        ref2: 方向参考物体ID（12点钟方向）
        use_3d: 如果为 True，使用 3D 位置投影到 XY 平面

    返回:
        包含计算的小时和象限的 TRRConstraint 对象

    示例:
        >>> objects = {
        ...     "Center": {"position": [0, 0, 0]},
        ...     "North": {"position": [1, 0, 0]},  # 12点方向
        ...     "Target": {"position": [1, 1, 0]},  # 1-2点方向
        ... }
        >>> constraint = compute_trr(objects, "Target", "Center", "North")
        >>> print(f"{constraint.hour}点，象限{constraint.quadrant}")

    Compute a TRR constraint from object data.

    Args:
        objects: Dictionary mapping object IDs to object data
        target: Target object ID
        ref1: Origin/reference object ID
        ref2: Direction reference object ID
        use_3d: If True, use 3D positions projected to XY plane

    Returns:
        TRRConstraint with computed hour and quadrant
    """
    if use_3d:
        # Use 3D coordinates (XY projection)
        target_pos = np.array(objects[target].get("position_3d", [0, 0, 0])[:2])
        ref1_pos = np.array(objects[ref1].get("position_3d", [0, 0, 0])[:2])
        ref2_pos = np.array(objects[ref2].get("position_3d", [0, 0, 0])[:2])
    else:
        # Use 2D image coordinates
        target_pos = np.array(objects[target].get("position_2d", [0, 0])[:2])
        ref1_pos = np.array(objects[ref1].get("position_2d", [0, 0])[:2])
        ref2_pos = np.array(objects[ref2].get("position_2d", [0, 0])[:2])

    # Compute angle
    angle_deg = compute_angle_2d(target_pos, ref1_pos, ref2_pos)

    # Convert to hour
    hour = angle_to_hour(angle_deg)
    quadrant = hour_to_quadrant(hour)

    return TRRConstraint(
        target=target,
        ref1=ref1,
        ref2=ref2,
        hour=hour,
        quadrant=quadrant,
        angle_deg=angle_deg,
    )


def extract_all_qrr(
    objects: Dict[str, Dict],
    metric: MetricType,
    tau: float = 0.10,
    disjoint_only: bool = True
) -> List[QRRConstraint]:
    """
    Extract all QRR constraints for given objects and metric.

    Args:
        objects: Dictionary mapping object IDs to object data
        metric: Type of metric to compare
        tau: Tolerance for approximate equality
        disjoint_only: If True, only compare disjoint pairs (no shared objects)

    Returns:
        List of QRRConstraint objects
    """
    from itertools import combinations

    obj_ids = list(objects.keys())
    pairs = list(combinations(obj_ids, 2))
    constraints = []

    for i, pair1 in enumerate(pairs):
        for pair2 in pairs[i + 1:]:
            # Check for disjoint pairs
            if disjoint_only:
                if set(pair1) & set(pair2):
                    continue

            constraint = compute_qrr(objects, pair1, pair2, metric, tau)
            constraints.append(constraint)

    return constraints


def extract_all_trr(
    objects: Dict[str, Dict],
    use_3d: bool = False
) -> List[TRRConstraint]:
    """
    Extract all TRR constraints for given objects.

    Args:
        objects: Dictionary mapping object IDs to object data
        use_3d: If True, use 3D positions

    Returns:
        List of TRRConstraint objects
    """
    from itertools import permutations

    obj_ids = list(objects.keys())
    constraints = []

    # For each ordered triple (target, ref1, ref2)
    for triple in permutations(obj_ids, 3):
        target, ref1, ref2 = triple
        constraint = compute_trr(objects, target, ref1, ref2, use_3d)
        constraints.append(constraint)

    return constraints


def clock_angular_error(pred_hour: int, gt_hour: int) -> float:
    """
    Compute angular error between predicted and ground-truth clock hours.

    Handles wrap-around (e.g., hour 12 vs hour 1 = 30 degrees, not 330).

    Args:
        pred_hour: Predicted hour (1-12)
        gt_hour: Ground-truth hour (1-12)

    Returns:
        Angular error in degrees (0-180)
    """
    # Convert to 0-11 for easier math
    pred_idx = (pred_hour - 1) % 12
    gt_idx = (gt_hour - 1) % 12

    # Compute minimum circular distance
    diff = abs(pred_idx - gt_idx)
    min_diff = min(diff, 12 - diff)

    return min_diff * 30.0  # Each hour is 30 degrees
