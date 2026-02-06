"""
场景配置的退化检测。

本模块检测并标记在数据集生成过程中应被拒绝的退化空间配置。

退化类型包括：
- 近等情况（NEAR_EQUALITY）：过多约等于关系，导致比较模糊
- 共线排列（COLLINEARITY）：物体在一条直线上
- 重合物体（COINCIDENT）：物体位置过于接近
- 平凡支配（TRIVIAL_DOMINANCE）：一对距离远大于所有其他对
- 低标签多样性（LOW_DIVERSITY）：约束类型分布不均
- 聚类（CLUSTERING）：物体过度聚集

使用场景：
- 数据集生成时过滤无效场景
- 确保约束分布有意义
- 避免模糊或不明确的测试样例

Degeneracy checking for scene configurations.

This module detects and flags degenerate spatial configurations
that should be rejected during dataset generation.

Degeneracies include:
- Near-equality cases (ambiguous comparisons)
- Collinear object arrangements
- Coincident objects
- Trivial dominance (one pair >> all others)
- Low label diversity
"""

from typing import List, Dict, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
import numpy as np
from collections import Counter

from ordinal_spatial.dsl.predicates import QRRConstraint, compute_dist_3d
from ordinal_spatial.dsl.comparators import Comparator


class DegeneracyType(Enum):
    """Types of degenerate configurations."""
    NEAR_EQUALITY = "near_equality"
    COLLINEARITY = "collinearity"
    COINCIDENT = "coincident"
    TRIVIAL_DOMINANCE = "trivial_dominance"
    LOW_DIVERSITY = "low_diversity"
    CLUSTERING = "clustering"


@dataclass
class Degeneracy:
    """
    A detected degeneracy in the scene.

    Attributes:
        type: Type of degeneracy
        severity: Severity score (0-1, higher = worse)
        objects: Objects involved in the degeneracy
        description: Human-readable description
        data: Additional data about the degeneracy
    """
    type: DegeneracyType
    severity: float
    objects: List[str]
    description: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DegeneracyReport:
    """
    Report of all degeneracies found in a scene.

    Attributes:
        is_valid: Whether the scene is valid (no severe degeneracies)
        degeneracies: List of detected degeneracies
        severity_score: Overall severity (0-1)
        recommendation: Action recommendation
    """
    is_valid: bool
    degeneracies: List[Degeneracy]
    severity_score: float
    recommendation: str

    def __str__(self) -> str:
        if self.is_valid:
            return f"Valid scene (severity: {self.severity_score:.2f})"
        else:
            issues = [d.description for d in self.degeneracies[:3]]
            return f"Invalid scene: {'; '.join(issues)}"


class DegeneracyChecker:
    """
    Check scenes for degenerate configurations.

    Usage:
        checker = DegeneracyChecker()
        report = checker.check(objects)
        if not report.is_valid:
            print(f"Rejected: {report.recommendation}")
    """

    def __init__(
        self,
        tau: float = 0.10,
        min_separation: float = 0.3,
        collinearity_threshold: float = 0.1,
        diversity_threshold: float = 0.5,
        max_severity: float = 0.5,
    ):
        """
        Initialize the checker.

        Args:
            tau: Tolerance parameter for comparisons
            min_separation: Minimum distance between objects
            collinearity_threshold: Threshold for collinearity detection
            diversity_threshold: Minimum label entropy
            max_severity: Maximum allowed severity for valid scene
        """
        self.tau = tau
        self.min_separation = min_separation
        self.collinearity_threshold = collinearity_threshold
        self.diversity_threshold = diversity_threshold
        self.max_severity = max_severity

    def check(self, objects: Dict[str, Dict]) -> DegeneracyReport:
        """
        Check a scene for degeneracies.

        Args:
            objects: Dictionary of object_id -> object_data

        Returns:
            DegeneracyReport with findings
        """
        degeneracies = []

        # Check for coincident objects
        degeneracies.extend(self._check_coincident(objects))

        # Check for collinearity
        degeneracies.extend(self._check_collinearity(objects))

        # Check for near-equality in distances
        degeneracies.extend(self._check_near_equality(objects))

        # Check for trivial dominance
        degeneracies.extend(self._check_trivial_dominance(objects))

        # Check for low diversity
        degeneracies.extend(self._check_low_diversity(objects))

        # Calculate overall severity
        if degeneracies:
            severity_score = max(d.severity for d in degeneracies)
        else:
            severity_score = 0.0

        is_valid = severity_score <= self.max_severity

        # Generate recommendation
        if is_valid:
            recommendation = "Scene is valid for use"
        elif severity_score > 0.8:
            recommendation = "Reject and regenerate scene"
        else:
            recommendation = "Consider adjusting object positions"

        return DegeneracyReport(
            is_valid=is_valid,
            degeneracies=degeneracies,
            severity_score=severity_score,
            recommendation=recommendation,
        )

    def _check_coincident(self, objects: Dict[str, Dict]) -> List[Degeneracy]:
        """
        检查物体是否过于靠近。

        Check for objects that are too close together.
        """
        degeneracies = []
        obj_ids = list(objects.keys())

        for i, id1 in enumerate(obj_ids):
            for id2 in obj_ids[i + 1:]:
                dist = compute_dist_3d(objects[id1], objects[id2])

                if dist < self.min_separation:
                    severity = 1.0 - (dist / self.min_separation)
                    degeneracies.append(Degeneracy(
                        type=DegeneracyType.COINCIDENT,
                        severity=min(1.0, severity),
                        objects=[id1, id2],
                        description=f"Objects {id1} and {id2} are too close (dist={dist:.3f})",
                        data={"distance": dist, "threshold": self.min_separation},
                    ))

        return degeneracies

    def _check_collinearity(self, objects: Dict[str, Dict]) -> List[Degeneracy]:
        """
        检查物体是否共线排列。

        Check for collinear object arrangements.
        """
        degeneracies = []
        obj_ids = list(objects.keys())

        if len(obj_ids) < 3:
            return degeneracies

        # Get 3D positions
        positions = {}
        for obj_id in obj_ids:
            pos = objects[obj_id].get("position_3d",
                   objects[obj_id].get("3d_coords", [0, 0, 0]))
            positions[obj_id] = np.array(pos)

        # Check all triples
        for triple in combinations(obj_ids, 3):
            id1, id2, id3 = triple
            p1, p2, p3 = positions[id1], positions[id2], positions[id3]

            # Compute angle at p2
            v1 = p1 - p2
            v2 = p3 - p2

            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)

            if norm1 < 1e-6 or norm2 < 1e-6:
                continue

            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            cos_angle = np.clip(cos_angle, -1, 1)
            sin_angle = np.sqrt(1 - cos_angle ** 2)

            if sin_angle < self.collinearity_threshold:
                severity = 1.0 - (sin_angle / self.collinearity_threshold)
                degeneracies.append(Degeneracy(
                    type=DegeneracyType.COLLINEARITY,
                    severity=severity * 0.7,  # Lower weight for collinearity
                    objects=list(triple),
                    description=f"Objects {triple} are nearly collinear (sin={sin_angle:.3f})",
                    data={"sin_angle": sin_angle},
                ))

        return degeneracies

    def _check_near_equality(self, objects: Dict[str, Dict]) -> List[Degeneracy]:
        """
        检查模糊的近似相等情况。

        Check for ambiguous near-equality cases.
        """
        degeneracies = []
        obj_ids = list(objects.keys())
        pairs = list(combinations(obj_ids, 2))

        # Compute all pairwise distances
        distances = {}
        for pair in pairs:
            dist = compute_dist_3d(objects[pair[0]], objects[pair[1]])
            distances[pair] = dist

        # Check for near-boundary cases
        boundary_margin = 0.03  # 3% margin around tau
        dist_values = list(distances.values())

        for i, (pair1, d1) in enumerate(distances.items()):
            for pair2, d2 in list(distances.items())[i + 1:]:
                if set(pair1) & set(pair2):
                    continue  # Skip non-disjoint

                max_val = max(d1, d2)
                threshold = self.tau * max_val
                diff = abs(d1 - d2)

                # Check if near boundary
                if threshold * (1 - boundary_margin) < diff < threshold * (1 + boundary_margin):
                    severity = 0.5 - abs(diff - threshold) / (threshold * boundary_margin)
                    degeneracies.append(Degeneracy(
                        type=DegeneracyType.NEAR_EQUALITY,
                        severity=max(0, severity),
                        objects=list(pair1) + list(pair2),
                        description=f"Near-boundary comparison: {pair1} vs {pair2}",
                        data={
                            "distances": {str(pair1): d1, str(pair2): d2},
                            "diff": diff,
                            "threshold": threshold,
                        },
                    ))

        return degeneracies

    def _check_trivial_dominance(self, objects: Dict[str, Dict]) -> List[Degeneracy]:
        """
        检查是否有一对物体支配所有其他对。

        Check if one pair dominates all others.
        """
        degeneracies = []
        obj_ids = list(objects.keys())
        pairs = list(combinations(obj_ids, 2))

        if len(pairs) < 2:
            return degeneracies

        # Compute distances
        distances = [
            compute_dist_3d(objects[p[0]], objects[p[1]])
            for p in pairs
        ]

        max_dist = max(distances)
        other_dists = [d for d in distances if d < max_dist * 0.99]

        if other_dists:
            ratio = max_dist / max(other_dists)
            if ratio > 3.0:
                severity = min(1.0, (ratio - 3.0) / 3.0)
                degeneracies.append(Degeneracy(
                    type=DegeneracyType.TRIVIAL_DOMINANCE,
                    severity=severity * 0.6,
                    objects=list(obj_ids),
                    description=f"One pair dominates (ratio={ratio:.2f})",
                    data={"dominance_ratio": ratio},
                ))

        return degeneracies

    def _check_low_diversity(self, objects: Dict[str, Dict]) -> List[Degeneracy]:
        """
        检查约束标签的多样性是否过低。

        Check for low constraint label diversity.
        """
        degeneracies = []
        obj_ids = list(objects.keys())
        pairs = list(combinations(obj_ids, 2))

        if len(pairs) < 4:
            return degeneracies

        # Simulate constraint labels
        from ordinal_spatial.dsl.comparators import compare

        labels = []
        for i, pair1 in enumerate(pairs):
            for pair2 in pairs[i + 1:]:
                if set(pair1) & set(pair2):
                    continue

                d1 = compute_dist_3d(objects[pair1[0]], objects[pair1[1]])
                d2 = compute_dist_3d(objects[pair2[0]], objects[pair2[1]])
                label = compare(d1, d2, self.tau)
                labels.append(label)

        if not labels:
            return degeneracies

        # Calculate entropy
        counts = Counter(labels)
        total = len(labels)
        probs = [c / total for c in counts.values()]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(3)  # 3 possible labels
        normalized_entropy = entropy / max_entropy

        if normalized_entropy < self.diversity_threshold:
            severity = (self.diversity_threshold - normalized_entropy) / self.diversity_threshold
            dominant = counts.most_common(1)[0]
            degeneracies.append(Degeneracy(
                type=DegeneracyType.LOW_DIVERSITY,
                severity=severity * 0.5,
                objects=list(obj_ids),
                description=f"Low label diversity (entropy={normalized_entropy:.2f}), "
                           f"dominant label: {dominant[0]} ({dominant[1]}/{total})",
                data={
                    "entropy": normalized_entropy,
                    "label_counts": {str(k): v for k, v in counts.items()},
                },
            ))

        return degeneracies


# =============================================================================
# Standalone Functions
# =============================================================================

def check_scene_degeneracy(
    objects: Dict[str, Dict],
    tau: float = 0.10,
) -> DegeneracyReport:
    """
    Check a scene for degeneracies.

    Args:
        objects: Dictionary of object_id -> object_data
        tau: Tolerance parameter

    Returns:
        DegeneracyReport with findings
    """
    checker = DegeneracyChecker(tau=tau)
    return checker.check(objects)


def is_scene_valid(
    objects: Dict[str, Dict],
    tau: float = 0.10,
    max_severity: float = 0.5,
) -> bool:
    """
    Quick check if a scene is valid.

    Args:
        objects: Dictionary of object_id -> object_data
        tau: Tolerance parameter
        max_severity: Maximum allowed severity

    Returns:
        True if scene is valid, False otherwise
    """
    checker = DegeneracyChecker(tau=tau, max_severity=max_severity)
    report = checker.check(objects)
    return report.is_valid
