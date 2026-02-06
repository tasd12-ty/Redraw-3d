"""
约束提取智能体的基类定义。

Base class definitions for constraint extraction agents.

This module defines:
- ConstraintSet: Container for extracted constraints
- ConstraintAgent: Abstract base class for agents
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from ordinal_spatial.dsl.schema import (
    QRRConstraintSchema,
    TRRConstraintSchema,
    TopologyConstraint,
    OcclusionConstraint,
    AxialConstraint,
    SizeConstraint,
    CloserConstraint,
)


@dataclass
class ObjectInfo:
    """
    从图像中识别的物体信息。

    Object information identified from image.
    """
    id: str
    type: str  # cube, sphere, cylinder, etc.
    color: str
    size_class: str = "medium"  # tiny, small, medium, large
    position_2d: Optional[List[float]] = None  # [x, y] in image
    description: Optional[str] = None  # Natural language description


@dataclass
class ConstraintSet:
    """
    从图像中提取的约束集合 (QSP - Qualitative Scene Program)。

    Container for extracted constraints from images.
    Represents a Qualitative Scene Program (QSP).

    Attributes:
        objects: List of identified objects
        qrr: Quaternary relative relations (distance comparisons)
        trr: Ternary clock relations (directional)
        topology: Topological relations (disjoint, touching, overlapping)
        occlusion: Occlusion relations (view-dependent)
        axial: Axial ordering (left_of, right_of, above, below, etc.)
        size: Size comparisons (bigger/smaller)
        closer: Ternary distance comparisons
        confidence: Overall extraction confidence
        metadata: Additional information
    """
    objects: List[ObjectInfo] = field(default_factory=list)
    qrr: List[QRRConstraintSchema] = field(default_factory=list)
    trr: List[TRRConstraintSchema] = field(default_factory=list)
    topology: List[TopologyConstraint] = field(default_factory=list)
    occlusion: List[OcclusionConstraint] = field(default_factory=list)
    axial: List[AxialConstraint] = field(default_factory=list)
    size: List[SizeConstraint] = field(default_factory=list)
    closer: List[CloserConstraint] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式。

        Convert to dictionary format.
        """
        return {
            "objects": [
                {
                    "id": obj.id,
                    "type": obj.type,
                    "color": obj.color,
                    "size_class": obj.size_class,
                    "position_2d": obj.position_2d,
                    "description": obj.description,
                }
                for obj in self.objects
            ],
            "constraints": {
                "qrr": [c.model_dump() for c in self.qrr],
                "trr": [c.model_dump() for c in self.trr],
                "topology": [c.model_dump() for c in self.topology],
                "occlusion": [c.model_dump() for c in self.occlusion],
                "axial": [c.model_dump() for c in self.axial],
                "size": [c.model_dump() for c in self.size],
                "closer": [c.model_dump() for c in self.closer],
            },
            "confidence": self.confidence,
            "counts": self.count_by_arity(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConstraintSet":
        """
        从字典创建 ConstraintSet。

        Create ConstraintSet from dictionary.
        """
        objects = [
            ObjectInfo(
                id=obj["id"],
                type=obj.get("type", "unknown"),
                color=obj.get("color", "unknown"),
                size_class=obj.get("size_class", "medium"),
                position_2d=obj.get("position_2d"),
                description=obj.get("description"),
            )
            for obj in data.get("objects", [])
        ]

        constraints = data.get("constraints", {})

        return cls(
            objects=objects,
            qrr=[QRRConstraintSchema(**c) for c in constraints.get("qrr", [])],
            trr=[TRRConstraintSchema(**c) for c in constraints.get("trr", [])],
            topology=[TopologyConstraint(**c) for c in constraints.get("topology", [])],
            occlusion=[OcclusionConstraint(**c) for c in constraints.get("occlusion", [])],
            axial=[AxialConstraint(**c) for c in constraints.get("axial", [])],
            size=[SizeConstraint(**c) for c in constraints.get("size", [])],
            closer=[CloserConstraint(**c) for c in constraints.get("closer", [])],
            confidence=data.get("confidence", 0.0),
            metadata=data.get("metadata", {}),
        )

    def summary(self) -> str:
        """
        返回约束集合的摘要，按元数分组。

        Return a summary of the constraint set, grouped by arity.
        """
        counts = self.count_by_arity()
        n = counts["n_objects"]
        b = counts["binary"]
        t = counts["ternary"]
        q = counts["quaternary"]

        lines = [
            f"Objects: {n}",
            f"",
            f"Binary (pairs, expected C({n},2)={b['expected_pairs']}):",
            f"  Axial:    {b['axial']}",
            f"  Topology: {b['topology']}",
            f"  Size:     {b['size']}",
            f"  Occlusion:{b['occlusion']}",
            f"  Subtotal: {b['total']}",
            f"",
            f"Ternary (triples, expected C({n},3)={t['expected_triples']}):",
            f"  Closer:   {t['closer']}",
            f"  TRR:      {t['trr']}",
            f"  Subtotal: {t['total']}",
            f"",
            f"Quaternary (pure QRR, expected 3*C({n},4)={q['expected_pure_qrr']}):",
            f"  QRR:      {q['qrr']}",
            f"  Subtotal: {q['total']}",
            f"",
            f"Grand total: {counts['grand_total']}",
            f"Confidence: {self.confidence:.2f}",
        ]
        return "\n".join(lines)

    def total_constraints(self) -> int:
        """Return total number of constraints."""
        return (
            len(self.qrr) + len(self.trr) + len(self.topology) +
            len(self.occlusion) + len(self.axial) + len(self.size) +
            len(self.closer)
        )

    def count_by_arity(self) -> Dict[str, Dict[str, Any]]:
        """
        按关系元数分组计数，并给出理论期望值。

        Count constraints grouped by arity (binary/ternary/quaternary),
        with expected values based on combinatorial formulas.

        Returns:
            Dict with arity groups, each containing actual counts
            and expected counts based on number of objects.
        """
        n = len(self.objects)
        pairs = n * (n - 1) // 2  # C(n,2)
        triples = n * (n - 1) * (n - 2) // 6  # C(n,3)
        pure_qrr = n * (n - 1) * (n - 2) * (n - 3) // 8 if n >= 4 else 0  # 3*C(n,4)

        binary_total = len(self.axial) + len(self.topology) + len(self.size) + len(self.occlusion)
        ternary_total = len(self.closer) + len(self.trr)
        quaternary_total = len(self.qrr)

        return {
            "n_objects": n,
            "binary": {
                "axial": len(self.axial),
                "topology": len(self.topology),
                "size": len(self.size),
                "occlusion": len(self.occlusion),
                "total": binary_total,
                "expected_pairs": pairs,
            },
            "ternary": {
                "closer": len(self.closer),
                "trr": len(self.trr),
                "total": ternary_total,
                "expected_triples": triples,
            },
            "quaternary": {
                "qrr": len(self.qrr),
                "total": quaternary_total,
                "expected_pure_qrr": pure_qrr,
            },
            "grand_total": binary_total + ternary_total + quaternary_total,
        }


class ConstraintAgent(ABC):
    """
    约束提取智能体的抽象基类。

    Abstract base class for constraint extraction agents.

    Agents can extract spatial constraints from:
    - Single images (Task-3)
    - Multiple images (Task-2)
    - Blender scene data (Task-1)
    """

    @abstractmethod
    def extract_from_single_view(
        self,
        image: Union[str, Path, bytes],
        objects: Optional[List[Dict[str, Any]]] = None,
        tau: float = 0.10,
    ) -> ConstraintSet:
        """
        从单视角图像提取约束 (Task-3)。

        Extract constraints from a single-view image (Task-3).

        Args:
            image: Image path, bytes, or base64 string
            objects: Optional list of known objects in the scene
            tau: Tolerance parameter for comparisons

        Returns:
            ConstraintSet containing extracted constraints
        """
        pass

    @abstractmethod
    def extract_from_multi_view(
        self,
        images: List[Union[str, Path, bytes]],
        objects: Optional[List[Dict[str, Any]]] = None,
        tau: float = 0.10,
    ) -> ConstraintSet:
        """
        从多视角图像提取约束 (Task-2)。

        Extract constraints from multiple views (Task-2).

        Args:
            images: List of image paths, bytes, or base64 strings
            objects: Optional list of known objects in the scene
            tau: Tolerance parameter for comparisons

        Returns:
            ConstraintSet with view-invariant and view-dependent constraints
        """
        pass

    def extract(
        self,
        images: Union[str, Path, bytes, List[Union[str, Path, bytes]]],
        objects: Optional[List[Dict[str, Any]]] = None,
        tau: float = 0.10,
    ) -> ConstraintSet:
        """
        统一接口：根据输入自动选择单视角或多视角提取。

        Unified interface: automatically select single or multi-view extraction.

        Args:
            images: Single image or list of images
            objects: Optional list of known objects
            tau: Tolerance parameter

        Returns:
            ConstraintSet with extracted constraints
        """
        if isinstance(images, list):
            if len(images) == 1:
                return self.extract_from_single_view(images[0], objects, tau)
            return self.extract_from_multi_view(images, objects, tau)
        return self.extract_from_single_view(images, objects, tau)
