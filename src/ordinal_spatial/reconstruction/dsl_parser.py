"""
DSL 约束解析器。

解析 JSON 格式的约束 DSL，转化为内部数据结构。

DSL Constraint Parser.

Parses JSON format constraint DSL into internal data structures.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class AxialRelation(str, Enum):
    """
    轴向关系枚举。

    Axial relation enumeration.
    """
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"

    @classmethod
    def from_string(cls, s: str) -> "AxialRelation":
        """从字符串解析轴向关系。"""
        mapping = {
            "left_of": cls.LEFT_OF,
            "right_of": cls.RIGHT_OF,
            "above": cls.ABOVE,
            "below": cls.BELOW,
            "in_front_of": cls.IN_FRONT_OF,
            "behind": cls.BEHIND,
            "left": cls.LEFT_OF,
            "right": cls.RIGHT_OF,
            "front": cls.IN_FRONT_OF,
            "back": cls.BEHIND,
        }
        return mapping.get(s.lower(), cls.LEFT_OF)


class TopologyRelation(str, Enum):
    """
    拓扑关系枚举 (RCC-8 子集)。

    Topology relation enumeration (RCC-8 subset).
    """
    DISJOINT = "disjoint"       # DC - 分离
    TOUCHING = "touching"       # EC - 外部接触
    OVERLAPPING = "overlapping" # PO - 部分重叠


class MetricType(str, Enum):
    """
    度量类型枚举。

    Metric type enumeration.
    """
    DIST_3D = "DIST_3D"
    DIST_2D = "DIST_2D"
    DEPTH_GAP = "DEPTH_GAP"
    SIZE_RATIO = "SIZE_RATIO"


class Comparator(str, Enum):
    """
    比较器枚举。

    Comparator enumeration.
    """
    LT = "<"
    APPROX = "~="
    GT = ">"

    @classmethod
    def from_string(cls, s: str) -> "Comparator":
        """从字符串解析比较器。"""
        mapping = {
            "<": cls.LT,
            "lt": cls.LT,
            "less": cls.LT,
            "~=": cls.APPROX,
            "≈": cls.APPROX,
            "approx": cls.APPROX,
            "eq": cls.APPROX,
            ">": cls.GT,
            "gt": cls.GT,
            "greater": cls.GT,
        }
        return mapping.get(s.lower() if isinstance(s, str) else s, cls.APPROX)


@dataclass
class ObjectInfo:
    """
    物体信息。

    Object information.
    """
    id: str
    shape: str = "cube"
    color: str = "gray"
    size: str = "medium"
    material: str = "rubber"
    position_3d: Optional[List[float]] = None


@dataclass
class QRRConstraint:
    """
    QRR (四元相对关系) 约束。

    QRR (Quaternary Relative Relation) constraint.
    """
    pair1: Tuple[str, str]  # (A, B)
    pair2: Tuple[str, str]  # (C, D)
    metric: MetricType
    comparator: Comparator
    ratio: float = 1.0
    difficulty: int = 3
    boundary_flag: bool = False


@dataclass
class TRRConstraint:
    """
    TRR (三元时钟关系) 约束。

    TRR (Ternary clock Relation) constraint.
    """
    target: str
    ref1: str
    ref2: str
    hour: int  # 1-12
    quadrant: int = 1  # 1-4
    angle_deg: float = 0.0


@dataclass
class AxialConstraint:
    """
    轴向偏序约束。

    Axial ordering constraint.
    """
    obj_a: str
    obj_b: str
    axis: str  # "x", "y", "z"
    relation: AxialRelation

    @classmethod
    def from_dict(cls, d: Dict) -> "AxialConstraint":
        """从字典创建轴向约束。"""
        # 支持多种字段名格式
        obj_a = d.get("a") or d.get("obj_a") or d.get("object_a") or d.get("obj1")
        obj_b = d.get("b") or d.get("obj_b") or d.get("object_b") or d.get("obj2")
        axis = d.get("axis", "x")
        relation_str = d.get("relation", "left_of")

        return cls(
            obj_a=obj_a,
            obj_b=obj_b,
            axis=axis,
            relation=AxialRelation.from_string(relation_str),
        )


@dataclass
class TopologyConstraint:
    """
    拓扑关系约束。

    Topology relation constraint.
    """
    obj1: str
    obj2: str
    relation: TopologyRelation

    @classmethod
    def from_dict(cls, d: Dict) -> "TopologyConstraint":
        """从字典创建拓扑约束。"""
        obj1 = d.get("obj1") or d.get("object1") or d.get("a")
        obj2 = d.get("obj2") or d.get("object2") or d.get("b")
        relation_str = d.get("relation", "disjoint")

        return cls(
            obj1=obj1,
            obj2=obj2,
            relation=TopologyRelation(relation_str),
        )


@dataclass
class OcclusionConstraint:
    """
    遮挡关系约束（视角相关）。

    Occlusion relation constraint (view-dependent).
    """
    occluder: str
    occluded: str
    partial: bool = False


@dataclass
class ParsedConstraints:
    """
    解析后的约束集合。

    Parsed constraint collection.
    """
    scene_id: str = "unknown"
    objects: List[ObjectInfo] = field(default_factory=list)
    qrr: List[QRRConstraint] = field(default_factory=list)
    trr: List[TRRConstraint] = field(default_factory=list)
    axial: List[AxialConstraint] = field(default_factory=list)
    topology: List[TopologyConstraint] = field(default_factory=list)
    occlusion: List[OcclusionConstraint] = field(default_factory=list)
    tau: float = 0.10

    def get_object_ids(self) -> List[str]:
        """获取所有物体 ID。"""
        return [obj.id for obj in self.objects]

    def get_object_by_id(self, obj_id: str) -> Optional[ObjectInfo]:
        """通过 ID 获取物体。"""
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        return None

    def n_objects(self) -> int:
        """物体数量。"""
        return len(self.objects)

    def n_constraints(self) -> int:
        """约束总数。"""
        return (
            len(self.qrr)
            + len(self.trr)
            + len(self.axial)
            + len(self.topology)
            + len(self.occlusion)
        )


@dataclass
class ValidationResult:
    """
    验证结果。

    Validation result.
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class DSLParser:
    """
    DSL 约束解析器。

    DSL Constraint Parser.
    """

    def __init__(self, strict: bool = False):
        """
        初始化解析器。

        Args:
            strict: 是否使用严格模式（遇到错误时抛出异常）
        """
        self.strict = strict

    def parse_file(self, path: str) -> ParsedConstraints:
        """
        从文件解析约束。

        Parse constraints from file.

        Args:
            path: JSON 文件路径

        Returns:
            ParsedConstraints 对象
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.parse_json(data)

    def parse_json(self, data: Dict[str, Any]) -> ParsedConstraints:
        """
        从 JSON 数据解析约束。

        Parse constraints from JSON data.

        Args:
            data: JSON 数据字典

        Returns:
            ParsedConstraints 对象
        """
        result = ParsedConstraints()

        # 解析场景 ID
        result.scene_id = data.get("scene_id", "unknown")

        # 解析 tau
        result.tau = data.get("tau", 0.10)

        # 解析物体
        result.objects = self._parse_objects(data.get("objects", []))

        # 解析约束
        constraints = data.get("constraints", data)

        result.qrr = self._parse_qrr(constraints.get("qrr", []))
        result.trr = self._parse_trr(constraints.get("trr", []))
        result.axial = self._parse_axial(constraints.get("axial", []))
        result.topology = self._parse_topology(constraints.get("topology", []))
        result.occlusion = self._parse_occlusion(constraints.get("occlusion", []))

        # 验证
        if self.strict:
            validation = self.validate(result)
            if not validation.is_valid:
                raise ValueError(f"Validation failed: {validation.errors}")

        return result

    def _parse_objects(self, objects_data: List[Dict]) -> List[ObjectInfo]:
        """解析物体列表。"""
        objects = []
        for obj_data in objects_data:
            obj = ObjectInfo(
                id=obj_data.get("id") or obj_data.get("name") or f"obj_{len(objects)}",
                shape=obj_data.get("shape", "cube"),
                color=obj_data.get("color", "gray"),
                size=obj_data.get("size", "medium"),
                material=obj_data.get("material", "rubber"),
                position_3d=obj_data.get("position_3d") or obj_data.get("3d_coords"),
            )
            objects.append(obj)
        return objects

    def _parse_qrr(self, qrr_data: List[Dict]) -> List[QRRConstraint]:
        """解析 QRR 约束。"""
        constraints = []
        for c in qrr_data:
            try:
                pair1 = tuple(c.get("pair1", []))
                pair2 = tuple(c.get("pair2", []))

                if len(pair1) != 2 or len(pair2) != 2:
                    logger.warning(f"Invalid QRR pair: {c}")
                    continue

                metric_str = c.get("metric", "DIST_3D")
                try:
                    metric = MetricType(metric_str)
                except ValueError:
                    metric = MetricType.DIST_3D

                constraint = QRRConstraint(
                    pair1=pair1,
                    pair2=pair2,
                    metric=metric,
                    comparator=Comparator.from_string(c.get("comparator", "~=")),
                    ratio=c.get("ratio", 1.0),
                    difficulty=c.get("difficulty", 3),
                    boundary_flag=c.get("boundary_flag", False),
                )
                constraints.append(constraint)
            except Exception as e:
                logger.warning(f"Failed to parse QRR constraint: {c}, error: {e}")
        return constraints

    def _parse_trr(self, trr_data: List[Dict]) -> List[TRRConstraint]:
        """解析 TRR 约束。"""
        constraints = []
        for c in trr_data:
            try:
                constraint = TRRConstraint(
                    target=c.get("target"),
                    ref1=c.get("ref1"),
                    ref2=c.get("ref2"),
                    hour=c.get("hour", 12),
                    quadrant=c.get("quadrant", 1),
                    angle_deg=c.get("angle_deg", 0.0),
                )
                constraints.append(constraint)
            except Exception as e:
                logger.warning(f"Failed to parse TRR constraint: {c}, error: {e}")
        return constraints

    def _parse_axial(self, axial_data: List[Dict]) -> List[AxialConstraint]:
        """解析轴向约束。"""
        constraints = []
        for c in axial_data:
            try:
                constraint = AxialConstraint.from_dict(c)
                constraints.append(constraint)
            except Exception as e:
                logger.warning(f"Failed to parse axial constraint: {c}, error: {e}")
        return constraints

    def _parse_topology(self, topology_data: List[Dict]) -> List[TopologyConstraint]:
        """解析拓扑约束。"""
        constraints = []
        for c in topology_data:
            try:
                constraint = TopologyConstraint.from_dict(c)
                constraints.append(constraint)
            except Exception as e:
                logger.warning(f"Failed to parse topology constraint: {c}, error: {e}")
        return constraints

    def _parse_occlusion(self, occlusion_data: List[Dict]) -> List[OcclusionConstraint]:
        """解析遮挡约束。"""
        constraints = []
        for c in occlusion_data:
            try:
                constraint = OcclusionConstraint(
                    occluder=c.get("occluder"),
                    occluded=c.get("occluded"),
                    partial=c.get("partial", False),
                )
                constraints.append(constraint)
            except Exception as e:
                logger.warning(f"Failed to parse occlusion constraint: {c}, error: {e}")
        return constraints

    def validate(self, constraints: ParsedConstraints) -> ValidationResult:
        """
        验证约束的有效性。

        Validate constraint validity.

        Args:
            constraints: 解析后的约束

        Returns:
            ValidationResult 对象
        """
        errors = []
        warnings = []

        # 检查物体数量
        if constraints.n_objects() == 0:
            errors.append("No objects defined")

        # 获取所有物体 ID
        object_ids = set(constraints.get_object_ids())

        # 检查 QRR 约束引用的物体是否存在
        for qrr in constraints.qrr:
            for obj_id in qrr.pair1 + qrr.pair2:
                if obj_id not in object_ids:
                    warnings.append(f"QRR references unknown object: {obj_id}")

        # 检查 TRR 约束引用的物体是否存在
        for trr in constraints.trr:
            for obj_id in [trr.target, trr.ref1, trr.ref2]:
                if obj_id and obj_id not in object_ids:
                    warnings.append(f"TRR references unknown object: {obj_id}")

        # 检查轴向约束引用的物体是否存在
        for axial in constraints.axial:
            if axial.obj_a not in object_ids:
                warnings.append(f"Axial constraint references unknown object: {axial.obj_a}")
            if axial.obj_b not in object_ids:
                warnings.append(f"Axial constraint references unknown object: {axial.obj_b}")

        # 检查拓扑约束引用的物体是否存在
        for topo in constraints.topology:
            if topo.obj1 not in object_ids:
                warnings.append(f"Topology constraint references unknown object: {topo.obj1}")
            if topo.obj2 not in object_ids:
                warnings.append(f"Topology constraint references unknown object: {topo.obj2}")

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
        )

    def infer_objects_from_constraints(self, data: Dict[str, Any]) -> List[ObjectInfo]:
        """
        从约束中推断物体列表（当没有显式定义物体时）。

        Infer object list from constraints when not explicitly defined.
        """
        object_ids = set()
        constraints = data.get("constraints", data)

        # 从 QRR 收集
        for qrr in constraints.get("qrr", []):
            object_ids.update(qrr.get("pair1", []))
            object_ids.update(qrr.get("pair2", []))

        # 从 TRR 收集
        for trr in constraints.get("trr", []):
            if trr.get("target"):
                object_ids.add(trr["target"])
            if trr.get("ref1"):
                object_ids.add(trr["ref1"])
            if trr.get("ref2"):
                object_ids.add(trr["ref2"])

        # 从轴向约束收集
        for axial in constraints.get("axial", []):
            obj_a = axial.get("a") or axial.get("obj_a")
            obj_b = axial.get("b") or axial.get("obj_b")
            if obj_a:
                object_ids.add(obj_a)
            if obj_b:
                object_ids.add(obj_b)

        # 从拓扑约束收集
        for topo in constraints.get("topology", []):
            obj1 = topo.get("obj1") or topo.get("a")
            obj2 = topo.get("obj2") or topo.get("b")
            if obj1:
                object_ids.add(obj1)
            if obj2:
                object_ids.add(obj2)

        # 创建默认物体
        return [
            ObjectInfo(id=obj_id, shape="cube", color="gray", size="medium")
            for obj_id in sorted(object_ids)
        ]
