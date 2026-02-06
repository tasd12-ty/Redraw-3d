"""
ORDINAL-SPATIAL 数据结构的 Pydantic 模式定义。

本模块定义了以下数据模型：
- 物体规范（ObjectSpec）：形状、颜色、尺寸、位置等
- QRR/TRR 约束：查询和预测结果
- 序场景描述（OSD - Ordinal Scene Description）：完整场景信息
- 多视角支持（WORLD/VIEW/CONSENSUS）：世界/视角/共识约束

使用 Pydantic 进行类型验证和序列化，确保数据一致性。

Pydantic schemas for ORDINAL-SPATIAL data structures.

This module defines the data models for:
- Object specifications
- QRR/TRR constraints
- Ordinal Scene Descriptions (OSD)
- Multi-view support (WORLD/VIEW/CONSENSUS)
"""

from typing import List, Dict, Optional, Any, Tuple, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import json


class ShapeType(str, Enum):
    """
    支持的 3D 基本形状。

    Supported 3D primitive shapes.
    """
    CUBE = "cube"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    CONE = "cone"
    PYRAMID = "pyramid"
    CUBOID = "cuboid"
    TRIANGULAR_PRISM = "triangular_prism"
    HEXAGONAL_PRISM = "hexagonal_prism"
    TETRAHEDRON = "tetrahedron"
    OCTAHEDRON = "octahedron"
    TORUS = "torus"
    ELLIPSOID = "ellipsoid"


class SizeClass(str, Enum):
    """
    物体的离散尺寸类别。

    Discrete size classes for objects.
    """
    LARGE = "large"
    MEDIUM = "medium"
    SMALL = "small"


class MaterialType(str, Enum):
    """
    影响外观的材质类型。

    Material types affecting appearance.
    """
    RUBBER = "rubber"
    METAL = "metal"


# =============================================================================
# Object Specification
# =============================================================================

class ObjectSpec(BaseModel):
    """
    场景中单个物体的规范。

    属性:
        id: 物体的唯一标识符
        shape: 3D 基本形状类型
        color: 物体颜色（名称或 RGB）
        size: 尺寸类别或数值比例
        material: 表面材质类型
        position_3d: 3D 世界坐标 [x, y, z]
        position_2d: 2D 图像坐标 [u, v]
        depth: 相对相机的深度
        rotation: 旋转角度（度数）

    Specification for a single object in the scene.

    Attributes:
        id: Unique identifier for the object
        shape: Type of 3D primitive
        color: Object color (name or RGB)
        size: Size class or numeric scale
        material: Surface material type
        position_3d: 3D world coordinates [x, y, z]
        position_2d: 2D image coordinates [u, v]
        depth: Camera-relative depth
        rotation: Rotation angle in degrees
    """
    id: str
    shape: ShapeType
    color: str
    size: Union[SizeClass, float] = SizeClass.MEDIUM
    material: MaterialType = MaterialType.RUBBER
    position_3d: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    position_2d: List[float] = Field(default_factory=lambda: [0.0, 0.0])
    depth: float = 0.0
    rotation: float = 0.0
    bounding_box: Optional[Dict[str, List[float]]] = None
    visible_pixels: Optional[int] = None

    @field_validator('position_3d')
    @classmethod
    def validate_position_3d(cls, v):
        """
        验证 3D 位置必须有 3 个元素。

        Validate that 3D position has exactly 3 elements.
        """
        if len(v) != 3:
            raise ValueError(f"position_3d must have 3 elements, got {len(v)}")
        return v

    @field_validator('position_2d')
    @classmethod
    def validate_position_2d(cls, v):
        """
        验证 2D 位置至少有 2 个元素。

        Validate that 2D position has at least 2 elements.
        """
        if len(v) < 2:
            raise ValueError(f"position_2d must have at least 2 elements, got {len(v)}")
        return v[:2]

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典用于约束计算。

        Convert to dictionary for constraint computation.
        """
        return {
            "id": self.id,
            "shape": self.shape.value,
            "color": self.color,
            "size": self.size.value if isinstance(self.size, SizeClass) else self.size,
            "material": self.material.value,
            "position_3d": self.position_3d,
            "position_2d": self.position_2d,
            "depth": self.depth,
            "rotation": self.rotation,
        }


# =============================================================================
# Constraint Schemas
# =============================================================================

class QRRConstraintSchema(BaseModel):
    """
    QRR 约束序列化的模式定义。

    Schema for QRR constraint serialization.
    """
    pair1: List[str]
    pair2: List[str]
    metric: str
    comparator: str
    ratio: float = 1.0
    difficulty: int = 3
    boundary_flag: bool = False

    @field_validator('pair1', 'pair2')
    @classmethod
    def validate_pair(cls, v):
        """
        验证对必须有恰好 2 个元素。

        Validate that pair has exactly 2 elements.
        """
        if len(v) != 2:
            raise ValueError(f"Pair must have exactly 2 elements, got {len(v)}")
        return sorted(v)

    @field_validator('comparator')
    @classmethod
    def validate_comparator(cls, v):
        """
        验证比较器是否有效。

        Validate that comparator is valid.
        """
        valid = {"<", "~=", ">", "≈", "=", "lt", "eq", "gt", "approx"}
        if v.lower() not in valid and v not in valid:
            raise ValueError(f"Invalid comparator: {v}")
        return v


class TRRConstraintSchema(BaseModel):
    """
    TRR 约束序列化的模式定义。

    Schema for TRR constraint serialization.
    """
    target: str
    ref1: str
    ref2: str
    hour: int = Field(ge=1, le=12)
    quadrant: int = Field(ge=1, le=4, default=1)
    angle_deg: float = 0.0


class TopologyConstraint(BaseModel):
    """
    Topological relationship between objects.

    Based on RCC-8 subset:
    - DISJOINT: Objects don't touch (DC)
    - TOUCHING: Objects share boundary (EC)
    - OVERLAPPING: Objects partially overlap (PO)
    """
    obj1: str
    obj2: str
    relation: str = Field(pattern="^(disjoint|touching|overlapping)$")


class OcclusionConstraint(BaseModel):
    """
    Occlusion relationship (view-dependent).

    Represents that obj1 occludes (is in front of) obj2 from
    the current camera viewpoint.
    """
    occluder: str
    occluded: str
    partial: bool = False  # True if partial occlusion


# =============================================================================
# Extended Constraint Types (PDF Section 4)
# =============================================================================

class AxialRelation(str, Enum):
    """
    二元轴向偏序关系。

    Binary axial ordering relations for spatial positioning.
    Based on cardinal/depth directions.
    """
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"


class AxialConstraint(BaseModel):
    """
    轴向偏序约束。

    Axial ordering constraint between two objects.
    Example: obj1 is left_of obj2
    """
    obj1: str
    obj2: str
    relation: AxialRelation

    def inverse(self) -> "AxialConstraint":
        """Return the inverse constraint."""
        inverse_map = {
            AxialRelation.LEFT_OF: AxialRelation.RIGHT_OF,
            AxialRelation.RIGHT_OF: AxialRelation.LEFT_OF,
            AxialRelation.ABOVE: AxialRelation.BELOW,
            AxialRelation.BELOW: AxialRelation.ABOVE,
            AxialRelation.IN_FRONT_OF: AxialRelation.BEHIND,
            AxialRelation.BEHIND: AxialRelation.IN_FRONT_OF,
        }
        return AxialConstraint(
            obj1=self.obj2,
            obj2=self.obj1,
            relation=inverse_map[self.relation]
        )


class SizeConstraint(BaseModel):
    """
    二元大小比较约束。

    Binary size comparison constraint.
    Represents: bigger_obj is visually larger than smaller_obj.
    """
    bigger: str
    smaller: str

    def inverse(self) -> "SizeConstraint":
        """Return the inverse (swapped) constraint."""
        return SizeConstraint(bigger=self.smaller, smaller=self.bigger)


class CloserConstraint(BaseModel):
    """
    三元距离比较约束 (closer triplet)。

    Ternary distance comparison constraint.
    Represents: d(anchor, closer_obj) < d(anchor, farther_obj)

    This is different from QRR (quaternary) which compares two pairs.
    """
    anchor: str      # Reference point A
    closer: str      # Object B that is closer to A
    farther: str     # Object C that is farther from A

    def to_qrr_style(self) -> Dict[str, Any]:
        """
        Convert to QRR-style representation for compatibility.

        closer(A, B, C) means d(A,B) < d(A,C)
        """
        return {
            "pair1": [self.anchor, self.closer],
            "pair2": [self.anchor, self.farther],
            "metric": "dist3D",
            "comparator": "<",
        }


# =============================================================================
# Camera Specification
# =============================================================================

class CameraParams(BaseModel):
    """
    Camera parameters for a view.

    Attributes:
        camera_id: Unique identifier for this view
        position: Camera position in world coordinates
        look_at: Point the camera is looking at
        up: Up vector
        fov: Field of view in degrees
        resolution: Image resolution [width, height]
    """
    camera_id: str
    position: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    look_at: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    up: List[float] = Field(default_factory=lambda: [0.0, 0.0, 1.0])
    fov: float = 50.0
    resolution: List[int] = Field(default_factory=lambda: [1024, 1024])


# =============================================================================
# Multi-View Structures
# =============================================================================

class WorldConstraints(BaseModel):
    """
    View-invariant 3D constraints.

    These constraints hold regardless of camera position:
    - 3D distance comparisons (QRR)
    - Topological relationships (touching, disjoint)
    - Physical size ordering
    - 3D axial relations
    - Ternary closer relations
    """
    qrr: List[QRRConstraintSchema] = Field(default_factory=list)
    topology: List[TopologyConstraint] = Field(default_factory=list)
    size: List[SizeConstraint] = Field(default_factory=list)
    closer: List[CloserConstraint] = Field(default_factory=list)
    axial: List[AxialConstraint] = Field(default_factory=list)


class ViewConstraints(BaseModel):
    """
    Per-camera 2D observations.

    These constraints depend on the specific viewpoint:
    - 2D image-plane distances
    - Clock positions in image
    - Occlusion relationships
    - 2D axial relations (in image plane)
    """
    camera: CameraParams
    qrr_2d: List[QRRConstraintSchema] = Field(default_factory=list)
    trr: List[TRRConstraintSchema] = Field(default_factory=list)
    occlusion: List[OcclusionConstraint] = Field(default_factory=list)
    axial_2d: List[AxialConstraint] = Field(default_factory=list)
    size_apparent: List[SizeConstraint] = Field(default_factory=list)
    image_path: Optional[str] = None
    depth_path: Optional[str] = None


class ConsensusConstraints(BaseModel):
    """
    Multi-view derived constraints.

    Aggregates information from multiple views:
    - Verified constraints (consistent across views)
    - Conflicts (expected view-dependent differences)
    """
    verified: List[QRRConstraintSchema] = Field(default_factory=list)
    view_dependent: List[str] = Field(default_factory=list)  # Constraint IDs
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)


# =============================================================================
# Main OSD Schema
# =============================================================================

class OrdinalSceneDescription(BaseModel):
    """
    Complete Ordinal Scene Description (OSD).

    This is the main data structure for the benchmark, containing:
    - Scene metadata
    - Object specifications
    - View-invariant world constraints
    - Per-view constraints
    - Multi-view consensus

    Attributes:
        scene_id: Unique scene identifier
        objects: List of objects in the scene
        tau: Tolerance parameter used for comparisons
        world: View-invariant 3D constraints
        views: Per-camera observations
        consensus: Multi-view derived constraints
        metadata: Additional scene metadata
    """
    scene_id: str
    objects: List[ObjectSpec]
    tau: float = Field(default=0.10, ge=0.0, le=1.0)
    world: WorldConstraints = Field(default_factory=WorldConstraints)
    views: List[ViewConstraints] = Field(default_factory=list)
    consensus: Optional[ConsensusConstraints] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def get_object(self, obj_id: str) -> Optional[ObjectSpec]:
        """
        通过 ID 获取物体。

        Get object by ID.
        """
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        return None

    def get_objects_dict(self) -> Dict[str, Dict]:
        """
        获取物体字典用于约束计算。

        Get objects as dictionary for constraint computation.
        """
        return {obj.id: obj.to_dict() for obj in self.objects}

    def object_ids(self) -> List[str]:
        """
        获取物体 ID 列表。

        Get list of object IDs.
        """
        return [obj.id for obj in self.objects]

    def to_json(self, indent: int = 2) -> str:
        """
        序列化为 JSON 字符串。

        Serialize to JSON string.
        """
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "OrdinalSceneDescription":
        """
        从 JSON 字符串反序列化。

        Deserialize from JSON string.
        """
        data = json.loads(json_str)
        return cls.model_validate(data)

    def save(self, path: str):
        """
        保存到 JSON 文件。

        Save to JSON file.
        """
        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str) -> "OrdinalSceneDescription":
        """
        从 JSON 文件加载。

        Load from JSON file.
        """
        with open(path, 'r') as f:
            return cls.from_json(f.read())


# =============================================================================
# Query Schemas
# =============================================================================

class QRRQuery(BaseModel):
    """
    Query for T1-Q task (QRR classification).

    Attributes:
        scene_id: Scene to query
        query_id: Unique query identifier
        objects: Object identifiers {A, B, C, D}
        metric: Type of metric to compare
        tau: Tolerance parameter
        ground_truth: Optional ground truth for evaluation
    """
    scene_id: str
    query_id: str
    objects: Dict[str, str]  # {"A": "obj_1", "B": "obj_2", ...}
    metric: str
    tau: float = 0.10
    ground_truth: Optional[QRRConstraintSchema] = None


class TRRQuery(BaseModel):
    """
    Query for T1-C task (TRR classification).

    Attributes:
        scene_id: Scene to query
        query_id: Unique query identifier
        target: Target object ID
        ref1: Origin object ID
        ref2: Direction reference object ID
        ground_truth: Optional ground truth for evaluation
    """
    scene_id: str
    query_id: str
    target: str
    ref1: str
    ref2: str
    ground_truth: Optional[TRRConstraintSchema] = None


# =============================================================================
# Prediction Schemas
# =============================================================================

class QRRPrediction(BaseModel):
    """
    QRR 查询的预测结果。

    Prediction for QRR query.
    """
    query_id: str
    comparator: str
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    reasoning: Optional[str] = None


class TRRPrediction(BaseModel):
    """
    TRR 查询的预测结果。

    Prediction for TRR query.
    """
    query_id: str
    hour: int = Field(ge=1, le=12)
    quadrant: Optional[int] = Field(ge=1, le=4, default=None)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    reasoning: Optional[str] = None


class OSDPrediction(BaseModel):
    """
    T2 任务（完整 OSD 提取）的预测结果。

    Prediction for T2 task (full OSD extraction).
    Supports all constraint types from the formal DSL.
    """
    scene_id: str
    objects: List[str]
    qrr: List[QRRConstraintSchema] = Field(default_factory=list)
    trr: List[TRRConstraintSchema] = Field(default_factory=list)
    topology: List[TopologyConstraint] = Field(default_factory=list)
    occlusion: List[OcclusionConstraint] = Field(default_factory=list)
    axial: List[AxialConstraint] = Field(default_factory=list)
    size: List[SizeConstraint] = Field(default_factory=list)
    closer: List[CloserConstraint] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)


# =============================================================================
# JSON Schema Generation
# =============================================================================

def generate_json_schemas(output_dir: str = "."):
    """
    为所有数据结构生成 JSON Schema 文件。

    Generate JSON Schema files for all data structures.
    """
    import os

    schemas = {
        "qrr_query.json": QRRQuery.model_json_schema(),
        "trr_query.json": TRRQuery.model_json_schema(),
        "osd.json": OrdinalSceneDescription.model_json_schema(),
        "prediction.json": OSDPrediction.model_json_schema(),
        "axial_constraint.json": AxialConstraint.model_json_schema(),
        "size_constraint.json": SizeConstraint.model_json_schema(),
        "closer_constraint.json": CloserConstraint.model_json_schema(),
    }

    for filename, schema in schemas.items():
        path = os.path.join(output_dir, filename)
        with open(path, 'w') as f:
            json.dump(schema, f, indent=2)
        print(f"Generated: {path}")
