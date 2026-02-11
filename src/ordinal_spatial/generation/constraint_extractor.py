"""
从 3D 场景数据中提取约束。

本模块从已知的 3D 场景几何信息中提取真值序约束（QRR 和 TRR）。
这是生成基准测试真值的核心组件。

功能：
- 提取所有可能的 QRR 约束（成对距离比较）
- 提取所有可能的 TRR 约束（时钟方向关系）
- 支持多种度量类型（3D距离、2D距离、深度差等）
- 可配置最大约束数量
- 标记边界情况（接近阈值的约束）
- 支持仅不相交对（避免共享物体）

Constraint extraction from 3D scene data.

This module extracts ground-truth ordinal constraints (QRR and TRR)
from known 3D scene geometry. It's the core component for generating
benchmark ground truth.
"""

from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from itertools import combinations, permutations
import numpy as np

from ordinal_spatial.dsl.predicates import (
    QRRConstraint,
    TRRConstraint,
    MetricType,
    compute_qrr,
    compute_trr,
    compute_dist_3d,
    compute_dist_2d,
    compute_depth_gap,
)
from ordinal_spatial.dsl.comparators import compare, difficulty_from_ratio
from ordinal_spatial.dsl.schema import (
    OrdinalSceneDescription,
    ObjectSpec,
    WorldConstraints,
    ViewConstraints,
    CameraParams,
    QRRConstraintSchema,
    TRRConstraintSchema,
    AxialConstraint,
    AxialRelation,
    OcclusionConstraint,
    TopologyConstraint,
    SizeConstraint,
    CloserConstraint,
)
import math


@dataclass
class ExtractionConfig:
    """Configuration for constraint extraction."""
    tau: float = 0.10
    disjoint_pairs_only: bool = True
    include_qrr: bool = True
    include_trr: bool = True
    metrics: List[MetricType] = None
    max_qrr_per_scene: Optional[int] = None
    max_trr_per_scene: Optional[int] = None
    flag_boundary_cases: bool = True
    boundary_margin: float = 0.2  # Flag if within 20% of threshold

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [MetricType.DIST_3D]


class ConstraintExtractor:
    """
    Extract ordinal constraints from scene data.

    This class provides the core functionality for generating ground-truth
    constraints from known 3D geometry.
    """

    def __init__(self, config: ExtractionConfig = None):
        """
        Initialize the extractor.

        Args:
            config: Extraction configuration
        """
        self.config = config or ExtractionConfig()

    def extract(self, scene: Dict) -> OrdinalSceneDescription:
        """
        Extract all constraints from a scene.

        Args:
            scene: Scene dictionary with objects and camera info

        Returns:
            Complete OrdinalSceneDescription with constraints
        """
        # Parse objects
        objects = self._parse_objects(scene)
        objects_dict = {obj.id: obj.to_dict() for obj in objects}

        # Extract world constraints (view-invariant)
        world = self._extract_world_constraints(objects_dict)

        # Extract view constraints if camera info available
        views = []
        if "camera" in scene or "views" in scene:
            views = self._extract_view_constraints(objects_dict, scene)

        # Build OSD
        osd = OrdinalSceneDescription(
            scene_id=scene.get("scene_id", f"scene_{hash(str(scene)) % 10000:04d}"),
            objects=objects,
            tau=self.config.tau,
            world=world,
            views=views,
            metadata={
                "n_objects": len(objects),
                "extraction_config": {
                    "tau": self.config.tau,
                    "disjoint_only": self.config.disjoint_pairs_only,
                    "metrics": [str(m) for m in self.config.metrics],
                }
            }
        )

        return osd

    def _parse_objects(self, scene: Dict) -> List[ObjectSpec]:
        """
        从场景字典解析物体。

        Parse objects from scene dictionary.
        """
        objects = []
        raw_objects = scene.get("objects", [])

        for i, obj in enumerate(raw_objects):
            # Handle various input formats
            obj_id = obj.get("id", obj.get("name", f"obj_{i}"))
            shape = obj.get("shape", obj.get("type", "cube"))
            color = obj.get("color", "gray")
            size = obj.get("size", "medium")

            # Position formats
            pos_3d = obj.get("position_3d", obj.get("3d_coords", [0, 0, 0]))
            pos_2d = obj.get("position_2d", obj.get("pixel_coords", [0, 0])[:2])
            depth = obj.get("depth", 0)
            if len(obj.get("pixel_coords", [])) > 2:
                depth = obj["pixel_coords"][2]

            from ordinal_spatial.dsl.schema import ShapeType, SizeClass
            try:
                shape_type = ShapeType(shape.lower())
            except ValueError:
                shape_type = ShapeType.CUBE

            objects.append(ObjectSpec(
                id=obj_id,
                shape=shape_type,
                color=color,
                size=size,
                position_3d=list(pos_3d),
                position_2d=list(pos_2d),
                depth=depth,
            ))

        return objects

    def _extract_world_constraints(
        self,
        objects: Dict[str, Dict]
    ) -> WorldConstraints:
        """
        提取全部 7 类视角不变的 3D 约束。

        Extract all 7 types of view-invariant 3D constraints:
        QRR, topology, size, closer, axial, TRR, occlusion.
        """
        # --- QRR ---
        qrr_list = []
        if self.config.include_qrr:
            for metric in self.config.metrics:
                if metric == MetricType.DIST_3D:
                    constraints = extract_qrr_from_scene(
                        objects,
                        metric=metric,
                        tau=self.config.tau,
                        disjoint_only=self.config.disjoint_pairs_only,
                    )
                    qrr_list.extend(constraints)

        if (self.config.max_qrr_per_scene
            and len(qrr_list) > self.config.max_qrr_per_scene):
            qrr_list = self._sample_diverse(
                qrr_list, self.config.max_qrr_per_scene
            )

        qrr_schemas = [
            QRRConstraintSchema(
                pair1=list(c.pair1),
                pair2=list(c.pair2),
                metric=str(c.metric),
                comparator=str(c.comparator),
                ratio=c.ratio,
                difficulty=c.difficulty,
                boundary_flag=c.boundary_flag,
            )
            for c in qrr_list
        ]

        # --- TRR (3D) ---
        trr_schemas = []
        if self.config.include_trr:
            trr_list = extract_trr_from_scene(
                objects, use_3d=True
            )
            if (self.config.max_trr_per_scene
                and len(trr_list)
                > self.config.max_trr_per_scene):
                trr_list = trr_list[
                    :self.config.max_trr_per_scene
                ]
            trr_schemas = [
                TRRConstraintSchema(
                    target=c.target,
                    ref1=c.ref1,
                    ref2=c.ref2,
                    hour=c.hour,
                    quadrant=c.quadrant,
                    angle_deg=c.angle_deg,
                )
                for c in trr_list
            ]

        # --- Topology ---
        topology = extract_topology_from_scene(
            objects, tau=self.config.tau,
        )

        # --- Size ---
        size = extract_size_from_scene(
            objects, tau=self.config.tau,
        )

        # --- Closer ---
        closer = extract_closer_from_scene(
            objects, tau=self.config.tau,
        )

        # --- Axial (3D world coordinates) ---
        axial = extract_axial_3d_from_scene(
            objects, tau=self.config.tau,
        )

        # --- Occlusion (3D approximation) ---
        occlusion = extract_occlusion_3d_from_scene(
            objects, tau=self.config.tau,
        )

        return WorldConstraints(
            qrr=qrr_schemas,
            topology=topology,
            size=size,
            closer=closer,
            axial=axial,
            trr=trr_schemas,
            occlusion=occlusion,
        )

    def _build_view_objects(
        self,
        base_objects: Dict[str, Dict],
        view_data: Dict,
    ) -> Dict[str, Dict]:
        """
        用视角特定的 pixel_coords 覆盖物体的 position_2d。

        Override object position_2d with view-specific pixel_coords.
        """
        view_obj_list = view_data.get("objects", [])
        # obj_id → pixel_coords 查找表
        view_pixels = {}
        for obj in view_obj_list:
            obj_id = obj.get("id", obj.get("name"))
            if obj_id and "pixel_coords" in obj:
                view_pixels[obj_id] = obj["pixel_coords"]

        # 克隆并替换 2D 位置
        view_objects = {}
        for obj_id, obj_data in base_objects.items():
            obj_copy = dict(obj_data)
            if obj_id in view_pixels:
                pc = view_pixels[obj_id]
                obj_copy["position_2d"] = list(pc[:2])
                obj_copy["pixel_coords"] = list(pc)
                if len(pc) > 2:
                    obj_copy["depth"] = pc[2]
            view_objects[obj_id] = obj_copy

        return view_objects

    def _extract_view_constraints(
        self,
        objects: Dict[str, Dict],
        scene: Dict
    ) -> List[ViewConstraints]:
        """
        提取每个视角的 2D 约束 (含 axial_2d 和 occlusion)。

        Extract per-view 2D constraints including axial_2d and occlusion.
        """
        views = []

        # 处理单/多视角
        view_data = scene.get(
            "views", [{"camera": scene.get("camera", {})}]
        )

        for i, view in enumerate(view_data):
            camera_data = view.get("camera", {})
            camera = CameraParams(
                camera_id=camera_data.get("camera_id", f"view_{i}"),
                position=camera_data.get("position", [0, 0, 5]),
                look_at=camera_data.get("look_at", [0, 0, 0]),
                fov=camera_data.get("fov", 50),
            )

            # 构建视角特定的物体字典
            view_objects = self._build_view_objects(objects, view)

            # 提取 2D QRR
            qrr_2d = []
            if (self.config.include_qrr
                and MetricType.DIST_2D in self.config.metrics):
                constraints = extract_qrr_from_scene(
                    view_objects,
                    metric=MetricType.DIST_2D,
                    tau=self.config.tau,
                    disjoint_only=self.config.disjoint_pairs_only,
                )
                qrr_2d = [
                    QRRConstraintSchema(
                        pair1=list(c.pair1),
                        pair2=list(c.pair2),
                        metric=str(c.metric),
                        comparator=str(c.comparator),
                        ratio=c.ratio,
                        difficulty=c.difficulty,
                        boundary_flag=c.boundary_flag,
                    )
                    for c in constraints
                ]

            # 提取 TRR (使用视角特定 position_2d)
            trr = []
            if self.config.include_trr:
                trr_constraints = extract_trr_from_scene(
                    view_objects, use_3d=False
                )
                if (self.config.max_trr_per_scene
                    and len(trr_constraints)
                    > self.config.max_trr_per_scene):
                    trr_constraints = trr_constraints[
                        :self.config.max_trr_per_scene
                    ]

                trr = [
                    TRRConstraintSchema(
                        target=c.target,
                        ref1=c.ref1,
                        ref2=c.ref2,
                        hour=c.hour,
                        quadrant=c.quadrant,
                        angle_deg=c.angle_deg,
                    )
                    for c in trr_constraints
                ]

            # 提取视角 axial_2d
            axial_2d = extract_axial_2d_from_view(
                view_objects, tau=self.config.tau,
            )

            # 提取视角 occlusion
            occlusion = extract_occlusion_from_view(view_objects)

            views.append(ViewConstraints(
                camera=camera,
                qrr_2d=qrr_2d,
                trr=trr,
                axial_2d=axial_2d,
                occlusion=occlusion,
                image_path=view.get("image_path"),
                depth_path=view.get("depth_path"),
            ))

        return views

    def _sample_diverse(
        self,
        constraints: List[QRRConstraint],
        n: int
    ) -> List[QRRConstraint]:
        """
        采样约束以保持难度多样性。

        Sample constraints to maintain difficulty diversity.
        """
        if len(constraints) <= n:
            return constraints

        # Group by difficulty
        by_difficulty = {}
        for c in constraints:
            d = c.difficulty
            if d not in by_difficulty:
                by_difficulty[d] = []
            by_difficulty[d].append(c)

        # Sample proportionally from each difficulty level
        result = []
        per_level = max(1, n // len(by_difficulty))

        for level in sorted(by_difficulty.keys()):
            level_constraints = by_difficulty[level]
            sample_n = min(per_level, len(level_constraints))
            indices = np.random.choice(len(level_constraints), sample_n, replace=False)
            result.extend([level_constraints[i] for i in indices])

        # Fill remaining slots randomly
        while len(result) < n:
            remaining = [c for c in constraints if c not in result]
            if not remaining:
                break
            result.append(remaining[np.random.randint(len(remaining))])

        return result[:n]


# =============================================================================
# Standalone Extraction Functions
# =============================================================================

def extract_qrr_from_scene(
    objects: Dict[str, Dict],
    metric: MetricType = MetricType.DIST_3D,
    tau: float = 0.10,
    disjoint_only: bool = True,
) -> List[QRRConstraint]:
    """
    Extract all QRR constraints from scene objects.

    Args:
        objects: Dictionary of object_id -> object_data
        metric: Metric type to compare
        tau: Tolerance parameter
        disjoint_only: Only compare disjoint pairs

    Returns:
        List of QRRConstraint objects
    """
    obj_ids = list(objects.keys())
    pairs = list(combinations(obj_ids, 2))
    constraints = []

    for i, pair1 in enumerate(pairs):
        for pair2 in pairs[i + 1:]:
            # Check disjoint
            if disjoint_only and set(pair1) & set(pair2):
                continue

            try:
                constraint = compute_qrr(objects, pair1, pair2, metric, tau)
                constraints.append(constraint)
            except (KeyError, ValueError) as e:
                # Skip invalid pairs
                continue

    return constraints


def extract_trr_from_scene(
    objects: Dict[str, Dict],
    use_3d: bool = False,
) -> List[TRRConstraint]:
    """
    Extract all TRR constraints from scene objects.

    Args:
        objects: Dictionary of object_id -> object_data
        use_3d: Use 3D positions instead of 2D

    Returns:
        List of TRRConstraint objects
    """
    obj_ids = list(objects.keys())
    constraints = []

    for triple in permutations(obj_ids, 3):
        target, ref1, ref2 = triple
        try:
            constraint = compute_trr(objects, target, ref1, ref2, use_3d)
            constraints.append(constraint)
        except (KeyError, ValueError):
            continue

    return constraints


def extract_axial_2d_from_view(
    objects_for_view: Dict[str, Dict],
    tau: float = 0.10,
    image_width: int = 480,
    image_height: int = 320,
) -> List[AxialConstraint]:
    """
    从单视角的 pixel_coords 提取轴向约束。

    Extract 2D axial constraints from a view's pixel coordinates.

    Convention (image-plane):
      - px1 < px2 => obj1 LEFT_OF obj2
      - py1 < py2 => obj1 ABOVE obj2 (y increases downward)
      - depth1 < depth2 => obj1 IN_FRONT_OF obj2

    Args:
        objects_for_view: {obj_id: {"pixel_coords": [px, py, depth]}}
        tau: Tolerance (as fraction; scaled by image dimension)
        image_width: Image width in pixels (for tau scaling)
        image_height: Image height in pixels (for tau scaling)

    Returns:
        List of AxialConstraint
    """
    tau_px = tau * image_width
    tau_py = tau * image_height
    tau_d = tau  # depth 已归一化 [0, 1]

    # 过滤掉缺少 pixel_coords 的物体
    valid = {}
    for obj_id, obj in objects_for_view.items():
        pc = obj.get("pixel_coords", obj.get("position_2d"))
        if pc and len(pc) >= 2:
            depth = pc[2] if len(pc) > 2 else obj.get("depth", 0)
            valid[obj_id] = (pc[0], pc[1], depth)

    obj_ids = list(valid.keys())
    constraints = []

    for obj1_id, obj2_id in combinations(obj_ids, 2):
        px1, py1, d1 = valid[obj1_id]
        px2, py2, d2 = valid[obj2_id]

        # Left / Right
        dx = px1 - px2
        if abs(dx) > tau_px:
            rel = AxialRelation.LEFT_OF if dx < 0 else AxialRelation.RIGHT_OF
            constraints.append(AxialConstraint(
                obj1=obj1_id, obj2=obj2_id, relation=rel
            ))

        # Above / Below (图像 y 轴向下, py 小 = 上方)
        dy = py1 - py2
        if abs(dy) > tau_py:
            rel = AxialRelation.ABOVE if dy < 0 else AxialRelation.BELOW
            constraints.append(AxialConstraint(
                obj1=obj1_id, obj2=obj2_id, relation=rel
            ))

        # In front of / Behind (depth 小 = 离相机近)
        dd = d1 - d2
        if abs(dd) > tau_d:
            rel = (AxialRelation.IN_FRONT_OF if dd < 0
                   else AxialRelation.BEHIND)
            constraints.append(AxialConstraint(
                obj1=obj1_id, obj2=obj2_id, relation=rel
            ))

    return constraints


def extract_occlusion_from_view(
    objects_for_view: Dict[str, Dict],
    tau_depth: float = 0.05,
    size_radius: Optional[Dict[str, float]] = None,
) -> List[OcclusionConstraint]:
    """
    从单视角像素坐标推断遮挡关系。

    Approximate occlusion from pixel proximity + depth ordering.

    Two objects may have an occlusion relationship when:
    1. Their 2D pixel distance is small (bounding boxes overlap)
    2. One is clearly closer to the camera (smaller depth)

    Args:
        objects_for_view: {obj_id: {"pixel_coords": [px,py,depth],
                                     "size": "large"/"medium"/"small"}}
        tau_depth: Minimum depth difference to declare occlusion
        size_radius: Approximate screen radius per size class

    Returns:
        List of OcclusionConstraint
    """
    if size_radius is None:
        size_radius = {
            "large": 60, "medium": 40, "small": 25,
        }
    default_radius = 40

    # 收集有效物体
    valid = {}
    for obj_id, obj in objects_for_view.items():
        pc = obj.get("pixel_coords", obj.get("position_2d"))
        if pc and len(pc) >= 3 and pc[2] > 0:
            size_str = str(
                obj.get("size", obj.get("size_class", "medium"))
            ).lower()
            radius = size_radius.get(size_str, default_radius)
            valid[obj_id] = (pc[0], pc[1], pc[2], radius)

    obj_ids = list(valid.keys())
    constraints = []

    for obj_i, obj_j in combinations(obj_ids, 2):
        px_i, py_i, d_i, r_i = valid[obj_i]
        px_j, py_j, d_j, r_j = valid[obj_j]

        # 像素距离
        pixel_dist = math.sqrt(
            (px_i - px_j) ** 2 + (py_i - py_j) ** 2
        )
        # 是否在屏幕上重叠
        if pixel_dist >= (r_i + r_j):
            continue

        # 深度差是否显著
        depth_diff = abs(d_i - d_j)
        if depth_diff < tau_depth:
            continue

        # depth 更小的是 occluder (离相机更近)
        if d_i < d_j:
            constraints.append(OcclusionConstraint(
                occluder=obj_i, occluded=obj_j, partial=True
            ))
        else:
            constraints.append(OcclusionConstraint(
                occluder=obj_j, occluded=obj_i, partial=True
            ))

    return constraints


# =============================================================================
# World-level (3D) Extraction Functions
# =============================================================================

_SIZE_VALUE = {
    "tiny": 0.25, "small": 0.35,
    "medium": 0.50, "large": 0.70,
}


def _size_to_value(size: str) -> float:
    """尺寸类别 → 数值。"""
    return _SIZE_VALUE.get(str(size).lower(), 0.50)


def _size_to_radius(size: str) -> float:
    """尺寸类别 → 近似半径。"""
    return _size_to_value(size) * 0.5


def _get_pos_3d(obj: Dict) -> List[float]:
    """提取 3D 坐标, 兼容多种格式。"""
    return obj.get(
        "position_3d",
        obj.get("3d_coords", obj.get("position", [0, 0, 0]))
    )


def extract_topology_from_scene(
    objects: Dict[str, Dict],
    tau: float = 0.10,
) -> List[TopologyConstraint]:
    """
    从 3D 位置和尺寸推断拓扑关系。

    Extract topology constraints from 3D positions and sizes.

    Heuristic:
      combined_radius = r1 + r2
      dist > 1.2 * combined  →  disjoint
      dist > 0.8 * combined  →  touching
      else                   →  overlapping
    """
    from ordinal_spatial.dsl.comparators import compare as cmp_fn
    obj_ids = list(objects.keys())
    constraints = []

    for a, b in combinations(obj_ids, 2):
        pos_a = _get_pos_3d(objects[a])
        pos_b = _get_pos_3d(objects[b])

        dist = math.sqrt(sum(
            (x - y) ** 2 for x, y in zip(pos_a, pos_b)
        ))
        r_a = _size_to_radius(
            objects[a].get("size", "medium")
        )
        r_b = _size_to_radius(
            objects[b].get("size", "medium")
        )
        combined = r_a + r_b

        if dist > combined * 1.2:
            relation = "disjoint"
        elif dist > combined * 0.8:
            relation = "touching"
        else:
            relation = "overlapping"

        constraints.append(TopologyConstraint(
            obj1=a, obj2=b, relation=relation
        ))

    return constraints


def extract_size_from_scene(
    objects: Dict[str, Dict],
    tau: float = 0.10,
) -> List[SizeConstraint]:
    """
    从物理尺寸提取大小比较约束。

    Extract size comparison constraints from physical sizes.
    """
    from ordinal_spatial.dsl.comparators import (
        compare as cmp_fn, Comparator,
    )
    obj_ids = list(objects.keys())
    constraints = []

    for a, b in combinations(obj_ids, 2):
        va = _size_to_value(objects[a].get("size", "medium"))
        vb = _size_to_value(objects[b].get("size", "medium"))

        result = cmp_fn(va, vb, tau)
        if result == Comparator.GT:
            constraints.append(
                SizeConstraint(bigger=a, smaller=b)
            )
        elif result == Comparator.LT:
            constraints.append(
                SizeConstraint(bigger=b, smaller=a)
            )

    return constraints


def extract_closer_from_scene(
    objects: Dict[str, Dict],
    tau: float = 0.10,
) -> List[CloserConstraint]:
    """
    从 3D 距离提取三元距离比较约束。

    Extract ternary closer constraints from 3D distances.
    For each (anchor, obj_a, obj_b) triple, emit constraint
    if d(anchor,a) < d(anchor,b) with tolerance tau.
    """
    from ordinal_spatial.dsl.comparators import (
        compare as cmp_fn, Comparator,
    )
    obj_ids = list(objects.keys())
    constraints = []

    for anchor, a, b in permutations(obj_ids, 3):
        pa = _get_pos_3d(objects[anchor])
        p_a = _get_pos_3d(objects[a])
        p_b = _get_pos_3d(objects[b])

        d_a = math.sqrt(sum(
            (x - y) ** 2 for x, y in zip(pa, p_a)
        ))
        d_b = math.sqrt(sum(
            (x - y) ** 2 for x, y in zip(pa, p_b)
        ))

        result = cmp_fn(d_a, d_b, tau)
        if result == Comparator.LT:
            constraints.append(CloserConstraint(
                anchor=anchor, closer=a, farther=b,
            ))

    return constraints


def extract_axial_3d_from_scene(
    objects: Dict[str, Dict],
    tau: float = 0.10,
) -> List[AxialConstraint]:
    """
    从 3D 世界坐标提取轴向关系。

    Extract axial constraints from 3D world coordinates.

    Convention (Blender / CLEVR world frame):
      X: left (−) / right (+)
      Y: front (−) / behind (+)
      Z: below (−) / above (+)
    """
    obj_ids = list(objects.keys())
    constraints = []

    for a, b in combinations(obj_ids, 2):
        pos_a = _get_pos_3d(objects[a])
        pos_b = _get_pos_3d(objects[b])

        # X: left / right
        dx = pos_a[0] - pos_b[0]
        if abs(dx) > tau:
            rel = (AxialRelation.LEFT_OF if dx < 0
                   else AxialRelation.RIGHT_OF)
            constraints.append(AxialConstraint(
                obj1=a, obj2=b, relation=rel,
            ))

        # Z: above / below
        if len(pos_a) > 2 and len(pos_b) > 2:
            dz = pos_a[2] - pos_b[2]
            if abs(dz) > tau:
                rel = (AxialRelation.ABOVE if dz > 0
                       else AxialRelation.BELOW)
                constraints.append(AxialConstraint(
                    obj1=a, obj2=b, relation=rel,
                ))

        # Y: in front of / behind
        dy = pos_a[1] - pos_b[1]
        if abs(dy) > tau:
            rel = (AxialRelation.IN_FRONT_OF if dy < 0
                   else AxialRelation.BEHIND)
            constraints.append(AxialConstraint(
                obj1=a, obj2=b, relation=rel,
            ))

    return constraints


def extract_occlusion_3d_from_scene(
    objects: Dict[str, Dict],
    tau: float = 0.10,
) -> List[OcclusionConstraint]:
    """
    从 3D 位置近似遮挡关系。

    Approximate occlusion from 3D coordinates.

    Heuristic: two objects overlap in X/Z (screen-space proxy)
    AND one is significantly closer along Y (depth axis).
    """
    obj_ids = list(objects.keys())
    constraints = []

    for a, b in combinations(obj_ids, 2):
        pos_a = _get_pos_3d(objects[a])
        pos_b = _get_pos_3d(objects[b])

        # X/Z 重叠检查 (screen-space 代理)
        x_overlap = abs(pos_a[0] - pos_b[0]) < 1.0
        z_overlap = (
            abs(pos_a[2] - pos_b[2]) < 1.0
            if len(pos_a) > 2 and len(pos_b) > 2
            else True
        )

        if not (x_overlap and z_overlap):
            continue

        # Y 深度差 (CLEVR: Y 小 = 离相机近)
        dy = pos_a[1] - pos_b[1]
        if abs(dy) > tau:
            if dy < 0:
                constraints.append(OcclusionConstraint(
                    occluder=a, occluded=b, partial=True,
                ))
            else:
                constraints.append(OcclusionConstraint(
                    occluder=b, occluded=a, partial=True,
                ))

    return constraints


def extract_scene_constraints(
    scene: Dict,
    tau: float = 0.10,
    metrics: List[str] = None,
) -> OrdinalSceneDescription:
    """
    Convenience function to extract constraints from a scene.

    Args:
        scene: Scene dictionary
        tau: Tolerance parameter
        metrics: List of metric names (default: ["dist3D"])

    Returns:
        OrdinalSceneDescription with extracted constraints
    """
    if metrics is None:
        metrics = ["dist3D"]

    metric_types = [MetricType.from_string(m) for m in metrics]

    config = ExtractionConfig(
        tau=tau,
        metrics=metric_types,
    )

    extractor = ConstraintExtractor(config)
    return extractor.extract(scene)


# =============================================================================
# Validation Functions
# =============================================================================

def validate_extracted_constraints(
    osd: OrdinalSceneDescription
) -> Dict[str, Any]:
    """
    Validate extracted constraints for consistency and coverage.

    Args:
        osd: Ordinal Scene Description to validate

    Returns:
        Validation report dictionary
    """
    from ordinal_spatial.evaluation.consistency import check_qrr_consistency

    report = {
        "valid": True,
        "n_objects": len(osd.objects),
        "n_qrr": len(osd.world.qrr),
        "n_views": len(osd.views),
        "issues": [],
    }

    # Check QRR consistency
    qrr_dicts = [c.model_dump() for c in osd.world.qrr]
    consistency = check_qrr_consistency(qrr_dicts)

    if not consistency.is_consistent:
        report["valid"] = False
        report["issues"].append(f"QRR inconsistency: {len(consistency.cycles)} cycles")

    # Check coverage
    n_objects = len(osd.objects)
    expected_pairs = n_objects * (n_objects - 1) // 2

    # For disjoint comparisons: C(n,4) * 3 pairs
    if n_objects >= 4:
        from math import comb
        expected_qrr = comb(n_objects, 4) * 3
        actual_qrr = len(osd.world.qrr)
        coverage = actual_qrr / expected_qrr if expected_qrr > 0 else 1.0
        report["qrr_coverage"] = coverage

        if coverage < 0.5:
            report["issues"].append(f"Low QRR coverage: {coverage:.1%}")

    return report
