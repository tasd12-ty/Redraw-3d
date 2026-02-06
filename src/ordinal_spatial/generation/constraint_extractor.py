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
)


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
        提取视角不变的 3D 约束。

        Extract view-invariant 3D constraints.
        """
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

        # Limit if configured
        if self.config.max_qrr_per_scene and len(qrr_list) > self.config.max_qrr_per_scene:
            # Sample to keep diverse difficulties
            qrr_list = self._sample_diverse(qrr_list, self.config.max_qrr_per_scene)

        # Convert to schema format
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

        return WorldConstraints(qrr=qrr_schemas, topology=[])

    def _extract_view_constraints(
        self,
        objects: Dict[str, Dict],
        scene: Dict
    ) -> List[ViewConstraints]:
        """
        提取每个视角的 2D 约束。

        Extract per-view 2D constraints.
        """
        views = []

        # Handle single camera or multiple views
        view_data = scene.get("views", [{"camera": scene.get("camera", {})}])

        for i, view in enumerate(view_data):
            camera_data = view.get("camera", {})
            camera = CameraParams(
                camera_id=camera_data.get("camera_id", f"view_{i}"),
                position=camera_data.get("position", [0, 0, 5]),
                look_at=camera_data.get("look_at", [0, 0, 0]),
                fov=camera_data.get("fov", 50),
            )

            # Extract 2D QRR
            qrr_2d = []
            if self.config.include_qrr and MetricType.DIST_2D in self.config.metrics:
                constraints = extract_qrr_from_scene(
                    objects,
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

            # Extract TRR
            trr = []
            if self.config.include_trr:
                trr_constraints = extract_trr_from_scene(objects)
                if self.config.max_trr_per_scene and len(trr_constraints) > self.config.max_trr_per_scene:
                    trr_constraints = trr_constraints[:self.config.max_trr_per_scene]

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

            views.append(ViewConstraints(
                camera=camera,
                qrr_2d=qrr_2d,
                trr=trr,
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
