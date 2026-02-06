"""
Blender 约束提取智能体 (Task-1)。

Blender Constraint Extraction Agent for Task-1.

This agent extracts ground-truth spatial constraints from:
1. CLEVR_scenes.json (Blender render output)
2. .blend files directly (when run inside Blender)

Features:
- Extract all constraint types (QRR, TRR, topology, axial, size, closer)
- Compute transitive closures for ordering constraints
- Support for multiple scenes in batch
- Ground truth generation for benchmark
"""

import json
import logging
import math
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from itertools import combinations, permutations

from ordinal_spatial.agents.base import ConstraintAgent, ConstraintSet, ObjectInfo
from ordinal_spatial.dsl.schema import (
    QRRConstraintSchema,
    TRRConstraintSchema,
    TopologyConstraint,
    OcclusionConstraint,
    AxialConstraint,
    AxialRelation,
    SizeConstraint,
    CloserConstraint,
)
from ordinal_spatial.dsl.predicates import (
    compute_qrr,
    compute_trr,
    MetricType,
)
from ordinal_spatial.dsl.comparators import compare, Comparator


logger = logging.getLogger(__name__)


@dataclass
class BlenderAgentConfig:
    """
    Blender 智能体配置。

    Configuration for Blender Constraint Agent.

    Attributes:
        tau: Tolerance parameter for comparisons
        extract_qrr: Whether to extract QRR constraints
        extract_trr: Whether to extract TRR constraints
        extract_axial: Whether to extract axial ordering constraints
        extract_topology: Whether to extract topology constraints
        extract_size: Whether to extract size comparison constraints
        extract_closer: Whether to extract ternary closer constraints
        extract_occlusion: Whether to extract occlusion constraints
        disjoint_pairs_only: Only compare disjoint object pairs for QRR
        max_qrr_per_scene: Maximum QRR constraints per scene (None for all)
        max_trr_per_scene: Maximum TRR constraints per scene (None for all)
        compute_transitive_closure: Whether to compute transitive closures
    """
    tau: float = 0.10
    extract_qrr: bool = True
    extract_trr: bool = True
    extract_axial: bool = True
    extract_topology: bool = True
    extract_size: bool = True
    extract_closer: bool = True
    extract_occlusion: bool = True
    disjoint_pairs_only: bool = True
    max_qrr_per_scene: Optional[int] = None
    max_trr_per_scene: Optional[int] = None
    compute_transitive_closure: bool = True


class BlenderConstraintAgent(ConstraintAgent):
    """
    从 Blender 场景数据提取约束的智能体 (Task-1)。

    Agent for extracting constraints from Blender scene data (Task-1).

    This provides the ground truth upper bound for the benchmark.
    """

    def __init__(self, config: Optional[BlenderAgentConfig] = None):
        """
        初始化智能体。

        Initialize the agent.

        Args:
            config: Agent configuration
        """
        self.config = config or BlenderAgentConfig()

    def extract_from_single_view(
        self,
        image: Union[str, Path, bytes],
        objects: Optional[List[Dict[str, Any]]] = None,
        tau: float = 0.10,
    ) -> ConstraintSet:
        """
        从单视角场景数据提取约束。

        Extract constraints from single-view scene data.

        Note: For Blender extraction, 'image' can be:
        - Path to CLEVR_scenes.json
        - Path to single scene JSON file
        - Scene dictionary directly

        Args:
            image: Scene data source (JSON path or dict)
            objects: Optional override for objects
            tau: Tolerance parameter

        Returns:
            ConstraintSet with extracted constraints
        """
        self.config.tau = tau

        # Load scene data
        if isinstance(image, dict):
            scene_data = image
        elif isinstance(image, (str, Path)):
            path = Path(image)
            if path.suffix == ".json":
                with open(path) as f:
                    scene_data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        else:
            raise ValueError(f"Unsupported input type: {type(image)}")

        # Handle CLEVR_scenes.json format (multiple scenes)
        if "scenes" in scene_data:
            # Extract from first scene only for single view
            if len(scene_data["scenes"]) > 0:
                scene_data = scene_data["scenes"][0]
            else:
                return ConstraintSet(confidence=0.0, metadata={"error": "No scenes found"})

        # Override objects if provided
        if objects:
            scene_data["objects"] = objects

        return self._extract_from_scene(scene_data)

    def extract_from_multi_view(
        self,
        images: List[Union[str, Path, bytes]],
        objects: Optional[List[Dict[str, Any]]] = None,
        tau: float = 0.10,
    ) -> ConstraintSet:
        """
        从多视角场景数据提取约束。

        Extract constraints from multi-view scene data.

        For Blender extraction, this merges constraints from multiple scenes
        or multiple views of the same scene.

        Args:
            images: List of scene data sources
            objects: Optional override for objects
            tau: Tolerance parameter

        Returns:
            ConstraintSet with merged constraints
        """
        self.config.tau = tau

        all_constraints = []
        for img in images:
            cs = self.extract_from_single_view(img, objects, tau)
            all_constraints.append(cs)

        # Merge constraints (take intersection for view-invariant)
        return self._merge_constraint_sets(all_constraints)

    def extract_from_clevr_scenes(
        self,
        scenes_json_path: Union[str, Path],
        tau: float = 0.10,
    ) -> List[ConstraintSet]:
        """
        从 CLEVR_scenes.json 提取所有场景的约束。

        Extract constraints from all scenes in CLEVR_scenes.json.

        Args:
            scenes_json_path: Path to CLEVR_scenes.json
            tau: Tolerance parameter

        Returns:
            List of ConstraintSet, one per scene
        """
        self.config.tau = tau

        with open(scenes_json_path) as f:
            data = json.load(f)

        results = []
        scenes = data.get("scenes", [data])  # Handle single scene or multiple

        for i, scene in enumerate(scenes):
            logger.info(f"Extracting constraints from scene {i + 1}/{len(scenes)}")
            cs = self._extract_from_scene(scene)
            cs.metadata["scene_index"] = i
            cs.metadata["image_filename"] = scene.get("image_filename", f"scene_{i}")
            results.append(cs)

        return results

    def extract_from_blend_file(
        self,
        blend_path: Union[str, Path],
        tau: float = 0.10,
    ) -> ConstraintSet:
        """
        从 .blend 文件提取约束（需要在 Blender 中运行）。

        Extract constraints from .blend file (must run inside Blender).

        Args:
            blend_path: Path to .blend file
            tau: Tolerance parameter

        Returns:
            ConstraintSet with extracted constraints
        """
        self.config.tau = tau

        try:
            import bpy
        except ImportError:
            raise ImportError(
                "bpy module not available. This method must be run inside Blender. "
                "Use: blender --background --python your_script.py"
            )

        # Load blend file
        bpy.ops.wm.open_mainfile(filepath=str(blend_path))

        # Extract scene data from Blender
        scene_data = self._extract_scene_from_blender()

        return self._extract_from_scene(scene_data)

    def _extract_scene_from_blender(self) -> Dict[str, Any]:
        """
        从当前 Blender 场景提取数据。

        Extract scene data from current Blender scene.

        Returns:
            Scene data dictionary
        """
        import bpy

        scene = bpy.context.scene
        objects = []

        for obj in scene.objects:
            if obj.type != 'MESH':
                continue

            # Skip ground plane and other non-object meshes
            if obj.name.lower() in ('ground', 'plane', 'floor', 'ground_plane'):
                continue

            # Get object properties
            location = list(obj.location)
            scale = list(obj.scale)

            # Determine shape from mesh name or custom property
            shape = self._infer_shape_from_blender_object(obj)

            # Get color from material
            color = self._get_color_from_blender_object(obj)

            # Determine size class from scale
            avg_scale = sum(scale) / 3
            if avg_scale < 0.4:
                size_class = "small"
            elif avg_scale > 0.7:
                size_class = "large"
            else:
                size_class = "medium"

            objects.append({
                "id": obj.name,
                "shape": shape,
                "color": color,
                "size": size_class,
                "3d_coords": location,
                "scale": avg_scale,
                "rotation": list(obj.rotation_euler),
            })

        return {
            "objects": objects,
            "scene_id": scene.name,
        }

    def _infer_shape_from_blender_object(self, obj) -> str:
        """从 Blender 物体推断形状类型。"""
        name_lower = obj.name.lower()

        if 'cube' in name_lower or 'box' in name_lower:
            return 'cube'
        elif 'sphere' in name_lower or 'ball' in name_lower:
            return 'sphere'
        elif 'cylinder' in name_lower:
            return 'cylinder'
        elif 'cone' in name_lower:
            return 'cone'
        elif 'torus' in name_lower or 'donut' in name_lower:
            return 'torus'

        # Try to infer from mesh geometry
        mesh = obj.data
        if mesh:
            verts = len(mesh.vertices)
            faces = len(mesh.polygons)

            # Simple heuristics
            if verts == 8 and faces == 6:
                return 'cube'
            elif verts > 100:
                return 'sphere'

        return 'unknown'

    def _get_color_from_blender_object(self, obj) -> str:
        """从 Blender 物体获取颜色。"""
        if obj.active_material:
            mat = obj.active_material
            if mat.use_nodes:
                # Try to find Principled BSDF
                for node in mat.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        color = node.inputs['Base Color'].default_value
                        return self._rgb_to_color_name(color[:3])

            # Fallback to diffuse color
            if hasattr(mat, 'diffuse_color'):
                return self._rgb_to_color_name(mat.diffuse_color[:3])

        return 'gray'

    def _rgb_to_color_name(self, rgb: Tuple[float, float, float]) -> str:
        """将 RGB 值转换为颜色名称。"""
        r, g, b = rgb

        # Simple color mapping based on dominant channel
        colors = {
            (1, 0, 0): 'red',
            (0, 1, 0): 'green',
            (0, 0, 1): 'blue',
            (1, 1, 0): 'yellow',
            (1, 0, 1): 'purple',
            (0, 1, 1): 'cyan',
            (0.5, 0.5, 0.5): 'gray',
            (1, 1, 1): 'white',
            (0, 0, 0): 'black',
            (1, 0.5, 0): 'orange',
            (0.6, 0.3, 0): 'brown',
        }

        # Find closest color
        min_dist = float('inf')
        closest = 'gray'

        for color_rgb, name in colors.items():
            dist = sum((a - b) ** 2 for a, b in zip(rgb, color_rgb))
            if dist < min_dist:
                min_dist = dist
                closest = name

        return closest

    def _extract_from_scene(self, scene_data: Dict[str, Any]) -> ConstraintSet:
        """
        从场景数据提取所有约束。

        Extract all constraints from scene data.
        """
        # Parse objects
        raw_objects = scene_data.get("objects", [])
        objects_dict = {}
        object_infos = []

        for i, obj in enumerate(raw_objects):
            obj_id = obj.get("id", obj.get("name", f"obj_{i}"))

            # Get position (handle various formats)
            pos_3d = obj.get("3d_coords", obj.get("position_3d", obj.get("position", [0, 0, 0])))
            pos_2d = obj.get("pixel_coords", obj.get("position_2d", [0, 0]))[:2] if obj.get("pixel_coords") or obj.get("position_2d") else [0, 0]

            objects_dict[obj_id] = {
                "id": obj_id,
                "position": pos_3d,
                "position_3d": pos_3d,
                "position_2d": pos_2d,
                "shape": obj.get("shape", obj.get("type", "cube")),
                "color": obj.get("color", "gray"),
                "size": obj.get("size", "medium"),
                "material": obj.get("material", "rubber"),
                "rotation": obj.get("rotation", 0),
            }

            object_infos.append(ObjectInfo(
                id=obj_id,
                type=obj.get("shape", obj.get("type", "cube")),
                color=obj.get("color", "gray"),
                size_class=obj.get("size", "medium"),
                position_2d=pos_2d,
            ))

        # Extract constraints
        qrr = self._extract_qrr(objects_dict) if self.config.extract_qrr else []
        trr = self._extract_trr(objects_dict) if self.config.extract_trr else []
        axial = self._extract_axial(objects_dict) if self.config.extract_axial else []
        topology = self._extract_topology(objects_dict) if self.config.extract_topology else []
        size = self._extract_size(objects_dict) if self.config.extract_size else []
        closer = self._extract_closer(objects_dict) if self.config.extract_closer else []
        occlusion = self._extract_occlusion(objects_dict, scene_data) if self.config.extract_occlusion else []

        # Compute transitive closures if configured
        if self.config.compute_transitive_closure:
            axial = self._compute_axial_closure(axial)
            size = self._compute_size_closure(size)

        result = ConstraintSet(
            objects=object_infos,
            qrr=qrr,
            trr=trr,
            axial=axial,
            topology=topology,
            size=size,
            closer=closer,
            occlusion=occlusion,
            confidence=1.0,  # Ground truth = perfect confidence
            metadata={
                "source": "blender",
                "scene_id": scene_data.get("scene_id", scene_data.get("image_index", "unknown")),
                "tau": self.config.tau,
                "n_objects": len(object_infos),
            }
        )
        result.metadata["counts"] = result.count_by_arity()
        return result

    def _extract_qrr(self, objects: Dict[str, Dict]) -> List[QRRConstraintSchema]:
        """提取 QRR 约束。"""
        obj_ids = list(objects.keys())
        pairs = list(combinations(obj_ids, 2))
        constraints = []

        for i, pair1 in enumerate(pairs):
            for pair2 in pairs[i + 1:]:
                # Check disjoint
                if self.config.disjoint_pairs_only and set(pair1) & set(pair2):
                    continue

                try:
                    qrr = compute_qrr(objects, pair1, pair2, MetricType.DIST_3D, self.config.tau)
                    constraints.append(QRRConstraintSchema(
                        pair1=list(pair1),
                        pair2=list(pair2),
                        metric=str(qrr.metric),
                        comparator=str(qrr.comparator),
                        ratio=qrr.ratio,
                        difficulty=qrr.difficulty,
                        boundary_flag=qrr.boundary_flag,
                    ))
                except (KeyError, ValueError) as e:
                    logger.debug(f"Skipping QRR for {pair1}, {pair2}: {e}")

        # Limit if configured
        if self.config.max_qrr_per_scene and len(constraints) > self.config.max_qrr_per_scene:
            constraints = constraints[:self.config.max_qrr_per_scene]

        return constraints

    def _extract_trr(self, objects: Dict[str, Dict]) -> List[TRRConstraintSchema]:
        """提取 TRR 约束。"""
        obj_ids = list(objects.keys())
        constraints = []

        for triple in permutations(obj_ids, 3):
            target, ref1, ref2 = triple
            try:
                trr = compute_trr(objects, target, ref1, ref2, use_3d=False)
                constraints.append(TRRConstraintSchema(
                    target=trr.target,
                    ref1=trr.ref1,
                    ref2=trr.ref2,
                    hour=trr.hour,
                    quadrant=trr.quadrant,
                    angle_deg=trr.angle_deg,
                ))
            except (KeyError, ValueError) as e:
                logger.debug(f"Skipping TRR for {triple}: {e}")

        # Limit if configured
        if self.config.max_trr_per_scene and len(constraints) > self.config.max_trr_per_scene:
            constraints = constraints[:self.config.max_trr_per_scene]

        return constraints

    def _extract_axial(self, objects: Dict[str, Dict]) -> List[AxialConstraint]:
        """提取轴向偏序约束。"""
        obj_ids = list(objects.keys())
        constraints = []

        for obj1_id, obj2_id in combinations(obj_ids, 2):
            obj1 = objects[obj1_id]
            obj2 = objects[obj2_id]

            pos1 = obj1.get("position_3d", obj1.get("position", [0, 0, 0]))
            pos2 = obj2.get("position_3d", obj2.get("position", [0, 0, 0]))

            # X-axis (left/right)
            if pos1[0] < pos2[0] - self.config.tau:
                constraints.append(AxialConstraint(
                    obj1=obj1_id, obj2=obj2_id, relation=AxialRelation.LEFT_OF
                ))
            elif pos1[0] > pos2[0] + self.config.tau:
                constraints.append(AxialConstraint(
                    obj1=obj1_id, obj2=obj2_id, relation=AxialRelation.RIGHT_OF
                ))

            # Y-axis (above/below) - Note: in CLEVR, Y is typically forward/back
            # Z-axis is typically up/down
            if len(pos1) > 2 and len(pos2) > 2:
                if pos1[2] > pos2[2] + self.config.tau:
                    constraints.append(AxialConstraint(
                        obj1=obj1_id, obj2=obj2_id, relation=AxialRelation.ABOVE
                    ))
                elif pos1[2] < pos2[2] - self.config.tau:
                    constraints.append(AxialConstraint(
                        obj1=obj1_id, obj2=obj2_id, relation=AxialRelation.BELOW
                    ))

            # Y-axis for depth (in_front_of/behind)
            if pos1[1] < pos2[1] - self.config.tau:
                constraints.append(AxialConstraint(
                    obj1=obj1_id, obj2=obj2_id, relation=AxialRelation.IN_FRONT_OF
                ))
            elif pos1[1] > pos2[1] + self.config.tau:
                constraints.append(AxialConstraint(
                    obj1=obj1_id, obj2=obj2_id, relation=AxialRelation.BEHIND
                ))

        return constraints

    def _extract_topology(self, objects: Dict[str, Dict]) -> List[TopologyConstraint]:
        """提取拓扑约束。"""
        obj_ids = list(objects.keys())
        constraints = []

        for obj1_id, obj2_id in combinations(obj_ids, 2):
            obj1 = objects[obj1_id]
            obj2 = objects[obj2_id]

            pos1 = obj1.get("position_3d", obj1.get("position", [0, 0, 0]))
            pos2 = obj2.get("position_3d", obj2.get("position", [0, 0, 0]))

            # Calculate distance
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))

            # Get sizes (approximate radius)
            size1 = self._size_to_radius(obj1.get("size", "medium"))
            size2 = self._size_to_radius(obj2.get("size", "medium"))

            combined_radius = size1 + size2

            # Determine topology
            if dist > combined_radius * 1.2:
                relation = "disjoint"
            elif dist > combined_radius * 0.8:
                relation = "touching"
            else:
                relation = "overlapping"

            constraints.append(TopologyConstraint(
                obj1=obj1_id, obj2=obj2_id, relation=relation
            ))

        return constraints

    def _extract_size(self, objects: Dict[str, Dict]) -> List[SizeConstraint]:
        """提取大小比较约束。"""
        obj_ids = list(objects.keys())
        constraints = []

        for obj1_id, obj2_id in combinations(obj_ids, 2):
            obj1 = objects[obj1_id]
            obj2 = objects[obj2_id]

            size1 = self._size_to_value(obj1.get("size", "medium"))
            size2 = self._size_to_value(obj2.get("size", "medium"))

            # Compare with tolerance
            comp = compare(size1, size2, self.config.tau)

            if comp == Comparator.GT:
                constraints.append(SizeConstraint(bigger=obj1_id, smaller=obj2_id))
            elif comp == Comparator.LT:
                constraints.append(SizeConstraint(bigger=obj2_id, smaller=obj1_id))
            # Skip approximately equal

        return constraints

    def _extract_closer(self, objects: Dict[str, Dict]) -> List[CloserConstraint]:
        """提取三元距离比较约束。"""
        obj_ids = list(objects.keys())
        constraints = []

        for anchor, closer, farther in permutations(obj_ids, 3):
            anchor_pos = objects[anchor].get("position_3d", objects[anchor].get("position", [0, 0, 0]))
            closer_pos = objects[closer].get("position_3d", objects[closer].get("position", [0, 0, 0]))
            farther_pos = objects[farther].get("position_3d", objects[farther].get("position", [0, 0, 0]))

            dist_closer = math.sqrt(sum((a - b) ** 2 for a, b in zip(anchor_pos, closer_pos)))
            dist_farther = math.sqrt(sum((a - b) ** 2 for a, b in zip(anchor_pos, farther_pos)))

            comp = compare(dist_closer, dist_farther, self.config.tau)

            if comp == Comparator.LT:
                constraints.append(CloserConstraint(
                    anchor=anchor, closer=closer, farther=farther
                ))

        return constraints

    def _extract_occlusion(
        self,
        objects: Dict[str, Dict],
        scene_data: Dict[str, Any]
    ) -> List[OcclusionConstraint]:
        """提取遮挡约束。"""
        # Check if occlusion info is in scene data
        if "relationships" in scene_data:
            # CLEVR format
            rels = scene_data["relationships"]
            if "behind" in rels:
                constraints = []
                obj_ids = list(objects.keys())
                behind_matrix = rels["behind"]

                for i, row in enumerate(behind_matrix):
                    for j in row:
                        if i < len(obj_ids) and j < len(obj_ids):
                            # Object i is behind object j, so j occludes i
                            constraints.append(OcclusionConstraint(
                                occluder=obj_ids[j],
                                occluded=obj_ids[i],
                                partial=True,
                            ))
                return constraints

        # Fallback: compute from depth (Y coordinate typically)
        obj_ids = list(objects.keys())
        constraints = []

        for obj1_id, obj2_id in combinations(obj_ids, 2):
            obj1 = objects[obj1_id]
            obj2 = objects[obj2_id]

            pos1 = obj1.get("position_3d", obj1.get("position", [0, 0, 0]))
            pos2 = obj2.get("position_3d", obj2.get("position", [0, 0, 0]))

            # Check X/Z overlap (2D bounding box)
            x_overlap = abs(pos1[0] - pos2[0]) < 1.0
            z_overlap = abs(pos1[2] - pos2[2]) < 1.0 if len(pos1) > 2 and len(pos2) > 2 else True

            if x_overlap and z_overlap:
                # Check depth (Y)
                if pos1[1] < pos2[1] - self.config.tau:
                    # obj1 is in front, may occlude obj2
                    constraints.append(OcclusionConstraint(
                        occluder=obj1_id, occluded=obj2_id, partial=True
                    ))
                elif pos2[1] < pos1[1] - self.config.tau:
                    constraints.append(OcclusionConstraint(
                        occluder=obj2_id, occluded=obj1_id, partial=True
                    ))

        return constraints

    def _size_to_value(self, size: str) -> float:
        """将尺寸类别转换为数值。"""
        mapping = {
            "tiny": 0.25,
            "small": 0.35,
            "medium": 0.5,
            "large": 0.7,
        }
        return mapping.get(str(size).lower(), 0.5)

    def _size_to_radius(self, size: str) -> float:
        """将尺寸类别转换为近似半径。"""
        return self._size_to_value(size) * 0.5

    def _compute_axial_closure(
        self,
        constraints: List[AxialConstraint]
    ) -> List[AxialConstraint]:
        """
        计算轴向约束的传递闭包。

        Compute transitive closure for axial constraints.
        """
        # Group by relation type and axis
        by_relation = {}
        for c in constraints:
            rel = c.relation
            if rel not in by_relation:
                by_relation[rel] = []
            by_relation[rel].append(c)

        # Compute closure for each transitive relation
        transitive_relations = [
            (AxialRelation.LEFT_OF, AxialRelation.LEFT_OF),
            (AxialRelation.RIGHT_OF, AxialRelation.RIGHT_OF),
            (AxialRelation.ABOVE, AxialRelation.ABOVE),
            (AxialRelation.BELOW, AxialRelation.BELOW),
            (AxialRelation.IN_FRONT_OF, AxialRelation.IN_FRONT_OF),
            (AxialRelation.BEHIND, AxialRelation.BEHIND),
        ]

        new_constraints = list(constraints)
        existing = {(c.obj1, c.obj2, c.relation) for c in constraints}

        for rel, _ in transitive_relations:
            if rel not in by_relation:
                continue

            # Build graph
            graph = {}
            for c in by_relation[rel]:
                if c.obj1 not in graph:
                    graph[c.obj1] = set()
                graph[c.obj1].add(c.obj2)

            # Find transitive edges
            for a in graph:
                for b in graph.get(a, set()):
                    for c in graph.get(b, set()):
                        if a != c and (a, c, rel) not in existing:
                            new_constraints.append(AxialConstraint(
                                obj1=a, obj2=c, relation=rel
                            ))
                            existing.add((a, c, rel))

        return new_constraints

    def _compute_size_closure(
        self,
        constraints: List[SizeConstraint]
    ) -> List[SizeConstraint]:
        """
        计算大小约束的传递闭包。

        Compute transitive closure for size constraints.
        """
        # Build graph: bigger -> set of smaller
        graph = {}
        for c in constraints:
            if c.bigger not in graph:
                graph[c.bigger] = set()
            graph[c.bigger].add(c.smaller)

        existing = {(c.bigger, c.smaller) for c in constraints}
        new_constraints = list(constraints)

        # Find transitive edges
        for a in graph:
            for b in graph.get(a, set()):
                for c in graph.get(b, set()):
                    if a != c and (a, c) not in existing:
                        new_constraints.append(SizeConstraint(bigger=a, smaller=c))
                        existing.add((a, c))

        return new_constraints

    def _merge_constraint_sets(
        self,
        constraint_sets: List[ConstraintSet]
    ) -> ConstraintSet:
        """
        合并多个约束集合。

        Merge multiple constraint sets.
        """
        if not constraint_sets:
            return ConstraintSet()

        if len(constraint_sets) == 1:
            return constraint_sets[0]

        # Take union of objects
        all_objects = {}
        for cs in constraint_sets:
            for obj in cs.objects:
                all_objects[obj.id] = obj

        # Take intersection of constraints (conservative merge)
        def constraint_key(c):
            if hasattr(c, 'pair1'):  # QRR
                return ('qrr', tuple(sorted(c.pair1)), tuple(sorted(c.pair2)), c.comparator)
            elif hasattr(c, 'target'):  # TRR
                return ('trr', c.target, c.ref1, c.ref2)
            elif hasattr(c, 'relation'):  # Axial or Topology
                return ('rel', c.obj1, c.obj2, str(c.relation))
            elif hasattr(c, 'bigger'):  # Size
                return ('size', c.bigger, c.smaller)
            elif hasattr(c, 'anchor'):  # Closer
                return ('closer', c.anchor, c.closer, c.farther)
            elif hasattr(c, 'occluder'):  # Occlusion
                return ('occ', c.occluder, c.occluded)
            return str(c)

        # Count constraint occurrences
        counts = {}
        for cs in constraint_sets:
            for c in cs.qrr + cs.trr + cs.axial + cs.topology + cs.size + cs.closer + cs.occlusion:
                key = constraint_key(c)
                if key not in counts:
                    counts[key] = (0, c)
                counts[key] = (counts[key][0] + 1, c)

        # Keep constraints that appear in all sets
        n_sets = len(constraint_sets)
        qrr, trr, axial, topology, size, closer, occlusion = [], [], [], [], [], [], []

        for key, (count, c) in counts.items():
            if count == n_sets:  # Appears in all sets
                if key[0] == 'qrr':
                    qrr.append(c)
                elif key[0] == 'trr':
                    trr.append(c)
                elif key[0] == 'rel':
                    if isinstance(c, AxialConstraint):
                        axial.append(c)
                    else:
                        topology.append(c)
                elif key[0] == 'size':
                    size.append(c)
                elif key[0] == 'closer':
                    closer.append(c)
                elif key[0] == 'occ':
                    occlusion.append(c)

        return ConstraintSet(
            objects=list(all_objects.values()),
            qrr=qrr,
            trr=trr,
            axial=axial,
            topology=topology,
            size=size,
            closer=closer,
            occlusion=occlusion,
            confidence=1.0,
            metadata={
                "source": "blender_merged",
                "n_views": len(constraint_sets),
            }
        )
