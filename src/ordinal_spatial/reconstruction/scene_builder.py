"""
场景重建器。

将约束求解器的结果转化为完整的场景描述。

Scene Builder.

Converts constraint solver results into complete scene descriptions.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import json

from .dsl_parser import ParsedConstraints, ObjectInfo
from .constraint_solver import SolverResult


@dataclass
class SceneConfig:
    """
    场景配置。

    Scene configuration.
    """
    scene_bounds: Tuple[float, float] = (-5.0, 5.0)
    normalize: bool = True
    center_scene: bool = True


# 尺寸映射
SIZE_SCALE_MAP = {
    "tiny": 0.25,
    "small": 0.35,
    "medium": 0.5,
    "large": 0.7,
    "custom": 0.5,
}

# 颜色映射 (RGB hex)
COLOR_MAP = {
    "red": "#e74c3c",
    "blue": "#3498db",
    "green": "#2ecc71",
    "yellow": "#f1c40f",
    "purple": "#9b59b6",
    "cyan": "#1abc9c",
    "orange": "#e67e22",
    "pink": "#fd79a8",
    "gray": "#95a5a6",
    "brown": "#8B4513",
    "white": "#ffffff",
    "black": "#2c3e50",
}


@dataclass
class ReconstructedObject:
    """
    重建后的物体。

    Reconstructed object.
    """
    id: str
    shape: str
    color: str
    size: str
    material: str
    position: np.ndarray
    scale: float
    color_hex: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "id": self.id,
            "shape": self.shape,
            "color": self.color,
            "size": self.size,
            "material": self.material,
            "position_3d": self.position.tolist(),
            "scale": self.scale,
            "color_hex": self.color_hex,
        }


@dataclass
class ReconstructedScene:
    """
    重建后的场景。

    Reconstructed scene.
    """
    scene_id: str
    objects: List[ReconstructedObject]
    bounds: Tuple[float, float]
    satisfaction_rate: float
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典。"""
        return {
            "scene_id": self.scene_id,
            "objects": [obj.to_dict() for obj in self.objects],
            "bounds": self.bounds,
            "satisfaction_rate": self.satisfaction_rate,
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """转换为 JSON 字符串。"""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str):
        """保存到文件。"""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def get_object(self, obj_id: str) -> Optional[ReconstructedObject]:
        """通过 ID 获取物体。"""
        for obj in self.objects:
            if obj.id == obj_id:
                return obj
        return None


class SceneBuilder:
    """
    场景重建器。

    Scene builder.
    """

    def __init__(self, config: SceneConfig = None):
        """
        初始化场景重建器。

        Initialize scene builder.

        Args:
            config: 场景配置
        """
        self.config = config or SceneConfig()

    def build(
        self,
        solver_result: SolverResult,
        constraints: ParsedConstraints,
        scene_id: str = None,
    ) -> ReconstructedScene:
        """
        从求解结果构建场景。

        Build scene from solver result.

        Args:
            solver_result: 求解器结果
            constraints: 原始约束
            scene_id: 场景 ID

        Returns:
            ReconstructedScene 对象
        """
        positions = solver_result.positions

        # 归一化位置
        if self.config.normalize and positions:
            positions = self._normalize_positions(positions)

        # 构建重建物体列表
        reconstructed_objects = []

        for obj_info in constraints.objects:
            pos = positions.get(obj_info.id)
            if pos is None:
                # 如果没有求解到位置，使用随机位置
                pos = np.random.uniform(
                    self.config.scene_bounds[0],
                    self.config.scene_bounds[1],
                    size=3
                )

            # 获取尺寸比例
            scale = self._get_size_scale(obj_info.size)

            # 获取颜色 hex
            color_hex = COLOR_MAP.get(obj_info.color.lower(), "#888888")

            reconstructed_obj = ReconstructedObject(
                id=obj_info.id,
                shape=obj_info.shape,
                color=obj_info.color,
                size=obj_info.size,
                material=obj_info.material,
                position=np.array(pos),
                scale=scale,
                color_hex=color_hex,
            )
            reconstructed_objects.append(reconstructed_obj)

        return ReconstructedScene(
            scene_id=scene_id or constraints.scene_id,
            objects=reconstructed_objects,
            bounds=self.config.scene_bounds,
            satisfaction_rate=solver_result.satisfaction_rate,
            metadata={
                "iterations": solver_result.iterations,
                "final_loss": solver_result.final_loss,
                "n_objects": len(reconstructed_objects),
                "n_constraints": constraints.n_constraints(),
            },
        )

    def _normalize_positions(
        self,
        positions: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """
        归一化位置到场景边界。

        Normalize positions to scene bounds.

        Args:
            positions: 原始位置字典

        Returns:
            归一化后的位置字典
        """
        if not positions:
            return positions

        # 转换为数组
        pos_array = np.array(list(positions.values()))

        # 中心化
        if self.config.center_scene:
            centroid = pos_array.mean(axis=0)
            pos_array = pos_array - centroid

        # 缩放到边界
        max_extent = np.abs(pos_array).max()
        if max_extent > 0:
            target_extent = (self.config.scene_bounds[1] - self.config.scene_bounds[0]) / 2 * 0.8
            scale = target_extent / max_extent
            pos_array = pos_array * scale

        # 重建字典
        return {
            obj_id: pos_array[i]
            for i, obj_id in enumerate(positions.keys())
        }

    def _get_size_scale(self, size) -> float:
        """
        获取尺寸比例值。

        Get size scale value.
        """
        if isinstance(size, str):
            return SIZE_SCALE_MAP.get(size.lower(), 0.5)
        elif isinstance(size, (int, float)):
            return float(size)
        return 0.5

    def build_from_positions(
        self,
        positions: Dict[str, np.ndarray],
        object_infos: List[ObjectInfo] = None,
        scene_id: str = "reconstructed",
    ) -> ReconstructedScene:
        """
        直接从位置字典构建场景。

        Build scene directly from positions dictionary.

        Args:
            positions: 位置字典
            object_infos: 物体信息列表（可选）
            scene_id: 场景 ID

        Returns:
            ReconstructedScene 对象
        """
        # 归一化
        if self.config.normalize:
            positions = self._normalize_positions(positions)

        # 构建物体信息映射
        info_map = {}
        if object_infos:
            info_map = {obj.id: obj for obj in object_infos}

        # 构建物体列表
        reconstructed_objects = []

        for obj_id, pos in positions.items():
            info = info_map.get(obj_id)

            if info:
                shape = info.shape
                color = info.color
                size = info.size
                material = info.material
            else:
                # 默认属性
                shape = "cube"
                color = "gray"
                size = "medium"
                material = "rubber"

            scale = self._get_size_scale(size)
            color_hex = COLOR_MAP.get(color.lower(), "#888888")

            reconstructed_obj = ReconstructedObject(
                id=obj_id,
                shape=shape,
                color=color,
                size=size,
                material=material,
                position=np.array(pos),
                scale=scale,
                color_hex=color_hex,
            )
            reconstructed_objects.append(reconstructed_obj)

        return ReconstructedScene(
            scene_id=scene_id,
            objects=reconstructed_objects,
            bounds=self.config.scene_bounds,
            satisfaction_rate=1.0,
            metadata={
                "n_objects": len(reconstructed_objects),
            },
        )


def build_scene_from_constraints(
    constraints_json: Dict[str, Any],
    solver_result: SolverResult = None,
) -> ReconstructedScene:
    """
    从约束 JSON 直接构建场景的便捷函数。

    Convenience function to build scene directly from constraints JSON.

    Args:
        constraints_json: 约束 JSON 数据
        solver_result: 求解器结果（可选，如果为 None 则执行求解）

    Returns:
        ReconstructedScene 对象
    """
    from .dsl_parser import DSLParser
    from .constraint_solver import create_solver, SolverConfig

    # 解析约束
    parser = DSLParser()
    constraints = parser.parse_json(constraints_json)

    # 如果没有求解结果，执行求解
    if solver_result is None:
        solver = create_solver(SolverConfig(tau=constraints.tau))
        solver_result = solver.solve(constraints)

    # 构建场景
    builder = SceneBuilder()
    return builder.build(solver_result, constraints)
