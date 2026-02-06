"""
Task-3: 从相对约束 DSL 复原场景图模块。

本模块实现从形式化约束语言重建 3D 场景的完整流程：
1. DSL 解析器：解析 JSON 约束到内部数据结构
2. 约束求解器：使用 PyTorch 自动微分求解 3D 坐标
3. 场景重建器：将求解结果转化为 OrdinalSceneDescription
4. 3D 可视化器：使用 Matplotlib/Plotly 可视化场景

Task-3: Reconstruct 3D scene from relative constraint DSL.

This module implements the complete pipeline for reconstructing 3D scenes
from formal constraint language:
1. DSL Parser: Parse JSON constraints to internal data structures
2. Constraint Solver: Use PyTorch autograd to solve for 3D coordinates
3. Scene Builder: Convert solver results to OrdinalSceneDescription
4. 3D Visualizer: Visualize scenes using Matplotlib/Plotly
"""

from .dsl_parser import (
    DSLParser,
    ParsedConstraints,
    AxialConstraint,
    AxialRelation,
)
from .constraint_solver import (
    ConstraintSolver,
    GradientDescentSolver,
    NumpyGradientSolver,
    SolverResult,
    SolverConfig,
    create_solver,
    TORCH_AVAILABLE,
)
from .scene_builder import (
    SceneBuilder,
    SceneConfig,
)
from .visualizer import (
    SceneVisualizer,
    VisualizerConfig,
)

# 便捷函数
from .pipeline import reconstruct_and_visualize

__all__ = [
    # Parser
    "DSLParser",
    "ParsedConstraints",
    "AxialConstraint",
    "AxialRelation",
    # Solver
    "ConstraintSolver",
    "GradientDescentSolver",
    "NumpyGradientSolver",
    "SolverResult",
    "SolverConfig",
    "create_solver",
    "TORCH_AVAILABLE",
    # Builder
    "SceneBuilder",
    "SceneConfig",
    # Visualizer
    "SceneVisualizer",
    "VisualizerConfig",
    # Pipeline
    "reconstruct_and_visualize",
]
