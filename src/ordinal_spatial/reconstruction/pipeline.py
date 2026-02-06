"""
重建流水线 - 端到端便捷函数。

Reconstruction Pipeline - End-to-end convenience functions.
"""

from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

from .dsl_parser import DSLParser, ParsedConstraints
from .constraint_solver import (
    GradientDescentSolver,
    NumpyGradientSolver,
    SolverConfig,
    SolverResult,
    TORCH_AVAILABLE,
)
from .scene_builder import SceneBuilder, SceneConfig, ReconstructedScene
from .visualizer import SceneVisualizer, VisualizerConfig

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """
    流水线配置。

    Pipeline configuration.
    """
    # 求解器配置
    solver_type: str = "pytorch"  # "pytorch" 或 "numpy"
    n_dims: int = 3
    learning_rate: float = 0.05
    max_iterations: int = 2000
    tau: float = 0.10
    use_gpu: bool = False

    # 场景配置
    scene_bounds: tuple = (-5.0, 5.0)
    normalize: bool = True

    # 可视化配置
    visualizer: str = "matplotlib"  # "matplotlib", "plotly", "simple"
    figsize: tuple = (10, 10)
    show_labels: bool = True
    alpha: float = 0.8


@dataclass
class PipelineResult:
    """
    流水线结果。

    Pipeline result.
    """
    scene: ReconstructedScene
    solver_result: SolverResult
    constraints: ParsedConstraints
    figure: Any = None
    output_paths: Dict[str, str] = field(default_factory=dict)

    @property
    def satisfaction_rate(self) -> float:
        """约束满足率。"""
        return self.solver_result.satisfaction_rate

    @property
    def positions(self) -> Dict:
        """物体位置。"""
        return self.solver_result.positions

    def summary(self) -> str:
        """生成摘要。"""
        lines = [
            f"=== Reconstruction Result ===",
            f"Scene ID: {self.scene.scene_id}",
            f"Objects: {len(self.scene.objects)}",
            f"Constraints: {self.constraints.n_constraints()}",
            f"Satisfaction Rate: {self.satisfaction_rate:.1%}",
            f"Iterations: {self.solver_result.iterations}",
            f"Final Loss: {self.solver_result.final_loss:.6f}",
        ]

        if self.output_paths:
            lines.append("\nOutputs:")
            for name, path in self.output_paths.items():
                lines.append(f"  - {name}: {path}")

        return "\n".join(lines)


def reconstruct_and_visualize(
    constraints_input: Union[str, Dict[str, Any]],
    output_dir: str = None,
    config: PipelineConfig = None,
    show: bool = False,
) -> PipelineResult:
    """
    完整的重建和可视化流水线。

    Complete reconstruction and visualization pipeline.

    Args:
        constraints_input: 约束输入（文件路径或 JSON 字典）
        output_dir: 输出目录（可选）
        config: 流水线配置
        show: 是否显示可视化结果

    Returns:
        PipelineResult 对象
    """
    config = config or PipelineConfig()
    output_paths = {}

    # 1. 解析约束
    logger.info("Step 1: Parsing constraints...")
    parser = DSLParser()

    if isinstance(constraints_input, str):
        constraints = parser.parse_file(constraints_input)
    else:
        constraints = parser.parse_json(constraints_input)

    logger.info(f"  - Objects: {constraints.n_objects()}")
    logger.info(f"  - Constraints: {constraints.n_constraints()}")

    # 2. 创建求解器
    logger.info("Step 2: Solving constraints...")
    solver_config = SolverConfig(
        n_dims=config.n_dims,
        learning_rate=config.learning_rate,
        max_iterations=config.max_iterations,
        tau=config.tau or constraints.tau,
        use_gpu=config.use_gpu,
    )

    if config.solver_type == "pytorch" and TORCH_AVAILABLE:
        solver = GradientDescentSolver(solver_config)
    else:
        if config.solver_type == "pytorch":
            logger.warning("PyTorch not available, using NumPy solver")
        solver = NumpyGradientSolver(solver_config)

    # 3. 求解
    solver_result = solver.solve(constraints)
    logger.info(f"  - Satisfaction Rate: {solver_result.satisfaction_rate:.1%}")
    logger.info(f"  - Iterations: {solver_result.iterations}")
    logger.info(f"  - Final Loss: {solver_result.final_loss:.6f}")

    # 4. 构建场景
    logger.info("Step 3: Building scene...")
    scene_config = SceneConfig(
        scene_bounds=config.scene_bounds,
        normalize=config.normalize,
    )
    builder = SceneBuilder(scene_config)
    scene = builder.build(solver_result, constraints)

    # 5. 可视化
    figure = None
    if output_dir or show:
        logger.info("Step 4: Visualizing...")
        vis_config = VisualizerConfig(
            figsize=config.figsize,
            show_labels=config.show_labels,
            alpha=config.alpha,
        )
        visualizer = SceneVisualizer(vis_config)

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存场景 JSON
            scene_path = output_dir / "scene.json"
            scene.save(str(scene_path))
            output_paths["scene_json"] = str(scene_path)

            # 保存可视化
            if config.visualizer == "plotly":
                vis_path = output_dir / "scene.html"
                figure = visualizer.visualize_plotly(
                    scene, output_path=str(vis_path), show=show
                )
                output_paths["visualization"] = str(vis_path)
            elif config.visualizer == "simple":
                vis_path = output_dir / "scene.png"
                figure = visualizer.visualize_simple(
                    scene, output_path=str(vis_path), show=show
                )
                output_paths["visualization"] = str(vis_path)
            else:
                vis_path = output_dir / "scene.png"
                figure = visualizer.visualize_matplotlib(
                    scene, output_path=str(vis_path), show=show
                )
                output_paths["visualization"] = str(vis_path)

            logger.info(f"  - Saved to: {output_dir}")

        elif show:
            if config.visualizer == "plotly":
                figure = visualizer.visualize_plotly(scene, show=True)
            elif config.visualizer == "simple":
                figure = visualizer.visualize_simple(scene, show=True)
            else:
                figure = visualizer.visualize_matplotlib(scene, show=True)

    logger.info("Done!")

    return PipelineResult(
        scene=scene,
        solver_result=solver_result,
        constraints=constraints,
        figure=figure,
        output_paths=output_paths,
    )


def reconstruct_from_json(
    constraints_json: Dict[str, Any],
    **kwargs,
) -> ReconstructedScene:
    """
    从 JSON 快速重建场景。

    Quick scene reconstruction from JSON.

    Args:
        constraints_json: 约束 JSON 数据
        **kwargs: 其他配置参数

    Returns:
        ReconstructedScene 对象
    """
    result = reconstruct_and_visualize(constraints_json, **kwargs)
    return result.scene


def reconstruct_from_file(
    constraints_path: str,
    output_dir: str = None,
    **kwargs,
) -> PipelineResult:
    """
    从文件重建场景。

    Reconstruct scene from file.

    Args:
        constraints_path: 约束文件路径
        output_dir: 输出目录
        **kwargs: 其他配置参数

    Returns:
        PipelineResult 对象
    """
    return reconstruct_and_visualize(
        constraints_path,
        output_dir=output_dir,
        **kwargs,
    )
