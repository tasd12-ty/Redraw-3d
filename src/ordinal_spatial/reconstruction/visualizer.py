"""
3D 场景可视化器。

使用 Matplotlib 和 Plotly 可视化重建的 3D 场景。

3D Scene Visualizer.

Visualizes reconstructed 3D scenes using Matplotlib and Plotly.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import logging

from .scene_builder import ReconstructedScene, ReconstructedObject, COLOR_MAP
from .shapes.primitives import get_mesh_for_shape

logger = logging.getLogger(__name__)


@dataclass
class VisualizerConfig:
    """
    可视化器配置。

    Visualizer configuration.
    """
    figsize: Tuple[int, int] = (10, 10)
    dpi: int = 150
    show_labels: bool = True
    show_axes: bool = True
    show_grid: bool = True
    alpha: float = 0.8
    label_fontsize: int = 10
    title_fontsize: int = 14
    background_color: str = "#f0f0f0"
    camera_elevation: float = 30.0
    camera_azimuth: float = 45.0


class SceneVisualizer:
    """
    3D 场景可视化器。

    3D Scene Visualizer.
    """

    def __init__(self, config: VisualizerConfig = None):
        """
        初始化可视化器。

        Initialize visualizer.

        Args:
            config: 可视化器配置
        """
        self.config = config or VisualizerConfig()

    def visualize_matplotlib(
        self,
        scene: ReconstructedScene,
        title: str = None,
        output_path: str = None,
        show: bool = False,
    ):
        """
        使用 Matplotlib 可视化场景。

        Visualize scene using Matplotlib.

        Args:
            scene: 重建的场景
            title: 图标题
            output_path: 输出路径（可选）
            show: 是否显示图像

        Returns:
            matplotlib Figure 对象
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError:
            raise ImportError(
                "matplotlib required for visualization. "
                "Install with: pip install matplotlib"
            )

        fig = plt.figure(figsize=self.config.figsize)
        ax = fig.add_subplot(111, projection='3d')

        # 设置背景
        ax.set_facecolor(self.config.background_color)

        # 绘制每个物体
        for obj in scene.objects:
            self._draw_object_matplotlib(ax, obj)

        # 设置轴标签
        if self.config.show_axes:
            ax.set_xlabel('X', fontsize=self.config.label_fontsize)
            ax.set_ylabel('Y', fontsize=self.config.label_fontsize)
            ax.set_zlabel('Z', fontsize=self.config.label_fontsize)

        # 设置标题
        if title is None:
            title = f"Reconstructed Scene: {scene.scene_id}"
            if scene.satisfaction_rate < 1.0:
                title += f" (Satisfaction: {scene.satisfaction_rate:.1%})"
        ax.set_title(title, fontsize=self.config.title_fontsize)

        # 设置视角
        ax.view_init(
            elev=self.config.camera_elevation,
            azim=self.config.camera_azimuth
        )

        # 设置轴范围
        bounds = scene.bounds
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_zlim(bounds)

        # 保持比例
        ax.set_box_aspect([1, 1, 1])

        # 网格
        ax.grid(self.config.show_grid)

        plt.tight_layout()

        # 保存
        if output_path:
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Saved visualization to: {output_path}")

        # 显示
        if show:
            plt.show()

        return fig

    def _draw_object_matplotlib(self, ax, obj: ReconstructedObject):
        """
        在 Matplotlib 3D 轴上绘制单个物体。

        Draw single object on Matplotlib 3D axes.
        """
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # 获取网格
        try:
            vertices, faces = get_mesh_for_shape(
                obj.shape,
                obj.position,
                obj.scale,
            )
        except Exception as e:
            logger.warning(f"Failed to get mesh for {obj.shape}: {e}")
            # 回退到点表示
            ax.scatter(
                obj.position[0],
                obj.position[1],
                obj.position[2],
                c=obj.color_hex,
                s=100 * obj.scale,
                label=obj.id,
            )
            return

        # 创建多边形集合
        mesh_faces = vertices[faces]
        collection = Poly3DCollection(
            mesh_faces,
            alpha=self.config.alpha,
            facecolor=obj.color_hex,
            edgecolor='black',
            linewidth=0.5,
        )
        ax.add_collection3d(collection)

        # 添加标签
        if self.config.show_labels:
            ax.text(
                obj.position[0],
                obj.position[1],
                obj.position[2] + obj.scale + 0.3,
                obj.id,
                fontsize=self.config.label_fontsize,
                ha='center',
            )

    def visualize_plotly(
        self,
        scene: ReconstructedScene,
        title: str = None,
        output_path: str = None,
        show: bool = False,
    ):
        """
        使用 Plotly 可视化场景（交互式）。

        Visualize scene using Plotly (interactive).

        Args:
            scene: 重建的场景
            title: 图标题
            output_path: 输出 HTML 路径（可选）
            show: 是否在浏览器中显示

        Returns:
            plotly Figure 对象
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "plotly required for interactive visualization. "
                "Install with: pip install plotly"
            )

        traces = []

        for obj in scene.objects:
            trace = self._create_object_trace_plotly(obj)
            if trace is not None:
                traces.append(trace)

        # 创建图形
        fig = go.Figure(data=traces)

        # 设置布局
        if title is None:
            title = f"Reconstructed Scene: {scene.scene_id}"
            if scene.satisfaction_rate < 1.0:
                title += f" (Satisfaction: {scene.satisfaction_rate:.1%})"

        bounds = scene.bounds

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(title='X', range=bounds),
                yaxis=dict(title='Y', range=bounds),
                zaxis=dict(title='Z', range=bounds),
                aspectmode='cube',
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
        )

        # 保存
        if output_path:
            if output_path.endswith('.html'):
                fig.write_html(output_path)
            else:
                fig.write_image(output_path)
            logger.info(f"Saved visualization to: {output_path}")

        # 显示
        if show:
            fig.show()

        return fig

    def _create_object_trace_plotly(self, obj: ReconstructedObject):
        """
        为 Plotly 创建物体轨迹。

        Create object trace for Plotly.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        try:
            vertices, faces = get_mesh_for_shape(
                obj.shape,
                obj.position,
                obj.scale,
            )
        except Exception as e:
            logger.warning(f"Failed to get mesh for {obj.shape}: {e}")
            # 回退到散点图
            return go.Scatter3d(
                x=[obj.position[0]],
                y=[obj.position[1]],
                z=[obj.position[2]],
                mode='markers',
                marker=dict(size=10, color=obj.color_hex),
                name=obj.id,
                text=f"{obj.id}: {obj.shape} ({obj.color})",
            )

        # 创建 Mesh3d
        return go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color=obj.color_hex,
            opacity=self.config.alpha,
            name=obj.id,
            hoverinfo='name+text',
            text=f"{obj.shape} ({obj.color}, {obj.size})",
        )

    def visualize_simple(
        self,
        scene: ReconstructedScene,
        title: str = None,
        output_path: str = None,
        show: bool = False,
    ):
        """
        简化的可视化（只使用点和标签）。

        Simplified visualization (points and labels only).

        Args:
            scene: 重建的场景
            title: 图标题
            output_path: 输出路径
            show: 是否显示

        Returns:
            matplotlib Figure 对象
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise ImportError("matplotlib required for visualization")

        fig = plt.figure(figsize=self.config.figsize)
        ax = fig.add_subplot(111, projection='3d')

        # 收集数据
        xs, ys, zs = [], [], []
        colors = []
        sizes = []
        labels = []

        for obj in scene.objects:
            xs.append(obj.position[0])
            ys.append(obj.position[1])
            zs.append(obj.position[2])
            colors.append(obj.color_hex)
            sizes.append(obj.scale * 500)  # 缩放到合适的显示大小
            labels.append(f"{obj.id}\n({obj.shape})")

        # 绘制点
        ax.scatter(xs, ys, zs, c=colors, s=sizes, alpha=self.config.alpha)

        # 添加标签
        if self.config.show_labels:
            for i, label in enumerate(labels):
                ax.text(
                    xs[i], ys[i], zs[i] + 0.5,
                    label,
                    fontsize=self.config.label_fontsize - 2,
                    ha='center',
                )

        # 设置
        if title is None:
            title = f"Scene: {scene.scene_id}"
        ax.set_title(title, fontsize=self.config.title_fontsize)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        bounds = scene.bounds
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_zlim(bounds)

        ax.view_init(elev=self.config.camera_elevation, azim=self.config.camera_azimuth)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def visualize_constraints(
        self,
        scene: ReconstructedScene,
        constraints_data: Dict[str, Any] = None,
        output_path: str = None,
        show: bool = False,
    ):
        """
        可视化场景及其约束关系。

        Visualize scene with constraint relations.

        Args:
            scene: 重建的场景
            constraints_data: 约束数据
            output_path: 输出路径
            show: 是否显示

        Returns:
            matplotlib Figure 对象
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
        except ImportError:
            raise ImportError("matplotlib required for visualization")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 绘制物体
        positions = {}
        for obj in scene.objects:
            positions[obj.id] = obj.position
            ax.scatter(
                obj.position[0],
                obj.position[1],
                obj.position[2],
                c=obj.color_hex,
                s=obj.scale * 500,
                alpha=self.config.alpha,
                label=obj.id,
            )

            if self.config.show_labels:
                ax.text(
                    obj.position[0],
                    obj.position[1],
                    obj.position[2] + 0.5,
                    obj.id,
                    fontsize=self.config.label_fontsize,
                    ha='center',
                )

        # 绘制约束关系线
        if constraints_data:
            # 绘制 QRR 约束
            for qrr in constraints_data.get("qrr", []):
                pair1 = qrr.get("pair1", [])
                pair2 = qrr.get("pair2", [])

                if len(pair1) == 2 and all(p in positions for p in pair1):
                    p1, p2 = positions[pair1[0]], positions[pair1[1]]
                    ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        'b--', alpha=0.3, linewidth=1
                    )

                if len(pair2) == 2 and all(p in positions for p in pair2):
                    p1, p2 = positions[pair2[0]], positions[pair2[1]]
                    ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        [p1[2], p2[2]],
                        'r--', alpha=0.3, linewidth=1
                    )

            # 绘制轴向约束
            for axial in constraints_data.get("axial", []):
                obj_a = axial.get("a") or axial.get("obj_a")
                obj_b = axial.get("b") or axial.get("obj_b")

                if obj_a in positions and obj_b in positions:
                    pa, pb = positions[obj_a], positions[obj_b]
                    ax.annotate3D(
                        '',
                        xyz=pb,
                        xytext=pa,
                        arrowprops=dict(arrowstyle='->', color='green', alpha=0.5),
                    ) if hasattr(ax, 'annotate3D') else None

        ax.set_title(f"Scene with Constraints: {scene.scene_id}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        bounds = scene.bounds
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_zlim(bounds)

        ax.view_init(elev=self.config.camera_elevation, azim=self.config.camera_azimuth)
        ax.legend(loc='upper left', fontsize=8)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=self.config.dpi, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    def save(self, fig, path: str, dpi: int = None):
        """
        保存图像。

        Save figure.

        Args:
            fig: matplotlib Figure 对象
            path: 输出路径
            dpi: 分辨率
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required")

        fig.savefig(path, dpi=dpi or self.config.dpi, bbox_inches='tight')
        logger.info(f"Saved figure to: {path}")


def visualize_scene(
    scene: ReconstructedScene,
    backend: str = "matplotlib",
    output_path: str = None,
    show: bool = False,
    **kwargs,
):
    """
    便捷可视化函数。

    Convenience visualization function.

    Args:
        scene: 重建的场景
        backend: 后端选择 ("matplotlib" 或 "plotly")
        output_path: 输出路径
        show: 是否显示
        **kwargs: 其他参数

    Returns:
        Figure 对象
    """
    config = VisualizerConfig(**kwargs) if kwargs else None
    visualizer = SceneVisualizer(config)

    if backend == "plotly":
        return visualizer.visualize_plotly(scene, output_path=output_path, show=show)
    else:
        return visualizer.visualize_matplotlib(scene, output_path=output_path, show=show)
