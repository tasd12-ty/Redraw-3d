"""
重建模块单元测试。

Unit tests for reconstruction module.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path

from ordinal_spatial.reconstruction.dsl_parser import (
    DSLParser,
    ParsedConstraints,
    AxialConstraint,
    AxialRelation,
    QRRConstraint,
    Comparator,
    MetricType,
)
from ordinal_spatial.reconstruction.constraint_solver import (
    GradientDescentSolver,
    NumpyGradientSolver,
    SolverConfig,
    SolverResult,
    TORCH_AVAILABLE,
)
from ordinal_spatial.reconstruction.scene_builder import (
    SceneBuilder,
    SceneConfig,
    ReconstructedScene,
)
from ordinal_spatial.reconstruction.visualizer import (
    SceneVisualizer,
    VisualizerConfig,
)
from ordinal_spatial.reconstruction.pipeline import (
    reconstruct_and_visualize,
    PipelineConfig,
)
from ordinal_spatial.reconstruction.shapes.primitives import (
    sphere_mesh,
    cube_mesh,
    cylinder_mesh,
    get_mesh_for_shape,
)


# ============================================================================
# Test Data
# ============================================================================

SIMPLE_CONSTRAINTS = {
    "scene_id": "test_simple",
    "tau": 0.10,
    "objects": [
        {"id": "A", "shape": "cube", "color": "red", "size": "medium"},
        {"id": "B", "shape": "sphere", "color": "blue", "size": "medium"},
        {"id": "C", "shape": "cylinder", "color": "green", "size": "small"},
    ],
    "constraints": {
        "axial": [
            {"a": "A", "b": "B", "axis": "x", "relation": "left_of"},
            {"a": "B", "b": "C", "axis": "x", "relation": "left_of"},
        ],
    },
}

COMPLEX_CONSTRAINTS = {
    "scene_id": "test_complex",
    "tau": 0.10,
    "objects": [
        {"id": "obj_1", "shape": "cube", "color": "red", "size": "large"},
        {"id": "obj_2", "shape": "sphere", "color": "blue", "size": "medium"},
        {"id": "obj_3", "shape": "cylinder", "color": "green", "size": "small"},
        {"id": "obj_4", "shape": "cone", "color": "yellow", "size": "medium"},
    ],
    "constraints": {
        "qrr": [
            {
                "pair1": ["obj_1", "obj_2"],
                "pair2": ["obj_3", "obj_4"],
                "metric": "DIST_3D",
                "comparator": "<",
            },
        ],
        "axial": [
            {"a": "obj_1", "b": "obj_2", "axis": "x", "relation": "left_of"},
            {"a": "obj_2", "b": "obj_3", "axis": "x", "relation": "left_of"},
            {"a": "obj_3", "b": "obj_4", "axis": "x", "relation": "left_of"},
            {"a": "obj_1", "b": "obj_3", "axis": "z", "relation": "below"},
        ],
        "topology": [
            {"obj1": "obj_1", "obj2": "obj_2", "relation": "disjoint"},
        ],
    },
}


# ============================================================================
# DSL Parser Tests
# ============================================================================

class TestDSLParser:
    """DSL 解析器测试。"""

    def test_parse_simple_constraints(self):
        """测试解析简单约束。"""
        parser = DSLParser()
        constraints = parser.parse_json(SIMPLE_CONSTRAINTS)

        assert constraints.scene_id == "test_simple"
        assert constraints.tau == 0.10
        assert len(constraints.objects) == 3
        assert len(constraints.axial) == 2

    def test_parse_complex_constraints(self):
        """测试解析复杂约束。"""
        parser = DSLParser()
        constraints = parser.parse_json(COMPLEX_CONSTRAINTS)

        assert constraints.scene_id == "test_complex"
        assert len(constraints.objects) == 4
        assert len(constraints.qrr) == 1
        assert len(constraints.axial) == 4
        assert len(constraints.topology) == 1

    def test_parse_qrr_constraint(self):
        """测试解析 QRR 约束。"""
        parser = DSLParser()
        constraints = parser.parse_json(COMPLEX_CONSTRAINTS)

        qrr = constraints.qrr[0]
        assert qrr.pair1 == ("obj_1", "obj_2")
        assert qrr.pair2 == ("obj_3", "obj_4")
        assert qrr.metric == MetricType.DIST_3D
        assert qrr.comparator == Comparator.LT

    def test_parse_axial_constraint(self):
        """测试解析轴向约束。"""
        parser = DSLParser()
        constraints = parser.parse_json(SIMPLE_CONSTRAINTS)

        axial = constraints.axial[0]
        assert axial.obj_a == "A"
        assert axial.obj_b == "B"
        assert axial.axis == "x"
        assert axial.relation == AxialRelation.LEFT_OF

    def test_validate_constraints(self):
        """测试约束验证。"""
        parser = DSLParser()
        constraints = parser.parse_json(SIMPLE_CONSTRAINTS)

        result = parser.validate(constraints)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_parse_from_file(self, tmp_path):
        """测试从文件解析。"""
        # 写入临时文件
        file_path = tmp_path / "constraints.json"
        with open(file_path, "w") as f:
            json.dump(SIMPLE_CONSTRAINTS, f)

        # 解析
        parser = DSLParser()
        constraints = parser.parse_file(str(file_path))

        assert constraints.scene_id == "test_simple"
        assert len(constraints.objects) == 3

    def test_get_object_ids(self):
        """测试获取物体 ID。"""
        parser = DSLParser()
        constraints = parser.parse_json(SIMPLE_CONSTRAINTS)

        ids = constraints.get_object_ids()
        assert ids == ["A", "B", "C"]

    def test_n_constraints(self):
        """测试约束计数。"""
        parser = DSLParser()
        constraints = parser.parse_json(COMPLEX_CONSTRAINTS)

        assert constraints.n_constraints() == 6  # 1 qrr + 4 axial + 1 topology


# ============================================================================
# Constraint Solver Tests
# ============================================================================

class TestConstraintSolver:
    """约束求解器测试。"""

    @pytest.fixture
    def simple_constraints(self):
        """简单约束 fixture。"""
        parser = DSLParser()
        return parser.parse_json(SIMPLE_CONSTRAINTS)

    @pytest.fixture
    def complex_constraints(self):
        """复杂约束 fixture。"""
        parser = DSLParser()
        return parser.parse_json(COMPLEX_CONSTRAINTS)

    def test_numpy_solver_basic(self, simple_constraints):
        """测试 NumPy 求解器基本功能。"""
        config = SolverConfig(
            max_iterations=500,
            learning_rate=0.05,
        )
        solver = NumpyGradientSolver(config)
        result = solver.solve(simple_constraints)

        assert result.satisfiable
        assert len(result.positions) == 3
        assert "A" in result.positions
        assert "B" in result.positions
        assert "C" in result.positions

    def test_numpy_solver_axial_satisfaction(self, simple_constraints):
        """测试 NumPy 求解器满足轴向约束。"""
        config = SolverConfig(max_iterations=1000)
        solver = NumpyGradientSolver(config)
        result = solver.solve(simple_constraints)

        # A 应该在 B 左侧 (A.x < B.x)
        assert result.positions["A"][0] < result.positions["B"][0]
        # B 应该在 C 左侧 (B.x < C.x)
        assert result.positions["B"][0] < result.positions["C"][0]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_pytorch_solver_basic(self, simple_constraints):
        """测试 PyTorch 求解器基本功能。"""
        config = SolverConfig(max_iterations=500)
        solver = GradientDescentSolver(config)
        result = solver.solve(simple_constraints)

        assert result.satisfiable
        assert len(result.positions) == 3
        assert result.satisfaction_rate > 0.5

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_pytorch_solver_axial_satisfaction(self, simple_constraints):
        """测试 PyTorch 求解器满足轴向约束。"""
        config = SolverConfig(max_iterations=1000)
        solver = GradientDescentSolver(config)
        result = solver.solve(simple_constraints)

        # 检查轴向约束
        assert result.positions["A"][0] < result.positions["B"][0]
        assert result.positions["B"][0] < result.positions["C"][0]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_pytorch_solver_complex(self, complex_constraints):
        """测试 PyTorch 求解器处理复杂约束。"""
        config = SolverConfig(max_iterations=2000)
        solver = GradientDescentSolver(config)
        result = solver.solve(complex_constraints)

        assert result.satisfiable
        assert len(result.positions) == 4
        assert result.satisfaction_rate > 0.7

    def test_solver_empty_constraints(self):
        """测试空约束处理。"""
        empty = {
            "scene_id": "empty",
            "objects": [],
            "constraints": {},
        }
        parser = DSLParser()
        constraints = parser.parse_json(empty)

        solver = NumpyGradientSolver()
        result = solver.solve(constraints)

        assert not result.satisfiable
        assert result.error is not None


# ============================================================================
# Scene Builder Tests
# ============================================================================

class TestSceneBuilder:
    """场景重建器测试。"""

    def test_build_scene(self):
        """测试构建场景。"""
        parser = DSLParser()
        constraints = parser.parse_json(SIMPLE_CONSTRAINTS)

        solver = NumpyGradientSolver(SolverConfig(max_iterations=500))
        solver_result = solver.solve(constraints)

        builder = SceneBuilder()
        scene = builder.build(solver_result, constraints)

        assert scene.scene_id == "test_simple"
        assert len(scene.objects) == 3
        assert scene.satisfaction_rate > 0

    def test_scene_normalization(self):
        """测试坐标归一化。"""
        builder = SceneBuilder(SceneConfig(
            scene_bounds=(-5.0, 5.0),
            normalize=True,
        ))

        positions = {
            "A": np.array([100.0, 200.0, 0.0]),
            "B": np.array([300.0, 400.0, 0.0]),
        }

        normalized = builder._normalize_positions(positions)

        # 检查归一化后的坐标在边界内
        for pos in normalized.values():
            assert np.all(np.abs(pos) <= 5.0)

    def test_scene_to_dict(self):
        """测试场景转字典。"""
        parser = DSLParser()
        constraints = parser.parse_json(SIMPLE_CONSTRAINTS)

        solver = NumpyGradientSolver(SolverConfig(max_iterations=500))
        solver_result = solver.solve(constraints)

        builder = SceneBuilder()
        scene = builder.build(solver_result, constraints)

        scene_dict = scene.to_dict()

        assert "scene_id" in scene_dict
        assert "objects" in scene_dict
        assert len(scene_dict["objects"]) == 3

    def test_scene_save_load(self, tmp_path):
        """测试场景保存和加载。"""
        parser = DSLParser()
        constraints = parser.parse_json(SIMPLE_CONSTRAINTS)

        solver = NumpyGradientSolver(SolverConfig(max_iterations=500))
        solver_result = solver.solve(constraints)

        builder = SceneBuilder()
        scene = builder.build(solver_result, constraints)

        # 保存
        save_path = tmp_path / "scene.json"
        scene.save(str(save_path))

        assert save_path.exists()

        # 加载并验证
        with open(save_path) as f:
            loaded = json.load(f)

        assert loaded["scene_id"] == scene.scene_id
        assert len(loaded["objects"]) == len(scene.objects)


# ============================================================================
# Visualizer Tests
# ============================================================================

class TestVisualizer:
    """可视化器测试。"""

    @pytest.fixture
    def sample_scene(self):
        """示例场景 fixture。"""
        parser = DSLParser()
        constraints = parser.parse_json(SIMPLE_CONSTRAINTS)

        solver = NumpyGradientSolver(SolverConfig(max_iterations=500))
        solver_result = solver.solve(constraints)

        builder = SceneBuilder()
        return builder.build(solver_result, constraints)

    def test_matplotlib_visualization(self, sample_scene, tmp_path):
        """测试 Matplotlib 可视化。"""
        pytest.importorskip("matplotlib")

        visualizer = SceneVisualizer()
        output_path = tmp_path / "scene.png"

        fig = visualizer.visualize_matplotlib(
            sample_scene,
            output_path=str(output_path),
        )

        assert fig is not None
        assert output_path.exists()

    def test_simple_visualization(self, sample_scene, tmp_path):
        """测试简化可视化。"""
        pytest.importorskip("matplotlib")

        visualizer = SceneVisualizer()
        output_path = tmp_path / "simple.png"

        fig = visualizer.visualize_simple(
            sample_scene,
            output_path=str(output_path),
        )

        assert fig is not None
        assert output_path.exists()

    def test_plotly_visualization(self, sample_scene, tmp_path):
        """测试 Plotly 可视化。"""
        pytest.importorskip("plotly")

        visualizer = SceneVisualizer()
        output_path = tmp_path / "scene.html"

        fig = visualizer.visualize_plotly(
            sample_scene,
            output_path=str(output_path),
        )

        assert fig is not None
        assert output_path.exists()


# ============================================================================
# Shape Primitives Tests
# ============================================================================

class TestShapePrimitives:
    """形状基元测试。"""

    def test_sphere_mesh(self):
        """测试球体网格生成。"""
        center = np.array([0.0, 0.0, 0.0])
        vertices, faces = sphere_mesh(center, radius=1.0, resolution=10)

        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3
        assert len(vertices) > 0
        assert len(faces) > 0

    def test_cube_mesh(self):
        """测试立方体网格生成。"""
        center = np.array([0.0, 0.0, 0.0])
        vertices, faces = cube_mesh(center, size=1.0)

        assert vertices.shape == (8, 3)
        assert faces.shape == (12, 3)

    def test_cylinder_mesh(self):
        """测试圆柱体网格生成。"""
        center = np.array([0.0, 0.0, 0.0])
        vertices, faces = cylinder_mesh(center, radius=0.5, height=1.0, resolution=10)

        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3

    def test_get_mesh_for_shape(self):
        """测试形状选择器。"""
        center = np.array([1.0, 2.0, 3.0])

        for shape in ["sphere", "cube", "cylinder", "cone", "pyramid"]:
            vertices, faces = get_mesh_for_shape(shape, center, size=0.5)
            assert vertices.shape[1] == 3
            assert faces.shape[1] == 3


# ============================================================================
# Pipeline Tests
# ============================================================================

class TestPipeline:
    """流水线测试。"""

    def test_full_pipeline(self, tmp_path):
        """测试完整流水线。"""
        output_dir = tmp_path / "output"

        config = PipelineConfig(
            solver_type="numpy",
            max_iterations=500,
            visualizer="simple",
        )

        result = reconstruct_and_visualize(
            constraints_input=SIMPLE_CONSTRAINTS,
            output_dir=str(output_dir),
            config=config,
        )

        assert result.scene is not None
        assert result.solver_result is not None
        assert result.satisfaction_rate > 0
        assert (output_dir / "scene.json").exists()
        assert (output_dir / "scene.png").exists()

    def test_pipeline_summary(self, tmp_path):
        """测试流水线摘要。"""
        config = PipelineConfig(
            solver_type="numpy",
            max_iterations=500,
        )

        result = reconstruct_and_visualize(
            constraints_input=SIMPLE_CONSTRAINTS,
            config=config,
        )

        summary = result.summary()
        assert "Scene ID" in summary
        assert "Objects" in summary
        assert "Satisfaction Rate" in summary

    def test_pipeline_from_file(self, tmp_path):
        """测试从文件运行流水线。"""
        # 写入约束文件
        constraints_path = tmp_path / "constraints.json"
        with open(constraints_path, "w") as f:
            json.dump(COMPLEX_CONSTRAINTS, f)

        output_dir = tmp_path / "output"

        config = PipelineConfig(
            solver_type="numpy",
            max_iterations=1000,
        )

        result = reconstruct_and_visualize(
            constraints_input=str(constraints_path),
            output_dir=str(output_dir),
            config=config,
        )

        assert result.scene.scene_id == "test_complex"
        assert len(result.scene.objects) == 4


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """集成测试。"""

    def test_end_to_end_simple(self):
        """端到端简单测试。"""
        # 解析
        parser = DSLParser()
        constraints = parser.parse_json(SIMPLE_CONSTRAINTS)

        # 求解
        solver = NumpyGradientSolver(SolverConfig(max_iterations=1000))
        solver_result = solver.solve(constraints)

        # 构建
        builder = SceneBuilder()
        scene = builder.build(solver_result, constraints)

        # 验证
        assert scene.scene_id == "test_simple"
        assert len(scene.objects) == 3

        # 验证约束满足
        positions = {obj.id: obj.position for obj in scene.objects}
        assert positions["A"][0] < positions["B"][0]
        assert positions["B"][0] < positions["C"][0]

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_end_to_end_pytorch(self):
        """端到端 PyTorch 测试。"""
        parser = DSLParser()
        constraints = parser.parse_json(COMPLEX_CONSTRAINTS)

        solver = GradientDescentSolver(SolverConfig(max_iterations=2000))
        solver_result = solver.solve(constraints)

        builder = SceneBuilder()
        scene = builder.build(solver_result, constraints)

        assert len(scene.objects) == 4
        assert solver_result.satisfaction_rate > 0.7

    def test_multiple_runs_consistency(self):
        """测试多次运行的一致性。"""
        parser = DSLParser()
        constraints = parser.parse_json(SIMPLE_CONSTRAINTS)

        config = SolverConfig(
            max_iterations=500,
            random_seed=42,
        )

        solver1 = NumpyGradientSolver(config)
        result1 = solver1.solve(constraints)

        solver2 = NumpyGradientSolver(config)
        result2 = solver2.solve(constraints)

        # 相同的随机种子应该产生相同的结果
        for obj_id in result1.positions:
            np.testing.assert_array_almost_equal(
                result1.positions[obj_id],
                result2.positions[obj_id],
                decimal=5,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
