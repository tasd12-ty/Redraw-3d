"""
约束求解器 - 使用 PyTorch 自动微分。

从相对约束集合求解满足约束的 3D 坐标。

Constraint Solver - Using PyTorch Autograd.

Solves for 3D coordinates that satisfy relative constraints.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
import logging
import warnings

logger = logging.getLogger(__name__)

# 检测 PyTorch 可用性（包括与 NumPy 的兼容性）
TORCH_AVAILABLE = False
torch = None

try:
    import torch as _torch
    import torch.nn as _nn
    import torch.optim as _optim

    # 测试 PyTorch 与 NumPy 的兼容性
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _test_tensor = _torch.tensor([1.0, 2.0])
        try:
            _test_numpy = _test_tensor.numpy()
            TORCH_AVAILABLE = True
            torch = _torch
            nn = _nn
            optim = _optim
        except RuntimeError:
            # PyTorch 与 NumPy 版本不兼容
            TORCH_AVAILABLE = False
            logger.warning(
                "PyTorch is installed but incompatible with NumPy version. "
                "Using NumPy solver instead."
            )
        finally:
            del _test_tensor
except ImportError:
    TORCH_AVAILABLE = False

from .dsl_parser import (
    ParsedConstraints,
    QRRConstraint,
    TRRConstraint,
    AxialConstraint,
    TopologyConstraint,
    AxialRelation,
    Comparator,
    MetricType,
)

logger = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    """
    求解器配置。

    Solver configuration.
    """
    n_dims: int = 3
    learning_rate: float = 0.05
    max_iterations: int = 2000
    convergence_threshold: float = 1e-6
    margin: float = 0.1
    tau: float = 0.10
    use_gpu: bool = False
    random_seed: Optional[int] = 42

    # 损失权重
    qrr_weight: float = 1.0
    axial_weight: float = 2.0
    topology_weight: float = 1.5
    regularization_weight: float = 0.01

    # PyTorch 优化选项
    optimizer: str = "adam"  # "adam", "lbfgs", "sgd"
    use_lr_scheduler: bool = True
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 100
    gradient_clip: Optional[float] = 1.0
    use_vectorized: bool = True  # 使用向量化计算加速
    early_stopping_patience: int = 200  # 早停耐心值


@dataclass
class SolverResult:
    """
    求解器结果。

    Solver result.
    """
    positions: Dict[str, np.ndarray]  # object_id -> [x, y, z]
    satisfiable: bool = True
    satisfaction_rate: float = 1.0
    iterations: int = 0
    final_loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    constraint_details: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


class ConstraintSolver(ABC):
    """
    约束求解器抽象基类。

    Abstract base class for constraint solvers.
    """

    @abstractmethod
    def solve(self, constraints: ParsedConstraints) -> SolverResult:
        """
        求解约束。

        Solve constraints.

        Args:
            constraints: 解析后的约束集合

        Returns:
            SolverResult 对象
        """
        pass


class GradientDescentSolver(ConstraintSolver):
    """
    基于 PyTorch 梯度下降的约束求解器。

    Gradient descent constraint solver based on PyTorch.
    """

    def __init__(self, config: SolverConfig = None):
        """
        初始化求解器。

        Initialize solver.

        Args:
            config: 求解器配置
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch required for GradientDescentSolver. "
                "Install with: pip install torch"
            )

        self.config = config or SolverConfig()

        if self.config.random_seed is not None:
            torch.manual_seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

        self.device = torch.device(
            "cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu"
        )

    def solve(self, constraints: ParsedConstraints) -> SolverResult:
        """
        使用梯度下降求解约束。

        Solve constraints using gradient descent.

        Args:
            constraints: 解析后的约束集合

        Returns:
            SolverResult 对象
        """
        # 获取物体列表
        object_ids = constraints.get_object_ids()
        n_objects = len(object_ids)

        if n_objects == 0:
            return SolverResult(
                positions={},
                satisfiable=False,
                error="No objects to solve",
            )

        # 创建 ID 到索引的映射
        id_to_idx = {obj_id: i for i, obj_id in enumerate(object_ids)}

        # 初始化位置（可学习参数）
        positions = self._initialize_positions(n_objects, constraints)
        positions.requires_grad_(True)

        # 预构建约束索引以加速向量化计算
        if self.config.use_vectorized:
            constraint_indices = self._build_constraint_indices(constraints, id_to_idx)
        else:
            constraint_indices = None

        # 创建优化器
        optimizer = self._create_optimizer(positions)

        # 学习率调度器
        scheduler = None
        if self.config.use_lr_scheduler and self.config.optimizer != "lbfgs":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.config.lr_decay_factor,
                patience=self.config.lr_decay_patience,
                min_lr=1e-6
            )

        # 优化循环
        loss_history = []
        prev_loss = float("inf")
        best_loss = float("inf")
        no_improve_count = 0
        loss_val = 0.0
        iteration = 0

        for iteration in range(self.config.max_iterations):
            # LBFGS 需要闭包
            if self.config.optimizer == "lbfgs":
                def closure():
                    optimizer.zero_grad()
                    loss, _ = self._compute_total_loss(
                        positions, constraints, id_to_idx, constraint_indices
                    )
                    loss.backward()
                    return loss
                loss = optimizer.step(closure)
                loss_val = loss.item() if loss is not None else 0.0
            else:
                optimizer.zero_grad()

                # 计算总损失
                loss, loss_details = self._compute_total_loss(
                    positions, constraints, id_to_idx, constraint_indices
                )

                # 反向传播
                loss.backward()

                # 梯度裁剪
                if self.config.gradient_clip is not None:
                    torch.nn.utils.clip_grad_norm_([positions], self.config.gradient_clip)

                optimizer.step()
                loss_val = loss.item()

            loss_history.append(loss_val)

            # 学习率调度
            if scheduler is not None:
                scheduler.step(loss_val)

            # 早停检测
            if loss_val < best_loss:
                best_loss = loss_val
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= self.config.early_stopping_patience:
                logger.info(f"Early stopping at iteration {iteration}")
                break

            # 检查收敛
            if abs(prev_loss - loss_val) < self.config.convergence_threshold:
                logger.info(f"Converged at iteration {iteration}")
                break

            prev_loss = loss_val

            # 定期日志
            if iteration % 200 == 0:
                current_lr = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else self.config.learning_rate
                logger.debug(f"Iteration {iteration}, Loss: {loss_val:.6f}, LR: {current_lr:.6f}")

        # 提取结果
        final_positions = positions.detach().cpu().numpy()
        result_positions = {
            obj_id: final_positions[idx]
            for obj_id, idx in id_to_idx.items()
        }

        # 计算约束满足率
        satisfaction_rate = self._compute_satisfaction_rate(
            positions.detach(), constraints, id_to_idx
        )

        # 获取最终的损失详情
        with torch.no_grad():
            _, loss_details = self._compute_total_loss(
                positions, constraints, id_to_idx, constraint_indices
            )

        return SolverResult(
            positions=result_positions,
            satisfiable=True,
            satisfaction_rate=satisfaction_rate,
            iterations=iteration + 1,
            final_loss=loss_val,
            loss_history=loss_history,
            constraint_details={k: v.item() if hasattr(v, 'item') else v for k, v in loss_details.items()},
        )

    def _create_optimizer(self, positions: Any) -> Any:
        """
        创建优化器。

        Create optimizer based on config.
        """
        if self.config.optimizer == "adam":
            return optim.Adam([positions], lr=self.config.learning_rate)
        elif self.config.optimizer == "lbfgs":
            return optim.LBFGS(
                [positions],
                lr=self.config.learning_rate,
                max_iter=20,
                history_size=10
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                [positions],
                lr=self.config.learning_rate,
                momentum=0.9,
                nesterov=True
            )
        else:
            return optim.Adam([positions], lr=self.config.learning_rate)

    def _build_constraint_indices(
        self,
        constraints: ParsedConstraints,
        id_to_idx: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        预构建约束索引以加速向量化计算。

        Pre-build constraint indices for vectorized computation.
        """
        indices = {}

        # 构建 QRR 约束索引
        if constraints.qrr:
            qrr_i1, qrr_j1, qrr_i2, qrr_j2 = [], [], [], []
            qrr_types = []  # 0: LT, 1: GT, 2: APPROX
            for qrr in constraints.qrr:
                i1 = id_to_idx.get(qrr.pair1[0])
                j1 = id_to_idx.get(qrr.pair1[1])
                i2 = id_to_idx.get(qrr.pair2[0])
                j2 = id_to_idx.get(qrr.pair2[1])
                if None not in (i1, j1, i2, j2):
                    qrr_i1.append(i1)
                    qrr_j1.append(j1)
                    qrr_i2.append(i2)
                    qrr_j2.append(j2)
                    if qrr.comparator == Comparator.LT:
                        qrr_types.append(0)
                    elif qrr.comparator == Comparator.GT:
                        qrr_types.append(1)
                    else:
                        qrr_types.append(2)

            if qrr_i1:
                indices['qrr'] = {
                    'i1': torch.tensor(qrr_i1, device=self.device),
                    'j1': torch.tensor(qrr_j1, device=self.device),
                    'i2': torch.tensor(qrr_i2, device=self.device),
                    'j2': torch.tensor(qrr_j2, device=self.device),
                    'types': torch.tensor(qrr_types, device=self.device),
                }

        # 构建轴向约束索引
        if constraints.axial:
            axis_to_dim = {"x": 0, "y": 1, "z": 2}
            relation_to_axis = {
                AxialRelation.LEFT_OF: "x", AxialRelation.RIGHT_OF: "x",
                AxialRelation.ABOVE: "z", AxialRelation.BELOW: "z",
                AxialRelation.IN_FRONT_OF: "y", AxialRelation.BEHIND: "y",
            }
            axial_a, axial_b, axial_dims, axial_dirs = [], [], [], []

            for axial in constraints.axial:
                idx_a = id_to_idx.get(axial.obj_a)
                idx_b = id_to_idx.get(axial.obj_b)
                if idx_a is not None and idx_b is not None:
                    axial_a.append(idx_a)
                    axial_b.append(idx_b)
                    axis = axial.axis if axial.axis else relation_to_axis.get(axial.relation, "x")
                    axial_dims.append(axis_to_dim.get(axis, 0))
                    # -1 for A < B, 1 for A > B
                    if axial.relation in [AxialRelation.LEFT_OF, AxialRelation.BELOW, AxialRelation.BEHIND]:
                        axial_dirs.append(-1)
                    else:
                        axial_dirs.append(1)

            if axial_a:
                indices['axial'] = {
                    'a': torch.tensor(axial_a, device=self.device),
                    'b': torch.tensor(axial_b, device=self.device),
                    'dims': torch.tensor(axial_dims, device=self.device),
                    'dirs': torch.tensor(axial_dirs, device=self.device, dtype=torch.float32),
                }

        return indices

    def _initialize_positions(
        self,
        n_objects: int,
        constraints: ParsedConstraints,
    ) -> Any:
        """
        初始化物体位置。

        Initialize object positions.
        """
        # 检查是否有已知位置
        known_positions = {}
        for obj in constraints.objects:
            if obj.position_3d is not None:
                known_positions[obj.id] = obj.position_3d

        # 创建初始位置
        positions = torch.randn(
            n_objects, self.config.n_dims, device=self.device
        ) * 2.0  # 扩大初始分布

        # 如果有已知位置，使用它们
        if known_positions:
            object_ids = constraints.get_object_ids()
            for i, obj_id in enumerate(object_ids):
                if obj_id in known_positions:
                    positions[i] = torch.tensor(
                        known_positions[obj_id][:self.config.n_dims],
                        device=self.device,
                    )

        return positions

    def _compute_total_loss(
        self,
        positions: Any,
        constraints: ParsedConstraints,
        id_to_idx: Dict[str, int],
        constraint_indices: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        计算总损失。

        Compute total loss.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        loss_details = {}

        # QRR 损失（使用向量化计算如果可用）
        if constraints.qrr:
            if constraint_indices and 'qrr' in constraint_indices:
                qrr_loss = self._compute_qrr_loss_vectorized(positions, constraint_indices['qrr'])
            else:
                qrr_loss = self._compute_qrr_loss(positions, constraints.qrr, id_to_idx)
            total_loss = total_loss + self.config.qrr_weight * qrr_loss
            loss_details["qrr"] = qrr_loss

        # 轴向约束损失（使用向量化计算如果可用）
        if constraints.axial:
            if constraint_indices and 'axial' in constraint_indices:
                axial_loss = self._compute_axial_loss_vectorized(positions, constraint_indices['axial'])
            else:
                axial_loss = self._compute_axial_loss(positions, constraints.axial, id_to_idx)
            total_loss = total_loss + self.config.axial_weight * axial_loss
            loss_details["axial"] = axial_loss

        # 拓扑约束损失
        if constraints.topology:
            topo_loss = self._compute_topology_loss(
                positions, constraints.topology, id_to_idx, constraints.objects
            )
            total_loss = total_loss + self.config.topology_weight * topo_loss
            loss_details["topology"] = topo_loss

        # 正则化（防止位置过于分散或集中）
        reg_loss = self._compute_regularization_loss(positions)
        total_loss = total_loss + self.config.regularization_weight * reg_loss
        loss_details["regularization"] = reg_loss

        return total_loss, loss_details

    def _compute_qrr_loss_vectorized(
        self,
        positions: Any,
        qrr_indices: Dict[str, Any],
    ) -> Any:
        """
        向量化计算 QRR 约束损失。

        Vectorized QRR constraint loss computation.
        """
        i1, j1, i2, j2 = qrr_indices['i1'], qrr_indices['j1'], qrr_indices['i2'], qrr_indices['j2']
        types = qrr_indices['types']
        margin = self.config.margin

        # 批量计算距离
        d1 = torch.norm(positions[i1] - positions[j1], dim=1) + 1e-8
        d2 = torch.norm(positions[i2] - positions[j2], dim=1) + 1e-8

        # LT: d1 < d2 -> relu(d1 - d2 + margin)
        lt_mask = (types == 0)
        lt_loss = torch.relu(d1 - d2 + margin) * lt_mask.float()

        # GT: d1 > d2 -> relu(d2 - d1 + margin)
        gt_mask = (types == 1)
        gt_loss = torch.relu(d2 - d1 + margin) * gt_mask.float()

        # APPROX: (d1 - d2)^2
        approx_mask = (types == 2)
        approx_loss = ((d1 - d2) ** 2) * approx_mask.float()

        return (lt_loss + gt_loss + approx_loss).sum()

    def _compute_axial_loss_vectorized(
        self,
        positions: Any,
        axial_indices: Dict[str, Any],
    ) -> Any:
        """
        向量化计算轴向约束损失。

        Vectorized axial constraint loss computation.
        """
        a_idx, b_idx = axial_indices['a'], axial_indices['b']
        dims = axial_indices['dims']
        dirs = axial_indices['dirs']
        margin = self.config.margin

        # 获取各轴坐标
        # 由于不同约束可能在不同轴上，我们需要逐个处理
        # 但可以用 gather 进行向量化
        n_constraints = len(a_idx)
        losses = torch.zeros(n_constraints, device=self.device)

        for i in range(n_constraints):
            pos_a = positions[a_idx[i], dims[i]]
            pos_b = positions[b_idx[i], dims[i]]
            if dirs[i] < 0:
                # A < B
                losses[i] = torch.relu(pos_a - pos_b + margin)
            else:
                # A > B
                losses[i] = torch.relu(pos_b - pos_a + margin)

        return losses.sum()

    def _compute_qrr_loss(
        self,
        positions: Any,
        qrr_constraints: List[QRRConstraint],
        id_to_idx: Dict[str, int],
    ) -> Any:
        """
        计算 QRR 约束损失。

        Compute QRR constraint loss.
        """
        loss = torch.tensor(0.0, device=self.device)
        margin = self.config.margin

        for qrr in qrr_constraints:
            try:
                # 获取物体索引
                i1 = id_to_idx.get(qrr.pair1[0])
                j1 = id_to_idx.get(qrr.pair1[1])
                i2 = id_to_idx.get(qrr.pair2[0])
                j2 = id_to_idx.get(qrr.pair2[1])

                if None in (i1, j1, i2, j2):
                    continue

                # 计算距离
                d1 = torch.norm(positions[i1] - positions[j1]) + 1e-8
                d2 = torch.norm(positions[i2] - positions[j2]) + 1e-8

                if qrr.comparator == Comparator.LT:
                    # d1 < d2: 惩罚 d1 >= d2 - margin
                    loss = loss + torch.relu(d1 - d2 + margin)
                elif qrr.comparator == Comparator.GT:
                    # d1 > d2: 惩罚 d1 <= d2 + margin
                    loss = loss + torch.relu(d2 - d1 + margin)
                else:  # APPROX
                    # d1 ≈ d2: 惩罚差异
                    loss = loss + (d1 - d2) ** 2

            except Exception as e:
                logger.warning(f"Error computing QRR loss: {e}")

        return loss

    def _compute_axial_loss(
        self,
        positions: Any,
        axial_constraints: List[AxialConstraint],
        id_to_idx: Dict[str, int],
    ) -> Any:
        """
        计算轴向约束损失。

        Compute axial constraint loss.
        """
        loss = torch.tensor(0.0, device=self.device)
        margin = self.config.margin

        # 轴向到维度索引的映射
        axis_to_dim = {"x": 0, "y": 1, "z": 2}

        # 关系到比较方向的映射
        relation_to_direction = {
            AxialRelation.LEFT_OF: -1,     # A.x < B.x
            AxialRelation.RIGHT_OF: 1,     # A.x > B.x
            AxialRelation.BELOW: -1,       # A.z < B.z (或 A.y < B.y)
            AxialRelation.ABOVE: 1,        # A.z > B.z (或 A.y > B.y)
            AxialRelation.IN_FRONT_OF: 1,  # A.y > B.y (靠近相机)
            AxialRelation.BEHIND: -1,      # A.y < B.y
        }

        # 关系到轴的映射（如果未指定轴）
        relation_to_axis = {
            AxialRelation.LEFT_OF: "x",
            AxialRelation.RIGHT_OF: "x",
            AxialRelation.ABOVE: "z",
            AxialRelation.BELOW: "z",
            AxialRelation.IN_FRONT_OF: "y",
            AxialRelation.BEHIND: "y",
        }

        for axial in axial_constraints:
            try:
                idx_a = id_to_idx.get(axial.obj_a)
                idx_b = id_to_idx.get(axial.obj_b)

                if idx_a is None or idx_b is None:
                    continue

                # 确定轴
                axis = axial.axis if axial.axis else relation_to_axis.get(axial.relation, "x")
                dim = axis_to_dim.get(axis, 0)

                # 确定方向
                direction = relation_to_direction.get(axial.relation, -1)

                # 获取坐标
                pos_a = positions[idx_a, dim]
                pos_b = positions[idx_b, dim]

                if direction < 0:
                    # A < B: 惩罚 A >= B - margin
                    loss = loss + torch.relu(pos_a - pos_b + margin)
                else:
                    # A > B: 惩罚 A <= B + margin
                    loss = loss + torch.relu(pos_b - pos_a + margin)

            except Exception as e:
                logger.warning(f"Error computing axial loss: {e}")

        return loss

    def _compute_topology_loss(
        self,
        positions: Any,
        topology_constraints: List[TopologyConstraint],
        id_to_idx: Dict[str, int],
        objects: List,
    ) -> Any:
        """
        计算拓扑约束损失。

        Compute topology constraint loss.
        """
        loss = torch.tensor(0.0, device=self.device)

        # 获取物体大小
        size_map = {obj.id: self._get_size_value(obj.size) for obj in objects}

        for topo in topology_constraints:
            try:
                idx1 = id_to_idx.get(topo.obj1)
                idx2 = id_to_idx.get(topo.obj2)

                if idx1 is None or idx2 is None:
                    continue

                # 计算距离
                dist = torch.norm(positions[idx1] - positions[idx2]) + 1e-8

                # 获取物体半径（近似）
                r1 = size_map.get(topo.obj1, 0.5)
                r2 = size_map.get(topo.obj2, 0.5)
                min_dist = r1 + r2

                if topo.relation.value == "disjoint":
                    # 分离：距离应大于两物体半径之和
                    loss = loss + torch.relu(min_dist - dist + self.config.margin)
                elif topo.relation.value == "touching":
                    # 接触：距离应接近两物体半径之和
                    loss = loss + (dist - min_dist) ** 2
                elif topo.relation.value == "overlapping":
                    # 重叠：距离应小于两物体半径之和
                    loss = loss + torch.relu(dist - min_dist + self.config.margin)

            except Exception as e:
                logger.warning(f"Error computing topology loss: {e}")

        return loss

    def _compute_regularization_loss(self, positions: Any) -> Any:
        """
        计算正则化损失。

        Compute regularization loss.
        """
        # 中心化：鼓励物体围绕原点分布
        center = positions.mean(dim=0)
        center_loss = torch.sum(center ** 2)

        # 分散：鼓励物体不要太集中
        pairwise_dists = torch.cdist(positions, positions)
        min_dist = 0.5  # 最小期望距离
        # 只考虑非对角线元素
        mask = 1 - torch.eye(positions.shape[0], device=self.device)
        spread_loss = torch.relu(min_dist - pairwise_dists * mask).sum()

        return center_loss * 0.1 + spread_loss * 0.1

    def _get_size_value(self, size) -> float:
        """获取尺寸数值。"""
        size_map = {
            "tiny": 0.25,
            "small": 0.35,
            "medium": 0.5,
            "large": 0.7,
        }
        if isinstance(size, str):
            return size_map.get(size.lower(), 0.5)
        elif isinstance(size, (int, float)):
            return float(size)
        return 0.5

    def _compute_satisfaction_rate(
        self,
        positions: Any,
        constraints: ParsedConstraints,
        id_to_idx: Dict[str, int],
    ) -> float:
        """
        计算约束满足率。

        Compute constraint satisfaction rate.
        """
        total = 0
        satisfied = 0
        margin = self.config.margin * 0.5

        with torch.no_grad():
            # 检查 QRR 约束
            for qrr in constraints.qrr:
                i1 = id_to_idx.get(qrr.pair1[0])
                j1 = id_to_idx.get(qrr.pair1[1])
                i2 = id_to_idx.get(qrr.pair2[0])
                j2 = id_to_idx.get(qrr.pair2[1])

                if None in (i1, j1, i2, j2):
                    continue

                d1 = torch.norm(positions[i1] - positions[j1]).item()
                d2 = torch.norm(positions[i2] - positions[j2]).item()

                total += 1
                if qrr.comparator == Comparator.LT and d1 < d2 - margin:
                    satisfied += 1
                elif qrr.comparator == Comparator.GT and d1 > d2 + margin:
                    satisfied += 1
                elif qrr.comparator == Comparator.APPROX and abs(d1 - d2) < margin * 2:
                    satisfied += 1

            # 检查轴向约束
            axis_to_dim = {"x": 0, "y": 1, "z": 2}
            for axial in constraints.axial:
                idx_a = id_to_idx.get(axial.obj_a)
                idx_b = id_to_idx.get(axial.obj_b)

                if idx_a is None or idx_b is None:
                    continue

                dim = axis_to_dim.get(axial.axis, 0)
                pos_a = positions[idx_a, dim].item()
                pos_b = positions[idx_b, dim].item()

                total += 1
                if axial.relation in [AxialRelation.LEFT_OF, AxialRelation.BELOW, AxialRelation.BEHIND]:
                    if pos_a < pos_b - margin:
                        satisfied += 1
                else:
                    if pos_a > pos_b + margin:
                        satisfied += 1

        return satisfied / total if total > 0 else 1.0


class NumpyGradientSolver(ConstraintSolver):
    """
    纯 NumPy 实现的梯度下降求解器（无需 PyTorch）。

    Pure NumPy gradient descent solver (no PyTorch required).
    """

    def __init__(self, config: SolverConfig = None):
        """初始化求解器。"""
        self.config = config or SolverConfig()

        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    def solve(self, constraints: ParsedConstraints) -> SolverResult:
        """
        使用 NumPy 梯度下降求解约束。

        Solve constraints using NumPy gradient descent.
        """
        object_ids = constraints.get_object_ids()
        n_objects = len(object_ids)

        if n_objects == 0:
            return SolverResult(
                positions={},
                satisfiable=False,
                error="No objects to solve",
            )

        id_to_idx = {obj_id: i for i, obj_id in enumerate(object_ids)}

        # 初始化位置
        positions = np.random.randn(n_objects, self.config.n_dims) * 2.0

        # 梯度下降
        loss_history = []

        for iteration in range(self.config.max_iterations):
            grad = np.zeros_like(positions)

            # 计算梯度
            loss = self._compute_gradients(
                positions, grad, constraints, id_to_idx
            )

            # 更新位置
            positions -= self.config.learning_rate * grad

            loss_history.append(loss)

            # 检查收敛
            if iteration > 0 and abs(loss_history[-2] - loss) < self.config.convergence_threshold:
                break

        # 构建结果
        result_positions = {
            obj_id: positions[idx].copy()
            for obj_id, idx in id_to_idx.items()
        }

        return SolverResult(
            positions=result_positions,
            satisfiable=True,
            satisfaction_rate=self._compute_satisfaction_rate(positions, constraints, id_to_idx),
            iterations=iteration + 1,
            final_loss=loss,
            loss_history=loss_history,
        )

    def _compute_gradients(
        self,
        positions: np.ndarray,
        grad: np.ndarray,
        constraints: ParsedConstraints,
        id_to_idx: Dict[str, int],
    ) -> float:
        """计算梯度和损失。"""
        total_loss = 0.0
        margin = self.config.margin

        # 轴向约束
        axis_to_dim = {"x": 0, "y": 1, "z": 2}

        for axial in constraints.axial:
            idx_a = id_to_idx.get(axial.obj_a)
            idx_b = id_to_idx.get(axial.obj_b)

            if idx_a is None or idx_b is None:
                continue

            dim = axis_to_dim.get(axial.axis, 0)
            pos_a = positions[idx_a, dim]
            pos_b = positions[idx_b, dim]

            # LEFT_OF, BELOW, BEHIND: A < B
            if axial.relation in [AxialRelation.LEFT_OF, AxialRelation.BELOW, AxialRelation.BEHIND]:
                violation = pos_a - pos_b + margin
                if violation > 0:
                    total_loss += violation
                    grad[idx_a, dim] += 1.0
                    grad[idx_b, dim] -= 1.0
            else:  # RIGHT_OF, ABOVE, IN_FRONT_OF: A > B
                violation = pos_b - pos_a + margin
                if violation > 0:
                    total_loss += violation
                    grad[idx_a, dim] -= 1.0
                    grad[idx_b, dim] += 1.0

        # QRR 约束
        for qrr in constraints.qrr:
            i1 = id_to_idx.get(qrr.pair1[0])
            j1 = id_to_idx.get(qrr.pair1[1])
            i2 = id_to_idx.get(qrr.pair2[0])
            j2 = id_to_idx.get(qrr.pair2[1])

            if None in (i1, j1, i2, j2):
                continue

            v1 = positions[i1] - positions[j1]
            v2 = positions[i2] - positions[j2]
            d1 = np.linalg.norm(v1) + 1e-8
            d2 = np.linalg.norm(v2) + 1e-8

            if qrr.comparator == Comparator.LT:  # d1 < d2
                if d1 >= d2 - margin:
                    total_loss += d1 - d2 + margin
                    # d1 的梯度
                    grad[i1] += v1 / d1
                    grad[j1] -= v1 / d1
                    # d2 的梯度（负）
                    grad[i2] -= v2 / d2
                    grad[j2] += v2 / d2

            elif qrr.comparator == Comparator.GT:  # d1 > d2
                if d1 <= d2 + margin:
                    total_loss += d2 - d1 + margin
                    grad[i1] -= v1 / d1
                    grad[j1] += v1 / d1
                    grad[i2] += v2 / d2
                    grad[j2] -= v2 / d2

        return total_loss

    def _compute_satisfaction_rate(
        self,
        positions: np.ndarray,
        constraints: ParsedConstraints,
        id_to_idx: Dict[str, int],
    ) -> float:
        """计算约束满足率。"""
        total = 0
        satisfied = 0
        margin = self.config.margin * 0.5

        axis_to_dim = {"x": 0, "y": 1, "z": 2}

        for axial in constraints.axial:
            idx_a = id_to_idx.get(axial.obj_a)
            idx_b = id_to_idx.get(axial.obj_b)

            if idx_a is None or idx_b is None:
                continue

            dim = axis_to_dim.get(axial.axis, 0)
            pos_a = positions[idx_a, dim]
            pos_b = positions[idx_b, dim]

            total += 1
            if axial.relation in [AxialRelation.LEFT_OF, AxialRelation.BELOW, AxialRelation.BEHIND]:
                if pos_a < pos_b - margin:
                    satisfied += 1
            else:
                if pos_a > pos_b + margin:
                    satisfied += 1

        for qrr in constraints.qrr:
            i1 = id_to_idx.get(qrr.pair1[0])
            j1 = id_to_idx.get(qrr.pair1[1])
            i2 = id_to_idx.get(qrr.pair2[0])
            j2 = id_to_idx.get(qrr.pair2[1])

            if None in (i1, j1, i2, j2):
                continue

            d1 = np.linalg.norm(positions[i1] - positions[j1])
            d2 = np.linalg.norm(positions[i2] - positions[j2])

            total += 1
            if qrr.comparator == Comparator.LT and d1 < d2 - margin:
                satisfied += 1
            elif qrr.comparator == Comparator.GT and d1 > d2 + margin:
                satisfied += 1
            elif qrr.comparator == Comparator.APPROX and abs(d1 - d2) < margin * 2:
                satisfied += 1

        return satisfied / total if total > 0 else 1.0


def create_solver(config: SolverConfig = None, use_pytorch: bool = True) -> ConstraintSolver:
    """
    创建约束求解器的工厂函数。

    Factory function to create constraint solver.

    Args:
        config: 求解器配置
        use_pytorch: 是否使用 PyTorch（默认 True）

    Returns:
        ConstraintSolver 实例
    """
    if use_pytorch and TORCH_AVAILABLE:
        return GradientDescentSolver(config)
    else:
        logger.info("PyTorch not available, using NumPy solver")
        return NumpyGradientSolver(config)
