"""
T3（重构任务）的序嵌入基线。

本基线使用基于梯度的优化方法从序约束中重构点配置。

核心思想：
- 给定序约束（距离比较），找到满足尽可能多约束的点配置
- 使用梯度下降优化点位置
- 最小化约束违反惩罚

损失函数：
- 对于每个约束 dist(A,B) < dist(C,D)：
  如果违反，添加 max(0, dist(A,B) - dist(C,D) + margin)²
- margin 提供软边界，避免数值不稳定

实现版本：
- OrdinalEmbedding: 使用 PyTorch（支持 GPU 加速）
- NumpyOrdinalEmbedding: 纯 NumPy 实现（无需 PyTorch）

适用于：
- T3 任务评估
- 约束可满足性分析
- 点云重构

Ordinal embedding baseline for T3 (reconstruction) task.

This baseline reconstructs point configurations from ordinal constraints
using gradient-based optimization.
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class EmbeddingConfig:
    """
    序嵌入的配置。

    Configuration for ordinal embedding.
    """
    n_dims: int = 3
    learning_rate: float = 0.01
    max_iterations: int = 1000
    margin: float = 0.1
    convergence_threshold: float = 1e-6
    use_gpu: bool = False
    random_seed: Optional[int] = None


class OrdinalEmbedding:
    """
    通过梯度下降优化的序嵌入。

    给定序约束（距离比较），找到满足尽可能多约束的点配置。

    Ordinal embedding via gradient descent optimization.

    Given ordinal constraints (distance comparisons), finds a point
    configuration that satisfies as many constraints as possible.
    """

    def __init__(self, config: EmbeddingConfig = None):
        """
        初始化嵌入优化器。

        参数:
            config: 嵌入配置

        Initialize embedding optimizer.

        Args:
            config: Embedding configuration
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch required for ordinal embedding. "
                "Install with: pip install torch"
            )

        self.config = config or EmbeddingConfig()

        if self.config.random_seed is not None:
            torch.manual_seed(self.config.random_seed)
            np.random.seed(self.config.random_seed)

    def fit(
        self,
        n_points: int,
        qrr_constraints: List[Dict],
        initial_positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        拟合点配置以满足约束。

        参数:
            n_points: 要嵌入的点数
            qrr_constraints: QRR 约束字典列表
            initial_positions: 可选的初始位置 (n_points, n_dims)

        返回:
            优化后的位置 (n_points, n_dims)

        Fit point configuration to satisfy constraints.

        Args:
            n_points: Number of points to embed
            qrr_constraints: List of QRR constraint dictionaries
            initial_positions: Optional initial positions (n_points, n_dims)

        Returns:
            Optimized positions (n_points, n_dims)
        """
        device = torch.device("cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu")

        # Initialize positions
        if initial_positions is not None:
            positions = torch.tensor(
                initial_positions,
                dtype=torch.float32,
                device=device,
                requires_grad=True
            )
        else:
            positions = torch.randn(
                n_points,
                self.config.n_dims,
                device=device,
                requires_grad=True
            )

        # Parse constraints into index format
        constraints = self._parse_constraints(qrr_constraints, n_points)

        # Optimizer
        optimizer = optim.Adam([positions], lr=self.config.learning_rate)

        # Optimization loop
        prev_loss = float('inf')
        for iteration in range(self.config.max_iterations):
            optimizer.zero_grad()

            loss = self._compute_loss(positions, constraints)
            loss.backward()
            optimizer.step()

            # Check convergence
            loss_val = loss.item()
            if abs(prev_loss - loss_val) < self.config.convergence_threshold:
                break
            prev_loss = loss_val

        return positions.detach().cpu().numpy()

    def _parse_constraints(
        self,
        qrr_constraints: List[Dict],
        n_points: int
    ) -> List[Tuple]:
        """
        将约束解析为索引格式。

        Parse constraints into index format.
        """
        constraints = []

        # Build object ID to index mapping
        obj_ids = set()
        for c in qrr_constraints:
            for obj in c.get("pair1", []) + c.get("pair2", []):
                obj_ids.add(obj)

        obj_to_idx = {obj: i for i, obj in enumerate(sorted(obj_ids))}

        for c in qrr_constraints:
            pair1 = c.get("pair1", [])
            pair2 = c.get("pair2", [])
            comparator = c.get("comparator", "~=")

            if len(pair1) < 2 or len(pair2) < 2:
                continue

            try:
                i1 = obj_to_idx.get(pair1[0])
                j1 = obj_to_idx.get(pair1[1])
                i2 = obj_to_idx.get(pair2[0])
                j2 = obj_to_idx.get(pair2[1])

                if None in (i1, j1, i2, j2):
                    continue
                if max(i1, j1, i2, j2) >= n_points:
                    continue

                constraints.append((i1, j1, i2, j2, comparator))
            except (KeyError, ValueError):
                continue

        return constraints

    def _compute_loss(
        self,
        positions: torch.Tensor,
        constraints: List[Tuple]
    ) -> torch.Tensor:
        """
        计算约束满足损失。

        Compute constraint satisfaction loss.
        """
        loss = torch.tensor(0.0, device=positions.device)
        margin = self.config.margin

        for i1, j1, i2, j2, comparator in constraints:
            # Compute distances
            d1 = torch.norm(positions[i1] - positions[j1])
            d2 = torch.norm(positions[i2] - positions[j2])

            if comparator == "<":
                # d1 should be less than d2
                # Loss = max(0, d1 - d2 + margin)
                loss = loss + torch.relu(d1 - d2 + margin)

            elif comparator == ">":
                # d1 should be greater than d2
                # Loss = max(0, d2 - d1 + margin)
                loss = loss + torch.relu(d2 - d1 + margin)

            else:  # "~=" approximate equality
                # d1 and d2 should be close
                # Loss = (d1 - d2)^2
                loss = loss + (d1 - d2) ** 2

        return loss

    def fit_with_progress(
        self,
        n_points: int,
        qrr_constraints: List[Dict],
        callback: Optional[callable] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        拟合并跟踪进度。

        参数:
            n_points: 点的数量
            qrr_constraints: 约束
            callback: 进度回调函数

        返回:
            (positions, history) 元组

        Fit with progress tracking.

        Args:
            n_points: Number of points
            qrr_constraints: Constraints
            callback: Progress callback function

        Returns:
            (positions, history) tuple
        """
        device = torch.device("cuda" if self.config.use_gpu and torch.cuda.is_available() else "cpu")

        positions = torch.randn(
            n_points,
            self.config.n_dims,
            device=device,
            requires_grad=True
        )

        constraints = self._parse_constraints(qrr_constraints, n_points)
        optimizer = optim.Adam([positions], lr=self.config.learning_rate)

        history = {
            "loss": [],
            "satisfaction_rate": [],
        }

        for iteration in range(self.config.max_iterations):
            optimizer.zero_grad()

            loss = self._compute_loss(positions, constraints)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            sat_rate = self._compute_satisfaction_rate(
                positions.detach(), constraints
            )

            history["loss"].append(loss_val)
            history["satisfaction_rate"].append(sat_rate)

            if callback:
                callback(iteration, loss_val, sat_rate)

            # Early stopping if fully satisfied
            if sat_rate >= 1.0:
                break

        return positions.detach().cpu().numpy(), history

    def _compute_satisfaction_rate(
        self,
        positions: torch.Tensor,
        constraints: List[Tuple]
    ) -> float:
        """
        计算满足的约束比例。

        Compute fraction of satisfied constraints.
        """
        if not constraints:
            return 1.0

        satisfied = 0
        margin = self.config.margin * 0.5  # Stricter for evaluation

        with torch.no_grad():
            for i1, j1, i2, j2, comparator in constraints:
                d1 = torch.norm(positions[i1] - positions[j1]).item()
                d2 = torch.norm(positions[i2] - positions[j2]).item()

                if comparator == "<" and d1 < d2 - margin:
                    satisfied += 1
                elif comparator == ">" and d1 > d2 + margin:
                    satisfied += 1
                elif comparator == "~=" and abs(d1 - d2) < margin * 2:
                    satisfied += 1

        return satisfied / len(constraints)


# =============================================================================
# Standalone Functions
# =============================================================================

def embed_from_constraints(
    qrr_constraints: List[Dict],
    n_dims: int = 3,
    max_iterations: int = 1000,
) -> np.ndarray:
    """
    从约束快速嵌入点的函数。

    参数:
        qrr_constraints: QRR 约束列表
        n_dims: 嵌入维度
        max_iterations: 最大优化迭代次数

    返回:
        点位置数组

    Quick function to embed points from constraints.

    Args:
        qrr_constraints: QRR constraint list
        n_dims: Embedding dimensions
        max_iterations: Max optimization iterations

    Returns:
        Point positions array
    """
    # Determine number of points
    obj_ids = set()
    for c in qrr_constraints:
        for obj in c.get("pair1", []) + c.get("pair2", []):
            obj_ids.add(obj)

    n_points = len(obj_ids)

    config = EmbeddingConfig(
        n_dims=n_dims,
        max_iterations=max_iterations,
    )

    embedder = OrdinalEmbedding(config)
    positions = embedder.fit(n_points, qrr_constraints)

    return positions


class NumpyOrdinalEmbedding:
    """
    仅使用 NumPy 的序嵌入（无需 PyTorch）。

    使用简单的梯度下降进行优化。

    NumPy-only ordinal embedding (no PyTorch required).

    Uses simple gradient descent for optimization.
    """

    def __init__(
        self,
        n_dims: int = 3,
        learning_rate: float = 0.01,
        max_iterations: int = 1000,
        margin: float = 0.1,
    ):
        """
        初始化 NumPy 嵌入器。

        Initialize NumPy embedder.
        """
        self.n_dims = n_dims
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.margin = margin

    def fit(
        self,
        n_points: int,
        qrr_constraints: List[Dict],
    ) -> np.ndarray:
        """
        使用 NumPy 梯度下降进行拟合。

        Fit using NumPy gradient descent.
        """
        # Initialize
        positions = np.random.randn(n_points, self.n_dims)

        # Parse constraints
        obj_ids = set()
        for c in qrr_constraints:
            for obj in c.get("pair1", []) + c.get("pair2", []):
                obj_ids.add(obj)
        obj_to_idx = {obj: i for i, obj in enumerate(sorted(obj_ids))}

        constraints = []
        for c in qrr_constraints:
            pair1 = c.get("pair1", [])
            pair2 = c.get("pair2", [])
            if len(pair1) < 2 or len(pair2) < 2:
                continue

            try:
                i1 = obj_to_idx[pair1[0]]
                j1 = obj_to_idx[pair1[1]]
                i2 = obj_to_idx[pair2[0]]
                j2 = obj_to_idx[pair2[1]]
                comparator = c.get("comparator", "~=")
                constraints.append((i1, j1, i2, j2, comparator))
            except KeyError:
                continue

        # Optimization loop
        for _ in range(self.max_iterations):
            grad = np.zeros_like(positions)

            for i1, j1, i2, j2, comparator in constraints:
                v1 = positions[i1] - positions[j1]
                v2 = positions[i2] - positions[j2]
                d1 = np.linalg.norm(v1) + 1e-8
                d2 = np.linalg.norm(v2) + 1e-8

                if comparator == "<":
                    # Want d1 < d2
                    if d1 >= d2 - self.margin:
                        # Push d1 smaller, d2 larger
                        grad[i1] -= v1 / d1
                        grad[j1] += v1 / d1
                        grad[i2] += v2 / d2
                        grad[j2] -= v2 / d2

                elif comparator == ">":
                    # Want d1 > d2
                    if d1 <= d2 + self.margin:
                        grad[i1] += v1 / d1
                        grad[j1] -= v1 / d1
                        grad[i2] -= v2 / d2
                        grad[j2] += v2 / d2

                else:  # ~=
                    # Want d1 ≈ d2
                    diff = d1 - d2
                    grad[i1] += diff * v1 / d1
                    grad[j1] -= diff * v1 / d1
                    grad[i2] -= diff * v2 / d2
                    grad[j2] += diff * v2 / d2

            positions -= self.learning_rate * grad

        return positions
