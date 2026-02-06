"""
任务 T3：序重构运行器。

评估从序约束集中重构点配置的能力。

任务描述：
- 输入：QRR 约束集（无图像、无绝对坐标）
- 输出：满足约束的 3D 点配置
- 目标：找到与真值相似的配置

评估指标：
- NRMS（归一化 RMS）：Procrustes 对齐后的归一化均方根误差
  * 使用 Procrustes 分析消除旋转、平移、缩放的影响
  * 只评估配置的形状相似性
- 约束满足率：重构配置满足的约束比例
- 尺度因子：Procrustes 对齐的缩放系数

方法：
- 使用梯度下降优化点位置
- 最小化约束违反惩罚
- 支持 PyTorch（GPU）或纯 NumPy 实现

特点：
- 视角不变：不依赖具体视角
- 相似性不变：通过 Procrustes 对齐
- 无需绝对尺度信息

Task T3: Ordinal Reconstruction runner.

Evaluates the ability to reconstruct point configurations
from ordinal constraint sets.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import numpy as np

from ordinal_spatial.evaluation.metrics import compute_t3_metrics, T3Metrics

logger = logging.getLogger(__name__)


@dataclass
class T3Config:
    """Configuration for T3 task runner."""
    n_dims: int = 3
    max_iterations: int = 1000
    learning_rate: float = 0.01
    margin: float = 0.1
    use_gpu: bool = False
    save_predictions: bool = True
    output_dir: Optional[str] = None


@dataclass
class T3Result:
    """Result from T3 evaluation."""
    metrics: T3Metrics
    predictions: List[Dict]
    ground_truth: List[Dict]
    config: T3Config
    errors: List[str] = field(default_factory=list)


class T3Runner:
    """
    Runner for T3 (Ordinal Reconstruction) task.

    Evaluates ability to reconstruct point configurations
    from ordinal constraint sets.
    """

    def __init__(self, embedder=None, config: T3Config = None):
        """
        Initialize T3 runner.

        Args:
            embedder: Embedding implementation (optional, will create default)
            config: Task configuration
        """
        self.config = config or T3Config()
        self.embedder = embedder

        if self.embedder is None:
            self._init_default_embedder()

    def _init_default_embedder(self):
        """
        初始化默认嵌入优化器。

        Initialize default embedding optimizer.
        """
        try:
            from ordinal_spatial.baselines.embedding import (
                OrdinalEmbedding,
                EmbeddingConfig,
            )
            embed_config = EmbeddingConfig(
                n_dims=self.config.n_dims,
                learning_rate=self.config.learning_rate,
                max_iterations=self.config.max_iterations,
                margin=self.config.margin,
                use_gpu=self.config.use_gpu,
            )
            self.embedder = OrdinalEmbedding(embed_config)
        except ImportError:
            # Fall back to numpy-only version
            from ordinal_spatial.baselines.embedding import NumpyOrdinalEmbedding
            self.embedder = NumpyOrdinalEmbedding(
                n_dims=self.config.n_dims,
                learning_rate=self.config.learning_rate,
                max_iterations=self.config.max_iterations,
                margin=self.config.margin,
            )

    def run(
        self,
        dataset: List[Dict],
    ) -> T3Result:
        """
        Run T3 evaluation on dataset.

        Args:
            dataset: List of scene data with constraints and ground truth

        Returns:
            T3Result with metrics and predictions
        """
        all_predictions = []
        all_ground_truth = []
        errors = []

        for item in dataset:
            scene_id = item.get("scene_id", "unknown")
            constraints = item.get("constraints", item.get("qrr", []))
            gt_positions = item.get("ground_truth_positions", None)
            objects = item.get("objects", [])

            # Get QRR constraints
            if isinstance(constraints, dict):
                qrr = constraints.get("qrr", [])
            elif isinstance(constraints, list):
                qrr = constraints
            else:
                qrr = []

            # Determine number of points
            obj_ids = set()
            for c in qrr:
                for obj in c.get("pair1", []) + c.get("pair2", []):
                    obj_ids.add(obj)

            # Also include objects from scene
            for obj in objects:
                obj_id = obj.get("id", obj.get("name", ""))
                if obj_id:
                    obj_ids.add(obj_id)

            n_points = len(obj_ids)
            obj_to_idx = {obj: i for i, obj in enumerate(sorted(obj_ids))}

            if n_points < 2:
                logger.warning(f"Scene {scene_id}: insufficient points ({n_points})")
                errors.append(f"{scene_id}: insufficient points")
                continue

            try:
                # Run embedding
                if hasattr(self.embedder, "fit"):
                    predicted_positions = self.embedder.fit(n_points, qrr)
                else:
                    # Custom embedder interface
                    predicted_positions = self.embedder.embed(qrr)

                # Convert to dict format
                pred_dict = {
                    obj: predicted_positions[idx].tolist()
                    for obj, idx in obj_to_idx.items()
                }

                all_predictions.append({
                    "scene_id": scene_id,
                    "positions": pred_dict,
                    "n_points": n_points,
                    "n_constraints": len(qrr),
                })

                # Process ground truth
                if gt_positions is not None:
                    if isinstance(gt_positions, np.ndarray):
                        gt_dict = {
                            obj: gt_positions[idx].tolist()
                            for obj, idx in obj_to_idx.items()
                        }
                    elif isinstance(gt_positions, dict):
                        gt_dict = gt_positions
                    else:
                        gt_dict = {}
                else:
                    # Try to extract from objects
                    gt_dict = {}
                    for obj in objects:
                        obj_id = obj.get("id", obj.get("name", ""))
                        if obj_id and "position" in obj:
                            gt_dict[obj_id] = obj["position"]

                all_ground_truth.append({
                    "scene_id": scene_id,
                    "positions": gt_dict,
                })

            except Exception as e:
                logger.warning(f"Error on scene {scene_id}: {e}")
                errors.append(f"{scene_id}: {e}")
                all_predictions.append({
                    "scene_id": scene_id,
                    "positions": {},
                    "error": str(e),
                })
                all_ground_truth.append({
                    "scene_id": scene_id,
                    "positions": {},
                })

        # Compute metrics
        metrics = self._compute_batch_metrics(all_predictions, all_ground_truth)

        # Save if configured
        if self.config.save_predictions and self.config.output_dir:
            self._save_results(all_predictions, all_ground_truth, metrics)

        return T3Result(
            metrics=metrics,
            predictions=all_predictions,
            ground_truth=all_ground_truth,
            config=self.config,
            errors=errors,
        )

    def _compute_batch_metrics(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
    ) -> T3Metrics:
        """
        计算所有场景的指标。

        Compute metrics across all scenes.
        """
        nrms_values = []
        satisfaction_rates = []
        valid_count = 0

        for pred, gt in zip(predictions, ground_truth):
            if "error" in pred or not pred.get("positions") or not gt.get("positions"):
                continue

            pred_positions = pred["positions"]
            gt_positions = gt["positions"]

            # Align object IDs
            common_objs = set(pred_positions.keys()) & set(gt_positions.keys())
            if len(common_objs) < 2:
                continue

            # Build arrays
            sorted_objs = sorted(common_objs)
            pred_array = np.array([pred_positions[obj] for obj in sorted_objs])
            gt_array = np.array([gt_positions[obj] for obj in sorted_objs])

            # Handle dimension mismatch
            if pred_array.shape[1] != gt_array.shape[1]:
                min_dim = min(pred_array.shape[1], gt_array.shape[1])
                pred_array = pred_array[:, :min_dim]
                gt_array = gt_array[:, :min_dim]

            try:
                metrics = compute_t3_metrics(pred_array, gt_array)
                nrms_values.append(metrics.nrms)
                satisfaction_rates.append(metrics.constraint_satisfaction_rate)
                valid_count += 1
            except Exception as e:
                logger.warning(f"Metrics computation failed: {e}")
                continue

        # Aggregate
        if nrms_values:
            return T3Metrics(
                nrms=float(np.mean(nrms_values)),
                constraint_satisfaction_rate=float(np.mean(satisfaction_rates)),
                scale=1.0,
            )
        else:
            return T3Metrics(
                nrms=float('inf'),
                constraint_satisfaction_rate=0.0,
                scale=1.0,
            )

    def _save_results(self, predictions, ground_truth, metrics):
        """
        保存结果到输出目录。

        Save results to output directory.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_dir / "predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)

        with open(output_dir / "ground_truth.json", "w") as f:
            json.dump(ground_truth, f, indent=2)

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)


def run_t3_evaluation(
    dataset: List[Dict],
    n_dims: int = 3,
    max_iterations: int = 1000,
) -> T3Metrics:
    """
    Quick function to run T3 evaluation.

    Args:
        dataset: Evaluation dataset with constraints
        n_dims: Embedding dimensions
        max_iterations: Max optimization iterations

    Returns:
        T3Metrics
    """
    config = T3Config(
        n_dims=n_dims,
        max_iterations=max_iterations,
        save_predictions=False,
    )
    runner = T3Runner(config=config)
    result = runner.run(dataset)
    return result.metrics


def reconstruct_from_constraints(
    qrr_constraints: List[Dict],
    n_dims: int = 3,
    max_iterations: int = 1000,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Reconstruct point positions from QRR constraints.

    Args:
        qrr_constraints: List of QRR constraint dicts
        n_dims: Output dimensions
        max_iterations: Optimization iterations

    Returns:
        (positions array, object_id to index mapping)
    """
    # Determine objects
    obj_ids = set()
    for c in qrr_constraints:
        for obj in c.get("pair1", []) + c.get("pair2", []):
            obj_ids.add(obj)

    n_points = len(obj_ids)
    obj_to_idx = {obj: i for i, obj in enumerate(sorted(obj_ids))}

    config = T3Config(
        n_dims=n_dims,
        max_iterations=max_iterations,
    )
    runner = T3Runner(config=config)

    # Run embedding
    positions = runner.embedder.fit(n_points, qrr_constraints)

    return positions, obj_to_idx
