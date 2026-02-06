"""
任务 T1：序关系分类运行器。

T1-Q: QRR（四元相对关系）分类
      给定四个物体 A, B, C, D，判断 dist(A,B) 与 dist(C,D) 的大小关系
      输出：< (小于) | ~= (约等于) | > (大于)

T1-C: TRR（三元时钟关系）分类
      给定三个物体 A (目标), B (原点), C (参考方向)
      判断 A 相对于 B->C 轴的时钟位置
      输出：1-12 小时 + 1-4 象限

评估指标：
- T1-Q: 准确率、宏平均F1、序距离误差
- T1-C: 小时准确率、象限准确率、角度误差

增强记录（论文级）：
- raw_response: 原始VLM回复
- ground_truth: 完整GT（含difficulty, ratio, boundary_flag）
- evaluation: 每个query的正确性、ordinal_distance_error、is_flip

Task T1: Ordinal Classification runners.

T1-Q: QRR (Quaternary Relative Relations) classification
T1-C: TRR (Ternary Clock Relations) classification
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

from ordinal_spatial.evaluation.metrics import (
    compute_t1_qrr_metrics,
    compute_t1_trr_metrics,
    T1QRRMetrics,
    T1TRRMetrics,
    normalize_comparator,
)
from ordinal_spatial.dsl.comparators import Comparator, ordinal_distance, is_flip

logger = logging.getLogger(__name__)


@dataclass
class T1Config:
    """Configuration for T1 task runner."""
    tau: float = 0.10
    with_cot: bool = False
    save_predictions: bool = True
    output_dir: Optional[str] = None
    save_detailed: bool = True  # Save enhanced predictions with raw_response, evaluation


@dataclass
class T1Result:
    """Result from T1 evaluation."""
    metrics: Union[T1QRRMetrics, T1TRRMetrics]
    predictions: List[Dict]
    ground_truth: List[Dict]
    config: T1Config
    errors: List[str] = field(default_factory=list)


class T1QRRRunner:
    """
    Runner for T1-Q (QRR classification) task.

    Evaluates a baseline's ability to compare pairwise distances
    between objects in a scene.
    """

    def __init__(self, baseline, config: T1Config = None):
        """
        Initialize T1-Q runner.

        Args:
            baseline: Baseline implementation with predict_qrr method
            config: Task configuration
        """
        self.baseline = baseline
        self.config = config or T1Config()

    def run(
        self,
        dataset: List[Dict],
        images_dir: Optional[str] = None,
    ) -> T1Result:
        """
        Run T1-Q evaluation on dataset.

        Args:
            dataset: List of scene/query data
            images_dir: Directory containing scene images

        Returns:
            T1Result with metrics and predictions
        """
        predictions = []
        ground_truth = []
        detailed = []  # 增强记录
        errors = []

        for item in dataset:
            scene = item.get("scene", item)
            scene_id = scene.get("scene_id", "unknown")
            queries = item.get("qrr_queries", [])

            # Get image if available
            image = None
            image_path_str = item.get("image_path", "")
            if images_dir:
                if image_path_str:
                    full_path = Path(images_dir) / image_path_str
                    if full_path.exists():
                        image = str(full_path)

            # Build objects dict
            objects = {
                obj.get("id", f"obj_{i}"): obj
                for i, obj in enumerate(scene.get("objects", []))
            }

            for query in queries:
                query_id = query.get("query_id", "")

                # Extract full ground truth info
                gt_raw = query.get("ground_truth", "~=")
                if isinstance(gt_raw, dict):
                    gt_comparator = gt_raw.get("comparator", "~=")
                    gt_difficulty = gt_raw.get("difficulty")
                    gt_ratio = gt_raw.get("ratio")
                    gt_boundary_flag = gt_raw.get("boundary_flag")
                else:
                    gt_comparator = gt_raw
                    gt_difficulty = query.get("difficulty")
                    gt_ratio = query.get("ratio")
                    gt_boundary_flag = query.get("boundary_flag")

                try:
                    # Get prediction from baseline
                    raw_response = None
                    parse_success = True
                    parse_error = None
                    reasoning = None

                    if hasattr(self.baseline, "predict_qrr"):
                        if image and hasattr(self.baseline, "config"):
                            # VLM baseline with image
                            pred = self.baseline.predict_qrr(
                                image,
                                objects,
                                (query["A"], query["B"]),
                                (query["C"], query["D"]),
                                metric=query.get("metric", "dist3D"),
                                tau=self.config.tau,
                            )
                            # Extract enhanced fields
                            raw_response = pred.get("raw_response")
                            reasoning = pred.get("reasoning")
                            parse_error = pred.get("parse_error")
                            parse_success = not bool(parse_error)
                        else:
                            # Oracle baseline without image
                            from ordinal_spatial.dsl.schema import QRRQuery
                            q = QRRQuery(
                                scene_id=scene.get("scene_id", ""),
                                query_id=query_id,
                                objects={
                                    "A": query["A"],
                                    "B": query["B"],
                                    "C": query["C"],
                                    "D": query["D"],
                                },
                                metric=query.get("metric", "dist3D"),
                                tau=self.config.tau,
                            )
                            pred_obj = self.baseline.predict_qrr(objects, q)
                            pred = {
                                "comparator": pred_obj.comparator,
                                "confidence": pred_obj.confidence,
                            }
                    else:
                        raise AttributeError("Baseline missing predict_qrr method")

                    pred_comparator = normalize_comparator(pred.get("comparator", "~="))
                    pred_confidence = pred.get("confidence", 0.5)

                    # Compute evaluation metrics for this query
                    gt_comp_normalized = normalize_comparator(gt_comparator)
                    is_correct = (pred_comparator == gt_comp_normalized)

                    pred_c = Comparator.from_string(pred_comparator)
                    gt_c = Comparator.from_string(gt_comp_normalized)
                    ord_dist_err = ordinal_distance(pred_c, gt_c)
                    is_flip_error = is_flip(pred_c, gt_c)

                    predictions.append({
                        "query_id": query_id,
                        "comparator": pred_comparator,
                        "confidence": pred_confidence,
                    })

                    ground_truth.append({
                        "query_id": query_id,
                        "comparator": gt_comp_normalized,
                    })

                    # Enhanced detailed record
                    detailed.append({
                        "query_id": query_id,
                        "scene_id": scene_id,
                        "image_path": image_path_str,

                        "prediction": {
                            "comparator": pred_comparator,
                            "confidence": pred_confidence,
                            "reasoning": reasoning,
                        },

                        "ground_truth": {
                            "comparator": gt_comp_normalized,
                            "difficulty": gt_difficulty,
                            "ratio": gt_ratio,
                            "boundary_flag": gt_boundary_flag,
                        },

                        "evaluation": {
                            "is_correct": is_correct,
                            "ordinal_distance_error": ord_dist_err,
                            "is_flip": is_flip_error,
                        },

                        "raw_response": raw_response,
                        "parse_success": parse_success,
                        "parse_error": str(parse_error) if parse_error else None,
                    })

                except Exception as e:
                    logger.warning(f"Error on query {query_id}: {e}")
                    errors.append(f"{query_id}: {e}")
                    predictions.append({
                        "query_id": query_id,
                        "comparator": "~=",
                        "confidence": 0.0,
                        "error": str(e),
                    })
                    ground_truth.append({
                        "query_id": query_id,
                        "comparator": normalize_comparator(gt_comparator),
                    })
                    # Error detailed record
                    detailed.append({
                        "query_id": query_id,
                        "scene_id": scene_id,
                        "image_path": image_path_str,
                        "prediction": {"comparator": "~=", "confidence": 0.0, "reasoning": None},
                        "ground_truth": {
                            "comparator": normalize_comparator(gt_comparator),
                            "difficulty": gt_difficulty,
                            "ratio": gt_ratio,
                            "boundary_flag": gt_boundary_flag,
                        },
                        "evaluation": {"is_correct": False, "ordinal_distance_error": 0, "is_flip": False},
                        "raw_response": None,
                        "parse_success": False,
                        "parse_error": str(e),
                    })

        # Compute metrics
        metrics = compute_t1_qrr_metrics(predictions, ground_truth)

        # Save if configured
        if self.config.save_predictions and self.config.output_dir:
            self._save_results(predictions, ground_truth, metrics, detailed)

        return T1Result(
            metrics=metrics,
            predictions=predictions,
            ground_truth=ground_truth,
            config=self.config,
            errors=errors,
        )

    def _save_results(self, predictions, ground_truth, metrics, detailed=None):
        """
        保存结果到输出目录。

        Save results to output directory.
        Includes enhanced detailed predictions with raw_response, evaluation.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Basic outputs (backward compatible)
        with open(output_dir / "predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        # Enhanced detailed predictions (论文级记录)
        if detailed and self.config.save_detailed:
            with open(output_dir / "predictions_detailed.json", "w") as f:
                json.dump(detailed, f, indent=2, ensure_ascii=False)

            # Compute metrics by difficulty
            by_difficulty = {}
            by_boundary = {"boundary": [], "non_boundary": []}

            for d in detailed:
                difficulty = d["ground_truth"].get("difficulty")
                boundary_flag = d["ground_truth"].get("boundary_flag")
                is_correct = d["evaluation"]["is_correct"]

                if difficulty is not None:
                    if difficulty not in by_difficulty:
                        by_difficulty[difficulty] = {"correct": 0, "total": 0}
                    by_difficulty[difficulty]["total"] += 1
                    if is_correct:
                        by_difficulty[difficulty]["correct"] += 1

                if boundary_flag is not None:
                    key = "boundary" if boundary_flag else "non_boundary"
                    by_boundary[key].append(is_correct)

            # Format difficulty metrics
            difficulty_metrics = {}
            for diff, counts in sorted(by_difficulty.items()):
                acc = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
                difficulty_metrics[f"difficulty_{diff}"] = {
                    "accuracy": acc,
                    "n_samples": counts["total"],
                }

            # Format boundary metrics
            boundary_metrics = {}
            for key, results in by_boundary.items():
                if results:
                    boundary_metrics[key] = {
                        "accuracy": sum(results) / len(results),
                        "n_samples": len(results),
                    }

            metrics_by_difficulty = {
                "overall": metrics.to_dict(),
                "by_difficulty": difficulty_metrics,
                "by_boundary": boundary_metrics,
            }

            with open(output_dir / "metrics_by_difficulty.json", "w") as f:
                json.dump(metrics_by_difficulty, f, indent=2)

            logger.info(f"Saved detailed predictions to {output_dir / 'predictions_detailed.json'}")
            logger.info(f"Saved difficulty metrics to {output_dir / 'metrics_by_difficulty.json'}")


class T1TRRRunner:
    """
    Runner for T1-C (TRR classification) task.

    Evaluates a baseline's ability to determine clock-face
    directional positions.
    """

    def __init__(self, baseline, config: T1Config = None):
        """
        Initialize T1-C runner.

        Args:
            baseline: Baseline implementation with predict_trr method
            config: Task configuration
        """
        self.baseline = baseline
        self.config = config or T1Config()

    def run(
        self,
        dataset: List[Dict],
        images_dir: Optional[str] = None,
    ) -> T1Result:
        """
        Run T1-C evaluation on dataset.

        Args:
            dataset: List of scene/query data
            images_dir: Directory containing scene images

        Returns:
            T1Result with metrics and predictions
        """
        predictions = []
        ground_truth = []
        errors = []

        for item in dataset:
            scene = item.get("scene", item)
            queries = item.get("trr_queries", [])

            image = None
            if images_dir:
                image_path = item.get("image_path", "")
                if image_path:
                    full_path = Path(images_dir) / image_path
                    if full_path.exists():
                        image = str(full_path)

            objects = {
                obj.get("id", f"obj_{i}"): obj
                for i, obj in enumerate(scene.get("objects", []))
            }

            for query in queries:
                query_id = query.get("query_id", "")

                try:
                    if hasattr(self.baseline, "predict_trr"):
                        if image and hasattr(self.baseline, "config"):
                            pred = self.baseline.predict_trr(
                                image,
                                objects,
                                query["target"],
                                query["ref1"],
                                query["ref2"],
                            )
                        else:
                            from ordinal_spatial.dsl.schema import TRRQuery
                            q = TRRQuery(
                                scene_id=scene.get("scene_id", ""),
                                query_id=query_id,
                                target=query["target"],
                                ref1=query["ref1"],
                                ref2=query["ref2"],
                            )
                            pred_obj = self.baseline.predict_trr(objects, q)
                            pred = {
                                "hour": pred_obj.hour,
                                "quadrant": pred_obj.quadrant,
                                "confidence": pred_obj.confidence,
                            }
                    else:
                        raise AttributeError("Baseline missing predict_trr method")

                    predictions.append({
                        "query_id": query_id,
                        "hour": pred.get("hour", 12),
                        "quadrant": pred.get("quadrant", 1),
                        "confidence": pred.get("confidence", 0.5),
                    })

                    gt = query.get("ground_truth", {})
                    ground_truth.append({
                        "query_id": query_id,
                        "hour": gt.get("hour", 12),
                        "quadrant": gt.get("quadrant", 1),
                    })

                except Exception as e:
                    logger.warning(f"Error on query {query_id}: {e}")
                    errors.append(f"{query_id}: {e}")
                    predictions.append({
                        "query_id": query_id,
                        "hour": 12,
                        "quadrant": 1,
                        "confidence": 0.0,
                        "error": str(e),
                    })
                    gt = query.get("ground_truth", {})
                    ground_truth.append({
                        "query_id": query_id,
                        "hour": gt.get("hour", 12),
                        "quadrant": gt.get("quadrant", 1),
                    })

        metrics = compute_t1_trr_metrics(predictions, ground_truth)

        if self.config.save_predictions and self.config.output_dir:
            self._save_results(predictions, ground_truth, metrics)

        return T1Result(
            metrics=metrics,
            predictions=predictions,
            ground_truth=ground_truth,
            config=self.config,
            errors=errors,
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

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)


# =============================================================================
# Convenience Functions
# =============================================================================

def run_t1_qrr_evaluation(
    baseline,
    dataset: List[Dict],
    tau: float = 0.10,
    images_dir: Optional[str] = None,
) -> T1QRRMetrics:
    """
    Quick function to run T1-Q evaluation.

    Args:
        baseline: Baseline implementation
        dataset: Evaluation dataset
        tau: Tolerance parameter
        images_dir: Optional images directory

    Returns:
        T1QRRMetrics
    """
    config = T1Config(tau=tau, save_predictions=False)
    runner = T1QRRRunner(baseline, config)
    result = runner.run(dataset, images_dir)
    return result.metrics


def run_t1_trr_evaluation(
    baseline,
    dataset: List[Dict],
    images_dir: Optional[str] = None,
) -> T1TRRMetrics:
    """
    Quick function to run T1-C evaluation.

    Args:
        baseline: Baseline implementation
        dataset: Evaluation dataset
        images_dir: Optional images directory

    Returns:
        T1TRRMetrics
    """
    config = T1Config(save_predictions=False)
    runner = T1TRRRunner(baseline, config)
    result = runner.run(dataset, images_dir)
    return result.metrics
