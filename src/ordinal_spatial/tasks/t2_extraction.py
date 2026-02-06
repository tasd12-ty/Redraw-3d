"""
任务 T2：约束提取运行器。

评估基线从图像中提取完整、一致的序空间约束集的能力。

任务描述：
- 输入：场景图像 + 物体列表
- 输出：完整的 QRR 和 TRR 约束集
- 目标：提取所有或大部分约束，且保持逻辑一致性

评估指标：
- 精确率（Precision）：预测约束中正确的比例
- 召回率（Recall）：真值约束中被发现的比例
- F1 分数：精确率和召回率的调和平均
- 一致性：约束集是否无矛盾
- 调整后 F1：F1 × 一致性

一致性检查：
- 使用图循环检测算法
- 发现传递性违反（如 A<B<C<A）
- 统计一致和不一致的样本数

增强记录（论文级）：
- raw_response: 原始VLM回复
- parsed_constraints: 解析后的约束
- constraint_diff: 正确/错误/缺失/多余分类
- ground_truth_constraints: 真值约束

Task T2: Constraint Extraction runner.

Evaluates a baseline's ability to extract complete, consistent
sets of ordinal spatial constraints from images.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging

from ordinal_spatial.evaluation.metrics import compute_t2_metrics, T2Metrics, normalize_comparator
from ordinal_spatial.evaluation.consistency import check_qrr_consistency
from ordinal_spatial.dsl.comparators import Comparator, is_flip

logger = logging.getLogger(__name__)


@dataclass
class T2Config:
    """Configuration for T2 task runner."""
    tau: float = 0.10
    with_cot: bool = False
    check_consistency: bool = True
    save_predictions: bool = True
    output_dir: Optional[str] = None
    save_detailed: bool = True  # Save enhanced predictions with constraint_diff


def _qrr_key(c: Dict) -> Tuple:
    """
    Create canonical key for QRR constraint matching.

    匹配规则: (pair1, pair2, metric)，允许对象对顺序交换
    """
    p1 = tuple(sorted(c.get("pair1", [])))
    p2 = tuple(sorted(c.get("pair2", [])))
    metric = c.get("metric", "dist3D")
    # Canonicalize order
    if p1 > p2:
        p1, p2 = p2, p1
    return (p1, p2, metric)


def compute_constraint_diff(
    predicted_qrr: List[Dict],
    ground_truth_qrr: List[Dict],
) -> Dict[str, List[Dict]]:
    """
    计算约束差异分类。

    Compute constraint difference classification:
    - correct: 预测正确（对象对和comparator都匹配）
    - incorrect: 对象对匹配但comparator错误（含is_flip标记）
    - missing: GT中有但预测中没有
    - spurious: 预测中有但GT中没有

    Args:
        predicted_qrr: List of predicted QRR constraints
        ground_truth_qrr: List of ground truth QRR constraints

    Returns:
        Dict with correct, incorrect, missing, spurious lists
    """
    # Build lookups
    pred_by_key = {}
    for i, c in enumerate(predicted_qrr):
        key = _qrr_key(c)
        pred_by_key[key] = {
            "index": i,
            "constraint": c,
            "comparator": normalize_comparator(c.get("comparator", "~=")),
        }

    gt_by_key = {}
    for i, c in enumerate(ground_truth_qrr):
        key = _qrr_key(c)
        gt_by_key[key] = {
            "index": i,
            "constraint": c,
            "comparator": normalize_comparator(c.get("comparator", "~=")),
        }

    correct = []
    incorrect = []
    missing = []
    spurious = []

    # Check predicted constraints
    for key, pred_info in pred_by_key.items():
        pred_c = pred_info["constraint"]
        pred_comp = pred_info["comparator"]

        if key in gt_by_key:
            gt_info = gt_by_key[key]
            gt_c = gt_info["constraint"]
            gt_comp = gt_info["comparator"]

            if pred_comp == gt_comp:
                # Correct
                correct.append({
                    "type": "qrr",
                    "predicted": {
                        "pair1": pred_c.get("pair1", []),
                        "pair2": pred_c.get("pair2", []),
                        "metric": pred_c.get("metric", "dist3D"),
                        "comparator": pred_comp,
                    },
                    "ground_truth": {
                        "pair1": gt_c.get("pair1", []),
                        "pair2": gt_c.get("pair2", []),
                        "metric": gt_c.get("metric", "dist3D"),
                        "comparator": gt_comp,
                        "ratio": gt_c.get("ratio"),
                        "difficulty": gt_c.get("difficulty"),
                        "boundary_flag": gt_c.get("boundary_flag"),
                    },
                })
            else:
                # Incorrect
                is_flip_error = is_flip(
                    Comparator.from_string(pred_comp),
                    Comparator.from_string(gt_comp)
                )
                incorrect.append({
                    "type": "qrr",
                    "predicted": {
                        "pair1": pred_c.get("pair1", []),
                        "pair2": pred_c.get("pair2", []),
                        "metric": pred_c.get("metric", "dist3D"),
                        "comparator": pred_comp,
                    },
                    "ground_truth": {
                        "pair1": gt_c.get("pair1", []),
                        "pair2": gt_c.get("pair2", []),
                        "metric": gt_c.get("metric", "dist3D"),
                        "comparator": gt_comp,
                        "ratio": gt_c.get("ratio"),
                        "difficulty": gt_c.get("difficulty"),
                        "boundary_flag": gt_c.get("boundary_flag"),
                    },
                    "is_flip": is_flip_error,
                })
        else:
            # Spurious
            spurious.append({
                "type": "qrr",
                "predicted": {
                    "pair1": pred_c.get("pair1", []),
                    "pair2": pred_c.get("pair2", []),
                    "metric": pred_c.get("metric", "dist3D"),
                    "comparator": pred_comp,
                },
            })

    # Check missing constraints
    for key, gt_info in gt_by_key.items():
        if key not in pred_by_key:
            gt_c = gt_info["constraint"]
            missing.append({
                "type": "qrr",
                "ground_truth": {
                    "pair1": gt_c.get("pair1", []),
                    "pair2": gt_c.get("pair2", []),
                    "metric": gt_c.get("metric", "dist3D"),
                    "comparator": gt_info["comparator"],
                    "ratio": gt_c.get("ratio"),
                    "difficulty": gt_c.get("difficulty"),
                    "boundary_flag": gt_c.get("boundary_flag"),
                },
            })

    return {
        "correct": correct,
        "incorrect": incorrect,
        "missing": missing,
        "spurious": spurious,
    }


@dataclass
class T2Result:
    """Result from T2 evaluation."""
    metrics: T2Metrics
    predictions: List[Dict]
    ground_truth: List[Dict]
    consistency_stats: Dict[str, Any]
    config: T2Config
    errors: List[str] = field(default_factory=list)


class T2Runner:
    """
    Runner for T2 (Constraint Extraction) task.

    Evaluates a baseline's ability to extract complete and
    consistent constraint sets from scene images.
    """

    def __init__(self, baseline, config: T2Config = None):
        """
        Initialize T2 runner.

        Args:
            baseline: Baseline with extract_constraints method
            config: Task configuration
        """
        self.baseline = baseline
        self.config = config or T2Config()

    def run(
        self,
        dataset: List[Dict],
        images_dir: Optional[str] = None,
    ) -> T2Result:
        """
        Run T2 evaluation on dataset.

        Args:
            dataset: List of scene data with ground truth
            images_dir: Directory containing scene images

        Returns:
            T2Result with metrics and predictions
        """
        all_predictions = []
        all_ground_truth = []
        all_detailed = []  # 增强记录：包含constraint_diff
        errors = []

        consistency_results = {
            "n_consistent": 0,
            "n_inconsistent": 0,
            "n_cycles_total": 0,
        }

        for item in dataset:
            scene = item.get("scene", item)
            scene_id = scene.get("scene_id", "unknown")

            # Get image path for recording
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

            # Get ground truth first (needed for constraint_diff)
            gt = item.get("ground_truth", item.get("constraints", {}))
            gt_qrr = gt.get("qrr", scene.get("constraints", {}).get("qrr", []))
            gt_trr = gt.get("trr", scene.get("constraints", {}).get("trr", []))

            gt_record = {
                "scene_id": scene_id,
                "qrr": gt_qrr,
                "trr": gt_trr,
            }
            all_ground_truth.append(gt_record)

            try:
                # Get prediction from baseline
                raw_response = None
                parse_success = True
                parse_error = None

                if hasattr(self.baseline, "extract_constraints"):
                    if image:
                        pred = self.baseline.extract_constraints(
                            image, objects, tau=self.config.tau
                        )
                        # Extract raw_response if available
                        raw_response = pred.get("raw_response")
                        parse_error = pred.get("parse_error")
                        parse_success = not bool(parse_error)
                    else:
                        # Oracle baseline
                        pred = self.baseline.extract_osd(scene)
                        if hasattr(pred, "model_dump"):
                            pred = pred.model_dump()
                        elif hasattr(pred, "to_dict"):
                            pred = pred.to_dict()
                elif hasattr(self.baseline, "extract_osd"):
                    pred = self.baseline.extract_osd(scene)
                    if hasattr(pred, "model_dump"):
                        pred = pred.model_dump()
                else:
                    raise AttributeError("Baseline missing extraction method")

                # Check consistency
                consistency_info = {"is_consistent": True, "n_cycles": 0, "cycles": []}
                if self.config.check_consistency:
                    qrr = pred.get("qrr", [])
                    report = check_qrr_consistency(qrr)

                    if report.is_consistent:
                        consistency_results["n_consistent"] += 1
                    else:
                        consistency_results["n_inconsistent"] += 1
                        consistency_results["n_cycles_total"] += len(report.cycles)

                    consistency_info = {
                        "is_consistent": report.is_consistent,
                        "n_cycles": len(report.cycles),
                        "cycles": [str(c) for c in report.cycles] if report.cycles else [],
                    }
                    pred["consistency"] = consistency_info

                # Compute constraint_diff
                pred_qrr = pred.get("qrr", [])
                constraint_diff = compute_constraint_diff(pred_qrr, gt_qrr)

                # Scene-level metrics
                n_correct = len(constraint_diff["correct"])
                n_incorrect = len(constraint_diff["incorrect"])
                n_missing = len(constraint_diff["missing"])
                n_spurious = len(constraint_diff["spurious"])
                n_flip = sum(1 for c in constraint_diff["incorrect"] if c.get("is_flip", False))

                n_pred = len(pred_qrr)
                n_gt = len(gt_qrr)

                scene_precision = n_correct / n_pred if n_pred > 0 else 0.0
                scene_recall = n_correct / n_gt if n_gt > 0 else 0.0
                scene_f1 = (2 * scene_precision * scene_recall /
                           (scene_precision + scene_recall) if (scene_precision + scene_recall) > 0 else 0.0)

                # Basic prediction record
                all_predictions.append({
                    "scene_id": scene_id,
                    **pred,
                })

                # Enhanced detailed record
                detailed_record = {
                    "scene_id": scene_id,
                    "image_path": image_path_str,

                    # Raw and parsed
                    "raw_response": raw_response,
                    "parse_success": parse_success,
                    "parse_error": str(parse_error) if parse_error else None,

                    # Parsed constraints
                    "parsed_constraints": {
                        "qrr": pred_qrr,
                        "trr": pred.get("trr", []),
                    },

                    # Ground truth constraints
                    "ground_truth_constraints": {
                        "qrr": gt_qrr,
                        "trr": gt_trr,
                    },

                    # Constraint diff (核心分类)
                    "constraint_diff": constraint_diff,

                    # Scene-level metrics
                    "scene_metrics": {
                        "n_correct": n_correct,
                        "n_incorrect": n_incorrect,
                        "n_missing": n_missing,
                        "n_spurious": n_spurious,
                        "n_flip": n_flip,
                        "precision": scene_precision,
                        "recall": scene_recall,
                        "f1": scene_f1,
                    },

                    # Consistency
                    "consistency": consistency_info,
                }
                all_detailed.append(detailed_record)

            except Exception as e:
                logger.warning(f"Error on scene {scene_id}: {e}")
                errors.append(f"{scene_id}: {e}")
                all_predictions.append({
                    "scene_id": scene_id,
                    "qrr": [],
                    "trr": [],
                    "error": str(e),
                })
                # Error detailed record
                all_detailed.append({
                    "scene_id": scene_id,
                    "image_path": image_path_str,
                    "raw_response": None,
                    "parse_success": False,
                    "parse_error": str(e),
                    "parsed_constraints": {"qrr": [], "trr": []},
                    "ground_truth_constraints": {"qrr": gt_qrr, "trr": gt_trr},
                    "constraint_diff": {"correct": [], "incorrect": [], "missing": [], "spurious": []},
                    "scene_metrics": {
                        "n_correct": 0, "n_incorrect": 0, "n_missing": len(gt_qrr),
                        "n_spurious": 0, "n_flip": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    },
                    "consistency": {"is_consistent": True, "n_cycles": 0, "cycles": []},
                    "error": str(e),
                })

        # Compute aggregated metrics
        metrics = self._compute_batch_metrics(all_predictions, all_ground_truth)

        # Save if configured
        if self.config.save_predictions and self.config.output_dir:
            self._save_results(all_predictions, all_ground_truth, metrics, all_detailed)

        return T2Result(
            metrics=metrics,
            predictions=all_predictions,
            ground_truth=all_ground_truth,
            consistency_stats=consistency_results,
            config=self.config,
            errors=errors,
        )

    def _compute_batch_metrics(
        self,
        predictions: List[Dict],
        ground_truth: List[Dict],
    ) -> T2Metrics:
        """
        计算所有场景的指标。

        Compute metrics across all scenes.
        """
        from ordinal_spatial.evaluation.metrics import compute_t2_metrics_batch

        return compute_t2_metrics_batch(predictions, ground_truth)

    def _save_results(self, predictions, ground_truth, metrics, detailed=None):
        """
        保存结果到输出目录。

        Save results to output directory.
        Includes enhanced detailed predictions with constraint_diff.
        """
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Basic outputs (backward compatible)
        with open(output_dir / "predictions.json", "w") as f:
            json.dump(predictions, f, indent=2)

        with open(output_dir / "ground_truth.json", "w") as f:
            json.dump(ground_truth, f, indent=2)

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics.to_dict(), f, indent=2)

        # Enhanced detailed predictions (论文级记录)
        if detailed and self.config.save_detailed:
            with open(output_dir / "predictions_detailed.json", "w") as f:
                json.dump(detailed, f, indent=2, ensure_ascii=False)

            # Constraint diff summary across all scenes
            total_correct = sum(len(d["constraint_diff"]["correct"]) for d in detailed)
            total_incorrect = sum(len(d["constraint_diff"]["incorrect"]) for d in detailed)
            total_missing = sum(len(d["constraint_diff"]["missing"]) for d in detailed)
            total_spurious = sum(len(d["constraint_diff"]["spurious"]) for d in detailed)
            total_flip = sum(d["scene_metrics"]["n_flip"] for d in detailed)

            constraint_diff_all = {
                "total_summary": {
                    "n_scenes": len(detailed),
                    "n_correct": total_correct,
                    "n_incorrect": total_incorrect,
                    "n_missing": total_missing,
                    "n_spurious": total_spurious,
                    "n_flip": total_flip,
                },
                "per_scene": [
                    {
                        "scene_id": d["scene_id"],
                        "image_path": d["image_path"],
                        "n_correct": len(d["constraint_diff"]["correct"]),
                        "n_incorrect": len(d["constraint_diff"]["incorrect"]),
                        "n_missing": len(d["constraint_diff"]["missing"]),
                        "n_spurious": len(d["constraint_diff"]["spurious"]),
                        "n_flip": d["scene_metrics"]["n_flip"],
                        "correct": d["constraint_diff"]["correct"],
                        "incorrect": d["constraint_diff"]["incorrect"],
                        "missing": d["constraint_diff"]["missing"],
                        "spurious": d["constraint_diff"]["spurious"],
                    }
                    for d in detailed
                ]
            }

            with open(output_dir / "constraint_diff_all.json", "w") as f:
                json.dump(constraint_diff_all, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved detailed predictions to {output_dir / 'predictions_detailed.json'}")
            logger.info(f"Saved constraint diff to {output_dir / 'constraint_diff_all.json'}")


def run_t2_evaluation(
    baseline,
    dataset: List[Dict],
    tau: float = 0.10,
    images_dir: Optional[str] = None,
) -> T2Metrics:
    """
    Quick function to run T2 evaluation.

    Args:
        baseline: Baseline implementation
        dataset: Evaluation dataset
        tau: Tolerance parameter
        images_dir: Optional images directory

    Returns:
        T2Metrics
    """
    config = T2Config(tau=tau, save_predictions=False)
    runner = T2Runner(baseline, config)
    result = runner.run(dataset, images_dir)
    return result.metrics
