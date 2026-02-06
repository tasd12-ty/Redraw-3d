"""
ORDINAL-SPATIAL 基准测试的评估指标。

本模块实现了三个任务的评估指标：

T1（分类任务）：
    - 准确率（精确匹配）
    - 宏平均准确率（类别平衡）
    - 序距离误差（惩罚翻转错误）
    - 时钟角度误差（TRR专用）

T2（约束提取任务）：
    - 精确率、召回率、F1分数
    - 翻转率（严重错误）
    - 自一致性得分
    - 调整后F1（F1 × 一致性）

T3（序重构任务）：
    - Procrustes对齐RMS
    - 归一化RMS（NRMS）
    - 约束满足率

所有指标都提供 to_dict() 方法用于序列化。

Evaluation metrics for ORDINAL-SPATIAL benchmark.

This module implements metrics for all three tasks:

T1 (Classification):
    - Accuracy (exact match)
    - Macro-averaged accuracy (class-balanced)
    - Ordinal distance error (penalizes flips)
    - Clock angular error (for TRR)

T2 (Constraint Extraction):
    - Precision, Recall, F1
    - Flip rate (severe errors)
    - Self-consistency score
    - Adjusted F1 (F1 × consistency)

T3 (Reconstruction):
    - Procrustes-aligned RMS
    - Normalized RMS (NRMS)
    - Constraint satisfaction rate
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter
import numpy as np

from ordinal_spatial.dsl.comparators import Comparator, ordinal_distance, is_flip
from ordinal_spatial.dsl.predicates import clock_angular_error
from ordinal_spatial.evaluation.consistency import check_qrr_consistency, check_trr_consistency


# =============================================================================
# Data Classes for Metrics
# =============================================================================

@dataclass
class T1QRRMetrics:
    """Metrics for T1-Q (QRR classification) task."""
    accuracy: float = 0.0
    macro_accuracy: float = 0.0
    ordinal_distance_error: float = 0.0
    flip_rate: float = 0.0
    n_samples: int = 0
    per_class_accuracy: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典。

        Convert to dictionary.
        """
        return {
            "accuracy": self.accuracy,
            "macro_accuracy": self.macro_accuracy,
            "ordinal_distance_error": self.ordinal_distance_error,
            "flip_rate": self.flip_rate,
            "n_samples": self.n_samples,
            "per_class_accuracy": self.per_class_accuracy,
            "confusion_matrix": self.confusion_matrix,
        }


@dataclass
class T1TRRMetrics:
    """
    T1-C（TRR 分类）任务的评估指标。

    Metrics for T1-C (TRR classification) task.
    """
    hour_accuracy: float = 0.0
    quadrant_accuracy: float = 0.0
    mean_angular_error: float = 0.0
    median_angular_error: float = 0.0
    n_samples: int = 0
    angular_error_histogram: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典。

        Convert to dictionary.
        """
        return {
            "hour_accuracy": self.hour_accuracy,
            "quadrant_accuracy": self.quadrant_accuracy,
            "mean_angular_error": self.mean_angular_error,
            "median_angular_error": self.median_angular_error,
            "n_samples": self.n_samples,
            "angular_error_histogram": self.angular_error_histogram,
        }


@dataclass
class T2Metrics:
    """
    T2（约束提取）任务的评估指标。

    Metrics for T2 (constraint extraction) task.
    """
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    flip_rate: float = 0.0
    self_consistency: float = 0.0
    adjusted_f1: float = 0.0
    n_predicted: int = 0
    n_ground_truth: int = 0
    n_correct: int = 0
    n_flips: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典。

        Convert to dictionary.
        """
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "flip_rate": self.flip_rate,
            "self_consistency": self.self_consistency,
            "adjusted_f1": self.adjusted_f1,
            "n_predicted": self.n_predicted,
            "n_ground_truth": self.n_ground_truth,
            "n_correct": self.n_correct,
            "n_flips": self.n_flips,
        }


@dataclass
class T3Metrics:
    """
    T3（序重构）任务的评估指标。

    Metrics for T3 (ordinal reconstruction) task.
    """
    rms_aligned: float = 0.0
    nrms: float = 0.0
    constraint_satisfaction_rate: float = 0.0
    scale: float = 1.0
    n_points: int = 0
    n_satisfied: int = 0
    n_total_constraints: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典。

        Convert to dictionary.
        """
        return {
            "rms_aligned": self.rms_aligned,
            "nrms": self.nrms,
            "constraint_satisfaction_rate": self.constraint_satisfaction_rate,
            "scale": self.scale,
            "n_points": self.n_points,
            "n_satisfied": self.n_satisfied,
            "n_total_constraints": self.n_total_constraints,
        }


# =============================================================================
# T1 Metrics Implementation
# =============================================================================

def compute_t1_qrr_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict]
) -> T1QRRMetrics:
    """
    Compute T1-Q (QRR classification) metrics.

    Args:
        predictions: List of {"query_id": str, "comparator": str, ...}
        ground_truth: List of {"query_id": str, "comparator": str, ...}

    Returns:
        T1QRRMetrics with all classification metrics
    """
    # Build lookup by query_id
    gt_lookup = {g["query_id"]: g for g in ground_truth}

    # Track per-class statistics
    class_correct = Counter()
    class_total = Counter()
    confusion = {c: {c2: 0 for c2 in ["<", "~=", ">"]} for c in ["<", "~=", ">"]}

    n_correct = 0
    n_flips = 0
    total_ord_dist = 0.0
    n_samples = 0

    for pred in predictions:
        qid = pred["query_id"]
        if qid not in gt_lookup:
            continue

        gt = gt_lookup[qid]
        pred_comp = normalize_comparator(pred.get("comparator", "~="))
        gt_comp = normalize_comparator(gt.get("comparator", "~="))

        n_samples += 1
        class_total[gt_comp] += 1

        # Exact match
        if pred_comp == gt_comp:
            n_correct += 1
            class_correct[gt_comp] += 1

        # Ordinal distance
        pred_c = Comparator.from_string(pred_comp)
        gt_c = Comparator.from_string(gt_comp)
        total_ord_dist += ordinal_distance(pred_c, gt_c)

        # Flip detection
        if is_flip(pred_c, gt_c):
            n_flips += 1

        # Confusion matrix
        confusion[gt_comp][pred_comp] += 1

    # Compute metrics
    accuracy = n_correct / n_samples if n_samples > 0 else 0.0

    # Macro accuracy (average per-class accuracy)
    per_class_acc = {}
    for c in ["<", "~=", ">"]:
        if class_total[c] > 0:
            per_class_acc[c] = class_correct[c] / class_total[c]
        else:
            per_class_acc[c] = 0.0

    macro_acc = np.mean(list(per_class_acc.values())) if per_class_acc else 0.0

    # Average ordinal distance error
    ord_dist_err = total_ord_dist / n_samples if n_samples > 0 else 0.0

    # Flip rate
    flip_rate = n_flips / n_samples if n_samples > 0 else 0.0

    return T1QRRMetrics(
        accuracy=accuracy,
        macro_accuracy=macro_acc,
        ordinal_distance_error=ord_dist_err,
        flip_rate=flip_rate,
        n_samples=n_samples,
        per_class_accuracy=per_class_acc,
        confusion_matrix=confusion,
    )


def compute_t1_trr_metrics(
    predictions: List[Dict],
    ground_truth: List[Dict]
) -> T1TRRMetrics:
    """
    Compute T1-C (TRR classification) metrics.

    Args:
        predictions: List of {"query_id": str, "hour": int, "quadrant": int, ...}
        ground_truth: List of {"query_id": str, "hour": int, "quadrant": int, ...}

    Returns:
        T1TRRMetrics with all classification metrics
    """
    gt_lookup = {g["query_id"]: g for g in ground_truth}

    n_hour_correct = 0
    n_quadrant_correct = 0
    angular_errors = []
    n_samples = 0

    for pred in predictions:
        qid = pred["query_id"]
        if qid not in gt_lookup:
            continue

        gt = gt_lookup[qid]
        pred_hour = pred.get("hour", 12)
        gt_hour = gt.get("hour", 12)
        pred_quadrant = pred.get("quadrant", hour_to_quadrant(pred_hour))
        gt_quadrant = gt.get("quadrant", hour_to_quadrant(gt_hour))

        n_samples += 1

        # Hour accuracy
        if pred_hour == gt_hour:
            n_hour_correct += 1

        # Quadrant accuracy
        if pred_quadrant == gt_quadrant:
            n_quadrant_correct += 1

        # Angular error
        ang_err = clock_angular_error(pred_hour, gt_hour)
        angular_errors.append(ang_err)

    # Compute metrics
    hour_acc = n_hour_correct / n_samples if n_samples > 0 else 0.0
    quad_acc = n_quadrant_correct / n_samples if n_samples > 0 else 0.0
    mean_ang_err = np.mean(angular_errors) if angular_errors else 0.0
    median_ang_err = np.median(angular_errors) if angular_errors else 0.0

    # Build histogram (binned by 30-degree increments)
    error_bins = {i: 0 for i in range(0, 181, 30)}
    for err in angular_errors:
        bin_val = int(err // 30) * 30
        bin_val = min(bin_val, 180)
        error_bins[bin_val] += 1

    return T1TRRMetrics(
        hour_accuracy=hour_acc,
        quadrant_accuracy=quad_acc,
        mean_angular_error=mean_ang_err,
        median_angular_error=median_ang_err,
        n_samples=n_samples,
        angular_error_histogram=error_bins,
    )


def hour_to_quadrant(hour: int) -> int:
    """
    将时钟小时数转换为象限。

    Convert clock hour to quadrant.
    """
    if hour in (12, 1, 2):
        return 1
    elif hour in (3, 4, 5):
        return 2
    elif hour in (6, 7, 8):
        return 3
    else:
        return 4


def normalize_comparator(comp: str) -> str:
    """
    将比较器字符串规范化为标准形式。

    Normalize comparator string to canonical form.
    """
    comp = comp.strip().lower()
    if comp in ("<", "lt"):
        return "<"
    elif comp in (">", "gt"):
        return ">"
    else:
        return "~="


# =============================================================================
# T2 Metrics Implementation
# =============================================================================

def compute_t2_metrics(
    predicted: Dict,
    ground_truth: Dict
) -> T2Metrics:
    """
    Compute T2 (constraint extraction) metrics.

    Args:
        predicted: {"qrr": [...], "trr": [...]}
        ground_truth: {"qrr": [...], "trr": [...]}

    Returns:
        T2Metrics with precision, recall, F1, etc.
    """
    # Extract QRR constraints
    pred_qrr = predicted.get("qrr", [])
    gt_qrr = ground_truth.get("qrr", [])

    # Create canonical keys for matching
    def qrr_key(c):
        p1 = tuple(sorted(c.get("pair1", [])))
        p2 = tuple(sorted(c.get("pair2", [])))
        metric = c.get("metric", "dist3D")
        # Canonicalize order
        if p1 > p2:
            p1, p2 = p2, p1
        return (p1, p2, metric)

    def qrr_value(c):
        return normalize_comparator(c.get("comparator", "~="))

    # Build lookup
    gt_lookup = {qrr_key(c): qrr_value(c) for c in gt_qrr}
    pred_lookup = {qrr_key(c): qrr_value(c) for c in pred_qrr}

    # Count matches
    n_correct = 0
    n_flips = 0

    for key, pred_val in pred_lookup.items():
        if key in gt_lookup:
            gt_val = gt_lookup[key]
            if pred_val == gt_val:
                n_correct += 1
            elif is_flip(
                Comparator.from_string(pred_val),
                Comparator.from_string(gt_val)
            ):
                n_flips += 1

    n_pred = len(pred_lookup)
    n_gt = len(gt_lookup)

    # Precision, Recall, F1
    precision = n_correct / n_pred if n_pred > 0 else 0.0
    recall = n_correct / n_gt if n_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Flip rate
    flip_rate = n_flips / n_pred if n_pred > 0 else 0.0

    # Self-consistency
    consistency_report = check_qrr_consistency(pred_qrr)
    self_consistency = 1.0 if consistency_report.is_consistent else 0.0

    # Adjusted F1
    adjusted_f1 = f1 * self_consistency

    return T2Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        flip_rate=flip_rate,
        self_consistency=self_consistency,
        adjusted_f1=adjusted_f1,
        n_predicted=n_pred,
        n_ground_truth=n_gt,
        n_correct=n_correct,
        n_flips=n_flips,
    )


def compute_t2_metrics_batch(
    predictions: List[Dict],
    ground_truths: List[Dict]
) -> T2Metrics:
    """
    Compute aggregated T2 metrics over multiple scenes.

    Args:
        predictions: List of scene predictions
        ground_truths: List of scene ground truths

    Returns:
        Aggregated T2Metrics
    """
    total_correct = 0
    total_pred = 0
    total_gt = 0
    total_flips = 0
    n_consistent = 0
    n_scenes = 0

    for pred, gt in zip(predictions, ground_truths):
        metrics = compute_t2_metrics(pred, gt)
        total_correct += metrics.n_correct
        total_pred += metrics.n_predicted
        total_gt += metrics.n_ground_truth
        total_flips += metrics.n_flips
        if metrics.self_consistency > 0:
            n_consistent += 1
        n_scenes += 1

    precision = total_correct / total_pred if total_pred > 0 else 0.0
    recall = total_correct / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    flip_rate = total_flips / total_pred if total_pred > 0 else 0.0
    self_consistency = n_consistent / n_scenes if n_scenes > 0 else 0.0
    adjusted_f1 = f1 * self_consistency

    return T2Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        flip_rate=flip_rate,
        self_consistency=self_consistency,
        adjusted_f1=adjusted_f1,
        n_predicted=total_pred,
        n_ground_truth=total_gt,
        n_correct=total_correct,
        n_flips=total_flips,
    )


# =============================================================================
# T3 Metrics Implementation
# =============================================================================

def procrustes_align(
    predicted: np.ndarray,
    ground_truth: np.ndarray
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Compute Procrustes alignment with scaling.

    Finds optimal similarity transform (scale, rotation, translation)
    minimizing ||s*R*pred + t - gt||^2

    Args:
        predicted: (n, d) array of predicted positions
        ground_truth: (n, d) array of ground truth positions

    Returns:
        (aligned_pred, scale, rotation, translation)
    """
    # Center both sets
    pred_mean = predicted.mean(axis=0)
    gt_mean = ground_truth.mean(axis=0)

    pred_centered = predicted - pred_mean
    gt_centered = ground_truth - gt_mean

    # Compute optimal rotation using SVD
    H = pred_centered.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute optimal scale
    pred_rotated = pred_centered @ R.T
    scale = np.sum(gt_centered * pred_rotated) / np.sum(pred_rotated * pred_rotated)

    # Compute translation
    translation = gt_mean - scale * (R @ pred_mean)

    # Apply transform
    aligned = scale * (predicted @ R.T) + translation

    return aligned, scale, R, translation


def compute_t3_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    constraints: Optional[List[Dict]] = None,
    tau: float = 0.10
) -> T3Metrics:
    """
    Compute T3 (ordinal reconstruction) metrics.

    Args:
        predicted: (n, d) array of predicted positions
        ground_truth: (n, d) array of ground truth positions
        constraints: Optional list of QRR constraints to check satisfaction
        tau: Tolerance for constraint satisfaction check

    Returns:
        T3Metrics with alignment and satisfaction metrics
    """
    # Procrustes alignment
    aligned, scale, R, t = procrustes_align(predicted, ground_truth)

    # RMS error after alignment
    residuals = aligned - ground_truth
    rms = np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))

    # Normalized RMS (by diameter)
    gt_diameter = np.max(np.linalg.norm(
        ground_truth[:, None, :] - ground_truth[None, :, :],
        axis=2
    ))
    nrms = rms / gt_diameter if gt_diameter > 0 else 0.0

    # Constraint satisfaction (if constraints provided)
    n_satisfied = 0
    n_total = 0

    if constraints:
        from ordinal_spatial.dsl.comparators import compare

        # Build object index mapping (assumes objects are 0..n-1)
        for c in constraints:
            p1 = c.get("pair1", [])
            p2 = c.get("pair2", [])
            gt_comp = normalize_comparator(c.get("comparator", "~="))

            if len(p1) < 2 or len(p2) < 2:
                continue

            # Get indices (assuming object IDs are like "obj_0", "obj_1", etc.)
            try:
                i1, j1 = int(p1[0].split("_")[-1]), int(p1[1].split("_")[-1])
                i2, j2 = int(p2[0].split("_")[-1]), int(p2[1].split("_")[-1])
            except (ValueError, IndexError):
                continue

            if max(i1, j1, i2, j2) >= len(aligned):
                continue

            # Compute distances in reconstructed positions
            d1 = np.linalg.norm(aligned[i1] - aligned[j1])
            d2 = np.linalg.norm(aligned[i2] - aligned[j2])

            pred_comp = str(compare(d1, d2, tau))
            pred_comp = normalize_comparator(pred_comp)

            n_total += 1
            if pred_comp == gt_comp:
                n_satisfied += 1

    csr = n_satisfied / n_total if n_total > 0 else 1.0

    return T3Metrics(
        rms_aligned=rms,
        nrms=nrms,
        constraint_satisfaction_rate=csr,
        scale=scale,
        n_points=len(predicted),
        n_satisfied=n_satisfied,
        n_total_constraints=n_total,
    )


# =============================================================================
# Aggregate Metrics
# =============================================================================

@dataclass
class BenchmarkMetrics:
    """
    完整基准测试的组合指标。

    Combined metrics for the full benchmark.
    """
    t1_qrr: Optional[T1QRRMetrics] = None
    t1_trr: Optional[T1TRRMetrics] = None
    t2: Optional[T2Metrics] = None
    t3: Optional[T3Metrics] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典。

        Convert to dictionary.
        """
        result = {}
        if self.t1_qrr:
            result["t1_qrr"] = self.t1_qrr.to_dict()
        if self.t1_trr:
            result["t1_trr"] = self.t1_trr.to_dict()
        if self.t2:
            result["t2"] = self.t2.to_dict()
        if self.t3:
            result["t3"] = self.t3.to_dict()
        return result

    def summary(self) -> str:
        """
        生成指标摘要字符串。

        Generate summary string of metrics.
        """
        """Generate human-readable summary."""
        lines = ["=== ORDINAL-SPATIAL Benchmark Results ===", ""]

        if self.t1_qrr:
            lines.append("T1-Q (QRR Classification):")
            lines.append(f"  Accuracy: {self.t1_qrr.accuracy:.3f}")
            lines.append(f"  Macro Accuracy: {self.t1_qrr.macro_accuracy:.3f}")
            lines.append(f"  Ordinal Distance Error: {self.t1_qrr.ordinal_distance_error:.3f}")
            lines.append(f"  Flip Rate: {self.t1_qrr.flip_rate:.3f}")
            lines.append("")

        if self.t1_trr:
            lines.append("T1-C (TRR Classification):")
            lines.append(f"  Hour Accuracy: {self.t1_trr.hour_accuracy:.3f}")
            lines.append(f"  Quadrant Accuracy: {self.t1_trr.quadrant_accuracy:.3f}")
            lines.append(f"  Mean Angular Error: {self.t1_trr.mean_angular_error:.1f}°")
            lines.append("")

        if self.t2:
            lines.append("T2 (Constraint Extraction):")
            lines.append(f"  Precision: {self.t2.precision:.3f}")
            lines.append(f"  Recall: {self.t2.recall:.3f}")
            lines.append(f"  F1: {self.t2.f1:.3f}")
            lines.append(f"  Self-Consistency: {self.t2.self_consistency:.3f}")
            lines.append(f"  Adjusted F1: {self.t2.adjusted_f1:.3f}")
            lines.append("")

        if self.t3:
            lines.append("T3 (Ordinal Reconstruction):")
            lines.append(f"  NRMS: {self.t3.nrms:.4f}")
            lines.append(f"  Constraint Satisfaction: {self.t3.constraint_satisfaction_rate:.3f}")
            lines.append("")

        return "\n".join(lines)
