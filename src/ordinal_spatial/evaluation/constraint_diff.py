"""
Constraint-Diff 评估指标。

Constraint-Diff metrics for evaluating spatial constraint extraction.

This module implements the evaluation metrics defined in the benchmark paper:
- Missing: Constraints in GT but not in prediction (recall issues)
- Spurious: Constraints in prediction but not in GT (precision issues)
- Violated: Constraints with flipped direction (severe errors)

The metrics provide fine-grained analysis beyond simple P/R/F1.

.. deprecated::
    This module is superseded by ``evaluation.comparison`` which provides:
    - All 7 constraint types (adds closer, trr)
    - Canonical key normalization with pair/obj swap handling
    - Wrong-value sub-classification (flip/shift/other)
    - Optional GT annotation for hallucinated constraints
    - Batch comparison and structured ComparisonResult

    Use ``from ordinal_spatial.evaluation.comparison import
    ConstraintComparator`` for new code.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from collections import defaultdict

from ordinal_spatial.dsl.comparators import Comparator, is_flip


@dataclass
class ConstraintDiffMetrics:
    """
    Constraint-Diff 评估指标。

    Constraint-Diff metrics as defined in the benchmark paper (Section 6).

    Compares ground truth constraints R_GT with predicted constraints R_pred:
    - Missing: R_GT \\ R_pred (constraints not extracted)
    - Spurious: R_pred \\ R_GT (hallucinated constraints)
    - Violated: Constraints present in both but with flipped direction

    Attributes:
        n_ground_truth: Total number of GT constraints |R_GT|
        n_predicted: Total number of predicted constraints |R_pred|
        n_correct: Correctly matched constraints |R_GT ∩ R_pred| with same value
        n_missing: Missing constraints |R_GT \\ R_pred|
        n_spurious: Spurious/hallucinated constraints |R_pred \\ R_GT|
        n_violated: Constraints with flipped direction (e.g., < vs >)
        missing_rate: n_missing / n_ground_truth
        spurious_rate: n_spurious / n_predicted
        violated_rate: n_violated / (n_correct + n_violated)
        precision: n_correct / n_predicted
        recall: n_correct / n_ground_truth
        f1: Harmonic mean of precision and recall
    """
    # Absolute counts
    n_ground_truth: int = 0
    n_predicted: int = 0
    n_correct: int = 0
    n_missing: int = 0
    n_spurious: int = 0
    n_violated: int = 0

    # Rates
    missing_rate: float = 0.0
    spurious_rate: float = 0.0
    violated_rate: float = 0.0

    # Standard metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # Detailed breakdown by constraint type
    by_type: Dict[str, 'ConstraintDiffMetrics'] = field(default_factory=dict)

    # Lists of specific constraint keys for debugging
    missing_keys: List[str] = field(default_factory=list)
    spurious_keys: List[str] = field(default_factory=list)
    violated_keys: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "n_ground_truth": self.n_ground_truth,
            "n_predicted": self.n_predicted,
            "n_correct": self.n_correct,
            "n_missing": self.n_missing,
            "n_spurious": self.n_spurious,
            "n_violated": self.n_violated,
            "missing_rate": self.missing_rate,
            "spurious_rate": self.spurious_rate,
            "violated_rate": self.violated_rate,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }

        if self.by_type:
            result["by_type"] = {
                k: v.to_dict() for k, v in self.by_type.items()
            }

        return result

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "Constraint-Diff Metrics:",
            f"  GT: {self.n_ground_truth}, Pred: {self.n_predicted}",
            f"  Correct: {self.n_correct}, Missing: {self.n_missing}, "
            f"Spurious: {self.n_spurious}, Violated: {self.n_violated}",
            f"  Missing Rate: {self.missing_rate:.3f}",
            f"  Spurious Rate: {self.spurious_rate:.3f}",
            f"  Violated Rate: {self.violated_rate:.3f}",
            f"  P/R/F1: {self.precision:.3f}/{self.recall:.3f}/{self.f1:.3f}",
        ]
        return "\n".join(lines)


def _normalize_comparator(comp: str) -> str:
    """Normalize comparator string to canonical form."""
    comp = str(comp).strip().lower()
    if comp in ("<", "lt", "less"):
        return "<"
    elif comp in (">", "gt", "greater"):
        return ">"
    elif comp in ("~=", "eq", "equal", "approximately_equal"):
        return "~="
    return comp


def _qrr_key(constraint: Dict) -> Tuple:
    """
    Create canonical key for QRR constraint.

    Args:
        constraint: QRR constraint dict with pair1, pair2, metric

    Returns:
        Canonical tuple key (sorted pairs, metric)
    """
    pair1 = tuple(sorted(constraint.get("pair1", [])))
    pair2 = tuple(sorted(constraint.get("pair2", [])))
    metric = constraint.get("metric", "dist3D")

    # Canonicalize order: pair1 < pair2
    if pair1 > pair2:
        pair1, pair2 = pair2, pair1

    return (pair1, pair2, metric)


def _axial_key(constraint: Dict) -> Tuple:
    """Create canonical key for axial constraint."""
    obj1 = constraint.get("object1", constraint.get("obj1", ""))
    obj2 = constraint.get("object2", constraint.get("obj2", ""))
    relation = constraint.get("relation", "")
    return (obj1, obj2, relation)


def _size_key(constraint: Dict) -> Tuple:
    """Create canonical key for size constraint."""
    obj1 = constraint.get("object1", constraint.get("obj1", ""))
    obj2 = constraint.get("object2", constraint.get("obj2", ""))
    return (obj1, obj2)


def _topology_key(constraint: Dict) -> Tuple:
    """Create canonical key for topology constraint."""
    obj1 = constraint.get("object1", constraint.get("obj1", ""))
    obj2 = constraint.get("object2", constraint.get("obj2", ""))
    # Canonicalize order
    if obj1 > obj2:
        obj1, obj2 = obj2, obj1
    return (obj1, obj2)


def compute_constraint_diff(
    predicted: Dict[str, Any],
    ground_truth: Dict[str, Any],
    constraint_types: Optional[List[str]] = None
) -> ConstraintDiffMetrics:
    """
    计算 Constraint-Diff 指标。

    Compute Constraint-Diff metrics comparing predicted vs ground truth.

    Args:
        predicted: Predicted constraint set dict with qrr, axial, size, etc.
        ground_truth: Ground truth constraint set dict
        constraint_types: List of constraint types to evaluate
                         (default: ["qrr", "axial", "size", "topology"])

    Returns:
        ConstraintDiffMetrics with detailed breakdown
    """
    if constraint_types is None:
        constraint_types = [
            "qrr", "axial", "size", "topology", "occlusion",
        ]

    # Initialize metrics
    metrics = ConstraintDiffMetrics()
    metrics.by_type = {}

    # Compute metrics for each constraint type
    for ctype in constraint_types:
        if ctype == "qrr":
            type_metrics = _compute_qrr_diff(
                predicted.get("qrr", []),
                ground_truth.get("qrr", [])
            )
        elif ctype == "axial":
            type_metrics = _compute_axial_diff(
                predicted.get("axial", []),
                ground_truth.get("axial", [])
            )
        elif ctype == "size":
            type_metrics = _compute_size_diff(
                predicted.get("size", []),
                ground_truth.get("size", [])
            )
        elif ctype == "topology":
            type_metrics = _compute_topology_diff(
                predicted.get("topology", []),
                ground_truth.get("topology", [])
            )
        elif ctype == "occlusion":
            type_metrics = _compute_occlusion_diff(
                predicted.get("occlusion", []),
                ground_truth.get("occlusion", [])
            )
        else:
            continue

        metrics.by_type[ctype] = type_metrics

        # Aggregate counts
        metrics.n_ground_truth += type_metrics.n_ground_truth
        metrics.n_predicted += type_metrics.n_predicted
        metrics.n_correct += type_metrics.n_correct
        metrics.n_missing += type_metrics.n_missing
        metrics.n_spurious += type_metrics.n_spurious
        metrics.n_violated += type_metrics.n_violated

        # Aggregate detailed keys
        metrics.missing_keys.extend(type_metrics.missing_keys)
        metrics.spurious_keys.extend(type_metrics.spurious_keys)
        metrics.violated_keys.extend(type_metrics.violated_keys)

    # Compute rates
    if metrics.n_ground_truth > 0:
        metrics.missing_rate = metrics.n_missing / metrics.n_ground_truth
        metrics.recall = metrics.n_correct / metrics.n_ground_truth

    if metrics.n_predicted > 0:
        metrics.spurious_rate = metrics.n_spurious / metrics.n_predicted
        metrics.precision = metrics.n_correct / metrics.n_predicted

    matched_total = metrics.n_correct + metrics.n_violated
    if matched_total > 0:
        metrics.violated_rate = metrics.n_violated / matched_total

    if metrics.precision + metrics.recall > 0:
        metrics.f1 = 2 * metrics.precision * metrics.recall / (
            metrics.precision + metrics.recall
        )

    return metrics


def _compute_qrr_diff(
    predicted: List[Dict],
    ground_truth: List[Dict]
) -> ConstraintDiffMetrics:
    """Compute Constraint-Diff for QRR constraints."""
    metrics = ConstraintDiffMetrics()

    # Build lookup tables
    gt_lookup = {}
    for c in ground_truth:
        key = _qrr_key(c)
        value = _normalize_comparator(c.get("comparator", "~="))
        gt_lookup[key] = value

    pred_lookup = {}
    for c in predicted:
        key = _qrr_key(c)
        value = _normalize_comparator(c.get("comparator", "~="))
        pred_lookup[key] = value

    metrics.n_ground_truth = len(gt_lookup)
    metrics.n_predicted = len(pred_lookup)

    # Find matches, missing, spurious, violated
    gt_keys = set(gt_lookup.keys())
    pred_keys = set(pred_lookup.keys())

    # Missing: in GT but not in pred
    missing = gt_keys - pred_keys
    metrics.n_missing = len(missing)
    metrics.missing_keys = [str(k) for k in missing]

    # Spurious: in pred but not in GT
    spurious = pred_keys - gt_keys
    metrics.n_spurious = len(spurious)
    metrics.spurious_keys = [str(k) for k in spurious]

    # Check matched keys for correct vs violated
    matched = gt_keys & pred_keys
    for key in matched:
        gt_val = gt_lookup[key]
        pred_val = pred_lookup[key]

        if pred_val == gt_val:
            metrics.n_correct += 1
        elif _is_comparator_flip(pred_val, gt_val):
            metrics.n_violated += 1
            metrics.violated_keys.append(f"{key}: {pred_val} vs {gt_val}")
        else:
            # Different but not flipped (e.g., < vs ~=)
            # Count as neither correct nor violated (partial match)
            pass

    # Compute rates
    if metrics.n_ground_truth > 0:
        metrics.missing_rate = metrics.n_missing / metrics.n_ground_truth
        metrics.recall = metrics.n_correct / metrics.n_ground_truth

    if metrics.n_predicted > 0:
        metrics.spurious_rate = metrics.n_spurious / metrics.n_predicted
        metrics.precision = metrics.n_correct / metrics.n_predicted

    matched_total = metrics.n_correct + metrics.n_violated
    if matched_total > 0:
        metrics.violated_rate = metrics.n_violated / matched_total

    if metrics.precision + metrics.recall > 0:
        metrics.f1 = 2 * metrics.precision * metrics.recall / (
            metrics.precision + metrics.recall
        )

    return metrics


def _compute_axial_diff(
    predicted: List[Dict],
    ground_truth: List[Dict]
) -> ConstraintDiffMetrics:
    """Compute Constraint-Diff for axial ordering constraints."""
    metrics = ConstraintDiffMetrics()

    # Build lookup tables
    gt_set = set()
    for c in ground_truth:
        key = _axial_key(c)
        gt_set.add(key)

    pred_set = set()
    for c in predicted:
        key = _axial_key(c)
        pred_set.add(key)

    metrics.n_ground_truth = len(gt_set)
    metrics.n_predicted = len(pred_set)

    # For axial, check exact match and opposite relations
    # Opposite pairs: left_of/right_of, above/below, in_front_of/behind
    opposites = {
        "left_of": "right_of",
        "right_of": "left_of",
        "above": "below",
        "below": "above",
        "in_front_of": "behind",
        "behind": "in_front_of",
    }

    # Build lookup by (obj1, obj2) -> relation
    gt_by_pair = {}
    for c in ground_truth:
        obj1 = c.get("object1", c.get("obj1", ""))
        obj2 = c.get("object2", c.get("obj2", ""))
        rel = c.get("relation", "")
        gt_by_pair[(obj1, obj2)] = rel

    pred_by_pair = {}
    for c in predicted:
        obj1 = c.get("object1", c.get("obj1", ""))
        obj2 = c.get("object2", c.get("obj2", ""))
        rel = c.get("relation", "")
        pred_by_pair[(obj1, obj2)] = rel

    # Analyze
    gt_pairs = set(gt_by_pair.keys())
    pred_pairs = set(pred_by_pair.keys())

    missing_pairs = gt_pairs - pred_pairs
    spurious_pairs = pred_pairs - gt_pairs
    matched_pairs = gt_pairs & pred_pairs

    metrics.n_missing = len(missing_pairs)
    metrics.n_spurious = len(spurious_pairs)
    metrics.missing_keys = [str(p) for p in missing_pairs]
    metrics.spurious_keys = [str(p) for p in spurious_pairs]

    for pair in matched_pairs:
        gt_rel = gt_by_pair[pair]
        pred_rel = pred_by_pair[pair]

        if gt_rel == pred_rel:
            metrics.n_correct += 1
        elif opposites.get(gt_rel) == pred_rel:
            metrics.n_violated += 1
            metrics.violated_keys.append(f"{pair}: {pred_rel} vs {gt_rel}")

    # Compute rates
    if metrics.n_ground_truth > 0:
        metrics.missing_rate = metrics.n_missing / metrics.n_ground_truth
        metrics.recall = metrics.n_correct / metrics.n_ground_truth

    if metrics.n_predicted > 0:
        metrics.spurious_rate = metrics.n_spurious / metrics.n_predicted
        metrics.precision = metrics.n_correct / metrics.n_predicted

    matched_total = metrics.n_correct + metrics.n_violated
    if matched_total > 0:
        metrics.violated_rate = metrics.n_violated / matched_total

    if metrics.precision + metrics.recall > 0:
        metrics.f1 = 2 * metrics.precision * metrics.recall / (
            metrics.precision + metrics.recall
        )

    return metrics


def _compute_size_diff(
    predicted: List[Dict],
    ground_truth: List[Dict]
) -> ConstraintDiffMetrics:
    """Compute Constraint-Diff for size comparison constraints."""
    metrics = ConstraintDiffMetrics()

    # Build lookup: (obj1, obj2) -> comparator
    gt_lookup = {}
    for c in ground_truth:
        key = _size_key(c)
        value = _normalize_comparator(c.get("comparator", "~="))
        gt_lookup[key] = value

    pred_lookup = {}
    for c in predicted:
        key = _size_key(c)
        value = _normalize_comparator(c.get("comparator", "~="))
        pred_lookup[key] = value

    metrics.n_ground_truth = len(gt_lookup)
    metrics.n_predicted = len(pred_lookup)

    gt_keys = set(gt_lookup.keys())
    pred_keys = set(pred_lookup.keys())

    missing = gt_keys - pred_keys
    spurious = pred_keys - gt_keys
    matched = gt_keys & pred_keys

    metrics.n_missing = len(missing)
    metrics.n_spurious = len(spurious)
    metrics.missing_keys = [str(k) for k in missing]
    metrics.spurious_keys = [str(k) for k in spurious]

    for key in matched:
        gt_val = gt_lookup[key]
        pred_val = pred_lookup[key]

        if pred_val == gt_val:
            metrics.n_correct += 1
        elif _is_comparator_flip(pred_val, gt_val):
            metrics.n_violated += 1
            metrics.violated_keys.append(f"{key}: {pred_val} vs {gt_val}")

    # Compute rates
    if metrics.n_ground_truth > 0:
        metrics.missing_rate = metrics.n_missing / metrics.n_ground_truth
        metrics.recall = metrics.n_correct / metrics.n_ground_truth

    if metrics.n_predicted > 0:
        metrics.spurious_rate = metrics.n_spurious / metrics.n_predicted
        metrics.precision = metrics.n_correct / metrics.n_predicted

    matched_total = metrics.n_correct + metrics.n_violated
    if matched_total > 0:
        metrics.violated_rate = metrics.n_violated / matched_total

    if metrics.precision + metrics.recall > 0:
        metrics.f1 = 2 * metrics.precision * metrics.recall / (
            metrics.precision + metrics.recall
        )

    return metrics


def _compute_topology_diff(
    predicted: List[Dict],
    ground_truth: List[Dict]
) -> ConstraintDiffMetrics:
    """Compute Constraint-Diff for topology constraints."""
    metrics = ConstraintDiffMetrics()

    # Build lookup: (obj1, obj2) -> relation
    gt_lookup = {}
    for c in ground_truth:
        key = _topology_key(c)
        value = c.get("relation", "DC")
        gt_lookup[key] = value

    pred_lookup = {}
    for c in predicted:
        key = _topology_key(c)
        value = c.get("relation", "DC")
        pred_lookup[key] = value

    metrics.n_ground_truth = len(gt_lookup)
    metrics.n_predicted = len(pred_lookup)

    gt_keys = set(gt_lookup.keys())
    pred_keys = set(pred_lookup.keys())

    missing = gt_keys - pred_keys
    spurious = pred_keys - gt_keys
    matched = gt_keys & pred_keys

    metrics.n_missing = len(missing)
    metrics.n_spurious = len(spurious)
    metrics.missing_keys = [str(k) for k in missing]
    metrics.spurious_keys = [str(k) for k in spurious]

    for key in matched:
        gt_val = gt_lookup[key]
        pred_val = pred_lookup[key]

        if pred_val == gt_val:
            metrics.n_correct += 1
        else:
            # For topology, different relation is not necessarily a "flip"
            # But DC vs EC vs PO are distinct states
            metrics.n_violated += 1
            metrics.violated_keys.append(f"{key}: {pred_val} vs {gt_val}")

    # Compute rates
    if metrics.n_ground_truth > 0:
        metrics.missing_rate = metrics.n_missing / metrics.n_ground_truth
        metrics.recall = metrics.n_correct / metrics.n_ground_truth

    if metrics.n_predicted > 0:
        metrics.spurious_rate = metrics.n_spurious / metrics.n_predicted
        metrics.precision = metrics.n_correct / metrics.n_predicted

    matched_total = metrics.n_correct + metrics.n_violated
    if matched_total > 0:
        metrics.violated_rate = metrics.n_violated / matched_total

    if metrics.precision + metrics.recall > 0:
        metrics.f1 = 2 * metrics.precision * metrics.recall / (
            metrics.precision + metrics.recall
        )

    return metrics


def _occlusion_key(constraint: Dict) -> Tuple:
    """Create canonical key for occlusion constraint."""
    occluder = constraint.get("occluder", "")
    occluded = constraint.get("occluded", "")
    return (occluder, occluded)


def _compute_occlusion_diff(
    predicted: List[Dict],
    ground_truth: List[Dict]
) -> ConstraintDiffMetrics:
    """
    计算遮挡约束的 Constraint-Diff。

    Compute Constraint-Diff for occlusion constraints.
    """
    metrics = ConstraintDiffMetrics()

    # 构建查找表: (occluder, occluded) 有向对
    gt_set = set()
    for c in ground_truth:
        gt_set.add(_occlusion_key(c))

    pred_set = set()
    for c in predicted:
        pred_set.add(_occlusion_key(c))

    metrics.n_ground_truth = len(gt_set)
    metrics.n_predicted = len(pred_set)

    # 缺失: 在 GT 中但不在预测中
    missing = gt_set - pred_set
    metrics.n_missing = len(missing)
    metrics.missing_keys = [str(k) for k in missing]

    # 冗余: 在预测中但不在 GT 中
    spurious = pred_set - gt_set
    metrics.spurious_keys = [str(k) for k in spurious]

    # 检查完全匹配和方向反转
    matched = gt_set & pred_set
    metrics.n_correct = len(matched)

    # 反转: (A,B) 在预测中但 GT 有 (B,A)
    violated_count = 0
    real_spurious = set()
    for key in spurious:
        reversed_key = (key[1], key[0])
        if reversed_key in gt_set:
            violated_count += 1
        else:
            real_spurious.add(key)

    metrics.n_violated = violated_count
    metrics.n_spurious = len(real_spurious)
    metrics.spurious_keys = [str(k) for k in real_spurious]
    metrics.violated_keys = [
        str(k) for k in spurious if (k[1], k[0]) in gt_set
    ]

    # 计算率
    if metrics.n_ground_truth > 0:
        metrics.missing_rate = (
            metrics.n_missing / metrics.n_ground_truth
        )
        metrics.recall = (
            metrics.n_correct / metrics.n_ground_truth
        )

    if metrics.n_predicted > 0:
        metrics.spurious_rate = (
            metrics.n_spurious / metrics.n_predicted
        )
        metrics.precision = (
            metrics.n_correct / metrics.n_predicted
        )

    matched_total = metrics.n_correct + metrics.n_violated
    if matched_total > 0:
        metrics.violated_rate = (
            metrics.n_violated / matched_total
        )

    if metrics.precision + metrics.recall > 0:
        metrics.f1 = (
            2 * metrics.precision * metrics.recall
            / (metrics.precision + metrics.recall)
        )

    return metrics


def compute_grouped_constraint_diff(
    predicted: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, ConstraintDiffMetrics]:
    """
    计算分组 Constraint-Diff (view-invariant vs view-dependent)。

    Compute grouped Constraint-Diff metrics:
    - "view_invariant": qrr, topology, size
    - "view_dependent": axial, occlusion
    - "overall": all combined

    Args:
        predicted: Predicted constraint set dict
        ground_truth: Ground truth constraint set dict

    Returns:
        Dict with "view_invariant", "view_dependent", "overall"
    """
    vi_types = ["qrr", "topology", "size"]
    vd_types = ["axial", "occlusion"]

    vi = compute_constraint_diff(
        predicted, ground_truth, vi_types
    )
    vd = compute_constraint_diff(
        predicted, ground_truth, vd_types
    )
    overall = compute_constraint_diff(
        predicted, ground_truth, vi_types + vd_types
    )

    return {
        "view_invariant": vi,
        "view_dependent": vd,
        "overall": overall,
    }


def _is_comparator_flip(pred: str, gt: str) -> bool:
    """
    Check if two comparators represent a flip (direction reversal).

    A flip is when < becomes > or vice versa.
    ~= vs < or ~= vs > is not a flip, just a different degree of error.
    """
    if pred == "<" and gt == ">":
        return True
    if pred == ">" and gt == "<":
        return True
    return False


def compute_constraint_diff_batch(
    predictions: List[Dict],
    ground_truths: List[Dict],
    constraint_types: Optional[List[str]] = None
) -> ConstraintDiffMetrics:
    """
    计算多个场景的聚合 Constraint-Diff 指标。

    Compute aggregated Constraint-Diff metrics over multiple scenes.

    Args:
        predictions: List of predicted constraint sets
        ground_truths: List of ground truth constraint sets
        constraint_types: Constraint types to evaluate

    Returns:
        Aggregated ConstraintDiffMetrics
    """
    # Aggregate raw counts
    total = ConstraintDiffMetrics()
    type_totals = defaultdict(lambda: ConstraintDiffMetrics())

    for pred, gt in zip(predictions, ground_truths):
        metrics = compute_constraint_diff(pred, gt, constraint_types)

        total.n_ground_truth += metrics.n_ground_truth
        total.n_predicted += metrics.n_predicted
        total.n_correct += metrics.n_correct
        total.n_missing += metrics.n_missing
        total.n_spurious += metrics.n_spurious
        total.n_violated += metrics.n_violated

        for ctype, type_metrics in metrics.by_type.items():
            type_totals[ctype].n_ground_truth += type_metrics.n_ground_truth
            type_totals[ctype].n_predicted += type_metrics.n_predicted
            type_totals[ctype].n_correct += type_metrics.n_correct
            type_totals[ctype].n_missing += type_metrics.n_missing
            type_totals[ctype].n_spurious += type_metrics.n_spurious
            type_totals[ctype].n_violated += type_metrics.n_violated

    # Compute rates for total
    if total.n_ground_truth > 0:
        total.missing_rate = total.n_missing / total.n_ground_truth
        total.recall = total.n_correct / total.n_ground_truth

    if total.n_predicted > 0:
        total.spurious_rate = total.n_spurious / total.n_predicted
        total.precision = total.n_correct / total.n_predicted

    matched_total = total.n_correct + total.n_violated
    if matched_total > 0:
        total.violated_rate = total.n_violated / matched_total

    if total.precision + total.recall > 0:
        total.f1 = 2 * total.precision * total.recall / (
            total.precision + total.recall
        )

    # Compute rates for each type
    for ctype, type_metrics in type_totals.items():
        if type_metrics.n_ground_truth > 0:
            type_metrics.missing_rate = type_metrics.n_missing / type_metrics.n_ground_truth
            type_metrics.recall = type_metrics.n_correct / type_metrics.n_ground_truth

        if type_metrics.n_predicted > 0:
            type_metrics.spurious_rate = type_metrics.n_spurious / type_metrics.n_predicted
            type_metrics.precision = type_metrics.n_correct / type_metrics.n_predicted

        matched = type_metrics.n_correct + type_metrics.n_violated
        if matched > 0:
            type_metrics.violated_rate = type_metrics.n_violated / matched

        if type_metrics.precision + type_metrics.recall > 0:
            type_metrics.f1 = 2 * type_metrics.precision * type_metrics.recall / (
                type_metrics.precision + type_metrics.recall
            )

    total.by_type = dict(type_totals)

    return total
