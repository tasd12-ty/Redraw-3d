#!/usr/bin/env python3
"""
在 ORDINAL-SPATIAL 基准测试上评估预测结果。

从预测文件和真值文件计算评估指标。

功能：
- 加载预测和真值 JSON 文件
- 计算任务专用指标（T1-Q/T1-C/T2/T3）
- 执行故障分析（可选）
- 生成评估报告
- 保存结果到文件

评估指标：
- T1-Q: 准确率、宏F1、序距离误差、翻转率
- T1-C: 小时准确率、象限准确率、角度误差
- T2: 精确率、召回率、F1、一致性
- T3: NRMS、约束满足率

故障分析：
- 比较符翻转（< ↔ >）
- 约等号误用
- 边界情况错误
- 物体混淆

使用示例：
    python evaluate.py -p predictions.json -g ground_truth.json -t t1-q
    python evaluate.py -p preds.json -g gt.json -t t2 --analyze-failures

Evaluate predictions on ORDINAL-SPATIAL benchmark.

Computes metrics from prediction and ground truth files.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json(path: str) -> Any:
    """
    加载 JSON 文件。

    Load JSON file.
    """
    with open(path) as f:
        return json.load(f)


def evaluate_t1_qrr(
    predictions: List[Dict],
    ground_truth: List[Dict],
) -> Dict:
    """
    评估 T1-Q（QRR 分类）预测结果。

    Evaluate T1-Q predictions.
    """
    from ordinal_spatial.evaluation.metrics import compute_t1_qrr_metrics

    metrics = compute_t1_qrr_metrics(predictions, ground_truth)
    return metrics.to_dict()


def evaluate_t1_trr(
    predictions: List[Dict],
    ground_truth: List[Dict],
) -> Dict:
    """
    评估 T1-C（TRR 时钟分类）预测结果。

    Evaluate T1-C predictions.
    """
    from ordinal_spatial.evaluation.metrics import compute_t1_trr_metrics

    metrics = compute_t1_trr_metrics(predictions, ground_truth)
    return metrics.to_dict()


def evaluate_t2(
    predictions: List[Dict],
    ground_truth: List[Dict],
) -> Dict:
    """
    评估 T2（约束提取）预测结果。

    Evaluate T2 predictions.
    """
    from ordinal_spatial.evaluation.metrics import compute_t2_metrics_batch

    metrics = compute_t2_metrics_batch(predictions, ground_truth)
    return metrics.to_dict()


def evaluate_t3(
    predictions: List[Dict],
    ground_truth: List[Dict],
) -> Dict:
    """
    评估 T3（序重构）预测结果。

    Evaluate T3 predictions.
    """
    import numpy as np
    from ordinal_spatial.evaluation.metrics import compute_t3_metrics

    nrms_values = []
    satisfaction_values = []

    for pred, gt in zip(predictions, ground_truth):
        pred_pos = pred.get("positions", {})
        gt_pos = gt.get("positions", {})

        if not pred_pos or not gt_pos:
            continue

        # Align objects
        common = set(pred_pos.keys()) & set(gt_pos.keys())
        if len(common) < 2:
            continue

        sorted_objs = sorted(common)
        pred_arr = np.array([pred_pos[o] for o in sorted_objs])
        gt_arr = np.array([gt_pos[o] for o in sorted_objs])

        # Handle dimension mismatch
        if pred_arr.shape[1] != gt_arr.shape[1]:
            min_dim = min(pred_arr.shape[1], gt_arr.shape[1])
            pred_arr = pred_arr[:, :min_dim]
            gt_arr = gt_arr[:, :min_dim]

        try:
            metrics = compute_t3_metrics(pred_arr, gt_arr)
            nrms_values.append(metrics.nrms)
            satisfaction_values.append(metrics.constraint_satisfaction)
        except Exception as e:
            logger.warning(f"Error computing T3 metrics: {e}")

    if nrms_values:
        return {
            "nrms": float(np.mean(nrms_values)),
            "nrms_std": float(np.std(nrms_values)),
            "constraint_satisfaction": float(np.mean(satisfaction_values)),
            "n_evaluated": len(nrms_values),
        }
    else:
        return {
            "nrms": float('inf'),
            "constraint_satisfaction": 0.0,
            "n_evaluated": 0,
        }


def analyze_failures(
    predictions: List[Dict],
    ground_truth: List[Dict],
    task: str,
) -> Dict:
    """
    分析故障模式。

    Analyze failure modes.
    """
    failures = {
        "total_errors": 0,
        "comparator_flips": 0,
        "approx_errors": 0,
        "boundary_errors": 0,
    }

    for pred, gt in zip(predictions, ground_truth):
        if pred.get("error"):
            failures["total_errors"] += 1
            continue

        if task == "t1-q":
            pred_comp = pred.get("comparator", "~=")
            gt_comp = gt.get("comparator", "~=")

            if pred_comp != gt_comp:
                # Check flip type
                if (pred_comp == "<" and gt_comp == ">") or \
                   (pred_comp == ">" and gt_comp == "<"):
                    failures["comparator_flips"] += 1
                elif pred_comp == "~=" or gt_comp == "~=":
                    failures["approx_errors"] += 1

    return failures


def main():
    """
    主入口点。

    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate ORDINAL-SPATIAL predictions"
    )
    parser.add_argument(
        "--predictions", "-p",
        required=True,
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--ground-truth", "-g",
        required=True,
        help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--task", "-t",
        required=True,
        choices=["t1-q", "t1-c", "t2", "t3"],
        help="Task type"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file for results (optional)"
    )
    parser.add_argument(
        "--analyze-failures", "-a",
        action="store_true",
        help="Include failure analysis"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load data
    logger.info(f"Loading predictions: {args.predictions}")
    predictions = load_json(args.predictions)

    logger.info(f"Loading ground truth: {args.ground_truth}")
    ground_truth = load_json(args.ground_truth)

    # Validate counts
    if len(predictions) != len(ground_truth):
        logger.warning(
            f"Count mismatch: {len(predictions)} predictions vs "
            f"{len(ground_truth)} ground truth"
        )

    # Evaluate
    logger.info(f"Evaluating task: {args.task}")

    if args.task == "t1-q":
        metrics = evaluate_t1_qrr(predictions, ground_truth)
    elif args.task == "t1-c":
        metrics = evaluate_t1_trr(predictions, ground_truth)
    elif args.task == "t2":
        metrics = evaluate_t2(predictions, ground_truth)
    elif args.task == "t3":
        metrics = evaluate_t3(predictions, ground_truth)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    results = {
        "task": args.task,
        "n_predictions": len(predictions),
        "n_ground_truth": len(ground_truth),
        "metrics": metrics,
    }

    # Failure analysis
    if args.analyze_failures:
        logger.info("Analyzing failures...")
        failures = analyze_failures(predictions, ground_truth, args.task)
        results["failures"] = failures

    # Output
    print("\n" + "=" * 50)
    print(f"EVALUATION RESULTS - {args.task.upper()}")
    print("=" * 50)
    print(f"Samples: {len(predictions)}")
    print("\nMetrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    if args.analyze_failures and "failures" in results:
        print("\nFailure Analysis:")
        for key, value in results["failures"].items():
            print(f"  {key}: {value}")

    print("=" * 50 + "\n")

    # Save to file
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
