#!/usr/bin/env python3
"""
可视化 ORDINAL-SPATIAL 基准测试结果。

生成诊断图表和分析可视化。

功能：
- 混淆矩阵（分类任务）
- 难度分布图（按难度级别的准确率）
- 一致性分析（饼图）
- 重构误差分布（直方图）
- 指标对比（多基线）
- 文本报告生成

图表类型：
1. 混淆矩阵 - T1 任务的预测 vs 真值
2. 难度分解 - 各难度级别的准确率和样本数
3. 一致性饼图 - T2 任务的一致/不一致比例
4. 重构误差 - T3 任务的 NRMS 分布
5. 指标对比 - 多个基线的性能对比
6. 文本报告 - 结构化的评估摘要

依赖：
- matplotlib: 绘图库
- seaborn: 统计可视化（可选）

使用示例：
    python visualize.py -p preds.json -g gt.json -t t1-q -o ./plots
    python visualize.py --compare result1.json result2.json -o ./comparison

Visualize ORDINAL-SPATIAL benchmark results.

Generates diagnostic plots and analysis visualizations.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json(path: str) -> Any:
    """
    加载 JSON 文件。

    Load JSON file.
    """
    with open(path) as f:
        return json.load(f)


def plot_confusion_matrix(
    predictions: List[Dict],
    ground_truth: List[Dict],
    output_path: str,
    title: str = "Confusion Matrix",
):
    """
    绘制分类任务的混淆矩阵。

    Plot confusion matrix for classification tasks.
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.error("matplotlib and seaborn required for visualization")
        logger.error("Install with: pip install matplotlib seaborn")
        return

    from collections import Counter

    # Build confusion matrix
    labels = ["<", "~=", ">"]
    matrix = np.zeros((3, 3), dtype=int)

    label_to_idx = {l: i for i, l in enumerate(labels)}

    for pred, gt in zip(predictions, ground_truth):
        pred_comp = pred.get("comparator", "~=")
        gt_comp = gt.get("comparator", "~=")

        # Normalize
        if pred_comp == "≈":
            pred_comp = "~="
        if gt_comp == "≈":
            gt_comp = "~="

        if pred_comp in label_to_idx and gt_comp in label_to_idx:
            matrix[label_to_idx[gt_comp], label_to_idx[pred_comp]] += 1

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved confusion matrix: {output_path}")


def plot_difficulty_breakdown(
    predictions: List[Dict],
    ground_truth: List[Dict],
    dataset: List[Dict],
    output_path: str,
):
    """
    按难度级别绘制准确率。

    Plot accuracy by difficulty level.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for visualization")
        return

    from collections import defaultdict

    # Group by difficulty
    difficulty_results = defaultdict(lambda: {"correct": 0, "total": 0})

    for pred, gt, item in zip(predictions, ground_truth, dataset):
        # Get difficulty from item or estimate
        difficulty = item.get("difficulty", 3)

        pred_comp = pred.get("comparator", "~=")
        gt_comp = gt.get("comparator", "~=")

        difficulty_results[difficulty]["total"] += 1
        if pred_comp == gt_comp:
            difficulty_results[difficulty]["correct"] += 1

    # Compute accuracy
    difficulties = sorted(difficulty_results.keys())
    accuracies = []
    counts = []

    for d in difficulties:
        stats = difficulty_results[d]
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
        else:
            acc = 0
        accuracies.append(acc)
        counts.append(stats["total"])

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy by difficulty
    ax1.bar(difficulties, accuracies, color='steelblue')
    ax1.set_xlabel("Difficulty Level")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy by Difficulty")
    ax1.set_ylim(0, 1)

    # Sample count by difficulty
    ax2.bar(difficulties, counts, color='coral')
    ax2.set_xlabel("Difficulty Level")
    ax2.set_ylabel("Sample Count")
    ax2.set_title("Sample Distribution")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved difficulty breakdown: {output_path}")


def plot_consistency_analysis(
    results: Dict,
    output_path: str,
):
    """
    绘制 T2 结果的一致性分析。

    Plot consistency analysis for T2 results.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for visualization")
        return

    consistency_stats = results.get("consistency_stats", {})

    n_consistent = consistency_stats.get("n_consistent", 0)
    n_inconsistent = consistency_stats.get("n_inconsistent", 0)

    if n_consistent + n_inconsistent == 0:
        logger.warning("No consistency data available")
        return

    # Pie chart
    fig, ax = plt.subplots(figsize=(8, 6))

    sizes = [n_consistent, n_inconsistent]
    labels = ["Consistent", "Inconsistent"]
    colors = ["#2ecc71", "#e74c3c"]
    explode = (0.05, 0)

    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        shadow=True,
        startangle=90,
    )
    ax.set_title("Constraint Set Consistency")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved consistency analysis: {output_path}")


def plot_reconstruction_error(
    predictions: List[Dict],
    ground_truth: List[Dict],
    output_path: str,
):
    """
    绘制 T3 的重构误差分布。

    Plot reconstruction error distribution for T3.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for visualization")
        return

    from ordinal_spatial.evaluation.metrics import compute_t3_metrics

    nrms_values = []

    for pred, gt in zip(predictions, ground_truth):
        pred_pos = pred.get("positions", {})
        gt_pos = gt.get("positions", {})

        if not pred_pos or not gt_pos:
            continue

        common = set(pred_pos.keys()) & set(gt_pos.keys())
        if len(common) < 2:
            continue

        sorted_objs = sorted(common)
        pred_arr = np.array([pred_pos[o] for o in sorted_objs])
        gt_arr = np.array([gt_pos[o] for o in sorted_objs])

        if pred_arr.shape[1] != gt_arr.shape[1]:
            min_dim = min(pred_arr.shape[1], gt_arr.shape[1])
            pred_arr = pred_arr[:, :min_dim]
            gt_arr = gt_arr[:, :min_dim]

        try:
            metrics = compute_t3_metrics(pred_arr, gt_arr)
            nrms_values.append(metrics.nrms)
        except Exception:
            pass

    if not nrms_values:
        logger.warning("No valid NRMS values to plot")
        return

    # Histogram
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(nrms_values, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(nrms_values), color='red', linestyle='--', label=f'Mean: {np.mean(nrms_values):.3f}')
    ax.axvline(np.median(nrms_values), color='green', linestyle='--', label=f'Median: {np.median(nrms_values):.3f}')

    ax.set_xlabel("NRMS Error")
    ax.set_ylabel("Count")
    ax.set_title("Reconstruction Error Distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved reconstruction error plot: {output_path}")


def plot_metrics_comparison(
    results_dict: Dict[str, Dict],
    output_path: str,
):
    """
    绘制多次运行的指标对比。

    Plot metrics comparison across multiple runs.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib required for visualization")
        return

    baselines = list(results_dict.keys())
    metrics_data = {}

    # Extract common metrics
    for baseline, results in results_dict.items():
        metrics = results.get("metrics", {})
        for metric_name, value in metrics.items():
            if isinstance(value, (int, float)):
                if metric_name not in metrics_data:
                    metrics_data[metric_name] = {}
                metrics_data[metric_name][baseline] = value

    if not metrics_data:
        logger.warning("No metrics to compare")
        return

    n_metrics = len(metrics_data)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))

    if n_metrics == 1:
        axes = [axes]

    for ax, (metric_name, values) in zip(axes, metrics_data.items()):
        x = range(len(baselines))
        heights = [values.get(b, 0) for b in baselines]

        ax.bar(x, heights, color='steelblue')
        ax.set_xticks(x)
        ax.set_xticklabels(baselines, rotation=45, ha='right')
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved metrics comparison: {output_path}")


def generate_report(
    results: Dict,
    output_path: str,
):
    """
    生成文本报告。

    Generate text report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ORDINAL-SPATIAL EVALUATION REPORT")
    lines.append("=" * 60)
    lines.append("")

    task = results.get("task", "unknown")
    lines.append(f"Task: {task}")
    lines.append(f"Samples: {results.get('n_predictions', 0)}")
    lines.append("")

    lines.append("METRICS")
    lines.append("-" * 40)

    metrics = results.get("metrics", {})
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"  {key}: {value:.4f}")
        else:
            lines.append(f"  {key}: {value}")

    lines.append("")

    if "failures" in results:
        lines.append("FAILURE ANALYSIS")
        lines.append("-" * 40)
        for key, value in results["failures"].items():
            lines.append(f"  {key}: {value}")
        lines.append("")

    lines.append("=" * 60)

    report = "\n".join(lines)

    with open(output_path, "w") as f:
        f.write(report)

    logger.info(f"Saved report: {output_path}")


def main():
    """
    主入口点。

    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description="Visualize ORDINAL-SPATIAL results"
    )
    parser.add_argument(
        "--results", "-r",
        help="Path to results JSON file"
    )
    parser.add_argument(
        "--predictions", "-p",
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--ground-truth", "-g",
        help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--dataset", "-d",
        help="Path to dataset JSON file"
    )
    parser.add_argument(
        "--task", "-t",
        choices=["t1-q", "t1-c", "t2", "t3"],
        help="Task type"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./visualizations",
        help="Output directory for plots"
    )
    parser.add_argument(
        "--plot-type",
        choices=["confusion", "difficulty", "consistency", "reconstruction", "report", "all"],
        default="all",
        help="Type of visualization"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="Paths to multiple result files for comparison"
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Comparison mode
    if args.compare:
        results_dict = {}
        for path in args.compare:
            name = Path(path).stem
            results_dict[name] = load_json(path)

        plot_metrics_comparison(
            results_dict,
            str(output_dir / "comparison.png")
        )
        return

    # Single results mode
    if args.results:
        results = load_json(args.results)
        task = results.get("task", args.task)

        if args.plot_type in ["report", "all"]:
            generate_report(results, str(output_dir / "report.txt"))

        if args.plot_type in ["consistency", "all"] and task == "t2":
            plot_consistency_analysis(results, str(output_dir / "consistency.png"))

    # Predictions + ground truth mode
    if args.predictions and args.ground_truth:
        predictions = load_json(args.predictions)
        ground_truth = load_json(args.ground_truth)

        task = args.task or "t1-q"

        if args.plot_type in ["confusion", "all"] and task in ["t1-q", "t1-c"]:
            plot_confusion_matrix(
                predictions,
                ground_truth,
                str(output_dir / "confusion_matrix.png"),
                title=f"Confusion Matrix - {task.upper()}"
            )

        if args.plot_type in ["difficulty", "all"] and args.dataset:
            dataset = load_json(args.dataset)
            plot_difficulty_breakdown(
                predictions,
                ground_truth,
                dataset,
                str(output_dir / "difficulty_breakdown.png")
            )

        if args.plot_type in ["reconstruction", "all"] and task == "t3":
            plot_reconstruction_error(
                predictions,
                ground_truth,
                str(output_dir / "reconstruction_error.png")
            )

    logger.info(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
