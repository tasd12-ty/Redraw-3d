#!/usr/bin/env python3
"""
在 ORDINAL-SPATIAL 基准测试上运行基线模型。

支持所有基线实现和任务类型。

功能：
- 加载数据集（支持多种划分）
- 初始化基线模型（oracle/vlm_direct/vlm_cot/hybrid/embedding）
- 运行评估（T1-Q/T1-C/T2/T3/all）
- 保存结果和指标
- 生成评估摘要

基线类型：
- oracle: 真值基线（100%准确率）
- vlm_direct: VLM 直接预测
- vlm_cot: VLM + 思维链
- hybrid: 预测-验证-修复循环
- embedding: 序嵌入优化

使用示例：
    python run_baseline.py --baseline oracle --task t1-q --data ./data
    python run_baseline.py --baseline vlm_direct --task t2 --model openai/gpt-4o

Run baseline models on ORDINAL-SPATIAL benchmark.

Supports all baseline implementations and task types.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(data_path: str, split: str = "test_iid") -> List[Dict]:
    """
    加载数据集。

    Load dataset. Supports:
    1. Flat mode: data_path/dataset.json (list of scene entries)
    2. Legacy split mode: data_path/splits/{split}.json
    3. Direct JSON file path
    """
    path = Path(data_path)

    # Flat mode: dataset.json
    flat_file = path / "dataset.json"
    if flat_file.exists():
        with open(flat_file) as f:
            return json.load(f)

    # Legacy split mode
    split_file = path / "splits" / f"{split}.json"
    if split_file.exists():
        with open(split_file) as f:
            return json.load(f)

    # Direct file
    if path.suffix == ".json" and path.exists():
        with open(path) as f:
            return json.load(f)

    raise FileNotFoundError(
        f"Dataset not found: tried {flat_file}, "
        f"{split_file}, {path}"
    )


def preprocess_for_t1_qrr(
    dataset: List[Dict],
    data_path: str,
    max_queries_per_scene: int = 10,
) -> List[Dict]:
    """
    Preprocess benchmark dataset for T1-Q evaluation.

    Loads metadata and converts QRR constraints to query format.
    """
    processed = []
    data_dir = Path(data_path)

    for item in dataset:
        meta_path = data_dir / item.get("metadata_path", "")
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        # Get objects
        objects = meta.get("objects", [])

        # Get QRR constraints from metadata
        # 支持 {world, views} 和旧扁平格式
        cs = meta.get("constraints", {})
        world = cs.get("world", cs)
        qrr_list = world.get("qrr", [])

        # Convert to queries
        queries = []
        for i, qrr in enumerate(qrr_list[:max_queries_per_scene]):
            pair1 = qrr.get("pair1", [])
            pair2 = qrr.get("pair2", [])
            if len(pair1) == 2 and len(pair2) == 2:
                queries.append({
                    "query_id": f"{item['scene_id']}_qrr_{i}",
                    "A": pair1[0],
                    "B": pair1[1],
                    "C": pair2[0],
                    "D": pair2[1],
                    "metric": qrr.get("metric", "dist3D"),
                    "ground_truth": qrr.get("comparator", ""),
                    # Enhanced: include difficulty metadata for detailed analysis
                    "difficulty": qrr.get("difficulty"),
                    "ratio": qrr.get("ratio"),
                    "boundary_flag": qrr.get("boundary_flag"),
                })

        if queries:
            # Extract just the filename from the image path
            image_path = item.get("single_view_image", "")
            if "/" in image_path:
                image_path = image_path.split("/")[-1]
            processed.append({
                "scene_id": item.get("scene_id", ""),
                "image_path": image_path,
                "scene": {"scene_id": item["scene_id"], "objects": objects},
                "qrr_queries": queries,
            })

    return processed


def preprocess_for_t2(
    dataset: List[Dict],
    data_path: str,
) -> List[Dict]:
    """
    Preprocess benchmark dataset for T2 evaluation.

    Loads metadata and formats for constraint extraction evaluation.
    """
    processed = []
    data_dir = Path(data_path)

    for item in dataset:
        meta_path = data_dir / item.get("metadata_path", "")
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        # Get objects
        objects = meta.get("objects", [])

        # Get ground truth constraints
        # 支持 {world, views} 和旧扁平格式
        cs = meta.get("constraints", {})
        constraints = cs.get("world", cs)

        # Extract image filename
        image_path = item.get("single_view_image", "")
        if "/" in image_path:
            image_path = image_path.split("/")[-1]

        processed.append({
            "scene_id": item.get("scene_id", ""),
            "image_path": image_path,
            "scene": {"scene_id": item["scene_id"], "objects": objects},
            "ground_truth": {
                "qrr": constraints.get("qrr", []),
                "trr": constraints.get("trr", []),
            },
        })

    return processed


def get_baseline(baseline_name: str, config: Optional[Dict] = None) -> Any:
    """
    按名称获取基线实例。

    Get baseline instance by name.
    """
    if baseline_name == "oracle":
        from ordinal_spatial.baselines.oracle import OracleBaseline
        return OracleBaseline()

    elif baseline_name == "vlm_direct":
        from ordinal_spatial.baselines.vlm_direct import VLMDirectBaseline, VLMConfig
        vlm_config = VLMConfig(
            model=config.get("model", "openai/gpt-4o") if config else "openai/gpt-4o",
            api_key=config.get("api_key", os.environ.get("OPENROUTER_API_KEY", "")) if config else os.environ.get("OPENROUTER_API_KEY", ""),
        )
        return VLMDirectBaseline(vlm_config)

    elif baseline_name == "vlm_cot":
        from ordinal_spatial.baselines.vlm_direct import VLMDirectBaseline, VLMConfig
        vlm_config = VLMConfig(
            model=config.get("model", "openai/gpt-4o") if config else "openai/gpt-4o",
            api_key=config.get("api_key", os.environ.get("OPENROUTER_API_KEY", "")) if config else os.environ.get("OPENROUTER_API_KEY", ""),
            use_cot=True,
        )
        return VLMDirectBaseline(vlm_config)

    elif baseline_name == "hybrid":
        from ordinal_spatial.baselines.hybrid import HybridBaseline, HybridConfig
        from ordinal_spatial.baselines.vlm_direct import VLMConfig
        vlm_config = VLMConfig(
            model=config.get("model", "openai/gpt-4o") if config else "openai/gpt-4o",
            api_key=config.get("api_key", os.environ.get("OPENROUTER_API_KEY", "")) if config else os.environ.get("OPENROUTER_API_KEY", ""),
        )
        hybrid_config = HybridConfig(vlm_config=vlm_config)
        return HybridBaseline(hybrid_config)

    elif baseline_name == "embedding":
        from ordinal_spatial.baselines.embedding import OrdinalEmbedding, EmbeddingConfig
        embed_config = EmbeddingConfig(
            n_dims=config.get("n_dims", 3) if config else 3,
            max_iterations=config.get("max_iterations", 1000) if config else 1000,
        )
        return OrdinalEmbedding(embed_config)

    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")


def run_t1_qrr(baseline, dataset: List[Dict], config: Dict) -> Dict:
    """
    运行 T1-Q 评估。

    Run T1-Q evaluation.
    """
    from ordinal_spatial.tasks import T1QRRRunner, T1Config

    task_config = T1Config(
        tau=config.get("tau", 0.10),
        save_predictions=True,
        output_dir=config.get("output_dir"),
    )

    runner = T1QRRRunner(baseline, task_config)
    result = runner.run(dataset, config.get("images_dir"))

    return {
        "task": "T1-Q",
        "metrics": result.metrics.to_dict(),
        "n_predictions": len(result.predictions),
        "n_errors": len(result.errors),
    }


def run_t1_trr(baseline, dataset: List[Dict], config: Dict) -> Dict:
    """
    运行 T1-C 评估。

    Run T1-C evaluation.
    """
    from ordinal_spatial.tasks import T1TRRRunner, T1Config

    task_config = T1Config(
        tau=config.get("tau", 0.10),
        save_predictions=True,
        output_dir=config.get("output_dir"),
    )

    runner = T1TRRRunner(baseline, task_config)
    result = runner.run(dataset, config.get("images_dir"))

    return {
        "task": "T1-C",
        "metrics": result.metrics.to_dict(),
        "n_predictions": len(result.predictions),
        "n_errors": len(result.errors),
    }


def run_t2(baseline, dataset: List[Dict], config: Dict) -> Dict:
    """
    运行 T2 评估。

    Run T2 evaluation.
    """
    from ordinal_spatial.tasks import T2Runner
    from ordinal_spatial.tasks.t2_extraction import T2Config

    task_config = T2Config(
        tau=config.get("tau", 0.10),
        check_consistency=True,
        save_predictions=True,
        output_dir=config.get("output_dir"),
    )

    runner = T2Runner(baseline, task_config)
    result = runner.run(dataset, config.get("images_dir"))

    return {
        "task": "T2",
        "metrics": result.metrics.to_dict(),
        "consistency_stats": result.consistency_stats,
        "n_predictions": len(result.predictions),
        "n_errors": len(result.errors),
    }


def run_t3(baseline, dataset: List[Dict], config: Dict) -> Dict:
    """
    运行 T3 评估。

    Run T3 evaluation.
    """
    from ordinal_spatial.tasks import T3Runner
    from ordinal_spatial.tasks.t3_reconstruction import T3Config

    task_config = T3Config(
        n_dims=config.get("n_dims", 3),
        max_iterations=config.get("max_iterations", 1000),
        save_predictions=True,
        output_dir=config.get("output_dir"),
    )

    runner = T3Runner(baseline, task_config)
    result = runner.run(dataset)

    return {
        "task": "T3",
        "metrics": result.metrics.to_dict(),
        "n_predictions": len(result.predictions),
        "n_errors": len(result.errors),
    }


def main():
    """
    主入口点。

    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description="Run baseline on ORDINAL-SPATIAL benchmark"
    )
    parser.add_argument(
        "--baseline", "-b",
        required=True,
        choices=["oracle", "vlm_direct", "vlm_cot", "hybrid", "embedding"],
        help="Baseline to run"
    )
    parser.add_argument(
        "--task", "-t",
        required=True,
        choices=["t1-q", "t1-c", "t2", "t3", "all"],
        help="Task to evaluate"
    )
    parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to dataset directory or file"
    )
    parser.add_argument(
        "--split", "-s",
        default="test_iid",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--output", "-o",
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--images-dir", "-i",
        help="Directory containing scene images"
    )
    parser.add_argument(
        "--tau", type=float, default=0.10,
        help="Tolerance parameter"
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o",
        help="VLM model name (for VLM baselines)"
    )
    parser.add_argument(
        "--max-samples", type=int,
        help="Limit number of samples"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load dataset
    logger.info(f"Loading dataset: {args.data}/{args.split}")
    dataset = load_dataset(args.data, args.split)

    if args.max_samples:
        dataset = dataset[:args.max_samples]
        logger.info(f"Limited to {len(dataset)} samples")

    # Setup config
    config = {
        "tau": args.tau,
        "model": args.model,
        "images_dir": args.images_dir,
        "output_dir": args.output,
    }

    # Get baseline
    logger.info(f"Loading baseline: {args.baseline}")
    baseline = get_baseline(args.baseline, config)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # Run tasks
    if args.task in ["t1-q", "all"]:
        logger.info("Running T1-Q evaluation...")
        config["output_dir"] = str(output_dir / "t1_qrr")
        config["images_dir"] = str(Path(args.data) / "images" / "single_view")
        # Preprocess dataset for T1-Q
        t1q_dataset = preprocess_for_t1_qrr(dataset, args.data)
        logger.info(f"Preprocessed {len(t1q_dataset)} scenes with QRR queries")
        results["t1_qrr"] = run_t1_qrr(baseline, t1q_dataset, config)
        logger.info(f"T1-Q Results: {results['t1_qrr']['metrics']}")

    if args.task in ["t1-c", "all"]:
        logger.info("Running T1-C evaluation...")
        config["output_dir"] = str(output_dir / "t1_trr")
        results["t1_trr"] = run_t1_trr(baseline, dataset, config)
        logger.info(f"T1-C Results: {results['t1_trr']['metrics']}")

    if args.task in ["t2", "all"]:
        logger.info("Running T2 evaluation...")
        config["output_dir"] = str(output_dir / "t2")
        config["images_dir"] = str(Path(args.data) / "images" / "single_view")
        # Preprocess dataset for T2
        t2_dataset = preprocess_for_t2(dataset, args.data)
        logger.info(f"Preprocessed {len(t2_dataset)} scenes for T2")
        results["t2"] = run_t2(baseline, t2_dataset, config)
        logger.info(f"T2 Results: {results['t2']['metrics']}")

    if args.task in ["t3", "all"]:
        logger.info("Running T3 evaluation...")
        config["output_dir"] = str(output_dir / "t3")
        results["t3"] = run_t3(baseline, dataset, config)
        logger.info(f"T3 Results: {results['t3']['metrics']}")

    # Save summary
    summary = {
        "baseline": args.baseline,
        "task": args.task,
        "split": args.split,
        "n_samples": len(dataset),
        "tau": args.tau,
        "results": results,
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
