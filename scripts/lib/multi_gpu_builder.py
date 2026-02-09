"""多GPU并行构建器核心实现"""

import hashlib
import json
import logging
import multiprocessing as mp
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any

from .gpu_worker import GPUWorker
from .merger import ResultMerger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetSize(Enum):
    """数据集规模预设"""
    TINY = {
        "train": 8, "val": 8, "test_iid": 8,
        "test_comp": 8, "test_hard": 8
    }
    SMALL = {
        "train": 800, "val": 160, "test_iid": 400,
        "test_comp": 80, "test_hard": 80
    }
    MEDIUM = {
        "train": 8000, "val": 1600, "test_iid": 4000,
        "test_comp": 800, "test_hard": 800
    }
    LARGE = {
        "train": 80000, "val": 16000, "test_iid": 40000,
        "test_comp": 8000, "test_hard": 8000
    }


@dataclass
class SplitConfig:
    """分割配置"""
    name: str
    n_scenes: int
    min_objects: int
    max_objects: int
    tau: float


class MultiGPUBuilder:
    """多GPU并行构建器 - 支持任意数量 GPU"""

    # 质量预设
    QUALITY_PRESETS = {
        "draft": {"samples": 64, "width": 480, "height": 320},
        "normal": {"samples": 256, "width": 1024, "height": 768},
        "high": {"samples": 512, "width": 1024, "height": 768},
    }

    def __init__(
        self,
        output_dir: str,
        blender_path: str,
        n_gpus: int = 1,
        dataset_size: DatasetSize = DatasetSize.SMALL,
        quality: str = "normal",
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.blender_path = blender_path
        # n_gpus=0 表示 CPU 渲染（单进程）
        self.n_gpus = n_gpus
        self.use_gpu = n_gpus > 0
        self.n_workers = max(n_gpus, 1)
        self.seed = seed

        # 获取质量配置
        quality_config = self.QUALITY_PRESETS[quality]
        self.render_samples = quality_config["samples"]
        self.image_width = quality_config["width"]
        self.image_height = quality_config["height"]

        # 构建分割配置
        sizes = dataset_size.value
        self.splits = [
            SplitConfig("train", sizes["train"], 3, 10, 0.10),
            SplitConfig("val", sizes["val"], 3, 10, 0.10),
            SplitConfig("test_iid", sizes["test_iid"], 3, 10, 0.10),
            SplitConfig("test_comp", sizes["test_comp"], 10, 15, 0.10),
            SplitConfig("test_hard", sizes["test_hard"], 3, 10, 0.05),
        ]

        self._create_directories()

    def _create_directories(self):
        """创建目录结构"""
        dirs = [
            self.output_dir / "images" / "single_view",
            self.output_dir / "images" / "multi_view",
            self.output_dir / "splits",
            self.output_dir / "metadata",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    def build(self):
        """构建数据集"""
        logger.info("Starting dataset build...")
        start_time = time.time()
        all_stats = {}

        for split_config in self.splits:
            logger.info(f"\n{'='*50}")
            logger.info(f"Building: {split_config.name} ({split_config.n_scenes} scenes)")
            logger.info(f"{'='*50}")

            stats = self._build_split(split_config)
            all_stats[split_config.name] = stats

        self._save_info(all_stats)

        elapsed = time.time() - start_time
        logger.info(f"\nDone! Total time: {elapsed/3600:.1f} hours")

    def _build_split(self, split_config: SplitConfig) -> Dict:
        """构建单个分割"""
        # 分配任务到GPU
        tasks = self._create_gpu_tasks(split_config)

        # 并行渲染
        logger.info(f"Launching {len(tasks)} worker(s)...")
        with mp.Pool(processes=len(tasks)) as pool:
            results = pool.map(GPUWorker.render, tasks)

        # 合并结果
        logger.info("Merging results...")
        merger = ResultMerger(self.output_dir)
        stats = merger.merge(split_config, results)

        return stats

    def _create_gpu_tasks(self, split_config: SplitConfig) -> List[Dict]:
        """创建渲染任务，每个 worker 分配不重叠的索引区间和确定性种子"""
        tasks = []
        # worker 数量不超过场景数
        effective_workers = min(self.n_workers, split_config.n_scenes)
        scenes_per_worker = split_config.n_scenes // effective_workers
        remainder = split_config.n_scenes % effective_workers
        start_idx = 0

        for worker_id in range(effective_workers):
            n_scenes = scenes_per_worker + (1 if worker_id < remainder else 0)
            if n_scenes == 0:
                continue

            # 每个 worker 的种子由全局种子 + split 名 + start_idx 确定
            # 用 hashlib 替代 hash() 保证跨进程/跨运行确定性
            split_hash = int.from_bytes(
                hashlib.sha256(split_config.name.encode()).digest()[:4],
                'little'
            )
            worker_seed = self.seed + split_hash + start_idx

            tasks.append({
                "worker_id": worker_id,
                "gpu_id": worker_id if self.use_gpu else -1,
                "use_gpu": self.use_gpu,
                "split_name": split_config.name,
                "start_idx": start_idx,
                "n_scenes": n_scenes,
                "min_objects": split_config.min_objects,
                "max_objects": split_config.max_objects,
                "tau": split_config.tau,
                "blender_path": self.blender_path,
                "output_dir": str(self.output_dir),
                "seed": worker_seed,
                "render_config": {
                    "samples": self.render_samples,
                    "width": self.image_width,
                    "height": self.image_height,
                }
            })
            start_idx += n_scenes

        return tasks

    def _save_info(self, stats: Dict):
        """保存数据集信息"""
        info = {
            "name": "ORDINAL-SPATIAL Benchmark",
            "created": datetime.now().isoformat(),
            "n_gpus": self.n_gpus,
            "seed": self.seed,
            "render_quality": {
                "samples": self.render_samples,
                "resolution": [self.image_width, self.image_height]
            },
            "splits": {s.name: s.n_scenes for s in self.splits},
            "statistics": stats,
        }

        with open(self.output_dir / "dataset_info.json", 'w') as f:
            json.dump(info, f, indent=2)
