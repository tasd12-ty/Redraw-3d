"""多GPU并行构建器（扁平模式）"""

import hashlib
import json
import logging
import multiprocessing as mp
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from .gpu_worker import GPUWorker
from .merger import ResultMerger

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 多生成比例
OVERSHOOT_RATIO = 0.10


class MultiGPUBuilder:
  """多GPU并行构建器 - 扁平数据集，无分割"""

  # 质量预设
  QUALITY_PRESETS = {
    "draft": {"samples": 64, "width": 480, "height": 320},
    "normal": {
      "samples": 256, "width": 1024, "height": 768,
    },
    "high": {
      "samples": 512, "width": 1024, "height": 768,
    },
  }

  def __init__(
    self,
    output_dir: str,
    blender_path: str,
    n_gpus: int = 1,
    n_scenes: int = 1000,
    min_objects: int = 3,
    max_objects: int = 10,
    tau: float = 0.10,
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

    # 数据集参数
    self.n_scenes = n_scenes
    self.min_objects = min_objects
    self.max_objects = max_objects
    self.tau = tau

    # 获取质量配置
    quality_config = self.QUALITY_PRESETS[quality]
    self.render_samples = quality_config["samples"]
    self.image_width = quality_config["width"]
    self.image_height = quality_config["height"]

    self._create_directories()

  def _create_directories(self):
    """创建目录结构"""
    for d in [
      self.output_dir / "images" / "single_view",
      self.output_dir / "images" / "multi_view",
      self.output_dir / "metadata",
    ]:
      d.mkdir(parents=True, exist_ok=True)

  def build(self):
    """构建数据集"""
    n_levels = self.max_objects - self.min_objects + 1

    logger.info("=" * 60)
    logger.info("ORDINAL-SPATIAL Parallel Dataset Builder")
    logger.info(f"  Target scenes: {self.n_scenes}")
    logger.info(
      f"  Objects: {self.min_objects}"
      f"-{self.max_objects} ({n_levels} levels)"
    )
    logger.info(f"  Tau: {self.tau}")
    logger.info(
      f"  Workers: {self.n_workers} "
      f"({'GPU' if self.use_gpu else 'CPU'})"
    )
    logger.info("=" * 60)

    # 多生成
    extra = max(
      n_levels,
      int(self.n_scenes * OVERSHOOT_RATIO),
    )
    render_count = self.n_scenes + extra
    logger.info(
      f"Over-generate: {render_count} "
      f"(+{extra} extra for trim)"
    )

    start_time = time.time()

    # 分配任务到 worker
    tasks = self._create_tasks(render_count)

    # 并行渲染
    logger.info(f"Launching {len(tasks)} worker(s)...")
    with mp.Pool(processes=len(tasks)) as pool:
      results = pool.map(GPUWorker.render, tasks)

    # 合并结果
    logger.info("Merging results...")
    merger = ResultMerger(self.output_dir)
    all_scenes = merger.merge(
      self.tau, "scene", results
    )

    # 裁剪到均衡数量
    logger.info("Trimming to balanced counts...")
    per_level = self.n_scenes // n_levels
    remainder = self.n_scenes % n_levels
    selected = self._trim_to_balanced(
      all_scenes, per_level, remainder, n_levels
    )

    # 保存 dataset.json
    dataset_file = self.output_dir / "dataset.json"
    with open(dataset_file, 'w') as f:
      json.dump(selected, f, indent=2)

    # 打印分布
    dist = Counter(e["n_objects"] for e in selected)
    logger.info("Object count distribution:")
    for k in sorted(dist):
      logger.info(f"  {k} objects: {dist[k]} scenes")

    # 保存信息
    self._save_info(len(selected))

    elapsed = time.time() - start_time
    logger.info(
      f"\nDone! {len(selected)} scenes, "
      f"{elapsed / 3600:.1f} hours"
    )

  def _trim_to_balanced(
    self,
    entries: List[Dict],
    per_level: int,
    remainder: int,
    n_levels: int,
  ) -> List[Dict]:
    """按物体数量分组裁剪到目标数量"""
    from collections import defaultdict
    by_count = defaultdict(list)
    for entry in entries:
      by_count[entry["n_objects"]].append(entry)

    selected = []
    for k in range(n_levels):
      obj_count = self.min_objects + k
      target = per_level + (1 if k < remainder else 0)
      available = by_count.get(obj_count, [])

      if len(available) < target:
        logger.warning(
          f"  {obj_count} objects: "
          f"only {len(available)}/{target}"
        )
        selected.extend(available)
      else:
        selected.extend(available[:target])

    return selected

  def _create_tasks(
    self, render_count: int
  ) -> List[Dict]:
    """创建渲染任务"""
    tasks = []
    effective_workers = min(
      self.n_workers, render_count
    )
    scenes_per_worker = render_count // effective_workers
    remainder = render_count % effective_workers
    start_idx = 0

    for worker_id in range(effective_workers):
      n_scenes = scenes_per_worker + (
        1 if worker_id < remainder else 0
      )
      if n_scenes == 0:
        continue

      # 确定性种子
      seed_hash = int.from_bytes(
        hashlib.sha256(
          f"scene_{start_idx}".encode()
        ).digest()[:4],
        'little',
      )
      worker_seed = self.seed + seed_hash

      tasks.append({
        "worker_id": worker_id,
        "gpu_id": worker_id if self.use_gpu else -1,
        "use_gpu": self.use_gpu,
        "prefix": "scene",
        "start_idx": start_idx,
        "n_scenes": n_scenes,
        "min_objects": self.min_objects,
        "max_objects": self.max_objects,
        "tau": self.tau,
        "blender_path": self.blender_path,
        "output_dir": str(self.output_dir),
        "seed": worker_seed,
        "render_config": {
          "samples": self.render_samples,
          "width": self.image_width,
          "height": self.image_height,
        },
      })
      start_idx += n_scenes

    return tasks

  def _save_info(self, n_scenes: int):
    """保存数据集信息"""
    info = {
      "name": "ORDINAL-SPATIAL Dataset",
      "version": "2.0",
      "created": datetime.now().isoformat(),
      "n_gpus": self.n_gpus,
      "seed": self.seed,
      "config": {
        "n_scenes": self.n_scenes,
        "min_objects": self.min_objects,
        "max_objects": self.max_objects,
        "tau": self.tau,
      },
      "render_quality": {
        "samples": self.render_samples,
        "resolution": [
          self.image_width, self.image_height,
        ],
      },
      "statistics": {
        "total_scenes": n_scenes,
        "total_single_view_images": n_scenes,
        "total_multi_view_images": n_scenes * 4,
      },
    }

    with open(
      self.output_dir / "dataset_info.json", 'w'
    ) as f:
      json.dump(info, f, indent=2)
