#!/usr/bin/env python3
"""
Build ORDINAL-SPATIAL benchmark dataset (unified single/multi-GPU).

生成指定数量的场景，物体数量在 min-max 范围内严格均分。
多生成 ~10%，按物体数量分组裁剪到目标数量。
无 train/val/test 划分，生成扁平数据集。

支持 CPU / 单 GPU / 多 GPU 模式:
  --n-gpus 0  → CPU 渲染 (单进程)
  --n-gpus 1  → 单 GPU 渲染
  --n-gpus 4  → 4 GPU 并行渲染

Usage:
    uv run os-benchmark -o ./data -b /path/to/blender -n 1000
    uv run os-benchmark -o ./data -b /path/to/blender -n 1000 \
        --n-gpus 4 --quality normal --use-gpu
"""

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 多生成比例
OVERSHOOT_RATIO = 0.10

# 质量预设
QUALITY_PRESETS = {
  "draft":  {"samples": 64,  "width": 480,  "height": 320},
  "normal": {"samples": 256, "width": 1024, "height": 768},
  "high":   {"samples": 512, "width": 1024, "height": 768},
}


@dataclass
class BenchmarkConfig:
  """数据集生成配置。"""
  output_dir: str
  blender_path: str
  n_scenes: int = 1000
  min_objects: int = 3
  max_objects: int = 10
  tau: float = 0.10
  n_views: int = 4
  image_width: int = 480
  image_height: int = 320
  camera_distance: float = 12.0
  elevation: float = 30.0
  use_gpu: bool = False
  n_gpus: int = 0
  render_samples: int = 256
  random_seed: int = 42


def _get_rendering_dir() -> Path:
  """获取渲染模块目录路径。"""
  return Path(__file__).parent.parent / "rendering"


# =============================================================================
# Module-level worker function (pickle-safe for multiprocessing)
# =============================================================================

def _run_render_worker(task: Dict[str, Any]) -> Dict[str, Any]:
  """
  单个渲染 worker (module-level, 可被 pickle)。

  每个 worker 调用 Blender 渲染一批场景, 然后提取约束。

  Args:
    task: 渲染任务配置

  Returns:
    渲染结果, 含 output_path 和约束
  """
  worker_id = task["worker_id"]
  gpu_id = task["gpu_id"]
  use_gpu = task["use_gpu"]
  prefix = task["prefix"]

  # 设置 CUDA 设备
  if use_gpu and gpu_id >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

  mode = f"GPU {gpu_id}" if use_gpu else "CPU"
  logger.info(
    f"Worker {worker_id} ({mode}): "
    f"Rendering {task['n_scenes']} scenes "
    f"(idx {task['start_idx']}~"
    f"{task['start_idx'] + task['n_scenes'] - 1})"
  )

  # 准备输出目录
  render_output = (
    Path(task["output_dir"]) / f"worker{worker_id}_temp"
  )
  render_output.mkdir(parents=True, exist_ok=True)

  # 构建 Blender 命令
  rendering_dir = _get_rendering_dir()
  render_script = str(
    rendering_dir / "render_multiview.py"
  )
  data_dir = rendering_dir / "data"

  cmd = [
    task["blender_path"],
    "--background",
    "--python", render_script,
    "--",
    "--base_scene_blendfile",
    str(data_dir / "base_scene_v5.blend"),
    "--properties_json",
    str(data_dir / "properties.json"),
    "--shape_dir", str(data_dir / "shapes_v5"),
    "--material_dir", str(data_dir / "materials_v5"),
    "--output_dir", str(render_output),
    "--prefix", prefix,
    "--num_images", str(task["n_scenes"]),
    "--start_idx", str(task["start_idx"]),
    "--min_objects", str(task["min_objects"]),
    "--max_objects", str(task["max_objects"]),
    "--balanced_objects", "1",
    "--n_views", str(task["n_views"]),
    "--camera_distance", str(task["camera_distance"]),
    "--elevation", str(task["elevation"]),
    "--width", str(task["width"]),
    "--height", str(task["height"]),
    "--render_num_samples", str(task["render_samples"]),
    "--use_gpu", "1" if use_gpu else "0",
    "--seed", str(task["seed"]),
  ]

  # 执行渲染
  start_time = time.time()
  try:
    result = subprocess.run(
      cmd, timeout=3600 * 8,
    )
    if result.returncode != 0:
      raise RuntimeError(
        f"Worker {worker_id} render failed"
      )
    elapsed = time.time() - start_time
    logger.info(
      f"Worker {worker_id}: Done ({elapsed:.1f}s)"
    )
  except subprocess.TimeoutExpired:
    logger.error(f"Worker {worker_id}: Timeout")
    raise

  return {
    "worker_id": worker_id,
    "output_path": str(render_output),
    "prefix": prefix,
    "n_scenes": task["n_scenes"],
    "render_time": time.time() - start_time,
  }


# =============================================================================
# Builder
# =============================================================================

class BenchmarkBuilder:
  """统一数据集生成器 (单进程 / 多 GPU 并行)。"""

  def __init__(self, config: BenchmarkConfig):
    self.config = config
    self.output_dir = Path(config.output_dir)
    self.n_workers = max(config.n_gpus, 1)

  def build(self) -> Dict[str, Any]:
    """生成完整数据集。"""
    n_levels = (
      self.config.max_objects - self.config.min_objects + 1
    )
    per_level = self.config.n_scenes // n_levels
    remainder = self.config.n_scenes % n_levels

    logger.info("=" * 60)
    logger.info("ORDINAL-SPATIAL Dataset Builder")
    logger.info(f"  Target scenes: {self.config.n_scenes}")
    logger.info(
      f"  Objects: {self.config.min_objects}"
      f"-{self.config.max_objects} ({n_levels} levels)"
    )
    logger.info(f"  Per level: {per_level}"
                + (f" (+1 for first {remainder})"
                   if remainder else ""))
    logger.info(f"  Tau: {self.config.tau}")
    logger.info(
      f"  Resolution: "
      f"{self.config.image_width}x{self.config.image_height}"
    )
    device = (
      f"{self.config.n_gpus} GPU(s)"
      if self.config.use_gpu else "CPU"
    )
    logger.info(f"  Device: {device}")
    logger.info(
      f"  Workers: {self.n_workers}"
    )
    logger.info("=" * 60)

    # 计算多生成数量
    extra = max(
      n_levels, int(self.config.n_scenes * OVERSHOOT_RATIO)
    )
    render_count = self.config.n_scenes + extra
    logger.info(
      f"Over-generate: {render_count} "
      f"(+{extra} extra for trim)"
    )

    # 创建目录结构
    self._create_directories()

    # 渲染
    start_time = time.time()
    logger.info(
      f"Step 1: Rendering {render_count} scenes "
      f"with {self.n_workers} worker(s)..."
    )
    if self.n_workers == 1:
      worker_results = [
        self._render_single(render_count)
      ]
    else:
      worker_results = self._render_parallel(
        render_count
      )

    render_time = time.time() - start_time
    logger.info(f"  Render completed in {render_time:.0f}s")

    # 收集所有场景
    logger.info("Step 2: Collecting scenes...")
    all_scenes = self._collect_scenes(worker_results)
    logger.info(f"  Collected {len(all_scenes)} scenes")

    if not all_scenes:
      raise RuntimeError("No scenes were rendered")

    # 按物体数量分组裁剪
    logger.info("Step 3: Trimming to balanced counts...")
    selected = self._trim_to_balanced(
      all_scenes, per_level, remainder, n_levels
    )
    logger.info(f"  Selected {len(selected)} scenes")

    # 提取约束
    logger.info(
      "Step 4: Extracting ground truth constraints..."
    )
    constraints_data = self._extract_constraints(selected)

    # 整理数据集
    logger.info("Step 5: Building dataset...")
    dataset = self._build_dataset(
      worker_results, selected, constraints_data
    )

    # 保存数据集索引
    index_file = self.output_dir / "dataset.json"
    with open(index_file, 'w') as f:
      json.dump(dataset, f, indent=2)

    # 保存数据集信息
    self._save_dataset_info(len(dataset))

    # 清理临时文件
    for result in worker_results:
      temp_dir = Path(result["output_path"])
      if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)

    # 打印物体数量分布
    dist = Counter(
      entry["n_objects"] for entry in dataset
    )
    logger.info("\nObject count distribution:")
    for k in sorted(dist):
      logger.info(f"  {k} objects: {dist[k]} scenes")

    total_time = time.time() - start_time
    logger.info("\n" + "=" * 60)
    logger.info("Dataset generation complete!")
    logger.info(f"  Output: {self.output_dir}")
    logger.info(f"  Scenes: {len(dataset)}")
    logger.info(f"  Time: {total_time:.0f}s")
    logger.info("=" * 60)

    return {"n_scenes": len(dataset)}

  # -----------------------------------------------------------
  # Rendering
  # -----------------------------------------------------------

  def _render_single(
    self, render_count: int
  ) -> Dict[str, Any]:
    """单进程渲染 (CPU 或 单 GPU)。"""
    task = self._make_task(
      worker_id=0,
      gpu_id=0 if self.config.use_gpu else -1,
      start_idx=0,
      n_scenes=render_count,
    )
    return _run_render_worker(task)

  def _render_parallel(
    self, render_count: int
  ) -> List[Dict[str, Any]]:
    """多 GPU 并行渲染。"""
    tasks = self._create_tasks(render_count)
    logger.info(f"Launching {len(tasks)} worker(s)...")

    with mp.Pool(processes=len(tasks)) as pool:
      results = pool.map(_run_render_worker, tasks)

    return results

  def _make_task(
    self,
    worker_id: int,
    gpu_id: int,
    start_idx: int,
    n_scenes: int,
    seed: Optional[int] = None,
  ) -> Dict[str, Any]:
    """构建单个渲染任务配置。"""
    return {
      "worker_id": worker_id,
      "gpu_id": gpu_id,
      "use_gpu": self.config.use_gpu,
      "prefix": "scene",
      "start_idx": start_idx,
      "n_scenes": n_scenes,
      "min_objects": self.config.min_objects,
      "max_objects": self.config.max_objects,
      "tau": self.config.tau,
      "blender_path": self.config.blender_path,
      "output_dir": str(self.output_dir),
      "n_views": self.config.n_views,
      "camera_distance": self.config.camera_distance,
      "elevation": self.config.elevation,
      "width": self.config.image_width,
      "height": self.config.image_height,
      "render_samples": self.config.render_samples,
      "seed": seed or self.config.random_seed,
    }

  def _create_tasks(
    self, render_count: int
  ) -> List[Dict[str, Any]]:
    """为多 GPU 模式创建渲染任务。"""
    tasks = []
    effective = min(self.n_workers, render_count)
    per_worker = render_count // effective
    remainder = render_count % effective
    start_idx = 0

    for wid in range(effective):
      n = per_worker + (1 if wid < remainder else 0)
      if n == 0:
        continue

      # 确定性种子
      seed_hash = int.from_bytes(
        hashlib.sha256(
          f"scene_{start_idx}".encode()
        ).digest()[:4],
        'little',
      )
      worker_seed = self.config.random_seed + seed_hash

      tasks.append(self._make_task(
        worker_id=wid,
        gpu_id=wid,
        start_idx=start_idx,
        n_scenes=n,
        seed=worker_seed,
      ))
      start_idx += n

    return tasks

  # -----------------------------------------------------------
  # Scene Collection & Trimming
  # -----------------------------------------------------------

  def _collect_scenes(
    self, worker_results: List[Dict[str, Any]]
  ) -> List[Dict]:
    """从所有 worker 收集渲染场景。"""
    all_scenes = []
    for result in worker_results:
      output = Path(result["output_path"])
      prefix = result.get("prefix", "scene")
      scenes_file = output / f"{prefix}_scenes.json"

      if not scenes_file.exists():
        logger.warning(
          f"Worker {result['worker_id']}: "
          f"Scene file not found: {scenes_file}"
        )
        continue

      with open(scenes_file) as f:
        scenes = json.load(f).get("scenes", [])

      # 标记来源 worker, 用于后续图片复制
      for scene in scenes:
        scene["_source_output"] = str(output)
      all_scenes.extend(scenes)

    return all_scenes

  def _trim_to_balanced(
    self,
    scenes: List[Dict],
    per_level: int,
    remainder: int,
    n_levels: int
  ) -> List[Dict]:
    """按物体数量分组，每组裁剪到目标数量。"""
    by_count: Dict[int, List[Dict]] = defaultdict(list)
    for scene in scenes:
      n = scene.get(
        "n_objects", len(scene.get("objects", []))
      )
      by_count[n].append(scene)

    selected = []
    for k in range(n_levels):
      obj_count = self.config.min_objects + k
      target = per_level + (1 if k < remainder else 0)
      available = by_count.get(obj_count, [])

      if len(available) < target:
        logger.warning(
          f"  {obj_count} objects: only "
          f"{len(available)}/{target} scenes"
        )
        selected.extend(available)
      else:
        selected.extend(available[:target])

    return selected

  # -----------------------------------------------------------
  # Constraint Extraction
  # -----------------------------------------------------------

  def _extract_constraints(
    self, scenes: List[Dict]
  ) -> Dict[str, Dict]:
    """使用 ConstraintExtractor 提取全部 7 类约束。"""
    try:
      from ordinal_spatial.generation.constraint_extractor import (
        ConstraintExtractor,
        ExtractionConfig,
      )
      from ordinal_spatial.dsl.predicates import MetricType
    except ImportError:
      logger.warning(
        "ConstraintExtractor not available, "
        "using scene data directly"
      )
      return {}

    config = ExtractionConfig(
      tau=self.config.tau,
      metrics=[MetricType.DIST_3D],
    )
    extractor = ConstraintExtractor(config)
    constraints_data = {}

    for scene in scenes:
      scene_id = scene.get("scene_id", "")
      try:
        osd = extractor.extract(scene)
        constraints_data[scene_id] = {
          "world": osd.world.model_dump()
            if hasattr(osd.world, "model_dump")
            else osd.world,
          "views": [
            v.model_dump()
            if hasattr(v, "model_dump") else v
            for v in osd.views
          ],
        }
      except Exception as e:
        logger.warning(
          f"Failed to extract constraints "
          f"for {scene_id}: {e}"
        )
        constraints_data[scene_id] = {}

    return constraints_data

  # -----------------------------------------------------------
  # Dataset Assembly
  # -----------------------------------------------------------

  def _build_dataset(
    self,
    worker_results: List[Dict[str, Any]],
    scenes: List[Dict],
    constraints_data: Dict[str, Dict]
  ) -> List[Dict]:
    """整理渲染结果到最终目录。"""
    dataset = []

    for scene in scenes:
      scene_id = scene.get("scene_id", "")
      source = Path(
        scene.get("_source_output", "")
      )

      # 复制多视角图片
      src_mv = source / "multi_view" / scene_id
      dst_mv = (
        self.output_dir / "images"
        / "multi_view" / scene_id
      )
      if src_mv.exists():
        if dst_mv.exists():
          shutil.rmtree(dst_mv)
        shutil.copytree(src_mv, dst_mv)

      # 复制单视角图片
      src_sv = (
        source / "single_view" / f"{scene_id}.png"
      )
      dst_sv = (
        self.output_dir / "images" / "single_view"
        / f"{scene_id}.png"
      )
      if src_sv.exists():
        shutil.copy(src_sv, dst_sv)

      # 保存元数据 (规范格式: constraints.{world, views})
      # 移除内部标记
      scene_clean = {
        k: v for k, v in scene.items()
        if not k.startswith("_")
      }
      metadata = {
        **scene_clean,
        "constraints": constraints_data.get(
          scene_id, {}
        ),
      }
      metadata_file = (
        self.output_dir / "metadata"
        / f"{scene_id}.json"
      )
      with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

      # 数据集条目
      n_objects = scene.get(
        "n_objects", len(scene.get("objects", []))
      )
      multi_view_images = [
        f"images/multi_view/{scene_id}/view_{i}.png"
        for i in range(self.config.n_views)
      ]

      dataset.append({
        "scene_id": scene_id,
        "single_view_image":
          f"images/single_view/{scene_id}.png",
        "multi_view_images": multi_view_images,
        "metadata_path": f"metadata/{scene_id}.json",
        "n_objects": n_objects,
        "tau": self.config.tau,
      })

    return dataset

  def _create_directories(self):
    """创建目录结构。"""
    for d in [
      self.output_dir,
      self.output_dir / "images" / "single_view",
      self.output_dir / "images" / "multi_view",
      self.output_dir / "metadata",
    ]:
      d.mkdir(parents=True, exist_ok=True)

  def _save_dataset_info(self, n_scenes: int):
    """保存数据集信息。"""
    info = {
      "name": "ORDINAL-SPATIAL Dataset",
      "version": "2.0",
      "created": datetime.now().isoformat(),
      "config": {
        "n_scenes": self.config.n_scenes,
        "min_objects": self.config.min_objects,
        "max_objects": self.config.max_objects,
        "tau": self.config.tau,
        "n_views": self.config.n_views,
        "n_gpus": self.config.n_gpus,
        "image_size": [
          self.config.image_width,
          self.config.image_height,
        ],
        "camera_distance": self.config.camera_distance,
        "elevation": self.config.elevation,
        "render_samples": self.config.render_samples,
        "seed": self.config.random_seed,
      },
      "statistics": {
        "total_scenes": n_scenes,
        "total_single_view_images": n_scenes,
        "total_multi_view_images":
          n_scenes * self.config.n_views,
      },
    }

    info_file = self.output_dir / "dataset_info.json"
    with open(info_file, 'w') as f:
      json.dump(info, f, indent=2)
    logger.info(f"Saved dataset info: {info_file}")


def main():
  """Main entry point."""
  parser = argparse.ArgumentParser(
    description="Build ORDINAL-SPATIAL dataset"
  )

  parser.add_argument(
    "--output-dir", "-o", required=True,
    help="Output directory"
  )
  parser.add_argument(
    "--blender-path", "-b", default="blender",
    help="Path to Blender executable"
  )

  # 数据集规模
  parser.add_argument(
    "--n-scenes", "-n", type=int, default=1000,
    help="Total number of scenes"
  )

  # 物体数量范围
  parser.add_argument(
    "--min-objects", type=int, default=3,
    help="Minimum objects per scene (default: 3)"
  )
  parser.add_argument(
    "--max-objects", type=int, default=10,
    help="Maximum objects per scene (default: 10)"
  )

  # 约束参数
  parser.add_argument(
    "--tau", type=float, default=0.10,
    help="Tolerance threshold (default: 0.10)"
  )

  # 渲染设置
  parser.add_argument("--n-views", type=int, default=4)
  parser.add_argument("--width", type=int, default=None,
    help="Image width (overrides quality preset)")
  parser.add_argument("--height", type=int, default=None,
    help="Image height (overrides quality preset)")
  parser.add_argument("--use-gpu", action="store_true")
  parser.add_argument(
    "--render-samples", type=int, default=None,
    help="Render samples (overrides quality preset)")

  # GPU / 并行
  parser.add_argument(
    "--n-gpus", type=int, default=0,
    help=(
      "Number of GPUs for parallel rendering "
      "(0=CPU single process, default: 0)"
    )
  )

  # 质量预设
  parser.add_argument(
    "--quality",
    choices=["draft", "normal", "high"],
    default="draft",
    help="Render quality preset (default: draft)"
  )

  # 相机设置
  parser.add_argument(
    "--camera-distance", type=float, default=12.0
  )
  parser.add_argument(
    "--elevation", type=float, default=30.0
  )

  parser.add_argument("--seed", type=int, default=42)

  args = parser.parse_args()

  # 解析质量预设
  preset = QUALITY_PRESETS[args.quality]
  width = args.width or preset["width"]
  height = args.height or preset["height"]
  samples = args.render_samples or preset["samples"]

  # --n-gpus > 0 隐含 --use-gpu
  use_gpu = args.use_gpu or args.n_gpus > 0

  # 打印配置
  n_levels = args.max_objects - args.min_objects + 1
  per_level = args.n_scenes // n_levels
  remainder = args.n_scenes % n_levels
  device = (
    f"{args.n_gpus} GPU(s)" if use_gpu
    else "CPU"
  )

  print("\n" + "=" * 60)
  print("ORDINAL-SPATIAL Dataset Builder")
  print("=" * 60)
  print(f"Output    : {args.output_dir}")
  print(f"Scenes    : {args.n_scenes}")
  print(
    f"Objects   : {args.min_objects}"
    f"-{args.max_objects} ({n_levels} levels, "
    f"~{per_level}/level)"
  )
  print(f"Quality   : {args.quality}")
  print(f"Resolution: {width}x{height}")
  print(f"Samples   : {samples}")
  print(f"Device    : {device}")
  print(f"Tau       : {args.tau}")
  print("=" * 60 + "\n")

  config = BenchmarkConfig(
    output_dir=args.output_dir,
    blender_path=args.blender_path,
    n_scenes=args.n_scenes,
    min_objects=args.min_objects,
    max_objects=args.max_objects,
    tau=args.tau,
    n_views=args.n_views,
    image_width=width,
    image_height=height,
    camera_distance=args.camera_distance,
    elevation=args.elevation,
    use_gpu=use_gpu,
    n_gpus=args.n_gpus,
    render_samples=samples,
    random_seed=args.seed,
  )

  builder = BenchmarkBuilder(config)
  stats = builder.build()

  print(f"\nDone: {stats['n_scenes']} scenes generated")


if __name__ == "__main__":
  main()
