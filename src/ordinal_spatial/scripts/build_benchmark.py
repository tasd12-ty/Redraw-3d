#!/usr/bin/env python3
"""
Build ORDINAL-SPATIAL benchmark dataset.

生成指定数量的场景，物体数量在 min-max 范围内严格均分。
多生成 ~10%，按物体数量分组裁剪到目标数量。
无 train/val/test 划分，生成扁平数据集。

Usage:
    uv run os-benchmark -o ./data/bench -b /path/to/blender -n 1000
    uv run os-benchmark -o ./data/bench -b /path/to/blender -n 1000 \
        --min-objects 3 --max-objects 10 --use-gpu
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 多生成比例
OVERSHOOT_RATIO = 0.10


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
  render_samples: int = 256
  random_seed: int = 42


class BenchmarkBuilder:
  """数据集生成器（扁平模式，带多生成裁剪）。"""

  def __init__(self, config: BenchmarkConfig):
    self.config = config
    self.output_dir = Path(config.output_dir)
    self.rendering_dir = Path(__file__).parent.parent / "rendering"

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
    logger.info(f"  Per level: {per_level} (+1 for first "
                f"{remainder})" if remainder else "")
    logger.info(f"  Tau: {self.config.tau}")
    logger.info(
      f"  Resolution: "
      f"{self.config.image_width}x{self.config.image_height}"
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
    logger.info(
      f"Step 1: Rendering {render_count} scenes..."
    )
    render_output = self._render(render_count)

    # 加载渲染结果
    scenes_file = render_output / "scene_scenes.json"
    if not scenes_file.exists():
      raise RuntimeError(
        f"Render produced no output: {scenes_file}"
      )

    with open(scenes_file) as f:
      all_rendered = json.load(f).get("scenes", [])

    if not all_rendered:
      raise RuntimeError(
        "Render succeeded but produced 0 scenes"
      )

    logger.info(f"  Rendered {len(all_rendered)} scenes")

    # 按物体数量分组裁剪
    logger.info("Step 2: Trimming to balanced counts...")
    selected = self._trim_to_balanced(
      all_rendered, per_level, remainder, n_levels
    )
    logger.info(f"  Selected {len(selected)} scenes")

    # 提取约束
    logger.info(
      "Step 3: Extracting ground truth constraints..."
    )
    constraints_data = self._extract_constraints(selected)

    # 整理数据集
    logger.info("Step 4: Building dataset...")
    dataset = self._build_dataset(
      render_output, selected, constraints_data
    )

    # 保存数据集索引
    index_file = self.output_dir / "dataset.json"
    with open(index_file, 'w') as f:
      json.dump(dataset, f, indent=2)

    # 保存数据集信息
    self._save_dataset_info(len(dataset))

    # 清理临时文件
    render_temp = self.output_dir / "render_temp"
    if render_temp.exists():
      shutil.rmtree(render_temp)

    # 打印物体数量分布
    dist = Counter(
      entry["n_objects"] for entry in dataset
    )
    logger.info("\nObject count distribution:")
    for k in sorted(dist):
      logger.info(f"  {k} objects: {dist[k]} scenes")

    logger.info("\n" + "=" * 60)
    logger.info("Dataset generation complete!")
    logger.info(f"  Output: {self.output_dir}")
    logger.info(f"  Scenes: {len(dataset)}")
    logger.info("=" * 60)

    return {"n_scenes": len(dataset)}

  def _trim_to_balanced(
    self,
    scenes: List[Dict],
    per_level: int,
    remainder: int,
    n_levels: int
  ) -> List[Dict]:
    """
    按物体数量分组，每组裁剪到目标数量。
    不足的组保留全部，不补齐。
    """
    # 按物体数量分组
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
          f"  {obj_count} objects: only {len(available)}"
          f"/{target} scenes available"
        )
        selected.extend(available)
      else:
        selected.extend(available[:target])

    return selected

  def _create_directories(self):
    """创建目录结构。"""
    for d in [
      self.output_dir,
      self.output_dir / "images" / "single_view",
      self.output_dir / "images" / "multi_view",
      self.output_dir / "metadata",
    ]:
      d.mkdir(parents=True, exist_ok=True)

  def _render(self, render_count: int) -> Path:
    """调用 Blender 渲染场景。"""
    render_output = self.output_dir / "render_temp"
    render_output.mkdir(parents=True, exist_ok=True)

    render_script = str(
      self.rendering_dir / "render_multiview.py"
    )
    data_dir = self.rendering_dir / "data"

    cmd = [
      self.config.blender_path,
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
      "--prefix", "scene",
      "--num_images", str(render_count),
      "--min_objects", str(self.config.min_objects),
      "--max_objects", str(self.config.max_objects),
      "--n_views", str(self.config.n_views),
      "--camera_distance", str(self.config.camera_distance),
      "--elevation", str(self.config.elevation),
      "--width", str(self.config.image_width),
      "--height", str(self.config.image_height),
      "--render_num_samples",
      str(self.config.render_samples),
      "--balanced_objects", "1",
      "--seed", str(self.config.random_seed),
    ]

    if self.config.use_gpu:
      cmd.extend(["--use_gpu", "1"])

    logger.info("Running Blender render...")

    try:
      result = subprocess.run(cmd, timeout=3600 * 8)
      if result.returncode != 0:
        raise RuntimeError("Blender render failed")
      logger.info("Render completed successfully")
    except subprocess.TimeoutExpired:
      logger.error("Render timed out")
      raise

    return render_output

  def _extract_constraints(
    self, scenes: List[Dict]
  ) -> Dict[str, Dict]:
    """从渲染结果中提取约束。"""
    try:
      from ordinal_spatial.agents import (
        BlenderConstraintAgent,
      )
    except ImportError:
      logger.warning(
        "BlenderConstraintAgent not available, "
        "using scene data directly"
      )
      return {}

    agent = BlenderConstraintAgent()
    constraints_data = {}

    for scene in scenes:
      scene_id = scene.get("scene_id", "")
      try:
        constraint_set = agent.extract_from_single_view(
          image=scene, tau=self.config.tau
        )
        constraints_data[scene_id] = constraint_set.to_dict()
      except Exception as e:
        logger.warning(
          f"Failed to extract constraints "
          f"for {scene_id}: {e}"
        )
        constraints_data[scene_id] = scene.get(
          "world_constraints", {}
        )

    return constraints_data

  def _build_dataset(
    self,
    render_output: Path,
    scenes: List[Dict],
    constraints_data: Dict[str, Dict]
  ) -> List[Dict]:
    """整理渲染结果到最终目录。"""
    dataset = []

    for scene in scenes:
      scene_id = scene.get("scene_id", "")

      # 复制多视角图片
      src_mv = render_output / "multi_view" / scene_id
      dst_mv = (
        self.output_dir / "images" / "multi_view" / scene_id
      )
      if src_mv.exists():
        if dst_mv.exists():
          shutil.rmtree(dst_mv)
        shutil.copytree(src_mv, dst_mv)

      # 复制单视角图片
      src_sv = (
        render_output / "single_view" / f"{scene_id}.png"
      )
      dst_sv = (
        self.output_dir / "images" / "single_view"
        / f"{scene_id}.png"
      )
      if src_sv.exists():
        shutil.copy(src_sv, dst_sv)

      # 保存元数据
      metadata_file = (
        self.output_dir / "metadata" / f"{scene_id}.json"
      )
      scene_metadata = {
        **scene,
        "constraints": constraints_data.get(scene_id, {})
      }
      with open(metadata_file, 'w') as f:
        json.dump(scene_metadata, f, indent=2)

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
    help="Total number of scenes to generate"
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
  parser.add_argument("--width", type=int, default=480)
  parser.add_argument("--height", type=int, default=320)
  parser.add_argument("--use-gpu", action="store_true")
  parser.add_argument(
    "--render-samples", type=int, default=256
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

  # 打印配置
  n_levels = args.max_objects - args.min_objects + 1
  per_level = args.n_scenes // n_levels
  remainder = args.n_scenes % n_levels
  print("Dataset configuration:")
  print(f"  Total scenes: {args.n_scenes}")
  print(
    f"  Objects range: {args.min_objects}"
    f"-{args.max_objects} ({n_levels} levels)"
  )
  if remainder:
    print(
      f"  Per level: {per_level} "
      f"(+1 for first {remainder})"
    )
  else:
    print(f"  Per level: {per_level}")
  print(f"  Resolution: {args.width}x{args.height}")
  print(f"  Render samples: {args.render_samples}")
  print(f"  Tau: {args.tau}")

  config = BenchmarkConfig(
    output_dir=args.output_dir,
    blender_path=args.blender_path,
    n_scenes=args.n_scenes,
    min_objects=args.min_objects,
    max_objects=args.max_objects,
    tau=args.tau,
    n_views=args.n_views,
    image_width=args.width,
    image_height=args.height,
    camera_distance=args.camera_distance,
    elevation=args.elevation,
    use_gpu=args.use_gpu,
    render_samples=args.render_samples,
    random_seed=args.seed,
  )

  builder = BenchmarkBuilder(config)
  stats = builder.build()

  print(f"\nDone: {stats['n_scenes']} scenes generated")


if __name__ == "__main__":
  main()
