"""GPU 工作进程 - 单个 GPU 的渲染任务。"""

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def _get_rendering_dir() -> Path:
  """获取渲染模块目录路径。"""
  return Path(__file__).parent.parent.parent / "src" / "ordinal_spatial" / "rendering"


class GPUWorker:
  """GPU 工作进程。"""

  @staticmethod
  def render(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    在单个 GPU 上执行渲染任务。

    Args:
      task: 任务配置字典

    Returns:
      渲染结果统计
    """
    worker_id = task["worker_id"]
    gpu_id = task["gpu_id"]
    use_gpu = task["use_gpu"]
    split_name = task["split_name"]

    # 设置 CUDA 设备（CPU 模式下不设置）
    if use_gpu and gpu_id >= 0:
      os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    mode = f"GPU {gpu_id}" if use_gpu else "CPU"
    logger.info(
      f"Worker {worker_id} ({mode}): Starting render {split_name} "
      f"(idx {task['start_idx']} ~ "
      f"{task['start_idx'] + task['n_scenes'] - 1}, "
      f"seed {task['seed']})"
    )

    # 准备输出目录
    render_output = Path(task["output_dir"]) / f"worker{worker_id}_temp"
    render_output.mkdir(parents=True, exist_ok=True)

    # 构建 Blender 命令
    rendering_dir = _get_rendering_dir()
    render_script = str(rendering_dir / "render_multiview.py")
    data_dir = rendering_dir / "data"

    cmd = [
      task["blender_path"],
      "--background",
      "--python", render_script,
      "--",
      "--base_scene_blendfile", str(data_dir / "base_scene_v5.blend"),
      "--properties_json", str(data_dir / "properties.json"),
      "--shape_dir", str(data_dir / "shapes_v5"),
      "--material_dir", str(data_dir / "materials_v5"),
      "--output_dir", str(render_output),
      "--split", split_name,
      "--num_images", str(task["n_scenes"]),
      "--start_idx", str(task["start_idx"]),
      "--min_objects", str(task["min_objects"]),
      "--max_objects", str(task["max_objects"]),
      "--n_views", "4",
      "--camera_distance", "12.0",
      "--elevation", "30.0",
      "--width", str(task["render_config"]["width"]),
      "--height", str(task["render_config"]["height"]),
      "--render_num_samples", str(task["render_config"]["samples"]),
      "--use_gpu", "1" if use_gpu else "0",
      "--seed", str(task["seed"]),
    ]

    # 执行渲染
    start_time = time.time()
    try:
      result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=3600 * 4,
        cwd=Path(__file__).parent.parent.parent
      )

      if result.returncode != 0:
        logger.error(
          f"Worker {worker_id}: Render failed\n"
          f"{result.stderr[-500:]}"
        )
        raise RuntimeError(f"Worker {worker_id} render failed")

      elapsed = time.time() - start_time
      logger.info(
        f"Worker {worker_id}: Done ({elapsed:.1f}s, "
        f"{elapsed/task['n_scenes']:.1f}s/scene)"
      )

    except subprocess.TimeoutExpired:
      logger.error(f"Worker {worker_id}: Timeout")
      raise

    # 提取约束
    constraints = GPUWorker._extract_constraints(
      render_output, split_name, task["tau"], worker_id
    )

    return {
      "worker_id": worker_id,
      "output_path": str(render_output),
      "n_scenes": task["n_scenes"],
      "render_time": elapsed,
      "constraints": constraints,
    }

  @staticmethod
  def _extract_constraints(
      output_dir: Path,
      split_name: str,
      tau: float,
      worker_id: int
  ) -> Dict[str, Dict]:
    """提取真值约束。"""
    try:
      from ordinal_spatial.agents import BlenderConstraintAgent
    except ImportError:
      logger.warning(
        f"Worker {worker_id}: Cannot import constraint extractor"
      )
      return {}

    scenes_file = output_dir / f"{split_name}_scenes.json"
    if not scenes_file.exists():
      logger.warning(f"Worker {worker_id}: Scene file not found")
      return {}

    with open(scenes_file) as f:
      scenes = json.load(f).get("scenes", [])

    agent = BlenderConstraintAgent()
    constraints = {}

    for i, scene in enumerate(scenes):
      scene_id = scene.get("scene_id", "")
      try:
        cs = agent.extract_from_single_view(scene, tau)
        constraints[scene_id] = cs.to_dict()
      except Exception as e:
        logger.warning(
          f"Worker {worker_id}: Constraint extraction failed "
          f"for {scene_id}: {e}"
        )
        constraints[scene_id] = scene.get(
          "world_constraints", {}
        )

    logger.info(
      f"GPU {gpu_id}: Extracted {len(constraints)} constraints"
    )
    return constraints
