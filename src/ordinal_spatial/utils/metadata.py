"""
元数据规范化工具。

统一处理新旧两种约束格式:
- 新格式: constraints = {world: {...}, views: [...]}
- 旧格式: constraints = {qrr: [...], trr: [...], ...}  (扁平)

Metadata normalization utilities.
Supports both the canonical {world, views} format and legacy flat format.
"""

from typing import Dict, Any, Optional, List
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# 视角不变约束类型 (属于 world)
VIEW_INVARIANT_TYPES = [
  "qrr", "topology", "size", "closer",
]

# 视角相关约束类型 (属于 views[k])
VIEW_DEPENDENT_TYPES = [
  "axial_2d", "occlusion", "trr", "qrr_2d", "size_apparent",
]


def get_world_constraints(metadata: Dict) -> Dict:
  """
  提取 world 约束, 支持新旧格式。

  新格式: metadata["constraints"]["world"]
  旧格式: metadata["constraints"] (整个 dict 就是 world)

  Args:
      metadata: 场景元数据 dict

  Returns:
      World 约束 dict (含 qrr, topology, size, closer, axial 等)
  """
  cs = metadata.get("constraints", {})
  return cs.get("world", cs)


def get_view_constraints(
  metadata: Dict, view_idx: int
) -> Dict:
  """
  提取指定视角的约束。

  Args:
      metadata: 场景元数据 dict
      view_idx: 视角索引

  Returns:
      视角约束 dict; 不存在时返回空 dict
  """
  cs = metadata.get("constraints", {})
  views = cs.get("views", [])
  if view_idx < len(views):
    return views[view_idx]
  return {}


def get_merged_gt(
  metadata: Dict, view_idx: Optional[int] = None
) -> Dict:
  """
  构建评估用 GT。

  视角不变约束 (qrr, topology, size, closer) 取自 world;
  视角相关约束 (axial, occlusion, trr) 取自 views[view_idx]。

  当 view_idx=None 时, 只返回 world 约束 (含 world.axial)。

  Args:
      metadata: 场景元数据 dict
      view_idx: 视角索引; None 表示仅用世界坐标

  Returns:
      合并后的约束 dict, 可直接用于比较
  """
  world = get_world_constraints(metadata)

  # 视角不变部分
  gt = {}
  for k in VIEW_INVARIANT_TYPES:
    gt[k] = world.get(k, [])

  if view_idx is None:
    # 仅世界坐标: axial 用 world.axial
    gt["axial"] = world.get("axial", [])
    gt["occlusion"] = world.get("occlusion", [])
    gt["trr"] = world.get("trr", [])
  else:
    # 视角相关部分
    view = get_view_constraints(metadata, view_idx)
    if view:
      # axial_2d → axial (评估时统一键名)
      gt["axial"] = view.get(
        "axial_2d", view.get("axial", [])
      )
      gt["occlusion"] = view.get("occlusion", [])
      gt["trr"] = view.get("trr", [])
    else:
      # Fallback: 无视角数据, 用 world
      gt["axial"] = world.get("axial", [])
      gt["occlusion"] = world.get("occlusion", [])
      gt["trr"] = world.get("trr", [])

  return gt


def load_scene_metadata(path: str) -> Dict:
  """
  加载场景元数据文件。

  Args:
      path: metadata JSON 文件路径

  Returns:
      场景元数据 dict
  """
  with open(path, "r", encoding="utf-8") as f:
    return json.load(f)
