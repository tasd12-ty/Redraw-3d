"""
ORDINAL-SPATIAL 约束提取智能体模块。

本模块提供从图像/场景数据中提取空间约束的智能体实现：
- Task-1: Blender 场景数据约束提取 (Ground Truth)
- Task-2: 多视角约束提取 (VLM)
- Task-3: 单视角约束提取 (VLM)

Constraint Extraction Agents for ORDINAL-SPATIAL.

This module provides agents for extracting spatial constraints:
- Task-1: Blender scene data extraction (Ground Truth)
- Task-2: Multi-view VLM extraction
- Task-3: Single-view VLM extraction
"""

from ordinal_spatial.agents.base import ConstraintAgent, ConstraintSet, ObjectInfo
from ordinal_spatial.agents.vlm_constraint_agent import (
    VLMConstraintAgent,
    VLMAgentConfig,
)
from ordinal_spatial.agents.blender_constraint_agent import (
    BlenderConstraintAgent,
    BlenderAgentConfig,
)

__all__ = [
    "ConstraintAgent",
    "ConstraintSet",
    "ObjectInfo",
    "VLMConstraintAgent",
    "VLMAgentConfig",
    "BlenderConstraintAgent",
    "BlenderAgentConfig",
]
