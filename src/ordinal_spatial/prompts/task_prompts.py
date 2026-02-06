"""
ORDINAL-SPATIAL 基准测试的任务专用提示词构建器。

本模块提供函数来为 VLM 查询构建完整提示词，
将系统提示与任务专用模板和场景数据结合。

功能：
- 构建 T1-Q（QRR 分类）提示
- 构建 T1-C（TRR 分类）提示
- 构建 T2（约束提取）提示
- 构建修复提示（用于混合基线）
- 格式化物体描述
- 加载提示模板文件

提示词结构：
1. 系统提示（任务规则和格式）
2. 场景信息（物体列表、位置等）
3. 查询说明（具体要求）
4. 输出格式规范（JSON schema）

Task-specific prompt builders for ORDINAL-SPATIAL benchmark.

This module provides functions to construct complete prompts for VLM queries,
combining system prompts with task-specific templates and scene data.
"""

import os
from typing import Dict, List, Optional, Any
from pathlib import Path

from ordinal_spatial.prompts.system_prompts import (
    get_system_prompt,
    format_repair_prompt,
)


# Template directory
TEMPLATE_DIR = Path(__file__).parent / "templates"


def load_template(name: str) -> str:
    """
    从文件加载提示模板。

    Load a prompt template from file.
    """
    path = TEMPLATE_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")
    return path.read_text()


def get_position_description(obj: Dict) -> str:
    """
    根据3D坐标生成位置描述。

    Generate position description from 3D coordinates.

    使用坐标系：
    - X轴：正方向为右 (right)，负方向为左 (left)
    - Y轴：正方向为后/远 (back/far)，负方向为前/近 (front/near)

    Returns:
        位置描述字符串，如 "front-left", "back-right", "center"
    """
    coords = obj.get("3d_coords", obj.get("position_3d", None))
    if coords is None:
        return ""

    x, y = coords[0], coords[1]

    # X轴位置判定（阈值1.0）
    if x < -1.0:
        x_pos = "left"
    elif x > 1.0:
        x_pos = "right"
    else:
        x_pos = "center"

    # Y轴位置判定（阈值0）
    # Y负值 = 靠近相机 = front
    # Y正值 = 远离相机 = back
    if y < -1.0:
        y_pos = "front"
    elif y > 1.0:
        y_pos = "back"
    else:
        y_pos = "middle"

    # 组合位置描述
    if x_pos == "center" and y_pos == "middle":
        return "center"
    elif x_pos == "center":
        return y_pos
    elif y_pos == "middle":
        return x_pos
    else:
        return f"{y_pos}-{x_pos}"


def format_object_description(obj: Dict, include_position: bool = True) -> str:
    """
    将单个物体格式化为丰富的描述字符串。

    Format a single object as a rich description string.

    格式：[位置] [颜色] [大小] [材质] [形状]
    例如："front-left brown small rubber cylinder"

    Args:
        obj: 物体数据字典
        include_position: 是否包含位置描述（默认True）

    Returns:
        描述字符串
    """
    parts = []

    # 位置描述（可选）
    if include_position:
        pos = get_position_description(obj)
        if pos:
            parts.append(pos)

    # 颜色
    if "color" in obj:
        parts.append(obj["color"])

    # 大小
    if "size" in obj:
        parts.append(obj["size"])

    # 材质
    if "material" in obj:
        parts.append(obj["material"])

    # 形状
    if "shape" in obj:
        parts.append(obj["shape"])

    return " ".join(parts) if parts else obj.get("id", "object")


def format_object_list(objects: List[Dict]) -> str:
    """
    格式化物体列表用于显示。

    Format a list of objects for display.
    """
    lines = []
    for obj in objects:
        obj_id = obj.get("id", "unknown")
        desc = format_object_description(obj)
        lines.append(f"- {obj_id}: {desc}")
    return "\n".join(lines)


# =============================================================================
# T1-Q Prompt Builder
# =============================================================================

def build_t1_qrr_prompt(
    objects: Dict[str, Dict],
    query: Dict,
    tau: float = 0.10,
    with_cot: bool = False
) -> Dict[str, str]:
    """
    Build prompt for T1-Q (QRR classification) task.

    Args:
        objects: Dictionary of object_id -> object data
        query: Query with "A", "B", "C", "D" object IDs
        tau: Tolerance parameter
        with_cot: Include chain-of-thought guidance

    Returns:
        Dict with "system" and "user" prompt strings
    """
    # Get object references
    obj_a = query.get("A", query.get("objects", {}).get("A", ""))
    obj_b = query.get("B", query.get("objects", {}).get("B", ""))
    obj_c = query.get("C", query.get("objects", {}).get("C", ""))
    obj_d = query.get("D", query.get("objects", {}).get("D", ""))

    # Get object descriptions
    obj_a_data = objects.get(obj_a, {})
    obj_b_data = objects.get(obj_b, {})
    obj_c_data = objects.get(obj_c, {})
    obj_d_data = objects.get(obj_d, {})

    # Format all objects
    all_objects = list(objects.values())
    object_descriptions = format_object_list(all_objects)

    # Load and format template
    template = load_template("t1_qrr_classification")
    user_prompt = template.format(
        object_descriptions=object_descriptions,
        obj_A=obj_a,
        obj_A_desc=format_object_description(obj_a_data),
        obj_B=obj_b,
        obj_B_desc=format_object_description(obj_b_data),
        obj_C=obj_c,
        obj_C_desc=format_object_description(obj_c_data),
        obj_D=obj_d,
        obj_D_desc=format_object_description(obj_d_data),
        tau=tau,
        tau_percent=int(tau * 100),
    )

    system_prompt = get_system_prompt("t1_qrr", with_cot=with_cot)

    return {
        "system": system_prompt,
        "user": user_prompt,
    }


# =============================================================================
# T1-C Prompt Builder
# =============================================================================

def build_t1_trr_prompt(
    objects: Dict[str, Dict],
    query: Dict,
    with_cot: bool = False
) -> Dict[str, str]:
    """
    Build prompt for T1-C (TRR classification) task.

    Args:
        objects: Dictionary of object_id -> object data
        query: Query with "target", "ref1", "ref2" object IDs
        with_cot: Include chain-of-thought guidance

    Returns:
        Dict with "system" and "user" prompt strings
    """
    # Get object references
    target = query.get("target", "")
    ref1 = query.get("ref1", "")
    ref2 = query.get("ref2", "")

    # Get object descriptions
    target_data = objects.get(target, {})
    ref1_data = objects.get(ref1, {})
    ref2_data = objects.get(ref2, {})

    # Format all objects
    all_objects = list(objects.values())
    object_descriptions = format_object_list(all_objects)

    # Load and format template
    template = load_template("t1_trr_classification")
    user_prompt = template.format(
        object_descriptions=object_descriptions,
        obj_A=target,
        obj_A_desc=format_object_description(target_data),
        obj_B=ref1,
        obj_B_desc=format_object_description(ref1_data),
        obj_C=ref2,
        obj_C_desc=format_object_description(ref2_data),
    )

    system_prompt = get_system_prompt("t1_trr", with_cot=with_cot)

    return {
        "system": system_prompt,
        "user": user_prompt,
    }


# =============================================================================
# T2 Prompt Builder
# =============================================================================

def build_t2_extraction_prompt(
    objects: Dict[str, Dict],
    tau: float = 0.10,
    with_cot: bool = False,
    constraint_types: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Build prompt for T2 (constraint extraction) task.

    Args:
        objects: Dictionary of object_id -> object data
        tau: Tolerance parameter
        with_cot: Include chain-of-thought guidance
        constraint_types: Types to extract ("qrr", "trr", or both)

    Returns:
        Dict with "system" and "user" prompt strings
    """
    if constraint_types is None:
        constraint_types = ["qrr", "trr"]

    # Format object list
    all_objects = list(objects.values())
    object_list = format_object_list(all_objects)
    n_objects = len(all_objects)

    # Load and format template
    template = load_template("t2_constraint_extraction")
    user_prompt = template.format(
        object_list=object_list,
        n_objects=n_objects,
        tau=tau,
    )

    system_prompt = get_system_prompt("t2", with_cot=with_cot)

    return {
        "system": system_prompt,
        "user": user_prompt,
    }


# =============================================================================
# Repair Prompt Builder
# =============================================================================

def build_repair_prompt(
    original_response: Dict,
    conflicts: List[str],
    objects: Dict[str, Dict]
) -> Dict[str, str]:
    """
    Build repair prompt for hybrid predict-verify-repair loop.

    Args:
        original_response: The VLM's original (inconsistent) response
        conflicts: List of conflict descriptions
        objects: Object data for context

    Returns:
        Dict with "system" and "user" prompt strings
    """
    system_prompt = get_system_prompt("t2", with_cot=True)
    user_prompt = format_repair_prompt(conflicts)

    # Add context about original response
    user_prompt += f"\n\nYour original response contained {len(original_response.get('qrr', []))} QRR constraints."

    return {
        "system": system_prompt,
        "user": user_prompt,
    }


# =============================================================================
# Batch Prompt Generation
# =============================================================================

def generate_t1_qrr_queries(
    scene_data: Dict,
    tau: float = 0.10
) -> List[Dict]:
    """
    Generate all T1-Q queries for a scene.

    Args:
        scene_data: Scene with objects and constraints
        tau: Tolerance parameter

    Returns:
        List of query dictionaries with prompts
    """
    from itertools import combinations

    objects = {obj["id"]: obj for obj in scene_data.get("objects", [])}
    obj_ids = list(objects.keys())

    # Generate all disjoint pair combinations
    pairs = list(combinations(obj_ids, 2))
    queries = []

    for i, pair1 in enumerate(pairs):
        for pair2 in pairs[i + 1:]:
            if set(pair1) & set(pair2):  # Skip non-disjoint
                continue

            query = {
                "query_id": f"qrr_{pair1[0]}_{pair1[1]}_{pair2[0]}_{pair2[1]}",
                "A": pair1[0],
                "B": pair1[1],
                "C": pair2[0],
                "D": pair2[1],
            }

            prompts = build_t1_qrr_prompt(objects, query, tau)
            query["prompts"] = prompts
            queries.append(query)

    return queries


def generate_t1_trr_queries(scene_data: Dict) -> List[Dict]:
    """
    Generate all T1-C queries for a scene.

    Args:
        scene_data: Scene with objects

    Returns:
        List of query dictionaries with prompts
    """
    from itertools import permutations

    objects = {obj["id"]: obj for obj in scene_data.get("objects", [])}
    obj_ids = list(objects.keys())

    queries = []

    for triple in permutations(obj_ids, 3):
        target, ref1, ref2 = triple

        query = {
            "query_id": f"trr_{target}_{ref1}_{ref2}",
            "target": target,
            "ref1": ref1,
            "ref2": ref2,
        }

        prompts = build_t1_trr_prompt(objects, query)
        query["prompts"] = prompts
        queries.append(query)

    return queries
