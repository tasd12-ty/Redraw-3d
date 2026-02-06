"""
VLM 约束提取的提示词模块。

Prompt module for VLM constraint extraction.
"""

from ordinal_spatial.agents.prompts.constraint_extraction import (
    build_single_view_prompt,
    build_multi_view_prompt,
    get_system_prompt,
    get_output_schema,
)

__all__ = [
    "build_single_view_prompt",
    "build_multi_view_prompt",
    "get_system_prompt",
    "get_output_schema",
]
