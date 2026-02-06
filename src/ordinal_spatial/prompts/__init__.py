"""
VLM prompt templates for ORDINAL-SPATIAL benchmark.

This module provides:
- System prompts defining task context and rules
- Task-specific user prompt templates
- Prompt builders for constructing complete queries
"""

from ordinal_spatial.prompts.system_prompts import (
    SYSTEM_PROMPT_BASE,
    SYSTEM_PROMPT_T1_QRR,
    SYSTEM_PROMPT_T1_TRR,
    SYSTEM_PROMPT_T2,
    COT_SPATIAL_ANALYSIS,
    COT_CLOCK_ANALYSIS,
    REPAIR_PROMPT,
    get_system_prompt,
    format_repair_prompt,
)

from ordinal_spatial.prompts.task_prompts import (
    build_t1_qrr_prompt,
    build_t1_trr_prompt,
    build_t2_extraction_prompt,
    build_repair_prompt,
    generate_t1_qrr_queries,
    generate_t1_trr_queries,
    load_template,
    format_object_description,
    format_object_list,
)

__all__ = [
    # System prompts
    "SYSTEM_PROMPT_BASE",
    "SYSTEM_PROMPT_T1_QRR",
    "SYSTEM_PROMPT_T1_TRR",
    "SYSTEM_PROMPT_T2",
    "COT_SPATIAL_ANALYSIS",
    "COT_CLOCK_ANALYSIS",
    "REPAIR_PROMPT",
    "get_system_prompt",
    "format_repair_prompt",
    # Task prompts
    "build_t1_qrr_prompt",
    "build_t1_trr_prompt",
    "build_t2_extraction_prompt",
    "build_repair_prompt",
    "generate_t1_qrr_queries",
    "generate_t1_trr_queries",
    "load_template",
    "format_object_description",
    "format_object_list",
]
