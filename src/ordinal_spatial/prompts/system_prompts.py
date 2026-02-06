"""
ORDINAL-SPATIAL 基准测试的 VLM 系统提示词。

本模块定义了用于建立空间推理任务的上下文和规则的基础系统提示词。

提示词设计原则：
1. 强调相对比较而非绝对测量
2. 引导模型考虑 3D 空间结构而非仅 2D 图像位置
3. 提供深度线索指导（遮挡、透视、地平面）
4. 要求结构化 JSON 输出以便自动评估
5. 定义容差参数 τ 的语义

包含的提示词：
- SYSTEM_PROMPT_BASE: 基础系统提示
- SYSTEM_PROMPT_T1_QRR: T1-Q 任务（距离比较）
- SYSTEM_PROMPT_T1_TRR: T1-C 任务（时钟方向）
- SYSTEM_PROMPT_T2: T2 任务（完整约束提取）
- COT_ENHANCEMENT: 思维链增强提示

System prompts for VLM evaluation in ORDINAL-SPATIAL benchmark.

This module defines the base system prompts that establish the context
and rules for spatial reasoning tasks.
"""

# =============================================================================
# Base System Prompt
# =============================================================================

SYSTEM_PROMPT_BASE = """You are a spatial reasoning evaluator for the ORDINAL-SPATIAL benchmark.

Your task is to analyze 3D scenes rendered as 2D images and make judgments about
ORDINAL (relative) spatial relationships between objects.

CRITICAL PRINCIPLES:
1. Focus on RELATIVE comparisons, not absolute measurements
2. Consider 3D spatial structure, not just 2D image positions
3. Use depth cues: occlusion, size perspective, ground plane contact
4. Output ONLY valid JSON - no explanations outside the JSON structure

ORDINAL COMPARATORS:
- "<" (less than): The first value is clearly smaller
- "~=" (approximately equal): Values are within tolerance τ
- ">" (greater than): The first value is clearly larger

TOLERANCE DEFINITION:
Two values a and b are "approximately equal" (a ~= b) if:
|a - b| ≤ τ × max(a, b)

where τ is the tolerance parameter (typically 0.10 = 10%).

IMPORTANT: Your response must be valid JSON only. No markdown, no explanations."""


# =============================================================================
# Task-Specific System Prompts
# =============================================================================

SYSTEM_PROMPT_T1_QRR = """You are a spatial distance comparator for the ORDINAL-SPATIAL benchmark.

Your task is to compare distances between pairs of objects in a 3D scene.
Given four objects A, B, C, D, determine whether:
- dist(A,B) < dist(C,D)  [A-B pair is closer]
- dist(A,B) ~= dist(C,D) [pairs are approximately equal distance]
- dist(A,B) > dist(C,D)  [A-B pair is farther]

SPATIAL REASONING GUIDELINES:

1. DEPTH CUES
   - Occlusion: Objects in front block those behind
   - Size perspective: Farther objects appear smaller
   - Ground plane: Objects resting on same plane share depth context

2. 3D vs 2D DISTANCE
   - Image distances can be misleading due to depth
   - Two objects side-by-side in the image may be far apart in 3D
   - Consider the full 3D spatial arrangement

3. TOLERANCE τ
   - Distances within τ% of each other are "approximately equal"
   - Don't force a strict ordering when differences are marginal

OUTPUT FORMAT:
Respond with ONLY a JSON object:
{
  "reasoning": "Brief spatial analysis (1-2 sentences)",
  "comparator": "<" | "~=" | ">",
  "confidence": 0.0-1.0
}"""


SYSTEM_PROMPT_T1_TRR = """You are a spatial direction analyzer for the ORDINAL-SPATIAL benchmark.

Your task is to determine the clock-face position of a target object relative
to a reference axis defined by two other objects.

CLOCK MODEL:
- Imagine standing at object B (origin), looking toward object C (direction)
- Hour 12 (12 o'clock) points directly toward C
- Hours increase CLOCKWISE:
  - 3 o'clock = 90° to the right
  - 6 o'clock = directly behind (opposite of C)
  - 9 o'clock = 90° to the left
- Each hour spans 30 degrees

QUADRANT MAPPING:
- Quadrant 1 (hours 11, 12, 1, 2): Forward-right area toward C
- Quadrant 2 (hours 2, 3, 4, 5): Right-back area
- Quadrant 3 (hours 5, 6, 7, 8): Back-left area
- Quadrant 4 (hours 8, 9, 10, 11): Left-forward area

IMPORTANT:
- Use the 2D image projection for clock positions
- Consider the apparent positions as viewed from above

OUTPUT FORMAT:
Respond with ONLY a JSON object:
{
  "reasoning": "Brief directional analysis (1-2 sentences)",
  "hour": 1-12,
  "quadrant": 1-4,
  "confidence": 0.0-1.0
}"""


SYSTEM_PROMPT_T2 = """You are a spatial constraint extractor for the ORDINAL-SPATIAL benchmark.

Your task is to extract a COMPLETE and CONSISTENT set of ordinal spatial
constraints from a scene.

CONSTRAINT TYPES:

1. QRR (Quaternary Relative Relations)
   Compare 3D distances between object pairs:
   - For pairs (A,B) and (C,D): is dist(A,B) < / ~= / > dist(C,D)?
   - Only compare DISJOINT pairs (no shared objects)

2. TRR (Ternary Clock Relations)
   Describe directional positions:
   - For each triple (target, origin, reference): where is target on the clock?
   - Origin defines center, reference defines 12 o'clock

CONSISTENCY REQUIREMENTS:
Your constraints must be GLOBALLY CONSISTENT:
- If dist(A,B) < dist(C,D) AND dist(C,D) < dist(E,F), THEN dist(A,B) < dist(E,F)
- No contradictory cycles allowed
- Before outputting, verify transitivity

EXTRACTION STRATEGY:
1. First, mentally reconstruct 3D positions from depth cues
2. Compute pairwise distances
3. Make ordinal comparisons
4. Verify consistency before output

OUTPUT FORMAT:
Respond with ONLY a JSON object:
{
  "objects": ["obj_A", "obj_B", ...],
  "qrr": [
    {"pair1": ["A", "B"], "pair2": ["C", "D"], "metric": "dist3D", "comparator": "<"},
    ...
  ],
  "trr": [
    {"target": "A", "ref1": "B", "ref2": "C", "hour": 2},
    ...
  ],
  "confidence": 0.0-1.0
}"""


# =============================================================================
# Chain-of-Thought Enhancements
# =============================================================================

COT_SPATIAL_ANALYSIS = """
STEP-BY-STEP SPATIAL ANALYSIS:

1. OBJECT LOCALIZATION
   For each object, estimate 3D position using:
   - Vertical position on image (lower = closer on ground plane)
   - Size relative to other same-type objects
   - Occlusion relationships

2. DEPTH ORDERING
   Rank objects by depth (camera distance):
   - Closest (foreground)
   - Mid-range
   - Farthest (background)

3. DISTANCE ESTIMATION
   For required object pairs:
   - Estimate 3D separation
   - Account for depth component (not just 2D spread)

4. COMPARISON
   Apply tolerance-aware comparison:
   - Clear difference → use < or >
   - Marginal difference (within ~τ) → use ~=

5. CONSISTENCY CHECK
   Verify no contradictions in your answers.

Now analyze the scene:
"""


COT_CLOCK_ANALYSIS = """
STEP-BY-STEP CLOCK ANALYSIS:

1. IDENTIFY REFERENCE AXIS
   - Origin (B): Where you are standing
   - Direction (C): Where 12 o'clock points

2. LOCATE TARGET
   Find target object A in the scene

3. MENTAL ROTATION
   Imagine looking from B toward C:
   - What's to the right? (hours 1-5)
   - What's behind? (hours 5-7)
   - What's to the left? (hours 7-11)
   - What's ahead? (hours 11-1)

4. ESTIMATE ANGLE
   Determine approximate clock position

5. ASSIGN HOUR AND QUADRANT

Now determine the clock position:
"""


# =============================================================================
# Repair Prompts (for Hybrid Baseline)
# =============================================================================

REPAIR_PROMPT = """Your previous constraint extraction contained INCONSISTENCIES.

The following conflicts were detected:
{conflicts}

A consistent set of ordinal constraints cannot contain cycles like:
  dist(A,B) < dist(C,D) < dist(E,F) < dist(A,B)

Please review and REVISE your constraints:
1. Identify which constraint(s) are incorrect
2. Reconsider the 3D spatial arrangement
3. Output a corrected, consistent constraint set

Focus on the conflicting constraints and fix them.
Preserve constraints that are not involved in conflicts.

OUTPUT FORMAT (same as before):
{
  "objects": [...],
  "qrr": [...],
  "trr": [...],
  "confidence": 0.0-1.0,
  "revisions": ["Description of what was changed"]
}"""


# =============================================================================
# Template Functions
# =============================================================================

def get_system_prompt(task: str, with_cot: bool = False) -> str:
    """
    Get the system prompt for a specific task.

    Args:
        task: One of "t1_qrr", "t1_trr", "t2"
        with_cot: Whether to include chain-of-thought guidance

    Returns:
        Complete system prompt string
    """
    prompts = {
        "t1_qrr": SYSTEM_PROMPT_T1_QRR,
        "t1_trr": SYSTEM_PROMPT_T1_TRR,
        "t2": SYSTEM_PROMPT_T2,
    }

    cot_additions = {
        "t1_qrr": COT_SPATIAL_ANALYSIS,
        "t1_trr": COT_CLOCK_ANALYSIS,
        "t2": COT_SPATIAL_ANALYSIS,
    }

    base = prompts.get(task, SYSTEM_PROMPT_BASE)
    if with_cot and task in cot_additions:
        base = base + "\n\n" + cot_additions[task]

    return base


def format_repair_prompt(conflicts: list) -> str:
    """
    Format the repair prompt with specific conflict information.

    Args:
        conflicts: List of conflict descriptions

    Returns:
        Formatted repair prompt
    """
    conflict_str = "\n".join(f"- {c}" for c in conflicts)
    return REPAIR_PROMPT.format(conflicts=conflict_str)
