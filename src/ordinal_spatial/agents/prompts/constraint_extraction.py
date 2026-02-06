"""
约束提取的提示词构建器。

Prompt builders for constraint extraction from images.

This module provides prompts for:
- Single-view constraint extraction (Task-3)
- Multi-view constraint extraction (Task-2)
"""

from typing import Dict, List, Optional, Any
from pathlib import Path


# =============================================================================
# System Prompts
# =============================================================================

SYSTEM_PROMPT_BASE = """You are a spatial reasoning expert analyzing 3D scenes from images.

## Core Principles

1. **Relative, Not Absolute**: Focus on RELATIVE spatial relationships, not exact coordinates.
2. **3D Reasoning**: Reason about 3D space, not just 2D image positions.
3. **Use Depth Cues**: Leverage occlusion, perspective, shadows, and relative sizes.
4. **Consistency**: Ensure all constraints are globally consistent (no contradictions).
5. **Exhaustive Enumeration**: You MUST systematically enumerate ALL combinatorial pairs and triples. Do NOT skip any.

## Combinatorial Completeness — CRITICAL

You MUST exhaustively cover ALL combinatorial groups. Here is how to calculate:

**Combination formula**: C(N, K) = N! / (K! × (N-K)!)

**Binary relations — Pairs (K=2)**:
  C(N, 2) = N × (N-1) / 2
  Example: N=5 → C(5,2) = 5×4/2 = 10 pairs.
  Applies to: axial, topology, size, occlusion.

**Ternary relations — Triples (K=3)**:
  C(N, 3) = N × (N-1) × (N-2) / 6
  Example: N=5 → C(5,3) = 5×4×3/6 = 10 triples.
  Applies to: closer (each triple yields up to 3 entries, one per anchor), TRR.

**Quaternary relations — Pure QRR (K=4)**:
  Compare dist(A,B) vs dist(C,D) where A,B,C,D are 4 DISTINCT objects.
  (If two pairs share an object like dist(A,B) vs dist(A,C), that is already
  a ternary "closer" relation. QRR only covers the disjoint-pair case.)
  Pure QRR count = 3 × C(N, 4) = N × (N-1) × (N-2) × (N-3) / 8
  Example: N=5 → 3×C(5,4) = 3×5 = 15 pure QRR.

**Your workflow**:
1. Count objects → N
2. Calculate C(N,2), C(N,3), and 3×C(N,4) using the formulas above
3. Enumerate ALL pairs and evaluate binary relations
4. Enumerate ALL triples and evaluate ternary relations
5. Enumerate ALL 4-object groups and evaluate quaternary (pure QRR) relations
6. Before outputting, verify your counts match the expected numbers

## Output Format

Return ONLY valid JSON. No markdown, no explanation outside JSON.
"""

SYSTEM_PROMPT_SINGLE_VIEW = SYSTEM_PROMPT_BASE + """
## Task: Single-View Constraint Extraction (Task-3)

Analyze the image and extract ALL spatial constraints by EXHAUSTIVE ENUMERATION.

### Step-by-Step Procedure:

**Step 1: Identify ALL objects** (assign id, type, color, size_class)

**Step 2: Enumerate ALL C(N,2) pairs** — For EACH pair (A, B), extract:

  (a) **Axial Relations** — Check ALL 3 spatial axes for each pair:
    - X-axis: Is A left_of or right_of B?
    - Z-axis (depth): Is A in_front_of or behind B?
    - Y-axis (vertical): Is A above or below B?
    - ONLY skip an axis if the two objects are genuinely at the same position on that axis
      (e.g., both resting on the ground plane → skip above/below).
    - Expected: For N objects, you should produce close to C(N,2) × 3 axial entries
      (minus those pairs that share the same position on some axis).

  (b) **Topology** — Exactly ONE relation per pair:
    - disjoint / touching / overlapping
    - Expected: exactly C(N,2) topology entries.

  (c) **Size Comparison** — For each pair, which is bigger:
    - Expected: exactly C(N,2) size entries.

  (d) **Occlusion** — Only if one object visually occludes another from this viewpoint:
    - Record occluder, occluded, partial (true/false).
    - This is view-dependent; only include actual occlusions.

**Step 3: Enumerate ALL C(N,3) triples** — For EACH triple (A, B, C), extract:

  (a) **Closer (Ternary Distance)** — Pick each object as anchor in turn:
    - closer(anchor=A, closer=?, farther=?) — which of B,C is closer to A?
    - Expected: For each triple, up to 3 closer entries (one per anchor choice).
      Total up to C(N,3) × 3.

  (b) **TRR (Clock Direction)** (if confident):
    - For target T with reference axis defined by ref1→ref2, what clock hour (1-12) is T at?

**Step 4: Enumerate ALL 3×C(N,4) quaternary groups** — Pure QRR:

  For each group of 4 DISTINCT objects (A, B, C, D), there are 3 ways to
  partition them into 2 pairs: (AB,CD), (AC,BD), (AD,BC).
  For each partition, compare dist(pair1) vs dist(pair2).
  - Use comparator: "<" (less), "~=" (approximately equal), ">" (greater).
  - IMPORTANT: Both pairs must have NO shared objects (4 distinct objects).
    Comparisons with shared objects (like dist(A,B) vs dist(A,C)) belong to
    "closer" in Step 3, NOT here.
  - Expected: 3 × C(N,4) = N(N-1)(N-2)(N-3)/8 QRR entries.

### Tolerance Parameter (tau)

When comparing distances, use tolerance tau={tau}:
- "<" means strictly less (outside tolerance band)
- "~=" means approximately equal (within tolerance)
- ">" means strictly greater

### Consistency Rules

- If A left_of B, then B right_of A (do NOT output both; output one direction per axis per pair)
- If A in_front_of B and B in_front_of C, then A in_front_of C (transitivity)
- If A bigger than B and B bigger than C, then A bigger than C
- Distance comparisons must not form contradictory cycles

### Important Notes

- For AXIAL: output only ONE direction per axis per pair (e.g., "A left_of B", NOT both "A left_of B" and "B right_of A")
- For SIZE: output only ONE entry per pair (e.g., "bigger: A, smaller: B")
- For TOPOLOGY: output exactly ONE entry per pair
- For CLOSER: output one entry per anchor choice per triple
"""

SYSTEM_PROMPT_MULTI_VIEW = SYSTEM_PROMPT_BASE + """
## Task: Multi-View Constraint Extraction (Task-2)

Analyze multiple images of the SAME scene from different viewpoints.
Apply the same exhaustive enumeration procedure as single-view.

### Goals:

1. Identify view-INVARIANT constraints (hold across all views):
   - 3D topology, physical sizes, 3D distances, closer relations

2. Identify view-DEPENDENT constraints (vary by viewpoint):
   - Occlusion, axial relations (left/right/front/behind depend on camera)

3. Resolve ambiguities using multiple views

### Process:

1. Identify ALL objects across views (same objects, different angles)
2. Calculate C(N,2) pairs and C(N,3) triples
3. For EACH pair: evaluate all 3 axes, topology, size, occlusion per view
4. For EACH triple: evaluate closer relations
5. Cross-reference views to validate and resolve inconsistencies
"""


# =============================================================================
# Output Schema
# =============================================================================

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {"type": "string"},
                    "color": {"type": "string"},
                    "size_class": {"type": "string", "enum": ["tiny", "small", "medium", "large"]},
                },
                "required": ["id", "type", "color"]
            }
        },
        "constraints": {
            "type": "object",
            "properties": {
                "axial": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "obj1": {"type": "string"},
                            "obj2": {"type": "string"},
                            "relation": {
                                "type": "string",
                                "enum": ["left_of", "right_of", "above", "below", "in_front_of", "behind"]
                            }
                        },
                        "required": ["obj1", "obj2", "relation"]
                    }
                },
                "topology": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "obj1": {"type": "string"},
                            "obj2": {"type": "string"},
                            "relation": {"type": "string", "enum": ["disjoint", "touching", "overlapping"]}
                        },
                        "required": ["obj1", "obj2", "relation"]
                    }
                },
                "occlusion": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "occluder": {"type": "string"},
                            "occluded": {"type": "string"},
                            "partial": {"type": "boolean"}
                        },
                        "required": ["occluder", "occluded"]
                    }
                },
                "size": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "bigger": {"type": "string"},
                            "smaller": {"type": "string"}
                        },
                        "required": ["bigger", "smaller"]
                    }
                },
                "closer": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "anchor": {"type": "string"},
                            "closer": {"type": "string"},
                            "farther": {"type": "string"}
                        },
                        "required": ["anchor", "closer", "farther"]
                    }
                },
                "qrr": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pair1": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                            "pair2": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2},
                            "metric": {"type": "string"},
                            "comparator": {"type": "string", "enum": ["<", "~=", ">"]}
                        },
                        "required": ["pair1", "pair2", "comparator"]
                    }
                },
                "trr": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "target": {"type": "string"},
                            "ref1": {"type": "string"},
                            "ref2": {"type": "string"},
                            "hour": {"type": "integer", "minimum": 1, "maximum": 12}
                        },
                        "required": ["target", "ref1", "ref2", "hour"]
                    }
                }
            }
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    },
    "required": ["objects", "constraints", "confidence"]
}


# =============================================================================
# User Prompt Templates
# =============================================================================

SINGLE_VIEW_USER_TEMPLATE = """Analyze this image and extract ALL spatial constraints by exhaustive enumeration.

{object_context}

Parameters:
- Tolerance (tau): {tau}

## Procedure

**Step 1: Identify ALL objects** — List every visible object with id, type, color, size_class.

**Step 2: Binary relations** — For EVERY pair (check all C(N,2) = N*(N-1)/2 pairs):
  - **Axial**: Judge all 3 axes (left/right, front/behind, above/below) per pair.
    Skip an axis ONLY if the two objects are at the same position on that axis.
    Output one direction per axis (e.g., obj1 left_of obj2, not both directions).
  - **Topology**: Exactly one of disjoint/touching/overlapping per pair.
  - **Size**: One entry per pair (bigger, smaller). Skip only if truly same size.
  - **Occlusion**: Only if one actually occludes the other from this viewpoint.

**Step 3: Ternary relations** — For EVERY triple (check all C(N,3) = N*(N-1)*(N-2)/6 triples):
  - **Closer**: For each triple (A,B,C), use each as anchor in turn and state which
    of the other two is closer. Up to 3 entries per triple.
  - **TRR**: Clock direction (1-12) if confident.

**Step 4: Quaternary relations (Pure QRR)** — For EVERY group of 4 distinct objects:
  - Each group of 4 gives 3 pair partitions. Compare dist(pair1) vs dist(pair2).
  - Total: 3 * C(N,4) = N*(N-1)*(N-2)*(N-3)/8 entries.
  - IMPORTANT: Both pairs must share NO objects. Shared-object comparisons
    (like dist(A,B) vs dist(A,C)) are already covered by "closer" above.

## Expected Counts (calculate and verify)

Compute N = number of objects, then:
- Binary:     C(N,2) = N*(N-1)/2 pairs
  - topology: exactly C(N,2)
  - size:     exactly C(N,2) (minus same-size pairs)
  - axial:    ~C(N,2)*2 to C(N,2)*3 (per-axis, minus co-located)
  - occlusion: only actual occlusions
- Ternary:    C(N,3) = N*(N-1)*(N-2)/6 triples
  - closer:   up to C(N,3)*3
- Quaternary: 3*C(N,4) = N*(N-1)*(N-2)*(N-3)/8
  - qrr:      exactly 3*C(N,4)

## Output JSON

Return ONLY valid JSON with keys exactly as shown:
```json
{{
  "objects": [
    {{"id": "color_shape", "type": "shape", "color": "color", "size_class": "small"}}
  ],
  "constraints": {{
    "axial": [
      {{"obj1": "A", "obj2": "B", "relation": "left_of"}}
    ],
    "topology": [
      {{"obj1": "A", "obj2": "B", "relation": "disjoint"}}
    ],
    "occlusion": [
      {{"occluder": "A", "occluded": "B", "partial": true}}
    ],
    "size": [
      {{"bigger": "A", "smaller": "B"}}
    ],
    "closer": [
      {{"anchor": "A", "closer": "B", "farther": "C"}}
    ],
    "qrr": [
      {{"pair1": ["A","B"], "pair2": ["C","D"], "comparator": "<"}}
    ],
    "trr": [
      {{"target": "T", "ref1": "R1", "ref2": "R2", "hour": 3}}
    ]
  }},
  "confidence": 0.0-1.0
}}
```"""

MULTI_VIEW_USER_TEMPLATE = """Analyze these {n_views} images of the same scene from different viewpoints.

{object_context}

Parameters:
- Tolerance (tau): {tau}

For each constraint, determine if it's view-invariant (3D) or view-dependent.

Return JSON with the same schema as single-view, plus metadata about view consistency."""


# =============================================================================
# Prompt Building Functions
# =============================================================================

def get_system_prompt(mode: str = "single", tau: float = 0.10) -> str:
    """
    获取系统提示词。

    Get system prompt for the specified mode.

    Args:
        mode: "single" or "multi"
        tau: Tolerance parameter

    Returns:
        System prompt string
    """
    if mode == "multi":
        return SYSTEM_PROMPT_MULTI_VIEW.format(tau=tau)
    return SYSTEM_PROMPT_SINGLE_VIEW.format(tau=tau)


def get_output_schema() -> Dict[str, Any]:
    """
    获取输出 JSON Schema。

    Get output JSON schema for validation.
    """
    return OUTPUT_SCHEMA


def format_object_context(objects: Optional[List[Dict[str, Any]]] = None) -> str:
    """
    格式化已知物体信息。

    Format known object information for the prompt.

    Args:
        objects: Optional list of known objects

    Returns:
        Formatted string describing objects
    """
    if not objects:
        return "Objects: Not specified (identify all visible objects)"

    lines = ["Known objects in scene:"]
    for obj in objects:
        obj_id = obj.get("id", "unknown")
        obj_type = obj.get("shape", obj.get("type", "unknown"))
        color = obj.get("color", "unknown")
        size = obj.get("size", "medium")
        lines.append(f"  - {obj_id}: {color} {obj_type} ({size})")

    return "\n".join(lines)


def build_single_view_prompt(
    objects: Optional[List[Dict[str, Any]]] = None,
    tau: float = 0.10,
) -> Dict[str, str]:
    """
    构建单视角提取提示词。

    Build prompt for single-view constraint extraction.

    Args:
        objects: Optional list of known objects
        tau: Tolerance parameter

    Returns:
        Dict with "system" and "user" prompts
    """
    object_context = format_object_context(objects)

    user_prompt = SINGLE_VIEW_USER_TEMPLATE.format(
        object_context=object_context,
        tau=tau,
    )

    return {
        "system": get_system_prompt("single", tau),
        "user": user_prompt,
    }


def build_multi_view_prompt(
    n_views: int,
    objects: Optional[List[Dict[str, Any]]] = None,
    tau: float = 0.10,
) -> Dict[str, str]:
    """
    构建多视角提取提示词。

    Build prompt for multi-view constraint extraction.

    Args:
        n_views: Number of views/images
        objects: Optional list of known objects
        tau: Tolerance parameter

    Returns:
        Dict with "system" and "user" prompts
    """
    object_context = format_object_context(objects)

    user_prompt = MULTI_VIEW_USER_TEMPLATE.format(
        n_views=n_views,
        object_context=object_context,
        tau=tau,
    )

    return {
        "system": get_system_prompt("multi", tau),
        "user": user_prompt,
    }
