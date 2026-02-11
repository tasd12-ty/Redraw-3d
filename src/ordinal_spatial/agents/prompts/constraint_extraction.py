"""
约束提取提示词构建器。

Prompt builders for VLM constraint extraction.

This module provides prompts for:
- Single-view constraint extraction (Task-3)
- Multi-view constraint extraction (Task-2)
"""

from typing import Any, Dict, List, Optional


CONSTRAINT_TYPE_DEFINITIONS = """
## Formal Constraint Types (7)

1) Axial (binary, ordered)
- Fields: obj1, obj2, relation
- relation ∈ {left_of, right_of, in_front_of, behind, above, below}
- Rule: for each pair and each axis, output at most one directional relation.
- Example: {"obj1": "obj_0", "obj2": "obj_1", "relation": "left_of"}

2) Topology (binary, unordered)
- Fields: obj1, obj2, relation
- relation ∈ {disjoint, touching, overlapping}
- Rule: exactly one topology relation per unordered pair.
- Example: {"obj1": "obj_0", "obj2": "obj_1", "relation": "disjoint"}

3) Size (binary, ordered)
- Fields: bigger, smaller
- Rule: one direction per pair; skip only if indistinguishable in size.
- Example: {"bigger": "obj_2", "smaller": "obj_4"}

4) Occlusion (binary, ordered, view-dependent)
- Fields: occluder, occluded, partial
- Rule: output only when actual visual blocking exists in the chosen view.
- Example: {"occluder": "obj_3", "occluded": "obj_1", "partial": true}

5) Closer (ternary, ordered by anchor)
- Fields: anchor, closer, farther
- Rule: for each triple, up to three entries (one anchor at a time).
- Example: {"anchor": "obj_0", "closer": "obj_1", "farther": "obj_2"}

6) TRR (ternary clock relation)
- Fields: target, ref1, ref2, hour
- hour ∈ [1, 12]
- Rule: infer clock direction of target relative to axis ref1→ref2.
- Example: {"target": "obj_1", "ref1": "obj_2", "ref2": "obj_4", "hour": 3}

7) QRR (quaternary distance comparison)
- Fields: pair1, pair2, metric, comparator
- comparator ∈ {"<", "~=", ">"}
- Rule: pair1 and pair2 must be disjoint (4 distinct objects).
- metric default: "dist3D"
- Example:
  {"pair1": ["obj_0", "obj_1"], "pair2": ["obj_2", "obj_3"],
   "metric": "dist3D", "comparator": "<"}
"""


THINKING_ANGLES = """
## Visual Reasoning Angles

- Depth cues: lower image position and stronger perspective usually imply
  nearer objects, but verify with occlusion and scale.
- Perspective: 2D proximity does not always mean 3D proximity.
- Ground-plane prior: all objects rest on the same support plane.
- Lighting and shadows: use as auxiliary depth hints, never as sole evidence.
- Occlusion order: if A blocks B, A is in front of B from that viewpoint.
- Multi-view disambiguation: use other views to resolve uncertain
  depth ordering.
"""


SYSTEM_PROMPT_BASE = """You are a strict spatial-reasoning parser.

Your job is to output a complete and self-consistent JSON constraint set from
images. You must reason in 3D using visual cues and combinatorial enumeration.

## Core Rules
- Focus on relative relations, not absolute coordinates.
- Enumerate systematically; do not skip combinations.
- Keep global consistency across all relations.
- Return ONLY valid JSON with the required schema.
""" + CONSTRAINT_TYPE_DEFINITIONS + "\n" + THINKING_ANGLES + """
## Combinatorial Coverage
Given N objects:
- Pairs: C(N,2) = N*(N-1)/2
- Triples: C(N,3) = N*(N-1)*(N-2)/6
- Pure QRR groups: 3*C(N,4) = N*(N-1)*(N-2)*(N-3)/8

Before final output, verify your counts against these formulas.
"""


SYSTEM_PROMPT_SINGLE_VIEW = SYSTEM_PROMPT_BASE + """
## Task Mode: Single View

Infer constraints from one image. Use depth cues carefully and stay
conservative when evidence is ambiguous.

For QRR comparison with tolerance tau={tau}:
- "<": clearly smaller than
- "~=": approximately equal within tolerance
- ">": clearly greater than
"""


SYSTEM_PROMPT_MULTI_VIEW = SYSTEM_PROMPT_BASE + """
## Task Mode: Multi View

You receive multiple views of the same scene.

- Build one unified object identity set across all views.
- View-invariant constraints (topology, size, closer, qrr) must be consistent
  across views.
- View-dependent constraints (axial, occlusion) should be anchored to view_0
  unless explicitly specified otherwise.
- Use cross-view evidence to resolve single-view ambiguity.

For QRR comparison with tolerance tau={tau}:
- "<": clearly smaller than
- "~=": approximately equal within tolerance
- ">": clearly greater than
"""


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
          "size_class": {
            "type": "string",
            "enum": ["tiny", "small", "medium", "large"],
          },
        },
        "required": ["id", "type", "color"],
      },
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
                "enum": [
                  "left_of",
                  "right_of",
                  "above",
                  "below",
                  "in_front_of",
                  "behind",
                ],
              },
            },
            "required": ["obj1", "obj2", "relation"],
          },
        },
        "topology": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "obj1": {"type": "string"},
              "obj2": {"type": "string"},
              "relation": {
                "type": "string",
                "enum": ["disjoint", "touching", "overlapping"],
              },
            },
            "required": ["obj1", "obj2", "relation"],
          },
        },
        "occlusion": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "occluder": {"type": "string"},
              "occluded": {"type": "string"},
              "partial": {"type": "boolean"},
            },
            "required": ["occluder", "occluded"],
          },
        },
        "size": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "bigger": {"type": "string"},
              "smaller": {"type": "string"},
            },
            "required": ["bigger", "smaller"],
          },
        },
        "closer": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "anchor": {"type": "string"},
              "closer": {"type": "string"},
              "farther": {"type": "string"},
            },
            "required": ["anchor", "closer", "farther"],
          },
        },
        "qrr": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "pair1": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
              },
              "pair2": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 2,
                "maxItems": 2,
              },
              "metric": {"type": "string"},
              "comparator": {
                "type": "string",
                "enum": ["<", "~=", ">"],
              },
            },
            "required": ["pair1", "pair2", "comparator"],
          },
        },
        "trr": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "target": {"type": "string"},
              "ref1": {"type": "string"},
              "ref2": {"type": "string"},
              "hour": {"type": "integer", "minimum": 1, "maximum": 12},
            },
            "required": ["target", "ref1", "ref2", "hour"],
          },
        },
      },
    },
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
  },
  "required": ["objects", "constraints", "confidence"],
}


SINGLE_VIEW_USER_TEMPLATE = """Analyze this image and extract ALL spatial \
constraints by exhaustive enumeration.

{object_context}

Parameters:
- Tolerance (tau): {tau}

## Required Procedure

**Step 1: Confirm objects** — Verify each listed object is visible.
Use the provided IDs exactly (do not rename).

**Step 2: Binary relations** — Enumerate ALL C(N,2) pairs:
  (a) Axial: check all 3 axes per pair (left/right, front/behind,
      above/below). Output one direction per axis. Skip an axis only
      when truly co-located. Expected: ~C(N,2)*2 to C(N,2)*3 entries.
  (b) Topology: exactly one of disjoint/touching/overlapping per pair.
      Expected: exactly C(N,2) entries.
  (c) Size: one entry per pair where sizes differ.
      Expected: up to C(N,2) entries.
  (d) Occlusion: only when one object visually blocks another.

**Step 3: Ternary relations** — Enumerate ALL C(N,3) triples:
  (a) Closer: for each triple, use each object as anchor in turn.
      Up to 3 entries per triple. Total up to C(N,3)*3.
  (b) TRR: clock direction (1-12) if confident.

**Step 4: Quaternary (pure QRR)** — For each group of 4 distinct
objects, 3 pair partitions. Compare dist(pair1) vs dist(pair2).
  - Include "metric": "dist3D" for every QRR entry.
  - Both pairs must share NO objects.
  - Expected: 3*C(N,4) entries.

**Step 5: Verify counts** — Before outputting, check:
  - topology count == C(N,2)
  - axial count ≈ C(N,2)*2 to C(N,2)*3
  - closer count ≈ C(N,3)*3
  - qrr count == 3*C(N,4)

Return ONLY valid JSON.
"""


MULTI_VIEW_USER_TEMPLATE = """Analyze these {n_views} images of the \
same scene from different camera angles.

{object_context}

Parameters:
- Tolerance (tau): {tau}

## Required Procedure

**Step 1: Cross-view identification** — The same objects appear in
every view. Use the provided IDs. Locate each object across views.

**Step 2: View-INVARIANT constraints** (must agree across views):
  (a) Topology: exactly C(N,2) entries (disjoint/touching/overlapping).
  (b) Size: up to C(N,2) entries where sizes differ.
  (c) Closer: up to C(N,3)*3 entries (3D distance ordering).
  (d) QRR: 3*C(N,4) entries (3D distance comparison).
  Use multiple views to resolve single-view ambiguity.

**Step 3: View-DEPENDENT constraints** (anchored to view_0):
  (a) Axial: left/right, front/behind, above/below from view_0.
      Expected: ~C(N,2)*2 to C(N,2)*3 entries.
  (b) Occlusion: report any occlusion visible from view_0.
  (c) TRR: clock direction from view_0 perspective.

**Step 4: Cross-view validation** — Check depth orderings against
perpendicular views. Resolve any contradictions.

**Step 5: Verify counts** — Same formulas as single-view.

## Output Rules
- Return ONE unified JSON (not per-view fragments).
- Keys must match the schema exactly.

Return ONLY valid JSON.
"""


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
  tau_str = str(tau)
  if mode == "multi":
    return SYSTEM_PROMPT_MULTI_VIEW.replace("{tau}", tau_str)
  return SYSTEM_PROMPT_SINGLE_VIEW.replace("{tau}", tau_str)


def get_output_schema() -> Dict[str, Any]:
  """
  获取输出 JSON Schema。

  Get output JSON schema for validation.
  """
  return OUTPUT_SCHEMA


def _comb(n: int, k: int) -> int:
  """组合数计算。"""
  if k < 0 or k > n:
    return 0
  if k == 0 or k == n:
    return 1
  if k == 1:
    return n
  if k == 2:
    return n * (n - 1) // 2
  if k == 3:
    return n * (n - 1) * (n - 2) // 6
  if k == 4:
    return n * (n - 1) * (n - 2) * (n - 3) // 24
  numer = 1
  denom = 1
  k = min(k, n - k)
  for i in range(1, k + 1):
    numer *= (n - k + i)
    denom *= i
  return numer // denom


def format_object_context(
  objects: Optional[List[Dict[str, Any]]] = None,
) -> str:
  """
  格式化已知物体信息。

  Format known object information for prompt context.

  Args:
    objects: Optional list of known objects.

  Returns:
    Markdown-formatted object context.
  """
  if not objects:
    return (
      "## Known Objects in Scene\n\n"
      "Objects are not provided. You must detect all visible objects first "
      "and then compute combinatorial counts from detected N."
    )

  n_objects = len(objects)
  pairs = _comb(n_objects, 2)
  triples = _comb(n_objects, 3)
  qrr_groups = 3 * _comb(n_objects, 4)

  lines = [
    f"## Known Objects in Scene (N={n_objects})",
    "",
    "| ID | Color | Size | Material | Shape |",
    "|---|---|---|---|---|",
  ]

  for idx, obj in enumerate(objects):
    obj_id = obj.get("id", f"obj_{idx}")
    color = obj.get("color", "unknown")
    size = obj.get("size", obj.get("size_class", "unknown"))
    material = obj.get("material", "unknown")
    shape = obj.get("shape", obj.get("type", "unknown"))
    lines.append(
      f"| {obj_id} | {color} | {size} | {material} | {shape} |"
    )

  lines.extend([
    "",
    f"With N={n_objects} objects:",
    f"- Pairs: C({n_objects},2) = {pairs}",
    f"- Triples: C({n_objects},3) = {triples}",
    f"- QRR groups: 3×C({n_objects},4) = {qrr_groups}",
  ])
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
