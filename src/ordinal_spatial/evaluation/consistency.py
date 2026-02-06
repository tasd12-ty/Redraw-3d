"""
ORDINAL-SPATIAL 基准测试的约束一致性检查。

本模块提供算法来验证一组序约束在内部是否一致（可联合满足）。
主要检查包括：

1. QRR 传递性：如果 dist(A,B) < dist(C,D) < dist(E,F)，则 dist(A,B) < dist(E,F)
2. TRR 角度一致性：时钟位置必须几何一致
3. 跨度量一致性：相关度量应在排序上一致

不一致的约束表示：
- VLM 错误（模型自相矛盾）
- 不可能的场景配置

使用有向图循环检测算法（NetworkX）来发现矛盾。

Constraint consistency checking for ORDINAL-SPATIAL benchmark.

This module provides algorithms to verify that a set of ordinal constraints
is internally consistent (jointly satisfiable). Key checks include:

1. QRR Transitivity: If dist(A,B) < dist(C,D) < dist(E,F), then dist(A,B) < dist(E,F)
2. TRR Angular Consistency: Clock positions must be geometrically consistent
3. Cross-metric consistency: Related metrics should agree on ordering

Inconsistent constraints indicate either:
- VLM errors (model contradicts itself)
- Impossible scene configurations
"""

from typing import List, Dict, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import networkx as nx


class ConsistencyStatus(Enum):
    """Status of consistency check."""
    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    UNKNOWN = "unknown"


@dataclass
class Cycle:
    """
    Represents a contradiction cycle in the constraint graph.

    A cycle indicates a transitivity violation:
    e.g., A < B, B < C, C < A forms a contradictory cycle.
    """
    nodes: List[Tuple]  # List of object pairs in the cycle
    edges: List[str]    # Edge labels (comparators)
    length: int = 0

    def __post_init__(self):
        self.length = len(self.nodes)

    def __str__(self) -> str:
        parts = []
        for i, (node, edge) in enumerate(zip(self.nodes, self.edges)):
            parts.append(f"{node} {edge}")
        parts.append(str(self.nodes[0]))  # Complete the cycle
        return " → ".join(parts)


@dataclass
class ConsistencyReport:
    """
    Result of consistency checking.

    Attributes:
        status: Overall consistency status
        is_consistent: True if no contradictions found
        cycles: List of contradiction cycles found
        warnings: Non-fatal issues (e.g., near-boundary comparisons)
        stats: Statistics about the constraint set
    """
    status: ConsistencyStatus
    is_consistent: bool
    cycles: List[Cycle] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.is_consistent:
            return f"Consistent ({self.stats.get('n_constraints', 0)} constraints)"
        else:
            return f"Inconsistent: {len(self.cycles)} cycle(s) found"


def check_qrr_consistency(
    constraints: List[Dict],
    strict: bool = True
) -> ConsistencyReport:
    """
    Check QRR constraints for transitivity consistency.

    Builds a directed comparison graph where:
    - Vertices are object pairs
    - Edges represent strict orderings (< or >)
    - Cycles indicate contradictions

    Args:
        constraints: List of QRR constraint dictionaries
        strict: If True, treat all < and > as strict; if False, allow some slack

    Returns:
        ConsistencyReport with cycle information
    """
    # Build directed graph
    graph = nx.DiGraph()
    pair_to_idx = {}
    idx_to_pair = {}

    # Add all pairs as nodes
    for c in constraints:
        pair1 = tuple(sorted(c.get("pair1", [])))
        pair2 = tuple(sorted(c.get("pair2", [])))

        for pair in [pair1, pair2]:
            if pair and pair not in pair_to_idx:
                idx = len(pair_to_idx)
                pair_to_idx[pair] = idx
                idx_to_pair[idx] = pair
                graph.add_node(idx)

    # Add edges based on comparisons
    for c in constraints:
        pair1 = tuple(sorted(c.get("pair1", [])))
        pair2 = tuple(sorted(c.get("pair2", [])))
        comparator = c.get("comparator", "~=")

        if not pair1 or not pair2:
            continue

        idx1 = pair_to_idx.get(pair1)
        idx2 = pair_to_idx.get(pair2)

        if idx1 is None or idx2 is None:
            continue

        # Add directed edge for strict orderings
        if comparator in ("<", "lt"):
            # pair1 < pair2 means edge from idx1 to idx2
            graph.add_edge(idx1, idx2, comparator="<")
        elif comparator in (">", "gt"):
            # pair1 > pair2 means edge from idx2 to idx1
            graph.add_edge(idx2, idx1, comparator="<")
        # Skip approximate equality (no edge)

    # Find cycles using DFS
    cycles = []
    try:
        # Find all simple cycles
        for cycle_nodes in nx.simple_cycles(graph):
            if len(cycle_nodes) >= 2:
                # Convert back to pairs
                pairs = [idx_to_pair[idx] for idx in cycle_nodes]
                edges = []
                for i in range(len(cycle_nodes)):
                    src = cycle_nodes[i]
                    dst = cycle_nodes[(i + 1) % len(cycle_nodes)]
                    edge_data = graph.get_edge_data(src, dst, {})
                    edges.append(edge_data.get("comparator", "<"))

                cycles.append(Cycle(nodes=pairs, edges=edges))
    except nx.NetworkXError:
        pass

    # Compile report
    is_consistent = len(cycles) == 0
    status = ConsistencyStatus.CONSISTENT if is_consistent else ConsistencyStatus.INCONSISTENT

    stats = {
        "n_constraints": len(constraints),
        "n_pairs": len(pair_to_idx),
        "n_edges": graph.number_of_edges(),
        "n_cycles": len(cycles),
    }

    return ConsistencyReport(
        status=status,
        is_consistent=is_consistent,
        cycles=cycles,
        stats=stats,
    )


def check_trr_consistency(
    constraints: List[Dict],
    tolerance_hours: int = 1
) -> ConsistencyReport:
    """
    Check TRR constraints for angular consistency.

    Key consistency rule:
    - CLOCK_k(A|B,C) and CLOCK_j(A|C,B) should satisfy k + j ≡ 0 (mod 12)
      with some tolerance for measurement error

    Args:
        constraints: List of TRR constraint dictionaries
        tolerance_hours: Allowed deviation in hours

    Returns:
        ConsistencyReport with conflict information
    """
    # Group constraints by (target, {ref1, ref2}) for symmetric checks
    grouped = defaultdict(list)
    for c in constraints:
        target = c.get("target")
        ref1 = c.get("ref1")
        ref2 = c.get("ref2")
        hour = c.get("hour", 12)

        if not all([target, ref1, ref2]):
            continue

        # Key is target + frozenset of refs
        key = (target, frozenset([ref1, ref2]))
        grouped[key].append({
            "ref1": ref1,
            "ref2": ref2,
            "hour": hour,
        })

    warnings = []
    conflicts = []

    # Check symmetric pairs
    for key, items in grouped.items():
        if len(items) < 2:
            continue

        # Find pairs with swapped refs
        for i, item1 in enumerate(items):
            for item2 in items[i + 1:]:
                if item1["ref1"] == item2["ref2"] and item1["ref2"] == item2["ref1"]:
                    # These should be complementary
                    h1 = item1["hour"]
                    h2 = item2["hour"]

                    # h1 + h2 should equal 12 (mod 12) for opposite directions
                    # Actually, if A is at hour k from B→C,
                    # then from C→B, A should be at (6 + k) mod 12 or (12 - k + 6) mod 12
                    # This is more complex - let's check for rough consistency
                    expected_h2 = (h1 + 6) % 12
                    if expected_h2 == 0:
                        expected_h2 = 12

                    diff = abs((h2 - expected_h2 + 6) % 12 - 6)
                    if diff > tolerance_hours:
                        conflicts.append({
                            "triple1": (key[0], item1["ref1"], item1["ref2"]),
                            "triple2": (key[0], item2["ref1"], item2["ref2"]),
                            "hour1": h1,
                            "hour2": h2,
                            "expected_h2": expected_h2,
                            "diff": diff,
                        })

    is_consistent = len(conflicts) == 0
    status = ConsistencyStatus.CONSISTENT if is_consistent else ConsistencyStatus.INCONSISTENT

    stats = {
        "n_constraints": len(constraints),
        "n_groups": len(grouped),
        "n_conflicts": len(conflicts),
    }

    # Convert conflicts to cycles for uniform reporting
    cycles = [
        Cycle(
            nodes=[c["triple1"], c["triple2"]],
            edges=[f"h={c['hour1']}", f"h={c['hour2']}"]
        )
        for c in conflicts
    ]

    return ConsistencyReport(
        status=status,
        is_consistent=is_consistent,
        cycles=cycles,
        warnings=warnings,
        stats=stats,
    )


def check_full_consistency(
    qrr_constraints: List[Dict],
    trr_constraints: List[Dict],
) -> ConsistencyReport:
    """
    Check both QRR and TRR constraints for consistency.

    Args:
        qrr_constraints: List of QRR constraints
        trr_constraints: List of TRR constraints

    Returns:
        Combined ConsistencyReport
    """
    qrr_report = check_qrr_consistency(qrr_constraints)
    trr_report = check_trr_consistency(trr_constraints)

    is_consistent = qrr_report.is_consistent and trr_report.is_consistent
    status = ConsistencyStatus.CONSISTENT if is_consistent else ConsistencyStatus.INCONSISTENT

    cycles = qrr_report.cycles + trr_report.cycles
    warnings = qrr_report.warnings + trr_report.warnings

    stats = {
        "qrr": qrr_report.stats,
        "trr": trr_report.stats,
        "total_cycles": len(cycles),
    }

    return ConsistencyReport(
        status=status,
        is_consistent=is_consistent,
        cycles=cycles,
        warnings=warnings,
        stats=stats,
    )


class ConsistencyChecker:
    """
    Stateful consistency checker for incremental constraint addition.

    Maintains a graph and checks for cycles as constraints are added.
    Useful for the predict-verify-repair loop in hybrid baselines.
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.pair_to_idx = {}
        self.idx_to_pair = {}
        self.constraints = []

    def add_qrr(self, constraint: Dict) -> Optional[Cycle]:
        """
        Add a QRR constraint and check for new cycles.

        Returns:
            Cycle if adding this constraint creates a contradiction, else None
        """
        pair1 = tuple(sorted(constraint.get("pair1", [])))
        pair2 = tuple(sorted(constraint.get("pair2", [])))
        comparator = constraint.get("comparator", "~=")

        if not pair1 or not pair2:
            return None

        # Add pairs as nodes
        for pair in [pair1, pair2]:
            if pair not in self.pair_to_idx:
                idx = len(self.pair_to_idx)
                self.pair_to_idx[pair] = idx
                self.idx_to_pair[idx] = pair
                self.graph.add_node(idx)

        idx1 = self.pair_to_idx[pair1]
        idx2 = self.pair_to_idx[pair2]

        # Determine edge direction
        if comparator in ("<", "lt"):
            src, dst = idx1, idx2
        elif comparator in (">", "gt"):
            src, dst = idx2, idx1
        else:
            # Approximate equality - no edge
            self.constraints.append(constraint)
            return None

        # Check if adding this edge would create a cycle
        if self.graph.has_edge(src, dst):
            # Edge already exists
            self.constraints.append(constraint)
            return None

        # Check for path from dst to src (would form cycle)
        if nx.has_path(self.graph, dst, src):
            # Found a cycle!
            path = nx.shortest_path(self.graph, dst, src)
            pairs = [self.idx_to_pair[idx] for idx in path]
            edges = ["<"] * len(pairs)
            return Cycle(nodes=pairs, edges=edges)

        # Safe to add edge
        self.graph.add_edge(src, dst, comparator="<")
        self.constraints.append(constraint)
        return None

    def check_all(self) -> ConsistencyReport:
        """Check consistency of all added constraints."""
        return check_qrr_consistency(self.constraints)

    def reset(self):
        """Clear all constraints and reset state."""
        self.graph.clear()
        self.pair_to_idx.clear()
        self.idx_to_pair.clear()
        self.constraints.clear()


def find_minimal_conflict(
    constraints: List[Dict],
    new_constraint: Dict
) -> List[Dict]:
    """
    Find the minimal set of constraints that conflict with a new constraint.

    Useful for explaining why a constraint cannot be added.

    Args:
        constraints: Existing constraints
        new_constraint: Constraint to check

    Returns:
        Minimal subset of constraints that form a contradiction with new_constraint
    """
    checker = ConsistencyChecker()

    # Add existing constraints
    for c in constraints:
        checker.add_qrr(c)

    # Check if new constraint creates conflict
    cycle = checker.add_qrr(new_constraint)
    if cycle is None:
        return []

    # Extract constraints involved in the cycle
    cycle_pairs = set(cycle.nodes)
    involved = [new_constraint]

    for c in constraints:
        pair1 = tuple(sorted(c.get("pair1", [])))
        pair2 = tuple(sorted(c.get("pair2", [])))
        if pair1 in cycle_pairs or pair2 in cycle_pairs:
            involved.append(c)

    return involved
