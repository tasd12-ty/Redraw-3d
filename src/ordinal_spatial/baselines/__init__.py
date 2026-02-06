"""
Baseline implementations for ORDINAL-SPATIAL benchmark.

Available baselines:
- Oracle: Direct computation from ground-truth 3D positions
- VLM Direct: Zero-shot VLM prediction
- VLM CoT: VLM with chain-of-thought prompting
- Hybrid: Predict-verify-repair loop with consistency checking
- Embedding: Ordinal embedding optimization for T3
"""

from ordinal_spatial.baselines.oracle import (
    OracleBaseline,
    oracle_predict_qrr,
    oracle_predict_trr,
    oracle_extract_osd,
)

from ordinal_spatial.baselines.vlm_direct import (
    VLMDirectBaseline,
    VLMConfig,
)

from ordinal_spatial.baselines.hybrid import (
    HybridBaseline,
    HybridConfig,
)

__all__ = [
    # Oracle
    "OracleBaseline",
    "oracle_predict_qrr",
    "oracle_predict_trr",
    "oracle_extract_osd",
    # VLM Direct
    "VLMDirectBaseline",
    "VLMConfig",
    # Hybrid
    "HybridBaseline",
    "HybridConfig",
    # Embedding (延迟导入，需要 torch)
    "OrdinalEmbedding",
    "embed_from_constraints",
]


def __getattr__(name):
    """延迟导入需要 torch 的模块。"""
    if name in ("OrdinalEmbedding", "embed_from_constraints"):
        from ordinal_spatial.baselines.embedding import (
            OrdinalEmbedding,
            embed_from_constraints,
        )
        _map = {
            "OrdinalEmbedding": OrdinalEmbedding,
            "embed_from_constraints": embed_from_constraints,
        }
        return _map[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
