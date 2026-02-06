"""
简单的 Token 计数工具。

Simple token counting utility.
支持: GPT, Gemma, Qwen 系列模型。
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# 缓存已加载的 tokenizer
_tokenizer_cache = {}


def get_tokenizer(model_name: str):
    """
    根据模型名获取 tokenizer。

    Args:
        model_name: 模型名称 (如 "google/gemma-3-27b-it", "gpt-4o", "qwen2.5")

    Returns:
        tokenizer 对象
    """
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    model_lower = model_name.lower()

    # GPT 系列 - 使用 tiktoken
    if "gpt" in model_lower or "openai" in model_lower:
        try:
            import tiktoken
            tokenizer = tiktoken.get_encoding("cl100k_base")
            _tokenizer_cache[model_name] = ("tiktoken", tokenizer)
            return _tokenizer_cache[model_name]
        except ImportError:
            logger.warning("tiktoken not installed, using transformers fallback")

    # Gemma, Qwen, 其他模型 - 使用 transformers
    try:
        from transformers import AutoTokenizer

        # 选择合适的 tokenizer 模型
        if "gemma" in model_lower:
            tokenizer_id = "google/gemma-2-2b"
        elif "qwen" in model_lower:
            tokenizer_id = "Qwen/Qwen2.5-0.5B"
        elif "llama" in model_lower:
            tokenizer_id = "meta-llama/Llama-2-7b-hf"
        else:
            tokenizer_id = "gpt2"  # fallback

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_id,
            trust_remote_code=True,
            use_fast=True,
        )
        _tokenizer_cache[model_name] = ("transformers", tokenizer)
        return _tokenizer_cache[model_name]

    except Exception as e:
        logger.warning(f"Failed to load tokenizer: {e}, using char estimate")
        _tokenizer_cache[model_name] = ("estimate", None)
        return _tokenizer_cache[model_name]


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    计算文本的 token 数量。

    Args:
        text: 输入文本
        model_name: 模型名称

    Returns:
        Token 数量
    """
    if not text:
        return 0

    tokenizer_type, tokenizer = get_tokenizer(model_name)

    try:
        if tokenizer_type == "tiktoken":
            return len(tokenizer.encode(text))
        elif tokenizer_type == "transformers":
            return len(tokenizer.encode(text, add_special_tokens=False))
        else:
            # 粗略估计: ~4 字符/token
            return len(text) // 4
    except Exception as e:
        logger.warning(f"Token counting failed: {e}")
        return len(text) // 4
