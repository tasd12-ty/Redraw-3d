"""
ORDINAL-SPATIAL 基准测试的 VLM 直接基线。

本基线使用视觉-语言模型从图像中直接预测序空间关系。

特性：
- 零样本预测（无需微调）
- 支持多种 VLM（通过 OpenRouter 或直接 API）
- 可选思维链（Chain-of-Thought）提示
- 结构化 JSON 输出
- 自动重试和错误处理

支持的任务：
- T1-Q: QRR 分类
- T1-C: TRR 分类
- T2: 完整约束提取

VLM Direct baseline for ORDINAL-SPATIAL benchmark.

This baseline uses vision-language models to directly predict
ordinal spatial relations from images.
"""

import json
import base64
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class VLMConfig:
    """
    VLM 基线的配置。

    Configuration for VLM baseline.
    """
    model: str = "google/gemini-2.0-flash-001"
    api_base: str = "https://openrouter.ai/api/v1"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None  # None = use API default (max)
    timeout: float = 60.0
    retry_count: int = 3
    retry_delay: float = 1.0
    with_cot: bool = False


class VLMDirectBaseline:
    """
    直接查询视觉-语言模型的 VLM 基线。

    通过 OpenRouter 或直接 API 支持多种 VLM 提供商。

    VLM baseline that directly queries a vision-language model.

    Supports multiple VLM providers through OpenRouter or direct API.
    """

    def __init__(self, config: VLMConfig = None):
        """
        初始化 VLM 基线。

        参数:
            config: VLM 配置

        Initialize VLM baseline.

        Args:
            config: VLM configuration
        """
        self.config = config or VLMConfig()
        self._client = None

    @property
    def client(self):
        """
        延迟初始化 API 客户端。

        Lazy initialization of API client.
        """
        if self._client is None:
            try:
                from openai import OpenAI
                import os

                api_key = self.config.api_key or os.environ.get("OPENROUTER_API_KEY")
                if not api_key:
                    raise ValueError(
                        "API key required. Set OPENROUTER_API_KEY env var or pass in config."
                    )

                self._client = OpenAI(
                    base_url=self.config.api_base,
                    api_key=api_key,
                )
            except ImportError:
                raise ImportError("openai package required. Install with: pip install openai")

        return self._client

    def predict_qrr(
        self,
        image: Union[str, bytes],
        objects: Dict[str, Dict],
        pair1: tuple,
        pair2: tuple,
        metric: str = "dist3D",
        tau: float = 0.10,
    ) -> Dict[str, Any]:
        """
        使用 VLM 预测 QRR 比较。

        参数:
            image: 图像路径或 base64 编码的字节
            objects: 物体数据字典
            pair1: 第一对物体 ID
            pair2: 第二对物体 ID
            metric: 度量类型
            tau: 容差

        返回:
            包含比较器和置信度的预测字典

        Predict QRR comparison using VLM.

        Args:
            image: Image path or base64-encoded bytes
            objects: Object data dictionary
            pair1: First pair of object IDs
            pair2: Second pair of object IDs
            metric: Metric type
            tau: Tolerance

        Returns:
            Prediction dictionary with comparator and confidence
        """
        from ordinal_spatial.prompts import build_t1_qrr_prompt

        # Build prompt
        query = {
            "objects": {"A": pair1[0], "B": pair1[1], "C": pair2[0], "D": pair2[1]},
        }
        prompts = build_t1_qrr_prompt(
            objects, query, tau=tau, with_cot=self.config.with_cot
        )

        # Build messages
        messages = self._build_messages(prompts, image)

        # Call API
        response = self._call_api(messages)

        # Parse response
        result = self._parse_qrr_response(response)
        return result

    def predict_trr(
        self,
        image: Union[str, bytes],
        objects: Dict[str, Dict],
        target: str,
        ref1: str,
        ref2: str,
    ) -> Dict[str, Any]:
        """
        使用 VLM 预测 TRR 时钟位置。

        参数:
            image: 图像路径或 base64 编码的字节
            objects: 物体数据字典
            target: 目标物体 ID
            ref1: 原点物体 ID
            ref2: 方向参考物体 ID

        返回:
            包含小时和置信度的预测字典

        Predict TRR clock position using VLM.

        Args:
            image: Image path or base64-encoded bytes
            objects: Object data dictionary
            target: Target object ID
            ref1: Origin object ID
            ref2: Direction reference object ID

        Returns:
            Prediction dictionary with hour and confidence
        """
        from ordinal_spatial.prompts import build_t1_trr_prompt

        query = {"target": target, "ref1": ref1, "ref2": ref2}
        prompts = build_t1_trr_prompt(
            objects, query, with_cot=self.config.with_cot
        )

        messages = self._build_messages(prompts, image)
        response = self._call_api(messages)
        result = self._parse_trr_response(response)
        return result

    def extract_constraints(
        self,
        image: Union[str, bytes],
        objects: Dict[str, Dict],
        tau: float = 0.10,
    ) -> Dict[str, Any]:
        """
        使用 VLM 提取完整约束集（T2 任务）。

        参数:
            image: 图像路径或 base64 编码的字节
            objects: 物体数据字典
            tau: 容差

        返回:
            提取的 OSD 预测

        Extract full constraint set using VLM (T2 task).

        Args:
            image: Image path or base64-encoded bytes
            objects: Object data dictionary
            tau: Tolerance

        Returns:
            Extracted OSD prediction
        """
        from ordinal_spatial.prompts import build_t2_extraction_prompt

        prompts = build_t2_extraction_prompt(
            objects, tau=tau, with_cot=self.config.with_cot
        )

        messages = self._build_messages(prompts, image)
        response = self._call_api(messages)
        result = self._parse_osd_response(response)
        return result

    def _build_messages(
        self,
        prompts: Dict[str, str],
        image: Union[str, bytes],
    ) -> List[Dict]:
        """
        构建包含图像的 API 消息列表。

        Build API message list with image.
        """
        messages = []

        # System message
        if prompts.get("system"):
            messages.append({
                "role": "system",
                "content": prompts["system"]
            })

        # User message with image
        image_data = self._encode_image(image)
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_data}"}
            },
            {
                "type": "text",
                "text": prompts["user"]
            }
        ]

        messages.append({
            "role": "user",
            "content": user_content
        })

        return messages

    def _encode_image(self, image: Union[str, bytes]) -> str:
        """
        将图像编码为 base64。

        Encode image to base64.
        """
        if isinstance(image, bytes):
            return base64.b64encode(image).decode()
        elif isinstance(image, str):
            path = Path(image)
            if path.exists():
                return base64.b64encode(path.read_bytes()).decode()
            else:
                # Assume already base64
                return image
        else:
            raise ValueError(f"Invalid image type: {type(image)}")

    def _call_api(self, messages: List[Dict]) -> str:
        """
        调用 VLM API 并包含重试逻辑。

        Call VLM API with retry logic.
        """
        import time

        last_error = None

        for attempt in range(self.config.retry_count):
            try:
                # Build request params
                params = {
                    "model": self.config.model,
                    "messages": messages,
                    "temperature": self.config.temperature,
                }
                if self.config.max_tokens is not None:
                    params["max_tokens"] = self.config.max_tokens

                response = self.client.chat.completions.create(**params)
                content = response.choices[0].message.content

                # Log token usage
                self._log_token_usage(messages, content, response)

                return content

            except Exception as e:
                last_error = e
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.config.retry_count - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))

        raise RuntimeError(f"API call failed after {self.config.retry_count} attempts: {last_error}")

    def _log_token_usage(self, messages: List[Dict], response_text: str, response):
        """
        记录 token 使用情况。

        Log token usage for debugging.
        """
        try:
            # 优先使用 API 返回的 usage 信息
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                prompt_tokens = getattr(usage, 'prompt_tokens', None)
                completion_tokens = getattr(usage, 'completion_tokens', None)
                total_tokens = getattr(usage, 'total_tokens', None)

                if prompt_tokens is not None:
                    logger.info(
                        f"[Token Usage] API reported: "
                        f"prompt={prompt_tokens}, completion={completion_tokens}, "
                        f"total={total_tokens}"
                    )
                    return

            # API 没有返回 usage，使用 tokenizer 估算
            from ordinal_spatial.utils.token_counter import count_tokens

            # 计算输入 tokens (只计算文本部分)
            input_text = ""
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str):
                    input_text += content
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            input_text += item.get("text", "")

            prompt_tokens = count_tokens(input_text, self.config.model)
            completion_tokens = count_tokens(response_text, self.config.model)

            logger.info(
                f"[Token Usage] Estimated ({self.config.model}): "
                f"prompt≈{prompt_tokens}, completion≈{completion_tokens}, "
                f"total≈{prompt_tokens + completion_tokens}"
            )

            # 检查是否可能被截断
            if completion_tokens > 3000:
                logger.warning(
                    f"[Token Warning] Large completion ({completion_tokens} tokens), "
                    "response may be truncated. Consider increasing max_tokens."
                )

        except Exception as e:
            logger.debug(f"Token counting failed: {e}")

    def _parse_qrr_response(self, response: str) -> Dict[str, Any]:
        """
        从 VLM 响应中解析 QRR 预测。

        Parse QRR prediction from VLM response.
        """
        try:
            data = self._extract_json(response)
            comparator = data.get("comparator", "~=")

            # Normalize comparator
            comparator = self._normalize_comparator(comparator)

            return {
                "comparator": comparator,
                "confidence": float(data.get("confidence", 0.5)),
                "reasoning": data.get("reasoning", ""),
                "raw_response": response,
            }

        except Exception as e:
            logger.warning(f"Failed to parse QRR response: {e}")
            return {
                "comparator": "~=",
                "confidence": 0.0,
                "reasoning": f"Parse error: {e}",
                "raw_response": response,
                "parse_error": True,
            }

    def _parse_trr_response(self, response: str) -> Dict[str, Any]:
        """
        从 VLM 响应中解析 TRR 预测。

        Parse TRR prediction from VLM response.
        """
        try:
            data = self._extract_json(response)
            hour = int(data.get("hour", 12))
            hour = max(1, min(12, hour))  # Clamp to valid range

            from ordinal_spatial.dsl.predicates import hour_to_quadrant
            quadrant = data.get("quadrant", hour_to_quadrant(hour))

            return {
                "hour": hour,
                "quadrant": int(quadrant),
                "confidence": float(data.get("confidence", 0.5)),
                "reasoning": data.get("reasoning", ""),
                "raw_response": response,
            }

        except Exception as e:
            logger.warning(f"Failed to parse TRR response: {e}")
            return {
                "hour": 12,
                "quadrant": 1,
                "confidence": 0.0,
                "reasoning": f"Parse error: {e}",
                "raw_response": response,
                "parse_error": True,
            }

    def _parse_osd_response(self, response: str) -> Dict[str, Any]:
        """
        从 VLM 响应中解析 OSD 提取结果。

        Parse OSD extraction from VLM response.
        """
        try:
            data = self._extract_json(response)

            # Normalize QRR constraints
            qrr = []
            for c in data.get("qrr", []):
                qrr.append({
                    "pair1": c.get("pair1", []),
                    "pair2": c.get("pair2", []),
                    "metric": c.get("metric", "dist3D"),
                    "comparator": self._normalize_comparator(c.get("comparator", "~=")),
                })

            # Normalize TRR constraints
            trr = []
            for c in data.get("trr", []):
                hour = int(c.get("hour", 12))
                hour = max(1, min(12, hour))
                trr.append({
                    "target": c.get("target", ""),
                    "ref1": c.get("ref1", ""),
                    "ref2": c.get("ref2", ""),
                    "hour": hour,
                })

            return {
                "objects": data.get("objects", []),
                "qrr": qrr,
                "trr": trr,
                "confidence": float(data.get("confidence", 0.5)),
                "raw_response": response,
            }

        except Exception as e:
            logger.warning(f"Failed to parse OSD response: {e}")
            return {
                "objects": [],
                "qrr": [],
                "trr": [],
                "confidence": 0.0,
                "raw_response": response,
                "parse_error": True,
            }

    def _extract_json(self, text: str) -> Dict:
        """
        从文本中提取 JSON，处理 Markdown 代码块。

        Extract JSON from text, handling markdown code blocks.
        """
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to extract from markdown code block
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{[^{}]*\}',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        raise ValueError(f"Could not extract JSON from response")

    def _normalize_comparator(self, comp: str) -> str:
        """
        将比较器规范化为标准形式。

        Normalize comparator to standard form.
        """
        comp = str(comp).strip().lower()
        if comp in ("<", "lt", "less"):
            return "<"
        elif comp in (">", "gt", "greater"):
            return ">"
        else:
            return "~="
