"""图片对比 — 设计稿 vs 实现截图的结构化差异分析"""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from openai import APIError

from .cache import get_cache, set_cache, compute_cache_key
from .image_utils import ImageData, image_to_data_uri
from .vision import _get_client

UI_COMPARE_SYSTEM_PROMPT = """你是一个专业的 UI 对比分析助手。你的任务是比较两张图片（设计稿 vs 实现截图），找出差异。

请从以下维度分析差异：

1. **layout** — 布局偏差：元素位置、间距、对齐方式不一致
2. **color** — 颜色差异：主色调、背景色、文字色不匹配
3. **spacing** — 间距问题：padding、margin、元素间距离不同
4. **typography** — 字体差异：字号、字重、行高不一致
5. **missing** — 缺少元素：实现中缺失了设计稿中的组件或内容
6. **extra** — 多余元素：实现中出现了设计稿没有的内容

请以 JSON 格式输出（不要 markdown 代码块标记）：

{
  "summary": "发现的差异总数和分类概述",
  "differences": [
    {
      "type": "layout|color|spacing|typography|missing|extra",
      "severity": "high|medium|low",
      "area": "差异所在的区域或组件名称",
      "expected": "设计稿中的预期表现",
      "actual": "实现中的实际表现"
    }
  ]
}

如果两张图片没有差异，differences 返回空数组。"""

COMPARE_USER_PROMPT = "请对比这两张图片，第一张是设计稿（预期），第二张是实现截图（实际），找出所有差异。"


@dataclass
class Difference:
    """单个差异项"""
    type: str
    severity: str
    area: str
    expected: str
    actual: str


@dataclass
class CompareResult:
    """对比结果"""
    summary: str
    differences: list[Difference]
    model: str
    cached: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转为 JSON 可序列化的 dict"""
        return {
            "summary": self.summary,
            "differences": [asdict(d) for d in self.differences],
            "model": self.model,
            "cached": self.cached,
        }


def _compute_compare_cache_key(hash_a: str, hash_b: str, mode: str) -> str:
    """计算对比缓存的 key"""
    raw = f"compare|{hash_a}|{hash_b}|{mode}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _parse_compare_response(content: str, model: str) -> CompareResult:
    """将 API 返回的 JSON 字符串解析为 CompareResult"""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return CompareResult(summary="", differences=[], model=model)

    diffs = []
    for item in data.get("differences", []):
        diffs.append(Difference(
            type=item.get("type", "unknown"),
            severity=item.get("severity", "low"),
            area=item.get("area", ""),
            expected=item.get("expected", ""),
            actual=item.get("actual", ""),
        ))
    return CompareResult(
        summary=data.get("summary", ""),
        differences=diffs,
        model=model,
    )


def _cached_to_result(cached: dict, model: str) -> CompareResult:
    """将缓存中的 dict 还原为 CompareResult"""
    diffs = [Difference(**d) for d in cached.get("differences", [])]
    return CompareResult(
        summary=cached.get("summary", ""),
        differences=diffs,
        model=cached.get("model", model),
        cached=True,
    )


def image_compare(expected_image: str, actual_image: str,
                  mode: str = "ui", force_refresh: bool = False) -> CompareResult:
    """对比两张图片，返回结构化差异结果

    Args:
        expected_image: 预期图片路径（设计稿）或 URL
        actual_image: 实际图片路径（实现截图）或 URL
        mode: 对比模式，默认 "ui"
        force_refresh: True 时跳过缓存

    Returns:
        CompareResult 实例
    """
    client = _get_client()

    expected = image_to_data_uri(expected_image)
    actual = image_to_data_uri(actual_image)

    # ── 缓存检查 ──
    if not force_refresh:
        cache_key = _compute_compare_cache_key(
            expected.image_hash, actual.image_hash, mode,
        )
        cached = get_cache(cache_key)
        if cached is not None:
            return _cached_to_result(cached, model=client.model)

    # ── 调用 API ──
    model = client.models[0]
    system_prompt = UI_COMPARE_SYSTEM_PROMPT

    try:
        response = client.client.chat.completions.create(
            model=model,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": expected.data_uri}},
                        {"type": "image_url", "image_url": {"url": actual.data_uri}},
                        {"type": "text", "text": COMPARE_USER_PROMPT},
                    ],
                },
            ],
        )

        content = response.choices[0].message.content
        result = _parse_compare_response(content, model=model)

        # 成功时写缓存
        if not force_refresh:
            set_cache(cache_key, provider=client.provider, model=model, result=result.to_dict())

        return result

    except APIError as e:
        return CompareResult(summary="", differences=[], model=model)
