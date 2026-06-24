"""图片对比功能测试 — Difference、CompareResult、image_compare"""

import json
import os
from dataclasses import asdict
from unittest.mock import MagicMock, patch

import pytest

from vision_mcp_server.compare import (
    UI_COMPARE_SYSTEM_PROMPT,
    COMPARE_USER_PROMPT,
    Difference,
    CompareResult,
    image_compare,
    _compute_compare_cache_key,
    _parse_compare_response,
)


class TestDifference:
    def test_字段赋值(self):
        d = Difference(type="layout", severity="high", area="header",
                       expected="高度 64px", actual="高度 48px")
        assert d.type == "layout"
        assert d.severity == "high"
        assert d.area == "header"
        assert d.expected == "高度 64px"
        assert d.actual == "高度 48px"

    def test_to_dict(self):
        d = Difference(type="color", severity="medium", area="button",
                       expected="#2563eb", actual="#3b82f6")
        expected = {"type": "color", "severity": "medium", "area": "button",
                    "expected": "#2563eb", "actual": "#3b82f6"}
        assert asdict(d) == expected


class TestCompareResult:
    def test_空差异列表(self):
        r = CompareResult(summary="无差异", differences=[], model="qwen-vl-max")
        assert r.summary == "无差异"
        assert r.differences == []
        assert r.model == "qwen-vl-max"
        assert r.cached is False

    def test_to_dict(self):
        diffs = [
            Difference(type="layout", severity="high", area="nav",
                       expected="64px", actual="48px"),
            Difference(type="color", severity="low", area="text",
                       expected="#333", actual="#666"),
        ]
        r = CompareResult(summary="发现 2 处差异", differences=diffs,
                          model="qwen-vl-max", cached=True)
        d = r.to_dict()
        assert d["summary"] == "发现 2 处差异"
        assert len(d["differences"]) == 2
        assert d["differences"][0]["type"] == "layout"
        assert d["cached"] is True

    def test_to_dict_JSON可序列化(self):
        diffs = [Difference(type="layout", severity="high", area="x",
                            expected="a", actual="b")]
        r = CompareResult(summary="test", differences=diffs, model="m")
        json_str = json.dumps(r.to_dict(), ensure_ascii=False)
        parsed = json.loads(json_str)
        assert parsed["summary"] == "test"
        assert len(parsed["differences"]) == 1


class TestParseCompareResponse:
    def test_正常JSON(self):
        content = json.dumps({
            "summary": "发现 1 处差异",
            "differences": [
                {"type": "color", "severity": "medium", "area": "button",
                 "expected": "#2563eb", "actual": "#3b82f6"},
            ],
        }, ensure_ascii=False)
        result = _parse_compare_response(content, model="qwen-vl-max")
        assert result.summary == "发现 1 处差异"
        assert len(result.differences) == 1
        assert result.differences[0].type == "color"
        assert result.model == "qwen-vl-max"

    def test_无差异(self):
        content = json.dumps({"summary": "无差异", "differences": []}, ensure_ascii=False)
        result = _parse_compare_response(content, model="m")
        assert result.summary == "无差异"
        assert result.differences == []

    def test_非法JSON_返回空结果(self):
        result = _parse_compare_response("不是 JSON", model="m")
        assert result.summary == ""
        assert result.differences == []

    def test_缺失字段_使用默认值(self):
        content = json.dumps({"summary": "test", "differences": [
            {"type": "layout"},
        ]}, ensure_ascii=False)
        result = _parse_compare_response(content, model="m")
        assert len(result.differences) == 1
        assert result.differences[0].severity == "low"
        assert result.differences[0].area == ""


class TestCompareCacheKey:
    def test_相同图片_相同key(self):
        k1 = _compute_compare_cache_key("hash_a", "hash_b", "ui")
        k2 = _compute_compare_cache_key("hash_a", "hash_b", "ui")
        assert k1 == k2

    def test_交换顺序_不同key(self):
        k1 = _compute_compare_cache_key("hash_a", "hash_b", "ui")
        k2 = _compute_compare_cache_key("hash_b", "hash_a", "ui")
        assert k1 != k2

    def test_不同模式_不同key(self):
        k1 = _compute_compare_cache_key("hash_a", "hash_b", "ui")
        k2 = _compute_compare_cache_key("hash_a", "hash_b", "detailed")
        assert k1 != k2


class TestComparePrompt:
    def test_UI对比提示词_包含必要关键词(self):
        assert "对比" in UI_COMPARE_SYSTEM_PROMPT
        assert "差异" in UI_COMPARE_SYSTEM_PROMPT
        assert "JSON" in UI_COMPARE_SYSTEM_PROMPT
        assert "type" in UI_COMPARE_SYSTEM_PROMPT
        assert "severity" in UI_COMPARE_SYSTEM_PROMPT
        assert "layout" in UI_COMPARE_SYSTEM_PROMPT

    def test_用户提示词包含对比指令(self):
        assert "对比" in COMPARE_USER_PROMPT
        assert "设计稿" in COMPARE_USER_PROMPT


class TestImageCompare:
    """image_compare 集成行为测试"""

    def test_请求包含两张图片(self):
        """验证 API 请求包含两张图片"""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "summary": "无差异", "differences": [],
            }, ensure_ascii=False)))
        ]

        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MODELS": "qwen-vl-max"}
        with patch.dict(os.environ, env):
            with patch("vision_mcp_server.compare.get_cache", return_value=None):
                with patch("vision_mcp_server.compare.set_cache"):
                    with patch("vision_mcp_server.compare._get_client") as mock_get_client:
                        mock_client = MagicMock()
                        mock_get_client.return_value = mock_client
                        mock_client.client.chat.completions.create.return_value = mock_response
                        mock_client.models = ["qwen-vl-max"]
                        mock_client.provider = "bailian"

                        image_compare(
                            expected_image="https://ex.com/ta_two_imgs_a.png",
                            actual_image="https://ex.com/ta_two_imgs_b.png",
                        )

        call_args = mock_client.client.chat.completions.create.call_args[1]
        messages = call_args["messages"]
        user_content = messages[1]["content"]

        image_parts = [c for c in user_content if c["type"] == "image_url"]
        assert len(image_parts) == 2

    def test_URL输入_返回CompareResult(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "summary": "发现 1 处差异",
                "differences": [
                    {"type": "color", "severity": "medium", "area": "button",
                     "expected": "#2563eb", "actual": "#3b82f6"},
                ],
            }, ensure_ascii=False)))
        ]

        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MODELS": "qwen-vl-max"}
        with patch.dict(os.environ, env):
            with patch("vision_mcp_server.compare._get_client") as mock_get_client:
                mock_client = MagicMock()
                mock_get_client.return_value = mock_client
                mock_client.client.chat.completions.create.return_value = mock_response
                mock_client.models = ["qwen-vl-max"]
                mock_client.provider = "bailian"

                result = image_compare(
                    expected_image="https://ex.com/tc_url_result_a.png",
                    actual_image="https://ex.com/tc_url_result_b.png",
                )

        assert isinstance(result, CompareResult)
        assert result.summary == "发现 1 处差异"
        assert len(result.differences) == 1
        assert result.differences[0].type == "color"
        assert result.model == "qwen-vl-max"

    def test_缓存命中_不调API(self):
        cached = CompareResult(
            summary="缓存结果",
            differences=[Difference(type="color", severity="low", area="x",
                                    expected="a", actual="b")],
            model="qwen-vl-max",
        )

        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MODELS": "qwen-vl-max"}
        with patch.dict(os.environ, env):
            with patch("vision_mcp_server.compare.get_cache", return_value=cached.to_dict()):
                with patch("vision_mcp_server.compare._get_client") as mock_get_client:
                    mock_client = MagicMock()
                    mock_get_client.return_value = mock_client
                    mock_client.models = ["qwen-vl-max"]
                    mock_client.provider = "bailian"

                    result = image_compare(
                        expected_image="https://ex.com/tc_cache_hit_a.png",
                        actual_image="https://ex.com/tc_cache_hit_b.png",
                    )

        assert result.summary == "缓存结果"
        assert result.cached is True
        mock_client.client.chat.completions.create.assert_not_called()

    def test_force_refresh_跳过缓存(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "summary": "新结果", "differences": [],
            }, ensure_ascii=False)))
        ]

        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MODELS": "qwen-vl-max"}
        with patch.dict(os.environ, env):
            with patch("vision_mcp_server.compare.get_cache") as mock_get:
                with patch("vision_mcp_server.compare._get_client") as mock_get_client:
                    mock_client = MagicMock()
                    mock_get_client.return_value = mock_client
                    mock_client.client.chat.completions.create.return_value = mock_response
                    mock_client.models = ["qwen-vl-max"]
                    mock_client.provider = "bailian"

                    result = image_compare(
                        expected_image="https://ex.com/tc_force_ref_a.png",
                        actual_image="https://ex.com/tc_force_ref_b.png",
                        force_refresh=True,
                    )

        assert result.summary == "新结果"
        mock_get.assert_not_called()

    def test_API错误_返回空结果(self):
        from openai import APIError

        mock_request = MagicMock()

        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MODELS": "qwen-vl-max"}
        with patch.dict(os.environ, env):
            with patch("vision_mcp_server.compare._get_client") as mock_get_client:
                mock_client = MagicMock()
                mock_get_client.return_value = mock_client
                mock_client.client.chat.completions.create.side_effect = APIError(
                    "服务不可用", request=mock_request, body=None)
                mock_client.models = ["qwen-vl-max"]
                mock_client.provider = "bailian"

                result = image_compare(
                    expected_image="https://ex.com/tc_api_err_a.png",
                    actual_image="https://ex.com/tc_api_err_b.png",
                )

        assert isinstance(result, CompareResult)
        assert result.summary == ""
        assert result.differences == []
        assert result.model == "qwen-vl-max"


class TestImageCompareTool:
    """server.py 中注册的工具函数测试"""

    def test_tool传递参数(self):
        mock_result = CompareResult(
            summary="无差异", differences=[], model="qwen-vl-max",
        )
        with patch("vision_mcp_server.server._compare", return_value=mock_result):
            from vision_mcp_server.server import image_compare as tool_compare
            result = tool_compare(
                expected_image="design.png",
                actual_image="page.png",
                mode="ui",
                force_refresh=True,
            )

        assert isinstance(result, dict)
        assert result["summary"] == "无差异"

    def test_tool返回dict(self):
        mock_result = CompareResult(
            summary="发现差异",
            differences=[Difference(type="color", severity="high", area="btn",
                                    expected="red", actual="blue")],
            model="qwen-vl-max",
        )
        with patch("vision_mcp_server.server._compare", return_value=mock_result):
            from vision_mcp_server.server import image_compare as tool_compare
            result = tool_compare(expected_image="a.png", actual_image="b.png")

        assert isinstance(result, dict)
        assert result["summary"] == "发现差异"
        assert len(result["differences"]) == 1
        assert result["differences"][0]["type"] == "color"
