"""视觉客户端多 Provider 集成测试"""

import os
from unittest.mock import MagicMock, patch

import pytest

from vision_mcp_server.vision import (
    PROVIDERS,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_QUICK,
    VisionClient,
    _resolve_config,
    describe,
)


class TestResolveConfig:
    def test_默认使用bailian(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-bailian"}, clear=True):
            base_url, model, api_key = _resolve_config()

        assert base_url == PROVIDERS["bailian"]["base_url"]
        assert model == PROVIDERS["bailian"]["model"]
        assert api_key == "sk-bailian"

    def test_通过VISION_PROVIDER切换(self):
        env = {
            "VISION_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-openai",
        }
        with patch.dict(os.environ, env, clear=True):
            base_url, model, api_key = _resolve_config()

        assert base_url == PROVIDERS["openai"]["base_url"]
        assert model == PROVIDERS["openai"]["model"]
        assert api_key == "sk-openai"

    def test_VISION_MODEL覆盖默认模型(self):
        env = {
            "DASHSCOPE_API_KEY": "sk-test",
            "VISION_MODEL": "qwen3.6-flash",
        }
        with patch.dict(os.environ, env, clear=True):
            base_url, model, api_key = _resolve_config()

        assert model == "qwen3.6-flash"

    def test_VISION_BASE_URL覆盖默认地址(self):
        env = {
            "DASHSCOPE_API_KEY": "sk-test",
            "VISION_BASE_URL": "https://my-proxy.example.com/v1",
        }
        with patch.dict(os.environ, env, clear=True):
            base_url, model, api_key = _resolve_config()

        assert base_url == "https://my-proxy.example.com/v1"

    def test_VISION_API_KEY覆盖provider的key(self):
        env = {
            "VISION_API_KEY": "sk-override",
            "VISION_PROVIDER": "openai",
        }
        with patch.dict(os.environ, env, clear=True):
            base_url, model, api_key = _resolve_config()

        assert api_key == "sk-override"

    def test_代码传参model覆盖所有(self):
        env = {
            "DASHSCOPE_API_KEY": "sk-test",
            "VISION_MODEL": "qwen3.6-flash",
        }
        with patch.dict(os.environ, env, clear=True):
            base_url, model, api_key = _resolve_config(model="my-custom-model")

        assert model == "my-custom-model"

    def test_无任何APIKey_抛出ValueError(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="未设置 API Key"):
                _resolve_config()

    def test_openrouter_provider(self):
        env = {
            "VISION_PROVIDER": "openrouter",
            "OPENROUTER_API_KEY": "sk-or",
        }
        with patch.dict(os.environ, env, clear=True):
            base_url, model, api_key = _resolve_config()

        assert base_url == PROVIDERS["openrouter"]["base_url"]
        assert model == PROVIDERS["openrouter"]["model"]
        assert api_key == "sk-or"


class TestVisionClient:
    def test_初始化使用bailian预设(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            assert client.model == PROVIDERS["bailian"]["model"]

    def test_初始化切换openai(self):
        env = {"VISION_PROVIDER": "openai", "OPENAI_API_KEY": "sk-openai"}
        with patch.dict(os.environ, env, clear=True):
            client = VisionClient()
            assert client.model == PROVIDERS["openai"]["model"]

    def test_describe_默认quick模式(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="- 整体布局: 顶部导航+侧边栏+主内容区的三栏布局"
                )
            )
        ]

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            with patch.object(client.client.chat.completions, "create", return_value=mock_response) as mock_create:
                result = client.describe("https://example.com/photo.jpg")

        assert result["description"].startswith("- 整体布局")
        assert result["status"] == "success"
        mock_create.assert_called_once()
        call_args = mock_create.call_args[1]
        assert call_args["messages"][0]["content"] == SYSTEM_PROMPT_QUICK
        assert call_args["max_tokens"] == 600

    def test_describe_detailed模式(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="**UI布局**: 页面采用顶部导航+侧边栏+主内容区的三栏布局"
                )
            )
        ]

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            with patch.object(client.client.chat.completions, "create", return_value=mock_response) as mock_create:
                result = client.describe("https://example.com/photo.jpg", mode="detailed")

        assert result["status"] == "success"
        call_args = mock_create.call_args[1]
        assert call_args["messages"][0]["content"] == SYSTEM_PROMPT
        assert call_args["max_tokens"] == 1500

    def test_describe_自定义max_tokens(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="简短描述"))
        ]

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            with patch.object(client.client.chat.completions, "create", return_value=mock_response) as mock_create:
                result = client.describe("https://example.com/photo.jpg", max_tokens=300)

        assert result["status"] == "success"
        assert mock_create.call_args[1]["max_tokens"] == 300

    def test_VISION_MAX_TOKENS环境变量(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="描述"))
        ]

        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MAX_TOKENS": "800"}
        with patch.dict(os.environ, env, clear=True):
            client = VisionClient()
            with patch.object(client.client.chat.completions, "create", return_value=mock_response) as mock_create:
                client.describe("https://example.com/photo.jpg")

        assert mock_create.call_args[1]["max_tokens"] == 800

    def test_describe_支持自定义prompt(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="截图包含一个登录表单"))
        ]

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            with patch.object(client.client.chat.completions, "create", return_value=mock_response) as mock_create:
                result = client.describe("https://example.com/photo.jpg", prompt="描述这个登录页面")

        assert result["description"] == "截图包含一个登录表单"
        call_args = mock_create.call_args[1]
        user_content = call_args["messages"][1]["content"]
        text_parts = [item["text"] for item in user_content if item["type"] == "text"]
        assert any("描述这个登录页面" in t for t in text_parts)

    def test_describe_API错误_返回错误状态(self):
        from openai import APIError

        mock_request = MagicMock()
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            with patch.object(
                client.client.chat.completions,
                "create",
                side_effect=APIError("服务不可用", request=mock_request, body=None),
            ):
                result = client.describe("https://example.com/photo.jpg")

        assert result["status"] == "error"
        assert "服务不可用" in result["error"]


def test_describe_便捷函数():
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="图片描述"))
    ]

    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
        client = VisionClient()
        with patch.object(client.client.chat.completions, "create", return_value=mock_response):
            with patch("vision_mcp_server.vision._default_client", client):
                result = describe("https://example.com/photo.jpg")

    assert result["description"] == "图片描述"
    assert result["status"] == "success"


def test_describe_便捷函数detailed模式():
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="详细图片描述"))
    ]

    with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
        client = VisionClient()
        with patch.object(client.client.chat.completions, "create", return_value=mock_response):
            with patch("vision_mcp_server.vision._default_client", client):
                result = describe("https://example.com/photo.jpg", mode="detailed")

    assert result["description"] == "详细图片描述"
    assert result["status"] == "success"
