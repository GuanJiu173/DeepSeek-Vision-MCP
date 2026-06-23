"""视觉客户端多 Provider 集成测试 — 适配 ImageData + 缓存"""

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
from vision_mcp_server.image_utils import ImageData


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

    def test_初始化记录provider名称(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            assert client.provider == "bailian"

    def test_初始化切换openai_provider(self):
        env = {"VISION_PROVIDER": "openai", "OPENAI_API_KEY": "sk-openai"}
        with patch.dict(os.environ, env, clear=True):
            client = VisionClient()
            assert client.provider == "openai"

    def test_describe_默认quick模式(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content="- 整体布局: 顶部导航+侧边栏+主内容区的三栏布局"
                )
            )
        ]
        image_data = ImageData(
            data_uri="https://example.com/photo.jpg",
            image_hash="test_default_quick",
        )

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            with patch.object(client.client.chat.completions, "create", return_value=mock_response) as mock_create:
                result = client.describe(image_data, force_refresh=True)

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
        image_data = ImageData(
            data_uri="https://example.com/photo.jpg",
            image_hash="test_detailed",
        )

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            with patch.object(client.client.chat.completions, "create", return_value=mock_response) as mock_create:
                result = client.describe(image_data, mode="detailed", force_refresh=True)

        assert result["status"] == "success"
        call_args = mock_create.call_args[1]
        assert call_args["messages"][0]["content"] == SYSTEM_PROMPT
        assert call_args["max_tokens"] == 1500

    def test_describe_自定义max_tokens(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="简短描述"))]
        image_data = ImageData(data_uri="https://example.com/photo.jpg", image_hash="test_max_tokens")

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            with patch.object(client.client.chat.completions, "create", return_value=mock_response) as mock_create:
                result = client.describe(image_data, max_tokens=300, force_refresh=True)

        assert result["status"] == "success"
        assert mock_create.call_args[1]["max_tokens"] == 300

    def test_VISION_MAX_TOKENS环境变量(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="描述"))]
        image_data = ImageData(data_uri="https://example.com/photo.jpg", image_hash="test_env_max_tokens")

        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MAX_TOKENS": "800"}
        with patch.dict(os.environ, env, clear=True):
            client = VisionClient()
            with patch.object(client.client.chat.completions, "create", return_value=mock_response) as mock_create:
                client.describe(image_data, force_refresh=True)

        assert mock_create.call_args[1]["max_tokens"] == 800

    def test_describe_支持自定义prompt(self):
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="截图包含一个登录表单"))
        ]
        image_data = ImageData(data_uri="https://example.com/photo.jpg", image_hash="test_prompt")

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            with patch.object(client.client.chat.completions, "create", return_value=mock_response) as mock_create:
                result = client.describe(image_data, prompt="描述这个登录页面", force_refresh=True)

        assert result["description"] == "截图包含一个登录表单"
        call_args = mock_create.call_args[1]
        user_content = call_args["messages"][1]["content"]
        text_parts = [item["text"] for item in user_content if item["type"] == "text"]
        assert any("描述这个登录页面" in t for t in text_parts)

    def test_describe_API错误_返回错误状态(self):
        from openai import APIError

        mock_request = MagicMock()
        image_data = ImageData(data_uri="https://example.com/photo.jpg", image_hash="test_error")

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            with patch.object(
                client.client.chat.completions,
                "create",
                side_effect=APIError("服务不可用", request=mock_request, body=None),
            ):
                result = client.describe(image_data, force_refresh=True)

        assert result["status"] == "error"
        assert "服务不可用" in result["error"]


class TestDescribeCache:
    """缓存行为测试"""

    def test_缓存命中_不调用API(self):
        image_data = ImageData(data_uri="https://ex.com/pic.jpg", image_hash="cached_hash")
        cached_result = {"description": "缓存结果", "model": "qwen-vl-max", "status": "success"}

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            with patch("vision_mcp_server.vision.get_cache", return_value=cached_result) as mock_get:
                with patch.object(client.client.chat.completions, "create") as mock_create:
                    result = client.describe(image_data)

        assert result["description"] == "缓存结果"
        mock_get.assert_called_once()
        mock_create.assert_not_called()

    def test_缓存未命中_调用API并写入缓存(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="新结果"))]
        image_data = ImageData(data_uri="https://ex.com/pic.jpg", image_hash="miss_hash")

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            with patch("vision_mcp_server.vision.get_cache", return_value=None) as mock_get:
                with patch("vision_mcp_server.vision.set_cache") as mock_set:
                    with patch.object(client.client.chat.completions, "create", return_value=mock_response):
                        result = client.describe(image_data)

        assert result["description"] == "新结果"
        assert result["status"] == "success"
        mock_get.assert_called_once()
        mock_set.assert_called_once()
        args, kwargs = mock_set.call_args
        assert kwargs.get("provider") == "bailian"
        assert kwargs.get("model") == "qwen-vl-max"

    def test_force_refresh_跳过缓存(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="刷新结果"))]
        image_data = ImageData(data_uri="https://ex.com/pic.jpg", image_hash="fresh_hash")

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            with patch("vision_mcp_server.vision.get_cache") as mock_get:
                with patch("vision_mcp_server.vision.set_cache") as mock_set:
                    with patch.object(client.client.chat.completions, "create", return_value=mock_response):
                        result = client.describe(image_data, force_refresh=True)

        assert result["description"] == "刷新结果"
        mock_get.assert_not_called()
        mock_set.assert_called_once()

    def test_API错误_不写入缓存(self):
        from openai import APIError

        mock_request = MagicMock()
        image_data = ImageData(data_uri="https://ex.com/pic.jpg", image_hash="err_cache")

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            with patch("vision_mcp_server.vision.get_cache", return_value=None):
                with patch("vision_mcp_server.vision.set_cache") as mock_set:
                    with patch.object(
                        client.client.chat.completions,
                        "create",
                        side_effect=APIError("超时", request=mock_request, body=None),
                    ):
                        result = client.describe(image_data)

        assert result["status"] == "error"
        mock_set.assert_not_called()

    def test_缓存命中_且命中的是之前成功的结果(self):
        from openai import APIError

        mock_request = MagicMock()
        image_data = ImageData(data_uri="https://ex.com/pic.jpg", image_hash="err_no_write")

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            with patch("vision_mcp_server.vision.get_cache", return_value=None):
                with patch("vision_mcp_server.vision.set_cache") as mock_set:
                    with patch.object(
                        client.client.chat.completions,
                        "create",
                        side_effect=APIError("error", request=mock_request, body=None),
                    ):
                        result = client.describe(image_data)

        assert result["status"] == "error"
        mock_set.assert_not_called()


class TestDescribeConvenience:
    def test_便捷函数(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="图片描述"))]
        image_data = ImageData(data_uri="https://example.com/photo.jpg", image_hash="conv_hash")

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            with patch.object(client.client.chat.completions, "create", return_value=mock_response):
                with patch("vision_mcp_server.vision._default_client", client):
                    with patch("vision_mcp_server.vision.get_cache", return_value=None):
                        with patch("vision_mcp_server.vision.set_cache"):
                            result = describe(image_data, force_refresh=True)

        assert result["description"] == "图片描述"
        assert result["status"] == "success"

    def test_便捷函数detailed模式(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="详细图片描述"))]
        image_data = ImageData(data_uri="https://example.com/photo.jpg", image_hash="conv_hash2")

        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            with patch.object(client.client.chat.completions, "create", return_value=mock_response):
                with patch("vision_mcp_server.vision._default_client", client):
                    with patch("vision_mcp_server.vision.get_cache", return_value=None):
                        with patch("vision_mcp_server.vision.set_cache"):
                            result = describe(image_data, mode="detailed", force_refresh=True)

        assert result["description"] == "详细图片描述"
        assert result["status"] == "success"
