"""模型回退路由测试 — VISION_MODELS 解析 + fallback 行为"""

import os
from unittest.mock import MagicMock, patch

import pytest

from vision_mcp_server.vision import (
    PROVIDERS,
    VisionClient,
    _parse_model_list,
    _is_fallback_error,
)
from vision_mcp_server.image_utils import ImageData


class TestParseModelList:
    def test_正常多模型(self):
        env = {"VISION_MODELS": "qwen-vl-max,qwen-vl-plus,qwen-3.7"}
        with patch.dict(os.environ, env, clear=True):
            assert _parse_model_list() == ["qwen-vl-max", "qwen-vl-plus", "qwen-3.7"]

    def test_单个模型(self):
        env = {"VISION_MODELS": "gpt-4o-mini"}
        with patch.dict(os.environ, env, clear=True):
            assert _parse_model_list() == ["gpt-4o-mini"]

    def test_含空白(self):
        env = {"VISION_MODELS": " qwen-vl-max , qwen-vl-plus "}
        with patch.dict(os.environ, env, clear=True):
            assert _parse_model_list() == ["qwen-vl-max", "qwen-vl-plus"]

    def test_空字符串(self):
        with patch.dict(os.environ, {"VISION_MODELS": ""}, clear=True):
            assert _parse_model_list() == []

    def test_未设置(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _parse_model_list() == []


class TestIsFallbackError:
    def _make_error(self, message: str, status_code: int | None = None):
        """安全构造 APIError，兼容不同 SDK 版本"""
        from openai import APIError
        err = APIError(message, request=MagicMock(), body=None)
        if status_code is not None:
            err.status_code = status_code
        return err

    def test_status_code_429(self):
        err = self._make_error("Too Many Requests", status_code=429)
        assert _is_fallback_error(err) is True

    def test_消息含quota(self):
        err = self._make_error("quota exceeded for model")
        assert _is_fallback_error(err) is True

    def test_消息含insufficient_balance(self):
        err = self._make_error("insufficient balance")
        assert _is_fallback_error(err) is True

    def test_消息含rate_limit(self):
        err = self._make_error("rate limit reached")
        assert _is_fallback_error(err) is True

    def test_普通API错误_不触发回退(self):
        err = self._make_error("service unavailable")
        assert _is_fallback_error(err) is False

    def test_消息含quota但大小写不同(self):
        err = self._make_error("Quota Exceeded")
        assert _is_fallback_error(err) is True

    def test_中文消息不触发(self):
        err = self._make_error("服务不可用")
        assert _is_fallback_error(err) is False


class TestVisionClientInitWithFallback:
    def test_VISION_MODELS_设置模型列表(self):
        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MODELS": "m1,m2,m3"}
        with patch.dict(os.environ, env):
            client = VisionClient()
            assert client.models == ["m1", "m2", "m3"]
            assert client.model == "m1"  # primary model

    def test_未设置VISION_MODELS_使用单模型列表(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient()
            assert client.models == ["qwen-vl-max"]
            assert client.model == "qwen-vl-max"

    def test_VISION_MODELS_覆盖provider默认(self):
        env = {
            "VISION_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-openai",
            "VISION_MODELS": "gpt-4o,gpt-4o-mini",
        }
        with patch.dict(os.environ, env):
            client = VisionClient()
            assert client.models == ["gpt-4o", "gpt-4o-mini"]

    def test_代码传参model在无VISION_MODELS时生效(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
            client = VisionClient(model="custom-model")
            assert client.models == ["custom-model"]

    def test_VISION_MODELS优先于代码传参(self):
        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MODELS": "list-model-a,list-model-b"}
        with patch.dict(os.environ, env):
            client = VisionClient(model="param-model")
            assert client.models == ["list-model-a", "list-model-b"]


class TestDescribeFallback:
    """describe 模型回退行为测试"""

    def test_单模型_正常返回(self):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="结果"))]
        image_data = ImageData(data_uri="https://ex.com/pic.jpg", image_hash="fb_single")

        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MODELS": "qwen-vl-max"}
        with patch.dict(os.environ, env):
            client = VisionClient()
            with patch.object(client.client.chat.completions, "create", return_value=mock_response) as mock_create:
                result = client.describe(image_data, force_refresh=True)

        assert result["status"] == "success"
        assert result["model"] == "qwen-vl-max"
        assert mock_create.call_args[1]["model"] == "qwen-vl-max"

    def test_model1失败_model2成功(self):
        """第一模型返回 429，自动回退到第二模型"""
        from openai import APIError

        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="回退结果"))]
        image_data = ImageData(data_uri="https://ex.com/pic.jpg", image_hash="fb_retry")

        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MODELS": "bad-model,good-model"}
        with patch.dict(os.environ, env):
            client = VisionClient()
            err1 = APIError("quota exceeded", request=mock_request, body=None)
            with patch.object(
                client.client.chat.completions,
                "create",
                side_effect=[
                    err1,  # model1 失败
                    mock_response,  # model2 成功
                ],
            ) as mock_create:
                result = client.describe(image_data, force_refresh=True)

        assert result["status"] == "success"
        assert result["description"] == "回退结果"
        assert result["model"] == "good-model"
        assert mock_create.call_count == 2
        assert mock_create.call_args_list[0][1]["model"] == "bad-model"
        assert mock_create.call_args_list[1][1]["model"] == "good-model"

    def test_model1_429_model2_429_model3成功(self):
        from openai import APIError

        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="三次终于成功"))]
        image_data = ImageData(data_uri="https://ex.com/pic.jpg", image_hash="fb_three")

        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MODELS": "a,b,c"}
        with patch.dict(os.environ, env):
            client = VisionClient()
            err1 = APIError("rate limit exceeded", request=mock_request, body=None)
            err1.status_code = 429
            err2 = APIError("insufficient balance", request=mock_request, body=None)
            with patch.object(
                client.client.chat.completions,
                "create",
                side_effect=[err1, err2, mock_response],
            ) as mock_create:
                result = client.describe(image_data, force_refresh=True)

        assert result["status"] == "success"
        assert result["model"] == "c"
        assert mock_create.call_count == 3

    def test_所有模型都失败_返回错误(self):
        from openai import APIError

        mock_request = MagicMock()
        image_data = ImageData(data_uri="https://ex.com/pic.jpg", image_hash="fb_all_fail")

        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MODELS": "a,b"}
        with patch.dict(os.environ, env):
            client = VisionClient()
            with patch.object(
                client.client.chat.completions,
                "create",
                side_effect=[
                    APIError("quota exceeded", request=mock_request, body=None),
                    APIError("rate limit", request=mock_request, body=None),
                ],
            ) as mock_create:
                result = client.describe(image_data, force_refresh=True)

        assert result["status"] == "error"
        assert "rate limit" in result["error"]
        assert mock_create.call_count == 2

    def test_model1非回退错误_立即返回(self):
        """model1 返回非回退错误（如 500），不应尝试 model2"""
        from openai import APIError

        mock_request = MagicMock()
        image_data = ImageData(data_uri="https://ex.com/pic.jpg", image_hash="fb_no_retry")

        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MODELS": "a,b"}
        with patch.dict(os.environ, env):
            client = VisionClient()
            with patch.object(
                client.client.chat.completions,
                "create",
                side_effect=APIError("Internal Server Error", request=mock_request, body=None),
            ) as mock_create:
                result = client.describe(image_data, force_refresh=True)

        assert result["status"] == "error"
        assert "Internal Server Error" in result["error"]
        assert mock_create.call_count == 1

    def test_API成功_写缓存时使用正确model(self):
        """缓存写入时的 model 应与成功调用的模型一致"""
        from openai import APIError

        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="model2结果"))]
        image_data = ImageData(data_uri="https://ex.com/pic.jpg", image_hash="fb_cache_model")

        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MODELS": "bad,good"}
        with patch.dict(os.environ, env):
            client = VisionClient()
            with patch.object(client.client.chat.completions, "create", side_effect=[
                APIError("quota exceeded", request=mock_request, body=None),
                mock_response,
            ]):
                with patch("vision_mcp_server.vision.get_cache", return_value=None):
                    with patch("vision_mcp_server.vision.set_cache") as mock_set:
                        result = client.describe(image_data)

        assert result["status"] == "success"
        assert result["model"] == "good"
        args, kwargs = mock_set.call_args
        assert kwargs.get("model") == "good"

    def test_model1缓存命中_不调用API(self):
        """model1 有缓存时直接返回，不走回退"""
        cached = {"description": "model1缓存", "model": "a", "status": "success"}
        image_data = ImageData(data_uri="https://ex.com/pic.jpg", image_hash="fb_cache_hit")

        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MODELS": "a,b"}
        with patch.dict(os.environ, env):
            client = VisionClient()
            with patch("vision_mcp_server.vision.get_cache", return_value=cached) as mock_get:
                with patch.object(client.client.chat.completions, "create") as mock_create:
                    result = client.describe(image_data)

        assert result == cached
        mock_create.assert_not_called()

    def test_model1未命中且失败_model2缓存命中(self):
        from openai import APIError

        mock_request = MagicMock()
        cached_model2 = {"description": "model2缓存", "model": "b", "status": "success"}
        image_data = ImageData(data_uri="https://ex.com/pic.jpg", image_hash="fb_fallback_cache")

        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MODELS": "a,b"}
        with patch.dict(os.environ, env):
            client = VisionClient()
            cache_values = iter([None, cached_model2])
            with patch("vision_mcp_server.vision.get_cache", side_effect=lambda *a, **kw: next(cache_values)):
                with patch.object(
                    client.client.chat.completions,
                    "create",
                    side_effect=APIError("quota exceeded", request=mock_request, body=None),
                ) as mock_create:
                    result = client.describe(image_data)

        assert result == cached_model2
        assert mock_create.call_count == 1
        assert mock_create.call_args[1]["model"] == "a"

    def test_force_refresh跳过所有缓存(self):
        """force_refresh=True 时不应查任何模型的缓存"""
        from openai import APIError

        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="force结果"))]
        image_data = ImageData(data_uri="https://ex.com/pic.jpg", image_hash="fb_force")

        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MODELS": "a,b"}
        with patch.dict(os.environ, env):
            client = VisionClient()
            with patch("vision_mcp_server.vision.get_cache") as mock_get:
                with patch.object(client.client.chat.completions, "create", side_effect=[
                    APIError("quota exceeded", request=mock_request, body=None),
                    mock_response,
                ]):
                    with patch("vision_mcp_server.vision.set_cache"):
                        result = client.describe(image_data, force_refresh=True)

        assert result["status"] == "success"
        assert result["model"] == "b"
        mock_get.assert_not_called()


class TestDescribeConvenienceWithFallback:
    def test_便捷函数使用fallback(self):
        from openai import APIError

        mock_request = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="便捷回退"))]
        image_data = ImageData(data_uri="https://ex.com/pic.jpg", image_hash="fb_conv")

        env = {"DASHSCOPE_API_KEY": "sk-test", "VISION_MODELS": "a,b"}
        with patch.dict(os.environ, env):
            client = VisionClient()
            with patch.object(client.client.chat.completions, "create", side_effect=[
                APIError("quota exceeded", request=mock_request, body=None),
                mock_response,
            ]):
                with patch("vision_mcp_server.vision._default_client", client):
                    from vision_mcp_server.vision import describe as module_describe
                    with patch("vision_mcp_server.vision.get_cache", return_value=None):
                        with patch("vision_mcp_server.vision.set_cache"):
                            result = module_describe(image_data, force_refresh=True)

        assert result["status"] == "success"
        assert result["model"] == "b"
