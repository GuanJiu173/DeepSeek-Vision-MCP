"""Server 入口测试 — image_understand 工具"""

import os
import tempfile
from unittest.mock import patch

import pytest

from vision_mcp_server.server import image_understand
from vision_mcp_server.image_utils import ImageData


class TestImageUnderstand:
    def test_调用describe_传递ImageData(self):
        """验证 image_understand 正确调用 describe 并传递参数"""
        with patch("vision_mcp_server.server.describe") as mock_describe:
            mock_describe.return_value = {
                "description": "测试",
                "model": "qwen-vl-max",
                "status": "success",
            }
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
                result = image_understand(
                    image_path="https://example.com/photo.jpg",
                    prompt="描述此图",
                    mode="detailed",
                )

        assert result["status"] == "success"
        mock_describe.assert_called_once()
        args, kwargs = mock_describe.call_args
        assert isinstance(args[0], ImageData)
        assert args[0].data_uri == "https://example.com/photo.jpg"
        assert kwargs["prompt"] == "描述此图"
        assert kwargs["mode"] == "detailed"
        assert kwargs["force_refresh"] is False

    def test_force_refresh_传递到describe(self):
        with patch("vision_mcp_server.server.describe") as mock_describe:
            mock_describe.return_value = {"status": "success", "description": ""}
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
                image_understand(
                    image_path="https://example.com/photo.jpg",
                    force_refresh=True,
                )

        args, kwargs = mock_describe.call_args
        assert kwargs["force_refresh"] is True

    def test_默认参数(self):
        with patch("vision_mcp_server.server.describe") as mock_describe:
            mock_describe.return_value = {"status": "success", "description": ""}
            with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
                image_understand(image_path="https://example.com/photo.jpg")

        args, kwargs = mock_describe.call_args
        assert kwargs["prompt"] is None
        assert kwargs["mode"] == "quick"
        assert kwargs["force_refresh"] is False

    def test_本地图片路径_计算hash(self):
        import struct
        import zlib

        # 1x1 红色 PNG
        signature = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", zlib.crc32(b"IHDR" + ihdr_data))
        raw_data = zlib.compress(b"\x00\xff\x00\x00")
        idat = struct.pack(">I", len(raw_data)) + b"IDAT" + raw_data + struct.pack(">I", zlib.crc32(b"IDAT" + raw_data))
        iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", zlib.crc32(b"IEND"))
        png_bytes = signature + ihdr + idat + iend

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_bytes)
            tmp_path = f.name

        try:
            from vision_mcp_server.image_utils import compute_image_hash
            expected_hash = compute_image_hash(png_bytes)
            with patch("vision_mcp_server.server.describe") as mock_describe:
                mock_describe.return_value = {"status": "success", "description": ""}
                with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "sk-test"}):
                    image_understand(image_path=tmp_path)

            args, kwargs = mock_describe.call_args
            assert args[0].image_hash == expected_hash
        finally:
            os.unlink(tmp_path)

    def test_文件不存在_抛出错误(self):
        with pytest.raises(FileNotFoundError):
            image_understand(image_path="/不存在/图片.png")
