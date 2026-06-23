"""图片工具模块测试"""

import base64
import os
import tempfile
from pathlib import Path

import pytest

from vision_mcp_server.image_utils import is_url, image_to_data_uri, SUPPORTED_FORMATS


class TestIsUrl:
    def test_http_url_返回True(self):
        assert is_url("http://example.com/image.jpg") is True

    def test_https_url_返回True(self):
        assert is_url("https://example.com/image.png") is True

    def test_本地路径_返回False(self):
        assert is_url("/home/user/photo.jpg") is False

    def test_相对路径_返回False(self):
        assert is_url("images/screenshot.png") is False

    def test_windows绝对路径_返回False(self):
        assert is_url(r"C:\Users\test\image.jpg") is False

    def test_data_uri_返回False(self):
        assert is_url("data:image/jpeg;base64,/9j/4AAQ") is False


class TestImageToDataUri:
    def test_png文件_正确编码(self):
        # 创建一个 1x1 的 PNG 文件
        import struct
        import zlib

        def create_minimal_png():
            """创建最小的合法 PNG 文件"""
            signature = b"\x89PNG\r\n\x1a\n"

            # IHDR chunk
            ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
            ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
            ihdr = (
                struct.pack(">I", 13)
                + b"IHDR"
                + ihdr_data
                + struct.pack(">I", ihdr_crc)
            )

            # IDAT chunk
            raw_data = zlib.compress(b"\x00\xff\x00\x00")  # 红色像素
            idat_crc = zlib.crc32(b"IDAT" + raw_data)
            idat = (
                struct.pack(">I", len(raw_data))
                + b"IDAT"
                + raw_data
                + struct.pack(">I", idat_crc)
            )

            # IEND chunk
            iend_crc = zlib.crc32(b"IEND")
            iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

            return signature + ihdr + idat + iend

        png_data = create_minimal_png()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_data)
            tmp_path = f.name

        try:
            result = image_to_data_uri(tmp_path)
            assert result.startswith("data:image/png;base64,")
            # 验证能解码回去
            encoded = result.split(",", 1)[1]
            decoded = base64.b64decode(encoded)
            assert decoded == png_data
        finally:
            os.unlink(tmp_path)

    def test_jpg扩展名_使用正确mime类型(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake jpg content")
            tmp_path = f.name

        try:
            result = image_to_data_uri(tmp_path)
            assert result.startswith("data:image/jpeg;base64,")
        finally:
            os.unlink(tmp_path)

    def test_URL输入_原样返回(self):
        url = "https://example.com/photo.jpg"
        result = image_to_data_uri(url)
        assert result == url

    def test_文件不存在_抛出FileNotFoundError(self):
        with pytest.raises(FileNotFoundError):
            image_to_data_uri("/不存在的路径/图片.jpg")

    def test_不支持的后缀_抛出ValueError(self):
        with tempfile.NamedTemporaryFile(suffix=".bmp", delete=False) as f:
            f.write(b"dummy")
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="不支持的图片格式"):
                image_to_data_uri(tmp_path)
        finally:
            os.unlink(tmp_path)
