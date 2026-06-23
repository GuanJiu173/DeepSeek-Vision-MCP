"""图片工具模块测试 — ImageData, normalize_prompt, compute_image_hash"""

import hashlib
import struct
import zlib
import base64
import os
import tempfile

import pytest

from vision_mcp_server.image_utils import (
    ImageData,
    is_url,
    image_to_data_uri,
    normalize_prompt,
    compute_image_hash,
    SUPPORTED_FORMATS,
)


class TestNormalizePrompt:
    def test_None_返回空字符串(self):
        assert normalize_prompt(None) == ""

    def test_首尾空白_折叠(self):
        assert normalize_prompt("  描述这张图  ") == "描述这张图"

    def test_连续空白_折叠(self):
        assert normalize_prompt("分析    这个UI") == "分析 这个UI"

    def test_换行_折叠(self):
        assert normalize_prompt("\n\n描述\n这张图\n") == "描述 这张图"

    def test_保留大小写(self):
        assert normalize_prompt("React Component") == "React Component"
        assert normalize_prompt("OCR") == "OCR"
        assert normalize_prompt("TailwindCSS") == "TailwindCSS"

    def test_空字符串(self):
        assert normalize_prompt("") == ""


class TestComputeImageHash:
    def test_已知内容_返回预期hash(self):
        data = b"hello"
        expected = hashlib.sha256(data).hexdigest()
        assert compute_image_hash(data) == expected

    def test_空字节(self):
        expected = hashlib.sha256(b"").hexdigest()
        assert compute_image_hash(b"") == expected

    def test_一致性_相同输入得相同输出(self):
        assert compute_image_hash(b"test data") == compute_image_hash(b"test data")

    def test_不同输入_不同hash(self):
        assert compute_image_hash(b"data1") != compute_image_hash(b"data2")


class TestImageData:
    def test_字段赋值(self):
        img = ImageData(data_uri="data:image/png;base64,abc", image_hash="deadbeef")
        assert img.data_uri == "data:image/png;base64,abc"
        assert img.image_hash == "deadbeef"

    def test_不可变字段类型(self):
        img = ImageData(data_uri="uri", image_hash="hash")
        assert isinstance(img.data_uri, str)
        assert isinstance(img.image_hash, str)


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
    def create_minimal_png(self):
        """创建最小的合法 PNG 文件"""
        signature = b"\x89PNG\r\n\x1a\n"

        ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data)
        ihdr = (
            struct.pack(">I", 13)
            + b"IHDR"
            + ihdr_data
            + struct.pack(">I", ihdr_crc)
        )

        raw_data = zlib.compress(b"\x00\xff\x00\x00")
        idat_crc = zlib.crc32(b"IDAT" + raw_data)
        idat = (
            struct.pack(">I", len(raw_data))
            + b"IDAT"
            + raw_data
            + struct.pack(">I", idat_crc)
        )

        iend_crc = zlib.crc32(b"IEND")
        iend = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)

        return signature + ihdr + idat + iend

    def test_png文件_返回ImageData(self):
        png_data = self.create_minimal_png()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(png_data)
            tmp_path = f.name

        try:
            result = image_to_data_uri(tmp_path)
            assert isinstance(result, ImageData)
            assert result.data_uri.startswith("data:image/png;base64,")
            # ImageData.hash 应该是内容的 SHA256
            assert result.image_hash == compute_image_hash(png_data)
            # 验证 base64 能正确解码
            encoded = result.data_uri.split(",", 1)[1]
            decoded = base64.b64decode(encoded)
            assert decoded == png_data
        finally:
            os.unlink(tmp_path)

    def test_URL输入_返回ImageData(self):
        url = "https://example.com/photo.jpg"
        result = image_to_data_uri(url)
        assert isinstance(result, ImageData)
        assert result.data_uri == url
        # URL 输入的 hash = URL 本身
        assert result.image_hash == url

    def test_jpg扩展名_正确mime(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            f.write(b"fake jpg")
            tmp_path = f.name

        try:
            result = image_to_data_uri(tmp_path)
            assert result.data_uri.startswith("data:image/jpeg;base64,")
        finally:
            os.unlink(tmp_path)

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
