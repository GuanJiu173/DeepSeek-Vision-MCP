"""图片路径检测与编码"""

import base64
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

_MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@dataclass
class ImageData:
    """图片处理结果，包含 data URI 和内容 SHA256 hash"""
    data_uri: str
    image_hash: str


def compute_image_hash(data: bytes) -> str:
    """计算图片内容的 SHA256 摘要"""
    return hashlib.sha256(data).hexdigest()


def normalize_prompt(prompt: str | None) -> str:
    """归一化 prompt，折叠空白以提高缓存命中率

    保留大小写，只做首尾 trim 和连续空白折叠。
    """
    if not prompt:
        return ""
    return re.sub(r"\s+", " ", prompt.strip())


def is_url(path: str) -> bool:
    """判断输入是否为 HTTP/HTTPS URL"""
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")


def image_to_data_uri(image_path: str) -> ImageData:
    """将本地图片转为 ImageData（data URI + SHA256 hash），URL 则原样返回"""
    if is_url(image_path):
        return ImageData(data_uri=image_path, image_hash=image_path)

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"不支持的图片格式: {suffix}，支持: {SUPPORTED_FORMATS}")

    mime = _MIME_MAP.get(suffix, f"image/{suffix[1:]}")
    data = path.read_bytes()
    image_hash = compute_image_hash(data)
    encoded = base64.b64encode(data).decode("utf-8")
    return ImageData(data_uri=f"data:{mime};base64,{encoded}", image_hash=image_hash)
