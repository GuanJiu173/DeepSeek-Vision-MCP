"""图片路径检测与编码"""

import base64
import mimetypes
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


def is_url(path: str) -> bool:
    """判断输入是否为 HTTP/HTTPS URL"""
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")


def image_to_data_uri(image_path: str) -> str:
    """将本地图片转为 base64 data URI，URL 则原样返回"""
    if is_url(image_path):
        return image_path

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_FORMATS:
        raise ValueError(f"不支持的图片格式: {suffix}，支持: {SUPPORTED_FORMATS}")

    mime = _MIME_MAP.get(suffix, f"image/{suffix[1:]}")
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{encoded}"
