"""磁盘缓存 — 基于文件系统的 JSON 缓存"""

import hashlib
import json
import os
import platform
import time
from pathlib import Path


def _default_cache_dir() -> Path:
    """返回平台标准的全局缓存目录

    Windows: %LOCALAPPDATA%/vision-mcp-server/cache
    Linux/macOS: ~/.cache/vision-mcp-server/
    """
    if platform.system() == "Windows":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
        return Path(base) / "vision-mcp-server" / "cache"
    base = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(base) / "vision-mcp-server"


CACHE_DIR = Path(os.getenv("VISION_CACHE_DIR", str(_default_cache_dir())))
CACHE_TTL = int(os.getenv("VISION_CACHE_TTL", "604800"))  # 默认 7 天（秒）


def _is_cache_enabled() -> bool:
    return os.getenv("VISION_CACHE_ENABLED", "true").lower() != "false"


def compute_cache_key(image_hash: str, prompt: str, mode: str, provider: str, model: str) -> str:
    """计算缓存 key（SHA256）"""
    raw = f"{image_hash}|{prompt}|{mode}|{provider}|{model}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_path(cache_key: str, cache_dir: Path | None = None) -> Path:
    return (cache_dir or CACHE_DIR) / f"{cache_key}.json"


def get_cache(cache_key: str, cache_dir: Path | None = None) -> dict | None:
    """读取缓存，过期或损坏时返回 None"""
    if not _is_cache_enabled():
        return None

    path = _cache_path(cache_key, cache_dir)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        # 损坏的缓存文件，删掉
        try:
            path.unlink()
        except OSError:
            pass
        return None

    # TTL 检查
    if time.time() - data.get("created_at", 0) > CACHE_TTL:
        try:
            path.unlink()
        except OSError:
            pass
        return None

    return data.get("result")


def set_cache(cache_key: str, provider: str, model: str, result: dict, cache_dir: Path | None = None) -> None:
    """写入缓存"""
    if not _is_cache_enabled():
        return

    target = cache_dir or CACHE_DIR
    target.mkdir(parents=True, exist_ok=True)

    data = {
        "created_at": int(time.time()),
        "provider": provider,
        "model": model,
        "result": result,
    }
    path = target / f"{cache_key}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
