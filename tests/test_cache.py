"""缓存模块测试"""

import json
import os
import time
from pathlib import Path

import pytest

from vision_mcp_server.cache import (
    CACHE_DIR,
    CACHE_TTL,
    get_cache,
    set_cache,
    compute_cache_key,
)


class TestComputeCacheKey:
    def test_相同输入_相同key(self):
        k1 = compute_cache_key("hash1", "prompt1", "quick", "bailian", "qwen-vl-max")
        k2 = compute_cache_key("hash1", "prompt1", "quick", "bailian", "qwen-vl-max")
        assert k1 == k2

    def test_不同image_hash_不同key(self):
        k1 = compute_cache_key("hash_a", "", "quick", "bailian", "qwen-vl-max")
        k2 = compute_cache_key("hash_b", "", "quick", "bailian", "qwen-vl-max")
        assert k1 != k2

    def test_不同mode_不同key(self):
        k1 = compute_cache_key("hash", "", "quick", "bailian", "qwen-vl-max")
        k2 = compute_cache_key("hash", "", "detailed", "bailian", "qwen-vl-max")
        assert k1 != k2

    def test_不同provider_不同key(self):
        k1 = compute_cache_key("hash", "", "quick", "bailian", "qwen-vl-max")
        k2 = compute_cache_key("hash", "", "quick", "openai", "gpt-4o-mini")
        assert k1 != k2

    def test_不同prompt_不同key(self):
        k1 = compute_cache_key("hash", "描述此图", "quick", "bailian", "qwen-vl-max")
        k2 = compute_cache_key("hash", "提取文字", "quick", "bailian", "qwen-vl-max")
        assert k1 != k2


class TestGetSetCache:
    def test_未命中_返回None(self, tmp_path):
        assert get_cache("nonexistent_key", cache_dir=tmp_path) is None

    def test_设置后_命中返回结果(self, tmp_path):
        result = {"description": "测试", "model": "qwen-vl-max", "status": "success"}
        set_cache("testkey", "bailian", "qwen-vl-max", result, cache_dir=tmp_path)
        cached = get_cache("testkey", cache_dir=tmp_path)
        assert cached == result

    def test_缓存包含provider和model元信息(self, tmp_path):
        result = {"description": "test", "status": "success"}
        set_cache("metatest", "openai", "gpt-4o-mini", result, cache_dir=tmp_path)
        cache_file = tmp_path / "metatest.json"
        assert cache_file.exists()
        raw = json.loads(cache_file.read_text(encoding="utf-8"))
        assert raw["provider"] == "openai"
        assert raw["model"] == "gpt-4o-mini"

    def test_过期缓存_返回None(self, tmp_path):
        result = {"description": "过期数据", "status": "success"}
        set_cache("expired", "bailian", "qwen-vl-max", result, cache_dir=tmp_path)
        # 篡改 created_at 为 8 天前
        cache_file = tmp_path / "expired.json"
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        data["created_at"] = int(time.time()) - 8 * 86400  # 8 天前
        cache_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        # 默认 TTL 7 天，所以应返回 None
        assert get_cache("expired", cache_dir=tmp_path) is None
        # 过期文件应被删除
        assert not cache_file.exists()

    def test_损坏的缓存文件_返回None(self, tmp_path):
        cache_file = tmp_path / "corrupted.json"
        cache_file.write_text("这不是合法 JSON", encoding="utf-8")
        assert get_cache("corrupted", cache_dir=tmp_path) is None


class TestCacheDirCreation:
    def test_set_cache_自动创建目录(self, tmp_path):
        nested = tmp_path / "sub" / "dir"
        result = {"description": "test", "status": "success"}
        set_cache("autodir", "bailian", "qwen-vl-max", result, cache_dir=nested)
        assert (nested / "autodir.json").exists()
