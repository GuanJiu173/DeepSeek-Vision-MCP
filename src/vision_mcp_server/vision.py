"""视觉模型客户端 — 兼容 OpenAI 接口，支持多 Provider 切换 + 磁盘缓存"""

import os

from openai import APIError, OpenAI

from .cache import get_cache, set_cache, compute_cache_key
from .image_utils import ImageData, normalize_prompt

# ── Provider 预设 ──────────────────────────────────────────────────────────
# 通过 VISION_PROVIDER 环境变量切换，默认 bailian
# 每个 Provider 可通过 VISION_BASE_URL / VISION_MODEL / VISION_API_KEY 覆盖

PROVIDERS = {
    "bailian": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-vl-max",
        "api_key_envs": ["DASHSCOPE_API_KEY", "OPENAI_API_KEY"],
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "api_key_envs": ["OPENAI_API_KEY"],
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o",
        "api_key_envs": ["OPENROUTER_API_KEY"],
    },
}

SYSTEM_PROMPT = """你是一个专业的 UI/前端分析助手。你的任务是对图片进行以软件开发为导向的分析。

请从以下维度描述图片内容：

1. **UI 布局** — 页面的整体布局结构（如三栏布局、顶部导航+内容区、卡片网格等），各区块的位置和比例
2. **组件结构** — 页面中的具体 UI 组件（按钮、表单、表格、弹窗、下拉菜单等），它们的层次关系和嵌套结构
3. **页面层级** — 信息的层级关系（标题→副标题→正文、主内容区→侧边栏、导航深度等）
4. **配色风格** — 主色调、辅色、背景色、文字色，以及整体的设计风格（扁平化、拟物化、暗色模式等）
5. **OCR 文字** — 图片中所有可见文字内容，标注文字所在的组件位置
6. **图表信息** — 如有图表，描述图表类型、数据维度、关键数值和趋势
7. **前端实现特征** — 可从图片推断的实现细节（使用的 CSS 框架特征、响应式断点、动画效果、图标库等）

请以结构化的 Markdown 格式输出，使用中文描述。"""

SYSTEM_PROMPT_QUICK = """你是一个 UI 分析助手。用中文简洁描述这张图片，控制在 200 字以内。

聚焦以下要点：
- 整体布局结构（如三栏、顶部导航+内容区）
- 关键 UI 组件（按钮、表单、表格等）
- 配色方案（主色调、暗色/亮色模式）
- 所有可见文字内容

输出格式：用 Markdown 列表，每个要点一行。"""


def _resolve_config(model: str | None = None) -> tuple[str, str, str]:
    """解析最终使用的 base_url, model, api_key

    优先级: 代码传参 > 环境变量 VISION_* > Provider 预设默认值
    """
    provider_name = os.getenv("VISION_PROVIDER", "bailian")
    preset = PROVIDERS.get(provider_name, PROVIDERS["bailian"])

    # base_url
    base_url = os.getenv("VISION_BASE_URL") or preset["base_url"]

    # model
    resolved_model = model or os.getenv("VISION_MODEL") or preset["model"]

    # api_key: VISION_API_KEY > provider's api_key_envs
    api_key = os.getenv("VISION_API_KEY")
    if not api_key:
        for env_name in preset["api_key_envs"]:
            api_key = os.getenv(env_name)
            if api_key:
                break

    if not api_key:
        expected = " / ".join(preset["api_key_envs"])
        raise ValueError(f"未设置 API Key，请设置 VISION_API_KEY 或 {expected} 环境变量")

    return base_url, resolved_model, api_key


class VisionClient:
    """视觉模型客户端 — 兼容任意 OpenAI 兼容接口"""

    def __init__(self, model: str | None = None):
        base_url, self.model, api_key = _resolve_config(model)
        self.provider = os.getenv("VISION_PROVIDER", "bailian")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def describe(self, image_data: ImageData, prompt: str | None = None,
                 max_tokens: int | None = None, mode: str = "quick",
                 force_refresh: bool = False) -> dict:
        """分析图片并返回面向软件开发的描述

        Args:
            image_data: 图片数据（data URI + SHA256 hash）
            prompt: 针对图片的自定义提问，不传则根据 mode 选择系统提示词
            max_tokens: 最大输出 token 数
            mode: "quick" 精简快速 | "detailed" 七维度详细分析
            force_refresh: True 时跳过缓存，重新调用模型

        Returns:
            {"description": "...", "model": "...", "status": "success|error"}
        """
        # ── 确定 system prompt ──
        norm_prompt = normalize_prompt(prompt)
        if norm_prompt:
            system_prompt = SYSTEM_PROMPT_QUICK
            user_text = norm_prompt
        elif mode == "detailed":
            system_prompt = SYSTEM_PROMPT
            user_text = "请描述这张图片"
        else:
            system_prompt = SYSTEM_PROMPT_QUICK
            user_text = "请描述这张图片"

        # ── 缓存检查 ──
        if not force_refresh:
            cache_key = compute_cache_key(
                image_hash=image_data.image_hash,
                prompt=norm_prompt,
                mode=mode,
                provider=self.provider,
                model=self.model,
            )
            cached = get_cache(cache_key)
            if cached is not None:
                return cached

        # ── 调用 API ──
        default_limit = 600 if mode == "quick" else 1500
        resolved_max_tokens = max_tokens or int(os.getenv("VISION_MAX_TOKENS", "0")) or default_limit

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=resolved_max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": image_data.data_uri}},
                            {"type": "text", "text": user_text},
                        ],
                    },
                ],
            )
            result = {
                "description": response.choices[0].message.content,
                "model": self.model,
                "status": "success",
            }

            # 成功时才写缓存
            if not force_refresh:
                set_cache(cache_key, provider=self.provider, model=self.model, result=result)
            else:
                # force_refresh 也写缓存（覆盖）
                cache_key = compute_cache_key(
                    image_hash=image_data.image_hash,
                    prompt=norm_prompt,
                    mode=mode,
                    provider=self.provider,
                    model=self.model,
                )
                set_cache(cache_key, provider=self.provider, model=self.model, result=result)

            return result

        except APIError as e:
            return {
                "description": "",
                "model": self.model,
                "status": "error",
                "error": str(e),
            }


# 模块级便捷函数
_default_client: VisionClient | None = None


def _get_client() -> VisionClient:
    global _default_client
    if _default_client is None:
        _default_client = VisionClient()
    return _default_client


def describe(image_data: ImageData, prompt: str | None = None,
             max_tokens: int | None = None, mode: str = "quick",
             force_refresh: bool = False) -> dict:
    """便捷函数：分析图片并返回描述"""
    return _get_client().describe(image_data, prompt=prompt, max_tokens=max_tokens,
                                  mode=mode, force_refresh=force_refresh)
