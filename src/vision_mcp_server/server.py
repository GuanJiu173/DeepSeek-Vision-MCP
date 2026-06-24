"""Vision MCP Server — 提供图片理解工具"""

from mcp.server.fastmcp import FastMCP

from .compare import image_compare as _compare
from .image_utils import image_to_data_uri
from .vision import describe

mcp = FastMCP("vision-mcp-server")


@mcp.tool()
def image_understand(image_path: str, prompt: str | None = None,
                     mode: str = "quick", force_refresh: bool = False) -> dict:
    """理解图片内容，返回面向软件开发的描述。

    适合分析 UI 截图、设计稿、架构图等。

    Args:
        image_path: 图片文件路径（支持 PNG/JPG/GIF/WebP）或图片 URL
        prompt: 针对图片的自定义提问，不传则根据 mode 自动选择提示词
        mode: "quick" 精简快速（默认，5-15s）| "detailed" 七维度详细分析（15-30s）
        force_refresh: True 时跳过缓存，重新调用模型分析
    """
    image_data = image_to_data_uri(image_path)
    return describe(image_data, prompt=prompt, mode=mode, force_refresh=force_refresh)


@mcp.tool()
def image_compare(expected_image: str, actual_image: str,
                  mode: str = "ui", force_refresh: bool = False) -> dict:
    """对比两张图片（设计稿 vs 实现截图），返回结构化差异列表。

    适合 UI 还原度检查、回归测试等场景。

    Args:
        expected_image: 预期图片路径（设计稿）或 URL
        actual_image: 实际图片路径（实现截图）或 URL
        mode: 对比模式，默认 "ui"
        force_refresh: True 时跳过缓存，重新分析
    """
    result = _compare(
        expected_image=expected_image,
        actual_image=actual_image,
        mode=mode,
        force_refresh=force_refresh,
    )
    return result.to_dict()
