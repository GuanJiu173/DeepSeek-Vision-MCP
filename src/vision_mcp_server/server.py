"""Vision MCP Server — 提供图片理解工具"""

from mcp.server.fastmcp import FastMCP

from .image_utils import image_to_data_uri
from .vision import describe

mcp = FastMCP("vision-mcp-server")


@mcp.tool()
def image_understand(image_path: str, prompt: str | None = None, mode: str = "quick") -> dict:
    """理解图片内容，返回面向软件开发的描述。

    适合分析 UI 截图、设计稿、架构图等。

    Args:
        image_path: 图片文件路径（支持 PNG/JPG/GIF/WebP）或图片 URL
        prompt: 针对图片的自定义提问，不传则根据 mode 自动选择提示词
        mode: "quick" 精简快速（默认，5-15s）| "detailed" 七维度详细分析（15-30s）
    """
    image_uri = image_to_data_uri(image_path)
    return describe(image_uri, prompt=prompt, mode=mode)
